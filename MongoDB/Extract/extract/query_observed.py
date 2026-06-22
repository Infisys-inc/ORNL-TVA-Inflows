import pymongo
import utils

from bson import CodecOptions, json_util
from bson.codec_options import DatetimeConversion

import numpy as np
import pandas as pd
import xarray as xr
import multiprocessing as mp


def main():
	lock = mp.Manager().Lock()
	with open(__file__.replace('.py', '.json'), "r") as r:
		config = json_util.loads(r.read())
	connection = config["connection"]
	database = config["database"]
	queries = config["queries"]

	for query in queries:
		process = query["process"]
		if not process:
			continue

		collection = query["collection"]
		event_start_time_key = query["event_start_time_key"]
		event_end_time_key = query["event_end_time_key"]
		event_time_key = query["event_time_key"]
		event_value_key = query["event_value_key"]
		observed_time_chunk_size = query["observed_time_chunk_size"]
		file_format = query["file_format"]
		aggregate = query["aggregate"]
		folder_overrides = query["folder_overrides"]
		observed_start_time = query["observed_start_time"]
		threads = query["threads"]

		with pymongo.MongoClient(connection, connect=True) as c:
			first = c[database][collection].with_options(codec_options=CodecOptions(datetime_conversion=DatetimeConversion.DATETIME_CLAMP)).find_one(aggregate[0]["$match"], {"metaData.timeStepMinutes": 1, "metaData.localTimeZone": 1})
			time_step_minutes = first["metaData"]["timeStepMinutes"]
			timezone = "UTC" if event_time_key == "t" else first["metaData"]["localTimeZone"]
			time_step = f"{time_step_minutes // 60}-HOUR"

			min_max = c[database][collection].with_options(codec_options=CodecOptions(datetime_conversion=DatetimeConversion.DATETIME_CLAMP)).aggregate([aggregate[0], {"$group": {"_id": None, "min": {"$min": f"${event_start_time_key}"}, "max": {"$max": f"${event_end_time_key}"}}}]).next()
			event_time_start, event_time_end = utils.to_cardinal_time(min_max["min"], time_step_minutes), utils.to_cardinal_time(min_max["max"], time_step_minutes)
			event_time_start = max(observed_start_time, event_time_start) if observed_start_time else event_time_start
			event_times = pd.date_range(event_time_start, event_time_end, freq=f"{time_step_minutes}min")
			location_ids = np.array(sorted(c[database][collection].distinct("locationId", aggregate[0]["$match"])), dtype=str)

		path = utils.get_folder_path(folder_overrides, aggregate, None, time_step, timezone, file_format)
		utils.ensure_path(path)
		parameters = [(connection, database, collection, path, file_format, observed_time_chunk_size, time_step_minutes, event_times, location_ids, location_id, event_start_time_key, event_time_key, event_value_key, aggregate, lock) for location_id in location_ids]

		with mp.Pool(threads or mp.cpu_count()) as pool:
			pool.starmap(run_query, parameters)
		# for p in parameters:
		# 	run_query(*p)


def run_query(connection, database, collection, path, file_format, observed_time_chunk_size, time_step_minutes, event_times, location_ids, location_id, event_start_time_key, event_time_key, event_value_key, aggregate, lock):
	values = np.full((1, len(event_times)), np.nan, dtype=np.float32)
	observed_time_chunk_size = min(len(event_times), observed_time_chunk_size)
	module_instance_id = None
	parameter_id = None
	qualifier_id = None
	encoded_time_step_id = None

	with pymongo.MongoClient(connection, connect=True) as c:
		meta_data = c[database][collection].with_options(codec_options=CodecOptions(datetime_conversion=DatetimeConversion.DATETIME_CLAMP)).find_one({**aggregate[0]["$match"], "locationId": str(location_id)}, {"metaData": 1})["metaData"]
		actual_event_start_times = sorted(c[database][collection].distinct(event_start_time_key, {**aggregate[0]["$match"], "locationId": str(location_id)}))
		actual_event_start_times = [a for a in actual_event_start_times if utils.to_cardinal_time(a, time_step_minutes) >= event_times[0]]
		if not actual_event_start_times:
			return
		for result in c[database][collection].with_options(codec_options=CodecOptions(datetime_conversion=DatetimeConversion.DATETIME_CLAMP)).aggregate([
			{"$match": {**aggregate[0]["$match"], "locationId": str(location_id), event_start_time_key: {"$in": actual_event_start_times}}}, *aggregate[1:],
			{"$project": {"_id": 0, "moduleInstanceId": 1, "parameterId": 1, "qualifierId": 1, "encodedTimeStepId": 1, "locationId": 1, f"timeseries.{event_time_key}": 1, f"timeseries.{event_value_key}": 1}}]):

			module_instance_id = result["moduleInstanceId"]
			parameter_id = result["parameterId"]
			qualifier_id = result["qualifierId"]
			encoded_time_step_id = result["encodedTimeStepId"]

			timeseries = [(t[event_value_key], pd.to_datetime(utils.to_cardinal_time(t[event_time_key], time_step_minutes))) for t in result["timeseries"] if event_times[0] <= utils.to_cardinal_time(t[event_time_key], time_step_minutes) <= event_times[-1]]
			value = np.empty(len(timeseries), dtype=np.float32)
			time = np.empty(len(timeseries), dtype="datetime64[ns]")
			for i, x in enumerate(timeseries):
				value[i], time[i] = x
			event_time_idx = np.searchsorted(event_times, time)

			missing = ~((event_time_idx < len(event_times)) & (event_times[event_time_idx] == time))
			if np.any(missing):
				raise ValueError(f"event times not found: {time[missing]}")

			values[0, event_time_idx] = value

	ds = get_dataset(module_instance_id, parameter_id, qualifier_id, encoded_time_step_id, meta_data, values, location_ids, location_id, file_format, event_times, event_time_key)
	write_file(ds, path, file_format, location_ids, location_id, observed_time_chunk_size, lock)
	print(f"{path} -> {location_id}")


def write_file(ds, path, file_format, location_ids, location_id, observed_time_chunk_size, lock):
	if file_format in ["zarr3"]:
		ds.attrs.update({"locationIds": location_ids})
	else:
		ds.coords["locationId"] = ds.coords["locationId"].astype(f"U{max([len(v) for v in location_ids])}")

	if file_format.startswith("zarr"):
		encoding = {"chunks": (1, observed_time_chunk_size)}
		with lock:
			utils.write_zarr(ds, path, file_format, encoding)
	elif file_format in ["nc", "netcdf"]:
		encoding = {"compression": "zlib", "complevel": 5, "chunksizes": (1, observed_time_chunk_size)}
		utils.write_netcdf(ds, path, location_id, None, encoding)
	else:
		utils.write_csv(ds, path, location_id, None)


def get_dataset(module_instance_id, parameter_id, qualifier_id, encoded_time_step_id, meta_data, values, location_ids, location_id, file_format, event_times, event_time_key):
	return xr.Dataset(
		data_vars={
			"value": (["locationId", "eventTime"], values, {
				"units": meta_data["unit"]
			})
		},
		coords={
			"locationId": (["locationId"], [np.uint32(np.searchsorted(location_ids, location_id)) if file_format in ["zarr3"] else location_id], {
				"cf_role": "timeseries_id",
				"long_name": "Location identifier"
			}),
			"eventTime": (["eventTime"], event_times, {
				"standard_name": "time",
				"long_name": "Event time"
			})
		},
		attrs=utils.get_attrs(module_instance_id, parameter_id, qualifier_id, encoded_time_step_id, meta_data, event_time_key)
	)


if __name__ == '__main__':
	main()
