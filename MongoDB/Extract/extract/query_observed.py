import re
import utils
import pymongo

import numpy as np
import xarray as xr
import pandas as pd
import multiprocessing as mp

from bson import json_util


def main():
	lock = mp.Manager().Lock()
	with open(__file__.replace('.py', '.json'), "r") as r:
		config = json_util.loads(r.read())

	for query in config["queries"]:
		parameters = get_parameters(config, query, lock)
		with mp.Pool(query["threads"] or mp.cpu_count()) as pool:
			pool.starmap(run_query, parameters)
		# for p in parameters:
		# 	run_query(*p)


def get_parameters(config, query, lock):
	parameters = []
	with pymongo.MongoClient(config["connection"], connect=True, compressors='snappy') as c:
		db = c[config["database"]][query["collection"]]
		for module_instance_id in [m for m in db.distinct("moduleInstanceId") if not query["filter"]["moduleInstanceId"] or any([re.match(f, m) for f in query["filter"]["moduleInstanceId"]])]:
			for parameter_id in [m for m in db.distinct("parameterId", {"moduleInstanceId": module_instance_id}) if not query["filter"]["parameterId"] or any([bool(re.match(f, m)) for f in query["filter"]["parameterId"]])]:
				for qualifier_id in [m for m in db.distinct("qualifierId", {"moduleInstanceId": module_instance_id, "parameterId": parameter_id}) if not query["filter"]["qualifierId"] or any([bool(re.match(f, m)) for f in query["filter"]["qualifierId"]])]:
					for encoded_time_step_id in [m for m in db.distinct("encodedTimeStepId", {"moduleInstanceId": module_instance_id, "parameterId": parameter_id, "qualifierId": qualifier_id}) if not query["filter"]["encodedTimeStepId"] or any([bool(re.match(f, m)) for f in query["filter"]["encodedTimeStepId"]])]:
						utils.ensure_path(utils.get_path("Observed", module_instance_id, parameter_id, qualifier_id, encoded_time_step_id), query["resume"])
						location_ids = sorted([m for m in db.distinct("locationId", {"moduleInstanceId": module_instance_id, "parameterId": parameter_id, "qualifierId": qualifier_id, "encodedTimeStepId": encoded_time_step_id}) if not query["filter"]["locationId"] or any([bool(re.match(f, m)) for f in query["filter"]["locationId"]])])
						for location_id in location_ids:
							filter = {"moduleInstanceId": module_instance_id, "parameterId": parameter_id, "qualifierId": qualifier_id, "encodedTimeStepId": encoded_time_step_id, "locationId": location_id}
							parameters.append((config["connection"], config["database"], query["collection"], query["resume"], query["max_age_hours"], query["file_format"], query["observed_time_chunk_size"], query["event_start_time_key"], query["event_end_time_key"], query["event_time_key"], query["event_value_key"], filter, location_ids, lock))
	return parameters


def run_query(connection, database, collection, resume, max_age_hours, file_format, observed_time_chunk_size, event_start_time_key, event_end_time_key, event_time_key, event_value_key, filter, location_ids, lock):
	with pymongo.MongoClient(connection, connect=True, compressors='snappy') as c:
		db = c[database][collection]
		time_step_minutes = utils.get_time_step_minutes(db, filter)
		event_times = get_event_times(db, filter, event_start_time_key, event_end_time_key, time_step_minutes)

		observed_time_chunk_size = min(len(event_times), observed_time_chunk_size)

		path = utils.get_path("Observed", filter["moduleInstanceId"], filter["parameterId"], filter["qualifierId"], filter["encodedTimeStepId"])
		if resume and utils.exists(path, filter["locationId"], file_format, max_age_hours):
			print(f"{path} -> {filter["locationId"]} [resumed]")
		else:
			meta_data = utils.get_meta_data(db, filter)
			values = get_values(db, filter, event_time_key, event_value_key, time_step_minutes, event_times)
			dataset = get_dataset(filter, meta_data, values, location_ids, file_format, event_times, event_time_key)
			write_file(dataset, path, file_format, location_ids, filter["locationId"], observed_time_chunk_size, lock)
			print(f"{path} -> {filter["locationId"]}")


def get_event_times(db, filter, event_start_time_key, event_end_time_key, time_step_minutes):
	min_max = db.with_options(codec_options=utils.codec_options).aggregate([{"$match": filter}, {"$group": {"_id": None, "min": {"$min": f"${event_start_time_key}"}, "max": {"$max": f"${event_end_time_key}"}}}]).next()
	event_time_start, event_time_end = utils.to_cardinal_time(min_max["min"], time_step_minutes), utils.to_cardinal_time(min_max["max"], time_step_minutes)
	return pd.date_range(event_time_start, event_time_end, freq=f"{time_step_minutes}min")


def get_values(db, filter, event_time_key, event_value_key, time_step_minutes, event_times):
	values = np.full((1, len(event_times)), np.nan, dtype=np.float32)
	for result in db.with_options(codec_options=utils.codec_options).find(filter, {"_id": 0, f"timeseries.{event_time_key}": 1, f"timeseries.{event_value_key}": 1, f"timeseries.f": 1}):
		timeseries = [(t[event_value_key] if t["f"] <= 2 else np.float32(np.nan), pd.to_datetime(utils.to_cardinal_time(t[event_time_key], time_step_minutes))) for t in result["timeseries"] if event_times[0] <= utils.to_cardinal_time(t[event_time_key], time_step_minutes) <= event_times[-1]]
		value = np.empty(len(timeseries), dtype=np.float32)
		time = np.empty(len(timeseries), dtype="datetime64[ns]")
		for i, x in enumerate(timeseries):
			value[i], time[i] = x
		event_time_idx = np.searchsorted(event_times, time)

		missing = ~((event_time_idx < len(event_times)) & (event_times[event_time_idx] == time))
		if np.any(missing):
			raise ValueError(f"event times not found: {time[missing]} -> {filter}")

		values[0, event_time_idx] = value
	return values


def write_file(ds, path, file_format, location_ids, location_id, observed_time_chunk_size, lock):
	if file_format  == "zarr3":
		ds.attrs.update({"locationIds": location_ids})
		encoding = {"value": {"chunks": (1, observed_time_chunk_size)}}
		with lock:
			utils.write_zarr(ds, path, file_format, encoding)
	elif file_format == "nc":
		encoding = {
			"value": {"dtype": "float32", "_FillValue": np.float32(np.nan), "zlib": True, "complevel": 5, "shuffle": True, "chunksizes": (1, observed_time_chunk_size)},
			"eventTime": {"zlib": True, "complevel": 5, "shuffle": True, "chunksizes": (observed_time_chunk_size,)}
		}
		utils.write_netcdf(ds, path, location_id, encoding)
	else:
		utils.write_csv(ds, path, location_id)


def get_dataset(filter, meta_data, values, location_ids, file_format, event_times, event_time_key):
	return xr.Dataset(
		data_vars={
			"value": (["locationId", "eventTime"], values, {
				"units": meta_data["unit"]
			})
		},
		coords={
			"locationId": (["locationId"], [np.uint32(np.searchsorted(np.array(location_ids), filter["locationId"])) if file_format == "zarr3" else filter["locationId"]], {
				"cf_role": "timeseries_id",
				"long_name": "Location identifier"
			}),
			"eventTime": (["eventTime"], event_times, {
				"standard_name": "time",
				"long_name": "Event time"
			})
		},
		attrs=utils.get_attrs(filter, meta_data, event_time_key)
	)


if __name__ == '__main__':
	main()
