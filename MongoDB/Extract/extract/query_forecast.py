import pymongo
import utils

from datetime import timedelta
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
	[q['aggregate'][0]['$match'].pop("locationId") if "locationId" in q['aggregate'][0]['$match'] else None for q in config["queries"]]
	[q['aggregate'][0]['$match'].pop("locationId") if "locationId" in q['aggregate'][0]['$match'] else None for q in config["queries"]]
	queries = list({str(q['aggregate'][0]['$match']): q for q in config["queries"]}.values())

	for query in queries:
		process = query["process"]
		if not process:
			continue

		collection = query["collection"]
		forecast_time_key = query["forecast_time_key"]
		event_time_key = query["event_time_key"]
		event_value_key = query["event_value_key"]
		lead_time_days = query["lead_time_days"]
		forecast_time_chunk_size = query["forecast_time_chunk_size"]
		file_format = query["file_format"]
		find = query["aggregate"][0]["$match"]
		batch_size = query["batch_size"]
		forecast_start_time = query["forecast_start_time"]
		threads = query["threads"]

		parameters = []
		with pymongo.MongoClient(connection, connect=True, compressors='snappy') as c:
			ensemble_ids = {str(r["_id"]): (r["min"], r["max"]) for r in c[database][collection].with_options(codec_options=CodecOptions(datetime_conversion=DatetimeConversion.DATETIME_CLAMP)).aggregate([{"$match": find}, {"$group": {"_id": "$ensembleId", "min": {"$min": f"${forecast_time_key}"}, "max": {"$max": f"${forecast_time_key}"}}}])}
			for ensemble_id, min_max in ensemble_ids.items():
				location_ids = np.array(sorted(c[database][collection].distinct("locationId", {**find, "ensembleId": ensemble_id})), dtype=str)
				ensemble_member_ids = np.array(sorted(c[database][collection].distinct("ensembleMemberId", {**find, "ensembleId": ensemble_id})), dtype=str)

				first = c[database][collection].with_options(codec_options=CodecOptions(datetime_conversion=DatetimeConversion.DATETIME_CLAMP)).find_one({**find, "ensembleId": ensemble_id}, {"metaData.timeStepMinutes": 1, "metaData.localTimeZone": 1})
				time_step_minutes = first["metaData"]["timeStepMinutes"]

				min_time, max_time = min_max
				forecast_time_start, forecast_time_end = utils.to_cardinal_time(min_time, time_step_minutes), utils.to_cardinal_time(max_time, time_step_minutes)
				forecast_time_start = max(forecast_start_time, forecast_time_start) if forecast_start_time else forecast_time_start
				forecast_times = pd.date_range(forecast_time_start, forecast_time_end, freq=f"{time_step_minutes}min")
				lead_times = np.arange(0, lead_time_days * 24 * 60, time_step_minutes, dtype=np.uint32)
				event_times = np.array([pd.date_range(t0, t0 + timedelta(days=lead_time_days) - timedelta(minutes=time_step_minutes), freq=f"{time_step_minutes}min") for t0 in forecast_times])

				if isinstance(find['moduleInstanceId'], dict):
					for module_instance_id in find['moduleInstanceId']['$in']:
						utils.ensure_path(utils.get_path("Forecast", module_instance_id, find["parameterId"], find["qualifierId"], find["encodedTimeStepId"], ensemble_id))
				else:
					utils.ensure_path(utils.get_path("Forecast", find["moduleInstanceId"], find["parameterId"], find["qualifierId"], find["encodedTimeStepId"], ensemble_id))
				parameters.extend([(connection, database, collection, batch_size, file_format, forecast_time_chunk_size, forecast_times, time_step_minutes, event_times, lead_times, location_ids, location_id, ensemble_id, ensemble_ids, ensemble_member_ids, forecast_time_key, event_time_key, event_value_key, lead_time_days, find, lock) for location_id in location_ids])

		with mp.Pool(threads or mp.cpu_count()) as pool:
			pool.starmap(run_query, parameters)
		# for p in parameters:
		# 	run_query(*p)


def run_query(connection, database, collection, batch_size, file_format, forecast_time_chunk_size, forecast_times, time_step_minutes, event_times, lead_times, location_ids, location_id, ensemble_id, ensemble_ids, ensemble_member_ids, forecast_time_key, event_time_key, event_value_key, lead_time_days, find, lock):
	values = np.full((1, len(forecast_times), 1, len(ensemble_member_ids), len(lead_times)), np.nan, dtype=np.float32)
	forecast_time_chunk_size = min(len(forecast_times), forecast_time_chunk_size)
	module_instance_id = None
	parameter_id = None
	qualifier_id = None
	encoded_time_step_id = None
	with pymongo.MongoClient(connection, connect=True, compressors='snappy') as c:
		meta_data = c[database][collection].with_options(codec_options=CodecOptions(datetime_conversion=DatetimeConversion.DATETIME_CLAMP)).find_one({**find, "locationId": str(location_id), "ensembleId": ensemble_id}, {"metaData": 1})["metaData"]
		actual_forecast_times = sorted(c[database][collection].distinct(forecast_time_key, {**find, "locationId": str(location_id), "ensembleId": ensemble_id}))
		actual_forecast_times = [a for a in actual_forecast_times if forecast_times[0] <= utils.to_cardinal_time(a, time_step_minutes) <= forecast_times[-1]]
		if not actual_forecast_times:
			return
		for forecast_time in np.split(actual_forecast_times, np.arange(batch_size, len(actual_forecast_times), batch_size)):
			for result in c[database][collection].with_options(codec_options=CodecOptions(datetime_conversion=DatetimeConversion.DATETIME_CLAMP)).find(
				{**find, "locationId": str(location_id), forecast_time_key: {"$in": forecast_time.tolist()}, "ensembleId": str(ensemble_id)},
				{"_id": 0, "moduleInstanceId": 1, "parameterId": 1, "qualifierId": 1, "encodedTimeStepId": 1, "ensembleMemberId": 1, forecast_time_key: 1, f"timeseries.{event_time_key}": 1, f"timeseries.{event_value_key}": 1}).batch_size(100):

				forecast_time = utils.to_cardinal_time(result[forecast_time_key], time_step_minutes)
				forecast_time_idx = np.searchsorted(forecast_times, forecast_time)
				ensemble_member_id_idx = np.searchsorted(ensemble_member_ids, result["ensembleMemberId"])

				module_instance_id = result["moduleInstanceId"]
				parameter_id = result["parameterId"]
				qualifier_id = result["qualifierId"]
				encoded_time_step_id = result["encodedTimeStepId"]

				timeseries = [(t[event_value_key], (utils.to_cardinal_time(t[event_time_key], time_step_minutes) - forecast_time).total_seconds() / 60) for t in result["timeseries"] if forecast_time + timedelta(days=lead_time_days) > utils.to_cardinal_time(t[event_time_key], time_step_minutes) >= forecast_time]
				value = np.empty(len(timeseries), dtype=np.float32)
				lead_time = np.empty(len(timeseries), dtype=np.uint32)
				for i, x in enumerate(timeseries):
					value[i], lead_time[i] = x
				lead_time_idx = np.searchsorted(lead_times, lead_time)
				values[0, forecast_time_idx, 0, ensemble_member_id_idx, lead_time_idx] = value

	path = utils.get_path("Forecast", module_instance_id, parameter_id, qualifier_id, encoded_time_step_id, ensemble_id)
	ds = get_dataset(module_instance_id, parameter_id, qualifier_id, encoded_time_step_id, meta_data, forecast_times, values, ensemble_ids, ensemble_id, location_ids, location_id, file_format, ensemble_member_ids, lead_times, event_times, event_time_key)
	write_file(ds, path, file_format, location_ids, location_id, lead_times, ensemble_id, ensemble_member_ids, forecast_time_chunk_size, lock)
	print(f"{path} -> {location_id}")


def write_file(ds, path, file_format, location_ids, location_id, lead_times, ensemble_id, ensemble_member_ids, forecast_time_chunk_size, lock):
	if file_format in ["zarr3"]:
		ds.attrs.update({"locationIds": location_ids, "ensembleMemberIds": ensemble_member_ids})
	else:
		ds.coords["locationId"] = ds.coords["locationId"].astype(f"U{max([len(v) for v in location_ids])}")

	if file_format.startswith("zarr"):
		encoding = {"value": {"chunks": (1, forecast_time_chunk_size, 1, len(ensemble_member_ids), len(lead_times))}}
		with lock:
			utils.write_zarr(ds, path, file_format, encoding)
	elif file_format in ["nc", "netcdf"]:
		encoding = {
			"value": {"dtype": "float32", "_FillValue": np.float32(np.nan), "zlib": True, "complevel": 5, "shuffle": True, "chunksizes": (1, forecast_time_chunk_size, 1, len(ensemble_member_ids), len(lead_times))},
			"eventTime": {"zlib": True, "complevel": 5, "shuffle": True, "chunksizes": (forecast_time_chunk_size, len(lead_times))}
		}
		utils.write_netcdf(ds, path, location_id, encoding)
	else:
		utils.write_csv(ds, path, location_id)


def get_dataset(module_instance_id, parameter_id, qualifier_id, encoded_time_step_id, meta_data, forecast_times, values, ensemble_ids, ensemble_id, location_ids, location_id, file_format, ensemble_member_ids, lead_times, event_times, event_time_key):
	return xr.Dataset(
		data_vars={
			"value": (["locationId", "forecastTime", "ensembleId", "ensembleMemberId", "leadTime"], values, {
				"units": meta_data["unit"],
				"coordinates": "eventTime"
			})
		},
		coords={
			"locationId": (["locationId"], [np.uint32(np.searchsorted(location_ids, location_id)) if file_format in ["zarr3"] else location_id], {
				"cf_role": "timeseries_id",
				"long_name": "Location identifier"
			}),
			"forecastTime": (["forecastTime"], forecast_times, {
				"standard_name": "forecast_reference_time",
				"long_name": "Forecast reference time"
			}),
			"ensembleId": (["ensembleId"], [np.uint32(np.searchsorted(ensemble_ids, ensemble_id)) if file_format in ["zarr3"] else ensemble_id], {
				"long_name": "Ensemble identifier"
			}),
			"ensembleMemberId": (["ensembleMemberId"], np.arange(len(ensemble_member_ids), dtype=np.uint16) if file_format in ["zarr3"] else ensemble_member_ids, {
				"long_name": "Ensemble member identifier"
			}),
			"leadTime": (["leadTime"], lead_times, {
				"standard_name": "forecast_period",
				"long_name": "Lead time minutes from forecast reference time",
				"units": "minutes"
			}),
			"eventTime": (["forecastTime", "leadTime"], event_times, {
				"long_name": "Forecast valid time",
			}),
		},
		attrs=utils.get_attrs(module_instance_id, parameter_id, qualifier_id, encoded_time_step_id, meta_data, event_time_key)
	)


if __name__ == '__main__':
	main()
