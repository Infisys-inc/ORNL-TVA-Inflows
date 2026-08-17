import re
import utils
import pymongo

import numpy as np
import xarray as xr
import pandas as pd
import multiprocessing as mp

from bson import json_util
from datetime import timedelta

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
			for parameter_id in [m for m in db.distinct("parameterId", {"moduleInstanceId": module_instance_id}) if not query["filter"]["parameterId"] or any([re.match(f, m) for f in query["filter"]["parameterId"]])]:
				for qualifier_id in [m for m in db.distinct("qualifierId", {"moduleInstanceId": module_instance_id, "parameterId": parameter_id}) if not query["filter"]["qualifierId"] or any([re.match(f, m) for f in query["filter"]["qualifierId"]])]:
					for ensemble_id in [m for m in db.distinct("ensembleId", {"moduleInstanceId": module_instance_id, "parameterId": parameter_id, "qualifierId": qualifier_id}) if not query["filter"]["ensembleId"] or any([re.match(f, m) for f in query["filter"]["ensembleId"]])]:
						for encoded_time_step_id in [m for m in db.distinct("encodedTimeStepId", {"moduleInstanceId": module_instance_id, "parameterId": parameter_id, "qualifierId": qualifier_id, "ensembleId": ensemble_id}) if not query["filter"]["encodedTimeStepId"] or any([re.match(f, m) for f in query["filter"]["encodedTimeStepId"]])]:
							utils.ensure_path(utils.get_path("Forecast", module_instance_id, parameter_id, qualifier_id, encoded_time_step_id, ensemble_id), query["resume"])
							location_ids = sorted([m for m in db.distinct("locationId", {"moduleInstanceId": module_instance_id, "parameterId": parameter_id, "qualifierId": qualifier_id, "ensembleId": ensemble_id, "encodedTimeStepId": encoded_time_step_id}) if not query["filter"]["locationId"] or any([re.match(f, m) for f in query["filter"]["locationId"]])])
							for location_id in location_ids:
								filter = {"moduleInstanceId": module_instance_id, "parameterId": parameter_id, "qualifierId": qualifier_id, "ensembleId": ensemble_id, "encodedTimeStepId": encoded_time_step_id, "locationId": location_id}
								parameters.append((config["connection"], config["database"], query["collection"], query["resume"], query["max_age_hours"], query["lead_time_days"], query["batch_size"], query["file_format"], query["forecast_time_chunk_size"], query["forecast_time_key"], query["event_time_key"], query["event_value_key"], filter, location_ids, lock))
	return parameters


def run_query(connection, database, collection, resume, max_age_hours, lead_time_days, batch_size, file_format, forecast_time_chunk_size, forecast_time_key, event_time_key, event_value_key, filter, location_ids, lock):
	with pymongo.MongoClient(connection, connect=True, compressors='snappy') as c:
		db = c[database][collection]
		time_step_minutes = utils.get_time_step_minutes(db, filter)
		forecast_times = get_forecast_times(db, filter, forecast_time_key, time_step_minutes)
		event_times = get_event_times(lead_time_days, time_step_minutes, forecast_times)

		lead_times = np.arange(0, lead_time_days * 24 * 60, time_step_minutes, dtype=np.uint32)
		ensemble_member_ids = sorted(db.distinct("ensembleMemberId", filter))
		forecast_time_chunk_size = min(len(forecast_times), forecast_time_chunk_size)

		path = utils.get_path("Forecast", filter["moduleInstanceId"], filter["parameterId"], filter["qualifierId"], filter["encodedTimeStepId"], filter["ensembleId"])
		if resume and utils.exists(path, filter["locationId"], file_format, max_age_hours):
			print(f"{path} -> {filter["locationId"]} [resumed]")
		else:
			meta_data = utils.get_meta_data(db, filter)
			values = get_values(db, filter, batch_size, forecast_time_key, event_time_key, event_value_key, time_step_minutes, forecast_times, ensemble_member_ids, lead_times, lead_time_days)
			dataset = get_dataset(filter, meta_data, forecast_times, values, location_ids, file_format, ensemble_member_ids, lead_times, event_times, event_time_key)
			write_file(dataset, path, file_format, filter["ensembleId"], location_ids, filter["locationId"], lead_times, ensemble_member_ids, forecast_time_chunk_size, lock)
			print(f"{path} -> {filter["locationId"]}")


def get_event_times(lead_time_days, time_step_minutes, forecast_times):
	return np.array([pd.date_range(t0, t0 + timedelta(days=lead_time_days) - timedelta(minutes=time_step_minutes), freq=f"{time_step_minutes}min") for t0 in forecast_times])


def get_forecast_times(db, filter, forecast_time_key, time_step_minutes):
	min_max = db.with_options(codec_options=utils.codec_options).aggregate([{"$match": filter}, {"$group": {"_id": None, "min": {"$min": f"${forecast_time_key}"}, "max": {"$max": f"${forecast_time_key}"}}}]).next()
	forecast_time_start, forecast_time_end = utils.to_cardinal_time(min_max["min"], time_step_minutes), utils.to_cardinal_time(min_max["max"], time_step_minutes)
	return pd.date_range(forecast_time_start, forecast_time_end, freq=f"{time_step_minutes}min")


def get_values(db, filter, batch_size, forecast_time_key, event_time_key, event_value_key, time_step_minutes, forecast_times, ensemble_member_ids, lead_times, lead_time_days):
	values = np.full((1, len(forecast_times), 1, len(ensemble_member_ids), len(lead_times)), np.nan, dtype=np.float32)
	actual_forecast_times = sorted(db.distinct(forecast_time_key, filter))
	for forecast_time in np.split(actual_forecast_times, np.arange(batch_size, len(actual_forecast_times), batch_size)):
		for result in db.with_options(codec_options=utils.codec_options).find({**filter, forecast_time_key: {"$in": forecast_time.tolist()}}, {"_id": 0, "ensembleMemberId": 1, forecast_time_key: 1, f"timeseries.{event_time_key}": 1, f"timeseries.{event_value_key}": 1, f"timeseries.f": 1}):
			forecast_time = utils.to_cardinal_time(result[forecast_time_key], time_step_minutes)

			forecast_time_idx = np.searchsorted(forecast_times, forecast_time)
			if forecast_time_idx == len(forecast_times) or forecast_time != forecast_times[forecast_time_idx]:
				raise ValueError(f"forecast time not found: {forecast_time} -> {filter}")

			ensemble_member_id_idx = np.searchsorted(ensemble_member_ids, result["ensembleMemberId"])
			if ensemble_member_id_idx == len(ensemble_member_ids) or ensemble_member_ids[ensemble_member_id_idx] != result["ensembleMemberId"]:
				raise ValueError(f"ensemble member not found: {result["ensembleMemberId"]} -> {filter}")

			timeseries = [(t[event_value_key] if t["f"] <= 2 else np.float32(np.nan), (utils.to_cardinal_time(t[event_time_key], time_step_minutes) - forecast_time).total_seconds() / 60) for t in result["timeseries"] if forecast_time + timedelta(days=lead_time_days) > utils.to_cardinal_time(t[event_time_key], time_step_minutes) >= forecast_time]
			value = np.empty(len(timeseries), dtype=np.float32)
			lead_time = np.empty(len(timeseries), dtype=np.uint32)
			for i, x in enumerate(timeseries):
				value[i], lead_time[i] = x

			lead_time_idx = np.searchsorted(lead_times, lead_time)
			missing = ~((lead_time_idx < len(lead_times)) & (lead_times[lead_time_idx] == lead_time))
			if np.any(missing):
				raise ValueError(f"lead times not found: {lead_time[missing]} -> {filter}")

			values[0, forecast_time_idx, 0, ensemble_member_id_idx, lead_time_idx] = value
	return values


def write_file(ds, path, file_format, ensemble_id, location_ids, location_id, lead_times, ensemble_member_ids, forecast_time_chunk_size, lock):
	if file_format == "zarr3":
		ds.attrs.update({"locationIds": location_ids, "ensembleIds": [ensemble_id], "ensembleMemberIds": ensemble_member_ids})
		encoding = {"value": {"chunks": (1, forecast_time_chunk_size, 1, len(ensemble_member_ids), len(lead_times))}}
		with lock:
			utils.write_zarr(ds, path, file_format, encoding)
	elif file_format == "nc":
		encoding = {
			"value": {"dtype": "float32", "_FillValue": np.float32(np.nan), "zlib": True, "complevel": 5, "shuffle": True, "chunksizes": (1, forecast_time_chunk_size, 1, len(ensemble_member_ids), len(lead_times))},
			"eventTime": {"zlib": True, "complevel": 5, "shuffle": True, "chunksizes": (forecast_time_chunk_size, len(lead_times))}
		}
		utils.write_netcdf(ds, path, location_id, encoding)
	else:
		utils.write_csv(ds, path, location_id)


def get_dataset(filter, meta_data, forecast_times, values, location_ids, file_format, ensemble_member_ids, lead_times, event_times, event_time_key):
	return xr.Dataset(
		data_vars={
			"value": (["locationId", "forecastTime", "ensembleId", "ensembleMemberId", "leadTime"], values, {
				"units": meta_data["unit"],
				"coordinates": "eventTime"
			})
		},
		coords={
			"locationId": (["locationId"], [np.uint32(np.searchsorted(np.array(location_ids), filter["locationId"])) if file_format == "zarr3" else filter["locationId"]], {
				"cf_role": "timeseries_id",
				"long_name": "Location identifier"
			}),
			"forecastTime": (["forecastTime"], forecast_times, {
				"standard_name": "forecast_reference_time",
				"long_name": "Forecast reference time"
			}),
			"ensembleId": (["ensembleId"], [np.uint32(0) if file_format == "zarr3" else filter["ensembleId"]], {
				"long_name": "Ensemble identifier"
			}),
			"ensembleMemberId": (["ensembleMemberId"], np.arange(len(ensemble_member_ids), dtype=np.uint16) if file_format == "zarr3" else ensemble_member_ids, {
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
		attrs=utils.get_attrs(filter, meta_data, event_time_key)
	)


if __name__ == '__main__':
	main()
