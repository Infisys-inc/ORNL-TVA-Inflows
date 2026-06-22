import json
import os
import shutil
from datetime import datetime, timedelta


def to_cardinal_time(t, time_step_minutes):
	date = datetime(t.year, t.month, t.day)
	minutes = t.hour * 60 + t.minute
	remainder = minutes % time_step_minutes
	if remainder > 0:
		minutes += time_step_minutes - remainder if remainder * 2 >= time_step_minutes else -remainder
	return date + timedelta(minutes=minutes)


def get_folder_path(folder_overrides, aggregate, ensemble_id, time_step, timezone, file_format):
	module_instance_id = folder_overrides.get("moduleInstanceId", aggregate[0]["$match"]["moduleInstanceId"])
	parameter_id = folder_overrides.get("parameterId", aggregate[0]["$match"]["parameterId"])
	qualifier_id = folder_overrides.get("qualifierId", "_".join(json.loads(aggregate[0]["$match"]["qualifierId"])))
	return f'{module_instance_id}-{parameter_id}-{qualifier_id}-{ensemble_id}-{time_step}-{timezone}.{file_format}'


def ensure_path(path):
	if os.path.exists(path):
		shutil.rmtree(path)
	os.makedirs(path)


def write_zarr(ds, path, file_format, encoding):
	zarr_format = 3 if file_format == "zarr3" else 2
	if os.path.exists(path) and os.listdir(path):
		ds.to_zarr(path, mode="a", consolidated=False, zarr_format=zarr_format, append_dim="locationId")
	else:
		ds.to_zarr(path, mode="w", consolidated=False, zarr_format=zarr_format, encoding={"value": encoding})


def write_netcdf(ds, path, location_id, ensemble_id, encoding):
	parts = path.split(".")
	ds.to_netcdf(os.path.join(path, f"{'.'.join(parts[:-1])}-{location_id}-{ensemble_id}.{parts[-1]}"), mode="w", encoding={"value": encoding})


def write_csv(ds, path, location_id, ensemble_id):
	parts = path.split(".")
	ds.reset_index().to_csv(os.path.join(path, f"{'.'.join(parts[:-1])}-{location_id}-{ensemble_id}.{parts[-1]}"), index=False)


def get_attrs(module_instance_id, parameter_id, qualifier_id, encoded_time_step_id, meta_data, event_time_key):
	return {
			**{k: str(v) if isinstance(v, datetime) else v for k, v in meta_data.items() if k not in ["locationName"]},
			"moduleInstanceId": module_instance_id,
			"parameterId": parameter_id,
			"qualifierId": qualifier_id,
			"encodedTimeStepId": encoded_time_step_id,
			"isLocalTime": str(event_time_key == "lt"),
			"Conventions": "CF-1.12"
		}
