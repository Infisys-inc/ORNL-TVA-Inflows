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


def get_path(base, module_instance_id, parameter_id, qualifier_id, encoded_time_step_id, ensemble_id):
	qualifier_id = "_".join(json.loads(qualifier_id))
	return os.path.join(base, f'{module_instance_id}-{parameter_id}-{qualifier_id}-{encoded_time_step_id}-{ensemble_id}')


def ensure_path(path):
	if os.path.exists(path):
		shutil.rmtree(path)
	os.makedirs(path)


def write_zarr(ds, path, file_format, encoding):
	zarr_format = 3 if file_format == "zarr3" else 2
	if os.path.exists(path) and os.listdir(path):
		ds.to_zarr(path, mode="a", consolidated=False, zarr_format=zarr_format, append_dim="locationId")
	else:
		ds.to_zarr(path, mode="w", consolidated=False, zarr_format=zarr_format, encoding=encoding)


def write_netcdf(ds, path, location_id, encoding):
	ds.to_netcdf(os.path.join(path, f"{location_id}.nc"), mode="w", encoding=encoding)


def write_csv(ds, path, location_id):
	ds.reset_index().to_csv(os.path.join(path, f"{location_id}.csv"), index=False)


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
