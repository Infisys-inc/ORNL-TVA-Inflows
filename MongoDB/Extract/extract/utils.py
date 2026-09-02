import json
import os
import re
import shutil
from datetime import datetime, timedelta

from bson import CodecOptions, DatetimeConversion
from netCDF4 import Dataset

codec_options = CodecOptions(datetime_conversion=DatetimeConversion.DATETIME_CLAMP)


def to_cardinal_time(t, time_step_minutes):
	date = datetime(t.year, t.month, t.day)
	minutes = t.hour * 60 + t.minute
	remainder = minutes % time_step_minutes
	if remainder > 0:
		minutes += time_step_minutes - remainder if remainder * 2 >= time_step_minutes else -remainder
	return date + timedelta(minutes=minutes)


def get_path(base, module_instance_id, parameter_id, qualifier_id, encoded_time_step_id, ensemble_id=""):
	qualifier_id = "_".join(json.loads(qualifier_id))
	return os.path.join(base, f'{module_instance_id}-{parameter_id}-{qualifier_id}-{encoded_time_step_id}-{ensemble_id}')


def ensure_path(path, resume=False):
	if not resume and os.path.exists(path):
		shutil.rmtree(path)
	os.makedirs(path, exist_ok=True)


def write_zarr(ds, path, encoding):
	if os.path.exists(path) and os.listdir(path):
		ds.to_zarr(path, mode="a", consolidated=False, zarr_format=3, append_dim="locationId")
	else:
		ds.to_zarr(path, mode="w", consolidated=False, zarr_format=3, encoding=encoding)


def write_netcdf(ds, path, location_id, encoding):
	ds.to_netcdf(os.path.join(path, f"{clean(location_id)}.nc"), mode="w", encoding=encoding)


def write_csv(ds, path, location_id):
	ds.reset_index().to_csv(os.path.join(path, f"{clean(location_id)}.csv"), index=False)


def clean(s):
	return re.sub(r'[<>:"/\\|?*\x00-\x1F]', "_", s).rstrip(". ")


def exists(path, location_id, file_format, max_age_hours):
	path = os.path.join(path, f"{clean(location_id)}.{file_format}")
	if os.path.exists(path) and datetime.now() - datetime.fromtimestamp(os.path.getmtime(path)) <= timedelta(hours=max_age_hours):
		try:
			with Dataset(path):
				return True
		except:
			return False
	return False


def get_time_step_minutes(db, filter):
	return db.find_one(filter, {"metaData.timeStepMinutes": 1})["metaData"]["timeStepMinutes"]


def get_meta_data(db, filter):
	return db.with_options(codec_options=codec_options).find_one(filter, {"metaData": 1})["metaData"]


def get_attrs(filter, meta_data, event_time_key):
	return {
			**{k: str(v) if isinstance(v, datetime) else v for k, v in meta_data.items()},
			**filter,
			"isLocalTime": str(event_time_key == "lt"),
			"Conventions": "CF-1.12"
		}
