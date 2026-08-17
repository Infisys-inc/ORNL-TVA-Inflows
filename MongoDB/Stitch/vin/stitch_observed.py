import os
import json
import pathlib
import shutil

import numpy as np
import pandas as pd
import xarray as xr
import multiprocessing as mp


def main():
	for f in ['hourly_observed.json', 'six_hour_observed.json']:
		with open(f, 'r') as r:
			config = json.load(r)

		folder_patterns = config['folders']
		data_root = config['data_root']
		timestep_hours = config['timestep_hours']
		location_lat_lon = pd.read_csv("location_lat_lon.csv", sep='\t', index_col=['LocationID'])

		path = f'Observed_{timestep_hours}H'
		if os.path.exists(path):
			shutil.rmtree(path)
		os.makedirs(path)

		folders = [m.name for f in folder_patterns for m in pathlib.Path(data_root).glob(f) if m.is_dir()]
		locations = {(pathlib.Path(location).stem, pathlib.Path(location).suffix) for f in folders for location in os.listdir(os.path.join(data_root, f))}
		mp.Pool(mp.cpu_count()).starmap(read, [(data_root, folders, location, extension, timestep_hours, location_lat_lon, path) for location, extension in locations])
		# for location in locations:
		# 	read(location)


def read(data_root, folders, location, extension, timestep_hours, location_lat_lon, path):
	observed_data = [d for d in [get_observed_data(data_root, folder, location, extension) for folder in folders] if d is not None]
	if observed_data:
		combined = combine(observed_data)
		write(combined, location, extension, timestep_hours, location_lat_lon, path)
		print(location)


def get_observed_data(data_root, folder, location, extension):
	path = os.path.join(data_root, folder, f'{location}{extension}')
	if os.path.exists(path):
		with xr.open_dataset(path) as ds:
			name = f'{ds.attrs['moduleInstanceId']}_{ds.attrs['parameterId']}_{ds.attrs['qualifierId']}'
			da = ds['value'].squeeze('locationId', drop=True).rename(name).load()
			da.attrs = {
				'long_name': ds.attrs['parameterName'],
				'units': ds.attrs['unit'],
				'coordinates': 'lat lon'
			}
			return da
	return None


def combine(observed_data):
	observed = xr.merge(observed_data, join='outer')
	observed = observed.dropna(dim='eventTime', how='all')
	observed['eventTime'].encoding.clear()
	return observed


def write(combined, location, extension, timestep_hours, location_lat_lon, path):
	combined = format_netcdf(combined, location, timestep_hours, location_lat_lon)
	encoding = {
		'time': {'dtype': 'int64', 'units': 'hours since 1970-01-01', 'calendar': 'proleptic_gregorian', 'zlib': True, 'complevel': 5, 'shuffle': True},
		**{name: {'dtype': 'float32', '_FillValue': np.float32(np.nan), 'zlib': True, 'complevel': 5, 'shuffle': True} for name in combined.data_vars}
	}
	combined.to_netcdf(os.path.join(path, f'{location}{extension}'), encoding=encoding)


def format_netcdf(ds, location, timestep_hours, location_lat_lon):
	if location in location_lat_lon.index:
		lat = location_lat_lon.loc[location, 'Latitude']
		lon = location_lat_lon.loc[location, 'Longitude']
		location_name = location_lat_lon.loc[location, 'Location']
	else:
		lat = 0
		lon = 0
		location_name = location

	ds = ds.rename(eventTime='time')

	ds = ds.assign_coords(
		lat=xr.DataArray(lat, attrs={'standard_name': 'latitude', 'long_name': 'station latitude', 'units': 'degrees_north', 'axis': 'Y'}),
		lon=xr.DataArray(lon, attrs={'standard_name': 'longitude', 'long_name': 'station longitude', 'units': 'degrees_east', 'axis': 'X'}),
	)

	ds['time'].attrs = {'standard_name': 'time', 'long_name': 'time'}

	ds.attrs = {
		'title': '',
		'station_id': location,
		'station_name': location_name,
		'station_latitude_degrees_north': lat,
		'station_longitude_degrees_east': lon,
		'time_coverage_start': pd.Timestamp(ds.time.values[0]).strftime('%Y-%m-%d %H:%M:%S'),
		'time_coverage_end': pd.Timestamp(ds.time.values[-1]).strftime('%Y-%m-%d %H:%M:%S'),
		'time_frequency': f'{timestep_hours}H',
		'Conventions': 'CF-1.8',
		'note': ''
	}
	return ds


if __name__ == '__main__':
	main()
