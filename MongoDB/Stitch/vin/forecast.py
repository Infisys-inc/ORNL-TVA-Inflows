import os
import re
import json
import pathlib
import shutil

import numpy as np
import pandas as pd
import xarray as xr
import multiprocessing as mp


filter_seen = True
filter_locations = False
append_upstream_data = True


def main():
	seen = {}
	location_attributes = pd.read_csv("location_attributes.csv", sep='\t', index_col=['LocationID'], dtype={'USGSID': str}).to_dict(orient='index')
	dam_attributes = pd.read_csv("dam_attributes.csv", sep='\t', index_col=['DamID'], dtype=str, keep_default_na=False).to_dict(orient='index')
	measure_attributes = pd.read_csv("forecast_measure_attributes.csv", sep='\t', index_col=['folder']).to_dict(orient='index')

	with open('forecast.json', 'r') as r:
		forecast = json.load(r)

	data_root = forecast['data_root']

	for timestep in forecast['timesteps']:
		folder_patterns = timestep['folders']
		timestep_hours = timestep['timestep_hours']
		upstream_dams_flow_folder = timestep['upstream_dams_flow_folder']
		tributaries_flow_folder = timestep['tributaries_flow_folder']

		path = f'Forecast_{timestep_hours}H'
		if os.path.exists(path):
			shutil.rmtree(path)
		os.makedirs(path)

		folders = [m.name for f in folder_patterns for m in pathlib.Path(data_root).glob(f) if m.is_dir()]
		locations = {(pathlib.Path(location).stem, pathlib.Path(location).suffix) for f in folders for location in os.listdir(os.path.join(data_root, f))}
		locations = [(location, extension) for location, extension in locations if location in location_attributes] if filter_locations else locations
		parameters = [(data_root, folders, location, extension, timestep_hours, location_attributes, location_attributes.get(location, {}), dam_attributes.get(location, {}), upstream_dams_flow_folder, tributaries_flow_folder, measure_attributes, path, seen.get(location, set())) for location, extension in locations]
		with mp.Pool(mp.cpu_count()) as pool:
			[seen.update({location: timeseries | seen.get(location, set())}) for location, timeseries in pool.starmap(read, parameters) if filter_seen]
	# for p in parameters:
	# 	location, timeseries = read(*p)
	# 	seen.update({location: timeseries | seen.get(location, set())})


def read(data_root, folders, location, extension, timestep_hours, all_location_attributes, location_attributes, dam_attributes, upstream_dams_flow_folder, tributaries_flow_folder, measure_attributes, path, seen):
	forecast_data = [d for d in [get_forecast_data(data_root, folder, location, extension, next((v for k, v in measure_attributes.items() if re.match(k, folder)), {})) for folder in folders] if d is not None and d.name not in seen]
	upstream_data = []
	tributary_data = []

	if append_upstream_data:
		for upstream_dam in dam_attributes.get('UpstreamDams', '').split(','):
			data = get_forecast_data(data_root, upstream_dams_flow_folder, upstream_dam, extension, next((v for k, v in dam_attributes.items() if re.match(k, upstream_dams_flow_folder)), {}))
			if data is not None:
				location_name = all_location_attributes.get(upstream_dam, {}).get('Location', location)
				data = data.rename(f'{data.name}_[{upstream_dam}-US-DAM]')
				data.attrs['long_name'] = f'{data.attrs["long_name"]} [{upstream_dam}-US-DAM]'
				data.attrs['station_id'] = upstream_dam
				data.attrs['station_name'] = location_name
				upstream_data.append(data)

	[d.rename(f'{d.name}_[{t}-US-TRIB]') for t, d in [(tributary, get_forecast_data(data_root, tributaries_flow_folder, tributary, extension, next((v for k, v in measure_attributes.items() if re.match(k, tributaries_flow_folder)), {}))) for tributary in dam_attributes.get('Tributaries', '').split(',')] if d is not None]

	if forecast_data:
		combined = combine(forecast_data + upstream_data + tributary_data)
		write(combined, location, extension, timestep_hours, location_attributes, dam_attributes, path)
		print(location)
	return location, {n.name for n in forecast_data}


def get_forecast_data(data_root, folder, location, extension, measure_attributes):
	path = os.path.join(data_root, folder, f'{location}{extension}')
	if os.path.exists(path):
		with xr.open_dataset(path) as ds:
			name = f'{ds.attrs['moduleInstanceId']}_{ds.attrs['parameterId']}_{ds.attrs['qualifierId']}_{ds.attrs['ensembleId'] or 'main'}'
			da = ds['value'].squeeze('locationId', drop=True).drop_vars(['eventTime', 'ensembleId', 'ensembleMemberId']).rename(name).load()
			da.encoding.pop('coordinates', None)
			da.attrs = {
				'long_name': measure_attributes.get('long_name', ds.attrs['parameterName']),
				'description': measure_attributes.get('description', ''),
				'units': ds.attrs['unit'],
				'coordinates': 'lat lon'
			}
			return da
	return None


def combine(forecast_data):
	forecast = xr.merge(forecast_data, join='outer')
	forecast = forecast.dropna(dim='forecastTime', how='all')
	forecast['forecastTime'].encoding.clear()
	return forecast


def write(combined, location, extension, timestep_hours, location_attributes, dam_attributes, path):
	combined = format_netcdf(combined, location, timestep_hours, location_attributes, dam_attributes).rename({'leadTime': 'lead_time', 'forecastTime': 'forecast_time'})
	encoding = {
		'lead_time': {'dtype': 'int16', 'zlib': True, 'complevel': 5, 'shuffle': True},
		'forecast_time': {'dtype': 'int64', 'units': 'hours since 1970-01-01', 'calendar': 'proleptic_gregorian', 'zlib': True, 'complevel': 5, 'shuffle': True},
		**{name: {'dtype': 'float32', '_FillValue': np.float32(np.nan), 'zlib': True, 'complevel': 5, 'shuffle': True} for name in combined.data_vars}
	}
	combined.to_netcdf(os.path.join(path, f'{location}{extension}'), encoding=encoding)


def format_netcdf(ds, location, timestep_hours, location_attributes, dam_attributes):
	lat = location_attributes.get('Latitude', 0)
	lon = location_attributes.get('Longitude', 0)
	usgs_id = location_attributes.get('USGSID', '')
	location_name = location_attributes.get('Location', location)


	ds = ds.assign_coords(
		lat=xr.DataArray(lat, attrs={'standard_name': 'latitude', 'long_name': 'station latitude', 'units': 'degrees_north', 'axis': 'Y'}),
		lon=xr.DataArray(lon, attrs={'standard_name': 'longitude', 'long_name': 'station longitude', 'units': 'degrees_east', 'axis': 'X'}),
	)

	ds.attrs = {
		'title': '',
		'station_id': location,
		'station_name': location_name,
		'usgs_id': usgs_id,
		'station_latitude_degrees_north': lat,
		'station_longitude_degrees_east': lon,
		'time_coverage_start': pd.Timestamp(ds.forecastTime.values[0]).strftime('%Y-%m-%d %H:%M:%S'),
		'time_coverage_end': pd.Timestamp(ds.forecastTime.values[-1]).strftime('%Y-%m-%d %H:%M:%S'),
		'time_frequency': f'{timestep_hours}H',
		'Conventions': 'CF-1.8',
		**dam_attributes
	}
	return ds


if __name__ == '__main__':
	main()
