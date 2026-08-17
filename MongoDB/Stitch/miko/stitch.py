import os
import json
import xarray as xr


with open('six_hour.json', 'r') as r:
# with open('hourly.json', 'r') as r:
	config = json.load(r)

locations = config['locations']
data_root = config['data_root']
timestep_hours = config['timestep_hours']
lead_hours = config['lead_hours']
forward_fill_hours = config['forward_fill_hours']


def main():
	for location, measures in locations.items():
		observed_data = [get_observed_data(location, m['observed']) for m in measures]
		forecast_data = [get_forecasts_data(location, forecasts, m['n'], m['aggregation']) for m in measures for forecasts in m['forecasts']]
		if observed_data:
			combined = combine(observed_data, forecast_data)
			write(combined, location)


def get_forecasts_data(location, forecast, n, aggregation):
	current_forecast = None
	for forecast in forecast:
		f = get_forecast_data(location, forecast['forecast'], forecast['ensemble_member'])
		current_forecast = f if current_forecast is None else xr.concat([current_forecast, f.dropna("eventTime", how='all')], dim="eventTime", join='exact', combine_attrs='drop').drop_duplicates("eventTime", keep="last").sortby("eventTime")

	if current_forecast is not None:
		current_forecast = current_forecast.assign(aggregate(current_forecast, n, aggregation))

	return current_forecast


def get_forecast_data(location, forecast, ensemble_member):
	with xr.open_dataset(os.path.join(data_root, "Forecast", forecast, f"{location}.nc")) as ds:
		da = ds['value'].squeeze('locationId', drop=True).sel(ensembleMemberId=ensemble_member).rename(f'{ds.attrs['parameterName'].replace(" ", "-")}_{ds.attrs['unit']}')
		da = da.sel(leadTime=((60 <= da['leadTime']) & (da['leadTime'] <= (lead_hours + forward_fill_hours) * 60))).drop_vars("eventTime").load()
		da = ffill(da)
		lead_times = da.drop_vars(['ensembleId', 'ensembleMemberId']).rename({'forecastTime': 'eventTime'}).to_dataset(dim="leadTime")
		lead_times = lead_times.rename({t: f'{da.name}_Forecast_{t // 60:03d}h' for t in lead_times.data_vars})
		return lead_times


def get_observed_data(location, observed):
	with xr.open_dataset(os.path.join(data_root, "Observed", observed, f"{location}.nc")) as ds:
		return ds['value'].squeeze('locationId', drop=True).rename(f'{ds.attrs['parameterName'].replace(" ", "-")}_{ds.attrs['unit']}')


def ffill(da):
	filled = da.copy()
	other_dims = [d for d in da.dims if d != "forecastTime"]

	for age in range(1, forward_fill_hours // timestep_hours + 1, timestep_hours):
		candidates = da.shift(forecastTime=age, leadTime=-age)
		replace_rows = filled.isnull().all(dim=other_dims) & candidates.notnull().any(dim=other_dims)
		filled = filled.where(~replace_rows, candidates)

	return filled.sel(leadTime=da['leadTime'] <= lead_hours * 60)


def combine(observed_data, forecast_data):
	observed = xr.merge(observed_data, join='inner', combine_attrs='drop')
	observed = observed.dropna(dim='eventTime', how='any')
	combined = xr.merge([observed, *forecast_data], join='left', combine_attrs='drop')
	combined['eventTime'].encoding.clear()
	return combined


def write(combined, location):
	combined.to_netcdf(os.path.join(data_root, f'{location}_{timestep_hours}h.nc'))


def aggregate(forecast_data, n, aggregation):
	d = {}
	[d.setdefault((int(name.split("_")[-1][:-1]) - 1) // n, []).append(forecast_data[name]) for name in forecast_data]
	aggregated = {'_'.join(v[0].name.split('_')[:-1] + [aggregation, f'{k * n + n:03d}h']): getattr(xr.concat(v, dim='_agg'), aggregation)(dim='_agg', skipna=aggregation != 'sum') for k, v in d.items()}

	if aggregation == 'sum':
		aggregated.update(cumulative(aggregated))

	return aggregated


def cumulative(aggregated):
	accumulator = None
	accumulated = {}

	for name, da in aggregated.items():
		accumulator = da if accumulator is None else accumulator + da
		accumulated[name.replace('_sum_', '_cumulative_')] = accumulator

	return accumulated


if __name__ == '__main__':
	main()
