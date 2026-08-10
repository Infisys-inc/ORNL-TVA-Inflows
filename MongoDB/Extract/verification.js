var observed_queries = [];
db.getCollection('Observed').find({}).forEach(f => { f["Filters"].forEach(t => {
	if (!f["Name"].startsWith("HT_"))
		observed_queries.push({
			"process": true,
			"collection": f["Collection"],
			"event_start_time_key": "startTime",
			"event_end_time_key": "endTime",
			"event_time_key": "t",
			"event_value_key": "v",
			"observed_time_chunk_size": 1000,
			"file_format": "nc",
			"aggregate": [{"$match": t["Filter"]}],
			"observed_start_time": {"$date": "2017-01-01T00:00:00Z"},
			"threads": 16
		});
	});
});
print(observed_queries);

var forecast_queries = [];
db.getCollection('Forecast').find({}).forEach(f => { f["Filters"].forEach(t => {
	if (!f["Name"].startsWith("HT_"))
		forecast_queries.push( {
			"process": true,
			"collection": f["Collection"],
			"aggregate": [{"$match": t["Filter"]}],
			"forecast_time_key": "forecastTime",
			"event_time_key": "t",
			"event_value_key": "v",
			"lead_time_days": 14,
			"forecast_time_chunk_size": 1000,
			"file_format" : "nc",
			"batch_size" : 100,
			"forecast_start_time" : null,
			"threads" : 16
		});
	});
});
print(forecast_queries);