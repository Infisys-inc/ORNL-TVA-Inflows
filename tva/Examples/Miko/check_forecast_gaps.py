from datetime import datetime

import pandas as pd
from pymongo import MongoClient


def get_timeseries_avg_forecast_df(
    location: str,
    series_start_date: datetime,
    max_lead_time: int,
    matching: dict,
    projection: dict,
    collection,
) -> pd.DataFrame:
    matching["locationId"] = location
    matching["forecastTime"] = {"$gte": series_start_date}

    projection["locationId"] = "$locationId"
    projection["_id"] = 0

    it = collection.aggregate(
        [
            {"$match": matching},
            {"$sort": {"locationId": 1, "bucket": 1}},
            {"$unwind": "$timeseries"},
            {"$match": {"$expr": {"$gte": ["$timeseries.t", "$forecastTime"]}}},
            {
                "$match": {
                    "$expr": {
                        "$lte": [
                            {
                                "$dateDiff": {
                                    "startDate": "$forecastTime",
                                    "endDate": "$timeseries.t",
                                    "unit": "hour",
                                }
                            },
                            max_lead_time,
                        ]
                    }
                }
            },
            {"$project": projection},
        ]
    )

    rows = list(it)
    return pd.DataFrame(rows)


def main() -> None:
    connection_uri = "mongodb://fews_admin:AvjgHB1zgtBS*T#@knxpwmongodb1.main.tva.gov:27017/FEWS_ARCHIVE?authSource=admin&tls=true"
    client = MongoClient(connection_uri)

    location = "SMFV2"
    max_lead_time = 6
    date = datetime(2001, 1, 1, 0, 0, 0)

    db = client["FEWS_ARCHIVE"]
    collection = db["ExternalForecastingScalarTimeSeries"]

    matching = {
        "parameterId": "MAP",
        "moduleInstanceId": {"$in": ["QPF_to_MAP"]},
        "locationId": {"$in": ["GATV2"]},
        "forecastTime": {"$gte": date},
        "encodedTimeStepId": "SETS60",
        "qualifierId": '["HRRR"]',
    }

    projection = {
        "DateTime": "$timeseries.t",
        "ForecastedMAP": "$timeseries.v",
        "_id": 0,
        "locationId": "$locationId",
        "ForecastTime": "$forecastTime",
    }

    df = get_timeseries_avg_forecast_df(
        location=location,
        series_start_date=date,
        max_lead_time=max_lead_time,
        matching=matching,
        projection=projection,
        collection=collection,
    )

    df["avgValue"] = df.groupby("ForecastTime")["ForecastedMAP"].transform("mean")
    df2 = df.drop_duplicates(subset=["ForecastTime"])

    forecast_times = pd.to_datetime(df2["ForecastTime"]).sort_values().drop_duplicates().reset_index(drop=True)
    diffs = forecast_times.diff().dropna()
    gaps = diffs[diffs != pd.Timedelta(hours=1)]
    six_hour_gap_idx = diffs[diffs == pd.Timedelta(hours=6)].index

    print(f"rows in df: {len(df)}")
    print(f"rows in df2 (unique ForecastTime): {len(df2)}")
    print(f"steps checked: {len(diffs)}")
    print(f"non-hourly gaps found: {len(gaps)}")

    if gaps.empty:
        print("No gaps. ForecastTime is continuous hourly.")
    else:
        gap_points = pd.DataFrame(
            {
                "prev_forecast_time": forecast_times.shift(1).loc[gaps.index],
                "next_forecast_time": forecast_times.loc[gaps.index],
                "gap": gaps,
            }
        )
        print(gap_points.head(20).to_string(index=False))

    print(f"6-hour jumps found: {len(six_hour_gap_idx)}")
    if len(six_hour_gap_idx) > 0:
        six_hour_starts = forecast_times.shift(1).loc[six_hour_gap_idx]
        print("First 20 6-hour jump start times:")
        print(six_hour_starts.head(20).to_string(index=False))


if __name__ == "__main__":
    main()
