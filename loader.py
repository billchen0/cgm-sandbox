import json
import pandas as pd
from pathlib import Path
from dateutil import tz

TZ_ABBREV = {
    "PST": "US/Pacific",
    "PDT": "US/Pacific",
    "MST": "US/Mountain",
    "MDT": "US/Mountain",
    "CST": "US/Central",
    "CDT": "US/Central",
    "EST": "US/Eastern",
    "EDT": "US/Eastern",
    "UTC": "UTC"
}

def load_cgm_data(base_path: str, 
                  subject_id: int | None = None, 
                  timezone: bool=True,
                  filename: str | None = None) -> pd.DataFrame:
    base_path = Path(base_path)
    subject_dir = base_path / str(subject_id) if subject_id is not None else base_path

    if filename:
        filename = filename.format(subject_id=subject_id)
        candidates = [subject_dir / filename, base_path / filename]
    else:
        raise ValueError("Must specify either `filename` or `subject_id`.")

    json_file = next((p for p in candidates if p.exists()), None)
    if json_file is None:
        raise FileNotFoundError(f"No CGM file found (tried {candidates})")
    
    with open(json_file, "r") as file:
        data = json.load(file)
    
    body = data.get("body", [])
    raw_df = pd.json_normalize(body)
    
    df = pd.DataFrame()
    df["time"] = pd.to_datetime(raw_df["effective_time_frame.date_time"], utc=True)
    if timezone:
        tz_str = data.get("header", {}).get("timezone", "UTC").upper()
        tz_name = TZ_ABBREV.get(tz_str, tz_str)
        local_tz = tz.gettz(tz_name) or tz.UTC
        df["time"] = df["time"].dt.tz_convert(local_tz)
    df["gl"] = raw_df["blood_glucose.value"]

    return df


def load_sleep_data(base_path: str, 
                    subject_id: int | None = None,
                    filename: str | None = None) -> pd.DataFrame:
    base_path = Path(base_path)
    subject_dir = base_path / str(subject_id) if subject_id is not None else base_path

    if filename:
        filename = filename.format(subject_id=subject_id)
        candidates = [subject_dir / filename, base_path / filename]
    else:
        raise ValueError("Must specify either `filename` or `subject_id`.")
    
    json_file = next((p for p in candidates if p.exists()), None)
    if json_file is None:
        raise FileNotFoundError(f"No CGM file found (tried {candidates})")
    
    with open(json_file, "r") as file:
        data = json.load(file)
    
    records = []

    body = data.get("body", [])

    for item in body:
        episodes = item.get("sleep_stage_episodes", [])
        for ep in episodes:
            ti = ep.get("sleep_stage_time_frame", {}).get("time_interval", {})

            start = pd.to_datetime(ti.get("start_date_time"), utc=True, errors="coerce")
            end   = pd.to_datetime(ti.get("end_date_time"),   utc=True, errors="coerce")
            stage = ep.get("sleep_stage_state")

            records.append({"start": start, "end": end, "stage": stage})
    
    return pd.DataFrame(records)


def load_activity_data(base_path: str, subject_id: int) -> pd.DataFrame:
    base_path = Path(base_path)
    json_file = base_path / str(subject_id) / f"{subject_id}_activity.json"

    with open(json_file, "r") as file:
        data = json.load(file)
        raw = pd.json_normalize(data["body"]["activity"])
    
    df = pd.DataFrame()
    df["start"] = pd.to_datetime(raw["effective_time_frame.time_interval.start_date_time"])
    df["end"] = pd.to_datetime(raw["effective_time_frame.time_interval.end_date_time"])
    df["steps"] = pd.to_numeric(raw["base_movement_quantity.value"], errors="coerce").fillna(0)

    return df


def load_food_entry_data(base_path: str, 
                         subject_id: int | None = None, 
                         filename: str | None = None) -> pd.DataFrame:
    base_path = Path(base_path)
    subject_dir = base_path / str(subject_id) if subject_id is not None else base_path

    if filename:
        filename = filename.format(subject_id=subject_id)
        candidates = [subject_dir / filename, base_path / filename]
    else:
        raise ValueError("Must specify either `filename` or `subject_id`.")

    json_file = next((p for p in candidates if p.exists()), None)
    if json_file is None:
        raise FileNotFoundError(f"No food entry file found (tried {candidates})")

    with open(json_file, "r") as f:
        data = json.load(f)

    body = data.get("body", [])
    raw_df = pd.json_normalize(body)

    df = pd.DataFrame()
    df["time"] = pd.to_datetime(
        raw_df["effective_time_frame.date_time"], errors="coerce", utc=True
    )

    df["carbohydrate"] = pd.to_numeric(raw_df.get("carbohydrate.value"), errors="coerce")

    food_name_series = raw_df.get("food_name")
    if food_name_series is None:
        food_name_series = pd.Series(["Food"] * len(raw_df))
    df["food_name"] = food_name_series.astype(str).str.strip().replace({"": "food_entry"})

    df["calories"] = pd.to_numeric(raw_df["calories.value"], errors="coerce")

    return df