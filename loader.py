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
    
    body = data.get("body", {})
    raw_df = pd.json_normalize(body)
    
    df = pd.DataFrame()
    df["time"] = pd.to_datetime(raw_df["effective_time_frame.time_interval.start_date_time"], utc=True)
    if timezone:
        tz_str = data.get("header", {}).get("timezone", "UTC").upper()
        tz_name = TZ_ABBREV.get(tz_str, tz_str)
        local_tz = tz.gettz(tz_name) or tz.UTC
        df["time"] = df["time"].dt.tz_convert(local_tz)
    df["gl"] = raw_df["blood_glucose.value"]

    return df


def load_sleep_data(base_path: str, subject_id: int) -> pd.DataFrame:
    base_path = Path(base_path)
    json_file = base_path / str(subject_id) / f"{subject_id}_sleep.json"

    with open(json_file, "r") as file:
        data = json.load(file)
    
    records = []
    for entry in data.get("body", {}).get("sleep", []):
        start = pd.to_datetime(entry["sleep_stage_time_frame"]["time_interval"]["start_date_time"])
        end = pd.to_datetime(entry["sleep_stage_time_frame"]["time_interval"]["end_date_time"])
        stage = entry["sleep_stage_state"]
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


def load_diet_data(base_path: str, subject_id: int, tz=None) -> pd.DataFrame:
    base_path = Path(base_path)
    json_file = base_path / str(subject_id) / f"{subject_id}_FOODLOG.json"

    if not json_file.exists():
        return pd.DataFrame(columns=["time", "meal_type", "nutrients"])

    with open(json_file, "r") as file:
        data = json.load(file)
        raw = pd.json_normalize(data["body"]["diet"])

    if raw.empty:
        return pd.DataFrame(columns=["time", "meal_type", "nutrients"])

    df = pd.DataFrame()
    df["time"] = pd.to_datetime(raw["effective_time_frame.time_interval.start_date_time"])
    if tz is not None:
        df["time"] = df["time"].dt.tz_localize(tz)

    df["meal_type"] = raw["meal_type"]

    nutrient_cols = [c for c in raw.columns if c.startswith("nutrients.")]
    def build_nutrient_dict(row):
        nutrients = {}
        for col in nutrient_cols:
            val = row[col]
            if pd.notna(val):
                key = col.split(".")[1]
                nutrients[key] = float(val)
        return nutrients

    df["nutrients"] = raw.apply(build_nutrient_dict, axis=1)

    return df