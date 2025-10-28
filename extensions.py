from loader import load_sleep_data
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import pandas as pd


class HypnogramExtension:
    def __init__(self, sleep_base_path, filename, gap_threshold=30):
        self.sleep_base_path = sleep_base_path
        self.filename = filename
        self.gap_threshold = pd.Timedelta(minutes=gap_threshold)

    def draw(self):
        viewer = self.viewer
        ax = self.ax
        subject_id = viewer.subject_id
        start, end = viewer.view_start, viewer.view_end

        # Load sleep data for this subject and restrict to current day
        sleep_df = load_sleep_data(self.sleep_base_path, filename=self.filename)
        day_sleep = sleep_df[(sleep_df["start"] < end) & (sleep_df["end"] > start)]

        # Stage mapping and background shading
        stage_map = {"Awake": 4, "REM_sleep": 3, "Light_sleep": 2, "Deep_sleep": 1}
        base_color = "#1f77b4"
        alphas = {4: 0.1, 3: 0.25, 2: 0.45, 1: 0.7}

        for val in [1, 2, 3, 4]:
            ax.axhspan(val - 0.5, val + 0.5, color=base_color, alpha=alphas[val])

        if not day_sleep.empty:
            prev_end = None
            prev_stage = None

            for _, row in day_sleep.iterrows():
                y_val = stage_map.get(row["stage"], None)
                if y_val is None:
                    continue

                seg_start = max(row["start"], start)
                seg_end = min(row["end"], end)

                ax.hlines(y_val, seg_start, seg_end, color="navy", linewidth=2)

                if prev_end is not None and (row["start"] - prev_end) <= self.gap_threshold:
                    ax.vlines(row["start"], prev_stage, y_val, color="navy", linewidth=2)

                prev_end = row["end"]
                prev_stage = y_val

        # Axis formatting
        ax.set_xlim(start, end)
        ax.set_ylim(0.5, 4.5)
        ax.set_yticks([1, 2, 3, 4])
        ax.set_yticklabels(["Deep", "Light", "REM", "Awake"])
        ax.set_ylabel(
            "Sleep Stage", 
            color="0.2",
            fontsize=10
        )

        ax.xaxis.set_major_locator(mdates.HourLocator(interval=viewer.scale(2, 6), tz=start.tzinfo))
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%-I %p", tz=start.tzinfo))
        ax.tick_params(labelsize=viewer.scale(10, 7))