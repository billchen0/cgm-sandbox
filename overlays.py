from loader import load_sleep_data, load_activity_data, load_food_entry_data
from pandas import Timedelta
import pandas as pd
import numpy as np
import mplcursors

from cgm_methods import detect_stable_glucose, detect_fasting_segments, extract_wakeup_glucose


class WakeupGlucoseOverlay:
    def __init__(self, sleep_base_path, min_sleep_hours=8.0):
        self.sleep_base_path = sleep_base_path
        self.min_sleep_hours = min_sleep_hours

    def draw(self):
        viewer = self.viewer
        sleep_df = load_sleep_data(self.sleep_base_path, viewer.subject_id)
        wg_df = extract_wakeup_glucose(viewer.df, sleep_df, min_sleep_hours=self.min_sleep_hours)
        if wg_df.empty:
            return

        wg_df = wg_df[
            (wg_df["wakeup_time"] >= viewer.view_start) & (wg_df["wakeup_time"] < viewer.view_end)
        ]

        for ax, start, end in viewer.iter_axes_by_time():
            ax_df = wg_df[(wg_df["wakeup_time"] >= start) & (wg_df["wakeup_time"] < end)]
            for _, row in ax_df.iterrows():
                s = viewer.scale(150, 60)
                fs = viewer.scale(10, 7)

                ax.scatter(row["wakeup_time"], row["wakeup_glucose"],
                           color="green", s=s, edgecolor="white", linewidth=1.2, alpha=1, zorder=5)
                ax.text(
                    row["wakeup_time"], row["wakeup_glucose"] - viewer.scale(70, 30),
                    f"Wakeup Glucose:\n{row['wakeup_glucose']:.0f} mg/dL",
                    fontsize=fs, fontweight="bold", color="black", ha="center", va="bottom",
                    bbox=dict(boxstyle="round,pad=0.3", fc="lightgray", alpha=0.8, ec="black", lw=0.8),
                    clip_on=True
                )


class PhysicalActivityOverlay:
    def __init__(self, activity_base_path):
        self.activity_base_path = activity_base_path

    def draw(self):
        viewer = self.viewer
        subject_id = viewer.subject_id
        activity_df = load_activity_data(self.activity_base_path, subject_id)

        if activity_df.empty:
            return

        # Filter activities in view window
        view_df = activity_df[
            (activity_df["start"] < viewer.view_end) & (activity_df["end"] > viewer.view_start)
        ]
        view_df = view_df[view_df["steps"] > 0]
        if view_df.empty:
            return

        for ax, start, end in viewer.iter_axes_by_time():
            ax_df = view_df[(view_df["start"] < end) & (view_df["end"] > start)]
            for _, row in ax_df.iterrows():
                ax.axvspan(max(row["start"], start), min(row["end"], end), ymax=0.15,
                           color="orange", alpha=0.2, zorder=1)


class StableGlucoseOverlay:
    def __init__(self, method="low_variability", **kwargs):
        self.method = method
        self.kwargs = kwargs
        
    def draw(self):
        viewer = self.viewer
        classified = detect_stable_glucose(viewer.df, method=self.method, **self.kwargs)
        stable_df = classified[classified["stable_label"] == "stable"]

        if stable_df.empty:
            return
        
        stable_df = stable_df.reset_index(drop=True)

        for ax, start, end in viewer.iter_axes_by_time():
            subset = stable_df[(stable_df["time"] >= start) & (stable_df["time"] < end)]
            if subset.empty:
                continue 
                
            subset = subset.reset_index(drop=True)
            segment_id = (subset["time"].diff() > Timedelta(minutes=10)).cumsum()
            for _, seg in subset.groupby(segment_id):
                lw = viewer.scale(4, 2)
                ax.plot(
                    seg["time"], seg["gl"],
                    color="#B4CBF0", linewidth=lw, alpha=0.8,
                    solid_capstyle="round", zorder=4
                )


class FoodEntryOverlay:
    def __init__(self, food_entry_path, filename):
        self.food_entry_path = food_entry_path
        self.filename = filename
        self._cursor = None
    
    def _size_from_carbs(self, carbs: float, viewer) -> float:
        carbs = float(np.clip(carbs, 0.0, 100.0))
        s_min = viewer.scale(40, 15)
        s_max = viewer.scale(280, 80) 
        return np.interp(carbs, [0.0, 100.0], [s_min, s_max])

    def draw(self):
        viewer = self.viewer
        diet_df = load_food_entry_data(base_path=self.food_entry_path, 
                                       subject_id=viewer.subject_id,
                                       filename=self.filename)

        day_meals = diet_df[
            (diet_df["time"] >= viewer.view_start) & (diet_df["time"] < viewer.view_end)
        ]
        if day_meals.empty:
            return

        points, tooltips = [], {}
        for ax, start, end in viewer.iter_axes_by_time():
            ax_df = day_meals[(day_meals["time"] >= start) & (day_meals["time"] < end)]

            for _, row in ax_df.iterrows():
                food_name = row.get("food_name", "food_entry")
                carbs_val = row.get("carbohydrate")

                tip = f"{food_name}\n$\\bf{{Carbs:}}$ {carbs_val:.0f} g"
                
                idx = (viewer.df["time"] - row["time"]).abs().idxmin()
                gl = viewer.df.loc[idx, "gl"]
                
                s = self._size_from_carbs(carbs_val, viewer)

                pt = ax.scatter(row["time"], gl, s=s, color="orange", alpha=0.7,
                                linewidth=1.2, edgecolor="white", zorder=5)
                points.append(pt)
                tooltips[pt] = tip

        self._cursor = mplcursors.cursor(points, hover=True)

        @self._cursor.connect("add")
        def on_hover(sel):
            sel.annotation.set_text(tooltips.get(sel.artist, ""))
            sel.annotation.get_bbox_patch().set(fc="orange", alpha=0.4)

        @self._cursor.connect("remove")
        def on_unhover(sel):
            if sel.annotation is not None:
                sel.annotation.set_visible(False)
                if sel.annotation.figure:
                    sel.annotation.figure.canvas.draw_idle()


class FastingGlucoseOverlay:
    def __init__(self, diet_base_path, fasting_window_hours=8.0):
        self.diet_base_path = diet_base_path
        self.fasting_window_hours = fasting_window_hours

    def draw(self):
        viewer = self.viewer
        diet_df = load_food_entry_data(base_path=self.food_entry_path, 
                                       subject_id=viewer.subject_id,
                                       filename=self.filename)
        if diet_df.empty:
            return

        classified = detect_fasting_segments(viewer.df, diet_df, self.fasting_window_hours)
        classified = classified[
            (classified["time"] >= viewer.view_start) & (classified["time"] < viewer.view_end)
        ]
        classified = classified[classified["fasting_label"] == "fasting"]

        if classified.empty:
            return

        group_id = (classified.index.to_series().diff() > 1).cumsum()

        for ax, start, end in viewer.iter_axes_by_time():
            subset = classified[(classified["time"] >= start) & (classified["time"] < end)]
            if subset.empty:
                continue

            segment_id = (subset["time"].diff() > Timedelta(minutes=10)).cumsum()
            for _, seg in subset.groupby(segment_id):
                ax.plot(seg["time"], seg["gl"], color="orange", linewidth=viewer.scale(4, 2),
                        solid_capstyle="round", zorder=4)

                # Median annotation
                median = np.median(seg["gl"])
                mid_time = seg["time"].iloc[len(seg) // 2]
                ax.text(mid_time, median + viewer.scale(25, 15),
                        f"Median FG: {median:.1f}",
                        fontsize=viewer.scale(9, 6),
                        color="orange", ha="center", va="bottom",
                        bbox=dict(facecolor="white", alpha=0.8))


class TimeInRangeOverlay:
    def __init__(self, low=70, high=180,
                 colors=None, show_lines=True, gap_minutes=10):
        self.low = low
        self.high = high
        self.show_lines = show_lines
        self.gap = Timedelta(minutes=gap_minutes)
        self.colors = colors or {
            "in":   "#22a559",  # green
            "low":  "#d9534f",  # red
            "high": "#f39c12",  # orange
            "line": "#e0e0e0"   # threshold lines
        }

    def draw(self):
        v = self.viewer
        for ax, start, end in v.iter_axes_by_time():
            # Threshold lines
            if self.show_lines:
                ax.axhline(self.low,  color=self.colors["line"], linewidth=1.0, zorder=1)
                ax.axhline(self.high, color=self.colors["line"], linewidth=1.0, zorder=1)

            # Data in this axis window
            sub = v.df[(v.df["time"] >= start) & (v.df["time"] <= end)][["time", "gl"]].dropna()
            if sub.empty:
                continue
            sub = sub.sort_values("time").reset_index(drop=True)

            # Classify each point: -1 (below), 0 (in), +1 (above)
            state = np.where(sub["gl"] < self.low, -1,
                             np.where(sub["gl"] > self.high, 1, 0))
            sub["_state"] = state

            # Break segments at state changes or big time gaps
            breaks = (sub["time"].diff() > self.gap) | (sub["_state"].diff().fillna(0) != 0)
            seg_id = breaks.cumsum()

            lw = v.scale(2.5, 1.2)
            for _, seg in sub.groupby(seg_id):
                s = int(seg["_state"].iloc[0])
                color = self.colors["in"] if s == 0 else (self.colors["high"] if s > 0 else self.colors["low"])
                i0 = int(seg.index.min())
                if i0 > 0:
                    seg = pd.concat([sub.loc[[i0-1]], seg], axis=0)
                ax.plot(seg["time"], seg["gl"],
                        color=color, linewidth=lw,
                        solid_capstyle="round", zorder=4, clip_on=True)


class PPGROverlay:
    def __init__(self,
                 food_base_path,
                 window_minutes: int = 120,
                 gap_minutes: int | None = 10):
        self.food_base_path = food_base_path
        self.window = pd.Timedelta(minutes=window_minutes)
        self.gap = (pd.Timedelta(minutes=gap_minutes) if gap_minutes is not None else None)

    def draw(self):
        v = self.viewer
        tz = getattr(v.view_start, "tzinfo", None)

        # Load food log aligned to the viewer's timezone (same pattern you use elsewhere)
        food_df = load_food_entry_data(base_path=self.food_entry_path, 
                                subject_id=v.subject_id,
                                filename=self.filename)
        if food_df is None or food_df.empty:
            return

        # Meals that overlap the visible window once expanded by the PPGR window
        meals = food_df[(food_df["time"] < v.view_end) & (food_df["time"] + self.window > v.view_start)]
        if meals.empty:
            return

        lw = v.scale(3.0, 1.6)

        for ax, start, end in v.iter_axes_by_time():
            m_ax = meals[(meals["time"] < end) & (meals["time"] + self.window > start)]
            if m_ax.empty:
                continue

            for _, m in m_ax.iterrows():
                t0 = max(m["time"], start)
                t1 = min(m["time"] + self.window, end)

                sub = v.df[(v.df["time"] >= t0) & (v.df["time"] <= t1)][["time", "gl"]].dropna()
                if sub.shape[0] < 2:
                    continue
                sub = sub.sort_values("time").reset_index(drop=True)

                # Break only across large dropouts; keep continuous color otherwise
                if self.gap is not None:
                    seg_id = (sub["time"].diff() > self.gap).fillna(False).cumsum()
                else:
                    seg_id = pd.Series(0, index=sub.index)

                for _, seg in sub.groupby(seg_id):
                    if len(seg) < 2:
                        continue
                    ax.plot(
                        seg["time"], seg["gl"],
                        color="#6c5ce7",
                        linewidth=lw, solid_capstyle="round",
                        zorder=6, clip_on=True
                    )