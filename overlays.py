from loader import load_sleep_data, load_food_entry_data
from pandas import Timedelta
import pandas as pd
import numpy as np
import mplcursors
from typing import Literal, Optional

from cgm_methods import extract_wakeup_glucose
from cgmquantify import summarize_measures, cv, mage_ma_segments


# --------------------------------
# Make your own Overlay!
# --------------------------------
class MeanGlucoseOverlay:
    # TODO: Add a "scope" parameter which allows you to compute mean glucose
    # within a single day or across the entire tracing.
    def __init__(self):
        self.scope = "daily" # "daily" or "full"

    def draw(self):
        # TODO: Attach the viewer
        viewer = ...

        # Optional: Uncomment if you decide to take on the cursor challenge
        # lines = []

        for ax, start, end in viewer.iter_axes_by_time():
            # TODO: Access the CGM data based on given scope
            if self.scope == "full":
                ...
            else:
                ...

            # TODO: Compute the mean glucose for given window
            mean_glucose = ...

            # TODO: Draw the mean line using curren axis
            ln = ax.axhline(
                # Fill in parameters
                # Documentation on hline method:
                # https://matplotlib.org/stable/api/_as_gen/matplotlib.axes.Axes.axhline.html
            )

            # Optional: Uncomment if you decide to take on the cursor challenge
            # lines.append((ln, mean_glucose, start, end))
        
        # (Optional)
        # TODO: Add a cursor tooltip to show mean glucose value
        # if lines:
        #     cursor = mplcursors.cursor([ln for (ln, _, _, _) in lines], hover=True)

        # @cursor.connect("add")
        # def _on_add(sel):
        #     artist = sel.artist
        #     ...
        
        #     sel.annotation.set_text()


# --------------------------------
# Basic Overlays
# --------------------------------
class TimeInRangeOverlay:
    def __init__(self, low=70, high=180,
                 colors=None, show_lines=True, gap_minutes=10):
        self.low = low
        self.high = high
        self.show_lines = show_lines
        self.gap = Timedelta(minutes=gap_minutes)
        self.colors = colors or {
            "in":   "#22a559",
            "low":  "#d9534f",
            "high": "#f39c12",
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

                
class FoodEntryOverlay:
    def __init__(self, 
                 source: Literal["file", "client"] = "client",
                 base_path: str | None = None,
                 subject_id: int | None = None,
                 filename: str | None = None,
                 client_df: Optional[pd.DataFrame] = None
                ):
        self.source = source
        self.base_path = base_path
        self.filename = filename
        self.client_df = client_df
        self._cursor = None
    
    def _size_from_carbs(self, carbs: float, viewer) -> float:
        carbs = float(np.clip(carbs, 0.0, 100.0))
        s_min = viewer.scale(40, 15)
        s_max = viewer.scale(280, 80) 
        return np.interp(carbs, [0.0, 100.0], [s_min, s_max])

    def draw(self):
        viewer = self.viewer
        food_df = load_food_entry_data(source=self.source,
                                       base_path=self.base_path, 
                                       subject_id=viewer.subject_id,
                                       filename=self.filename,
                                       client_df=self.client_df)

        day_meals = food_df[
            (food_df["time"] >= viewer.view_start) & (food_df["time"] < viewer.view_end)
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


# --------------------------------
# CGM Biomarker Overlays
# --------------------------------
class CgmMeasuresOverlay:

    def __init__(self,
                 sd_multiplier: float = 1.0,
                 loc: str = "top-left",
                 y: float = 0.965,
                 margin: float = 0.008,
                 facecolor: str = "#f7f7f7",
                 edgecolor: str = "#dcdcdc",
                 text_color: str = "#111",
                 pad: float = 0.10):
        self.sd_multiplier = sd_multiplier
        self.loc = loc
        self.y = y
        self.margin = margin
        self.facecolor = facecolor
        self.edgecolor = edgecolor
        self.text_color = text_color
        self.pad = pad

    @staticmethod
    def _fmt(v, unit=""):
        if v is None or (isinstance(v, float) and not np.isfinite(v)):
            return "—"
        return f"{v:.1f}{unit}"

    @staticmethod
    def _math_bold(label: str) -> str:
        safe = label.replace("–", "-")
        safe = safe.replace(" ", r"\ ")
        return rf"$\bf{{{safe}}}$"

    def draw(self):
        viewer = getattr(self, "viewer", None)
        ax = getattr(viewer, "ax_cgm", None)
        if viewer is None or ax is None or viewer.df.empty:
            return

        m = summarize_measures(viewer.df, sd_multiplier=self.sd_multiplier)

        parts = [
            (self._math_bold("TIR"),        self._fmt(m.get("in_range_70_180"), "%")),
            (self._math_bold("CV"),         self._fmt(m.get("cv_percent"), "%")),
            (self._math_bold("GMI"),        self._fmt(m.get("gmi_percent"), "%")),
            (self._math_bold("MAGE"),       self._fmt(m.get("MAGE"), " mg/dL")),
            (self._math_bold("HBGI"),       self._fmt(m.get("HBGI"))),
            (self._math_bold("LBGI"),       self._fmt(m.get("LBGI"))),
        ]

        text = " | ".join([f"{lbl}: {val}" for lbl, val in parts])

        ha = "left" if self.loc == "top-left" else "right"
        x = self.margin if ha == "left" else 1.0 - self.margin

        ax.text(
            x, self.y, text,
            ha=ha, va="center",
            fontsize=viewer.scale(10, 8),
            family="monospace",
            color=self.text_color,
            transform=ax.transAxes,
            bbox=dict(
                facecolor=self.facecolor,
                edgecolor=self.edgecolor,
                boxstyle=f"round,pad={self.pad}",
            ),
            zorder=15,
            clip_on=False,
        )


class CvOverlay:
    def __init__(self, color="#4a90e2", alpha=0.14, scope="window"):
        self.color = color
        self.alpha = alpha
        self.scope = scope

    def draw(self):
        viewer = getattr(self, "viewer", None)
        ax = getattr(viewer, "ax_cgm", None)

        if ax is None:
            return

        if self.scope == "full":
            w = viewer.df
        else:
            start, end = viewer.view_start, viewer.view_end
            w = viewer.df[(viewer.df["time"] >= start) & (viewer.df["time"] < end)]

        g = pd.to_numeric(w["gl"], errors="coerce").dropna()

        mean = float(np.nanmean(g))
        cv_percent = float(cv(w))
        sd = (cv_percent * mean) / 100.0

        # Visualization
        ax.axhline(mean, color=self.color, linewidth=viewer.scale(1.8, 1.2), alpha=0.7, zorder=-2)
        ax.axhspan(mean - sd, mean + sd, color=self.color, alpha=self.alpha, zorder=-3)

        # Annotation pill (top-right), bold label via mathtext
        txt = f"$\\bf{{CV}}$: {cv_percent:.1f}%   $\\bf{{Mean}}$: {mean:.1f}   $\\bf{{±1SD}}$: {sd:.1f}"
        ax.text(
            0.995, 0.965, txt,
            ha="right", va="center",
            fontsize=viewer.scale(10, 8), family="monospace", color="#111",
            transform=ax.transAxes,
            bbox=dict(facecolor="#f7f7f7", edgecolor="#dcdcdc", boxstyle="round,pad=0.10"),
            zorder=5, clip_on=False
        )


class MageOverlay:
    """
    Visualizes MAGE (MA) excursions for the *current window* (daily panel):
      • shaded span over each counted segment [t0, t1]
      • vertical amplitude bar at segment midpoint from local min→max
      • optional short/long MAs for context (thin lines)
    No text values are shown; this is interpretation-first.

    Parameters
    ----------
    resample_rule : str        Default "5min" (must match your mage_ma_segments)
    short_win     : str        Default "30min"
    long_win      : str        Default "2H"
    sd_multiplier : float      Default 1.0 (threshold = sd_multiplier * SD)
    color_up      : str        Fill/line color when segment trends up (max after min)
    color_down    : str        Fill/line color when segment trends down (min after max)
    fill_alpha    : float      Span transparency
    line_alpha    : float      Amplitude bar transparency
    show_ma       : bool       Plot short/long MAs for context (thin)
    ma_colors     : tuple      (short_color, long_color)
    """
    def __init__(self,
                 resample_rule: str = "5min",
                 short_win: str = "30min",
                 long_win: str = "2h",
                 sd_multiplier: float = 1.0,
                 show_ma: bool = False):
        self.resample_rule = resample_rule
        self.short_win = short_win
        self.long_win = long_win
        self.sd_multiplier = sd_multiplier
        self.show_ma = show_ma

    def _subset_window(self, df: pd.DataFrame, start, end) -> pd.DataFrame:
        return df[(df["time"] >= start) & (df["time"] < end)].copy()

    def _local_minmax(self, df: pd.DataFrame, t0, t1) -> tuple[float, float, float, float]:
        w = df[(df["time"] >= t0) & (df["time"] <= t1)]
        g = pd.to_numeric(w["gl"], errors="coerce").dropna()

        gmin = float(g.min())
        gmax = float(g.max())

        g_first = float(g.iloc[0])
        g_last = float(g.iloc[-1])
        return gmin, gmax, g_first, g_last

    def _draw_ma_thin(self, ax, df: pd.DataFrame, viewer):
        s = pd.Series(pd.to_numeric(df["gl"], errors="coerce").values,
                      index=pd.to_datetime(df["time"], errors="coerce")).dropna()
        if s.empty:
            return
        s = s.sort_index()
        s5 = s.resample(self.resample_rule).mean().interpolate("time").dropna()
        sma = s5.rolling(self.short_win, min_periods=1, center=True).mean()
        lma = s5.rolling(self.long_win,  min_periods=1, center=True).mean()

        ax.plot(sma.index, sma.values, color="#2c7fb8",
                linewidth=viewer.scale(1.0, 0.8), alpha=0.7, zorder=2)
        ax.plot(lma.index, lma.values, color="#7b3294",
                linewidth=viewer.scale(1.0, 0.8), alpha=0.7, zorder=2)

    def draw(self):
        viewer = getattr(self, "viewer", None)
        ax = getattr(viewer, "ax_cgm", None)
        if viewer is None or ax is None:
            return

        window_df = self._subset_window(viewer.df, viewer.view_start, viewer.view_end)
        if window_df.empty:
            return

        segs = mage_ma_segments(
            window_df,
            resample_rule=self.resample_rule,
            short_win=self.short_win,
            long_win=self.long_win,
            sd_multiplier=self.sd_multiplier
        )

        if self.show_ma:
            self._draw_ma_thin(ax, window_df, viewer)

        dt_total = (viewer.view_end - viewer.view_start)
        cap_half = dt_total * 0.006

        for seg in segs:
            t0, t1 = seg["t0"], seg["t1"]
            t_mid = seg["t_mid"]

            gmin, gmax, g_first, g_last = self._local_minmax(window_df, t0, t1)

            is_up = (g_last >= g_first)
            color = "#8bd3c7" if is_up else "#f4a6b8"

            # Shade the segment duration
            ax.axvspan(t0, t1, color=color, alpha=0.18, zorder=1)

            # Amplitude whisker at midpoint
            ax.vlines(t_mid, gmin, gmax, color=color,
                      linewidth=viewer.scale(2.0, 1.5), alpha=0.9, zorder=3)

            # Small end-caps at min/max (to visually bracket amplitude)
            ax.hlines(gmin, t_mid - cap_half, t_mid + cap_half, color=color,
                      linewidth=viewer.scale(1.5, 1.2), alpha=0.9, zorder=3)
            ax.hlines(gmax, t_mid - cap_half, t_mid + cap_half, color=color,
                      linewidth=viewer.scale(1.5, 1.2), alpha=0.9, zorder=3)
            

# --------------------------------
# Multi-modal Biomarker Overlays
# --------------------------------
class WakeupGlucoseOverlay:
    def __init__(self,
                 source: Literal["file", "client"],
                 base_path: str | None = None, 
                 subject_id: int | None = None,
                 filename: str | None = None,
                 client_df: Optional[pd.DataFrame] = None,
                 min_sleep_hours=8.0):
        self.source = source
        self.base_path = base_path
        self.filename = filename
        self.client_df = client_df
        self.min_sleep_hours = min_sleep_hours
        self.filename = filename

    def draw(self):
        viewer = self.viewer
        
        sleep_df = load_sleep_data(source=self.source,
                           base_path=self.base_path,
                           filename=self.filename,
                           client_df=self.client_df
                          )
        
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
                           color="blue", s=s, edgecolor="white", linewidth=1.2, alpha=1, zorder=5)
                if getattr(viewer, "view_mode", None) != "full":
                    ax.text(
                        row["wakeup_time"], row["wakeup_glucose"] - viewer.scale(70, 30),
                        f"Wakeup Glucose:\n{row['wakeup_glucose']:.0f} mg/dL",
                        fontsize=fs, fontweight="bold", color="darkblue", ha="center", va="bottom",
                        bbox=dict(boxstyle="round,pad=0.3", fc="blue", alpha=0.1, ec="black", lw=0.8),
                        clip_on=True
                    )


class PPGROverlay:
    def __init__(self,
                 source: Literal["file", "client"],
                 base_path: str | None = None, 
                 subject_id: int | None = None,
                 filename: str | None = None,
                 client_df: Optional[pd.DataFrame] = None,
                 window_minutes: int = 120,
                 gap_minutes: int | None = 10):
        self.source = source
        self.base_path = base_path
        self.filename = filename
        self.client_df = client_df
        self.window = pd.Timedelta(minutes=window_minutes)
        self.gap = (pd.Timedelta(minutes=gap_minutes) if gap_minutes is not None else None)

    def draw(self):
        v = self.viewer
        tz = getattr(v.view_start, "tzinfo", None)

        # Load food log aligned to the viewer's timezone (same pattern you use elsewhere)
        food_df = load_food_entry_data(source=self.source,
                                       base_path=self.base_path, 
                                       subject_id=v.subject_id,
                                       filename=self.filename,
                                       client_df=self.client_df)
        
        # Meals that overlap the visible window once expanded by the PPGR window
        meals = food_df[(food_df["time"] < v.view_end) & (food_df["time"] + self.window > v.view_start)]

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


