class MeanGlucoseOverlay:
    """
    Overlay that draws a horizontal mean glucose line for each visible panel
    or across the full CGM tracing, with optional interactive hover tooltips
    showing the mean glucose value.

    Parameters
    ----------
    source : {'file', 'client'}
        Source of CGM data (either loaded from file or provided as an in-memory dataframe).
    base_path : str or None, optional
        Base directory path for file-based loading (used when ``source='file'``).
    subject_id : int or None, optional
        Subject identifier, required when loading from file-based datasets.
    filename : str or None, optional
        Specific filename for file-based data loading.
    client_df : pandas.DataFrame or None, optional
        In-memory dataframe containing CGM data when ``source='client'`` is used.
    scope : {'daily', 'full'}, default 'daily'
        Defines the time window over which the mean glucose is calculated.
        - ``'daily'``: compute mean glucose within each visible axis window.
        - ``'full'``: compute a single mean glucose value across the entire dataset.

    Notes
    -----
    This overlay:
        - Plots a horizontal line at the mean glucose level for each axis panel.
        - Uses a semi-transparent dark blue-gray line for clarity and simplicity.
        - Optionally attaches interactive hover tooltips (if `mplcursors` is installed)
          that display the mean glucose value in mg/dL.

    Requirements
    ------------
    The parent viewer must provide:
        - ``viewer.df`` with columns ``["time", "gl"]`` representing glucose readings.
        - ``viewer.iter_axes_by_time()`` yielding ``(ax, start, end)`` for each visible subplot.
        - ``viewer.scale(hi, lo)`` for DPI-aware linewidth scaling.
        - ``viewer.view_start`` and ``viewer.view_end`` indicating the current time bounds.

    See Also
    --------
    mplcursors : Library used to attach interactive hover tooltips to matplotlib artists.
    """
    def __init__(self, 
                 source: Literal["file", "client"],
                 base_path: str | None = None,
                 subject_id: int | None = None,
                 filename: str | None = None,
                 client_df: Optional[pd.DataFrame] = None,
                 scope: Literal["daily", "full"] = "daily",
                ):
        self.scope = scope
        self.source = source
        self.base_path = base_path,
        self.subject_id = subject_id,
        self.filename = filename
        self.client_df = client_df
        self._cursor = None

    def draw(self):
        viewer = self.viewer
        
        # Variables required for the cursor
        lines = []
        hover_info = {}

        # Iterate through main viewer axis and plot the mean glucose line
        for ax, start, end in viewer.iter_axes_by_time():
            # Compute mean based on the scope indicated
            if self.scope == "daily":
                # Only select the CGM data for a given start to end date provided by the viewer
                sub = viewer.df[(viewer.df["time"] >= start) & (viewer.df["time"] < end)]
                g = pd.to_numeric(sub["gl"], errors="coerce").dropna()
                mean_val = float(np.round(np.nanmean(g), 2))
            # Compute mean for the full (~10 days) CGM tracing
            elif self.scope == "full":
                # For the purpose of this tutorial, g=viewer.df["gl"] is also okay!
                g = pd.to_numeric(viewer.df["gl"], errors="coerce")
                # Compute mean and round to 2 decimal places
                mean_val = float(np.round(np.nanmean(g), 2))

            # Draw mean line
            ln = ax.axhline(mean_val, 
                            color="#2c3e50", 
                            alpha=0.7, 
                            lw=viewer.scale(2.2, 1.6), 
                            zorder=4)
            lines.append(ln)
            hover_info[ln] = {"mean": mean_val}

        # Optional hover tooltips using mplcursors
        try:
            # For more details, see mplcursors documentation: https://mplcursors.readthedocs.io/en/stable/
            import mplcursors
            self._cursor = mplcursors.cursor(lines, hover=True)

            @self._cursor.connect("add")
            def _on_hover(sel):
                data = hover_info.get(sel.artist, {})
                text = (f"$\\bf{{Mean}}$: {data.get('mean', np.nan):.0f} mg/dL")
                sel.annotation.set_text(text)
                patch = sel.annotation.get_bbox_patch()
                patch.set(fc="#f5f5f5", ec="#cfcfcf", lw=0.8, alpha=0.95)
        except Exception:
            self._cursor = None