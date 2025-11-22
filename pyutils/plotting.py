__all__ = ["multicolor_errorbar", "lighten_color", "multisavefig", "add_annotation_in_legend", "multicolor_errorbar", "multicolor_errorbar_from_df"]

import colorsys
import matplotlib.pyplot as plt
import matplotlib.colors as mc
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd

def lighten_color(color, amount=0.5):
    """
    Lightens the given color by multiplying (1-luminosity) by the given amount.
    Input can be matplotlib color string, hex string, or RGB tuple.

    Examples:
    >> lighten_color("g", 0.3)
    >> lighten_color("#F034A3", 0.6)
    >> lighten_color((.3,.55,.1), 0.5)

    Source: https://stackoverflow.com/questions/37765197/darken-or-lighten-a-color-in-matplotlib
    """
    
    try:
        c = mc.cnames[color]
    except:
        c = color
    c = colorsys.rgb_to_hls(*mc.to_rgb(c))
    return colorsys.hls_to_rgb(c[0], 1 - amount * (1 - c[1]), c[2])

def multisavefig(fn, vector_fmts=["pdf", "svg"], raster_fmts=["png"], dpi=600, **kwargs):
    """
    Save the current Matplotlib figure in multiple formats with one call.

    This function saves the active figure to both vector and raster formats.
    Vector formats (e.g., PDF, SVG) preserve resolution-independent graphics,
    while raster formats (e.g., PNG) store pixel-based images at a specified DPI.

    Parameters
    ----------
    fn : str
        Base filename (without extension) for the saved files. 
        Newlines (`\n`) are automatically removed from the filename.
    vector_fmts : list of str, optional
        List of vector file formats to save (default: ["pdf", "svg"]).
    raster_fmts : list of str, optional
        List of raster file formats to save (default: ["png"]).
    dpi : int, optional
        Resolution (dots per inch) for raster outputs (default: 600).
    **kwargs : dict
        Additional keyword arguments passed directly to `matplotlib.pyplot.savefig`.

    Notes
    -----
    - All formats are saved with `bbox_inches="tight"` to minimize whitespace.
    - If the filename contains newlines, they are replaced with spaces.
    - This function uses the currently active figure in Matplotlib.
    """
    clean_fn = fn.replace("\n", " ")  # .replace(".", "_")

    for vector_fmt in vector_fmts:
        plt.savefig(clean_fn + "." + vector_fmt, bbox_inches="tight", **kwargs)

    for raster_fmt in raster_fmts:
        plt.savefig(clean_fn + "." + raster_fmt, bbox_inches="tight", dpi=dpi, **kwargs)


def add_annotation_in_legend(axis, annotation):
    """
    Add a text-only annotation to a Matplotlib legend without any markers or lines.

    This function inserts a dummy legend entry that contains only the given
    annotation text. It is useful for displaying metadata, parameter values,
    or other descriptive information directly in the legend without
    associating it with a plotted line, marker, or patch.

    Parameters
    ----------
    axis : matplotlib.axes.Axes
        The Axes object on which to add the legend entry.
    annotation : str
        The text to display in the legend.

    Notes
    -----
    - The legend entry has no visible handle (marker or line).
    - The legend frame is removed (`frameon=False`) and set to full transparency.
    - Spacing is minimized by setting `handlelength=0` and `handletextpad=0`.
    """
    axis.legend(
        handles=[mpatches.Patch(color="none", label=annotation)],
        handlelength=0,
        handletextpad=0,
        frameon=False,
        framealpha=0
    )

def multicolor_errorbar(x, y, xerr=None, yerr=None, colors=None, ax=None, labels=None, legend=False, **kwargs):
    """
    Plot error bars with each marker and its associated error bars in a different color.

    Parameters
    ----------
    x : array-like
        x-coordinates of the data points.
    y : array-like
        y-coordinates of the data points.
    xerr : array-like or None, optional
        Horizontal error bar sizes.
    yerr : array-like or None, optional
        Vertical error bar sizes.
    colors : list of str, optional
        List of colors for each data point and its error bars.
    ax : matplotlib.axes.Axes, optional
        Axes to plot on. If None, uses current axes.
    **kwargs :
        Additional keyword arguments passed to `plt.errorbar`.
    """
    x = np.asarray(x)
    y = np.asarray(y)

    if colors is None:
        colors = ["C0"] * len(x)
    if ax is None:
        ax = plt.gca()

    if labels is None:
        labels = [None,]*len(x)

    for i in range(len(x)):
        ax.errorbar(
            x[i], y[i],
            xerr=xerr[i] if xerr is not None else None,
            yerr=yerr[i] if yerr is not None else None,
            fmt="o",
            color=colors[i],
            ecolor=colors[i],
            label=labels[i],
            **kwargs
        )

    if legend:
        ax.legend()


def multicolor_errorbar_from_df(
    data: pd.DataFrame,
    x: str,
    y: str,
    xerr: str | None = None,
    yerr: str | None = None,
    **kwargs
) -> None:
    """
    Plot error bars with each marker and its associated error bars in a different color,
    using column names from a pandas DataFrame.

    Parameters
    ----------
    data : pandas.DataFrame
        DataFrame containing the data to plot.
    x : str
        Column name for x-coordinates.
    y : str
        Column name for y-coordinates.
    xerr : str or None, optional
        Column name for horizontal error bar sizes.
    yerr : str or None, optional
        Column name for vertical error bar sizes.
    **kwargs :
        Additional keyword arguments passed to `multicolor_errorbar`.
        Supported kwargs include:
        - colors : str or list of str
            Column name for colors, or a list of color values.
        - labels : str or list of str
            Column name for labels, or a list of label values.
        - ax : matplotlib.axes.Axes
            Axes to plot on.
        - legend : bool
            Whether to display a legend.
        - Any other kwargs accepted by plt.errorbar.

    Returns
    -------
    None
    """
    
    # Extract x and y data
    x_data = data[x].values
    y_data = data[y].values
    
    # Extract error data if specified
    xerr_data = data[xerr].values if xerr is not None else None
    yerr_data = data[yerr].values if yerr is not None else None
    
    # Handle colors - can be a column name or passed directly
    if "colors" in kwargs:
        colors = kwargs.pop("colors")
        if isinstance(colors, str) and colors in data.columns:
            colors = data[colors].tolist()
    else:
        colors = None
    
    # Handle labels - can be a column name or passed directly
    if "labels" in kwargs:
        labels = kwargs.pop("labels")
        if isinstance(labels, str) and labels in data.columns:
            labels = data[labels].tolist()
    else:
        labels = None
    
    multicolor_errorbar(
        x=x_data,
        y=y_data,
        xerr=xerr_data,
        yerr=yerr_data,
        colors=colors,
        labels=labels,
        **kwargs
    )