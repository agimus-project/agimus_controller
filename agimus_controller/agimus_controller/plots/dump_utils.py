import json
import numpy as np


def dump_plot_data(
    filename: str,
    title: str,
    time: np.ndarray,
    values: np.ndarray,
    labels=None,
    ylabels=None,
    semilogs=None,
    ylimits=None,
    colors=None,
):
    """Dump plot data and metadata to a JSON file."""
    data = {
        "title": title,
        "time": time.tolist(),
        "values": values.tolist(),
        "labels": labels if labels is not None else [],
        "ylabels": ylabels if ylabels is not None else [],
        "semilogs": semilogs if semilogs is not None else [],
        "ylimits": ylimits if ylimits is not None else [],
        "colors": colors if colors is not None else [],
    }
    with open(filename, "w") as f:
        json.dump(data, f, indent=2)
