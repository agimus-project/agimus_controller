import json
import numpy as np


def dump_mpc_plot_data(filename, plot_data_dict):
    """
    Dump all plot data and metadata for MPC plots to a JSON file.
    plot_data_dict: dict of dicts, one per figure, with keys:
      - title
      - x (time or index)
      - y (data series, possibly 2D)
      - labels
      - ylabels
      - colors
      - etc.
    """

    # Convert all numpy arrays to lists for JSON serialization
    def convert(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, dict):
            return {k: convert(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [convert(v) for v in obj]
        return obj

    with open(filename, "w") as f:
        json.dump(convert(plot_data_dict), f, indent=2)
