# Plot Data Dump Format

The plotting utilities now automatically dump plot data and metadata to a JSON file for each plot. This allows you to reproduce and beautify plots later.

## File Naming
- Each plot creates a file named `<title>_plotdata.json` (spaces replaced by underscores).

## JSON Structure
```
{
  "title": "Plot Title",
  "time": [ ... ],           // List of time values (x-axis)
  "values": [ [...], ... ],  // 2D list of y-values (one list per series)
  "labels": [ ... ],         // Series labels (if any)
  "ylabels": [ ... ],        // Y-axis labels (if any)
  "semilogs": [ ... ],       // List of booleans for semilog axes (if any)
  "ylimits": [ [...], ... ], // List of [min, max] for y-limits (if any)
  "colors": [ ... ]          // List of color codes for each series
}
```

## Usage
- You can load this JSON in Python and use the data to recreate or beautify the plot with any plotting library.
- Example:
```python
import json
with open('my_plot_plotdata.json') as f:
    data = json.load(f)
# data['time'], data['values'], data['labels'], ...
```

## Notes
- All arrays are saved as lists for compatibility.
- Colors are Matplotlib color codes.
- Not all fields are always present; empty lists are used if not specified.
