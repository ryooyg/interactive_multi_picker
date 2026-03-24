# Interactive Multi-Station Phase Picker for machine learning in seismology.

A Python-based interactive GUI tool for visualizing seismograms, manually picking P and S wave arrivals, and utilizing deep learning models for automatic phase picking. Built with Matplotlib and ObsPy, this tool provides an intuitive environment for seismologists to rapidly review and refine earthquake data across multiple stations.

## 🌟 Key Features

- **3-Component Visualization**: View Vertical (Z), North (N), and East (E) components simultaneously.
- **Interactive Picking**: Easily mark and adjust P (red) and S (blue) wave arrivals using mouse clicks, drag-and-drop, and keyboard shortcuts.
- **Deep Learning Integration**: Run **EQTransformer** or **PhaseNet** (via `SeisBench`) on the fly. Generates automatic picks and visualizes continuous probability curves as background shading on the waveforms.
- **Hypocenter & Magnitude Analysis**: Automatically calculate earthquake location (Lat, Lon, Depth, Origin Time) and Local Magnitude (ML) using picked phases.
- **Interactive Mapping & Zooming**: Generate high-quality location maps with station labels and event parameters. Supports **mouse wheel zoom in/out** and displays precise coordinates above the epicenter.
- **Z-Stack (Record Section) View**: Automatically aligns and stacks the Z-components of all picked stations. Supports **adding, modifying, and clearing picks directly** in the record section with keyboard shortcuts.
- **Automated Batch Processing**: Run the entire pipeline (picking, locating, magnitude calculation, and plotting) in **headless batch mode** without opening a GUI.
- **High-Precision Time Handling**: Uses vectorized time arrays and Matplotlib's `AutoDateFormatter` to ensure microsecond-level accuracy during extreme zooming.
- **On-the-fly Signal Processing**: Toggle a 1-10Hz bandpass filter or remove instrument responses (velocity output) in real-time.
- **Advanced Data I/O**: Export detailed location reports, station-specific magnitude CSVs, and refined pick files to a smart auto-generated output directory.

---

## 🛠️ Installation

### 1. Clone the Repository
First, clone this repository to your local machine:
```bash
git clone https://github.com/ryooyg/interactive_multi_picker.git
cd interactive_multi_picker
```

### 2. Install Dependencies

#### Option A: Using `uv` (Recommended)
If you have [uv](https://github.com/astral-sh/uv) installed, simply run:
```bash
uv sync
```

#### Option B: Using `pip`
```bash
pip install obspy numpy pandas matplotlib Pillow scipy cartopy seisbench torch
```

---

## 🚀 Usage

Run the script from the command line, providing the target MiniSEED file.

### Basic Command
```bash
python interactive_multi_picker.py --mseed ./data/your_waveform_data.mseed --inv ./metadata/total_inv.xml
```

### Batch Mode Command (Headless)
```bash
python interactive_multi_picker.py --mseed ./data/20260311/* --model eqtransformer --pretrained korea --inv ./total_inv.xml --batch
```

### Advanced Command (GUI)
```bash
python interactive_multi_picker.py \
    --mseed ./data/your_waveform_data.mseed \
    --picks ./initial_picks.csv \
    --filter \
    --model eqtransformer \
    --pretrained korea \
    --inv ./metadata/total_inv.xml \
    --output ./results/event_01
```

### CLI Arguments
*   `--mseed` (Required): Path to the MiniSEED waveform data file. Supports wildcards (e.g., `./data/20260311/*`).
*   `--picks`: Path to a CSV file containing initial/auto picks. **If provided with `--inv`, the tool will automatically calculate the hypocenter and magnitude upon startup.**
*   `--filter`: Flag to apply a basic 1-10Hz Bandpass filter automatically.
*   `--model`: AI model to use (`eqtransformer` or `phasenet`).
*   `--pretrained`: Pretrained weights (`original`, `stead`, `ethz`, or `korea`).
*   `--inv`: Path to the StationXML file (`.xml`) for instrument response removal and **hypocenter location**.
*   `--output`: Directory to save results. 
    *   **Smart Default**: If not provided, it creates a folder based on your input. For `event.mseed`, it creates `event_out/`. For a wildcard path `data/dir/*`, it creates `data/dir_out/`.
*   `--batch`: Flag to run in **headless batch mode**. Automatically performs picking, location, and saves all outputs including waveform summary plots without opening the GUI.

---

## 🎮 Controls & Shortcuts

### Mouse Controls
*   **Left Click + Drag**: Adjust vertical Red (P) or Blue (S) lines.
*   **Scroll Wheel**: Zoom in/out horizontally (Time axis in Main/Z-Stack windows) or **Zoom the Map** (in Map window).

### Keyboard Shortcuts
*   `p`: Set **P-wave** pick at cursor. (Main Window and Z-Stack)
*   `s`: Set **S-wave** pick at cursor. (Main Window and Z-Stack)
*   `c`: **Clear** picks for the current station. (Main Window and Z-Stack)
*   `Right Arrow` / `Page Down`: **Next** station.
*   `Left Arrow` / `Page Up`: **Previous** station.

---

## 📂 Output Files

All outputs are saved in the directory specified by `--output`.

1.  **`refined_picks.csv`**: Finalized P and S arrival times.
2.  **`location_report.txt`**: Detailed text report (Origin Time, Lat/Lon, Depth, ML, residuals).
3.  **`station_magnitudes.csv`**: Per-channel magnitude details (Distance, Amplitude, ML).
4.  **`location_map.png`**: Map image showing epicenter, used stations, and event parameters.
5.  **`record_section_page_XX.png`**: (Batch mode only) Static waveform plots with 15 stations per page, sorted by arrival time. Each plot shows 150 seconds of data starting 10 seconds before the first P-arrival.
