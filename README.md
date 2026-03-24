# Interactive Multi-Station Phase Picker for machine learning in seismology.

A Python-based interactive GUI tool for visualizing seismograms, manually picking P and S wave arrivals, and utilizing deep learning models for automatic phase picking. Built with Matplotlib and ObsPy, this tool provides an intuitive environment for seismologists to rapidly review and refine earthquake data across multiple stations.

## 🌟 Key Features

- **3-Component Visualization**: View Vertical (Z), North (N), and East (E) components simultaneously.
- **Interactive Picking**: Easily mark and adjust P (red) and S (blue) wave arrivals using mouse clicks, drag-and-drop, and keyboard shortcuts.
- **Deep Learning Integration**: Run **EQTransformer** or **PhaseNet** (via `SeisBench`) on the fly. Generates automatic picks and visualizes continuous probability curves as background shading on the waveforms.
- **Hypocenter & Magnitude Analysis**: Automatically calculate earthquake location (Lat, Lon, Depth, Origin Time) and Local Magnitude (ML) using picked phases.
- **Interactive Mapping**: Generate high-quality location maps with station labels and event parameters (time, magnitude) using `Cartopy`.
- **Z-Stack (Record Section) View**: Automatically aligns and stacks the Z-components of all picked stations, sorted by P-wave arrival time. This view smartly zooms into the temporal region of interest, allowing you to drag and adjust picks across multiple stations at a glance.
- **High-Precision Time Handling**: Uses vectorized time arrays and Matplotlib's `AutoDateFormatter` to prevent float precision loss, ensuring accurate visual alignment of waveforms and picks down to the microsecond level during extreme zooming.
- **On-the-fly Signal Processing**: Toggle a 1-10Hz bandpass filter or remove instrument responses (velocity output) in real-time.
- **Advanced Data I/O**: Export detailed location reports, station-specific magnitude CSVs, and refined pick files to a user-specified output directory.

---

## 🛠️ Installation

### 1. Clone the Repository
First, clone this repository to your local machine:
```bash
git clone https://github.com/your-username/interactive_multi_picker.git
cd interactive_multi_picker
```

### 2. Install Dependencies

#### Option A: Using `uv` (Recommended)
If you have [uv](https://github.com/astral-sh/uv) installed, simply run:
```bash
# This will automatically create a venv and install all dependencies from uv.lock
uv sync
```

#### Option B: Using `pip`
It is recommended to use a virtual environment:
```bash
# Create and activate venv
python -m venv .venv
# Windows: .venv\Scripts\activate | macOS/Linux: source .venv/bin/activate

# Install dependencies
pip install obspy numpy pandas matplotlib Pillow scipy cartopy

# For AI features:
pip install seisbench torch
```

---

## 🚀 Usage

Run the script from the command line, providing the target MiniSEED file.

### Basic Command
```bash
python interactive_multi_picker.py --mseed ./data/your_waveform_data.mseed --inv ./metadata/total_inv.xml
```

### Advanced Command
```bash
python interactive_multi_picker.py \
    --mseed ./data/your_waveform_data.mseed \
    --picks ./initial_picks.csv \
    --filter \
    --model eqtransformer \
    --pretrained stead \
    --inv ./metadata/total_inv.xml \
    --output ./results/event_01
```

### CLI Arguments
*   `--mseed` (Required): Path to the MiniSEED waveform data file. Supports wildcards (e.g., `./data/20260311/*`).
*   `--picks`: Path to a CSV file containing initial/auto picks. **If provided with `--inv`, the tool will automatically calculate the hypocenter, magnitude, and generate a map upon startup.**
*   `--filter`: Flag to apply a basic 1-10Hz Bandpass filter automatically.
*   `--model`: AI model to use for automatic picking (`eqtransformer` or `phasenet`).
*   `--pretrained`: Pretrained weights for the AI model. Options: `original`, `stead`, `ethz`, or `korea`. Default is `original`.
*   `--inv`: Path to the StationXML file (`.xml`) for instrument response removal and **hypocenter location**.
*   `--output`: Directory to save all results (CSV, TXT, PNG). 
    *   **Smart Default**: If not provided, it automatically creates a folder based on your input. For a single file `event.mseed`, it creates `event_out/`. For a wildcard path `data/event_dir/*`, it smartly creates `data/event_dir_out/`.

### 🇰🇷 Using Custom 'Korea' Weights
You can use locally trained weights specifically optimized for the Korean peninsula. 
1. Place the trained weight files (`eqtransformer_korea.pth` and/or `phasenet_korea.pth`) directly in the **root directory of this project**.
2. Run the tool using the `--pretrained korea` flag.

---

## 🎮 Controls & Shortcuts

### Mouse Controls
*   **Left Click + Drag**: Drag vertical Red (P) or Blue (S) lines to adjust arrival times.
*   **Scroll Wheel**: Zoom in/out horizontally (Time axis).

### Keyboard Shortcuts
*   `p`: Set **P-wave** pick at cursor.
*   `s`: Set **S-wave** pick at cursor.
*   `c`: **Clear** picks for the current station.
*   `Right Arrow` / `Page Down`: **Next** station.
*   `Left Arrow` / `Page Up`: **Previous** station.

### GUI Buttons
*   **Locate**: Calculate hypocenter and magnitude (ML), save reports, and display the map. (Requires ≥3 P-picks and `--inv`).
*   **AI Pick**: Execute the selected AI model on the *current* station.
*   **Z-Stack**: Open the multi-station Record Section view.
*   **SAVE**: Export refined picks to `refined_picks.csv`.
*   **EXIT**: Save and close.

---

## 📂 Output Files

All outputs are saved in the directory specified by `--output`.

1.  **`refined_picks.csv`**: Finalized P and S arrival times.
2.  **`location_report.txt`**: Detailed text report including Origin Time, Lat/Lon, Depth, ML, and residuals.
3.  **`station_magnitudes.csv`**: Per-channel magnitude calculations (Distance, Amplitude, ML).
4.  **`location_map.png`**: Visual map showing the epicenter and used stations with event info in the title.
