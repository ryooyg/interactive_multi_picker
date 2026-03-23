# Interactive Multi-Station Phase Picker for machine learning in seismology.

A Python-based interactive GUI tool for visualizing seismograms, manually picking P and S wave arrivals, and utilizing deep learning models for automatic phase picking. Built with Matplotlib and ObsPy, this tool provides an intuitive environment for seismologists to rapidly review and refine earthquake data across multiple stations.

## 🌟 Key Features

- **3-Component Visualization**: View Vertical (Z), North (N), and East (E) components simultaneously.
- **Interactive Picking**: Easily mark and adjust P (red) and S (blue) wave arrivals using mouse clicks, drag-and-drop, and keyboard shortcuts.
- **Deep Learning Integration**: Run **EQTransformer** or **PhaseNet** (via `SeisBench`) on the fly. Generates automatic picks and visualizes continuous probability curves as background shading on the waveforms.
- **Z-Stack (Record Section) View**: Automatically aligns and stacks the Z-components of all picked stations, sorted by P-wave arrival time. This view smartly zooms into the temporal region of interest, allowing you to drag and adjust picks across multiple stations at a glance.
- **High-Precision Time Handling**: Uses vectorized time arrays and Matplotlib's `AutoDateFormatter` to prevent float precision loss, ensuring accurate visual alignment of waveforms and picks down to the microsecond level during extreme zooming.
- **On-the-fly Signal Processing**: Toggle a 1-10Hz bandpass filter or remove instrument responses (velocity output) in real-time.
- **Data I/O**: Import preliminary picks from a CSV file and export your finalized, refined picks to `refined_picks.csv`.

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
pip install obspy numpy pandas matplotlib Pillow

# For AI features:
pip install seisbench torch
```

---

## 🚀 Usage

Run the script from the command line, providing the target MiniSEED file.

### Basic Command
```bash
python interactive_multi_picker.py --mseed ./data/your_waveform_data.mseed
```

### Advanced Command
```bash
python interactive_multi_picker.py \
    --mseed ./data/your_waveform_data.mseed \
    --picks ./initial_picks.csv \
    --filter \
    --model eqtransformer \
    --pretrained stead \
    --inv ./metadata/total_inv.xml
```

### CLI Arguments
*   `--mseed` (Required): Path to the MiniSEED waveform data file.
*   `--picks`: Path to a CSV file containing initial/auto picks to load on startup.
*   `--filter`: Flag to apply a basic 1-10Hz Bandpass filter automatically.
*   `--model`: AI model to use for automatic picking (`eqtransformer` or `phasenet`).
*   `--pretrained`: Pretrained weights for the AI model (e.g., `original`, `stead`, `ethz`). Default is `original`.
*   `--inv`: Path to the StationXML file (`.xml`) for instrument response removal.

---

## 🎮 Controls & Shortcuts

The GUI is designed for rapid operation without needing to click through menus.

### Mouse Controls
*   **Left Click + Drag**: Drag the vertical Red (P) or Blue (S) lines to precisely adjust the arrival time.
*   **Scroll Wheel**: Zoom in/out horizontally (Time axis).

### Keyboard Shortcuts
*   `p`: Set the **P-wave** pick at the current mouse cursor location.
*   `s`: Set the **S-wave** pick at the current mouse cursor location.
*   `c`: **Clear** both P and S picks for the currently displayed station.
*   `Right Arrow`, `n`, or `Page Down`: Move to the **Next** station.
*   `Left Arrow`, `b`, or `Page Up`: Move to the **Previous** station.

### GUI Buttons
*   **< / >**: Navigate between stations.
*   **EQTRANSFORMER / PHASENET**: Toggle the active AI model.
*   **W: ORIGINAL / STEAD / ETHZ**: Toggle the AI model's pretrained weights.
*   **AI Pick**: Execute the selected AI model on the *current* station.
*   **Filter: ON/OFF**: Toggle the 1-10Hz bandpass filter globally.
*   **Resp: ON/OFF**: Toggle instrument response removal for the *current* station (requires `--inv`).
*   **Clear**: Delete picks for the current station.
*   **Z-Stack**: Open the multi-station Record Section view (requires at least 1 P-pick).
*   **SAVE**: Export the current state of all picks to `refined_picks.csv`.
*   **EXIT**: Save and safely close the application.

---

## 📂 Output Format

When you click **SAVE** or **EXIT**, the tool writes all active picks to `refined_picks.csv` in the current directory.

**Format example (`refined_picks.csv`):**
```csv
Network_Station,Phase,Arrival_Time,Confidence
IU.INCN,P,2025-11-25T09:01:40.050000Z,0.95
KS.CGWB,P,2025-11-25T09:02:09.228285Z,1.0000
KS.CGWB,S,2025-11-25T09:02:47.285756Z,1.0000
KS.CIGB,P,2025-11-25T09:02:25.538630Z,1.0000
KS.CIGB,S,2025-11-25T09:03:13.563534Z,1.0000
```
*(Note: Manual picks are saved with a default confidence of 1.0)*
