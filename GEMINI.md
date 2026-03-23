# Development History - Interactive Multi-Station Phase Picker

This document records the foundational mandates, architectural decisions, and the evolution of the Interactive Multi-Station Earthquake Phase Picker project.

## 🐉 Core Mandates & Preferences
- **Language**: Python is the preferred language for development.
- **Documentation**: All code comments and technical documentation must be written in **English**.
- **User Alias**: The assistant is referred to as **Dragon (드래곤)**.

## 🛠️ Key Technical Implementations

### 1. GUI & Visualization
- **Branding**: Added `kitvalley.png` as a logo in the bottom-right corner of the main interface using a dedicated Matplotlib axis.
- **Precision**: Implemented vectorized time array generation (`t0 + times / 86400.0`) to maintain microsecond-level accuracy and improve rendering performance compared to iterative datetime object creation.
- **Dynamic Formatting**: Integrated `mdates.AutoDateFormatter` in both the main view and Z-stack to display time units (including milliseconds) appropriate to the zoom level.
- **Navigation**: Enhanced accessibility by adding `Page Up` (Previous Station) and `Page Down` (Next Station) keyboard shortcuts.

### 2. Z-Stack (Record Section) Enhancements
- **Layout**: Relocated the vertical scrollbar from the left to the right side of the screen for better ergonomics.
- **Smart Zooming**: Configured the initial X-axis view to start 10 seconds before the first P-wave pick and end 60 seconds after the last P-wave pick.
- **Pick Filtering**: Added logic to exclude P/S pick markers from the visualization if they fall outside a 180-second window relative to the earliest P-wave arrival, keeping the display focused on the primary event.

### 3. AI Automatic Picking (SeisBench Integration)
- **Data Integrity**: Implemented a "3-Component Overlapping Window" logic in `_run_ai_picker`. The tool now automatically trims the Z, N, and E traces to their common intersection time window before feeding them to the AI model, ensuring synchronized input.
- **Local Model Support**: Added the `korea` weight option. When `--pretrained korea` is used, the system attempts to load local files named `eqtransformer_korea.pth` or `phasenet_korea.pth` from the root directory.
- **Efficiency**: Removed mandatory `use_backup_repository()` calls to prioritize local SeisBench cache and reduce startup warnings.

### 4. Data Handling & Output
- **Precision**: Updated the CSV export logic to format `Confidence` values to exactly 4 decimal places.
- **Flexibility**: Reverted mandatory date unification logic after user feedback, ensuring that the original MiniSEED header timestamps are preserved exactly as-is in the final output.

## 📂 Project Structure Notes
- **Weight Files**: Custom weights (`*_korea.pth`) must be placed in the project root alongside `interactive_multi_picker.py`.
- **Dependency Management**: Supports both standard `pip` environments and modern `uv` workflows (`uv sync`).
