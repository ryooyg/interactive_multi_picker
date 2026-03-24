import argparse
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.font_manager as fm
from obspy import read, UTCDateTime, Stream
import numpy as np
import pandas as pd
import os
import csv
from matplotlib.widgets import Button, RadioButtons

# Font setup (Windows/Mac/Linux support)
import platform
if platform.system() == 'Windows':
    plt.rc('font', family='Malgun Gothic')
elif platform.system() == 'Darwin': # Mac
    plt.rc('font', family='AppleGothic')
else: # Linux
    plt.rc('font', family='NanumGothic')
plt.rc('axes', unicode_minus=False) # Prevent minus sign breaking

class MultiStationPicker:
    def __init__(self, mseed_file, apply_filter=False, auto_picks_file=None, ai_model=None, pretrained="original", inv_file=None, output_dir=".", batch_mode=False):
        self.output_dir = output_dir
        self.batch_mode = batch_mode
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir, exist_ok=True)
            
        # Disable default matplotlib shortcuts (prevent s for save, etc. - global setting before figure creation)
        plt.rcParams['keymap.save'] = ''
        plt.rcParams['keymap.fullscreen'] = ''
        plt.rcParams['keymap.home'] = ''
        plt.rcParams['keymap.back'] = ''
        plt.rcParams['keymap.forward'] = ''

        self.mseed_file = mseed_file
        self.apply_filter = apply_filter
        
        self.auto_picks_file = auto_picks_file
        self.ai_model = ai_model if ai_model else 'eqtransformer'
        self.pretrained = pretrained
        
        self.inv = None
        if inv_file and os.path.exists(inv_file):
            try:
                from obspy import read_inventory
                print(f"[*] Loading Inventory: {inv_file}")
                self.inv = read_inventory(inv_file)
            except Exception as e:
                print(f"[-] Failed to load Inventory: {e}")
        
        # 1. Load data (separate raw and filtered)
        self._load_data()
        
        # Create initial groupings for all stations
        self.station_names = []
        groups = {}
        for tr in self.st_raw:
            net_sta = f"{tr.stats.network}.{tr.stats.station}"
            if net_sta not in groups:
                groups[net_sta] = {'Z': None, 'N': None, 'E': None}
                self.station_names.append(net_sta)
            comp = tr.stats.component
            if comp.endswith('Z'): groups[net_sta]['Z'] = tr
            elif comp.endswith('N') or comp.endswith('1'): groups[net_sta]['N'] = tr
            elif comp.endswith('E') or comp.endswith('2'): groups[net_sta]['E'] = tr
            
        self.stations_raw = {k: v for k, v in groups.items() if any(v.values())}
        self.station_names = list(self.stations_raw.keys())
        
        # Dictionary to store processed data (response removed/filtered) per station
        self.stations = {}
        
        # All stations start with instrument response removal disabled (False)
        self.resp_status = {sta: False for sta in self.station_names}
        self.current_idx = 0
        
        # Process only the station on the current screen at startup
        if self.station_names:
            self._process_station(self.station_names[self.current_idx])
        
        # 2. Initialize picking database
        self.picks_db = {sta: {'p': {'mpl_num': None, 'conf': 0.0}, 
                               's': {'mpl_num': None, 'conf': 0.0}} for sta in self.station_names}
        self.lines = {'p': [], 's': []}
        self.dragging = None
        self.ai_annotations = {} # Probability curve storage
        
        if not self.station_names:
            print("[-] No station data to process.")
            return

        # --- Batch Mode Execution ---
        if self.batch_mode:
            print(f"\n[*] Running in BATCH MODE")
            if ai_model:
                self._run_ai_picker(self.ai_model)
            elif self.auto_picks_file:
                self._load_auto_picks()
                
            if self.inv:
                self._btn_locate_clicked(None)
            else:
                print("[-] Skipping location calculation (No --inv provided in batch mode).")
                
            self._save_to_csv()
            self._generate_batch_waveform_plots()
            print("[*] Batch processing completed.")
            return

        # 3. Initialize UI and graphs (Interactive Mode only)
        self._setup_plot()

        # 4. Load auto picks or perform AI picking (execute after UI is ready)
        if self.auto_picks_file:
            self._load_auto_picks()
        elif ai_model: # Auto-run at startup if model is specified via CLI argument
            self._run_ai_picker(self.ai_model)
            
        # Draw the first screen reflecting the results (must run after step 4)
        self._draw_current_station()

        # 5. Auto-locate if picks were provided
        if self.auto_picks_file and self.inv:
            print("\n[*] Initializing Auto-Locate from provided picks...")
            # We use a slight delay using a timer so the UI has time to render fully first
            timer = self.fig.canvas.new_timer(interval=500)
            timer.single_shot = True
            timer.add_callback(self._btn_locate_clicked, None)
            timer.start()

    def _load_data(self):
        try:
            import glob
            
            # If wildcards are used, we need to carefully collect only actual files, not directories.
            if '*' in self.mseed_file or '?' in self.mseed_file:
                files = [f for f in glob.glob(self.mseed_file) if os.path.isfile(f)]
                if not files:
                    raise ValueError(f"No files found matching: {self.mseed_file}")
                
                self.st_raw = Stream()
                for f in files:
                    try:
                        self.st_raw += read(f)
                    except Exception as e:
                        print(f"[-] Warning: Failed to read {f}: {e}")
            else:
                self.st_raw = read(self.mseed_file)
                
            if len(self.st_raw) == 0:
                raise ValueError("No valid seismic data could be loaded.")
                
            self.st_raw.detrend('linear')
            print(f"[*] Data loaded: {len(self.st_raw)} traces")
        except Exception as e:
            print(f"[-] Failed to load data: {e}")
            raise

    # Removed previous global _apply_processing, added individual station processing function instead
    def _process_station(self, sta):
        """Extract data for a specific station (sta) and process it according to Resp/Filter status."""
        comps_raw = self.stations_raw[sta]
        
        # Gather raw traces and create a new stream
        st_sta = Stream()
        for comp_key in ['Z', 'N', 'E']:
            if comps_raw[comp_key]:
                st_sta.append(comps_raw[comp_key].copy())
                
        # Remove instrument response (check current station status)
        if self.resp_status[sta] and self.inv:
            try:
                print(f"[*] {sta} Station: Removing instrument response...")
                st_sta.remove_response(inventory=self.inv, output="VEL")
            except Exception as e:
                print(f"[-] {sta} failed to remove instrument response: {e}")
                self.resp_status[sta] = False
                
        # Apply filter (follows global toggle status)
        if self.apply_filter:
            st_sta.filter('bandpass', freqmin=1.0, freqmax=10.0)
            
        # Store processed results in self.stations[sta]
        self.stations[sta] = {'Z': None, 'N': None, 'E': None}
        for tr in st_sta:
            comp = tr.stats.component
            if comp.endswith('Z'): self.stations[sta]['Z'] = tr
            elif comp.endswith('N') or comp.endswith('1'): self.stations[sta]['N'] = tr
            elif comp.endswith('E') or comp.endswith('2'): self.stations[sta]['E'] = tr

    def _run_ai_picker(self, model_name, station_name=None):
        target_info = f"Station({station_name})" if station_name else "All data"
        print(f"\n[*] AI 모델({model_name}, weight={self.pretrained})로 {target_info} Analyzing...")
        
        # Update status display
        if hasattr(self, 'ax_z'):
            self.ax_z.set_title(f" Analyzing... Please wait ({target_info}) ", color='blue', fontweight='bold')
            self.fig.canvas.draw()

        try:
            import seisbench
            import seisbench.models as sbm
            import torch
        except ImportError:
            print("[-] seisbench is not installed. Please run pip install seisbench torch.")
            return

        try:
            if model_name.lower() == 'eqtransformer':
                if self.pretrained.lower() == 'korea':
                    model = sbm.EQTransformer()
                    model.load_state_dict(torch.load("eqtransformer_korea.pth", weights_only=True))
                    print("[*] Loaded local weights: eqtransformer_korea.pth")
                else:
                    model = sbm.EQTransformer.from_pretrained(self.pretrained)
            elif model_name.lower() == 'phasenet':
                if self.pretrained.lower() == 'korea':
                    model = sbm.PhaseNet()
                    model.load_state_dict(torch.load("phasenet_korea.pth", weights_only=True))
                    print("[*] Loaded local weights: phasenet_korea.pth")
                else:
                    model = sbm.PhaseNet.from_pretrained(self.pretrained)
            else:
                print(f"[-] Unsupported model: {model_name}")
                return

            if torch.cuda.is_available():
                model.cuda()
                print("[*] GPU acceleration enabled")

            # Filter target data and extract 3-component overlapping window
            if station_name:
                try:
                    net, stacode = station_name.split('.')
                    st_to_classify_raw = self.st_raw.select(network=net, station=stacode)
                except ValueError:
                    st_to_classify_raw = self.st_raw.select(station=station_name)
                    
                # Find common time window where all 3 components exist
                if len(st_to_classify_raw) > 0:
                    start_times = [tr.stats.starttime for tr in st_to_classify_raw]
                    end_times = [tr.stats.endtime for tr in st_to_classify_raw]
                    
                    # Find the latest start time and earliest end time (intersection)
                    common_start = max(start_times)
                    common_end = min(end_times)
                    
                    if common_start < common_end:
                        # Trim to intersection window (use copy to preserve raw data)
                        st_to_classify = st_to_classify_raw.copy().trim(common_start, common_end)
                        print(f"[*] Extracted 3-component common window: {common_start} ~ {common_end}")
                    else:
                        print(f"[-] {station_name} station does not have overlapping time windows for all 3 components.")
                        st_to_classify = Stream()
                else:
                    st_to_classify = Stream()
                    
                # Clear existing probability data when analyzing a specific station
                self.ai_annotations[station_name] = Stream()
            else:
                # All data를 분석할 때는 각 관측소별로 교집합 구간을 찾아야 함
                st_to_classify = Stream()
                groups = {}
                for tr in self.st_raw:
                    net_sta = f"{tr.stats.network}.{tr.stats.station}"
                    if net_sta not in groups: groups[net_sta] = []
                    groups[net_sta].append(tr)
                    
                for net_sta, tr_list in groups.items():
                    if len(tr_list) > 0:
                        start_times = [t.stats.starttime for t in tr_list]
                        end_times = [t.stats.endtime for t in tr_list]
                        common_start = max(start_times)
                        common_end = min(end_times)
                        if common_start < common_end:
                            st_to_classify += Stream(tr_list).copy().trim(common_start, common_end)
                            
                self.ai_annotations = {}

            if len(st_to_classify) == 0:
                print(f"[-] Could not find data to analyze: {station_name}")
                return

            # 1. Perform Classification (Picking)
            classify_output = model.classify(st_to_classify)
            picks = classify_output.picks if hasattr(classify_output, 'picks') else classify_output
            
            # 2. Perform Annotation (Probability curves)
            try:
                annotations = model.annotate(st_to_classify)
                for tr in annotations:
                    net = tr.stats.network
                    sta_code = tr.stats.station
                    # Match station names
                    sta_key = f"{net}.{sta_code}" if net else sta_code
                    if not sta_key and station_name: sta_key = station_name
                    
                    norm_key = self._normalize_name(sta_key)
                    target_sta = next((ls for ls in self.station_names if norm_key == self._normalize_name(ls)), None)

                    if target_sta:
                        if target_sta not in self.ai_annotations:
                            self.ai_annotations[target_sta] = Stream()
                        self.ai_annotations[target_sta].append(tr)
            except Exception as e:
                print(f"[-] Failed to generate probability curves: {e}")
            
            # 3. Apply picking results
            count = 0
            for p in picks:
                parts = p.trace_id.split('.')
                sta = f"{parts[0]}.{parts[1]}" if len(parts) >= 2 else ""
                if not sta: continue
                
                phase = p.phase.lower()
                if phase not in ['p', 's']: continue
                    
                norm_sta = self._normalize_name(sta)
                for loaded_sta in self.station_names:
                    if norm_sta == self._normalize_name(loaded_sta):
                        try:
                            peak_time = p.peak_time if hasattr(p, 'peak_time') and p.peak_time else p.start_time
                            mpl_num = mdates.date2num(peak_time.datetime.replace(tzinfo=None))
                            conf = float(p.peak_value)
                            
                            # Update only if confidence is higher than existing or if it is the first time
                            db_entry = self.picks_db[loaded_sta][phase]
                            if db_entry['mpl_num'] is None or conf > db_entry['conf']:
                                self.picks_db[loaded_sta][phase] = {'mpl_num': mpl_num, 'conf': conf}
                                count += 1
                        except:
                            pass
            print(f"[*] AI picking complete: {count} phase picks applied.\n")
        except Exception as e:
            print(f"[-] Error during AI picking: {e}")

    def _normalize_name(self, name):
        return str(name).replace('_', '.').strip().lower()

    def _load_auto_picks(self):
        if not os.path.exists(self.auto_picks_file):
            print(f"[-] File not found: {self.auto_picks_file}")
            return
            
        try:
            df = pd.read_csv(self.auto_picks_file)
            df.columns = [c.lower() for c in df.columns]
            count = 0
            for _, row in df.iterrows():
                csv_sta = str(row['network_station']) if 'network_station' in df.columns else str(row.get('station', ''))
                phase = str(row.get('phase', '')).lower()
                time_str = str(row.get('arrival_time', '')) if 'arrival_time' in df.columns else str(row.get('time', ''))
                conf = float(row.get('confidence', 1.0))
                if not csv_sta or not phase or not time_str or time_str == 'nan': continue

                norm_csv_sta = self._normalize_name(csv_sta)
                for loaded_sta in self.station_names:
                    if norm_csv_sta == self._normalize_name(loaded_sta):
                        if phase in ['p', 's']:
                            try:
                                parsed_dt = UTCDateTime(time_str)
                                # Fix: Reference always-existing self.stations_raw instead of lazy-loaded self.stations
                                ref_tr = next((tr for tr in self.stations_raw[loaded_sta].values() if tr is not None), None)
                                if ref_tr:
                                    ref_date = ref_tr.stats.starttime
                                    # If CSV Y/M/D differs from actual data, overwrite date using data as reference (keep time)
                                    if parsed_dt.year != ref_date.year or parsed_dt.month != ref_date.month or parsed_dt.day != ref_date.day:
                                        fixed_time_str = f"{ref_date.year:04d}-{ref_date.month:02d}-{ref_date.day:02d}T{parsed_dt.hour:02d}:{parsed_dt.minute:02d}:{parsed_dt.second:02d}.{parsed_dt.microsecond:06d}Z"
                                        parsed_dt = UTCDateTime(fixed_time_str)
                                        
                                mpl_num = mdates.date2num(parsed_dt.datetime.replace(tzinfo=None))
                                self.picks_db[loaded_sta][phase] = {'mpl_num': mpl_num, 'conf': conf}
                                count += 1
                            except Exception as e:
                                print(f"[-] Time parsing error ({time_str} / {loaded_sta}): {e}")
                        break
            print(f"[*] Auto-picks {count}loaded (applied automatic date correction)")
        except Exception as e:
            print(f"[-] Error during loading: {e}")

    def _setup_plot(self):
        # Disable default matplotlib shortcuts (prevent s for save, etc.)
        plt.rcParams['keymap.save'] = ''
        plt.rcParams['keymap.fullscreen'] = ''
        plt.rcParams['keymap.home'] = ''
        plt.rcParams['keymap.back'] = ''
        plt.rcParams['keymap.forward'] = ''
        
        self.fig, (self.ax_z, self.ax_n, self.ax_e) = plt.subplots(3, 1, figsize=(12, 9), sharex=True)
        # Set window title and icon
        try:
            manager = self.fig.canvas.manager
            manager.set_window_title('interactive multi picker of Earthquake')
            
            # Icon setup logic (enhanced method - includes Tkinter hack for Windows)
            icon_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'kitvalley.png')
            if os.path.exists(icon_path):
                import tkinter as tk
                if hasattr(manager, 'window'):
                    window = manager.window
                    try:
                        # 1. Read PNG using PIL(Pillow) (most stable)
                        from PIL import Image, ImageTk
                        pil_img = Image.open(icon_path)
                        tk_img = ImageTk.PhotoImage(pil_img)
                        
                        # 2. Apply icon to current window
                        window.iconphoto(True, tk_img)
                        
                        # 3. Force apply to hidden top-level root window if exists (fixes Windows taskbar bug)
                        if window.master is not None:
                            window.master.iconphoto(True, tk_img)
                            
                        # Prevent garbage collection
                        window._tk_icon = tk_img 
                        
                    except Exception as e_pil:
                        print(f"[*] Failed to apply PIL icon: {e_pil}")
                        try:
                            # 4. Last resort: default PhotoImage
                            img = tk.PhotoImage(file=icon_path)
                            window.iconphoto(True, img)
                            if window.master is not None:
                                window.master.iconphoto(True, img)
                            window._tk_icon_fallback = img
                        except Exception as e_tk:
                            pass
                            
        except Exception as e:
            print(f"[*] Notice: Issue occurred while setting window icon or title ({e})")
            
        plt.subplots_adjust(left=0.08, right=0.95, top=0.85, bottom=0.1, hspace=0.18)
        
        self.formatter = mdates.DateFormatter('%H:%M:%S')
        self.ax_e.xaxis.set_major_formatter(self.formatter)
        plt.setp(self.ax_e.get_xticklabels(), rotation=30, ha='right')
        
        # Create secondary axis (TwinX) - for probability curves
        self.ax_z_prob = self.ax_z.twinx()
        self.ax_n_prob = self.ax_n.twinx()
        self.ax_e_prob = self.ax_e.twinx()
        for ax in [self.ax_z_prob, self.ax_n_prob, self.ax_e_prob]:
            ax.set_ylim(0, 0.5) 
            ax.set_yticks([]) 
            ax.set_navigate(False)
        
        # --- 3D button creation helper function ---
        def add_3d_button(rect, label, color, hovercolor='skyblue'):
            shadow_offset = 0.002
            ax_shadow = self.fig.add_axes([rect[0]+shadow_offset, rect[1]-shadow_offset, rect[2], rect[3]], label=f"shadow_{label}")
            ax_shadow.patch.set_facecolor('black')
            ax_shadow.patch.set_alpha(0.3)
            ax_shadow.set_xticks([]); ax_shadow.set_yticks([])
            
            ax = self.fig.add_axes(rect, label=f"btn_{label}")
            btn = Button(ax, label, color=color, hovercolor=hovercolor)
            btn.label.set_fontsize(10)
            btn.label.set_weight('bold')
            return btn

        h = 0.045
        y = 0.925

        # 1. Navigation (Left)
        self.btn_prev = add_3d_button([0.01, y, 0.035, h], '<', 'lightgrey')
        self.btn_prev.on_clicked(self._btn_prev_clicked)
        
        self.btn_next = add_3d_button([0.05, y, 0.035, h], '>', 'lightgrey')
        self.btn_next.on_clicked(self._btn_next_clicked)

        # 2. Cyclical model & weight selection buttons
        self.available_models = ['eqtransformer', 'phasenet']
        self.available_weights = ['original', 'stead', 'ethz']
        
        self.btn_model = add_3d_button([0.095, y, 0.145, h], self.ai_model.upper(), 'lightgoldenrodyellow')
        self.btn_model.on_clicked(self._btn_model_clicked)

        self.btn_weight = add_3d_button([0.245, y, 0.12, h], f'W: {self.pretrained.upper()}', 'lightcyan')
        self.btn_weight.on_clicked(self._btn_weight_clicked)

        # 3. Option buttons (Center)
        self.btn_ai = add_3d_button([0.37, y, 0.08, h], 'AI Pick', 'plum')
        self.btn_ai.on_clicked(self._btn_ai_clicked)

        filter_color = 'lightgreen' if self.apply_filter else 'white'
        self.btn_filter = add_3d_button([0.455, y, 0.09, h], f'Filter: {"ON" if self.apply_filter else "OFF"}', filter_color)
        self.btn_filter.on_clicked(self._btn_filter_clicked)

        # Initial screen station Resp status (default False -> show ON)
        sta = self.station_names[self.current_idx]
        is_resp = self.resp_status.get(sta, False)
        resp_label = f'Resp: {"OFF" if is_resp else "ON"}'
        resp_color = 'lightblue' if is_resp else 'white'
        self.btn_resp = add_3d_button([0.55, y, 0.09, h], resp_label, resp_color)
        self.btn_resp.on_clicked(self._btn_resp_clicked)

        self.btn_clear = add_3d_button([0.645, y, 0.05, h], 'Clear', 'white')
        self.btn_clear.on_clicked(self._btn_clear_clicked)

        self.btn_zstack = add_3d_button([0.70, y, 0.065, h], 'Z-Stack', 'mediumpurple')
        self.btn_zstack.on_clicked(self._btn_zstack_clicked)
        
        self.btn_locate = add_3d_button([0.77, y, 0.065, h], 'Locate', 'orange')
        self.btn_locate.on_clicked(self._btn_locate_clicked)

        # 4. Save and Exit (Right)
        self.btn_save = add_3d_button([0.84, y, 0.055, h], 'SAVE', 'gold')
        self.btn_save.on_clicked(self._save_to_csv)

        self.btn_exit = add_3d_button([0.90, y, 0.055, h], 'EXIT', 'salmon')
        self.btn_exit.on_clicked(self._btn_exit_clicked)

        # Add kitvalley.png logo to the bottom right of the screen
        if os.path.exists(icon_path):
            try:
                # [left, bottom, width, height]
                ax_logo = self.fig.add_axes([0.88, 0.015, 0.1, 0.07], anchor='SE', zorder=10)
                img = plt.imread(icon_path)
                ax_logo.imshow(img)
                ax_logo.axis('off')
            except Exception as e:
                print(f"[*] Failed to display screen logo: {e}")

        self.fig.canvas.mpl_connect('key_press_event', self.on_key)
        self.fig.canvas.mpl_connect('button_press_event', self.on_click)
        self.fig.canvas.mpl_connect('motion_notify_event', self.on_motion)
        self.fig.canvas.mpl_connect('button_release_event', self.on_release)
        self.fig.canvas.mpl_connect('scroll_event', self.on_scroll)

    def _btn_model_clicked(self, event):
        current_idx = self.available_models.index(self.ai_model.lower())
        next_idx = (current_idx + 1) % len(self.available_models)
        self.ai_model = self.available_models[next_idx]
        self.btn_model.label.set_text(self.ai_model.upper())
        print(f"[*] AI model changed to '{self.ai_model}'.")
        self.fig.canvas.draw_idle()

    def _btn_weight_clicked(self, event):
        current_idx = self.available_weights.index(self.pretrained.lower()) if self.pretrained.lower() in self.available_weights else 0
        next_idx = (current_idx + 1) % len(self.available_weights)
        self.pretrained = self.available_weights[next_idx]
        self.btn_weight.label.set_text(f'W: {self.pretrained.upper()}')
        print(f"[*] AI weights changed to '{self.pretrained}'.")
        self.fig.canvas.draw_idle()

    def _btn_exit_clicked(self, event):
        print("\n[*] Exit button clicked. Exiting.")
        plt.close(self.fig)

    def _sync_resp_button(self):
        """Synchronize button UI according to the Resp status of the current station."""
        sta = self.station_names[self.current_idx]
        is_resp = self.resp_status.get(sta, False)
        if is_resp:
            self.btn_resp.label.set_text('Resp: OFF')
            self.btn_resp.ax.set_facecolor('lightblue')
        else:
            self.btn_resp.label.set_text('Resp: ON')
            self.btn_resp.ax.set_facecolor('white')
        self.fig.canvas.draw_idle()

    def _btn_prev_clicked(self, event):
        if self.current_idx > 0:
            self.current_idx -= 1
            sta = self.station_names[self.current_idx]
            if sta not in self.stations:
                self._process_station(sta)
            self._sync_resp_button()
            self._draw_current_station()

    def _btn_next_clicked(self, event):
        if self.current_idx < len(self.station_names) - 1:
            self.current_idx += 1
            sta = self.station_names[self.current_idx]
            if sta not in self.stations:
                self._process_station(sta)
            self._sync_resp_button()
            self._draw_current_station()

    def _btn_filter_clicked(self, event):
        self.apply_filter = not self.apply_filter
        if self.apply_filter:
            self.btn_filter.label.set_text('Filter: ON')
            self.btn_filter.ax.set_facecolor('lightgreen')
        else:
            self.btn_filter.label.set_text('Filter: OFF')
            self.btn_filter.ax.set_facecolor('white')
            
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()
        
        # Filter is a global setting, so re-process current station
        sta = self.station_names[self.current_idx]
        self._process_station(sta)
        self._draw_current_station()

    def _btn_resp_clicked(self, event):
        if not self.inv:
            print("[-] Cannot remove instrument response: No Inventory data. (--inv option required)")
            return
            
        sta = self.station_names[self.current_idx]
        # Toggle current station status
        self.resp_status[sta] = not self.resp_status[sta]
        
        is_resp = self.resp_status[sta]
        if is_resp:
            self.btn_resp.label.set_text('Resp: OFF')
            self.btn_resp.ax.set_facecolor('lightblue')
            self.ax_z.set_title(f" {sta} Removing instrument response... Please wait ", color='blue', fontweight='bold')
        else:
            self.btn_resp.label.set_text('Resp: ON')
            self.btn_resp.ax.set_facecolor('white')
            self.ax_z.set_title(f" {sta} Restoring original data... ", color='blue', fontweight='bold')
            
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()
        
        self._process_station(sta)
        self._draw_current_station()

    def _btn_clear_clicked(self, event):
        sta = self.station_names[self.current_idx]
        self.picks_db[sta] = {'p': {'mpl_num': None, 'conf': 0.0}, 's': {'mpl_num': None, 'conf': 0.0}}
        # Also delete probability curves for the station
        if sta in self.ai_annotations: del self.ai_annotations[sta]
        self._draw_current_station()

    def _btn_zstack_clicked(self, event):
        # 1. Collect stations with P-wave picks
        picked_stas = []
        for sta in self.station_names:
            p_num = self.picks_db[sta]['p']['mpl_num']
            if p_num is not None:
                picked_stas.append((sta, p_num))
        
        if not picked_stas:
            print("[-] No P-wave pick information. Please pick at least one P-wave first.")
            return
            
        # 2. Sort by P-wave arrival time (earliest first)
        picked_stas.sort(key=lambda x: x[1])
        first_p_time = picked_stas[0][1] # Very first P-wave time (earliest time)
        
        # Calculate allowed drawing range (10s before first P-wave ~ 180s after)
        valid_start_time = first_p_time - (10 / 86400.0)
        valid_end_time = valid_start_time + (180 / 86400.0)
        
        # 3. Create new Figure - secure left margin for scrollbar
        self.fig_stack, self.ax_stack = plt.subplots(figsize=(10, 8))
        try:
            self.fig_stack.canvas.manager.set_window_title('Z-Component Record Section')
        except:
            pass
            
        # Increase left margin to 0.15 to secure scrollbar space
        plt.subplots_adjust(left=0.15, right=0.95, top=0.9, bottom=0.1)
        
        yticks = []
        yticklabels = []
        
        # Z-Stack specific state variable storage for drag and drop
        self.zstack_lines = {} # { (sta, phase): Line2D_object }
        self.zstack_dragging = None # Currently dragged line info (sta, phase)
        self.zstack_y_to_sta = {} # { y_pos: sta }
        self.zstack_sta_to_y = {} # { sta: y_pos }
        
        print(f"[*] Z-Stack: Sorting complete for {len(picked_stas)} stations.")
        
        # 4. Draw Z-component waveforms in sorted station order (stacked without gaps)
        total_picked = len(picked_stas)
        all_pick_times = []
        for sta, p_num in picked_stas:
            # Check if both P and S waves are within range, then include in drawing targets
            if valid_start_time <= p_num <= valid_end_time:
                all_pick_times.append(p_num)
            
            s_num = self.picks_db[sta]['s']['mpl_num']
            if s_num is not None and valid_start_time <= s_num <= valid_end_time:
                all_pick_times.append(s_num)
                
        for i, (sta, p_num) in enumerate(picked_stas):
            if sta not in self.stations:
                self._process_station(sta)
            
            tr = self.stations[sta]['Z']
            if tr is None:
                continue
                
            # Vectorized operations for time precision fix and speed improvement
            t0 = mdates.date2num(tr.stats.starttime.datetime.replace(tzinfo=None))
            num_times = t0 + tr.times() / 86400.0
            
            # Waveform normalization: limit max amplitude to 0.5 so adjacent stations touch (-0.5 ~ 0.5)
            data = tr.data
            max_val = np.max(np.abs(data))
            if max_val > 0:
                norm_data = (data / max_val) * 0.5
            else:
                norm_data = data
                
            # Y-axis index: reverse order so the earliest (0th) station is at the top (largest Y value)
            y_pos = total_picked - 1 - i  
            self.zstack_y_to_sta[y_pos] = sta
            self.zstack_sta_to_y[sta] = y_pos
            
            # Draw waveform
            self.ax_stack.plot(num_times, norm_data + y_pos, 'k-', lw=0.5, picker=False)
            
            # Draw P-wave marker (only if within range)
            if valid_start_time <= p_num <= valid_end_time:
                line_p, = self.ax_stack.plot([p_num, p_num], [y_pos - 0.5, y_pos + 0.5], 'r-', lw=2, alpha=0.8, picker=True, pickradius=5)
                self.zstack_lines[(sta, 'p')] = line_p
            
            # Draw S-wave marker (only if within range)
            s_num = self.picks_db[sta]['s']['mpl_num']
            if s_num is not None and valid_start_time <= s_num <= valid_end_time:
                line_s, = self.ax_stack.plot([s_num, s_num], [y_pos - 0.5, y_pos + 0.5], 'b-', lw=2, alpha=0.8, picker=True, pickradius=5)
                self.zstack_lines[(sta, 's')] = line_s
            
            yticks.append(y_pos)
            yticklabels.append(sta)
            
        self.ax_stack.set_yticks(yticks)
        self.ax_stack.set_yticklabels(yticklabels)
        
        # Y-axis viewport fixing: show only 8 waveforms on one screen
        view_size = 8.0
        top_y_limit = total_picked - 0.5
        bottom_y_limit = -0.5
        
        # Initial view: earliest station is at the top (top_y_limit)
        current_top = top_y_limit
        current_bottom = max(bottom_y_limit, current_top - view_size)
        self.ax_stack.set_ylim(current_bottom, current_top)
        
        # --- Right scrollbar (Slider) addition logic ---
        if total_picked > 8:
            from matplotlib.widgets import Slider
            
            # Scrollbar positioning: vertical on the right side of ax_stack (moved from 0.02 to around 0.96)
            plt.subplots_adjust(left=0.08, right=0.92, top=0.9, bottom=0.1)
            ax_scroll = self.fig_stack.add_axes([0.94, 0.1, 0.02, 0.8], facecolor='lightgoldenrodyellow')
            
            # Create slider (min: bottom, max: top). Initial value is top.
            # Lowering the scroll handle (decreasing value) shifts the view down to show lower waveforms.
            max_scroll = top_y_limit
            min_scroll = bottom_y_limit + view_size
            
            self.zstack_slider = Slider(
                ax=ax_scroll,
                label='',
                valmin=min_scroll,
                valmax=max_scroll,
                valinit=max_scroll,
                valstep=0.1,
                orientation='vertical',
                color='gray'
            )
            
            # Change Y-axis range according to slider movement (drag)
            def update_scroll(val):
                new_top = val
                new_bottom = new_top - view_size
                self.ax_stack.set_ylim(new_bottom, new_top)
                self.fig_stack.canvas.draw_idle()
                
            self.zstack_slider.on_changed(update_scroll)

        # --- X-axis auto-scaling logic ---
        if all_pick_times:
            # Gather only P-waves (red lines) to calculate range
            p_pick_times = [p_num for sta, p_num in picked_stas]
            if p_pick_times:
                min_x = min(p_pick_times)
                max_x = max(p_pick_times)
                
                # 10s before first P-wave arrival, 60s after last P-wave arrival
                margin_left = 10 / 86400.0  # 10초
                margin_right = 60 / 86400.0 # 60초
                
                self.ax_stack.set_xlim(min_x - margin_left, max_x + margin_right)

        locator = mdates.AutoDateLocator(maxticks=10)
        formatter = mdates.AutoDateFormatter(locator)
        self.ax_stack.xaxis.set_major_locator(locator)
        self.ax_stack.xaxis.set_major_formatter(formatter)
        
        plt.setp(self.ax_stack.get_xticklabels(), rotation=30, ha='right')
        self.ax_stack.set_title("Z-Component Record Section (Drag Picks | Scroll Left Bar | Scroll Wheel to Zoom X)")
        self.ax_stack.grid(True, linestyle=':', alpha=0.5)
        
        # --- Z-Stack specific event handler bindings ---
        self.fig_stack.canvas.mpl_connect('scroll_event', self._on_zstack_scroll)
        self.fig_stack.canvas.mpl_connect('button_press_event', self._on_zstack_press)
        self.fig_stack.canvas.mpl_connect('motion_notify_event', self._on_zstack_motion)
        self.fig_stack.canvas.mpl_connect('button_release_event', self._on_zstack_release)
        self.fig_stack.canvas.mpl_connect('key_press_event', self._on_zstack_key)
        
        # Add instruction text at the bottom
        self.fig_stack.text(0.5, 0.02, "Keys: [p] Pick P-wave | [s] Pick S-wave | [c] Clear Picks | Mouse Wheel: Zoom X", 
                            ha='center', fontsize=9, color='gray', fontstyle='italic')
        
        self.fig_stack.show()

    # --- Z-Stack specific event handlers ---
    def _on_zstack_key(self, event):
        if event.inaxes != self.ax_stack: return
        if event.ydata is None or event.xdata is None: return
        
        y_idx = int(round(event.ydata))
        if y_idx not in self.zstack_y_to_sta: return
        
        sta = self.zstack_y_to_sta[y_idx]
        
        if event.key == 'p':
            self._update_zstack_line(sta, 'p', event.xdata)
        elif event.key == 's':
            self._update_zstack_line(sta, 's', event.xdata)
        elif event.key == 'c':
            self._clear_zstack_lines(sta)
            
        self.fig_stack.canvas.draw_idle()
        
        # Sync with main window if the modified station is currently displayed
        if hasattr(self, 'fig') and plt.fignum_exists(self.fig.number):
            if self.station_names[self.current_idx] == sta:
                if event.key == 'c':
                    self._draw_current_station()
                else:
                    self._draw_lines(event.key, event.xdata)
                    self.fig.canvas.draw_idle()

    def _update_zstack_line(self, sta, phase, x_num):
        y_pos = self.zstack_sta_to_y[sta]
        self.picks_db[sta][phase] = {'mpl_num': x_num, 'conf': 1.0}
        color = 'r-' if phase == 'p' else 'b-'
        
        if (sta, phase) in self.zstack_lines and self.zstack_lines[(sta, phase)] is not None:
            # Update existing
            line = self.zstack_lines[(sta, phase)]
            line.set_data([x_num, x_num], [y_pos - 0.5, y_pos + 0.5])
        else:
            # Create new
            line, = self.ax_stack.plot([x_num, x_num], [y_pos - 0.5, y_pos + 0.5], color, lw=2, alpha=0.8, picker=True, pickradius=5)
            self.zstack_lines[(sta, phase)] = line

    def _clear_zstack_lines(self, sta):
        self.picks_db[sta]['p'] = {'mpl_num': None, 'conf': 0.0}
        self.picks_db[sta]['s'] = {'mpl_num': None, 'conf': 0.0}
        for phase in ['p', 's']:
            if (sta, phase) in self.zstack_lines and self.zstack_lines[(sta, phase)] is not None:
                self.zstack_lines[(sta, phase)].remove()
                self.zstack_lines[(sta, phase)] = None

    def _on_zstack_scroll(self, event):
        if event.inaxes != self.ax_stack: return
        
        # Mouse wheel scroll only handles X-axis (time) zoom in/out
        scale = 1/1.5 if event.button == 'up' else 1.5
        cur_xlim = self.ax_stack.get_xlim()
        new_width = (cur_xlim[1] - cur_xlim[0]) * scale
        rel_pos = (cur_xlim[1] - event.xdata) / (cur_xlim[1] - cur_xlim[0])
        self.ax_stack.set_xlim([event.xdata - new_width * (1 - rel_pos), event.xdata + new_width * rel_pos])
        
        self.fig_stack.canvas.draw_idle()

    def _on_zstack_press(self, event):
        if event.button != 1 or event.inaxes is None: return
        # Find the clicked line
        for (sta, phase), line in self.zstack_lines.items():
            contains, _ = line.contains(event)
            if contains:
                self.zstack_dragging = (sta, phase)
                return

    def _on_zstack_motion(self, event):
        if self.zstack_dragging and event.inaxes:
            sta, phase = self.zstack_dragging
            line = self.zstack_lines[(sta, phase)]
            # Update line position (X-axis movement)
            y_data = line.get_ydata()
            line.set_data([event.xdata, event.xdata], y_data)
            
            # Instantly update pick values in the main DB
            self.picks_db[sta][phase]['mpl_num'] = event.xdata
            self.fig_stack.canvas.draw_idle()
            
            # --- Main window (fig) synchronization ---
            if hasattr(self, 'fig') and plt.fignum_exists(self.fig.number):
                # If the station displayed on the main window is the one currently being dragged
                if self.station_names[self.current_idx] == sta:
                    # Instantly move vertical lines on the 3 axes (Z, N, E) of the main window
                    if phase in self.lines:
                        for main_line in self.lines[phase]:
                            main_line.set_xdata([event.xdata, event.xdata])
                    self._update_title_only()
                    self.fig.canvas.draw_idle()

    def _on_zstack_release(self, event):
        self.zstack_dragging = None

    def _btn_ai_clicked(self, event):
        sta = self.station_names[self.current_idx]
        self._run_ai_picker(self.ai_model, station_name=sta)
        self._draw_current_station()

    def _draw_current_station(self):
        # Clear both main and secondary axes
        for ax in [self.ax_z, self.ax_n, self.ax_e, self.ax_z_prob, self.ax_n_prob, self.ax_e_prob]: 
            ax.clear()
        
        self.fig.suptitle("") # Clear location result when switching station
        
        for ax in [self.ax_z_prob, self.ax_n_prob, self.ax_e_prob]:
            ax.set_ylim(0, 0.5) 
            ax.set_yticks([])
            
        self.lines = {'p': [], 's': []}
        
        sta = self.station_names[self.current_idx]
        comps = self.stations[sta]
        min_mpl, max_mpl = float('inf'), float('-inf')

        for comp_key, ax in zip(['Z', 'N', 'E'], [self.ax_z, self.ax_n, self.ax_e]):
            tr = comps[comp_key]
            if tr:
                # Vectorized time calculation
                t0 = mdates.date2num(tr.stats.starttime.datetime.replace(tzinfo=None))
                num_times = t0 + tr.times() / 86400.0
                ax.plot(num_times, tr.data, 'k-', linewidth=0.5)
                
                # Remove unit display, show only channel name
                ax.set_ylabel(f"{tr.stats.channel}", fontsize=9)
                
                t_num_start = num_times[0]
                t_num_end = num_times[-1]
                min_mpl, max_mpl = min(min_mpl, t_num_start), max(max_mpl, t_num_end)
            
            # Auto-rescale Y-axis (essential for Count <-> Velocity conversion)
            ax.relim()
            ax.autoscale_view(scalex=False, scaley=True)
            ax.grid(True, linestyle=':', alpha=0.5)

        # Draw AI probability curves
        if sta in self.ai_annotations:
            ann_st = self.ai_annotations[sta]
            for tr in ann_st:
                ch = tr.stats.channel.upper()
                color = 'red' if ch.endswith('P') else 'blue' if ch.endswith('S') else None
                if color:
                    t0 = mdates.date2num(tr.stats.starttime.datetime.replace(tzinfo=None))
                    num_times = t0 + tr.times() / 86400.0
                    for prob_ax in [self.ax_z_prob, self.ax_n_prob, self.ax_e_prob]:
                        prob_ax.fill_between(num_times, 0, tr.data, color=color, alpha=0.15)
                        prob_ax.plot(num_times, tr.data, color=color, alpha=0.4, linewidth=0.8)

        # Draw picking lines
        for phase in ['p', 's']:
            pick_data = self.picks_db[sta][phase]
            if pick_data['mpl_num'] is not None:
                self._draw_lines(phase, pick_data['mpl_num'], update_db=False)

        if min_mpl != float('inf'):
            margin = (max_mpl - min_mpl) * 0.05
            if margin == 0: margin = 1 / 86400
            self.ax_e.set_xlim([min_mpl - margin, max_mpl + margin])
            
            locator = mdates.AutoDateLocator(maxticks=10)
            formatter = mdates.AutoDateFormatter(locator)
            self.ax_e.xaxis.set_major_locator(locator)
            self.ax_e.xaxis.set_major_formatter(formatter)

        self._update_title_only()
        self.fig.canvas.draw()

    def _draw_lines(self, phase, x_num, update_db=True):
        for line in self.lines[phase]:
            try: line.remove()
            except: pass
        
        color = 'red' if phase == 'p' else 'blue'
        l1 = self.ax_z.axvline(x_num, color=color, ls='-', lw=2, alpha=0.8)
        l2 = self.ax_n.axvline(x_num, color=color, ls='--', lw=2, alpha=0.8)
        l3 = self.ax_e.axvline(x_num, color=color, ls='--', lw=2, alpha=0.8)
        self.lines[phase] = [l1, l2, l3]

        if update_db:
            sta = self.station_names[self.current_idx]
            self.picks_db[sta][phase] = {'mpl_num': x_num, 'conf': 1.0}
            self._update_title_only()

    def _update_title_only(self):
        sta = self.station_names[self.current_idx]
        def get_info(ph):
            d = self.picks_db[sta][ph]
            if d['mpl_num'] is not None:
                dt_obj = mdates.num2date(d['mpl_num'])
                return f"{ph.upper()}: {dt_obj.strftime('%H:%M:%S.%f')[:-3]}"
            return f"{ph.upper()}: None"
        
        title = f"[{self.current_idx+1}/{len(self.station_names)}] {sta} | {get_info('p')} | {get_info('s')}"
        self.ax_z.set_title(title, loc='left', fontsize=10)

    def on_scroll(self, event):
        if event.inaxes is None: return
        scale = 1/1.5 if event.button == 'up' else 1.5
        ax = event.inaxes
        cur_xlim = ax.get_xlim()
        new_width = (cur_xlim[1] - cur_xlim[0]) * scale
        rel_pos = (cur_xlim[1] - event.xdata) / (cur_xlim[1] - cur_xlim[0])
        self.ax_e.set_xlim([event.xdata - new_width * (1 - rel_pos), event.xdata + new_width * rel_pos])
        self.fig.canvas.draw_idle()

    def on_key(self, event):
        if event.key in ['right', 'n', 'pagedown']: self._btn_next_clicked(None)
        elif event.key in ['left', 'b', 'pageup']: self._btn_prev_clicked(None)
        elif event.inaxes and event.key in ['p', 's']:
            self._draw_lines(event.key, event.xdata)
            self.fig.canvas.draw()
        elif event.key == 'c': self._btn_clear_clicked(None)

    def on_click(self, event):
        if event.button == 1 and event.inaxes and self.fig.canvas.toolbar.mode == '':
            for phase in ['p', 's']:
                for line in self.lines[phase]:
                    if line.contains(event)[0]:
                        self.dragging = phase
                        return

    def on_motion(self, event):
        if self.dragging and event.inaxes:
            for line in self.lines[self.dragging]: line.set_xdata([event.xdata, event.xdata])
            sta = self.station_names[self.current_idx]
            self.picks_db[sta][self.dragging]['mpl_num'] = event.xdata
            self._update_title_only()
            self.fig.canvas.draw_idle()

    def on_release(self, event): self.dragging = None

    def _btn_locate_clicked(self, event):
        if not self.inv:
            print("[-] Cannot locate without inventory (--inv) to provide station coordinates.")
            self.ax_z.set_title(" Cannot locate: No Inventory Data (--inv) ", color='red', fontweight='bold')
            self.fig.canvas.draw_idle()
            return

        try:
            from seismo_analyzer import locate_hypocenter, calculate_magnitude, plot_map
        except ImportError:
            print("[-] seismo_analyzer.py not found in the current directory.")
            return

        # Prepare station coordinates mapping
        station_coords = {}
        for net in self.inv:
            for sta_obj in net:
                net_sta = f"{net.code}.{sta_obj.code}"
                station_coords[net_sta] = (sta_obj.latitude, sta_obj.longitude, sta_obj.elevation)

        # Get reference time (earliest P-pick)
        valid_p_picks = [p['p']['mpl_num'] for p in self.picks_db.values() if p['p']['mpl_num'] is not None]
        if not valid_p_picks:
            print("[-] No P-picks available for location.")
            if hasattr(self, 'fig'):
                self.ax_z.set_title(" Cannot locate: No P-picks available ", color='red', fontweight='bold')
                self.fig.canvas.draw_idle()
            return
        ref_time = UTCDateTime(mdates.num2date(min(valid_p_picks)).replace(tzinfo=None))

        print("[*] Calculating hypocenter...")
        if hasattr(self, 'fig'):
            self.fig.suptitle(" Calculating Hypocenter... Please wait ", color='blue', fontweight='bold', fontsize=12, y=0.88)
            self.fig.canvas.draw_idle()
            self.fig.canvas.flush_events()

        try:
            # 1. Locate
            est_lat, est_lon, est_depth, origin_time, used_stations, used_arrivals = locate_hypocenter(
                self.picks_db, station_coords, ref_time
            )

            # 2. Magnitude
            total_ml, ml_results, station_ml_details = calculate_magnitude(
                est_lat, est_lon, origin_time, station_coords, self.stations_raw, self.inv
            )

            # --- Formatting and Reporting ---
            ml_str = f" | ML: {total_ml:.2f}" if total_ml is not None else ""
            res_str = f"--- Location Result ---\nOrigin Time: {origin_time}\nLatitude: {est_lat:.4f}\nLongitude: {est_lon:.4f}\nDepth: {est_depth:.2f} km\n"
            if total_ml is not None:
                res_str += f"Magnitude: ML {total_ml:.2f} ({len(ml_results)} channels)\n"
            else:
                res_str += "Magnitude: ML N/A\n"
            res_str += f"Used Stations: {len(used_stations)}"

            print(f"\n{res_str}\n")

            # Save Report
            report_path = os.path.join(self.output_dir, "location_report.txt")
            with open(report_path, 'w', encoding='utf-8') as f:
                f.write(res_str + "\n\n")
                f.write("=== Used Stations for Location ===\n")
                for i, s in enumerate(used_stations):
                    t_arr = ref_time + used_arrivals[i]
                    f.write(f"{s[0]:<12} | Lat: {s[1]:8.4f} | Lon: {s[2]:9.4f} | P-Arr: {t_arr}\n")
                
                if station_ml_details:
                    f.write("\n=== Station Magnitudes ===\n")
                    f.write(f"{'Station.Ch':<12} | {'Dist(km)':<8} | {'Amp(mm)':<10} | {'ML':<5}\n")
                    f.write("-" * 50 + "\n")
                    for d in sorted(station_ml_details, key=lambda x: x['dist_km']):
                        f.write(f"{d['station']+'.'+d['comp']:<12} | {d['dist_km']:>8.1f} | {d['max_amp_mm']:>10.4e} | {d['ml']:>5.2f}\n")
            
            print(f"[*] Location report saved to: {report_path}")

            # Save Station Magnitudes to CSV
            if station_ml_details:
                ml_csv_path = os.path.join(self.output_dir, "station_magnitudes.csv")
                try:
                    with open(ml_csv_path, 'w', newline='', encoding='utf-8') as f:
                        writer = csv.writer(f)
                        writer.writerow(['Station', 'Component', 'Distance_km', 'Max_Amp_mm', 'ML'])
                        for d in sorted(station_ml_details, key=lambda x: x['dist_km']):
                            writer.writerow([d['station'], d['comp'], f"{d['dist_km']:.1f}", f"{d['max_amp_mm']:.6e}", f"{d['ml']:.2f}"])
                    print(f"[*] Station magnitudes saved to: {ml_csv_path}")
                except Exception as e:
                    print(f"[-] Failed to save station magnitudes CSV: {e}")

            # 3. Plot Map
            print("[*] Generating location map...")
            map_path = plot_map(est_lat, est_lon, used_stations, origin_time=origin_time, magnitude=total_ml, output_dir=self.output_dir, show_plot=not self.batch_mode)
            if map_path:
                res_str += f"\n\nMap saved to:\n{map_path}"

            # Update UI Result (Top position) - Only if not batch mode
            if not self.batch_mode and hasattr(self, 'fig'):
                title_res = f"Loc: {origin_time.strftime('%H:%M:%S.%f')[:-3]} | {est_lat:.4f}N, {est_lon:.4f}E | Dep: {est_depth:.1f}km{ml_str}"
                self.fig.suptitle(title_res, color='green', fontweight='bold', fontsize=12, y=0.88)
                self.fig.canvas.draw_idle()

                import tkinter as tk
                from tkinter import messagebox
                root = tk.Tk(); root.withdraw(); root.attributes('-topmost', True)
                messagebox.showinfo("Hypocenter Location Result", res_str, parent=root)
                root.destroy()

        except Exception as e:
            print(f"[-] Analysis failed: {e}")
            if hasattr(self, 'fig'):
                self.fig.suptitle(f" Analysis Failed: {e} ", color='red', fontweight='bold', fontsize=12, y=0.88)
                self.fig.canvas.draw_idle()

    def _save_to_csv(self, event=None):
        out = os.path.join(self.output_dir, "refined_picks.csv")
        try:
            with open(out, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(['Network_Station', 'Phase', 'Arrival_Time', 'Confidence'])
                for sta in self.station_names:
                    for ph in ['p', 's']:
                        d = self.picks_db[sta][ph]
                        if d['mpl_num'] is not None:
                            t = mdates.num2date(d['mpl_num'])
                            time_str = t.strftime('%Y-%m-%dT%H:%M:%S.%f') + 'Z'
                            conf_str = f"{float(d['conf']):.4f}"
                            writer.writerow([sta, ph.upper(), time_str, conf_str])
            print(f"[*] Results saved: {out}")
            if event and hasattr(self, 'ax_z'):
                self.ax_z.set_title("[SAVED SUCCESS!]", color='red', fontweight='bold')
                self.fig.canvas.draw()
        except Exception as e:
            print(f"[-] Failed to save: {e}")

    def _generate_batch_waveform_plots(self):
        """Generate static Z-stack waveform plots (15 stations per page) in batch mode."""
        print("[*] Generating batch waveform plots...")
        picked_stas = []
        for sta in self.station_names:
            p_num = self.picks_db[sta]['p']['mpl_num']
            if p_num is not None:
                picked_stas.append((sta, p_num))
        
        if not picked_stas:
            print("[-] No P-wave picks available to plot.")
            return
            
        picked_stas.sort(key=lambda x: x[1])
        first_p_time = picked_stas[0][1]
        
        # Extend X-axis to 150 seconds (starting 10s before first P-wave)
        xlim_min = first_p_time - (10 / 86400.0)
        xlim_max = xlim_min + (150 / 86400.0)

        batch_size = 15
        total_stations = len(picked_stas)
        num_pages = int(np.ceil(total_stations / batch_size))

        for page in range(num_pages):
            start_idx = page * batch_size
            end_idx = min((page + 1) * batch_size, total_stations)
            page_stas = picked_stas[start_idx:end_idx]
            
            fig, ax = plt.subplots(figsize=(10, 8))
            plt.subplots_adjust(left=0.15, right=0.95, top=0.9, bottom=0.1)
            
            yticks = []
            yticklabels = []
            
            n_stas_in_page = len(page_stas)
            for i, (sta, p_num) in enumerate(page_stas):
                if sta not in self.stations:
                    self._process_station(sta)
                
                tr = self.stations[sta]['Z']
                if tr is None: continue
                    
                t0 = mdates.date2num(tr.stats.starttime.datetime.replace(tzinfo=None))
                num_times = t0 + tr.times() / 86400.0
                
                data = tr.data
                max_val = np.max(np.abs(data))
                norm_data = (data / max_val) * 0.5 if max_val > 0 else data
                
                # Reverse order: earliest at top
                y_pos = n_stas_in_page - 1 - i  
                
                ax.plot(num_times, norm_data + y_pos, 'k-', lw=0.5)
                
                # Plot P pick
                ax.plot([p_num, p_num], [y_pos - 0.5, y_pos + 0.5], 'r-', lw=2, alpha=0.8)
                
                # Plot S pick if available
                s_num = self.picks_db[sta]['s']['mpl_num']
                if s_num is not None:
                    ax.plot([s_num, s_num], [y_pos - 0.5, y_pos + 0.5], 'b-', lw=2, alpha=0.8)
                
                yticks.append(y_pos)
                yticklabels.append(sta)
                
            ax.set_yticks(yticks)
            ax.set_yticklabels(yticklabels)
            ax.set_ylim(-0.5, n_stas_in_page - 0.5)
            ax.set_xlim(xlim_min, xlim_max)
            
            locator = mdates.AutoDateLocator(maxticks=10)
            formatter = mdates.AutoDateFormatter(locator)
            ax.xaxis.set_major_locator(locator)
            ax.xaxis.set_major_formatter(formatter)
            plt.setp(ax.get_xticklabels(), rotation=30, ha='right')
            
            ax.set_title(f"Z-Component Record Section (Page {page+1}/{num_pages})")
            ax.grid(True, linestyle=':', alpha=0.5)
            
            # Save figure
            out_filename = f"record_section_page_{page+1:02d}.png"
            out_path = os.path.join(self.output_dir, out_filename)
            plt.savefig(out_path, dpi=300, bbox_inches='tight')
            plt.close(fig)
            print(f"[*] Saved waveform plot: {out_path}")

    def show(self):
        plt.show()
        self._save_to_csv()

def main():
    parser = argparse.ArgumentParser(description="Interactive Multi-Station Earthquake Phase Picker")
    parser.add_argument("--mseed", required=True, help="Path to the MiniSEED data file to analyze")
    parser.add_argument("--picks", help="Existing picking data CSV file (leave empty if none)")
    parser.add_argument("--filter", action="store_true", help="Apply default Bandpass Filter (1-10Hz)")
    parser.add_argument("--model", choices=["eqtransformer", "phasenet"], help="Automatic picking using SeisBench deep learning models (optional)")
    parser.add_argument("--pretrained", default="original", help="SeisBench pre-trained weights (e.g., original, stead, ethz, etc.)")
    parser.add_argument("--inv", default="total_inv.xml", help="Path to Inventory XML file (for instrument response removal)")
    parser.add_argument("--output", default=None, help="Output directory for results (default: mseed_dir + '_out')")
    parser.add_argument("--batch", action="store_true", help="Run in headless batch mode: auto-pick -> locate -> save -> exit without opening GUI")
    args = parser.parse_args()
    
    # Auto-generate output directory if not provided
    if args.output is None:
        mseed_abs = os.path.abspath(args.mseed)
        mseed_dir = os.path.dirname(mseed_abs)
        mseed_name = os.path.basename(mseed_abs)
        
        if '*' in mseed_name or '?' in mseed_name:
            # If wildcard used, append _out to the parent directory
            # Example: C:\data\20260311\* -> C:\data\20260311_out
            args.output = mseed_dir + "_out"
        else:
            # If specific file, create subfolder in same directory
            # Example: C:\data\event.mseed -> C:\data\event_out
            base_name = os.path.splitext(mseed_name)[0]
            args.output = os.path.join(mseed_dir, f"{base_name}_out")
            
        print(f"[*] Output directory automatically set to: {args.output}")
    
    picker = MultiStationPicker(args.mseed, args.filter, args.picks, args.model, args.pretrained, args.inv, args.output, args.batch)
    if not args.batch and hasattr(picker, 'fig'): 
        picker.show()

if __name__ == "__main__":
    main()