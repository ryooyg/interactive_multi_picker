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

# 한글 폰트 설정 (Windows/Mac/Linux 대응)
import platform
if platform.system() == 'Windows':
    plt.rc('font', family='Malgun Gothic')
elif platform.system() == 'Darwin': # Mac
    plt.rc('font', family='AppleGothic')
else: # Linux
    plt.rc('font', family='NanumGothic')
plt.rc('axes', unicode_minus=False) # 마이너스 기호 깨짐 방지

class MultiStationPicker:
    def __init__(self, mseed_file, apply_filter=False, auto_picks_file=None, ai_model=None, pretrained="original", inv_file=None):
        # matplotlib 기본 단축키 비활성화 (s: 저장 등 방지 - 창 생성 전 전역 설정)
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
                print(f"[*] Inventory 로드 중: {inv_file}")
                self.inv = read_inventory(inv_file)
            except Exception as e:
                print(f"[-] Inventory 로드 실패: {e}")
        
        # 1. 데이터 로드 (원본과 필터용 분리)
        self._load_data()
        
        # 초기에 모든 관측소의 그룹핑 생성
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
        
        # 개별 관측소별 처리된(계기응답/필터 반영) 데이터를 저장할 딕셔너리
        self.stations = {}
        
        # 모든 관측소는 초기에는 계기응답 미적용(False) 상태로 시작
        self.resp_status = {sta: False for sta in self.station_names}
        self.current_idx = 0
        
        # 시작할 때 현재 화면의 관측소만 필터 처리를 진행
        if self.station_names:
            self._process_station(self.station_names[self.current_idx])
        
        # 2. 피킹 데이터베이스 초기화
        self.picks_db = {sta: {'p': {'mpl_num': None, 'conf': 0.0}, 
                               's': {'mpl_num': None, 'conf': 0.0}} for sta in self.station_names}
        self.lines = {'p': [], 's': []}
        self.dragging = None
        self.ai_annotations = {} # 확률 곡선 저장소
        
        # 3. 화면 UI 및 그래프 초기화
        if not self.station_names:
            print("[-] 표시할 관측소 데이터가 없습니다.")
            return
            
        self._setup_plot()

        # 4. 자동 피킹 로드 또는 AI 피킹 수행 (UI가 준비된 후 실행)
        if self.auto_picks_file:
            self._load_auto_picks()
        elif ai_model: # 명령줄 인수로 모델이 지정된 경우 시작 시 자동 실행
            self._run_ai_picker(self.ai_model)
            
        # 결과 반영하여 첫 화면 그리기 (반드시 4번 이후에 실행)
        self._draw_current_station()

    def _load_data(self):
        try:
            self.st_raw = read(self.mseed_file)
            self.st_raw.detrend('linear')
            print(f"[*] 데이터 로드 완료: {len(self.st_raw)} 트레이스")
        except Exception as e:
            print(f"[-] 데이터 로드 실패: {e}")
            raise

    # 이전의 전역적인 _apply_processing 제거, 대신 개별 관측소 처리 함수 추가
    def _process_station(self, sta):
        """특정 관측소(sta)의 데이터만 뽑아서 Resp/Filter 상태에 맞게 가공합니다."""
        comps_raw = self.stations_raw[sta]
        
        # 원본 트레이스들을 모아 새로운 스트림 생성
        st_sta = Stream()
        for comp_key in ['Z', 'N', 'E']:
            if comps_raw[comp_key]:
                st_sta.append(comps_raw[comp_key].copy())
                
        # 계기응답 제거 (현재 관측소 상태 확인)
        if self.resp_status[sta] and self.inv:
            try:
                print(f"[*] {sta} 관측소: 계기응답(Response) 제거 중...")
                st_sta.remove_response(inventory=self.inv, output="VEL")
            except Exception as e:
                print(f"[-] {sta} 계기응답 제거 실패: {e}")
                self.resp_status[sta] = False
                
        # 필터 적용 (필터는 전체 토글 상태 따름)
        if self.apply_filter:
            st_sta.filter('bandpass', freqmin=1.0, freqmax=10.0)
            
        # 처리된 결과를 self.stations[sta]에 저장
        self.stations[sta] = {'Z': None, 'N': None, 'E': None}
        for tr in st_sta:
            comp = tr.stats.component
            if comp.endswith('Z'): self.stations[sta]['Z'] = tr
            elif comp.endswith('N') or comp.endswith('1'): self.stations[sta]['N'] = tr
            elif comp.endswith('E') or comp.endswith('2'): self.stations[sta]['E'] = tr

    def _run_ai_picker(self, model_name, station_name=None):
        target_info = f"관측소({station_name})" if station_name else "전체 데이터"
        print(f"\n[*] AI 모델({model_name}, weight={self.pretrained})로 {target_info} 분석 중...")
        
        # 상태 표시 업데이트
        if hasattr(self, 'ax_z'):
            self.ax_z.set_title(f" 분석 중... 잠시만 기다려주세요 ({target_info}) ", color='blue', fontweight='bold')
            self.fig.canvas.draw()

        try:
            import seisbench
            import seisbench.models as sbm
            import torch
        except ImportError:
            print("[-] seisbench가 설치되어 있지 않습니다. 'pip install seisbench torch'를 실행해주세요.")
            return

        try:
            if model_name.lower() == 'eqtransformer':
                model = sbm.EQTransformer.from_pretrained(self.pretrained)
            elif model_name.lower() == 'phasenet':
                model = sbm.PhaseNet.from_pretrained(self.pretrained)
            else:
                print(f"[-] 지원하지 않는 모델입니다: {model_name}")
                return

            if torch.cuda.is_available():
                model.cuda()
                print("[*] GPU 가속 활성화됨")

            # 대상 데이터 필터링 및 3성분 교집합 구간 추출
            if station_name:
                try:
                    net, stacode = station_name.split('.')
                    st_to_classify_raw = self.st_raw.select(network=net, station=stacode)
                except ValueError:
                    st_to_classify_raw = self.st_raw.select(station=station_name)
                    
                # 3성분이 모두 존재하는 공통 시간 구간(Overlapping window) 찾기
                if len(st_to_classify_raw) > 0:
                    start_times = [tr.stats.starttime for tr in st_to_classify_raw]
                    end_times = [tr.stats.endtime for tr in st_to_classify_raw]
                    
                    # 가장 늦은 시작 시간과 가장 빠른 종료 시간 찾기 (교집합)
                    common_start = max(start_times)
                    common_end = min(end_times)
                    
                    if common_start < common_end:
                        # 교집합 구간으로 자르기 (원본 데이터 보존을 위해 copy 사용)
                        st_to_classify = st_to_classify_raw.copy().trim(common_start, common_end)
                        print(f"[*] 3성분 공통 구간 추출: {common_start} ~ {common_end}")
                    else:
                        print(f"[-] {station_name} 관측소의 3성분 데이터가 겹치는 시간 구간이 없습니다.")
                        st_to_classify = Stream()
                else:
                    st_to_classify = Stream()
                    
                # 특정 관측소 분석 시 기존 확률 데이터 초기화
                self.ai_annotations[station_name] = Stream()
            else:
                # 전체 데이터를 분석할 때는 각 관측소별로 교집합 구간을 찾아야 함
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
                print(f"[-] 분석할 데이터를 찾을 수 없습니다: {station_name}")
                return

            # 1. 피킹(Classify) 수행
            classify_output = model.classify(st_to_classify)
            picks = classify_output.picks if hasattr(classify_output, 'picks') else classify_output
            
            # 2. 확률 곡선(Annotate) 수행
            try:
                annotations = model.annotate(st_to_classify)
                for tr in annotations:
                    net = tr.stats.network
                    sta_code = tr.stats.station
                    # 관측소 이름 매칭
                    sta_key = f"{net}.{sta_code}" if net else sta_code
                    if not sta_key and station_name: sta_key = station_name
                    
                    norm_key = self._normalize_name(sta_key)
                    target_sta = next((ls for ls in self.station_names if norm_key == self._normalize_name(ls)), None)

                    if target_sta:
                        if target_sta not in self.ai_annotations:
                            self.ai_annotations[target_sta] = Stream()
                        self.ai_annotations[target_sta].append(tr)
            except Exception as e:
                print(f"[-] 확률 곡선 생성 실패: {e}")
            
            # 3. 피킹 결과 반영
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
                            
                            # 기존 결과보다 높은 신뢰도이거나 처음인 경우만 업데이트
                            db_entry = self.picks_db[loaded_sta][phase]
                            if db_entry['mpl_num'] is None or conf > db_entry['conf']:
                                self.picks_db[loaded_sta][phase] = {'mpl_num': mpl_num, 'conf': conf}
                                count += 1
                        except:
                            pass
            print(f"[*] AI 피킹 완료: {count}개의 위상 픽 반영됨.\n")
        except Exception as e:
            print(f"[-] AI 피킹 중 오류 발생: {e}")

    def _normalize_name(self, name):
        return str(name).replace('_', '.').strip().lower()

    def _load_auto_picks(self):
        if not os.path.exists(self.auto_picks_file):
            print(f"[-] 파일을 찾을 수 없음: {self.auto_picks_file}")
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
                                # 수정: lazy load되는 self.stations 대신 항상 존재하는 self.stations_raw 참조
                                ref_tr = next((tr for tr in self.stations_raw[loaded_sta].values() if tr is not None), None)
                                if ref_tr:
                                    ref_date = ref_tr.stats.starttime
                                    # CSV의 연도/월/일이 실제 데이터의 연도/월/일과 다르면 데이터 기준으로 날짜를 덮어씌움 (시간은 유지)
                                    if parsed_dt.year != ref_date.year or parsed_dt.month != ref_date.month or parsed_dt.day != ref_date.day:
                                        fixed_time_str = f"{ref_date.year:04d}-{ref_date.month:02d}-{ref_date.day:02d}T{parsed_dt.hour:02d}:{parsed_dt.minute:02d}:{parsed_dt.second:02d}.{parsed_dt.microsecond:06d}Z"
                                        parsed_dt = UTCDateTime(fixed_time_str)
                                        
                                mpl_num = mdates.date2num(parsed_dt.datetime.replace(tzinfo=None))
                                self.picks_db[loaded_sta][phase] = {'mpl_num': mpl_num, 'conf': conf}
                                count += 1
                            except Exception as e:
                                print(f"[-] 시간 파싱 오류 ({time_str} / {loaded_sta}): {e}")
                        break
            print(f"[*] 자동 피킹 {count}개 로드 완료 (날짜 누락 자동 보정 적용)")
        except Exception as e:
            print(f"[-] 로드 중 오류: {e}")

    def _setup_plot(self):
        # matplotlib 기본 단축키 비활성화 (s: 저장 등 방지)
        plt.rcParams['keymap.save'] = ''
        plt.rcParams['keymap.fullscreen'] = ''
        plt.rcParams['keymap.home'] = ''
        plt.rcParams['keymap.back'] = ''
        plt.rcParams['keymap.forward'] = ''
        
        self.fig, (self.ax_z, self.ax_n, self.ax_e) = plt.subplots(3, 1, figsize=(12, 9), sharex=True)
        # 창 제목 및 아이콘 설정
        try:
            manager = self.fig.canvas.manager
            manager.set_window_title('interactive multi picker of Earthquake')
            
            # 아이콘 설정 로직 (강화된 방식 - Windows 전용 Tkinter 해킹 포함)
            icon_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'kitvalley.png')
            if os.path.exists(icon_path):
                import tkinter as tk
                if hasattr(manager, 'window'):
                    window = manager.window
                    try:
                        # 1. PIL(Pillow)를 사용해 PNG 읽기 (가장 안정적)
                        from PIL import Image, ImageTk
                        pil_img = Image.open(icon_path)
                        tk_img = ImageTk.PhotoImage(pil_img)
                        
                        # 2. 현재 창에 아이콘 적용
                        window.iconphoto(True, tk_img)
                        
                        # 3. 만약 숨겨진 최상위 루트 윈도우가 따로 있다면 거기도 강제 적용 (Windows 작업 표시줄 버그 해결용)
                        if window.master is not None:
                            window.master.iconphoto(True, tk_img)
                            
                        # 가비지 컬렉션 방지
                        window._tk_icon = tk_img 
                        
                    except Exception as e_pil:
                        print(f"[*] PIL 아이콘 적용 실패: {e_pil}")
                        try:
                            # 4. 최후의 수단: 기본 PhotoImage
                            img = tk.PhotoImage(file=icon_path)
                            window.iconphoto(True, img)
                            if window.master is not None:
                                window.master.iconphoto(True, img)
                            window._tk_icon_fallback = img
                        except Exception as e_tk:
                            pass
                            
        except Exception as e:
            print(f"[*] 알림: 창 아이콘 또는 제목 설정 중 문제 발생 ({e})")
            
        plt.subplots_adjust(left=0.08, right=0.95, top=0.85, bottom=0.1, hspace=0.18)
        
        self.formatter = mdates.DateFormatter('%H:%M:%S')
        self.ax_e.xaxis.set_major_formatter(self.formatter)
        plt.setp(self.ax_e.get_xticklabels(), rotation=30, ha='right')
        
        # 보조 축(TwinX) 생성 - 확률 곡선용
        self.ax_z_prob = self.ax_z.twinx()
        self.ax_n_prob = self.ax_n.twinx()
        self.ax_e_prob = self.ax_e.twinx()
        for ax in [self.ax_z_prob, self.ax_n_prob, self.ax_e_prob]:
            ax.set_ylim(0, 0.5) 
            ax.set_yticks([]) 
            ax.set_navigate(False)
        
        # --- 3D 버튼 생성 헬퍼 함수 ---
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

        # 1. Navigation (좌측)
        self.btn_prev = add_3d_button([0.01, y, 0.035, h], '<', 'lightgrey')
        self.btn_prev.on_clicked(self._btn_prev_clicked)
        
        self.btn_next = add_3d_button([0.05, y, 0.035, h], '>', 'lightgrey')
        self.btn_next.on_clicked(self._btn_next_clicked)

        # 2. 순환형 모델 & 가중치 선택 버튼
        self.available_models = ['eqtransformer', 'phasenet']
        self.available_weights = ['original', 'stead', 'ethz']
        
        self.btn_model = add_3d_button([0.095, y, 0.145, h], self.ai_model.upper(), 'lightgoldenrodyellow')
        self.btn_model.on_clicked(self._btn_model_clicked)

        self.btn_weight = add_3d_button([0.245, y, 0.12, h], f'W: {self.pretrained.upper()}', 'lightcyan')
        self.btn_weight.on_clicked(self._btn_weight_clicked)

        # 3. 옵션 버튼 (중앙)
        self.btn_ai = add_3d_button([0.37, y, 0.085, h], 'AI Pick', 'plum')
        self.btn_ai.on_clicked(self._btn_ai_clicked)

        filter_color = 'lightgreen' if self.apply_filter else 'white'
        self.btn_filter = add_3d_button([0.46, y, 0.115, h], f'Filter: {"ON" if self.apply_filter else "OFF"}', filter_color)
        self.btn_filter.on_clicked(self._btn_filter_clicked)

        # 초기 화면의 관측소 Resp 상태 (기본적으로 False -> ON 표시)
        sta = self.station_names[self.current_idx]
        is_resp = self.resp_status.get(sta, False)
        resp_label = f'Resp: {"OFF" if is_resp else "ON"}'
        resp_color = 'lightblue' if is_resp else 'white'
        self.btn_resp = add_3d_button([0.58, y, 0.10, h], resp_label, resp_color)
        self.btn_resp.on_clicked(self._btn_resp_clicked)

        self.btn_clear = add_3d_button([0.685, y, 0.055, h], 'Clear', 'white')
        self.btn_clear.on_clicked(self._btn_clear_clicked)

        self.btn_zstack = add_3d_button([0.745, y, 0.07, h], 'Z-Stack', 'mediumpurple')
        self.btn_zstack.on_clicked(self._btn_zstack_clicked)

        # 4. 저장 및 종료 (우측)
        self.btn_save = add_3d_button([0.82, y, 0.065, h], 'SAVE', 'gold')
        self.btn_save.on_clicked(self._save_to_csv)

        self.btn_exit = add_3d_button([0.89, y, 0.065, h], 'EXIT', 'salmon')
        self.btn_exit.on_clicked(self._btn_exit_clicked)

        # 화면 우측 하단에 kitvalley.png 로고 추가
        if os.path.exists(icon_path):
            try:
                # [left, bottom, width, height]
                ax_logo = self.fig.add_axes([0.88, 0.015, 0.1, 0.07], anchor='SE', zorder=10)
                img = plt.imread(icon_path)
                ax_logo.imshow(img)
                ax_logo.axis('off')
            except Exception as e:
                print(f"[*] 화면 로고 표시 실패: {e}")

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
        print(f"[*] AI 모델이 '{self.ai_model}'(으)로 변경되었습니다.")
        self.fig.canvas.draw_idle()

    def _btn_weight_clicked(self, event):
        current_idx = self.available_weights.index(self.pretrained.lower()) if self.pretrained.lower() in self.available_weights else 0
        next_idx = (current_idx + 1) % len(self.available_weights)
        self.pretrained = self.available_weights[next_idx]
        self.btn_weight.label.set_text(f'W: {self.pretrained.upper()}')
        print(f"[*] AI 가중치가 '{self.pretrained}'(으)로 변경되었습니다.")
        self.fig.canvas.draw_idle()

    def _btn_exit_clicked(self, event):
        print("\n[*] 종료 버튼 클릭됨. 데이터를 저장하고 종료합니다.")
        self._save_to_csv()
        plt.close(self.fig)

    def _sync_resp_button(self):
        """현재 관측소의 Resp 상태에 맞게 버튼 UI를 동기화합니다."""
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
        
        # 필터는 전역 설정이므로 현재 관측소 다시 처리
        sta = self.station_names[self.current_idx]
        self._process_station(sta)
        self._draw_current_station()

    def _btn_resp_clicked(self, event):
        if not self.inv:
            print("[-] Inventory 데이터가 없어 계기응답을 제거할 수 없습니다. (--inv 옵션 필요)")
            return
            
        sta = self.station_names[self.current_idx]
        # 현재 관측소의 상태 반전
        self.resp_status[sta] = not self.resp_status[sta]
        
        is_resp = self.resp_status[sta]
        if is_resp:
            self.btn_resp.label.set_text('Resp: OFF')
            self.btn_resp.ax.set_facecolor('lightblue')
            self.ax_z.set_title(f" {sta} 계기응답(Response) 제거 중... 잠시만 기다려주세요 ", color='blue', fontweight='bold')
        else:
            self.btn_resp.label.set_text('Resp: ON')
            self.btn_resp.ax.set_facecolor('white')
            self.ax_z.set_title(f" {sta} 원본 데이터 복구 중... ", color='blue', fontweight='bold')
            
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()
        
        self._process_station(sta)
        self._draw_current_station()

    def _btn_clear_clicked(self, event):
        sta = self.station_names[self.current_idx]
        self.picks_db[sta] = {'p': {'mpl_num': None, 'conf': 0.0}, 's': {'mpl_num': None, 'conf': 0.0}}
        # 해당 관측소의 확률 곡선도 삭제
        if sta in self.ai_annotations: del self.ai_annotations[sta]
        self._draw_current_station()

    def _btn_zstack_clicked(self, event):
        # 1. P파 피킹이 존재하는 관측소 수집
        picked_stas = []
        for sta in self.station_names:
            p_num = self.picks_db[sta]['p']['mpl_num']
            if p_num is not None:
                picked_stas.append((sta, p_num))
        
        if not picked_stas:
            print("[-] P파 피킹 정보가 없습니다. 최소 1개 이상 피킹 후 사용해주세요.")
            return
            
        # 2. P파 도착시간(가장 빠른 시간) 순으로 정렬
        picked_stas.sort(key=lambda x: x[1])
        first_p_time = picked_stas[0][1] # 제일 첫 P파 시간 (가장 빠른 시간)
        
        # 허용 그리기 범위 계산 (첫 P파 10초 전 ~ 그로부터 180초 후)
        valid_start_time = first_p_time - (10 / 86400.0)
        valid_end_time = valid_start_time + (180 / 86400.0)
        
        # 3. 새로운 창(Figure) 생성 - 스크롤바 공간을 위해 왼쪽 여백 확보
        self.fig_stack, self.ax_stack = plt.subplots(figsize=(10, 8))
        try:
            self.fig_stack.canvas.manager.set_window_title('Z-Component Record Section')
        except:
            pass
            
        # 왼쪽 여백을 0.15로 늘려서 스크롤바 공간 확보
        plt.subplots_adjust(left=0.15, right=0.95, top=0.9, bottom=0.1)
        
        yticks = []
        yticklabels = []
        
        # 드래그 앤 드롭을 위한 Z-Stack 전용 상태 변수 저장소
        self.zstack_lines = {} # { (sta, phase): Line2D_object }
        self.zstack_dragging = None # 현재 드래그 중인 라인 정보 (sta, phase)
        
        print(f"[*] Z-Stack: 총 {len(picked_stas)}개 관측소 정렬 완료.")
        
        # 4. 정렬된 관측소 순서대로 Z 성분 파형 그리기 (관측소 간격 없이 쌓음)
        total_picked = len(picked_stas)
        all_pick_times = []
        for sta, p_num in picked_stas:
            # P, S파 모두 범위 내에 있는지 확인 후 그리기 대상에 포함
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
                
            # 시간 정밀도 문제 해결 및 속도 향상을 위한 벡터화 연산
            t0 = mdates.date2num(tr.stats.starttime.datetime.replace(tzinfo=None))
            num_times = t0 + tr.times() / 86400.0
            
            # 파형 정규화: 상하 관측소 파형이 딱 맞닿도록 최대 진폭을 0.5로 제한 (-0.5 ~ 0.5)
            data = tr.data
            max_val = np.max(np.abs(data))
            if max_val > 0:
                norm_data = (data / max_val) * 0.5
            else:
                norm_data = data
                
            # y축 인덱스: 가장 빠른(0번째) 관측소가 맨 위(가장 큰 Y값)에 오도록 역순 배치
            y_pos = total_picked - 1 - i  
            
            # 파형 그리기
            self.ax_stack.plot(num_times, norm_data + y_pos, 'k-', lw=0.5, picker=False)
            
            # P파 마커 표시 (범위 내일 때만 그림)
            if valid_start_time <= p_num <= valid_end_time:
                line_p, = self.ax_stack.plot([p_num, p_num], [y_pos - 0.5, y_pos + 0.5], 'r-', lw=2, alpha=0.8, picker=True, pickradius=5)
                self.zstack_lines[(sta, 'p')] = line_p
            
            # S파 마커 표시 (범위 내일 때만 그림)
            s_num = self.picks_db[sta]['s']['mpl_num']
            if s_num is not None and valid_start_time <= s_num <= valid_end_time:
                line_s, = self.ax_stack.plot([s_num, s_num], [y_pos - 0.5, y_pos + 0.5], 'b-', lw=2, alpha=0.8, picker=True, pickradius=5)
                self.zstack_lines[(sta, 's')] = line_s
            
            yticks.append(y_pos)
            yticklabels.append(sta)
            
        self.ax_stack.set_yticks(yticks)
        self.ax_stack.set_yticklabels(yticklabels)
        
        # Y축 뷰포트 고정: 한 화면에 8개의 파형만 보이도록 설정
        view_size = 8.0
        top_y_limit = total_picked - 0.5
        bottom_y_limit = -0.5
        
        # 초기 뷰: 가장 빠른 관측소가 맨 위(top_y_limit)에 위치
        current_top = top_y_limit
        current_bottom = max(bottom_y_limit, current_top - view_size)
        self.ax_stack.set_ylim(current_bottom, current_top)
        
        # --- 오른쪽 스크롤바(Slider) 추가 로직 ---
        if total_picked > 8:
            from matplotlib.widgets import Slider
            
            # 스크롤바 위치 지정: ax_stack의 오른쪽에 세로로 길게 (기존 0.02에서 0.96 근처로 이동)
            plt.subplots_adjust(left=0.08, right=0.92, top=0.9, bottom=0.1)
            ax_scroll = self.fig_stack.add_axes([0.94, 0.1, 0.02, 0.8], facecolor='lightgoldenrodyellow')
            
            # 슬라이더 생성 (최솟값: 바닥, 최댓값: 맨 위). 초기값은 맨 위.
            # 스크롤 핸들을 아래로 내리면(값이 작아지면) 아래쪽 파형이 보이도록 뷰가 내려감.
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
            
            # 슬라이더 움직임(드래그)에 따라 Y축 범위 변경
            def update_scroll(val):
                new_top = val
                new_bottom = new_top - view_size
                self.ax_stack.set_ylim(new_bottom, new_top)
                self.fig_stack.canvas.draw_idle()
                
            self.zstack_slider.on_changed(update_scroll)

        # --- X축 범위 자동 조절 로직 ---
        if all_pick_times:
            # P파(빨간 선)들만 모아서 범위 계산
            p_pick_times = [p_num for sta, p_num in picked_stas]
            if p_pick_times:
                min_x = min(p_pick_times)
                max_x = max(p_pick_times)
                
                # 첫 P파 도달시간 10초 전, 마지막 P파 도달시간 60초 후
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
        
        # --- Z-Stack 전용 이벤트 핸들러 바인딩 ---
        self.fig_stack.canvas.mpl_connect('scroll_event', self._on_zstack_scroll)
        self.fig_stack.canvas.mpl_connect('button_press_event', self._on_zstack_press)
        self.fig_stack.canvas.mpl_connect('motion_notify_event', self._on_zstack_motion)
        self.fig_stack.canvas.mpl_connect('button_release_event', self._on_zstack_release)
        
        self.fig_stack.show()

    # --- Z-Stack 전용 이벤트 핸들러 ---
    def _on_zstack_scroll(self, event):
        if event.inaxes != self.ax_stack: return
        
        # 마우스 휠 스크롤은 오직 X축 (시간) 확대/축소만 담당
        scale = 1/1.5 if event.button == 'up' else 1.5
        cur_xlim = self.ax_stack.get_xlim()
        new_width = (cur_xlim[1] - cur_xlim[0]) * scale
        rel_pos = (cur_xlim[1] - event.xdata) / (cur_xlim[1] - cur_xlim[0])
        self.ax_stack.set_xlim([event.xdata - new_width * (1 - rel_pos), event.xdata + new_width * rel_pos])
        
        self.fig_stack.canvas.draw_idle()

    def _on_zstack_press(self, event):
        if event.button != 1 or event.inaxes is None: return
        # 클릭한 라인 찾기
        for (sta, phase), line in self.zstack_lines.items():
            contains, _ = line.contains(event)
            if contains:
                self.zstack_dragging = (sta, phase)
                return

    def _on_zstack_motion(self, event):
        if self.zstack_dragging and event.inaxes:
            sta, phase = self.zstack_dragging
            line = self.zstack_lines[(sta, phase)]
            # 선 위치 업데이트 (X축 위치 이동)
            y_data = line.get_ydata()
            line.set_data([event.xdata, event.xdata], y_data)
            
            # 메인 DB에도 피킹 값 즉시 업데이트
            self.picks_db[sta][phase]['mpl_num'] = event.xdata
            self.fig_stack.canvas.draw_idle()
            
            # --- 메인 창(fig) 동기화 ---
            if hasattr(self, 'fig') and plt.fignum_exists(self.fig.number):
                # 메인 창에 띄워진 관측소가 지금 드래그 중인 관측소와 같다면
                if self.station_names[self.current_idx] == sta:
                    # 메인 창의 3개 축(Z, N, E)에 있는 수직선 위치 즉시 이동
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
        # 주 축과 보조 축 모두 초기화
        for ax in [self.ax_z, self.ax_n, self.ax_e, self.ax_z_prob, self.ax_n_prob, self.ax_e_prob]: 
            ax.clear()
        
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
                # 벡터화 시간 계산
                t0 = mdates.date2num(tr.stats.starttime.datetime.replace(tzinfo=None))
                num_times = t0 + tr.times() / 86400.0
                ax.plot(num_times, tr.data, 'k-', linewidth=0.5)
                
                # 단위 표시 제거, 채널명만 표시
                ax.set_ylabel(f"{tr.stats.channel}", fontsize=9)
                
                t_num_start = num_times[0]
                t_num_end = num_times[-1]
                min_mpl, max_mpl = min(min_mpl, t_num_start), max(max_mpl, t_num_end)
            
            # Y축 스케일 자동 재조정 (Count <-> Velocity 변환 시 필수)
            ax.relim()
            ax.autoscale_view(scalex=False, scaley=True)
            ax.grid(True, linestyle=':', alpha=0.5)

        # AI 확률 곡선 그리기
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

        # 피킹 선 그리기
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

    def _save_to_csv(self, event=None):
        out = "refined_picks.csv"
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
            print(f"[*] 결과 저장 완료: {out}")
            if event:
                self.ax_z.set_title("[SAVED SUCCESS!]", color='red', fontweight='bold')
                self.fig.canvas.draw()
        except Exception as e:
            print(f"[-] 저장 실패: {e}")

    def show(self):
        plt.show()
        self._save_to_csv()

def main():
    parser = argparse.ArgumentParser(description="Interactive Multi-Station Earthquake Phase Picker")
    parser.add_argument("--mseed", required=True, help="분석할 miniseed 데이터 파일 경로")
    parser.add_argument("--picks", help="기존 피킹 데이터 CSV 파일 (없으면 비워둠)")
    parser.add_argument("--filter", action="store_true", help="기본 Bandpass Filter(1-10Hz) 적용 여부")
    parser.add_argument("--model", choices=["eqtransformer", "phasenet"], help="seisbench 딥러닝 모델을 이용한 자동 피킹 (옵션)")
    parser.add_argument("--pretrained", default="original", help="SeisBench 사전 학습 가중치 (예: original, stead, ethz 등)")
    parser.add_argument("--inv", default="total_inv.xml", help="Inventory XML 파일 경로 (계기응답 제거용)")
    args = parser.parse_args()
    
    picker = MultiStationPicker(args.mseed, args.filter, args.picks, args.model, args.pretrained, args.inv)
    if hasattr(picker, 'fig'): 
        picker.show()

if __name__ == "__main__":
    main()