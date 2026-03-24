import os
import math
import warnings
import numpy as np
import matplotlib.pyplot as plt
from obspy import UTCDateTime
from obspy.geodetics.base import gps2dist_azimuth
from obspy.taup import TauPyModel
from scipy.optimize import least_squares
import matplotlib.dates as mdates

def locate_hypocenter(picks_db, station_coords, ref_time):
    """
    Calculate hypocenter (lat, lon, depth, origin_time) using P-picks.
    Returns: (est_lat, est_lon, est_depth, origin_time, used_stations, used_arrivals)
    """
    used_stations, used_arrivals = [], []
    
    # Filter P-picks
    picks_data = {}
    for sta, pdata in picks_db.items():
        if pdata['p']['mpl_num'] is not None:
            p_time = UTCDateTime(mdates.num2date(pdata['p']['mpl_num']).replace(tzinfo=None))
            picks_data[sta] = {'time': p_time, 'conf': pdata['p']['conf']}
            
    if not picks_data:
        raise ValueError("No P-picks available for location.")

    sorted_stas = sorted(picks_data.items(), key=lambda x: x[1]['time'])
    # 5% outlier trimming (robust locating)
    n = len(sorted_stas)
    trim_count = int(n * 0.05)
    if trim_count > 0 and n > 2 * trim_count:
        sorted_stas = sorted_stas[trim_count:-trim_count]
    valid_p_stas = dict(sorted_stas)
    
    for sta, pdata in valid_p_stas.items():
        t = pdata['time']
        if sta in station_coords:
            used_stations.append((sta, station_coords[sta][0], station_coords[sta][1]))
            used_arrivals.append(t - ref_time)
            
    if len(used_stations) < 3:
        raise ValueError(f"Not enough stations with coordinates (need 3, got {len(used_stations)}).")

    taup_model = TauPyModel(model='iasp91')

    def calc_tt(ev_lat, ev_lon, ev_depth, s_lat, s_lon):
        d_m, _, _ = gps2dist_azimuth(ev_lat, ev_lon, s_lat, s_lon)
        try:
            arr = taup_model.get_travel_times(source_depth_in_km=max(0, ev_depth), 
                                            distance_in_degree=d_m/111194.9, 
                                            phase_list=['P', 'p'])
            return arr[0].time if arr else 999.0
        except: return 999.0

    def residuals(params):
        ev_lat, ev_lon, ev_depth, ot_off = params
        return [used_arrivals[i] - (ot_off + calc_tt(ev_lat, ev_lon, ev_depth, s[1], s[2])) for i, s in enumerate(used_stations)]

    initial = [np.mean([s[1] for s in used_stations]), np.mean([s[2] for s in used_stations]), 10.0, -10.0]
    result = least_squares(residuals, initial, 
                           bounds=([-90.0, -180.0, 0.0, -3600.0], [90.0, 180.0, 800.0, 3600.0]), 
                           loss='soft_l1')
    
    est_lat, est_lon, est_depth, est_ot_off = result.x
    origin_time = ref_time + est_ot_off
    
    return est_lat, est_lon, est_depth, origin_time, used_stations, used_arrivals

def calculate_magnitude(est_lat, est_lon, origin_time, station_coords, stations_raw, inv):
    """
    Calculate Local Magnitude (ML) based on Wood-Anderson simulation.
    Returns: (total_ml, ml_results, station_ml_details)
    """
    paz_wa_vel = {
        'poles': [-6.283 + 4.7124j, -6.283 - 4.7124j],
        'zeros': [0 + 0j],
        'gain': 1.0,
        'sensitivity': 2800.0
    }
    
    ml_results = []
    station_ml_details = []
    
    for net_sta, coords in station_coords.items():
        if net_sta not in stations_raw:
            continue
            
        dist_km = gps2dist_azimuth(est_lat, est_lon, coords[0], coords[1])[0] / 1000.0
        
        for comp_key in ['Z', 'N', 'E']:
            tr = stations_raw[net_sta].get(comp_key)
            if tr is None: continue
                
            tr_copy = tr.copy()
            tr_copy.trim(starttime=origin_time, endtime=origin_time + 600)
            if len(tr_copy.data) == 0: continue
                
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    tr_copy.remove_response(inventory=inv, output='VEL', pre_filt=None)
                    tr_copy.simulate(paz_remove=None, paz_simulate=paz_wa_vel)
                    
                max_amp_mm = (np.max(tr_copy.data) - np.min(tr_copy.data)) * 1000.0 / 2.0
                if max_amp_mm <= 1e-8: continue
                
                if comp_key in ['N', 'E']: 
                    a0 = -0.587 * math.log10(max(dist_km, 0.1) / 100.0) - 0.00168 * (dist_km - 100.0) - 3.0
                else: 
                    a0 = -0.511 * math.log10(max(dist_km, 0.1) / 100.0) - 0.00170 * (dist_km - 100.0) - 3.0
                    
                ml = math.log10(max_amp_mm) - a0
                ml_results.append(ml)
                station_ml_details.append({
                    'station': net_sta,
                    'comp': comp_key,
                    'dist_km': dist_km,
                    'max_amp_mm': max_amp_mm,
                    'ml': ml
                })
            except Exception:
                pass
    
    total_ml = np.median(ml_results) if ml_results else None
    return total_ml, ml_results, station_ml_details

def plot_map(eq_lat, eq_lon, used_stations, origin_time=None, magnitude=None, output_dir=".", show_plot=True):
    """
    Plots the earthquake epicenter and used stations on a map using Cartopy.
    used_stations is a list of tuples: (station_name, lat, lon)
    """
    try:
        import cartopy.crs as ccrs
        import cartopy.feature as cfeature
    except ImportError:
        print("[-] Cartopy is not installed. Map plotting disabled.")
        return None

    fig = plt.figure(figsize=(10, 10))
    ax = plt.axes(projection=ccrs.PlateCarree())
    
    # Zoom in around the epicenter (+/- 2 degrees)
    ax.set_extent([eq_lon - 2.5, eq_lon + 2.5, eq_lat - 2.5, eq_lat + 2.5], crs=ccrs.PlateCarree())

    ax.add_feature(cfeature.LAND, edgecolor='black', facecolor='whitesmoke')
    ax.add_feature(cfeature.OCEAN, facecolor='lightblue')
    ax.add_feature(cfeature.COASTLINE, linewidth=1.0)
    ax.add_feature(cfeature.BORDERS, linestyle=':')

    # Plot ONLY used stations
    if used_stations:
        st_lats = [s[1] for s in used_stations]
        st_lons = [s[2] for s in used_stations]
        st_names = [s[0].split('.')[-1] for s in used_stations] # Extract station code

        ax.plot(st_lons, st_lats, 'bv', markersize=8, transform=ccrs.PlateCarree(), label='Used Stations', alpha=0.8)
        
        # Add labels for stations
        for i, name in enumerate(st_names):
            ax.text(st_lons[i], st_lats[i] + 0.05, name, 
                    transform=ccrs.PlateCarree(), fontsize=9, ha='center', va='bottom', 
                    bbox=dict(facecolor='white', alpha=0.5, edgecolor='none', pad=1))

    # Plot epicenter
    ax.plot(eq_lon, eq_lat, 'r*', markersize=20, markeredgecolor='black', transform=ccrs.PlateCarree(), label='Epicenter', zorder=5)
    
    ax.gridlines(draw_labels=True, linestyle='--', alpha=0.5)
    
    title_str = 'Earthquake Location Map'
    info_parts = []
    if origin_time:
        info_parts.append(origin_time.strftime('%Y-%m-%d %H:%M:%S.%f')[:-3])
    if magnitude is not None:
        info_parts.append(f"M {magnitude:.2f}")
    
    if info_parts:
        title_str += f"\n({' | '.join(info_parts)})"
        
    plt.title(title_str, pad=20, fontsize=14, fontweight='bold')
    
    map_path = os.path.join(output_dir, "location_map.png")
    plt.savefig(map_path, dpi=300, bbox_inches='tight')
    print(f"[*] Map saved to {map_path}")
    
    if show_plot:
        try:
            fig.canvas.manager.set_window_title('Earthquake Location Map')
        except: pass
        fig.show()
    else:
        plt.close(fig)
        
    return map_path
