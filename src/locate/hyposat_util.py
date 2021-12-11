import math
from pyrocko.guts import Object, String, Float, List
from collections import defaultdict
from pyrocko.gui.pile_viewer import PhaseMarker, EventMarker
from pyrocko import util, model
import numpy as num
import subprocess
from pyrocko.io import quakeml
from pyrocko.example import get_example_data
zero_level_km = 0


def nsl_str(nsl):
    return '.'.join(nsl)


def to_min_sec(lat, lon):

    if lat >= 0:
        ns = 'N'
    else:
        ns = 'S'
    if lon >= 0:
        ew = 'E'
    else:
        ew = 'W'

    lat = abs(lat)
    lon = abs(lon)
    dlat = lat - math.floor(lat)
    dlon = lon - math.floor(lon)
    mlat = math.floor(dlat*60)
    mlon = math.floor(dlon*60)
    slat = (dlat*60-mlat)*60
    slon = (dlon*60-mlon)*60

    return '%2i%02i%04.1f%s%3i%02i%04.1f%s' % (math.floor(lat), mlat, slat, ns,
                                               math.floor(lon), mlon, slon, ew)


def hyposat_inp():
    hypo_param_tmpl = '''  hyposat-parameter
*GLOBAL MODEL 2                     : iasp91
GLOBAL MODEL                       : %(global_model)s
LOCAL OR REGIONAL MODEL            : %(locreg_model)s
PHASE INDEX FOR LOCAL MODEL        : 0000
OUTPUT OF REGIONAL MODEL (DEF 0)   : 1
STATION FILE                       : stations.dat
STATION CORRECTION FILE            : %(stations_correction_file)s
P-VELOCITY TO CORRECT ELEVATION    : %(vp_to_correct_elevation)g
S-VELOCITY TO CORRECT ELEVATION    : %(vs_to_correct_elevation)g
RG GROUP-VELOCITY (DEF 2.5  [km/s]): %(rg_group_velocity)g
LG GROUP-VELOCITY (DEF 3.5  [km/s]): 3.5752
LQ GROUP-VELOCITY (DEF 4.4  [km/s]):  4.4
LR GROUP-VELOCITY (DEF 3.95 [km/s]): 2.85
STARTING SOURCE TIME (EPOCHAL TIME): %(starting_source_time)s
STARTING TIME ERROR       [s]      : 0.2
STARTING SOURCE DEPTH     [km]     : %(starting_source_depth_km)g
STARTING DEPTH ERROR      [km]     : 2.
DEPTH FLAG (f,b,d,F,B,D)           : d
STARTING SOURCE LATITUDE  [deg]    : 49.2
STARTING LATITUDE ERROR   [deg]    : 1
STARTING SOURCE LONGITUDE [deg]    : 8.2
STARTING LONGITUDE ERROR  [deg]    : 1.
MAGNITUDE CALCULATION (DEF 0)      : 1
P-ATTENUATION MODEL (G-R or V-C)   : V-C
S-ATTENUATION MODEL (IASPEI or R-P): R-P
MAXIMUM # OF ITERATIONS            : 600
# TO SEARCH OSCILLATIONS (DEF 4)   : 20
LOCATION ACCURACY [km] (DEFAULT 1) : 0.1
CONSTRAIN SOLUTION (0/1)           : 1
CONFIDENCE LEVEL  (68.3 - 99.99 %%) : 99.
EPICENTER ERROR ELLIPSE (DEF 1)    : 1
MAXIMUM AZIMUTH ERROR     [deg]    : 20.
MAXIMUM SLOWNESS ERROR    [s/deg]  : 3.
SLOWNESS [S/DEG] ( 0 = APP. VEL)   : 0
FLAG USING TRAVEL-TIME DIFFERENCES : 1
INPUT FILE NAME (DEF hyposat-in)   : _
OUTPUT FILE NAME (DEF hyposat-out) : _
OUTPUT SWITCH  (YES = 1, DEFAULT)  : 1
OUTPUT LEVEL                       : 4
'''

    return hypo_param_tmpl


def read_hypo_output():
        f = open('hyposat-out', 'r')
        hypo_out = f.read()
        f.close()
        ev_marker_hyposat = []
        evhead = 'T0 LAT LON Z VPVS DLAT DLON DZ DT0 DVPVS DEF RMS'.split()
        phhead = 'Stat Delta Azi Phase [used] Onset time Res Baz Res Rayp Res Used'.split()
        ellipsehead = 'Epicenter error ellipse:'.split()
        state = 0
        source_time = None
        phmarks = []
        kind = 1
        event_markers = []
        ellipse_major = None
        ellipse_minor = None
        ellipse_azimuth = None
        for line in hypo_out.splitlines():
            if state == 0:
                if line.split() == evhead:
                    state = 1
                elif line.split() == phhead:
                    state = 2
                elif line.split() == ellipsehead:
                    state = 3
                elif line.lstrip().startswith('Source time  :'):
                    toks = line.split()
                    datestr = ' '.join(toks[3:9])
                    source_date_str = ' '.join(toks[3:6])
                    source_time = util.str_to_time(datestr, format='%Y %m %d %H %M %S.3FRAC')

            elif state == 1:
                toks = line.split()
                datestr = ' '.join(toks[:4])
                t = util.str_to_time(datestr, format='%Y-%m-%d %H %M %S.3FRAC')
                lat = float(toks[4])
                lon = float(toks[5])
                depth = float(toks[6])*1000.
                try:
                    dz = float(toks[10])
                except:
                    dz = 0.
                try:
                    dh = float(toks[9])
                except:
                    dh = 0.
                rms = float(toks[14])
                event = model.Event(lat=lat, lon=lon, time=t, depth=depth-(zero_level_km*1000.), name='HYPOSAT-%i' % kind)
                event.ellipse_major = ellipse_major
                event.ellipse_minor = ellipse_minor
                event.ellipse_azimuth = ellipse_azimuth
                evmark = EventMarker(event)
                event_markers.append(evmark)
                evmark.set_kind(kind)
                ev_marker_hyposat.append(evmark)
                return(event, evmark, dh, dz, rms, float(toks[6]))
                phmarks = []
                kind += 1
                state = 0

            elif state == 2:
                toks = line[:58].split()
                if len(toks) >= 8:
                    have_used = 0
                    if len(toks) == 9:
                        have_used = 1
                    station = toks[0]
                    phase = toks[3]
                    if have_used:
                        used_phase = toks[4]
                    else:
                        used_phase = phase
                    residual = float(toks[7+have_used])
                    timestr = ' '.join(toks[4+have_used:7+have_used])
                    datestr = source_date_str + ' ' + timestr
                    t = util.str_to_time(datestr, format='%Y %m %d %H %M %S.3FRAC') - residual
                    if t - source_time < - 12*3600.:
                        t += 24*3600.

                    if t - source_time > 12*3600.:
                        t -= 24*3600.

                    nslc = station_phase_to_nslc[station, phase]
                    phmarks.append(PhaseMarker([nslc], t, t, 0,
                                                phasename=used_phase))

                else:
                    if len(toks) == 0 and phmarks:
                        state = 0

            elif state == 3:
                toks = line.split()
                if len(toks) == 0:
                    state = 0
                elif toks[0] == 'Major':
                    print(toks)
                    ellipse_major = float(toks[3])*1000.
                    ellipse_minor = float(toks[8])*1000.
                elif toks[0] == 'Azimuth:':
                    ellipse_azimuth = float(toks[1])


def write_hyposat_earthmodel(mod, name, folder=None):
    file = open("data/"+name, 'w+')
    file.write("5.\n")
    for lay in mod.layers():
        file.write("     %s     %s     %s\n" % (lay.zbot/1000., (lay.mbot.vp), (lay.mbot.vs)))
    file.close()


def run_hyposat(ev, event, ev_list, stations_hyposat, mod, mod_name):
    starting_source_depth_km = 0.1
    zero_level_km = 0
    rg_group_velocity = 2.6
    vp_to_correct_elevation = 3.8
    vs_to_correct_elevation = 2.1
    p_stddev = 0.5
    s_stddev = 0.5
    event_marker_out = []

    crust_51_keys = [
        'Off',
        'For station corrections',
        'For local/regional model',
        'For station corrections, local/regional model and surface\
                    reflection corrections']

    crust_51_choices = dict(
        [(b, a) for (a, b) in enumerate(crust_51_keys)])

    crust_51 = "Off"

    write_hyposat_earthmodel(mod, mod_name)
    locreg_model = mod_name
    global_model = "ak135"

    params = {}
    params['starting_source_depth_km'] = starting_source_depth_km + zero_level_km
    params['global_model'] = global_model
    params['locreg_model'] = locreg_model
    params['vp_to_correct_elevation'] = vp_to_correct_elevation
    params['vs_to_correct_elevation'] = vs_to_correct_elevation
    params['crust_51'] = crust_51_choices[crust_51]
    params['rg_group_velocity'] = rg_group_velocity
    params['stations_correction_file'] = "_"
    station_phase_to_nslc = {}
    hypo_param_tmpl = hyposat_inp()
    hypo_in = []
    for ev_phase_markers in ev_list:
        for marker in ev_phase_markers:
            if not isinstance(marker, PhaseMarker):
                continue
            phasename = marker.get_phasename()
            phase = phasename
            if phase == "P<(moho)":
                phase = "Pg"
            if phase == "p<(moho)":
                phase = pg
            if phase == "S<(moho)":
                phase = "Sg"
            if phase == "s<(moho)":
                phase = "sg"
            if phase == "P>(moho)":
                phase = "PG"
            if phase == "p>(moho)":
                phase = "pG"
            if phase == "S>(moho)":
                phase = "SG"
            if phase == "s>(moho)":
                phase = "sG"
            if phase == 'Sv(moho)s':
                phase = "SmS"
            if phase == 'Pv(moho)p':
                phase = "PmP"
            if phase == 'Pv_(moho)p':
                phase = "Pn"
            if phase == 'Sv_(moho)s':
                phase = "Sn"
            if phase == 'p':
                phase = "P"
            if phase == 's':
                phase = "S"
            phasename = phase
            nslcs = list(marker.nslc_ids)
            station = nslcs[1]

            station_phase_to_nslc[station, phasename] = nslcs[0]

            backazi = -1.
            backazi_stddev = 0.0
            slowness = -1.
            slowness_stddev = 0.0
            period = 0.0
            amplitude = 0.0
            flags = 'T__DR_'
            t = marker.tmin

            date_str = util.time_to_str(t, '%Y %m %d %H %M %S.3FRAC')
            if phasename[0] == 'P':
                t_stddev = p_stddev
            elif phasename[0] == 'S':
                t_stddev = s_stddev
            else:
                t_stddev = p_stddev
            hypo_in.append((station, phasename, date_str, t_stddev,
                            backazi, backazi_stddev, slowness,
                            slowness_stddev, flags, period, amplitude))

        hypo_in.sort()
        print(hypo_in)
        if len(hypo_in) == 0:
            pass
        else:
            f = open("hyposat-in", 'w')
            f.write('\n')
            for vals in hypo_in:
                s = '%-5s %-8s %s %5.3f %6.2f %5.2f %5.2f %5.2f %-6s %6.3f %12.2f' % vals

                f.write(s+'\n')

            f.close()
    try:

        params['starting_source_time'] = util.time_to_str(num.min(event.time)-60.)

        f = open('stations.dat', 'w')
        sta_lat_lon = []
        for sta in stations_hyposat:
            s = '%-5s%1s%s%7.1f' % (
                sta.station, ' ', to_min_sec(sta.lat, sta.lon),
                sta.elevation - zero_level_km*1000.)

            f.write(s+'\n')
            sta_lat_lon.append((sta.lat, sta.lon))
        f.close()

        f = open("hyposat-parameter", 'w')
        f.write(hypo_param_tmpl % params)
        f.close()

        subprocess.run(["tcsh", "hypo.tcsh"])
        event, marker, dh, dz, rms, depth = read_hypo_output()
        event_marker_out.append(marker)
        event.tags = [str(rms), str(ev["id"])]

    except:
        event = model.Event(lat=ev["lat"], lon=ev["lon"], time=ev["time"], depth=ev["depth"], name="non hypo")
    return event
