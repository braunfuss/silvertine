import math
from pyrocko.guts import Object, String, Float, List
from collections import defaultdict

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



class hyposat_inp(Object):
    '''Returns hyposat input files'''
    phase_selection = String.T()
    fallback_time = Float.T(optional=True)

    def __init__(self):
        starting_source_depth_km = 0.1
        zero_level_km = 0
        rg_group_velocity = 2.6
        vp_to_correct_elevation = 3.8
        vs_to_correct_elevation = 2.1
        p_stddev = 0.1
        s_stddev = 0.2

        crust_51_keys = [
            'Off',
            'For station corrections',
            'For local/regional model',
            'For station corrections, local/regional model and surface\
                        reflection corrections']

        crust_51_choices = dict(
            [(b, a) for (a, b) in enumerate(crust_51_keys)])

        crust_51 = "Off"

        locreg_model = "insheim.dat"
        global_model = "ak135"

        params = {}
        params['starting_source_depth_km'] = starting_source_depth_km + zero_level_km
        params['global_model'] = global_model
        params['locreg_model'] = locreg_model
        params['vp_to_correct_elevation'] = vp_to_correct_elevation
        params['vs_to_correct_elevation'] = vs_to_correct_elevation
        params['crust_51'] = crust_51_choices[crust_51]
        params['rg_group_velocity'] = rg_group_velocity
        params['stations_correction_file'] = "station_corrections_insheim.txt"

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
        # TO SEARCH OSCILLATIONS (DEF 4)   : 6
        LOCATION ACCURACY [km] (DEFAULT 1) : 1.
        CONSTRAIN SOLUTION (0/1)           : 1
        CONFIDENCE LEVEL  (68.3 - 99.99 %%) : 95.
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


        def load_stations_from_meta(self, fn=None, db=None):
            if fn is not None:
                stations_landau = num.loadtxt(fn, delimiter=",", dtype='str')
                stations = []
                for st in stations_landau:
                    stations.append(model.station.Station(station=st[0], network="GR", lat=float(st[1]), lon=float(st[2]), elevation=float(st[3])))
                return stations
            elif db == "landau":
                stations_landau = num.loadtxt("stations_landau.pf", delimiter=",", dtype='str')
                stations_landau_pyrocko = []
                for st in stations_landau:
                    stations_landau_pyrocko.append(model.station.Station(station=st[0], network="GR", lat=float(st[1]), lon=float(st[2]), elevation=float(st[3])))
                return stations_landau_pyrocko

            elif db == "insheim":
                stations_insheim = num.loadtxt("stations_insheim.pf", delimiter=",", dtype='str')
                stations_insheim_pyrocko = []
                for st in stations_insheim:
                    stations_insheim_pyrocko.append(model.station.Station(station=st[0], network="GR", lat=float(st[1]), lon=float(st[2]), elevation=float(st[3])))
                return stations_insheim_pyrocko
            elif db == "meta":
                stations_meta_pyrocko = []
                stations_meta = num.loadtxt("meta.txt", delimiter=",", dtype='str')
                for st in stations_meta:
                    stations_meta_pyrocko.append(model.station.Station(station=st[0], network="GR", lat=float(st[1]), lon=float(st[2]), elevation=float(st[3])))
                return stations_meta_pyrocko






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
                event = model.Event(lat, lon, t, depth=depth-(zero_level_km*1000.), name='HYPOSAT-%i' % kind)
                event.ellipse_major = ellipse_major
                event.ellipse_minor = ellipse_minor
                event.ellipse_azimuth = ellipse_azimuth
                evmark = EventMarker(event)
                event_markers.append(evmark)
                evmark.set_kind(kind)
                ev_marker_hyposat.append(evmark)
                return(evmark, dh, dz, rms, float(toks[6]))
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
                    phmarks.append(PhaseMarker([nslc], t, t, 0, phasename=used_phase))

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
