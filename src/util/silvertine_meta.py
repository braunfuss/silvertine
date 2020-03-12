from pyrocko import util
from pyrocko import util, pile, model, config, trace, io, pile, catalog
from pyrocko.client import fdsn


def find_station(name, tmin=None, tmax=None):
    if tmin and tmax is None:
        tmin = util.stt('2013-10-23 21:06:54.400')
        tmax = util.stt('2013-10-23 21:11:59.000')

    selection = [('*', name, '*', '*', tmin, tmax)]

    sites = ["bgr", "http://192.168.11.220:8080",
             "http://gpiseisgate.gpi.kit.edu", "geofon", "iris", "orfeus",
             "koeri", "ethz", "lmu", "resif", "geonet", "ingv"]
    for site in sites:
        try:
            request_response = fdsn.station(
                site='%s' % site, selection=selection, level='response')
            pyrocko_stations = request_response.get_pyrocko_stations()
            stations_hyposat.append(pyrocko_stations[0])
            stations_inversion.append(pyrocko_stations[0])
            st = pyrocko_stations[0]
            return st.lat, st.lon
        except:
            return False


def load_silvertine_stations():

    stations_landau = num.loadtxt("data/stations_landau.pf", delimiter=",", dtype='str')
    stations_landau_pyrocko = []
    for st in stations_landau:
        stations_landau_pyrocko.append(model.station.Station(station=st[0], network="GR", lat=float(st[1]), lon=float(st[2]), elevation=float(st[3])))

    stations_insheim = num.loadtxt("data/stations_insheim.pf", delimiter=",", dtype='str')
    stations_insheim_pyrocko = []
    for st in stations_insheim:
        stations_insheim_pyrocko.append(model.station.Station(station=st[0], network="GR", lat=float(st[1]), lon=float(st[2]), elevation=float(st[3])))

    stations_meta_pyrocko = []
    stations_meta = num.loadtxt("data/meta.txt", delimiter=",", dtype='str')
    for st in stations_meta:
        stations_meta_pyrocko.append(model.station.Station(station=st[0], network="GR", lat=float(st[1]), lon=float(st[2]), elevation=float(st[3])))

    return stations_landau_pyrocko, stations_insheim_pyrocko, stations_meta_pyrocko



def load_geress_phase_picks():
    events = num.loadtxt("data/geres_epi.csv", delimiter="\t", dtype='str')
    event_marker_out = []
    ev_dict_list = []
    for ev in events:
        date = str(ev[1])
        time = str(ev[2])
        try:
            h, m = [int(s) for s in time.split('.')]
        except:
            h = time
            m = 0.
        if len(str(h)) == 5:
            time = "0"+time[0]+":"+time[1:3]+":"+time[3:5]+time[5:]
        elif len(str(h)) == 4:
            time = "00"+":"+time[0:2]+":"+time[3:5]+time[5:]
        elif len(str(h)) == 3:
            time = "00"+ ":"+ "0" +time[0]+":"+time[1:4]+time[4:]
        else:
            time = time[0:2]+":"+time[2:4]+":"+time[4:5]+time[5:]
        date = str(date[0:4])+"-"+str(date[4:6]+"-"+date[6:8]+" ")
        ev_time = util.str_to_time(date+time)
        try:
            ev_dict_list.append(dict(id=ev[0], time=ev_time, lat=float(ev[3]), lon=float(ev[4]), mag=float(ev[5]), mag_type=ev[6], source=ev[7], phases=[], depth=[], rms=[], error_h = [], error_z =[]  ))
        except:
            ev_dict_list.append(dict(id=ev[0], time=ev_time, lat=float(ev[3]), lon=float(ev[4]), mag=None, mag_type=None, source=ev[5], phases=[], depth=[], rms=[], error_h = [], error_z=[]  ))


    picks = num.loadtxt("data/geres_phas.csv", delimiter="\t", dtype='str')
    return ev_dict_list, picks


def convert_phase_picks_to_pyrocko(ev_dict_list, picks):
    pyrocko_stations = []
    for ev in ev_dict_list:
        phase_markers = []
        stations_event = []
        hypo_in = []
        station_phase_to_nslc = {}
        times = []
        for p in picks:
            if p[0] == ev["id"]:
                date = str(p[3])
                time = str(p[4])
                try:
                    h, m = [int(s) for s in time.split('.')]
                except:
                    h = time
                    m = 0.
                if len(str(h)) == 5:
                    time = "0"+time[0]+":"+time[1:3]+":"+time[3:5]+time[5:]
                elif len(str(h)) == 4:
                    time = "00"+":"+time[0:2]+":"+time[3:5]+time[5:]
                elif len(str(h)) == 3:
                    time = "00"+ ":"+ "0" +time[0]+":"+time[1:4]+time[4:]
                else:
                    time = time[0:2]+":"+time[2:4]+":"+time[4:5]+time[5:]
                date = str(date[0:4])+"-"+str(date[4:6]+"-"+date[6:8]+" ")
                ev_time = util.str_to_time(date+time)
                times.append(ev_time)
                ## depth!+
                if p[1] not in stations_event:
                    for st in stations_meta_pyrocko:
                            if st.station == p[1]:
                                print("found in list meta")
                                stations_event.append(p[1])
                                stations_hyposat.append(st)
                                stations_inversion.append(st)
                                station = st
                if p[1] not in stations_event:
                    for st in stations_insheim_pyrocko:
                            if st.station == p[1]:
                                print("found in list insheim")
                                stations_event.append(p[1])
                                stations_hyposat.append(st)
                                stations_inversion.append(st)
                                station = st
                if p[1] not in stations_event:
                    for st in stations_landau_pyrocko:
                            if st.station == p[1]:
                                print("found in list landau")
                                stations_event.append(p[1])
                                stations_hyposat.append(st)
                                stations_inversion.append(st)
                                station = st
                if p[1] not in stations_event:
                        print("finding meta on:",p[1])
                        try:
                            if p[1] not in stations_event:
                                lat, lon = find_station(p[1])
                                stations_event.append(p[1])
                                print(p[1])
                                station = model.station.Station(station=p[1], network="GR", lat=float(lat), lon=float(lon), elevation=0.)
                        except:
                            pass
                if p[1] not in stations_event:
                    print(p[1], "not found")
                else:
                    event = model.event.Event(lat=ev["lat"], lon=ev["lon"], time=ev["time"], catalog=ev["source"], magnitude=ev["mag"])
                    phase_markers.append(PhaseMarker(["0",p[1]], ev_time, ev_time, 0, phasename=p[2], event_hash=ev["id"], event=event))
                    ev["phases"].append(dict(station=p[1], phase=p[2], pick=ev_time))
                    pyrocko_stations.append(station)
        ev_list.append(phase_markers)

    return ev_list
