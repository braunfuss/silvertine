from pyrock import orthodrome


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

def to_refrence_igem(lat, lon):
