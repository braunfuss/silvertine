from pyrocko.guts import Float
from pyrocko import gf
from pyrocko.gf import meta
from pyrocko import moment_tensor as mtm
from pyrocko.gf.seismosizer import Source
from pyrocko import util, pile, model, config, trace, io, pile, catalog
from pyrocko.client import fdsn
import numpy as num
from pyrocko.gui.pile_viewer import PhaseMarker, EventMarker
from beat.utility import get_rotation_matrix
pi = num.pi
pi4 = pi / 4.
km = 1000.
d2r = pi / 180.
r2d = 180. / pi

sqrt3 = num.sqrt(3.)
sqrt2 = num.sqrt(2.)
sqrt6 = num.sqrt(6.)


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
        except Exception:
            return False


def load_silvertine_stations():

    stations_landau = num.loadtxt("data/stations_landau.pf", delimiter=",",
                                  dtype='str')
    stations_landau_pyrocko = []
    for st in stations_landau:
        stations_landau_pyrocko.append(model.station.Station(station=st[0],
                                                             network="GR",
                                                             lat=float(st[1]),
                                                             lon=float(st[2]),
                                                             elevation=float(st[3])))

    stations_insheim = num.loadtxt("data/stations_insheim.pf", delimiter=",",
                                   dtype='str')
    stations_insheim_pyrocko = []
    for st in stations_insheim:
        stations_insheim_pyrocko.append(model.station.Station(station=st[0],
                                                              network="GR",
                                                              lat=float(st[1]),
                                                              lon=float(st[2]),
                                                              elevation=float(st[3])))

    stations_meta_pyrocko = []
    stations_meta = num.loadtxt("data/meta.txt", delimiter=",", dtype='str')
    for st in stations_meta:
        stations_meta_pyrocko.append(model.station.Station(station=st[0],
                                                           network="GR",
                                                           lat=float(st[1]),
                                                           lon=float(st[2]),
                                                           elevation=float(st[3])))

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
        except Exception:
            h = time
            m = 0.
        if len(str(h)) == 5:
            time = "0"+time[0]+":"+time[1:3]+":"+time[3:5]+time[5:]
        elif len(str(h)) == 4:
            time = "00"+":"+time[0:2]+":"+time[3:5]+time[5:]
        elif len(str(h)) == 3:
            time = "00" + ":" + "0" + time[0]+":"+time[1:4]+time[4:]
        else:
            time = time[0:2]+":"+time[2:4]+":"+time[4:5]+time[5:]
        date = str(date[0:4])+"-"+str(date[4:6]+"-"+date[6:8]+" ")
        ev_time = util.str_to_time(date+time)
        try:
            ev_dict_list.append(dict(id=ev[0], time=ev_time, lat=float(ev[3]),
                                lon=float(ev[4]), mag=float(ev[5]),
                                mag_type=ev[6], source=ev[7], phases=[],
                                depth=[], rms=[], error_h=[], error_z=[]))
        except Exception:
            ev_dict_list.append(dict(id=ev[0], time=ev_time, lat=float(ev[3]),
                                lon=float(ev[4]), mag=None, mag_type=None,
                                source=ev[5], phases=[], depth=[], rms=[],
                                error_h=[], error_z=[]))

    picks = num.loadtxt("data/geres_phas.csv", delimiter="\t", dtype='str')
    return ev_dict_list, picks


def load_ev_dict_list(path=None, nevent=0):

    events = num.loadtxt("data/geres_epi.csv", delimiter="\t", dtype='str')
    event_marker_out = []
    ev_dict_list = []
    if nevent is not None:
        events = events[0:nevent]
    for ev in events:
        date = str(ev[1])
        time = str(ev[2])
        try:
            h, m = [int(s) for s in time.split('.')]
        except Exception:
            h = time
            m = 0.
        if len(str(h)) == 5:
            time = "0"+time[0]+":"+time[1:3]+":"+time[3:5]+time[5:]
        elif len(str(h)) == 4:
            time = "00"+":"+time[0:2]+":"+time[3:5]+time[5:]
        elif len(str(h)) == 3:
            time = "00" + ":" + "0" + time[0]+":"+time[1:4]+time[4:]
        else:
            time = time[0:2]+":"+time[2:4]+":"+time[4:5]+time[5:]
        date = str(date[0:4])+"-"+str(date[4:6]+"-"+date[6:8]+" ")
        ev_time = util.str_to_time(date+time)
        try:
            ev_dict_list.append(dict(id=ev[0], time=ev_time, lat=float(ev[3]),
                                lon=float(ev[4]), mag=float(ev[5]),
                                mag_type=ev[6], source=ev[7], phases=[],
                                depth=[], rms=[], error_h=[], error_z=[]))
        except:
            ev_dict_list.append(dict(id=ev[0], time=ev_time, lat=float(ev[3]),
                                lon=float(ev[4]), mag=None, mag_type=None,
                                source=ev[5], phases=[], depth=[], rms=[],
                                error_h=[], error_z=[]))

    picks = num.loadtxt("data/geres_phas.csv", delimiter="\t", dtype='str')
    return ev_dict_list, picks


def convert_phase_picks_to_pyrocko(ev_dict_list, picks, nevent=0):
    stations_landau_pyrocko, stations_insheim_pyrocko, stations_meta_pyrocko = load_silvertine_stations()
    pyrocko_stations = []
    stations_hyposat = []
    stations_inversion = []
    pyrocko_events = []
    ev_list = []
    for ev in ev_dict_list:
        pyrocko_station = []
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
                    time = "00" + ":" + "0" + time[0]+":"+time[1:4]+time[4:]
                else:
                    time = time[0:2]+":"+time[2:4]+":"+time[4:5]+time[5:]
                date = str(date[0:4])+"-"+str(date[4:6]+"-"+date[6:8]+" ")
                pick_time = util.str_to_time(date+time)
                times.append(pick_time)
                if p[1] not in stations_event:
                    for st in stations_meta_pyrocko:
                        if st.station == p[1]:
                            stations_event.append(p[1])
                            stations_hyposat.append(st)
                            stations_inversion.append(st)
                            station = st
                if p[1] not in stations_event:
                    for st in stations_insheim_pyrocko:
                        if st.station == p[1]:
                            stations_event.append(p[1])
                            stations_hyposat.append(st)
                            stations_inversion.append(st)
                            station = st
                if p[1] not in stations_event:
                    for st in stations_landau_pyrocko:
                        if st.station == p[1]:
                            stations_event.append(p[1])
                            stations_hyposat.append(st)
                            stations_inversion.append(st)
                            station = st
                if p[1] not in stations_event:
                    print("finding meta online on:", p[1])
                    try:
                        if p[1] not in stations_event:
                            lat, lon = find_station(p[1])
                            stations_event.append(p[1])
                            station = model.station.Station(station=p[1],
                                                            network="GR",
                                                            lat=float(lat),
                                                            lon=float(lon),
                                                            elevation=0.)
                    except Exception:
                        pass
                if p[1] not in stations_event:
                    print(p[1], "not found in lists or online")
                else:
                    try:
                        event = model.event.Event(lat=float(ev["lat"]),
                                                  lon=float(ev["lon"]),
                                                  time=float(ev["time"]),
                                                  catalog=str(ev["source"]),
                                                  magnitude=float(ev["mag"]))
                    except:
                        event = model.event.Event(lat=float(ev["lat"]),
                                                  lon=float(ev["lon"]),
                                                  time=float(ev["time"]),
                                                  catalog=str(ev["source"]),
                                                  magnitude=-9)

                    phase_markers.append(PhaseMarker(["0", p[1]], pick_time,
                                                     pick_time, 0,
                                                     phasename=p[2],
                                                     event_hash=ev["id"],
                                                     event=event))
                    ev["phases"].append(dict(station=p[1], phase=p[2],
                                        pick=pick_time-event.time))
                    pyrocko_station.append(station)
        pyrocko_events.append(event)

        pyrocko_stations.append(pyrocko_station)
        ev_list.append(phase_markers)

    return ev_list, pyrocko_stations, pyrocko_events, ev_dict_list


def load_data(data_folder=None, nevent=None):
    ev_dict_list, picks = load_ev_dict_list(path=data_folder,
                                            nevent=nevent)
    ev_list_picks, stations, ev_list, ev_dict_list = convert_phase_picks_to_pyrocko(ev_dict_list, picks, nevent=nevent)
    if nevent is None:
        return ev_list, stations, ev_dict_list, ev_list_picks
    else:
        return ev_list, stations, ev_dict_list, ev_list_picks


class MTQTSource(gf.SourceWithMagnitude):
    """
    A moment tensor point source.
    Notes
    -----
    Following Q-T parameterization after Tape & Tape 2015
    """

    discretized_source_class = meta.DiscretizedMTSource

    u = Float.T(
        default=0.,
        help='Lune co-latitude transformed to grid.'
             'Defined: 0 <= u <=3/4pi')

    v = Float.T(
        default=0.,
        help='Lune co-longitude transformed to grid.'
             'Definded: -1/3 <= v <= 1/3')

    kappa = Float.T(
        default=0.,
        help='Strike angle equivalent of moment tensor plane.'
             'Defined: 0 <= kappa <= 2pi')

    sigma = Float.T(
        default=0.,
        help='Rake angle equivalent of moment tensor slip angle.'
             'Defined: -pi/2 <= sigma <= pi/2')

    h = Float.T(
        default=0.,
        help='Dip angle equivalent of moment tensor plane.'
             'Defined: 0 <= h <= 1')

    def __init__(self, **kwargs):
        n = 1000
        self._beta_mapping = num.linspace(0, pi, n)
        self._u_mapping = \
            (3. / 4. * self._beta_mapping) - \
            (1. / 2. * num.sin(2. * self._beta_mapping)) + \
            (1. / 16. * num.sin(4. * self._beta_mapping))

        self.lambda_factor_matrix = num.array(
            [[sqrt3, -1., sqrt2],
             [0., 2., sqrt2],
             [-sqrt3, -1., sqrt2]], dtype='float64')

        self.R = get_rotation_matrix()
        self.roty_pi4 = self.R['y'](-pi4)
        self.rotx_pi = self.R['x'](pi)

        self._lune_lambda_matrix = num.zeros((3, 3), dtype='float64')

        Source.__init__(self, **kwargs)

    @property
    def gamma(self):
        """
        Lunar co-longitude, dependend on v
        """
        return (1. / 3.) * num.arcsin(3. * self.v)

    @property
    def beta(self):
        """
        Lunar co-latitude, dependend on u
        """
        return num.interp(self.u, self._u_mapping, self._beta_mapping)

    def delta(self):
        """
        From Tape & Tape 2012, delta measures departure of MT being DC
        Delta = Gamma = 0 yields pure DC
        """
        return (pi / 2.) - self.beta

    @property
    def theta(self):
        return num.arccos(self.h)

    @property
    def rot_theta(self):
        return self.R['x'](self.theta)

    @property
    def rot_kappa(self):
        return self.R['z'](-self.kappa)

    @property
    def rot_sigma(self):
        return self.R['z'](self.sigma)

    @property
    def lune_lambda(self):
        sin_beta = num.sin(self.beta)
        cos_beta = num.cos(self.beta)
        sin_gamma = num.sin(self.gamma)
        cos_gamma = num.cos(self.gamma)
        vec = num.array([sin_beta * cos_gamma, sin_beta * sin_gamma, cos_beta])
        return 1. / sqrt6 * self.lambda_factor_matrix.dot(vec)

    @property
    def lune_lambda_matrix(self):
        num.fill_diagonal(self._lune_lambda_matrix, self.lune_lambda)
        return self._lune_lambda_matrix

    @property
    def rot_V(self):
        return self.rot_kappa.dot(self.rot_theta).dot(self.rot_sigma)

    @property
    def rot_U(self):
        return self.rot_V.dot(self.roty_pi4)

    @property
    def m9_nwu(self):
        """
        MT orientation is in NWU
        """
        return self.rot_U.dot(
            self.lune_lambda_matrix).dot(num.linalg.inv(self.rot_U))

    @property
    def m9(self):
        """
        Pyrocko MT in NED
        """
        return self.rotx_pi.dot(self.m9_nwu).dot(self.rotx_pi.T)

    @property
    def m6(self):
        return mtm.to6(self.m9)

    @property
    def m6_astuple(self):
        return tuple(self.m6.ravel().tolist())

    def base_key(self):
        return Source.base_key(self) + self.m6_astuple

    def discretize_basesource(self, store, target=None):
        times, amplitudes = self.effective_stf_pre().discretize_t(
            store.config.deltat, self.time)
        m0 = mtm.magnitude_to_moment(self.magnitude)
        m6s = self.m6 * m0
        return meta.DiscretizedMTSource(
            m6s=m6s[num.newaxis, :] * amplitudes[:, num.newaxis],
            **self._dparams_base_repeated(times))

    def pyrocko_moment_tensor(self):
        return mtm.MomentTensor(m=mtm.symmat6(*self.m6_astuple) * self.moment)

    def pyrocko_event(self, **kwargs):
        mt = self.pyrocko_moment_tensor()
        return Source.pyrocko_event(
            self,
            moment_tensor=self.pyrocko_moment_tensor(),
            magnitude=float(mt.moment_magnitude()),
            **kwargs)

    @classmethod
    def from_pyrocko_event(cls, ev, **kwargs):
        d = {}
        mt = ev.moment_tensor
        if mt:
            d.update(m6=list(map(float, mt.m6())))

        d.update(kwargs)
        return super(MTQTSource, cls).from_pyrocko_event(ev, **d)

    def __getstate__(self):
        state = self.__dict__.copy()
        state['R'] = None
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        self.R = get_rotation_matrix()
