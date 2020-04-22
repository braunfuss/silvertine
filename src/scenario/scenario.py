import random
import math
import numpy as num
from os.path import join as pjoin
import os.path as op
from pyrocko.moment_tensor import MomentTensor

from pyrocko import util, model, io, trace, config
from pyrocko.gf import Target, DCSource, RectangularSource, PorePressureLineSource, PorePressurePointSource, VLVDSource, MTSource
from pyrocko import gf
from pyrocko.fdsn import ws
from pyrocko.fdsn import station as fs
from pyrocko.guts import Object, Int, String
from silvertine.shakemap import shakemap_fwd
from pyrocko import moment_tensor as pmt
from numpy import random, where, cos, sin, arctan2, abs
from pyrocko.io import stationxml
import copy
import os
km = 1000.


def get_random_ellipse(n, x0, y0, mi, ma):

    xout = numpy.zeros(n)
    yout = numpy.zeros(n)

    nkeep = 0

    while nkeep < n:
        x=2*x0*(random.random(n-nkeep) - 0.5)
        y=2*y0*(random.random(n-nkeep) - 0.5)

        w,=where(((x/x0)**2 + (y/y0)**2) < 1)
        if w.size > 0:
            xout[nkeep:nkeep+w.size] = x[w]
            yout[nkeep:nkeep+w.size] = y[w]
            nkeep += w.size

    return xout, yout


def get_random_point_in_ellipse(milat, malat, milon, malon, miz, maz):
    coefs = (10, 2, 1)  # Coefficients in a0/c x**2 + a1/c y**2 + a2/c z**2 = 1
    rx, ry, rz = 1/num.sqrt(coefs)

    u = num.linspace(0, 2 * num.pi, 100)
    v = num.linspace(0, num.pi, 100)

    x = rx * num.outer(num.cos(u), num.sin(v))
    y = ry * num.outer(num.sin(u), num.sin(v))
    z = rz * num.outer(num.ones_like(u), num.cos(v))

    nkeep = 0

    while nkeep == 0:
        rand = num.random.uniform(-1, 1, 3)
        if rand[0] > num.min(x) and rand[0] < num.max(x) and rand[1] > num.min(y) and rand[1] < num.max(y) and rand[2] > num.min(z) and rand[2] < num.max(z):
            nkeep = 1
            x = rand[0]*(malat-milat) + milat
            y = rand[1]*(malon-milon) + milon
            z = rand[2]*(maz-miz) + miz

    return x, y, z


def rand(mi, ma):

    mi = float(mi)
    ma = float(ma)
    return random.random()*(ma-mi) + mi


def randlat(mi, ma):

    mi_ = 0.5*(math.sin(mi * math.pi/180.)+1.)
    ma_ = 0.5*(math.sin(ma * math.pi/180.)+1.)
    return math.asin(rand(mi_, ma_)*2.-1.)*180./math.pi


def xjoin(basepath, path):
    if path is None and basepath is not None:
        return basepath
    elif op.isabs(path) or basepath is None:
        return path
    else:
        return op.join(basepath, path)


class Path(String):
    pass


class HasPaths(Object):
    path_prefix = Path.T(optional=True)

    def __init__(self, *args, **kwargs):
        Object.__init__(self, *args, **kwargs)
        self._basepath = None
        self._parent_path_prefix = None

    def set_basepath(self, basepath, parent_path_prefix=None):
        self._basepath = basepath
        self._parent_path_prefix = parent_path_prefix
        for (prop, val) in self.T.ipropvals(self):
            if isinstance(val, HasPaths):
                val.set_basepath(
                    basepath, self.path_prefix or self._parent_path_prefix)

    def get_basepath(self):
        assert self._basepath is not None
        return self._basepath

    def expand_path(self, path, extra=None):
        assert self._basepath is not None

        if extra is None:
            def extra(path):
                return path

        path_prefix = self.path_prefix or self._parent_path_prefix

        if path is None:
            return None
        elif isinstance(path, basestring):
            return extra(
                op.normpath(xjoin(self._basepath, xjoin(path_prefix, path))))
        else:
            return [
                extra(
                    op.normpath(xjoin(self._basepath, xjoin(path_prefix, p))))
                for p in path]


def rand_source(event, SourceType="MT"):

    if SourceType == "MT":
        mt = MomentTensor.random_dc(magnitude=event.magnitude)
        event.moment_tensor = mt
        source = MTSource(
            lat=event.lat,
            lon=event.lon,
            north_shift=event.north_shift,
            east_shift=event.east_shift,
            depth=event.depth,
            m6=mt.m6(),
            time=event.time)

    if SourceType == "VLVD":
        mt = MomentTensor.random_dc(magnitude=event.magnitude)
        event.moment_tensor = mt
        source = VLVDSource(
            lat=event.lat,
            lon=event.lon,
            north_shift=event.north_shift,
            east_shift=event.east_shift,
            depth=event.depth,
            azimuth=event.strike,
            dip=mt.dip1,
            volume_change=num.random.uniform(0.1,10), # here synthetic volume change
            time=event.time,
            clvd_moment=mt.moment()) # ?

    if SourceType == "PorePressurePointSource":
        mt = MomentTensor.random_dc(magnitude=event.magnitude)
        event.moment_tensor = mt
        source = PorePressurePointSource(
            lat=event.lat,
            lon=event.lon,
            north_shift=event.north_shift,
            east_shift=event.east_shift,
            depth=event.depth,
            pp=num.random.uniform(1,1),  # here change in pa
            time=event.time) # ?

    if SourceType == "PorePressureLineSource":
        mt = MomentTensor.random_dc(magnitude=event.magnitude)
        event.moment_tensor = mt
        source = PorePressureLineSource(
            lat=event.lat,
            lon=event.lon,
            north_shift=event.north_shift,
            east_shift=event.east_shift,
            depth=event.depth,
            azimuth=event.strike,
            dip=mt.dip1,
            pp=num.random.uniform(1,1), # here change in pa
            time=event.time,
            length=num.random.uniform(1,20)*km) # scaling!)

    if SourceType == "Rectangular":
        length = num.random.uniform(1,20)*km
        width = num.random.uniform(1,20)*km
        strike, dip, rake = MomentTensor.random_strike_dip_rake()
        event.moment_tensor = MomentTensor(strike1=strike, dip1=dip, rake1=rake)
        source = RectangularSource(
            lat=event.lat,
            lon=event.lon,
            north_shift=event.north_shift,
            east_shift=event.east_shift,
            depth=event.depth,
            strike=strike,
            dip=dip,
            rake=rake,
            length=length,
            width=width,
            time=event.time,
            magnitude=event.magnitude)

    return source, event


def gen_stations(nstations=5,
                 latmin=-90., latmax=90.,
                 lonmin=-180., lonmax=180.):

    stations = []
    for i in range(nstations):
        sta = 'S%02i' % i
        s = model.Station('', sta, '',
            lat=randlat(latmin, latmax),
            lon=rand(lonmin, lonmax))

        stations.append(s)

    return stations


def gen_real_stations(tmin=util.stt('2014-01-01 16:10:00.000'),
                      tmax=util.stt('2014-01-01 16:39:59.000')):

    stations = []
    selection = [
        ('*', '*', '*', 'BH*', tmin, tmax),
    ]
    request_response = ws.station(
        site='iris', selection=selection, level='response')

    request_response.dump_xml(filename='stations.xml')
    sx = fs.load_xml(filename='stations.xml')

    for station in sx.get_pyrocko_stations():
        stations.append(station)

    return stations


def gen_random_tectonic_event(scenario_id, magmin=-0.5, magmax=3.,
                              depmin=5, depmax=10,
                              latmin=49.09586, latmax=49.25,
                              lonmin=8.0578, lonmax=8.20578,
                              timemin=util.str_to_time('2007-01-01 16:10:00.000'),
                              timemax=util.str_to_time('2020-01-01 16:10:00.000')):

    name = "scenario"+str(scenario_id)
    depth = rand(depmin, depmax)*km
    magnitude = rand(magmin, magmax)
    lat = randlat(latmin, latmax)
    lon = rand(lonmin, lonmax)
    time = rand(timemin, timemax)
    event = model.Event(name=name, lat=lat, lon=lon,
                        magnitude=magnitude, depth=depth,
                        time=time)

    return event


def gen_induced_event(scenario_id, magmin=1., magmax=3.,
                      depmin=3.5, depmax=14,
                      latmin=48.9586, latmax=49.2,
                      lonmin=8.0578, lonmax=8.1578,
                      radius_min=0.1, radius_max=20.2,
                      stress_drop_min=4.e06, stress_drop_max=17e6,
                      velocity=3000.,
                      timemin=util.str_to_time('2007-01-01 16:10:00.000'),
                      timemax=util.str_to_time('2020-01-01 16:10:00.000'),
                      well="insheim"):

    name = "scenario"+str(scenario_id)
    depth = rand(depmin, depmax)*km
    # source time function (STF) based on Brune source model, to get
    # spectra roughly realistic
    radius = randlat(radius_min, radius_max)
    stress_drop = rand(stress_drop_min, stress_drop_max)
    magnitude = float(pmt.moment_to_magnitude(
        16./7. * stress_drop * radius**3))
    rupture_velocity = 0.9 * velocity
    duration = 1.5 * radius / rupture_velocity

    if well is "insheim":
        # injection
        latmin = 49.149488448967226
        latmax = 49.15474362306716
        lonmin = 8.154522608466785
        lonmax = 8.160353517676578
        # extraction
        latmin = 49.15452255594506
        latmax = 49.16078760284185
        lonmin = 8.154355077151537
        lonmax = 8.160920669126998
    else:
        # injection
        latmin = 49.187282830892414
        latmax = 49.1876069158743
        lonmin = 8.110819065944598
        lonmax = 8.12401755104507
        # extraction
        latmin = 49.18755938553847
        latmax = 49.187687407341066
        lonmin = 8.123798045195171
        lonmax = 8.130836557156785

    depth_min = depmin
    depth_max = depmax
    lat, lon, depth_ell = get_random_point_in_ellipse(latmin, latmax,  lonmin,
                                                      lonmax, depth_min,
                                                      depth_max)
    time = rand(timemin, timemax)
    event = model.Event(name=name, lat=lat, lon=lon,
                        magnitude=magnitude, depth=depth,
                        time=time, duration=duration,
                        tags=["stress"+str(stress_drop)])

    return event


def gen_noise_events(targets, synthetics, engine, noise_sources=1, delay=40):

    noise_events = []
    for i in range(noise_sources):
        event = gen_random_tectonic_event(i, magmin=-1., magmax=0.)
        time = rand(event.time-delay, event.time+delay)
        mt = MomentTensor.random_dc(magnitude=event.magnitude)
        source = DCSource(
                            lat=event.lat,
                            lon=event.lon,
                            depth=event.depth,
                            strike=mt.strike1,
                            dip=mt.dip1,
                            rake=mt.rake1,
                            time=time,
                            magnitude=event.magnitude)
        response = engine.process(source, targets)
        noise_traces = response.pyrocko_traces()
        for tr, tr_noise in zip(synthetics, noise_traces):
            noise = trace.Trace(deltat=tr.deltat, tmin=tr.tmin,
                                ydata=tr_noise.ydata)
            choice_ev = num.random.choice(5, 1)
            if choice_ev == 0:
                tr.add(noise)
            else:
                pass
        noise_events.append(event)
    return(noise_events)


def save(synthetic_traces, event, stations, savedir, noise_events=False):
    model.dump_stations(stations, savedir+'model.txt')
    io.save(synthetic_traces, savedir+'traces.mseed')
    model.dump_events(event, savedir+'event.txt')
    model.dump_stations(stations, savedir+'stations.pf')
    st_xml = stationxml.FDSNStationXML.from_pyrocko_stations(stations,
                                                            add_flat_responses_from='M')
    st_xml.dump_xml(filename=savedir+'stations.xml')
    if noise_events is not False:
        model.dump_events(noise_events, savedir+'events_noise.txt')


def add_white_noise(synthetic_traces, scale=2e-8, scale_spectral='False'):

    if scale_spectral == 'True':
        noise_refrence = io.load('bfo_150901_0411.bhz')
        scale = num.max(num.abs(noise_refrence.spectrum()))
    for tr in synthetic_traces:

        nsamples = len(tr.ydata)
        randdata = num.random.normal(size=nsamples)*scale
        white_noise = trace.Trace(deltat=tr.deltat, tmin=tr.tmin,
                                  ydata=randdata)
        tr.add(white_noise)


def gen_white_noise(synthetic_traces, scale=2e-8, scale_spectral='False'):

    if scale_spectral == 'True':
        noise_refrence = io.load('bfo_150901_0411.bhz')
        scale = num.max(num.abs(noise_refrence.spectrum()))
    synthetic_traces_empty = copy.deepcopy(synthetic_traces)
    for tr in synthetic_traces_empty:
        max_scale = num.max(tr.ydata)
        min_scale = num.min(tr.ydata)
        scale = num.random.uniform(min_scale, max_scale)
        nsamples = len(tr.ydata)
        tr.ydata = tr.ydata*0.
        randdata = num.random.normal(size=nsamples)*scale
        tr.ydata = randdata

    return synthetic_traces_empty


def gen_dataset(scenarios, projdir, store_id, modelled_channel_codes, magmin,
                magmax, depmin, depmax, latmin, latmax, lonmin, lonmax,
                stations_file, gf_store_superdirs, shakemap=True,
                add_noise=True):
    if gf_store_superdirs is None:
        engine = gf.LocalEngine(use_config=True)
    else:
        engine = gf.LocalEngine(store_superdirs=[gf_store_superdirs])
    for scenario in range(scenarios):
        try:
            choice = num.random.choice(4, 1)
            choice = 1
            if choice == 0 or choice == 2:
                event = gen_random_tectonic_event(scenario, magmin=magmin,
                                                  magmax=magmax, depmin=depmin,
                                                  depmax=depmax, latmin=latmin,
                                                  latmax=latmax, lonmin=lonmin,
                                                  lonmax=lonmax)

                source, events = rand_source(event, SourceType='MT')

            if choice == 1:
                well_choice = num.random.choice(2, 1)
                if well_choice == 0:
                    well = "landau"
                else:
                    well = "insheim"
                event = gen_induced_event(scenario, magmin=magmin,
                                          magmax=magmax, depmin=depmin,
                                          depmax=depmax, latmin=latmin,
                                          latmax=latmax, lonmin=lonmin,
                                          lonmax=lonmax, well=well)

                source, events = rand_source(event, SourceType='MT')

            if choice == 3:
                event = gen_random_tectonic_event(scenario, magmin=0,
                                                  magmax=2.5, depmin=0.1,
                                                  depmax=0.2, latmin=49.16,
                                                  latmax=49.24, lonmin=7.98,
                                                  lonmax=8.1)

                source, events = rand_source(event, SourceType='MT')

            savedir = projdir + '/scenario_' + str(scenario) + '/'
            if not os.path.exists(savedir):
                os.makedirs(savedir)

            if stations_file is not None:
                stations = model.load_stations(projdir + "/" + stations_file)
                targets = []
                for st in stations:
                    for cha in st.channels:
                        target = Target(
                                lat=st.lat,
                                lon=st.lon,
                                store_id=store_id,
                                interpolation='multilinear',
                                quantity='displacement',
                                codes=st.nsl() + (cha.name,))
                        targets.append(target)

            else:
                targets = []
                for st in stations:
                    channels = modelled_channel_codes
                    for cha in channels:
                        target = Target(
                                lat=st.lat,
                                lon=st.lon,
                                store_id=store_id,
                                interpolation='multilinear',
                                quantity='displacement',
                                codes=st.nsl() + (cha,))
                    targets.append(target)
            if shakemap is True:
                shakemap_fwd.make_shakemap(engine, source, store_id,
                                           savedir, stations=stations)
            gen_loop = True
            response = engine.process(source, targets)
            synthetic_traces = response.pyrocko_traces()
            if choice == 2:
                synthetic_traces = gen_white_noise(synthetic_traces)
                event.tags=["no_event"]
            if add_noise is True and choice != 2:
                add_white_noise(synthetic_traces)
            noise_events = gen_noise_events(targets, synthetic_traces, engine)

            events = [event]
            save(synthetic_traces, events, stations, savedir,
                 noise_events=noise_events)
        except:
            pass

def silvertineScenario(projdir, scenarios=10, modelled_channel_codes='ENZ',
                       store_id='landau_100hz', magmin=-1., magmax=3.,
                       depmin=5, depmax=10,
                       latmin=49.0586, latmax=49.25,
                       lonmin=8.0578, lonmax=8.2,
                       stations_file=None, ratio_events=1,
                       shakemap=True, gf_store_superdirs=None):

    gen_dataset(scenarios, projdir, store_id, modelled_channel_codes, magmin,
                magmax, depmin, depmax, latmin, latmax, lonmin, lonmax,
                stations_file, gf_store_superdirs=gf_store_superdirs,
                shakemap=True)
