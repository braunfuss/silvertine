import random
import math
import numpy as num
from os.path import join as pjoin
import os.path as op
from pyrocko.moment_tensor import MomentTensor

from pyrocko import util, model, io, trace, config
from pyrocko.gf import Target, DCSource, RectangularSource, PorePressureLineSource, PorePressurePointSource, VLVDSource
from pyrocko import gf
from pyrocko.fdsn import ws
from pyrocko.fdsn import station as fs
from pyrocko.guts import Object, Int, String
import os
km = 1000.


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
        source = DCSource(
            lat=event.lat,
            lon=event.lon,
            north_shift=event.north_shift,
            east_shift=event.east_shift,
            depth=event.depth,
            strike=mt.strike1,
            dip=mt.dip1,
            rake=mt.rake1,
            time=event.time,
            magnitude=event.magnitude)

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
            volume_change=num.random.uniform(1,1), # here synthetic volume change
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


def gen_random_tectonic_event(scenario_id, magmin=1., magmax=3.,
              depmin=5, depmax=10,
              latmin=48.9586, latmax=49.3,
              lonmin=8.1578, lonmax=8.4578,
              timemin=util.str_to_time('2007-01-01 16:10:00.000'),
              timemax=util.str_to_time('2020-01-01 16:10:00.000')):

    name = scenario_id
    depth = rand(depmin, depmax)*km
    magnitude = rand(magmin, magmax)
    lat = randlat(latmin, latmax)
    lon = rand(lonmin, lonmax)
    time = rand(timemin, timemax)
    event = model.Event(name=name, lat=lat, lon=lon,
                        magnitude=magnitude, depth=depth,
                        time=time)

    return event


def gen_noise_events(targets, synthetics, engine, noise_sources=1, delay=40):

    noise_events = []
    for i in range(noise_sources):
        event = gen_event(i, magmin=-1.,magmax=0.)
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
            tr.add(noise)
        noise_events.append(event)
    return(noise_events)


def save(synthetic_traces, event, stations, savedir, noise_events=False):

        model.dump_stations(stations, savedir+'model.txt')
        io.save(synthetic_traces, savedir+'traces.mseed')
        model.dump_events(event, savedir+'event.txt')
        model.dump_stations(stations, savedir+'stations.pf')
        if noise_events is not False:
            model.dump_events(noise_events, savedir+'events_noise.txt')


def gen_white_noise(synthetic_traces,scale=2e-8, scale_spectral='False'):

    if scale_spectral == 'True':
        noise_refrence = io.load('bfo_150901_0411.bhz')
        scale = num.max(num.abs(noise_refrence.spectrum()))
    for tr in synthetic_traces:

        nsamples = len(tr.ydata)
        randdata = num.random.normal(size=nsamples)*scale
        white_noise = trace.Trace(deltat=tr.deltat, tmin=tr.tmin, ydata=randdata)
        tr.add(white_noise)


def gen_dataset(scenarios, projdir, store_id, modelled_channel_codes, magmin, magmax, depmin, depmax, latmin, latmax, lonmin, lonmax, stations_file):
    engine = gf.LocalEngine(store_superdirs=['/home/steinberg/seiger/grond/gf_stores'])
    for scenario in range(scenarios):

        choice = num.random.choice(2,1)
        if choice == 0:
            event = gen_random_tectonic_event(scenario, magmin=magmin,
                                              magmax=magmax, depmin=depmin,
                                              depmax=depmax, latmin=latmin,
                                              latmax=latmax, lonmin=lonmin,
                                              lonmax=lonmax)

            source, events = rand_source(event, SourceType='MT')

        if choice == 1:

            event = gen_induced_event(scenario, magmin=magmin,
                                              magmax=magmax, depmin=depmin,
                                              depmax=depmax, latmin=latmin,
                                              latmax=latmax, lonmin=lonmin,
                                              lonmax=lonmax)

            source, events = rand_source(event, SourceType='VLVD')

        savedir = projdir + 'scenario_' + str(scenario) + '/'
        if not os.path.exists(savedir):
            os.makedirs(savedir)

        if stations_file is not None:
            stations = model.load_stations(projdir + "/" + stations_file)
        targets = []
        for st in stations:
            channel_codes = modelled_channel_codes
            for cha in channel_codes:
                target = Target(
                        lat=st.lat,
                        lon=st.lon,
                        store_id=store_id,
                        interpolation='multilinear',
                        quantity='displacement',
                        codes=st.nsl() + (cha,))

                targets.append(target)
        gen_loop = True

        response = engine.process(source, targets)
        synthetic_traces = response.pyrocko_traces()
        gen_white_noise(synthetic_traces)
        noise_events = gen_noise_events(targets, synthetic_traces, engine)

        events = [event]
        save(synthetic_traces, events, stations, savedir,
             noise_events=noise_events)


def seigerScenario(projdir, scenarios=10, modelled_channel_codes='ENZ',
                   store_id='landau_100hz', magmin=1., magmax=3.,
                   depmin=5, depmax=10,
                   latmin=48.9586, latmax=49.3,
                   lonmin=8.1578, lonmax=8.4578,
                   stations_file=None, ratio_events=1):

    gen_dataset(scenarios, projdir, store_id, modelled_channel_codes, magmin, magmax, depmin, depmax, latmin, latmax, lonmin, lonmax, stations_file)
