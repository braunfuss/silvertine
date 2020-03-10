from .beam_stack import *
import os
from collections import defaultdict
from pyrocko.fdsn import ws, station as fdsnstation
from pyrocko.gf import store
from pyrocko import model
from pyrocko import cake
from pyrocko import io
from pyrocko import orthodrome as ortho
from pyrocko.guts import Object, String, Float, List, Dict
import logging
from ..util.ref_mods import *
pjoin = os.path.join


class CakeTiming(Object):
    '''Calculates and caches phase arrivals.
    :param fallback_time: returned, when no phase arrival was found for the
                        given depth-distance-phase-selection-combination

    E.g.:
    definition = 'first(p,P)-20'
    CakeTiming(definition)'''
    phase_selection = String.T()
    fallback_time = Float.T(optional=True)

    def __init__(self, phase_selection, fallback_time=None):
        self.arrivals = defaultdict(dict)
        self.fallback_time = fallback_time
        self.which = None
        self.phase_selection = phase_selection
        _phase_selection = phase_selection
        if '+' in _phase_selection:
            _phase_selection, self.offset = _phase_selection.split('+')
            self.offset = float(self.offset)
        elif '-' in _phase_selection:
            _phase_selection, self.offset = _phase_selection.split('-')
            self.offset = float(self.offset)
            self.offset = -self.offset

        if 'first' in _phase_selection:
            self.which = 'first'
        if 'last' in _phase_selection:
            self.which = 'last'
        if self.which:
            _phase_selection = self.strip(_phase_selection)

        self.phases = _phase_selection.split('|')

    def return_time(self, ray):
        if ray is None:
            return self.fallback_time
        else:
            return ray.t + self.offset

    def t(self, mod, z_dist, get_ray=False):
        ''':param phase_selection: phase names speparated by vertical bars
        :param z_dist: tuple with (depth, distance)
        '''
        z, dist = z_dist
        if (dist, z) in self.arrivals.keys():
            return self.return_time(self.arrivals[(dist, z)])

        phases = [cake.PhaseDef(pid) for pid in self.phases]
        arrivals = mod.arrivals(
            distances=[dist*cake.m2d], phases=phases, zstart=z)
        if arrivals == []:
            logger.warn(
                'no phase at d=%s, z=%s. (return fallback time)' % (dist, z))
            want = None
        else:
            want = self.phase_selector(arrivals)
        self.arrivals[(dist, z)] = want
        if get_ray:
            return want
        else:
            return self.return_time(want)

    def phase_selector(self, _list):
        if self.which == 'first':
            return min(_list, key=lambda x: x.t)
        if self.which == 'last':
            return max(_list, key=lambda x: x.t)

    def strip(self, ps):
        ps = ps.replace(self.which, '')
        ps = ps.rstrip(')')
        ps = ps.lstrip('(')
        return ps


class Timings(Object):
    timings = List.T(CakeTiming.T())

    def __init__(self, timings):
        self.timings = timings


def beam(scenario_folder, n_tests=1, show=False):
    nstart = 8
    array_centers = []

    events = []
    stations = []
    mod = insheim_layered_model()

    for i in range(nstart, n_tests):
        print("%s/scenario_%s/event.txt" % (scenario_folder, i))

        events.append(model.load_events("%s/scenario_%s/event.txt" % (scenario_folder, i))[0])
        stations.append(model.load_stations("%s/scenario_%s/stations.pf" % (scenario_folder, i)))
        traces = io.load(pjoin("%sscenario_%s/" % (scenario_folder, i), 'traces.mseed'))

        event = events[0]
        stations = stations[0]
        min_dist = min(
            [ortho.distance_accurate50m(s, event) for s in stations])
        max_dist = max(
            [ortho.distance_accurate50m(s, event) for s in stations])
        tmin = CakeTiming(phase_selection='first(p|P|PP)-10', fallback_time=0.001)
        tmax = CakeTiming(phase_selection='first(p|P|PP)+52', fallback_time=1000.)
        timing=(tmin, tmax)
        tstart = timing[0].t(mod, (event.depth, min_dist))
        tend = timing[1].t(mod, (event.depth, max_dist))

        normalize = True
        bf = BeamForming(stations, traces, normalize=normalize)
        bf.process(event=event,
                   timing=tmin,
                   fn_dump_center=pjoin("%sscenario_%s/" % (scenario_folder, i), 'array_center.pf'),
                   fn_beam=pjoin("%sscenario_%s/" % (scenario_folder, i), 'beam.mseed'),
                   station="INS")
        if show is True:
            bf.plot(fn=pjoin("%sscenario_%s/" % (scenario_folder, i), 'beam_shifts.png'))

        array_centers.append(bf.station_c)


def process(args, scenario_folder, n_tests=1, show=True):
    nstart = 8
    array_centers = []
    from .guesstimate_depth_v02 import PlotSettings, plot

    events = []
    stations = []
    mod = insheim_layered_model()

    for i in range(nstart, nstart+1):
        i = 8
        scenario_folder = "scenarios/"
        print("%s/scenario_%s/event.txt" % (scenario_folder, i))

        events.append(model.load_events("%s/scenario_%s/event.txt" % (scenario_folder, i))[0])
        stations.append(model.load_stations("%s/scenario_%s/stations.pf" % (scenario_folder, i)))
        traces = io.load(pjoin("%sscenario_%s/" % (scenario_folder, i), 'traces.mseed'))

        event = events[0]
        stations = stations[0]
        min_dist = min(
            [ortho.distance_accurate50m(s, event) for s in stations])
        max_dist = max(
            [ortho.distance_accurate50m(s, event) for s in stations])
        tmin = CakeTiming(phase_selection='first(p|P|PP)-10', fallback_time=0.001)
        tmax = CakeTiming(phase_selection='first(p|P|PP)+52', fallback_time=1000.)
        timing=(tmin, tmax)

        fns = ['.']


        array_id = "INS"


        settings_fn = pjoin("%sscenario_%s/" % (scenario_folder, i), 'plot_settings.yaml')
        settings = PlotSettings.from_argument_parser(args)

        if not settings.trace_filename:
            settings.trace_filename = pjoin("%sscenario_%s/" % (scenario_folder, i), 'beam.mseed')
        if not settings.station_filename:
            fn_array_center = pjoin("%sscenario_%s/" % (scenario_folder, i), 'array_center.pf')
            settings.station_filename = fn_array_center
            station = model.load_stations(fn_array_center)

    #    settings.store_id = '%s_%s_%s' % (array_id,
    #                                      target_crust,
    #                                      source_crust)

            settings.store_id = 'landau_100hz'


        settings.event_filename = pjoin("%sscenario_%s/" % (scenario_folder, i), "event.txt")
        settings.save_as = pjoin("%sscenario_%s/" % (scenario_folder, i), "depth_%(array-id)s.png")
        plot(settings)
        if args.overwrite_settings:
            settings.dump(filename=settings_fn)
        if show is True:
            plt.show()
