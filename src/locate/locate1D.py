import random
import math
import numpy as num
from os.path import join as pjoin
import os.path as op
from pyrocko.moment_tensor import MomentTensor
from collections import OrderedDict
from pyrocko import util, model, io, trace, config
from pyrocko.gf import Target, DCSource, RectangularSource
from pyrocko import gf
from pyrocko.fdsn import ws
from pyrocko.fdsn import station as fs
from pyrocko.guts import Object, Int, String
import os
from pyrocko import orthodrome as ortho
from pyrocko import cake
from pyrocko import model
from pyrocko.gf import meta
import time as timesys
import scipy
from pyrocko.gui.pile_viewer import PhaseMarker, EventMarker
from ..util.ref_mods import *
from ..util.differential_evolution import differential_evolution
import ray
import psutil

num_cpus = psutil.cpu_count(logical=False)-1


def update_sources(params):
    for i, source in enumerate(sources):
        source.lat = float(params[0+4*i])
        source.lon = float(params[1+4*i])
        source.depth = float(params[2+4*i])
        source.time = float(source_dc.time - params[3+4*i])
    return sources


def update_depth(params):
    for i, source in enumerate(sources):
        source.depth = float(params[0+4*i])
    return sources


def picks_fit_parallel(params, events=None, sources=None, source_dc=None,
                       pyrocko_stations=None, interpolated_tts=None,
                       interpolate=True):
    misfits = 0.
    norms = 0.
    dists = []

    for ev, source in zip([events], [sources]):
        source.lat = float(params[0])
        source.lon = float(params[1])
        source.depth = float(params[2])
        source.time = float(source_dc.time - params[3])
        for st in ev["phases"]:
            for stp in pyrocko_stations:
                if stp.station == st["station"]:
                    phase = st["phase"]
                    dists_m = (ortho.distance_accurate50m(source.lat,
                                                          source.lon,
                                                          stp.lat,
                                                          stp.lon)+stp.elevation)

                    if phase == "Pg":
                        phase = "P<(moho)"
                    if phase == "pg":
                        phase = "p<(moho)"
                    if phase == "Sg":
                        phase = "S<(moho)"
                    if phase == "sg":
                        phase = "s<(moho)"
                    if phase == "PG":
                        phase = "P>(moho)"
                    if phase == "pG":
                        phase = "p>(moho)"
                    if phase == "SG":
                        phase = "S>(moho)"
                    if phase == "sG":
                        phase = "s>(moho)"
                    if phase == "P*":
                        phase = "P"
                    if phase == "S*":
                        phase = "S"
                    if phase == "SmS":
                        phase = 'Sv(moho)s'
                    if phase == "PmP":
                        phase = 'Pv(moho)p'
                    if phase == "Pn":
                        phase = 'Pv_(moho)p'
                    if phase == "Sn":
                        phase = 'Sv_(moho)s'
                    cake_phase = cake.PhaseDef(phase)

                    if interpolate is True:
                        coords = num.array((dists_m, num.tile(source.depth, 1))).T
                        onset = interpolated_tts[cake_phase.definition()].interpolate(coords)

                    elif interpolate is False:
                        tts = interpolated_tts[cake_phase.definition()]
                        absolute_difference_function = lambda list_value: abs(abs(list_value[:][0][1] - given_value[1]))+abs(list_value[:][0][0] - given_value[0])
                        d = [(k,v) for k,v in tts.f_values.items()]
                        given_value = (dists_m, source_depth)
                        onset = min(d, key=absolute_difference_function)[1]

                    tdiff = st["pick"]
                    try:
                        misfits += num.sqrt(num.sum((tdiff - onset)**2))
                        norms += num.sqrt(num.sum(onset**2))
                    except Exception:
                        pass
        misfit = num.sqrt(misfits**2 / norms**2)
    return misfit


def picks_fit(params, line=None, line2=None, line3=None, line4=None,
              interpolate=True):
    update_sources(params)
    global iiter
    misfits = 0.
    norms = 0.
    dists = []
    iter_event = 0
    iter_new = iiter + 1
    iiter = iter_new
    for ev, source in zip(ev_dict_list, sources):
        for st in ev["phases"]:
            for stp in pyrocko_stations[iter_event]:
                if stp.station == st["station"]:
                    phase = st["phase"]
                    dists_m = (ortho.distance_accurate50m(source.lat,
                                                          source.lon,
                                                          stp.lat,
                                                          stp.lon)+stp.elevation)

                    if phase == "Pg":
                        phase = "P<(moho)"
                    if phase == "pg":
                        phase = "p<(moho)"
                    if phase == "Sg":
                        phase = "S<(moho)"
                    if phase == "sg":
                        phase = "s<(moho)"
                    if phase == "PG":
                        phase = "P>(moho)"
                    if phase == "pG":
                        phase = "p>(moho)"
                    if phase == "SG":
                        phase = "S>(moho)"
                    if phase == "sG":
                        phase = "s>(moho)"
                    if phase == "P*":
                        phase = "P"
                    if phase == "S*":
                        phase = "S"
                    if phase == "SmS":
                        phase = 'Sv(moho)s'
                    if phase == "PmP":
                        phase = 'Pv(moho)p'
                    if phase == "Pn":
                        phase = 'Pv_(moho)p'
                    if phase == "Sn":
                        phase = 'Sv_(moho)s'
                    cake_phase = cake.PhaseDef(phase)

                    if interpolate is True:
                        coords = num.array((dists_m, num.tile(source.depth, 1))).T
                        onset = interpolated_tts[cake_phase.definition()].interpolate(coords)
                    elif interpolate is False:
                        tts = interpolated_tts[cake_phase.definition()]
                        absolute_difference_function = lambda list_value: abs(abs(list_value[:][0][1] - given_value[1]))+abs(list_value[:][0][0] - given_value[0])
                        d = [(k, v) for k, v in tts.f_values.items()]
                        given_value = (dists_m, source_depth)
                        onset = min(d, key=absolute_difference_function)[1]
                    tdiff = st["pick"]
                    try:
                        misfits += num.sqrt(num.sum((tdiff - onset)**2))
                        norms += num.sqrt(num.sum(onset**2))
                    except Exception:
                        pass
        if line2:
            data = {
                'y': [source.lat],
                'x': [source.lon],
            }
            line2.data_source.stream(data)
        if line3:
            data = {
                'y': [source.lat],
                'x': [source.depth],
            }
            line3.data_source.stream(data)
        if line4:
            data = {
                'y': [source.lon],
                'x': [source.depth],
            }
            line4.data_source.stream(data)
    misfit = num.sqrt(misfits**2 / norms**2)
    iter_event = iter_event + 1
    if line:
        data = {
            'y': [misfit],
            'x': [iiter],
        }
        line.data_source.stream(data)

    return misfit


def depth_fit(params, line=None):
    update_depth(params)
    global iiter
    misfits = 0.
    norms = 0.
    dists = []
    iter_event = 0
    iter_new = iiter + 1
    iiter = iter_new
    for ev, source in zip(ev_dict_list, sources):
        for st in ev["phases"]:
            for stp in pyrocko_stations[iter_event]:
                if stp.station == st["station"]:
                    phase = st["phase"]
                    if phase == "Pg":
                        phase = "P<(moho)"
                    if phase == "pg":
                        phase = "p<(moho)"
                    if phase == "Sg":
                        phase = "S<(moho)"
                    if phase == "sg":
                        phase = "s<(moho)"
                    if phase == "PG":
                        phase = "P>(moho)"
                    if phase == "pG":
                        phase = "p>(moho)"
                    if phase == "SG":
                        phase = "S>(moho)"
                    if phase == "sG":
                        phase = "s>(moho)"
                    if phase == "P*":
                        phase = "P"
                    if phase == "S*":
                        phase = "S"
                    cake_phase = cake.PhaseDef(phase)
                    phase_list = [cake_phase]
                    dists = (ortho.distance_accurate15nm(source.lat,
                                                         source.lon,
                                                         stp.lat,
                                                         stp.lon)+stp.elevation)*cake.m2d

                    for i, arrival in enumerate(mod.arrivals([dists],
                                                phases=phase_list,
                                                zstart=source.depth)):

                        tdiff = st["pick"]

                        used_phase = arrival.used_phase()
                        if phase == used_phase.given_name() or phase[0] == used_phase.given_name()[0]:
                            misfits += num.sqrt(num.sum((tdiff - arrival.t)**2))
                            norms += num.sqrt(num.sum(arrival.t**2))

    misfit = num.sqrt(misfits**2 / norms**2)
    iter_event = iter_event + 1
    if line:
        data = {
            'y': [misfit],
            'x': [iiter],
        }
        line.data_source.stream(data)

    return misfit


def load_synthetic_test(n_tests, scenario_folder, nstart=1, nend=None):
    events = []
    stations = []
    for i in range(nstart, n_tests):
        print("%s/scenario_%s/event.txt" % (scenario_folder, i))

        events.append(model.load_events("%s/scenario_%s/event.txt" % (scenario_folder, i))[0])
        stations.append(model.load_stations("%s/scenario_%s/stations.pf" % (scenario_folder, i)))
    return events, stations


def synthetic_ray_tracing_setup(events, stations, mod):
    k = 0
    for evs, stats in zip(events, stations):
        ev_dict_list.append(dict(id="%s" % k, time=evs.time, lat=evs.lat,
                            lon=evs.lon, mag=evs.magnitude, mag_type="syn",
                            source="syn", phases=[], depth=[], rms=[],
                            error_h=[], error_z=[]))
    stations = stations[0]
    for nev, ev in enumerate(ev_dict_list):
        phase_markers = []
        stations_event = []
        hypo_in = []
        station_phase_to_nslc = {}
        dists = []
        for st in stations:
            dist = (ortho.distance_accurate15nm(ev["lat"], ev["lon"],
                                                st.lat, st.lon)+st.elevation)*cake.m2d
            dists.append(dist)
            for i, arrival in enumerate(mod.arrivals([dist],
                                        phases=phase_list,
                                        zstart=events[nev].depth)):

                event_arr = model.event.Event(lat=ev["lat"], lon=ev["lon"],
                                              time=ev["time"],
                                              catalog=ev["source"],
                                              magnitude=ev["mag"])
                used_phase = arrival.used_phase()
                used_phase = used_phase.given_name()
                if used_phase == "p<(moho)":
                    used_phase = "p"
                if used_phase == "P<(moho)":
                    used_phase = "Pg"
                if used_phase == "s<(moho)":
                    used_phase = "sg"
                if used_phase == "S<(moho)":
                    used_phase = "Sg"
                if used_phase == "p":
                    used_phase = "p"
                tarr = events[nev].time+arrival.t
                phase_markers.append(PhaseMarker(["0", st.station], tarr,
                                                 tarr, 0,
                                                 phasename=used_phase,
                                                 event_hash=ev["id"],
                                                 event=events[nev]))
                ev["phases"].append(dict(station=st.station,
                                         phase=used_phase,
                                         pick=arrival.t))
                times.append(tarr)

        ev_list = []
        ev_list.append(phase_markers)


def load_data(data_folder=None, nevent=0):
    from silvertine.util import silvertine_meta
    ev_dict_list, picks = silvertine_meta.load_ev_dict_list(path=data_folder,
                                                            nevent=nevent)
    ev_list_picks, stations, ev_list, ev_dict_list = silvertine_meta.convert_phase_picks_to_pyrocko(ev_dict_list, picks, nevent=nevent)
    if nevent is None:
        return ev_list, stations, ev_dict_list, ev_list_picks
    else:
        return ev_list, stations, ev_dict_list, ev_list_picks


@ray.remote
def optim_parallel(event, sources, bounds, stations, interpolated_tts,
                   result_sources, result_events, name):
    source_dc = sources
    try:
        result = differential_evolution(
            picks_fit_parallel,
            args=[event, sources, source_dc, stations, interpolated_tts],
            bounds=tuple(bounds.values()),
            maxiter=25,
            seed=123,
            tol=0.0001)
        params_x = result.x
        source = gf.DCSource(
            lat=float(params_x[0]),
            lon=float(params_x[1]),
            depth=float(params_x[2]),
            time=float(params_x[3]))
        result_sources.append(source)
        event_result = model.event.Event(lat=source.lat, lon=source.lon,
                                         time=source_dc.time+source.time,
                                         depth=source.depth,
                                         tags=[str(event["id"])],
                                         extras={"misfit": result.fun})
        result_events.append(event)
        file = open(name, 'a+')
        event_result.dump(file)
        file.close()
    except Exception:
        pass


def solve(show=False, n_tests=1, scenario_folder="scenarios",
          optimize_depth=False, scenario=True, data_folder="data",
          parallel=True, adress=None, interpolate=True, mod_name="insheim",
          singular=False):
    global ev_dict_list, times, phase_list, km, mod, pyrocko_stations, bounds, sources, source_dc, iiter, interpolated_tts, result_sources, result_events

    km = 1000.
    iiter = 0
    if mod_name == "insheim":
        mod = insheim_layered_model()
    result_sources = []
    result_events = []
    t = timesys.time()
    sources = []
    if parallel is False and singular is False:
        bounds = OrderedDict()
    else:
        bounds_list = []
    if scenario is True:
        test_events, pyrocko_stations = load_synthetic_test(n_tests, scenario_folder)
    else:
        test_events, pyrocko_stations, ev_dict_list, ev_list_picks = load_data(data_folder, nevent=n_tests)
    ev_iter = 0
    for ev in test_events:
        if parallel is False and singular is False:
            bounds.update({'lat%s' % ev_iter: (ev.lat-0.5, ev.lat+0.5)})
            bounds.update({'lon%s' % ev_iter: (ev.lon-0.5, ev.lon+0.5)})
            if ev.depth is None:
                bounds.update({'depth%s' % ev_iter: (0*km, 15*km)})
            elif ev.depth >= 3*km:
#                bounds.update({'depth%s' % ev_iter: (ev.depth-3*km, ev.depth+3*km)})
                bounds.update({'depth%s' % ev_iter: (0*km, 15*km)})

            else:
#                bounds.update({'depth%s' % ev_iter: (0., ev.depth+3*km)})
                bounds.update({'depth%s' % ev_iter: (0*km, 15*km)})

            bounds.update({'timeshift%s' % ev_iter: (-0.5, 0.5)})
        else:
            bounds = OrderedDict()
            bounds.update({'lat%s' % ev_iter: (ev.lat-0.2, ev.lat+0.2)})
            bounds.update({'lon%s' % ev_iter: (ev.lon-0.2, ev.lon+0.2)})
            bounds.update({'depth%s' % ev_iter: (0*km, 15*km)})
            bounds.update({'timeshift%s' % ev_iter: (-0.1, 0.1)})
            bounds_list.append(bounds)

        ev_iter = ev_iter+1

        time = ev.time
        source_dc = DCSource(
            lat=ev.lat,
            lon=ev.lon,
            depth=ev.depth,
            time=time,
            magnitude=ev.magnitude)

        # start solution lat, lon
        source = gf.DCSource(
            lat=ev.lat,
            lon=ev.lon,
            depth=ev.depth,
            time=time,
            magnitude=ev.magnitude)
        sources.append(source)
    times = []

    inp_cake = mod

    # P-phase definitions
    Pg = cake.PhaseDef('P<(moho)')
    pg = cake.PhaseDef('p<(moho)')
    PG = cake.PhaseDef('P'+'\\')
    pG = cake.PhaseDef('p'+'\\')
    p = cake.PhaseDef('p')
    pS = cake.PhaseDef('pS')
    PP = cake.PhaseDef('PP')
    P = cake.PhaseDef('P')
    pP = cake.PhaseDef('pP')
    pPv3pP = cake.PhaseDef('pPv3pP')
    pPv3pPv3pP = cake.PhaseDef('pPv3pPv3pP')
    PmP = cake.PhaseDef('Pv(moho)p')
    Pn = cake.PhaseDef('Pv_(moho)p')

    # S-phase Definitions
    S = cake.PhaseDef('S')
    s = cake.PhaseDef('s')
    sP = cake.PhaseDef('sP')
    SP = cake.PhaseDef('SP')
    SS = cake.PhaseDef('SS')
    SmS = cake.PhaseDef('Sv(moho)s')
    Sg = cake.PhaseDef('S<(moho)')
    sg = cake.PhaseDef('s<(moho)')
    SG = cake.PhaseDef('S>(moho)')
    sG = cake.PhaseDef('s>(moho)')
    sS = cake.PhaseDef('sS')
    Sn = cake.PhaseDef('Sv_(moho)s')

    sSv3sS = cake.PhaseDef('sSv3sS')
    sSv3sSv3sS = cake.PhaseDef('sSv3sSv3sS')

#    phase_list = [P, p, Sg, sg, pg, S, s, Pg, PG, pG, SS, PP, pS, SP, sP, sS,
#                  pP, pPv3pP, pPv3pPv3pP, sSv3sS, sSv3sSv3sS, SmS, PmP]

    phase_list = [P, p, Sg, S, s, Pg, pg, sg, SmS, PmP, Sn, Pn]

    if scenario is True:
        ev_dict_list = []
        synthetic_ray_tracing_setup(test_events, pyrocko_stations, inp_cake)
    import cProfile
    import pstats
    pr = cProfile.Profile()
    pr.enable()

    from silvertine.util import ttt
    # Calculate Traveltime tabel for each phase (parallel)
    interpolated_tts, missing = ttt.load_sptree(phase_list, mod_name)
    calculated_ttt = False

    if len(missing) != 0:
        print("Calculating travel time look up table,\
                this may take some time.")
        ttt.calculate_ttt_parallel(pyrocko_stations, mod, missing, mod_name,
                                   adress=adress)
        interpolated_tts_new, missing = ttt.load_sptree(phase_list, mod_name)
        interpolated_tts = {**interpolated_tts, **interpolated_tts_new}
        calculated_ttt = True
    if show is True:
        from bokeh.client import push_session, show_session
        from bokeh.io import curdoc
        from bokeh.plotting import figure
        from bokeh.layouts import gridplot, column
        from bokeh.models import Button
        f1 = figure(title='SciPy Optimisation Progress',
                    x_axis_label='# Iteration',
                    y_axis_label='Misfit',
                    plot_width=1000,
                    plot_height=500)
        p1 = f1.scatter([], [])
        f2 = figure(title='Map',
                    x_axis_label='Lat',
                    y_axis_label='Lon',
                    plot_width=500,
                    plot_height=500)
        p2 = f2.scatter([], [])

        f3 = figure(title='Lat with depth',
                    x_axis_label='Depth [m]',
                    y_axis_label='Lat',
                    plot_width=500,
                    plot_height=500)
        p3 = f3.scatter([], [])

        f4 = figure(title='Lat with depth',
                    x_axis_label='Depth [m]',
                    y_axis_label='Lon',
                    plot_width=500,
                    plot_height=500)
        p4 = f4.scatter([], [])

        def button_callback(a, b):
            new_data = dict()

        button = Button(label="Update")
        curdoc().add_root(gridplot([[f1]]))
        curdoc().add_root(gridplot([[f3, f2], [None, f4]]))
        session = push_session(curdoc())
        session.show()

        if singular is False:
            result = differential_evolution(
                picks_fit,
                args=[p1, p2, p3, p4, interpolate],
                bounds=tuple(bounds.values()),
                seed=123,
                maxiter=25,
                tol=0.0001,
                callback=lambda a, convergence: curdoc().add_next_tick_callback(button_callback(a, convergence)))

            sources = update_sources(result.x)
            if optimize_depth is True:
                bounds = OrderedDict()
                for source in sources:
                    bounds.update({'depth%s' % ev_iter: (source.depth-300.,
                                                         source.depth+300.)})
                result = differential_evolution(
                    depth_fit,
                    args=[plot],
                    bounds=tuple(bounds.values()),
                    seed=123,
                    maxiter=6,
                    tol=0.001,
                    callback=lambda a, convergence: curdoc().add_next_tick_callback(button_callback(a, convergence)))
                for source in sources:
                    sources = update_depth(result.x)
            for i, source in enumerate(sources):
                result_sources.append(source)
                event = model.event.Event(lat=source.lat, lon=source.lon,
                                          time=source.time, magnitude=source.magnitude,
                                          tags=[str(ev_dict_list[i]["id"])],
                                          extras={"misfit":result.fun})
                result_events.append(event)
        else:
            ev_dict_list_copy = ev_dict_list.copy()
            sources_copy = sources.copy()
            bounds_list_copy = bounds_list.copy()
            pyrocko_stations_copy = pyrocko_stations.copy()
            for i in range(len(ev_dict_list)):
                ev_dict_list = [ev_dict_list_copy[i]]
                sources = [sources_copy[i]]
                bounds = bounds_list_copy[i]
                pyrocko_stations = [pyrocko_stations_copy[i]]
                result = differential_evolution(
                    picks_fit,
                    args=[p1, p2, p3, p4, interpolate],
                    bounds=tuple(bounds.values()),
                    seed=123,
                    maxiter=25,
                    tol=0.0001,
                    callback=lambda a, convergence: curdoc().add_next_tick_callback(button_callback(a, convergence)))
                params_x = result.x
                source = gf.DCSource(
                    lat=float(params_x[0]),
                    lon=float(params_x[1]),
                    depth=float(params_x[2]),
                    time=float(params_x[3]))
                result_sources.append(source)
                event_result = model.event.Event(lat=source.lat, lon=source.lon,
                                                 time=source.time,
                                                 depth=source.depth,
                                                 extras={"misfit":result.fun},
                                                 tags=ev_dict_list[0]["id"])
                result_events.append(event)
                sources = update_sources(result.x)
                if optimize_depth is True:
                    bounds = OrderedDict()
                    for source in sources:
                        bounds.update({'depth%s' % ev_iter: (source.depth-300.,
                                                             source.depth+300.)})
                    result = differential_evolution(
                        depth_fit,
                        args=[plot],
                        bounds=tuple(bounds.values()),
                        seed=123,
                        maxiter=6,
                        tol=0.001,
                        callback=lambda a, convergence: curdoc().add_next_tick_callback(button_callback(a, convergence)))
                    for source in sources:
                        sources = update_depth(result.x)
                for source in sources:
                    result_sources.append(source)

    else:

        if parallel is True or parallel is "True":
            name = "scenarios/events.txt"
            file = open(name, 'w+')
            file.close()
            if calculated_ttt is False:
                ray.init(num_cpus=num_cpus-1, memory=28500 * 1024 * 1024)
            event_dict = []
            ray.get([optim_parallel.remote(ev_dict_list[i], sources[i], bounds_list[i], pyrocko_stations[i], interpolated_tts, result_sources, result_events, name) for i in range(len(ev_dict_list))])
            result = None
            source = None
        else:
            if singular is True:
                ev_dict_list_copy = ev_dict_list.copy()
                sources_copy = sources.copy()
                bounds_list_copy = bounds_list.copy()
                pyrocko_stations_copy = pyrocko_stations.copy()
                for i in range(len(ev_dict_list)):
                    ev_dict_list = [ev_dict_list_copy[i]]
                    sources = [sources_copy[i]]
                    bounds = bounds_list_copy[i]
                    pyrocko_stations = [pyrocko_stations_copy[i]]
                    result = differential_evolution(
                        picks_fit,
                        args=[],
                        bounds=tuple(bounds.values()),
                        seed=123,
                        maxiter=25,
                        tol=0.0001)
                    params_x = result.x
                    source = gf.DCSource(
                        lat=float(params_x[0]),
                        lon=float(params_x[1]),
                        depth=float(params_x[2]),
                        time=float(params_x[3]))
                    result_sources.append(source)
                    event_result = model.event.Event(lat=source.lat, lon=source.lon,
                                                     time=source.time,
                                                     depth=source.depth,
                                                     extras={"misfit":result.fun},
                                                     tags=ev_dict_list[0]["id"])
                    result_events.append(event)
                    sources = update_sources(result.x)
                    if optimize_depth is True:
                        bounds = OrderedDict()
                        for source in sources:
                            bounds.update({'depth%s' % ev_iter: (source.depth-300.,
                                                                 source.depth+300.)})
                        result = differential_evolution(
                            depth_fit,
                            args=[plot],
                            bounds=tuple(bounds.values()),
                            seed=123,
                            maxiter=6,
                            tol=0.001,
                            callback=lambda a, convergence: curdoc().add_next_tick_callback(button_callback(a, convergence)))
                        for source in sources:
                            sources = update_depth(result.x)
                    for source in sources:
                        result_sources.append(source)
            else:
                result = differential_evolution(
                    picks_fit,
                    args=[],
                    bounds=tuple(bounds.values()),
                    maxiter=25,
                    seed=123,
                    tol=0.0001)

                sources = update_sources(result.x)
                for source in sources:
                    source.regularize()

                if optimize_depth is True:
                    bounds = OrderedDict()
                    for source in sources:
                        bounds.update({'depth%s' % ev_iter: (source.depth-300.,
                                                             source.depth+300.)})
                    result = differential_evolution(
                        depth_fit,
                        args=[],
                        bounds=tuple(bounds.values()),
                        seed=123,
                        maxiter=6,
                        tol=0.001,
                        callback=lambda a, convergence: curdoc().add_next_tick_callback(button_callback(a, convergence)))
                    for source in sources:
                        sources = update_depth(result.x)
                for source in sources:
                        result_sources.append(source)
                        event = model.event.Event(lat=source.lat, lon=source.lon,
                                                  time=source.time, magnitude=source.magnitude)
                        result_events.append(event)
    pr.disable()
    filename = 'profile.prof'
    pr.dump_stats(filename)
    for source in sources:
        print(source)
    model.dump_events(result_events, scenario_folder+"result_events.pf")
    return result, sources
