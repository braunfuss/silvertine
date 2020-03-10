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
import time as timesys
import scipy
from pyrocko.gui.pile_viewer import PhaseMarker, EventMarker
from ..util.ref_mods import *
from ..util.differential_evolution import differential_evolution


def update_sources(params):
    for i, source in enumerate(sources):
        source.lat = float(params[0+4*i])
        source.lon = float(params[1+4*i])
        source.depth = float(params[2+4*i])
        source.time = float(source_dc.time - params[3+4*i])
        print(source)
    return sources


def update_depth(params):
    for i, source in enumerate(sources):
        source.depth = float(params[0+4*i])
        print(source)
    return sources


def picks_fit(params, line=None):
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
                    dists = (ortho.distance_accurate15nm(source.lat, source.lon,
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



def depth_fit(params, line=None):
    update_depth(params)
    global iiter
    misfits = 0.
    norms = 0.
    dists = []
    iter_event = 0
    iter_new = iiter +1
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
                    dists = (ortho.distance_accurate15nm(source.lat, source.lon,
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


def load_synthetic_test(n_tests, scenario_folder, nstart=8, nend=None):
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
                    tarr=events[nev].time+arrival.t
                    phase_markers.append(PhaseMarker(["0", st.station], tarr,
                                                     tarr, 0,
                                                     phasename=used_phase,
                                                     event_hash=ev["id"],
                                                     event=events[nev]))
                    ev["phases"].append(dict(station=st.station,
                                             phase=used_phase,
                                             pick=arrival.t))
                    times.append(tarr)

        ev_list=[]
        ev_list.append(phase_markers)


def calculate_ttt()


def solve(show=False, n_tests=1, scenario_folder="scenarios", optimize_depth=False):
    global ev_dict_list, times, phase_list, km, mod, pyrocko_stations, bounds, sources, source_dc, iiter

    km = 1000.
    iiter = 0
    mod = insheim_layered_model()

    t = timesys.time()
    sources = []
    bounds = OrderedDict()
    test_events, pyrocko_stations = load_synthetic_test(n_tests, scenario_folder)
    ev_iter = 0
    for ev in test_events:
        bounds.update({'lat%s' %ev_iter:(ev.lat-0.4, ev.lat+0.4)})
        bounds.update({'lon%s'%ev_iter:(ev.lon-0.4, ev.lon+0.4)})
        bounds.update({'depth%s'%ev_iter:(0.1*km, 7.*km)})
        bounds.update({'timeshift%s'%ev_iter:(-0.1, 0.1)})
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
            lat=source_dc.lat,
            lon=source_dc.lon)
        sources.append(source)
    ev_dict_list = []
    times = []

    inp_cake = mod

    # P-phase definitions
    Pg=cake.PhaseDef('P<(moho)')
    pg=cake.PhaseDef('p<(moho)')
    PG=cake.PhaseDef('P'+'\\')
    pG=cake.PhaseDef('p'+'\\')
    p=cake.PhaseDef('p')
    pS=cake.PhaseDef('pS')
    PP=cake.PhaseDef('PP')
    P=cake.PhaseDef('P')

    # S-phase Definitions
    Phase_S=cake.PhaseDef('S')
    Phase_s=cake.PhaseDef('s')
    sP=cake.PhaseDef('sP')
    SP=cake.PhaseDef('SP')
    SS=cake.PhaseDef('SS')
    Sg=cake.PhaseDef('S<(moho)')
    sg=cake.PhaseDef('s<(moho)')
    SG=cake.PhaseDef('S>(moho)')
    sG=cake.PhaseDef('s>(moho)')
    phase_list=[P, p, Sg, sg, pg, Phase_S, Phase_s, Pg, PG, pG, SS, PP, pS, SP, sP]

    synthetic_ray_tracing_setup(test_events, pyrocko_stations, inp_cake)
    import cProfile, pstats
    pr = cProfile.Profile()
    pr.enable()
    if show is True:
        from bokeh.client import push_session, show_session
        from bokeh.io import curdoc
        from bokeh.plotting import figure
        from bokeh.layouts import column
        from bokeh.models import Button
        f = figure(title='SciPy Optimisation Progress',
                   x_axis_label='# Iteration',
                   y_axis_label='Misfit',
                   plot_width=1200,
                   plot_height=500)
        plot = f.scatter([], [])
        ds = plot.data_source

        def button_callback(a, b):
            new_data = dict()

        button = Button(label="Update")

        curdoc().add_root(column(f, button))
        session = push_session(curdoc())
        session.show()

        result = differential_evolution(
            picks_fit,
            args=[plot],
            bounds=tuple(bounds.values()),
            seed=123,
            maxiter=25,
            tol=0.0001,
            callback=lambda a, convergence: curdoc().add_next_tick_callback(button_callback(a, convergence)))

        sources = update_sources(result.x)
        for source in sources:
            source.regularize()
        if optimize_depth is True:
            bounds = OrderedDict()
            for source in sources:
                bounds.update({'depth%s'%ev_iter:(source.depth-300., source.depth+300.)})
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

    else:
        result = differential_evolution(
            picks_fit,
            args=[],
            bounds=tuple(bounds.values()),
            maxiter=6,
            seed=123,
            tol=0.0001)

        sources = update_sources(result.x)
        for source in sources:
            source.regularize()

        if optimize_depth is True:
            bounds = OrderedDict()
            for source in sources:
                bounds.update({'depth%s'%ev_iter:(source.depth-300., source.depth+300.)})
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
    pr.disable()
    filename = 'profile.prof'
    pr.dump_stats(filename)
    for source in sources:
        source.regularize()
        print(source)
    return result, sources
