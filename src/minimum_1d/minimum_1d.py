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


def update_layered_model_insheim(params, nevents):
    mod = cake.LayeredModel.from_scanlines(cake.read_nd_model_str('''
  0.             %s            %s           2.7         1264.           600.
  0.54           %s           %s           2.7         1264.           600.
  0.54           %s           %s            2.7         1264.           600.
  0.77           %s           %s            2.7         1264.           600.
  0.77           %s           %s            2.7         1264.           600.
  1.07           %s           %s            2.7         1264.           600.
  1.07           %s            %s            2.7         1264.           600.
  2.25           %s            %s            2.7         1264.           600.
  2.25           %s            %s            2.7         1264.           600.
  2.40           %s            %s            2.7         1264.           600.
  2.40           %s            %s            2.7         1264.           600.
  2.55           %s            %s            2.7         1264.           600.
  2.55           %s            %s            2.7         1264.           600.
  3.28           %s            %s            2.7         1264.           600.
  3.28           %s            %s            2.7         1264.           600.
  3.550          %s           %s            2.7         1264.           600.
  3.550          %s           %s            2.7         1264.           600.
  5.100          %s           %s            2.7         1264.           600.
  5.100          %s           %s            2.7         1264.           600.
  15.00          6.18          3.57           2.7         1264.           600.
  15.00          6.18          3.57           2.7         1264.           600.
  20.00          6.25          3.61           2.7         1264.           600.
  20.00          6.25          3.61           2.7         1264.           600.
  21.00          6.88          3.97           2.7         1264.           600.
  21.00          6.88          3.97           2.7         1264.           600.
 24.             8.1            4.69           2.7         1264.           600.
mantle
 24.             8.1            4.69           2.7         1264.           600.'''.lstrip() % (params[1+4*nevents], params[2+4*nevents],
                                                                                               params[3+4*nevents], params[4+4*nevents], params[3+4*nevents], params[4+4*nevents],
                                                                                               params[5+4*nevents], params[6+4*nevents], params[5+4*nevents], params[6+4*nevents],
                                                                                               params[7+4*nevents], params[8+4*nevents], params[7+4*nevents], params[8+4*nevents],
                                                                                               params[9+4*nevents], params[10+4*nevents], params[9+4*nevents], params[10+4*nevents],
                                                                                               params[11+4*nevents], params[12+4*nevents], params[11+4*nevents], params[12+4*nevents],
                                                                                               params[13+4*nevents], params[14+4*nevents], params[13+4*nevents], params[14+4*nevents],
                                                                                               params[15+4*nevents], params[16+4*nevents], params[15+4*nevents], params[16+4*nevents],
                                                                                               params[17+4*nevents], params[18+4*nevents], params[17+4*nevents], params[18+4*nevents],
                                                                                               params[19+4*nevents], params[20+4*nevents], params[19+4*nevents], params[20+4*nevents],)))

    return mod


def update_sources(params):
    for i, source in enumerate(sources):
        source.lat = float(params[0+4*i])
        source.lon = float(params[1+4*i])
        source.depth = float(params[2+4*i])
        source.time = float(source_dc.time - params[3+4*i])
    return sources


def picks_fit(params, line=None):
    update_sources(params)
    update_layered_model(params, len(ev_dict_list))
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
                    dists = ortho.distance_accurate50m(source.lat, source.lon,
                                                       stp.lat,
                                                       stp.lon)*cake.m2d

                    for i, arrival in enumerate(mod.arrivals([dists],
                                                phases=phase_list,
                                                zstart=source.depth)):

                        tdiff = st["pick"]
                        phase = st["phase"]
                        if phase == "Pg":
                            phase = "P<(moho)"
                        if phase == "pg":
                            phase = "p<(moho)"
                        used_phase = arrival.used_phase()

                        if phase == used_phase.given_name():
                            misfits += num.sqrt(num.sum((tdiff - arrival.t)**2))
                            norms += num.sqrt(num.sum(arrival.t**2))
    if misfits == 0:
        misfit = 9999999.
    else:
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
                dist = ortho.distance_accurate50m(ev["lat"], ev["lon"],
                                                  st.lat, st.lon)*cake.m2d
                dists.append(dist)
                for i, arrival in enumerate(mod.arrivals([dist],
                                            phases=phase_list,
                                            zstart=events[nev].depth)):

                    event_arr=model.event.Event(lat=ev["lat"], lon=ev["lon"],
                                                  time=ev["time"],
                                                  catalog=ev["source"],
                                                  magnitude=ev["mag"])
                    used_phase=arrival.used_phase()
                    used_phase=used_phase.given_name()
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


def solve(show=False, n_tests=1, scenario_folder="scenarios"):
    global ev_dict_list, times, phase_list, km, mod, pyrocko_stations, bounds, sources, source_dc, iiter

    km = 1000.
    iiter = 0
    mod = insheim_layered_model()

    t = timesys.time()
    sources = []
    bounds = OrderedDict()
    test_events, pyrocko_stations=load_synthetic_test(n_tests, scenario_folder)
    ev_iter = 0
    for ev in test_events:
        bounds.update({'lat%s' %ev_iter:(ev.lat-0.4, ev.lat+0.4)})
        bounds.update({'lon%s'%ev_iter:(ev.lon-0.2, ev.lon+0.2)})
        bounds.update({'depth%s'%ev_iter:(1.*km, 7.*km)})
        bounds.update({'timeshift%s'%ev_iter:(-0.01, 0.01)})
        ev_iter = ev_iter+1
    bounds.update({'p_vel%s' %ev_iter:(ev.lat-0.4, ev.lat+0.4)})
    bounds.update({'s_vel%s'%ev_iter:(ev.lon-0.2, ev.lon+0.2)})
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
    p=cake.PhaseDef('p')
    P=cake.PhaseDef('P')

    # S-phase Definitions
    Phase_S=cake.PhaseDef('S')
    Sg=cake.PhaseDef('S<(moho)')
    sg=cake.PhaseDef('s<(moho)')
    phase_list=[P, p, Sg, sg, pg, Phase_S]

    synthetic_ray_tracing_setup(test_events, pyrocko_stations, inp_cake)

    if show is True:
        from bokeh.client import push_session, show_session
        from bokeh.io import curdoc
        from bokeh.plotting import figure
        from bokeh.layouts import column
        from bokeh.models import Button
        f = figure(title='SciPy Optimisation Progress',
                   x_axis_label='# Iteration',
                   y_axis_label='Misfit',
                   plot_width=800,
                   plot_height=300)
        plot = f.scatter([], [])
        ds = plot.data_source

        def button_callback(a, b):
            new_data=dict()

        button=Button(label="Update")

        curdoc().add_root(column(f, button))
        session = push_session(curdoc())
        session.show()

        result=scipy.optimize.differential_evolution(
            picks_fit,
            args=[plot],
            bounds=tuple(bounds.values()),
            maxiter=4,
            tol=0.01,
            callback=lambda a, convergence: curdoc().add_next_tick_callback(button_callback(a, convergence)))
    else:
        result=scipy.optimize.differential_evolution(
            picks_fit,
            args=[],
            bounds=tuple(bounds.values()),
            maxiter=4,
            tol=0.01)

    source = update_source(result.x)
    source.regularize()

    for source in sources:
        print(source)
    return result, sources
