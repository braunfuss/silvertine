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


def update_source(params):
    s = source
    s.lat = float(params[0])
    s.lon = float(params[1])
    s.depth = float(params[2])
    s.time = float(source_dc.time - params[3])
    return source


def insheim_layered_model():
    mod = cake.LayeredModel.from_scanlines(cake.read_nd_model_str('''
  0.             1.81           0.85           2.7         1264.           600.
  0.54           2.25           1.11           2.7         1264.           600.
  0.54           2.25           1.11           2.7         1264.           600.
  0.77           2.85           1.44           2.7         1264.           600.
  0.77           2.85           1.44           2.7         1264.           600.
  1.07           3.18           1.61           2.7         1264.           600.
  1.07           3.18           1.61           2.7         1264.           600.
  2.25           3.81           2.01           2.7         1264.           600.
  2.25           3.81           2.01           2.7         1264.           600.
  2.40           5.12           2.65           2.7         1264.           600.
  2.40           5.12           2.65           2.7         1264.           600.
  2.55           4.53           2.39           2.7         1264.           600.
  2.55           4.53           2.39           2.7         1264.           600.
  3.28           5.14           2.91           2.7         1264.           600.
  3.28           5.14           2.91           2.7         1264.           600.
  3.550          5.688          3.276           2.7         1264.           600.
  3.550          5.688          3.276           2.7         1264.           600.
  5.100          5.98          3.76           2.7         1264.           600.
  5.100          5.98          3.76           2.7         1264.           600.
  15.00          6.18          3.57           2.7         1264.           600.
  15.00          6.18          3.57           2.7         1264.           600.
  20.00          6.25          3.61           2.7         1264.           600.
  20.00          6.25          3.61           2.7         1264.           600.
  21.00          6.88          3.97           2.7         1264.           600.
  21.00          6.88          3.97           2.7         1264.           600.
 24.             8.1            4.69           2.7         1264.           600.
mantle
 24.             8.1            4.69           2.7         1264.           600.'''.lstrip()))

    return mod



def picks_fit(params, line=None):
    update_source(params)
    global iiter
    iiter = 0
    misfits = 0.
    norms = 0.
    dists = []
    for ev in ev_dict_list:
        for st in ev["phases"]:
            for stp in pyrocko_stations:
                if stp.station == st["station"]:
                    dists = ortho.distance_accurate50m(source.lat, source.lon, stp.lat, stp.lon)*cake.m2d
                    try:
                        for i, arrival in enumerate(mod.arrivals([dists], phases=phase_list, zstart=source.depth)):

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
                    except:
                        pass
    misfit = num.sqrt(misfits**2 / norms**2)

    iiter += 1

    if line:
        data = {
            'y': [misfit],
            'x': [iiter],
        }
        line.data_source.stream(data)

    return misfit


def load_synthetic_test(n_tests, nstart=8, nend=None):
    events = []
    stations = []
    for i in range(nstart, n_tests):
        events.append(model.load_events("/home/steinberg/src/mini-1d/scenario_%s/event.txt" % i)[0])
        stations.append(model.load_stations("/home/steinberg/src/mini-1d/scenario_%s/stations.pf" % i))
    return events, stations


def synthetic_ray_tracing_setup(events, stations, mod):

    k = 0
    for evs, stats in zip(events, stations):
        ev_dict_list.append(dict(id="%s" % k, time=evs.time, lat=evs.lat,
                            lon=evs.lon, mag=evs.magnitude, mag_type="syn",
                            source="syn", phases=[], depth=[], rms=[],
                            error_h=[], error_z=[]))
    pyrocko_stations = stations[0]

    for nev, ev in enumerate(ev_dict_list):
        phase_markers = []
        stations_event = []
        hypo_in = []
        station_phase_to_nslc = {}
        dists = []
        for st in pyrocko_stations:
                dist = ortho.distance_accurate50m(ev["lat"], ev["lon"],
                                                  st.lat, st.lon)*cake.m2d
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


def solve(show=False):
    global ev_dict_list, times, phase_list, km, mod, pyrocko_stations, bounds, source, source_dc, iiter

    km = 1000.
    iiter = 0
    mod = insheim_layered_model()

    pyrocko_stations = model.load_stations("test_scenario" + "/" + "stations.raw.txt")

    bounds = OrderedDict([
        ('lat', (49.0, 49.4)),
        ('lon', (8.2, 8.6)),
        ('depth', (1.*km, 7.*km)),
        ('timeshift', (-0.01, 0.01)),
        ])

    time = util.str_to_time("2008-06-10 06:40:55.265")
    source_dc = DCSource(
        lat=49.2641236051,
        lon=8.40722205647,
        depth=5968.,
        strike=20.,
        dip=40.,
        rake=60.,
        time=time,
        magnitude=4.)

    source = gf.DCSource(
        lat=source_dc.lat,
        lon=source_dc.lon)

    t = timesys.time()

    n_tests = 9
    test_events, test_stations = load_synthetic_test(n_tests)
    ev_dict_list = []
    times = []

    inp_cake = mod
    Pg = cake.PhaseDef('P<(moho)')
    pg = cake.PhaseDef('p<(moho)')
    p = cake.PhaseDef('p')
    P = cake.PhaseDef('P')

    Phase_S = cake.PhaseDef('S')
    Sg = cake.PhaseDef('S<(moho)')
    sg = cake.PhaseDef('s<(moho)')
    phase_list = [P, p, Sg, sg, pg, Phase_S]

    synthetic_ray_tracing_setup(test_events, test_stations, inp_cake)

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
            new_data = dict()

        button = Button(label="Update")

        curdoc().add_root(column(f, button))
        session = push_session(curdoc())
        session.show()

        result = scipy.optimize.differential_evolution(
            picks_fit,
            args=[plot],
            bounds=tuple(bounds.values()),
            maxiter=4,
            tol=0.01,
            callback=lambda a, convergence: curdoc().add_next_tick_callback(button_callback(a, convergence)))
    else:
        result = scipy.optimize.differential_evolution(
            picks_fit,
            args=[],
            bounds=tuple(bounds.values()),
            maxiter=4,
            tol=0.01)

    source = update_source(result.x)
    source.regularize()

    print(source)
    return result, source
