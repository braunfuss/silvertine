import random
import math
import numpy as num
from os.path import join as pjoin
import os.path as op
from pyrocko.moment_tensor import MomentTensor
from collections import OrderedDict
from pyrocko import util, model, io, trace, config
from pyrocko.gf import DCSource, RectangularSource
from pyrocko.guts import Object, Int, String
import os
from pyrocko import orthodrome as ortho
from pyrocko import cake, model, gf
from pyrocko.gf import meta
import time as timesys
import scipy
from pyrocko.gui.pile_viewer import PhaseMarker, EventMarker
from ..util.ref_mods import *
from ..util.differential_evolution import differential_evolution
from ..util import store_variation
import ray
import psutil
from pathlib import Path
from silvertine.util import ttt
from matplotlib import pyplot as plt
from pyrocko.guts import Float
from pyrocko import plot

num_cpus = psutil.cpu_count(logical=False)


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
        source.time = float(source_dc.time + params[3])
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
                    try:
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

                        misfits += num.sqrt(num.sum((tdiff - onset)**2))
                        norms += num.sqrt(num.sum(onset**2))
                    except Exception:
                        pass
        misfit = num.sqrt(misfits**2 / norms**2)
    return misfit


def picks_fit(params, line=None, line2=None, line3=None, line4=None,
              interpolate=True):
    global iiter

    dists = []
    iter_event = 0
    iter_new = iiter + 1
    iiter = iter_new
    misfit_stations = 0
    for ev, source in zip(ev_dict_list, sources):
        misfits = 0.
        norms = 0.
        source.lat = float(params[0+4*iter_event])
        source.lon = float(params[1+4*iter_event])
        source.depth = float(params[2+4*iter_event])
        source.time = float(ev["time"] + params[3+4*iter_event])
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
                    try:
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

                        misfits += num.sqrt(num.sum((tdiff - onset)**2))
                        norms += num.sqrt(num.sum(onset**2))
                    except Exception:
                        pass
        iter_event = iter_event + 1
        try:
            misfit_stations = misfit_stations + num.sqrt(misfits**2 / norms**2)
        except:
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
    misfit = num.sqrt(misfit_stations**2 / len(ev_dict_list)**2)
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


def load_synthetic_test(n_tests, scenario_folder, nstart=0, nend=None):
    events = []
    stations = []
    for i in range(nstart, n_tests):
        try:
            print("%s/scenario_%s/event.txt" % (scenario_folder, i))
            event = model.load_events("%s/scenario_%s/event.txt" % (scenario_folder, i))[0]
            if len(event.tags) > 0:
                if event.tags[0] == "no_event":
                    pass
                else:
                    events.append(event)
                    stations.append(model.load_stations("%s/scenario_%s/stations.pf" % (scenario_folder, i)))
            else:
                events.append(event)
                stations.append(model.load_stations("%s/scenario_%s/stations.pf" % (scenario_folder, i)))
        except FileNotFoundError:
            pass
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
    return ev_list, ev_dict_list


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

    try:
        source_dc = sources
        result = differential_evolution(
            picks_fit_parallel,
            args=[event, sources, source_dc, stations, interpolated_tts],
            bounds=tuple(bounds.values()),
            maxiter=20,
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
                                         tags=[str(result.fun),
                                               str(event["id"])])
        result_events.append(event)
        file = open(name, 'a+')
        event_result.olddumpf(file)
        file.write('--------------------------------------------\n')
        file.close()
    except Exception:
        pass


def get_phases_list():
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

    return phase_list


def bokeh_plot():
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

    return p1, p2, p3, p4, curdoc, session


def associate_waveforms(test_events, stations_list, reference_events=None,
                        folder=None, plot=True,
                        scenario=False):
    ev_iter = 0
    traces_dict = OrderedDict()
    from silvertine.util.waveform import plot_waveforms
    if scenario is False:
        # hardcoded for bgr envs
        pathlist = Path(folder).glob('day*')
        for path in sorted(pathlist):
            d1 = str(path)[4:16]
            d2 = str(path)[17:]
            for ev in test_events:
                no_reference = True
                #if reference_events is not None:
                #    for ref_ev in reference_events:
                event_time = ev.time
                if ev.time > d1 and ev.time < d2:
                    traces = io.load(path)
                    for tr in traces:
                        tr.chop(ev.time-10, ev.time+60)
    if scenario is True:
        for i, event in enumerate(test_events):
            stations = stations_list[i]
            savedir = folder + '/scenario_' + str(i) + '/'
            traces = io.load(savedir+"traces.mseed")
            traces_dict.update({'%s' % i: traces})

            if plot is True:
                plot_waveforms(traces, event, stations, savedir)
    return traces_dict


def get_bounds(test_events, parallel, singular, bounds, sources, bounds_list,
               minimum_vel=False, reference_events=None, minimum_vel_mod=None):
    ev_iter = 0
    for ev in test_events:
        no_reference = True
        if reference_events is not None:
            for ref_ev in reference_events:
                ev.lat = ref_ev.lat
                ev.lon = ref_ev.lon
                if ev.time > ref_ev.time-5. and ev.time < ref_ev.time+5.:
                    if parallel is False and singular is False:
                        bounds.update({'lat%s' % ev_iter: (ref_ev.lat-0.3, ref_ev.lat+0.3)})
                        bounds.update({'lon%s' % ev_iter: (ref_ev.lon-0.3, ref_ev.lon+0.3)})
                        bounds.update({'depth%s' % ev_iter: (0*km, 12*km)})
                        bounds.update({'timeshift%s' % ev_iter: (-0.1, 0.1)})
                    else:
                        bounds = OrderedDict()
                        bounds.update({'lat%s' % ev_iter: (ref_ev.lat-0.3, ref_ev.lat+0.3)})
                        bounds.update({'lon%s' % ev_iter: (ref_ev.lon-0.3, ref_ev.lon+0.3)})
                        bounds.update({'depth%s' % ev_iter: (0*km, 12*km)})
                        bounds.update({'timeshift%s' % ev_iter: (-0.1, 0.1)})
                        bounds_list.append(bounds)
                    no_reference = False
            if no_reference is True:
                if parallel is False and singular is False:
                    bounds.update({'lat%s' % ev_iter: (ev.lat-0.3, ev.lat+0.3)})
                    bounds.update({'lon%s' % ev_iter: (ev.lon-0.3, ev.lon+0.3)})
                    bounds.update({'depth%s' % ev_iter: (0*km, 12*km)})

                    bounds.update({'timeshift%s' % ev_iter: (-0.1, 0.1)})
                else:
                    bounds = OrderedDict()
                    bounds.update({'lat%s' % ev_iter: (ev.lat-0.3, ev.lat+0.3)})
                    bounds.update({'lon%s' % ev_iter: (ev.lon-0.3, ev.lon+0.3)})
                    bounds.update({'depth%s' % ev_iter: (0*km, 12*km)})
                    bounds.update({'timeshift%s' % ev_iter: (-0.1, 0.1)})
                    bounds_list.append(bounds)

        else:
            if parallel is False and singular is False:
                bounds.update({'lat%s' % ev_iter: (ev.lat-0.3, ev.lat+0.3)})
                bounds.update({'lon%s' % ev_iter: (ev.lon-0.3, ev.lon+0.3)})
                bounds.update({'depth%s' % ev_iter: (0*km, 12*km)})
                bounds.update({'timeshift%s' % ev_iter: (-0.1, 0.1)})
            else:
                bounds = OrderedDict()
                bounds.update({'lat%s' % ev_iter: (ev.lat-0.3, ev.lat+0.3)})
                bounds.update({'lon%s' % ev_iter: (ev.lon-0.3, ev.lon+0.3)})
                bounds.update({'depth%s' % ev_iter: (0*km, 12*km)})
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
    if minimum_vel is True:
        pertub = 0.1
        layers = minimum_vel_mod.layers()
        for i in range(0, minimum_vel_mod.nlayers):
            layer = next(layers)
            p_vel = ((layer.mbot.vp + layer.mtop.vp)/2.)/1000.
            s_vel = ((layer.mbot.vs + layer.mtop.vs)/2.)/1000.
            if i == 0:
                bounds.update({'p_vel%s' %i:(p_vel*(1-pertub), p_vel*(1+pertub))})
                bounds.update({'s_vel%s'%i:(s_vel*(1-pertub), s_vel*(1+pertub))})
            else:
                bounds.update({'p_vel_up%s' %i:(p_vel*(1-pertub), p_vel*(1+pertub))})
                bounds.update({'s_vel_up%s'%i:(s_vel*(1-pertub), s_vel*(1+pertub))})

    return bounds, bounds_list, sources, source_dc


def update_sources(ev_dict_list, params):
    sources = []
    for i, ev in enumerate(ev_dict_list):
        source = gf.DCSource()
        source.lat = float(params[0+4*i])
        source.lon = float(params[1+4*i])
        source.depth = float(params[2+4*i])
        source.time = float(ev["time"] + params[3+4*i])
        sources.append(source)
    return sources


def get_result_events(params, sources, result, ev_dict_list):
    result_events = []
    for i, source in enumerate(sources):
        source.time = float(ev_dict_list[i]["time"] + params[3+4*i])
        result_sources.append(source)
        try:
            event = model.event.Event(lat=source.lat, lon=source.lon,
                                      time=source.time,
                                      magnitude=source.magnitude,
                                      depth=source.depth,
                                      tags=[str(result.fun),
                                            str(ev_dict_list[i]["id"])])
        except:
            event = model.event.Event(lat=source.lat, lon=source.lon,
                                      time=source.time,
                                      depth=source.depth,
                                      tags=[str(result.fun),
                                            str(ev_dict_list[i]["id"])])

        result_events.append(event)
    return result_events


def get_single_result_event(ev_dict_list_copy, params_x, result, i=0):
    try:
        source = gf.DCSource(
            lat=float(params_x[0]),
            lon=float(params_x[1]),
            depth=float(params_x[2]),
            magnitude=ev_dict_list_copy[i]["mag"],
            time=ev_dict_list_copy[i]["time"] + float(params_x[3]))
        result_sources.append(source)
        event_result = model.event.Event(lat=source.lat, lon=source.lon,
                                         time=source.time,
                                         depth=source.depth,
                                         magnitude=source.magnitude,
                                         tags=[str(result.fun),
                                               str(ev_dict_list_copy[i]["id"])])
    except Exception:
        source = gf.DCSource(
            lat=float(params_x[0]),
            lon=float(params_x[1]),
            depth=float(params_x[2]),
            time=ev_dict_list_copy[i]["time"] + float(params_x[3]))
        result_sources.append(source)
        event_result = model.event.Event(lat=source.lat, lon=source.lon,
                                         time=source.time,
                                         depth=source.depth,
                                         tags=[str(result.fun),
                                               str(ev_dict_list_copy[i]["id"])])
    return event_result, source


def str_float_vals(vals):
    return ' '.join(['%e' % val for val in vals])


def update_layered_model(mod, params, nevents):
    srows = []
    k = 1000.
    scanned_mod = mod.to_scanlines()
    s = 0
    rows = scanned_mod
    rows_cut = scanned_mod[1::2]
    for i in range(0, mod.nlayers):
        depth, vp, vs, rho, qp, qs = rows_cut[i]
        if i == 0:
            depth, vp, vs, rho, qp, qs = rows[0]
            vp_mod = params[0+4*nevents]
            vs_mod = params[1+4*nevents]
            s = s+1
        elif (i % 2) != 0:
            try:
                vp_mod = params[s*2+0+4*nevents]
                vs_mod = params[s*2+1+4*nevents]
            except:
                vp_mod = params[s-1*2+0+4*nevents]
                vs_mod = params[s-1*2+1+4*nevents]
        else:
            vp_mod = params[s*2+0+4*nevents]
            vs_mod = params[s*2+1+4*nevents]
            s = s+1
        row = [depth / k, vp_mod, vs_mod, rho]
        srows.append('%15s' % (str_float_vals(row)))
        if i != 0:
            srows.append('%15s' % (str_float_vals(row)))
        if i == mod.nlayers-2:
            srows.append("moho")

    d = '\n'.join(srows)
    mod_modified = cake.LayeredModel.from_scanlines(cake.read_nd_model_str(d))
    return mod_modified


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
                                                                                               params[19+4*nevents], params[20+4*nevents], params[19+4*nevents], params[20+4*nevents])))

    return mod


def minimum_1d_fit(params, mod, line=None):
    global iiter
#    mod = update_layered_model(mod, params, len(ev_dict_list))
    mod = update_layered_model_insheim(params, len(ev_dict_list))

    dists = []
    iter_event = 0
    iter_new = iiter + 1
    iiter = iter_new
    misfit_stations = 0
    for ev, source in zip(ev_dict_list, sources):
        misfits = 0.
        norms = 0.
        source.lat = float(params[0+4*iter_event])
        source.lon = float(params[1+4*iter_event])
        source.depth = float(params[2+4*iter_event])
        source.time = float(ev["time"] + params[3+4*iter_event])
        for st in ev["phases"]:
            for stp in pyrocko_stations[iter_event]:
                if stp.station == st["station"]:
                    phase = st["phase"]
                    dists = ortho.distance_accurate50m(source.lat, source.lon,
                                                       stp.lat,
                                                       stp.lon)*cake.m2d

                    try:
#                        for phase_p in phase_list:

                                tdiff = st["pick"]
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

                                for i, arrival in enumerate(mod.arrivals([dists],
                                                            phases=[cake_phase],
                                                            zstart=source.depth)):
                                    used_phase = arrival.used_phase()

                                    if phase == used_phase.given_name():
                                        misfits += num.sqrt(num.sum((tdiff - arrival.t)**2))
                                        norms += num.sqrt(num.sum(arrival.t**2))
                    except:
                        pass

    misfit = num.sqrt(misfits**2 / norms**2)
    iter_event = iter_event + 1
    if line:
        data = {
            'y': [misfit],
            'x': [iiter],
        }
        line.data_source.stream(data)

    return misfit


def solve(show=False, n_tests=1, scenario_folder="scenarios",
          optimize_depth=False, scenario=True, data_folder="data",
          parallel=True, adress=None, interpolate=True, mod_name="insheim",
          singular=False, nboot=1, hybrid=True,
          minimum_vel=False, reference="catalog",):
    global ev_dict_list, times, phase_list, km, mod, pyrocko_stations, bounds, sources, source_dc, iiter, interpolated_tts, result_sources, result_events

    if scenario is False:
        if reference == "catalog":
            reference_events = model.load_events("data/events_ler.pf")
        maxiter = 55
        folder_waveforms = "/md3/projects3/seiger/acquisition"
    else:
        reference_events = None
        maxiter = 25
        folder_waveforms = scenario_folder

    km = 1000.
    iiter = 0
    if mod_name == "insheim":
        mod = insheim_layered_model()
    if mod_name == "landau":
        mod = landau_layered_model()
    if mod_name == "vsp":
        mod = vsp_layered_model()

    if nboot > 1:
        pertubed_mods = store_variation.ensemble_earthmodel(mod,
                                                            num_vary=nboot,
                                                            error_depth=0.2,
                                                            error_velocities=0.2,
                                                            depth_limit_variation=600)
    else:
        pertubed_mods = [mod]

    meta_results = []
    for kmod, mod in enumerate(pertubed_mods):

        result_sources = []
        result_events = []
        t = timesys.time()
        sources = []
        bounds = OrderedDict()
        bounds_list = []
        times = []
        inp_cake = mod

        phase_list = get_phases_list()

        if scenario is True:
            test_events, pyrocko_stations = load_synthetic_test(n_tests,
                                                                scenario_folder)
            ev_dict_list = []
            ev_list, ev_dict_list = synthetic_ray_tracing_setup(test_events,
                                                                pyrocko_stations,
                                                                inp_cake)
            if reference == "hyposat":
                from .hyposat_util import run_hyposat
                for i, ev in enumerate(test_events):
                    run_hyposat(ev_dict_list[i], ev, [ev_list[i]],
                                pyrocko_stations[i])
        else:
            test_events, pyrocko_stations, ev_dict_list, ev_list_picks = load_data(data_folder, nevent=n_tests)
            if reference == "hyposat":
                reference_events = []
                from .hyposat_util import run_hyposat
                for i, ev in enumerate(test_events):
                    mod_name_hyposat = "pertubed_%s_%s.dat" % (mod_name, kmod)
                    event = run_hyposat(ev_dict_list[i], ev,
                                        [ev_list_picks[i]],
                                        pyrocko_stations[i], mod,
                                        mod_name_hyposat)
                    reference_events.append(event)
        bounds, bounds_list, sources, source_dc = get_bounds(test_events,
                                                             parallel,
                                                             singular,
                                                             bounds,
                                                             sources,
                                                             bounds_list,
                                                             minimum_vel=minimum_vel,
                                                             reference_events=reference_events,
                                                             minimum_vel_mod=mod)

        if hybrid is True:
            waveforms = associate_waveforms(test_events, pyrocko_stations,
                                            reference_events=None,
                                            folder=folder_waveforms,
                                            scenario=scenario)

        # Load/Calculate Traveltime tabel for each phase (parallel)
        interpolated_tts, missing = ttt.load_sptree(phase_list, mod_name)
        calculated_ttt = False
        if minimum_vel is False:
            if len(missing) != 0:
                print("Calculating travel time look up table,\
                        this may take some time.")
                ttt.calculate_ttt_parallel(pyrocko_stations, mod, missing,
                                           mod_name,
                                           adress=adress)
                interpolated_tts_new, missing = ttt.load_sptree(phase_list, mod_name)
                interpolated_tts = {**interpolated_tts, **interpolated_tts_new}
                calculated_ttt = True

        if show is True:
            p1, p2, p3, p4, curdoc, session = bokeh_plot()

            if singular is False:
                result = differential_evolution(
                    picks_fit,
                    args=[p1, p2, p3, p4, interpolate],
                    bounds=tuple(bounds.values()),
                    seed=123,
                    maxiter=maxiter,
                    tol=0.0001)

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
                        maxiter=maxiter,
                        tol=0.001)
                    for source in sources:
                        sources = update_depth(result.x)

                sources = update_sources(ev_dict_list, result.x)
                result_events = get_result_events(result.x, sources, result,
                                                  ev_dict_list)

            else:
                ev_dict_list_copy = ev_dict_list.copy()
                sources_copy = sources.copy()
                bounds_list_copy = bounds_list.copy()
                pyrocko_stations_copy = pyrocko_stations.copy()
                for i in range(len(ev_dict_list_copy)):
                    # p1, p2, p3, p4, curdoc, session = bokeh_plot()
                    ev_dict_list = [ev_dict_list_copy[i]]
                    sources = [sources_copy[i]]
                    bounds = bounds_list_copy[i]
                    pyrocko_stations = [pyrocko_stations_copy[i]]
                    result = differential_evolution(
                        picks_fit,
                        args=[p1, p2, p3, p4, interpolate],
                        bounds=tuple(bounds.values()),
                        seed=123,
                        maxiter=maxiter,
                        tol=0.00001)

                    event_result, source = get_single_result_event(ev_dict_list_copy,
                                                                   result.x,
                                                                   result,
                                                                   i=i)
                    result_events.append(event_result)

                    if optimize_depth is True:
                        bounds = OrderedDict()
                        for source in [result_sources[i]]:
                            bounds.update({'depth%s' % ev_iter: (source.depth-300.,
                                                                 source.depth+300.)})
                        result = differential_evolution(
                            depth_fit,
                            args=[plot],
                            bounds=tuple(bounds.values()),
                            seed=123,
                            maxiter=maxiter,
                            tol=0.001)
                        for source in sources:
                            sources = update_depth(result.x)
                    for source in sources:
                        result_sources.append(source)

        else:
            if parallel is True or parallel is "True":
                name = scenario_folder+"/events_parallel_%s.txt" % str(kmod)
                file = open(name, 'w+')
                file.close()
                if calculated_ttt is False:
                    ray.init(num_cpus=num_cpus-1)
                event_dict = []
                ray.get([optim_parallel.remote(ev_dict_list[i], sources[i],
                                               bounds_list[i],
                                               pyrocko_stations[i],
                                               interpolated_tts,
                                               result_sources,
                                               result_events,
                                               name) for i in range(len(ev_dict_list))])
                result = None
                source = None
            else:
                if singular is True:
                    ev_dict_list_copy = ev_dict_list.copy()
                    sources_copy = sources.copy()
                    bounds_list_copy = bounds_list.copy()
                    pyrocko_stations_copy = pyrocko_stations.copy()
                    for i in range(len(ev_dict_list_copy)):
                        ev_dict_list = [ev_dict_list_copy[i]]
                        sources = [sources_copy[i]]
                        bounds = bounds_list_copy[i]
                        pyrocko_stations = [pyrocko_stations_copy[i]]
                        result = differential_evolution(
                            picks_fit,
                            args=[],
                            bounds=tuple(bounds.values()),
                            seed=123,
                            maxiter=maxiter,
                            tol=0.0001)

                        event_result, source = get_single_result_event(ev_dict_list_copy,
                                                                       result.x,
                                                                       result,
                                                                       i=i)
                        result_sources.append(source)
                        result_events.append(event_result)

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
                                maxiter=maxiter,
                                tol=0.001,
                                callback=lambda a, convergence: curdoc().add_next_tick_callback(button_callback(a, convergence)))
                            for source in sources:
                                sources = update_depth(result.x)
                        for source in sources:
                            result_sources.append(source)
                else:
                    if minimum_vel is False:
                        result = differential_evolution(
                            picks_fit,
                            args=[],
                            bounds=tuple(bounds.values()),
                            maxiter=maxiter,
                            seed=123,
                            tol=0.000001)

                        sources = update_sources(ev_dict_list, result.x)

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
                                maxiter=1.,
                                tol=1.)
                            for source in sources:
                                sources = update_depth(result.x)
                        for i, source in enumerate(sources):
                            result_sources.append(source)
                            event = model.event.Event(lat=source.lat, lon=source.lon,
                                                      time=source.time, magnitude=source.magnitude,
                                                      depth=source.depth,
                                                      tags=[str(result.fun), str(ev_dict_list[i]["id"])])
                            result_events.append(event)
                    else:
                        result = differential_evolution(
                            minimum_1d_fit,
                            args=[mod],
                            bounds=tuple(bounds.values()),
                            maxiter=1,
                            seed=123,
                            tol=100)

                        sources = update_sources(ev_dict_list, result.x)
                        mod = update_layered_model_insheim(result.x, len(ev_dict_list))
                        if scenario is True:
                            mod_save = scenario_folder + '/min_1d_model'
                        if scenario is False and singular is True:
                            mod_save = data_folder + '/min_1d_model'
                        cake.write_nd_model(mod, mod_save)
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
                                maxiter=maxiter,
                                tol=0.001)
                            for source in sources:
                                sources = update_depth(result.x)
                        for i, source in enumerate(sources):
                            result_sources.append(source)
                            event = model.event.Event(lat=source.lat, lon=source.lon,
                                                      time=source.time, magnitude=source.magnitude,
                                                      depth=source.depth,
                                                      tags=[str(result.fun), str(ev_dict_list[i]["id"])])
                            result_events.append(event)

        if parallel is not True:
            if scenario is True:
                model.dump_events(result_events, scenario_folder+"/result_events_%s.pf" % str(kmod))
            else:
                savedir = data_folder + "/event_%s" %(i) + "/"
                util.ensuredir(savedir)
                model.dump_events(result_events, scenario_folder+"/result_events_%s.pf" % str(kmod))

        meta_results.append(result_events)
        nevents = len(result_events)
    nevent = 0


    for k in range(0, nevents):

        if scenario is True:
            savedir = scenario_folder + '/scenario_' + str(k) + '/'
        if scenario is False and singular is True:
            savedir = data_folder + "/event_%s" %(k) + "/"
            util.ensuredir(savedir)

        #plot.mpl_init()
        fig = plt.figure(figsize=plot.mpl_papersize('a5', 'landscape'))
        axes = fig.add_subplot(1, 1, 1, aspect=1.0)
        axes.set_xlabel('Lat')
        axes.set_ylabel('Lon')
        for i, result_events in enumerate(meta_results):
            source = result_events[k]
            axes.scatter(source.lat, source.lon)
        stations = pyrocko_stations[0]
        for st in stations:
            axes.scatter(st.lat, st.lon, c="k")
            axes.text(st.lat, st.lon, str(st.station))
        fig.savefig(savedir+'location.png')
        plt.close()
    return result, result_events
