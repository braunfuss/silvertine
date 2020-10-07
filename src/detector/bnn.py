import matplotlib.pyplot as plt
from pyrocko.gf import LocalEngine, Target, DCSource, ws
from pyrocko import trace, io
from pyrocko.marker import PhaseMarker
from silvertine.util.waveform import plot_waveforms_raw
from silvertine import scenario
from pyrocko import model, cake, orthodrome
from silvertine.locate.locate1D import get_phases_list
from keras.layers import Input, Embedding, LSTM, Dense
from keras.models import Model
import keras
import numpy as np
from matplotlib import figure
from matplotlib.backends import backend_agg
from silvertine import scenario
from silvertine.util.silvertine_meta import MTQTSource
from pyrocko import model, cake, orthodrome
from silvertine.util.ref_mods import landau_layered_model
from pyrocko import moment_tensor, util
from keras.layers import Conv1D, MaxPooling1D, Input
from keras import models
from PIL import Image
from pathlib import Path
from silvertine.util import waveform
import scipy
from mtpar import cmt2tt, cmt2tt15, tt2cmt, tt152cmt
from mtpar.basis import change_basis
import ray
import copy
import psutil
num_cpus = psutil.cpu_count(logical=False)


import os
try:
  import seaborn as sns  # pylint: disable=g-import-not-at-top
  HAS_SEABORN = True
except ImportError:
  HAS_SEABORN = False
#import tensorflow.compat.v2 as tf
#import tensorflow_probability as tfp
#tf.disable_eager_execution()

#tf.enable_v2_behavior()
pi = np.pi
deg = 180./pi

def _mat(m):
    return np.array(([[m[0], m[3], m[4]],
                      [m[3], m[1], m[5]],
                      [m[4], m[5], m[2]]]))


def omega_angle(M1x, M2x):
    M1 = _mat(M1x)
    M2 = _mat(M2x)

    n = len(M1)
    cosom = np.zeros(n)

    M1r = M1
    M2r = M2
    omega = 0.5 * (1-((np.sum(M1r*M2r))/(np.sqrt(np.sum(M1r**2))*np.sqrt(np.sum((M2r**2))))))

    return omega


def rgb2gray(rgb):
    return np.dot(rgb[...,:3], [0.2989, 0.5870, 0.1140])


def plot_weight_posteriors(names, qm_vals, qs_vals, fname):
  """Save a PNG plot with histograms of weight means and stddevs.

  Args:
    names: A Python `iterable` of `str` variable names.
      qm_vals: A Python `iterable`, the same length as `names`,
      whose elements are Numpy `array`s, of any shape, containing
      posterior means of weight varibles.
    qs_vals: A Python `iterable`, the same length as `names`,
      whose elements are Numpy `array`s, of any shape, containing
      posterior standard deviations of weight varibles.
    fname: Python `str` filename to save the plot to.
  """
  fig = figure.Figure(figsize=(6, 3))
  canvas = backend_agg.FigureCanvasAgg(fig)

  ax = fig.add_subplot(1, 2, 1)
  for n, qm in zip(names, qm_vals):
    sns.distplot(tf.reshape(qm, shape=[-1]), ax=ax, label=n)
  ax.set_title('weight means')
  ax.set_xlim([-1.5, 1.5])
  ax.legend()

  ax = fig.add_subplot(1, 2, 2)
  for n, qs in zip(names, qs_vals):
    sns.distplot(tf.reshape(qs, shape=[-1]), ax=ax)
  ax.set_title('weight stddevs')
  ax.set_xlim([0, 1.])

  fig.tight_layout()
  canvas.print_figure(fname, format='png')
  print('saved {}'.format(fname))


def plot_heldout_prediction(input_vals, probs,
                            fname, n=10, title=''):
  """Save a PNG plot visualizing posterior uncertainty on heldout data.

  Args:
    input_vals: A `float`-like Numpy `array` of shape
      `[num_heldout] + IMAGE_SHAPE`, containing heldout input images.
    probs: A `float`-like Numpy array of shape `[num_monte_carlo,
      num_heldout, num_classes]` containing Monte Carlo samples of
      class probabilities for each heldout sample.
    fname: Python `str` filename to save the plot to.
    n: Python `int` number of datapoints to vizualize.
    title: Python `str` title for the plot.
  """
  fig = figure.Figure(figsize=(9, 3*n))
  canvas = backend_agg.FigureCanvasAgg(fig)
  for i in range(n):
    ax = fig.add_subplot(n, 3, 3*i + 1)
    ax.imshow(input_vals[i, :].reshape(5,151), interpolation='None')

    ax = fig.add_subplot(n, 3, 3*i + 2)
    for prob_sample in probs:
      sns.barplot(np.arange(10), prob_sample[i, :], alpha=0.1, ax=ax)
      ax.set_ylim([0, 1])
    ax.set_title('posterior samples')

    ax = fig.add_subplot(n, 3, 3*i + 3)
    sns.barplot(np.arange(10), tf.reduce_mean(probs[:, i, :], axis=0), ax=ax)
    ax.set_ylim([0, 1])
    ax.set_title('predictive probs')
  fig.suptitle(title)
  fig.tight_layout()

  canvas.print_figure(fname, format='png')
  print('saved {}'.format(fname))


def normalize_by_std_deviation(traces):
    traces_data = []
    for tr in traces:
        trace_level = np.nanmean(tr.ydata)
        tr.ydata = tr.ydata - trace_level
        traces_data.append(tr.ydata)
        tr.ydata = tr.ydata + trace_level

    traces_data = np.asarray(traces_data)
    nanstd = np.nanstd(traces_data, axis=1)[:, np.newaxis]
    nanstd[nanstd==0] = 1.

    for i, tr in enumerate(traces):
        tr.ydata = tr.ydata  / nanstd[i]
    return traces


def normalize_by_tracemax(traces):
    traces_data = []
    for tr in traces:
        trace_level = np.nanmean(tr.ydata)
        tr.ydata = tr.ydata - trace_level
        tr.ydata = tr.ydata/np.max(tr.ydata)
        tr.ydata = tr.ydata + trace_level

    return traces


def normalize(traces):
    traces_data = []
    for tr in traces:
        trace_level = np.nanmean(tr.ydata)
        tr.ydata = tr.ydata - trace_level
        tr.ydata = (tr.ydata-np.min(tr.ydata))/(np.max(tr.ydata)-np.min(tr.ydata))
        #tr.ydata = tr.ydata + trace_level
    return traces

def normalize_chunk(traces):
    traces_data = []
    for tr in traces:
        traces_data.append(tr.ydata)

    traces_data = np.asarray(traces_data)
    max = np.max(traces_data)
    min = np.min(traces_data)

    for tr in traces:
        #trace_level = np.nanmean(tr.ydata)
        #tr.ydata = tr.ydata - trace_level
        tr.ydata = (tr.ydata-min)/(max-min)
        #tr.ydata = tr.ydata + trace_level
    return traces


def normalize_all(traces, min, max):

    for tr in traces:
        trace_level = np.nanmean(tr.ydata)
        tr.ydata = tr.ydata - trace_level
        tr.ydata = (tr.ydata-min)/(max-min)
        #tr.ydata = tr.ydata + trace_level
    return traces


def bnn_detector_data(waveforms, max_traces, events=True, multilabel=False,
                      mechanism=False, sources=None, source_type="DC",
                      mtqt_ps=None):

    add_spectrum = False
    data_traces = []
    maxsamples = 0
    max_traces = None
    min_traces = None
    for traces in waveforms:
        for tr in traces:
        #    tr.ydata = np.abs(tr.ydata)
            if max_traces is None:
                max_traces = np.max(tr.ydata)
            else:
                if max_traces < np.max(tr.ydata):
                    max_traces = np.max(tr.ydata)
            if min_traces is None:
                min_traces = np.min(tr.ydata)
            else:
                if min_traces > np.min(tr.ydata):
                    min_traces = np.min(tr.ydata)

            if len(tr.ydata) > maxsamples:
                maxsamples = len(tr.ydata)
    for k, traces in enumerate(waveforms):
        traces_coll = None
    #    traces = normalize_by_std_deviation(traces)
    #    traces = normalize_by_tracemax(traces)
        #traces = normalize(traces)

#        traces = normalize_chunk(traces)
        #traces = normalize_all(traces, min_traces, max_traces)

        for tr in traces:
            tr.lowpass(4, 5.)
            tr.highpass(4, 0.03)
            tr.ydata = tr.ydata/np.max(tr.ydata)
            nsamples = len(tr.ydata)
            data = tr.ydata

            if nsamples != maxsamples:
                data = np.pad(data, (0, maxsamples-nsamples), 'constant')
                nsamples = len(data)
                tr.ydata = data
            if add_spectrum is True:
                freqs, fdata = tr.spectrum()
                fdata = 10*np.log10(fdata)
                fdata_zero = np.zeros(len(fdata))
                fdata, dict = scipy.signal.find_peaks(fdata)
                for idx in fdata:
                    fdata_zero[idx] = 1
            #    fdata = fdata_zero
            #    data = np.concatenate((data, fdata),  axis=0)
                data = fdata
            if traces_coll is None:
                traces_coll = tr.ydata
            else:
                traces_coll = np.concatenate((traces_coll, data),  axis=0)
        nsamples = len(traces_coll)
        data_traces.append(traces_coll)

    # normalize coordinates
    if multilabel is True and events is not None:
        lons = []
        lats = []
        depths = []
        for i, ev in enumerate(events):
            lats.append(ev.lat)
            lons.append(ev.lon)
            depths.append(ev.depth)
        lats = np.asarray(lats)

        # normalize to grid coords
        lats = (lats-np.min(lats))/(np.max(lats)-np.min(lats))
        lons = np.asarray(lons)
        lons = (lons-np.min(lons))/(np.max(lons)-np.min(lons))
        depths = np.asarray(depths)
        depths = (depths-np.min(depths))/(np.max(depths)-np.min(depths))

    if events is not None:
        labels = []
        if multilabel is True:
            for i, ev in enumerate(events):
                if len(ev.tags) > 0:
                    if ev.tags[0] == "no_event":
                        tag = 0
                        labels.append([0, 0, 0])

                    else:
                        if mechanism is True:
                            rake = 0.5-(ev.moment_tensor.rake1/90.)*0.5
                            labels.append([ev.moment_tensor.strike1/360.,
                                           ev.moment_tensor.dip1/90., rake])
                        else:
                            tag = 1
                            labels.append([lats[i], lons[i], depths[i]])

                else:
                    tag = 1
                #    print(ev)
                #    print(ev.moment_tensor)
                #    print(sources[i])
                    mechanism_and_location = True
                    if mechanism is True:
                    #    rake = 0.5-(ev.moment_tensor.rake2/90.)*0.5
                        #labels.append([ev.moment_tensor.strike2/360.,
                    #                   ev.moment_tensor.dip2/90., rake])
                    #   rake = 0.5-(sources[i].rake/180.)*0.5
                     #  labels.append([sources[i].strike/360.,
                    #                  sources[i].dip/90., rake])
                        # rake2 = 0.5-(ev.moment_tensor.rake2/180.)*0.5
                        # rake1 = 0.5-(ev.moment_tensor.rake1/180.)*0.5
                        # labels.append([ev.moment_tensor.strike2/360.,
                        #                ev.moment_tensor.dip2/90., rake2,
                        #                ev.moment_tensor.strike1/360.,
                        #                ev.moment_tensor.dip1/90., rake1])

                        if source_type == "MTQT":
                            mtqt_p = mtqt_ps[i]
                            if mechanism_and_location is True:
                                labels.append([mtqt_p[0]/((3/4)*pi),
                                               0.5-(mtqt_p[1]/(1/3))*0.5,
                                               mtqt_p[2]/(2*pi),
                                               0.5-(mtqt_p[3]/(pi/2))*0.5,
                                               mtqt_p[4]/1,
                                               lats[i], lons[i], depths[i]])
                            else:
                                labels.append([mtqt_p[0]/((3/4)*pi),
                                               0.5-(mtqt_p[1]/(1/3))*0.5,
                                               mtqt_p[2]/(2*pi),
                                               0.5-(mtqt_p[3]/(pi/2))*0.5,
                                               mtqt_p[4]/1])
                        if source_type == "DC":
                            if mechanism_and_location is True:
                                # make label data vector lat/lon/depth?
                                labels.append([0.5-((ev.moment_tensor.mnn/ev.moment_tensor.moment)/2)*0.5,
                                                0.5-((ev.moment_tensor.mee/ev.moment_tensor.moment)/2)*0.5,
                                                0.5-((ev.moment_tensor.mdd/ev.moment_tensor.moment)/2)*0.5,
                                                0.5-((ev.moment_tensor.mne/ev.moment_tensor.moment)/2)*0.5,
                                                0.5-((ev.moment_tensor.mnd/ev.moment_tensor.moment)/2)*0.5,
                                                0.5-((ev.moment_tensor.med/ev.moment_tensor.moment)/2)*0.5,
                                                lats[i], lons[i], depths[i]])
                            else:
                                labels.append([0.5-((ev.moment_tensor.mnn/ev.moment_tensor.moment)/2)*0.5,
                                                0.5-((ev.moment_tensor.mee/ev.moment_tensor.moment)/2)*0.5,
                                                0.5-((ev.moment_tensor.mdd/ev.moment_tensor.moment)/2)*0.5,
                                                0.5-((ev.moment_tensor.mne/ev.moment_tensor.moment)/2)*0.5,
                                                0.5-((ev.moment_tensor.mnd/ev.moment_tensor.moment)/2)*0.5,
                                                0.5-((ev.moment_tensor.med/ev.moment_tensor.moment)/2)*0.5])
                    #    print(labels)
                    #    print(ev)
                    else:
                        labels.append([lats[i], lons[i], depths[i]])
            labels = np.asarray(labels)

        else:
            for i, ev in enumerate(events):
                if len(ev.tags) > 0:
                    found_tag = False
                    for tag in ev.tags:
                        if tag == "no_event":
                            labels.append(0)
                            found_tag = True
                    if found_tag is False:
                        labels.append(1)
                else:
                    labels.append(1)
    else:
        if multilabel is True:
            labels = np.ones((len(data_traces), 3), dtype=np.int32)*0
        else:
            labels = np.ones(len(data_traces), dtype=np.int32)*0
    data_traces = np.asarray(data_traces)
    labels = np.asarray(labels)
    return data_traces, labels, len(data_traces), nsamples

@ray.remote
def get_parallel_scenario(i, targets, store_id, noised, real_noise_traces, post, pre, no_events, stations, mod):
            engine = LocalEngine(store_superdirs=['gf_stores'])
            store = engine.get_store(store_id)
            event = scenario.gen_random_tectonic_event(i, magmin=-0.5, magmax=3.)
            source_dc = DCSource(
                lat=scenario.randlat(49., 49.3),
                lon=scenario.rand(8.0, 8.3),
                depth=scenario.rand(3000., 13000.),
                strike=scenario.rand(0., 360.),
                dip=scenario.rand(0., 90.),
                rake=scenario.rand(-180., 180.),
                magnitude=scenario.rand(-1., 3.))
     #       try:
            response = engine.process(source_dc, targets)
            traces = response.pyrocko_traces()
            event.lat = source_dc.lat
            event.lon = source_dc.lon
            event.depth = source_dc.depth
            mt = moment_tensor.MomentTensor(strike=source_dc.strike, dip=source_dc.dip, rake=source_dc.rake,
                                    magnitude=source_dc.magnitude)

            event.moment_tensor = mt

            for tr in traces:
                    for st in stations:
                        if st.station == tr.station:
                            dists = (orthodrome.distance_accurate50m(source_dc.lat,
                                                     source_dc.lon,
                                                     st.lat,
                                                     st.lon)+st.elevation)*cake.m2d
                            processed = False
                            for i, arrival in enumerate(mod.arrivals([dists],
                                            phases=get_phases_list(),
                                            zstart=source_dc.depth)):
                                if processed is False:
                                    tr.chop(arrival.t-pre, arrival.t+post)
                                    processed = True
                                else:
                                    pass


                    nsamples = len(tr.ydata)
                    randdata = np.random.normal(size=nsamples)*np.min(tr.ydata)
                    white_noise = trace.Trace(deltat=tr.deltat, tmin=tr.tmin,
                                  ydata=randdata)
                    if noised is True:
                        tr.add(white_noise)
         #   except:
               #         traces = None
                #        event = None
                 #       nsamples = None
            return [traces, event, nsamples]


def generate_test_data(store_id, nevents=50, noised=True,
                       real_noise_traces=None, post=2., pre=2.,
                       no_events=False, parallel=True):
    mod = landau_layered_model()
    engine = LocalEngine(store_superdirs=['gf_stores'])
    store = engine.get_store(store_id)
    store.config.earthmodel_1d
    scale = 2e-14
    cake_phase = cake.PhaseDef("P")
    phase_list = [cake_phase]
    waveforms_events = []
    waveforms_noise = []
    stations = model.load_stations("scenarios/stations.raw.txt")
    targets = []
    events = []
    for st in stations:
        for cha in st.channels:
            if cha.name is not "R" and cha.name is not "T" and cha.name is not "Z":
                        target = Target(
		            lat=st.lat,
		            lon=st.lon,
		            store_id=store_id,
		            interpolation='multilinear',
		            quantity='displacement',
		            codes=st.nsl() + (cha.name,))
                        targets.append(target)
    if parallel is True:
        ray.init(num_cpus=num_cpus-1)
        results = ray.get([get_parallel_scenario.remote(i, targets, store_id, noised, real_noise_traces, post, pre, no_events, stations, mod) for i in range(0, nevents)])
        for res in results:
            events.append(res[1])
            waveforms_events.append(res[0])
            nsamples = res[2]
    else:
        for i in range(0, nevents):
        #    try:
                event = scenario.gen_random_tectonic_event(i, magmin=-0.5, magmax=3.)
                source_dc = DCSource(
                    lat=scenario.randlat(49., 49.3),
                    lon=scenario.rand(8.0, 8.3),
                    depth=scenario.rand(3000., 13000.),
                    strike=scenario.rand(0., 360.),
                    dip=scenario.rand(0., 90.),
                    rake=scenario.rand(-180., 180.),
                    magnitude=scenario.rand(-1., 3.))
                response = engine.process(source_dc, targets)
                traces = response.pyrocko_traces()
                event.lat = source_dc.lat
                event.lon = source_dc.lon
                event.depth = source_dc.depth
                mt = moment_tensor.MomentTensor(strike=source_dc.strike, dip=source_dc.dip, rake=source_dc.rake,
                                                magnitude=source_dc.magnitude)

                event.moment_tensor = mt

                events.append(event)
                for tr in traces:
                    for st in stations:
                        if st.station == tr.station:
                            dists = (orthodrome.distance_accurate50m(source_dc.lat,
                                                                 source_dc.lon,
                                                                 st.lat,
                                                                 st.lon)+st.elevation)*cake.m2d
                            processed = False
                            for i, arrival in enumerate(mod.arrivals([dists],
                                                        phases=get_phases_list(),
                                                        zstart=source_dc.depth)):
                                if processed is False:
                                    tr.chop(arrival.t-pre, arrival.t+post)
                                    processed = True
                                else:
                                    pass


                    nsamples = len(tr.ydata)
                    randdata = np.random.normal(size=nsamples)*np.min(tr.ydata)
                    white_noise = trace.Trace(deltat=tr.deltat, tmin=tr.tmin,
                                              ydata=randdata)
                    if noised is True:
                        tr.add(white_noise)

                waveforms_events.append(traces)
        #    except:
        #        pass
        # same number of non-events
        if no_events is True:
            for i in range(0, nevents):
                try:
                    source_dc = DCSource(
                        lat=scenario.randlat(49., 49.2),
                        lon=scenario.rand(8.1, 8.2),
                        depth=scenario.rand(100., 3000.),
                        strike=scenario.rand(0., 360.),
                        dip=scenario.rand(0., 90.),
                        rake=scenario.rand(-180., 180.),
                        magnitude=scenario.rand(-1., 0.1))

                    response = engine.process(source_dc, targets)
                    traces = response.pyrocko_traces()
                    for tr in traces:
                        for st in stations:
                            if st.station == tr.station:
                                dists = (orthodrome.distance_accurate50m(source_dc.lat,
                                                                         source_dc.lon,
                                                                         st.lat,
                                                                         st.lon)+st.elevation)*cake.m2d
                                processed = False
                                for i, arrival in enumerate(mod.arrivals([dists],
                                                            phases=get_phases_list(),
                                                            zstart=source_dc.depth)):
                                    if processed is False:
                                        tr.chop(arrival.t-2, arrival.t+2)
                                        processed = True
                                    else:
                                        pass
                        tr.ydata = tr.ydata*0.

                        nsamples = len(tr.ydata)
                        randdata = np.random.normal(size=nsamples)*scale
                        white_noise = trace.Trace(deltat=tr.deltat, tmin=tr.tmin,
                                                  ydata=randdata)
                        tr.add(white_noise)
                    waveforms_noise.append(traces)

                except:
                    pass

    return waveforms_events, waveforms_noise, nsamples, len(stations), events


def make_grid(center, dimx, dimy, zmin, zmax, dim_step, depth_step,
              latmin=None, latmax=None, lonmin=None, lonmax=None):

    if latmin is None:
        lats = np.arange(center[0]-dimx, center[0]+dimx, dim_step)
        lons = np.arange(center[1]-dimy, center[1]+dimy, dim_step)
    else:
        lats = np.arange(latmin, latmax, dim_step)
        lons = np.arange(lonmin, lonmax, dim_step)
    if depth_step != 0:
        depths = np.arange(zmin, zmax, depth_step)
    else:
        depths = []
    # single val

    return lats, lons, depths


def make_reciever_grid(center, dimx, dimy):
    lats, lons, depths = make_grid(center, dimx, dimy, zmin=0, zmax=0,
                                   dim_step=0, depth_step=0)


def nearest_station(lats, lons, lat_s, lon_s):
    dist_min = 9999999999.
    i_s = 0
    for lat, lon in zip(lats, lons):
        dist = orthodrome.distance_accurate50m(lat_s, lon_s, lat, lon)
        if dist < dist_min:
            i_min = i
        i_s = i_s+1
    return lats[i_min], lons[i_min]


def get_scn_mechs():
    mechs = num.loadtxt("ridgecrest/scn_test.mech", dtype="str")
    dates = []
    strikes = []
    rakes = []
    dips = []
    depths = []
    lats = []
    lons = []

    for i in mechs:
        dates.append(i[1][0:4]+"-"+i[1][5:7]+"-"+i[1][8:]+" "+i[2])
        strikes.append(float(i[16]))
        dips.append(float(i[17]))
        rakes.append(float(i[18]))
        lats.append(float(i[7]))
        lons.append(float(i[8]))
        depths.append(float(i[9]))

    return mechs, dates, strikes, dips, rakes, lats, lons, depths


@ray.remote
def get_parallel_dc(i, targets, store_id, noised, real_noise_traces, post, pre, no_events, stations, mod, params, strikes, dips, rakes):
    engine = LocalEngine(store_superdirs=['gf_stores'])
    store = engine.get_store(store_id)
    lat, lon, depth = params
    traces_uncuts = []
    tracess = []
    sources = []
    events = []
    mag = 5
    for strike in strikes:
        for dip in dips:
            for rake in rakes:
                event = scenario.gen_random_tectonic_event(i, magmin=-0.5, magmax=3.)
                source_dc = DCSource(
                    lat=lat,
                    lon=lon,
                    depth=depth,
                    strike=strike,
                    dip=dip,
                    rake=rake,
                    magnitude=mag)
                response = engine.process(source_dc, targets)
                traces = response.pyrocko_traces()
                event.lat = source_dc.lat
                event.lon = source_dc.lon
                event.depth = source_dc.depth
                mt = moment_tensor.MomentTensor(strike=source_dc.strike, dip=source_dc.dip, rake=source_dc.rake,
                                                magnitude=source_dc.magnitude)
                event.moment_tensor = mt
                traces_uncut = copy.deepcopy(traces)
                traces_uncuts.append(traces_uncut)
                for tr in traces:
                    for st in stations:
                        if st.station == tr.station:
                            dist = (orthodrome.distance_accurate50m(source_dc.lat,
                                                                    source_dc.lon,
                                                                    st.lat,
                                                                    st.lon)+st.elevation)*cake.m2d
                            processed = False
                            while processed is False:
                                for i, arrival in enumerate(mod.arrivals([dist],
                                                            phases=get_phases_list(),
                                                            zstart=source_dc.depth)):
                                    if processed is False:
                                        tr.chop(arrival.t-pre, arrival.t+post)
                                        processed = True
                                    else:
                                        pass


                    nsamples = len(tr.ydata)
                    randdata = np.random.normal(size=nsamples)*np.min(tr.ydata)
                    white_noise = trace.Trace(deltat=tr.deltat, tmin=tr.tmin,
                                              ydata=randdata)
                    if noised is True:
                        tr.add(white_noise)
                tracess.append(traces)
                events.append(event)
                sources.append(source_dc)
    return [tracess, events, nsamples, sources, traces_uncuts]


def generate_test_data_grid(store_id, nevents=50, noised=False,
                            real_noise_traces=None, strike_min=0.,
                            strike_max=360., strike_step=2.,
                            dip_min=70., dip_max=90., dip_step=1.,
                            rake_min=-180., rake_max=-160., rake_step=1.,
                            mag_min=5.0, mag_max=5.2, mag_step=0.1,
                            depth_step=200., zmin=4000., zmax=5000.,
                            dimx=0.2, dimy=0.2, center=None, source_type="DC",
                            kappa_min=0, kappa_max=2*pi, kappa_step=0.05,
                            sigma_min=-pi/2, sigma_max=pi/2, sigma_step=0.05,
                            h_min=0, h_max=1, h_step=0.05,
                            v_min=-1/3, v_max=1/3, v_step=0.05,
                            u_min=0, u_max=(3/4)*pi, u_step=0.05,
                            pre=2., post=2., no_events=False,
                            parallel=True):

    mod = landau_layered_model()
    engine = LocalEngine(store_superdirs=['gf_stores'])
    store = engine.get_store(store_id)
    store.config.earthmodel_1d
    scale = 2e-14
    cake_phase = cake.PhaseDef("P")
    phase_list = [cake_phase]
    waveforms_events = []
    waveforms_events_uncut = []
    waveforms_noise = []
    sources = []
    #change stations
    stations = model.load_stations("ridgecrest/stations_dist.txt")
    targets = []
    events = []
    mean_lat = []
    mean_lon = []
    for st in stations:
        mean_lat.append(st.lat)
        mean_lon.append(st.lon)
        for cha in st.channels:
            if cha.name is not "R" and cha.name is not "T" and cha.name is not "Z":
                target = Target(
                        lat=st.lat,
                        lon=st.lon,
                        store_id=store_id,
                        interpolation='multilinear',
                        quantity='displacement',
                        codes=st.nsl() + (cha.name,))
                targets.append(target)
  #  latmin=35.81
  #  latmax=35.964
  #  lonmin=-117.771
 #   lonmax=-117.61
    latmin = 35.7076667
    latmax = 35.9976667
    lonmin = -117.9091667
    lonmax = -117.5091667
    center = [np.mean([latmin, latmax]), np.mean([lonmin, lonmax])]
    dim_step = 0.01
    lats, lons, depths = make_grid(center, dimx, dimy, zmin, zmax, dim_step, depth_step, )
    strikes = np.arange(strike_min, strike_max, strike_step)
    dips = np.arange(dip_min, dip_max, dip_step)
    rakes = np.arange(rake_min, rake_max, rake_step)
    kappas = np.arange(kappa_min, kappa_max, kappa_step)
    sigmas = np.arange(sigma_min, sigma_max, sigma_step)
    vs = np.arange(v_min, v_max, v_step)
    us = np.arange(u_min, u_max, u_step)
    hs = np.arange(h_min, h_max, h_step)
    magnitudes = np.arange(mag_min, mag_max, mag_step)

    i = 0
    # loop over all mechanisms needed to desribe each grid point
    if source_type == "MTQT" or source_type == "MTQT_DC":
        for lat in lats:
            for lon in lons:
                for depth in depths:
                    for kappa in kappas:
                        for v in vs:
                            for u in us:
                                for h in hs:
                                    event = scenario.gen_random_tectonic_event(i, magmin=-0.5, magmax=3.)
                                    i = i+1
                                    source_dc = MTQTSource(
                                        lat=lat,
                                        lon=lon,
                                        depth=depth,
                                        u=u,
                                        v=v,
                                        kappa=kappa,
                                        sigma=sigma,
                                        h=h,
                                    )

                                    response = engine.process(source_dc, targets)
                                    traces = response.pyrocko_traces()
                                    event.lat = source_dc.lat
                                    event.lon = source_dc.lon
                                    event.depth = source_dc.depth
                                    mt = moment_tensor.MomentTensor(strike=source_dc.strike, dip=source_dc.dip, rake=source_dc.rake,
                                                                    magnitude=source_dc.magnitude)

                                    event.moment_tensor = mt
                                    sources.append(source_dc)
                                    events.append(event)
                                    for tr in traces:
                                        for st in stations:
                                            if st.station == tr.station:
                                                dists = (orthodrome.distance_accurate50m(source_dc.lat,
                                                                                     source_dc.lon,
                                                                                     st.lat,
                                                                                     st.lon)+st.elevation)*cake.m2d
                                                processed = False
                                                for ar, arrival in enumerate(mod.arrivals([dists],
                                                                            phases=get_phases_list(),
                                                                            zstart=source_dc.depth)):
                                                    if processed is False:
                                                        tr.chop(arrival.t-pre, arrival.t+post)
                                                        processed = True
                                                    else:
                                                        pass


                                        nsamples = len(tr.ydata)
                                        randdata = np.random.normal(size=nsamples)*np.min(tr.ydata)
                                        white_noise = trace.Trace(deltat=tr.deltat, tmin=tr.tmin,
                                                                  ydata=randdata)
                                        if noised is True:
                                            tr.add(white_noise)
                                    waveforms_events.append(traces)

    mtqt_ps = []
    if source_type == "MTQT2":
        if i < 10:
            for lat in lats:
                for lon in lons:
                    for depth in depths:
                        for strike in strikes:
                            for dip in dips:
                                for rake in rakes:
                                    for mag in magnitudes:
                                    #    try:
                                            event = scenario.gen_random_tectonic_event(i, magmin=-0.5, magmax=3.)
                                            i = i+1
                                            mt = moment_tensor.MomentTensor(strike=strike, dip=dip, rake=rake,
                                                                            magnitude=mag)

                                            event.moment_tensor = mt
                                            mt_use = mt.m6_up_south_east()
                                            rho, v, u, kappa, sigma, h = cmt2tt15(mt_use)
                                            source_mtqt = MTQTSource(
                                            lon=lon,
                                                lat=lat,
                                                depth=depth,
                                                u=u,
                                                v=v,
                                                kappa=kappa,
                                                sigma=sigma,
                                                h=h,
                                            )
                                            response = engine.process(source_mtqt, targets)
                                            traces = response.pyrocko_traces()
                                            event.lat = source_mtqt.lat
                                            event.lon = source_mtqt.lon
                                            event.depth = source_mtqt.depth
                                            mtqt_ps.append([u, v, kappa, sigma, h])

                                            sources.append(source_mtqt)
                                            events.append(event)
                                            for tr in traces:
                                                for st in stations:
                                                    if st.station == tr.station:
                                                        dists = (orthodrome.distance_accurate50m(source_mtqt.lat,
                                                                                             source_mtqt.lon,
                                                                                             st.lat,
                                                                                             st.lon)+st.elevation)*cake.m2d
                                                        processed = False
                                                        for ar, arrival in enumerate(mod.arrivals([dists],
                                                                                     phases=get_phases_list(),
                                                                                     zstart=source_mtqt.depth)):
                                                            if processed is False:
                                                                tr.chop(arrival.t-pre, arrival.t+post)
                                                                processed = True
                                                            else:
                                                                pass


                                                nsamples = len(tr.ydata)
                                                randdata = np.random.normal(size=nsamples)*np.min(tr.ydata)
                                                white_noise = trace.Trace(deltat=tr.deltat, tmin=tr.tmin,
                                                                          ydata=randdata)
                                                if noised is True:
                                                    tr.add(white_noise)
                                            waveforms_events.append(traces)
                                    #    except:
                                    #        pass


    if source_type == "DC":
        if parallel is True:
            params = []
            for lat in lats:
                for lon in lons:
                    for depth in depths:
    #                    for strike in strikes:
    #                        for dip in dips:
    #                            for rake in rakes:
                                        params.append([lat, lon, depth])

            ray.init(num_cpus=num_cpus-1)
            npm = len(lats)*len(lons)*len(depths)
            print("parallel")
            results = ray.get([get_parallel_dc.remote(i, targets, store_id, noised, real_noise_traces, post, pre, no_events, stations, mod, params[i], strikes, dips, rakes) for i in range(len(params))])
            for rests in results:
                for res in rests:
                    events.append(res[1])
                    waveforms_events.append(res[0])
                    nsamples = res[2]
                    sources.append(res[3])
                    waveforms_events.uncut(res[4])
            del results, params
        else:
            for lat in lats:
                for lon in lons:
                    for depth in depths:
                        for strike in strikes:
                            for dip in dips:
                                for rake in rakes:
                                    for mag in magnitudes:
                                        #try:
                                            event = scenario.gen_random_tectonic_event(i, magmin=-0.5, magmax=3.)
                                            i = i+1
                                            source_dc = DCSource(
                                                lat=lat,
                                                lon=lon,
                                                depth=depth,
                                                strike=strike,
                                                dip=dip,
                                                rake=rake,
                                                magnitude=mag)
                                            response = engine.process(source_dc, targets)
                                            traces = response.pyrocko_traces()
                                            event.lat = source_dc.lat
                                            event.lon = source_dc.lon
                                            event.depth = source_dc.depth
                                            mt = moment_tensor.MomentTensor(strike=source_dc.strike, dip=source_dc.dip, rake=source_dc.rake,
                                                                            magnitude=source_dc.magnitude)

                                            event.moment_tensor = mt
                                            sources.append(source_dc)
                                            events.append(event)
                                            for tr in traces:
                                                for st in stations:
                                                    if st.station == tr.station:
                                                        dist = (orthodrome.distance_accurate50m(source_dc.lat,
                                                                                             source_dc.lon,
                                                                                             st.lat,
                                                                                             st.lon)+st.elevation)#*cake.m2d
                                                        processed = False
                                                        depth = source_dc.depth
                                                        arrival = store.t('begin', (depth, dist))
                                                        if processed is False:
                                                            tr.chop(arrival-pre, arrival+post)
                                                            processed = True
                                                        else:
                                                            pass


                                                nsamples = len(tr.ydata)
                                                randdata = np.random.normal(size=nsamples)*np.min(tr.ydata)
                                                white_noise = trace.Trace(deltat=tr.deltat, tmin=tr.tmin,
                                                                          ydata=randdata)
                                                if noised is True:
                                                    tr.add(white_noise)
                                            waveforms_events.append(traces)
                                    #except:
                                    #    pass
        # same number of non-events
        if no_events is True:
            for i in range(0, nevents):
                try:
                    source_dc = DCSource(
                        lat=scenario.randlat(49., 49.2),
                        lon=scenario.rand(8.1, 8.2),
                        depth=scenario.rand(100., 3000.),
                        strike=scenario.rand(0., 360.),
                        dip=scenario.rand(0., 90.),
                        rake=scenario.rand(-180., 180.),
                        magnitude=scenario.rand(-1., 0.1))

                    response = engine.process(source_dc, targets)
                    traces = response.pyrocko_traces()
                    for tr in traces:
                        for st in stations:
                            if st.station == tr.station:
                                dists = (orthodrome.distance_accurate50m(source_dc.lat,
                                                                         source_dc.lon,
                                                                         st.lat,
                                                                         st.lon)+st.elevation)*cake.m2d
                                processed = False
                                for ar, arrival in enumerate(mod.arrivals([dists],
                                                            phases=get_phases_list(),
                                                            zstart=source_dc.depth)):
                                    if processed is False:
                                        tr.chop(arrival.t-pre, arrival.t+post)
                                        processed = True
                                    else:
                                        pass
                        tr.ydata = tr.ydata*0.

                        nsamples = len(tr.ydata)
                        randdata = np.random.normal(size=nsamples)*scale
                        white_noise = trace.Trace(deltat=tr.deltat, tmin=tr.tmin,
                                                  ydata=randdata)
                        tr.add(white_noise)
                    waveforms_noise.append(traces)

                except:
                    pass

    return waveforms_events, waveforms_noise, nsamples, len(stations), events, sources, mtqt_ps


def m6_ridgecrest():
    return [-0.25898825,  0.61811539, -0.35912714, -0.67312731,  0.35961476,  0.35849677]


def rgb2gray(rgb):
    return np.dot(rgb[..., :3], [0.2989, 0.5870, 0.1140])


def load_data(data_dir, store_id, scenario=False, stations=None, pre=2.,
              post=2.):
    mod = landau_layered_model()
    engine = LocalEngine(store_superdirs=['/home/asteinbe/gf_stores'])
    store = engine.get_store(store_id)
    store.config.earthmodel_1d
    cake_phase = cake.PhaseDef("P")
    phase_list = [cake_phase]
    from pathlib import Path
    events = []
    waveforms = []
    if scenario is True:
        pathlist = Path(data_dir).glob('scenario*/')
    else:
        pathlist = Path(data_dir).glob('ev*/')
    for path in sorted(pathlist):
        try:
            targets = []
            path = str(path)+"/"
            traces_event = []
            event = model.load_events(path+"event.txt")[0]
            traces_loaded = io.load(path+"traces.mseed")
            if scenario is True:
                stations = model.load_stations(path+"stations.pf")
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
            well_event = False
            for tr in traces_loaded:
                for st in stations:
                    if st.station == tr.station:
                        traces.append(tr)
            if len(event.tags) > 0:
                if event.tags[0] == "no_event":
                    for tr in traces:
                        nsamples = len(tr.ydata)

                        tr.chop(tr.tmin-pre, tr.tmin+post)
                        nsamples = len(tr.ydata)
                    traces_event.append(traces)
                else:
                    well_event = True
            if len(event.tags) == 0 or well_event is True:
                for tr in traces:
                    nsamples = len(tr.ydata)
                    for st in stations:
                        if st.station == tr.station:
                            dists = (orthodrome.distance_accurate50m(event.lat,
                                                                     event.lon,
                                                                     st.lat,
                                                                     st.lon)+st.elevation)*cake.m2d
                            processed = False
                            for i, arrival in enumerate(mod.arrivals([dists],
                                                        phases=get_phases_list(),
                                                        zstart=event.depth)):
                                if processed is False:
                                    tr.chop(tr.tmin+arrival.t-pre, tr.tmin+arrival.t+post)
                                    processed = True
                                else:
                                    pass
                    nsamples = len(tr.ydata)
                traces_event.append(traces)

            nsamples = len(tr.ydata)
            events.append(event)
            waveforms.append(traces)
        except FileNotFoundError:
            pass
    return waveforms, nsamples, len(stations), events


def plot_prescission(input, output):
    mislocation_rel = []
    for inp, outp in zip(input, output):
        mislocation_rel.append(inp-outp)
    mislocation_rel = np.asarray(mislocation_rel)
    plt.figure()
    plt.plot(mislocation_rel)
    plt.show()


def bnn_detector(waveforms_events=None, waveforms_noise=None, load=True,
                 multilabel=True, data_dir=None, train_model=True,
                 detector_only=False, validation_data=None, wanted_start=None,
                 wanted_end=None, mode="detector_only"):
    import _pickle as pickle
    import cProfile, pstats
    pr = cProfile.Profile()
    pr.enable()
    if detector_only is True:
        multilabel = False
    if mode == "mechanism_mode":
        mechanism = True
    else:
        mechanism = False

    if mechanism is False:
        if data_dir is not None:
            try:
                f = open("data_unseen_waveforms_bnn_gt_loaded", 'rb')
                waveforms_events, nsamples, nstations, events = pickle.load(f)
                f.close()
            except:
                waveforms_events, nsamples, nstations, events = load_data(data_dir,
                                                                          "mojave_large_ml")
                f = open("data_unseen_waveforms_bnn_gt_loaded", 'wb')
                pickle.dump([waveforms_events, nsamples, nstations, events], f)
                f.close()
            waveforms_unseen = waveforms_events
            events_unseen = events
        if load is True:
            try:
                f = open("data_waveforms_bnn_gt_loaded", 'rb')
                waveforms_events, nsamples, nstations, events = pickle.load(f)
                f.close()
            except:
                data_dir = "./shaky"
                waveforms_events, nsamples, nstations, events = load_data(data_dir, "mojave_large_ml")
                f = open("data_waveforms_bnn_gt_loaded", 'wb')
                pickle.dump([waveforms_events, nsamples, nstations, events], f)
                f.close()
        else:
            try:
                f = open("data_waveforms_bnn_gt", 'rb')
                waveforms_events, waveforms_noise, nsamples, nstations, events = pickle.load(f)
                f.close()
            except:

                waveforms_events, waveforms_noise, nsamples, nstations, events = generate_test_data("mojave_large_ml", nevents=1200)
                f = open("data_waveforms_bnn_gt", 'wb')
                pickle.dump([waveforms_events, waveforms_noise, nsamples, nstations,
                             events], f)
                f.close()
            sources = None
    else:
        try:
            print("loading")
            f = open("data_waveforms_bnn_mechanism", 'rb')
            waveforms_events, waveforms_noise, nsamples, nstations, events, sources, mtqt_ps = pickle.load(f)
            f.close()
        except:

            waveforms_events, waveforms_noise, nsamples, nstations, events, sources, mtqt_ps = generate_test_data_grid("mojave_large_ml", nevents=1200)
            f = open("data_waveforms_bnn_mechanism", 'wb')
            print("dump")
            pickle.dump([waveforms_events, waveforms_noise, nsamples, nstations,
                         events, sources, mtqt_ps], f)
            f.close()
    pr.disable()
    filename = 'profile_bnn.prof'
    pr.dump_stats(filename)
    if validation_data is None:
        max_traces = 0.
        for traces in waveforms_events:
            for tr in traces:
                if np.max(tr.ydata) > max_traces:
                    max_traces = np.max(tr.ydata)

        data_events, labels_events, nstations, nsamples = bnn_detector_data(waveforms_events, max_traces, events=events, multilabel=multilabel, mechanism=mechanism, sources=sources)
#        print(len(data_events))
#        print(labels_events)
        if data_dir is not None:
            data_events_unseen, labels_events_unseen, nstations_unseen, nsamples_unseen = bnn_detector_data(waveforms_unseen, max_traces, events=events_unseen, multilabel=multilabel,
                                                                                                        mechanism=mechanism)
        if detector_only is True:
            data_noise, labels_noise, nstations, nsamples = bnn_detector_data(waveforms_noise, max_traces, events=None, multilabel=multilabel, mechanism=mechanism)
            x_data = np.concatenate((data_events, data_noise), axis=0)
            y_data = np.concatenate((labels_events, labels_noise), axis= 0)
            #x_data = data_events
            #y_data = labels_events
            from keras.utils import to_categorical
            y_array = None
        else:
            x_data = data_events
            y_data = labels_events
    #        print(len(x_data))
    #        print(len(y_data))

    else:
        # hardcoded for bgr envs
        trace_comp_event = waveforms_events[0][0]
        gf_freq = trace_comp_event.deltat
        waveforms_unseen, stations_unseen = waveform.load_data_archieve(validation_data,
                                                                        gf_freq=gf_freq,
                                                                        wanted_start=wanted_start,
                                                                        wanted_end=wanted_end)
        max_traces = 0.
        calculate_max = False
        nstations = 0
        for st in stations_unseen:
            st_len = len(st)
            if st_len > nstations:
                nstations = st_len
        if calculate_max is True:
            for traces in waveforms_unseen:
                for tr in traces:
                    if np.max(tr.ydata) > max_traces:
                        max_traces = np.max(tr.ydata)
        data_events_unseen, labels_events_unseen, nstations_unseen, nsamples_unseen = bnn_detector_data(waveforms_unseen, max_traces, events=None, multilabel=multilabel, mechanism=mechanism)

        x_data = data_events_unseen
        y_data = labels_events_unseen
    ncomponents = 3
    nstations = nstations*ncomponents

    if multilabel is False:
        nlabels = 1
    else:
        nlabels = (y_data.shape[1])
    dat = x_data.copy()
    labels = y_data.copy()
    print('shape of x_data: ', x_data.shape)
    print('shape of y_data: ', y_data.shape)
    dat = dat[::2]
    #print(np.shape(dat[0]))
    #dat[2][0:2400] = np.ones(2400)*-1
    labels = labels[::2]

    np.random.seed(42)  # Set a random seed for reproducibility

    headline_data = dat
    headline_labels = labels
    additional_labels = labels
    additional_data = labels
    from sklearn.model_selection import train_test_split
    from keras.models import Sequential
    from keras.layers import Dense, Activation, Masking
    from keras.layers import Dense, Dropout, Activation
# For a single-input model with 2 classes (binary classi    fication):
    if train_model is True:
        #print(nstations)
        model = Sequential()
        model.add(Activation('relu'))

#        model.add(Dense(2056, activation='relu', input_dim=nsamples))

    #    model.add(Dense(1060, activation='relu', input_dim=nsamples))
    #    model.add(Dropout(0.5))
        #model.add(Masking(mask_value=-99999999))
    #    model.add(Dense(3600, activation='relu', input_dim=nsamples))
#        model.add(Dropout(0.5))
    #    model.add(Dense(64, activation='relu', input_dim=nsamples))
    #    model.add(Dropout(0.5))
        model.add(Dense(300, activation='relu', input_dim=nsamples))
    #    model.add(Dense(100, activation='relu', input_dim=nsamples))
    #    model.add(Dense(21, activation='relu', input_dim=nsamples))

        #model.add(Dense(11664, activation='relu', input_dim=nsamples))
    #    model.add(Dense(36, activation='relu', input_dim=nsamples))
    #    model.add(Dense(12, activation='relu', input_dim=nsamples))

#        model.add(Dense(38, activation='relu', input_dim=nsamples))
    #    model.add(Dense(3600, activation='relu', input_dim=nsamples))
    #    model.add(Dense(36, activation='relu', input_dim=nsamples))
    #    model.add(Dense(12, activation='relu', input_dim=nsamples))

    #    model.add(Dense(nstations, activation='relu', input_dim=nsamples))
    #    model.add(Dense(3600, activation='relu', input_dim=nsamples))

    #    model.add(Dropout(0.5))
    #    model.add(Dropout(0.5))
    #    model.add(Dense(int(nstations/ncomponents), activation='relu', input_dim=nsamples))
#        model.add(Dropout(0.5))

    #    model.add(Dense(8, activation='relu', input_dim=nsamples))
    #    model.add(Dense(5, activation='relu', input_dim=nsamples))
    #    model.add(Dropout(0.5))

#        model.add(Dense(3, activation='relu', input_dim=nsamples))
    #    model.add(Dense(64, activation='relu', input_dim=nsamples))

        model.add(Dense(nlabels, activation='sigmoid'))
        # adadelta
        #opt = keras.optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=True)
        model.compile(optimizer='rmsprop',
                      loss='binary_crossentropy',
                      metrics=['accuracy'])
    #    model.compile(optimizer='rmsprop',
    #                  loss='categorical_crossentropy',
    #                  metrics=['accuracy'])
        from keras.optimizers import SGD
    #    sgd = SGD(lr=0.001, decay=1e-6, momentum=0.9, nesterov=True)
    #    model.compile(optimizer=sgd,
    #                  loss='binary_crossentropy',
    #                  metrics=['accuracy'])
        #model.compile(loss='categorical_crossentropy',
        #              optimizer=sgd,
        #              metrics=['accuracy'])
        # Generate dummy data
    data = dat

    viz_steps = 2
    #pred = model.predict(train)
    bayesian = False

    if bayesian is False:
        train, x_val, y_train, y_val = train_test_split(dat, labels,
                                                        test_size=0.05,
                                                        random_state=10)
        x_val = dat
        y_val = labels
        # Train the model, iterating on the data in batches of 32 samples
        from keras.callbacks import ModelCheckpoint
        checkpointer = ModelCheckpoint(filepath="best_weights.hdf5",
                                       monitor = 'val_accuracy',
                                       verbose=1,
                                       save_best_only=True)

        if train_model is True:
            if detector_only is True:
                history = model.fit(train, y_train, epochs=5, batch_size=400,
                                    callbacks=[checkpointer])
            else:
                history = model.fit(dat, labels, epochs=600, batch_size=400,
                                    callbacks=[checkpointer])

            plot_model(model)
        #    layer_outputs = [layer.output for layer in model.layers[:]]
            #  Extracts the outputs of the top 12 layers
        #    activation_model = models.Model(inputs=model.input, outputs=layer_outputs)
        #    plot_acc_loss(history)
        else:
            if detector_only is True:
                model = keras.models.load_model('model_detector')
            if mode == "mechanism_mode":
                model = keras.models.load_model('model_mechanism')
            else:
                model = keras.models.load_model('model_locator')
        if data_dir is not None or validation_data is not None:
            pred = model.predict(data_events_unseen)
        else:
            pred = model.predict(x_val)
    #    print(np.shape(pred))
    #    print("here", pred)

    else:
        train, x_val, y_train, y_val = train_test_split(dat,
                                                        labels,
                                                        test_size=0.2,
                                                        random_state=10)

        num_epochs = 5
        batchsize = 32
        num_monte_carlo = 5
        step = 0
        for epoch in range(num_epochs):
            epoch_accuracy, epoch_loss = [], []
            for i in range(len(y_train) // batchsize):
                batch_x = train[i * batchsize: (i + 1) * batchsize]
                batch_y = y_train[i * batchsize: (i + 1) * batchsize]
                model.fit(batch_x, batch_y)

                if (step+1) % viz_steps == 0:
                # Compute log prob of heldout set by averaging draws from the model:
                # p(heldout | train) = int_model p(heldout|model) p(model|train)
                #                   ~= 1/n * sum_{i=1}^n p(heldout | model_i)
                # where model_i is a draw from the posterior p(model|train).
                    print(' ... Running monte carlo inference')
                    probs = tf.stack([model.predict(x_val)
                                      for _ in range(num_monte_carlo)], axis=0)
                    mean_probs = tf.reduce_mean(probs, axis=0)
                    heldout_log_prob = tf.reduce_mean(tf.math.log(mean_probs))
                    print(' ... Held-out nats: {:.3f}'.format(heldout_log_prob))
                    if HAS_SEABORN:
                      names = [layer.name for layer in model.layers]
                     # qm_vals = [layer.kernel_posterior.mean()
                    #             for layer in model.layers]
                     # qs_vals = [layer.kernel_posterior.stddev()
                    #             for layer in model.layers]
                     # plot_weight_posteriors(names, qm_vals, qs_vals,
                    #                         fname=os.path.join(
                    #                             ".",
                    #                             'epoch{}_step{:05d}_weights.png'.format(
                    #                                 epoch, step)))
                      plot_heldout_prediction(x_val, probs,
                                              fname=os.path.join(
                                                  ".",
                                                  'epoch{}_step{}_pred.png'.format(
                                                      epoch, step)),
                                              title='mean heldout logprob {:.2f}'
                                              .format(heldout_log_prob))

                step = step+1
    print(y_val[0:3], pred[0:3])
    print(y_val[-3:-1], pred[-3:-1])

    #print(abs(y_val)-abs(pred))
#    print(np.sum(abs(y_val)-abs(pred)))
    recover_real_value = False
    sdr = False
    if multilabel is True:
        if mechanism is True:
            if sdr is False:
                real_values = []
                real_ms = []
                for i, vals in enumerate(y_val[0:2]):
                    real_value = [(1.-(2.*vals[0]))*events[i].moment_tensor.moment*2, (1.-(2.*vals[1]))*events[i].moment_tensor.moment*2,
                                        (1.-(2.*vals[2]))*events[i].moment_tensor.moment*2, (1.-(2.*vals[3]))*events[i].moment_tensor.moment*2,
                                        (1.-(2.*vals[4]))*events[i].moment_tensor.moment*2, (1.-(2.*vals[5]))*events[i].moment_tensor.moment*2]
                    real_values.append(real_value)
                    m = moment_tensor.MomentTensor(mnn=(1.-(2.*vals[0]))*events[i].moment_tensor.moment*2, mee=(1.-(2.*vals[1]))*events[i].moment_tensor.moment*2,
                                        mdd=(1.-(2.*vals[2]))*events[i].moment_tensor.moment*2, mne=(1.-(2.*vals[3]))*events[i].moment_tensor.moment*2,
                                        mnd=(1.-(2.*vals[4]))*events[i].moment_tensor.moment*2, med=(1.-(2.*vals[5]))*events[i].moment_tensor.moment*2)
                    #print(real_value)
                    #print(vals)
                #    print(m.both_strike_dip_rake())
                #    print(events[i])
                #    print(sources[i])
                    real_ms.append(m)
                real_values_pred = []
                diff_values = []
                pred_ms = []
                for i, vals in enumerate(pred[0:2]):
                    real_value = [(1.-(2.*vals[0]))*events[i].moment_tensor.moment*2, (1.-(2.*vals[1]))*events[i].moment_tensor.moment*2,
                                        (1.-(2.*vals[2]))*events[i].moment_tensor.moment*2, (1.-(2.*vals[3]))*events[i].moment_tensor.moment*2,
                                        (1.-(2.*vals[4]))*events[i].moment_tensor.moment*2, (1.-(2.*vals[5]))*events[i].moment_tensor.moment*2]
                    real_values_pred.append(real_value)
                    m = moment_tensor.MomentTensor(mnn=(1.-(2.*vals[0]))*events[i].moment_tensor.moment*2, mee=(1.-(2.*vals[1]))*events[i].moment_tensor.moment*2,
                                        mdd=(1.-(2.*vals[2]))*events[i].moment_tensor.moment*2, mne=(1.-(2.*vals[3]))*events[i].moment_tensor.moment*2,
                                        mnd=(1.-(2.*vals[4]))*events[i].moment_tensor.moment*2, med=(1.-(2.*vals[5]))*events[i].moment_tensor.moment*2)
                #    print(m.both_strike_dip_rake())
                #    print(real_value)
                    pred_ms.append(m)
                    #diff_values.append([real_values[i][0]-real_values_pred[i][0], real_values[i][1]-real_values_pred[i][1], real_values[i][2]-real_values_pred[i][2]])
            for pred_m, real_m in zip(pred_ms, real_ms):
                from pyrocko import plot
                from pyrocko.plot import beachball
                omega = omega_angle(real_m.m6(), pred_m.m6())
                kagan = moment_tensor.kagan_angle(real_m, pred_m)
                fig = plt.figure()
                axes = fig.add_subplot(1, 1, 1)
                axes.set_xlim(-2., 4.)
                axes.set_ylim(-2., 2.)
                axes.set_axis_off()
                plot.beachball.plot_beachball_mpl(
                            real_m,
                            axes,
                            beachball_type='deviatoric',
                            size=60.,
                            position=(0, 1),
                            color_t=plot.mpl_color('scarletred2'),
                            linewidth=1.0)
                plot.beachball.plot_beachball_mpl(
                            pred_m,
                            axes,
                            beachball_type='deviatoric',
                            size=60.,
                            position=(1.5, 1),
                            color_t=plot.mpl_color('scarletred2'),
                            linewidth=1.0)
                plt.show()
                print(omega, kagan)

                real_values = []
                real_ms = []
                for i, vals in enumerate(y_val[-3:-1]):
                    real_value = [(1.-(2.*vals[0]))*events[i].moment_tensor.moment*2, (1.-(2.*vals[1]))*events[i].moment_tensor.moment*2,
                                        (1.-(2.*vals[2]))*events[i].moment_tensor.moment*2, (1.-(2.*vals[3]))*events[i].moment_tensor.moment*2,
                                        (1.-(2.*vals[4]))*events[i].moment_tensor.moment*2, (1.-(2.*vals[5]))*events[i].moment_tensor.moment*2]
                    real_values.append(real_value)
                    m = moment_tensor.MomentTensor(mnn=(1.-(2.*vals[0]))*events[i].moment_tensor.moment*2, mee=(1.-(2.*vals[1]))*events[i].moment_tensor.moment*2,
                                        mdd=(1.-(2.*vals[2]))*events[i].moment_tensor.moment*2, mne=(1.-(2.*vals[3]))*events[i].moment_tensor.moment*2,
                                        mnd=(1.-(2.*vals[4]))*events[i].moment_tensor.moment*2, med=(1.-(2.*vals[5]))*events[i].moment_tensor.moment*2)
                    #print(real_value)
                    #print(vals)
                #    print(m.both_strike_dip_rake())
                #    print(events[i])
                #    print(sources[i])
                    real_ms.append(m)
                real_values_pred = []
                diff_values = []
                pred_ms = []
                for i, vals in enumerate(pred[-3:-1]):
                    real_value = [(1.-(2.*vals[0]))*events[i].moment_tensor.moment*2, (1.-(2.*vals[1]))*events[i].moment_tensor.moment*2,
                                        (1.-(2.*vals[2]))*events[i].moment_tensor.moment*2, (1.-(2.*vals[3]))*events[i].moment_tensor.moment*2,
                                        (1.-(2.*vals[4]))*events[i].moment_tensor.moment*2, (1.-(2.*vals[5]))*events[i].moment_tensor.moment*2]
                    real_values_pred.append(real_value)
                    m = moment_tensor.MomentTensor(mnn=(1.-(2.*vals[0]))*events[i].moment_tensor.moment*2, mee=(1.-(2.*vals[1]))*events[i].moment_tensor.moment*2,
                                        mdd=(1.-(2.*vals[2]))*events[i].moment_tensor.moment*2, mne=(1.-(2.*vals[3]))*events[i].moment_tensor.moment*2,
                                        mnd=(1.-(2.*vals[4]))*events[i].moment_tensor.moment*2, med=(1.-(2.*vals[5]))*events[i].moment_tensor.moment*2)
                #    print(m.both_strike_dip_rake())
                #    print(real_value)
                    pred_ms.append(m)
                    #diff_values.append([real_values[i][0]-real_values_pred[i][0], real_values[i][1]-real_values_pred[i][1], real_values[i][2]-real_values_pred[i][2]])
            for pred_m, real_m in zip(pred_ms, real_ms):
                from pyrocko import plot
                from pyrocko.plot import beachball
                omega = omega_angle(real_m.m6(), pred_m.m6())
                kagan = moment_tensor.kagan_angle(real_m, pred_m)
                fig = plt.figure()
                axes = fig.add_subplot(1, 1, 1)
                axes.set_xlim(-2., 4.)
                axes.set_ylim(-2., 2.)
                axes.set_axis_off()
                plot.beachball.plot_beachball_mpl(
                            real_m,
                            axes,
                            beachball_type='deviatoric',
                            size=60.,
                            position=(0, 1),
                            color_t=plot.mpl_color('scarletred2'),
                            linewidth=1.0)
                plot.beachball.plot_beachball_mpl(
                            pred_m,
                            axes,
                            beachball_type='deviatoric',
                            size=60.,
                            position=(1.5, 1),
                            color_t=plot.mpl_color('scarletred2'),
                            linewidth=1.0)
                plt.show()
                print(omega, kagan)
            mechanism_and_location = True
            if mechanism_and_location is True:
                for pred_m, real_m in zip(pred_ms, real_ms):

                    lons = []
                    lats = []
                    depths = []
                    for i, ev in enumerate(events):
                        lats.append(ev.lat)
                        lons.append(ev.lon)
                        depths.append(ev.depth)
                    lats = np.asarray(lats)
                    lats_max = np.max(lats)
                    lats_min = np.min(lats)
                    lons = np.asarray(lons)
                    lons_max = np.max(lons)
                    lons_min = np.min(lons)
                    depths = np.asarray(depths)
                    depths_max = np.max(depths)
                    depths_min = np.min(depths)
                    real_values = []
                    #lons = (lons-np.min(lons))/(np.max(lons)-np.min(lons))
                    #vals*(lons_max-lons_min)/lons_min
                    for i, vals in enumerate(y_val):
                        lat = (-vals[6]*lats_min)+(vals[6]*lats_max)+lats_min
                        lon =  (-vals[7]*lons_min)+(vals[7]*lons_max)+lons_min
                        depth = (-vals[8]*depths_min)+(vals[8]*depths_max)+depths_min

                        real_values.append([lat, lon, depth])
                    real_values_pred = []
                    diff_values = []
                    for i, vals in enumerate(pred):
                        lat = (-vals[6]*lats_min)+(vals[6]*lats_max)+lats_min
                        lon = (-vals[7]*lons_min)+(vals[7]*lons_max)+lons_min
                        depth = (-vals[8]*depths_min)+(vals[8]*depths_max)+depths_min
                        real_values_pred.append([lat, lon, depth])
                        diff_values.append([real_values[i][0]-real_values_pred[i][0], real_values[i][1]-real_values_pred[i][1], real_values[i][2]-real_values_pred[i][2]])
                #    print("location", real_values, real_values_pred)
                #    print(diff_values)
                    print(np.max(diff_values), np.min(diff_values))
            if sdr is True:
                strikes = []
                dips = []
                rakes = []
                for i, ev in enumerate(events):
                    strikes.append(ev.moment_tensor.strike1)
                    dips.append(ev.moment_tensor.dip1)
                    rakes.append(ev.moment_tensor.rake1)
                real_values = []
                for i, vals in enumerate(y_val):
                    real_values.append([vals[0]*360., vals[1]*90.,
                                        180.-(360.*vals[2])])
                real_values_pred = []
                diff_values = []
                for i, vals in enumerate(pred):
                    real_values_pred.append([vals[0]*360., vals[1]*90., 180.-(360.*vals[2])])
                    diff_values.append([real_values[i][0]-real_values_pred[i][0], real_values[i][1]-real_values_pred[i][1], real_values[i][2]-real_values_pred[i][2]])
                plot_beachball = False
                if plot_beachball is True:
                    from pyrocko import gf, trace, plot, beachball, util, orthodrome, model
                    for i, vals in enumerate(pred):
                        fig = plt.figure()
                        axes = fig.add_subplot(1, 1, 1)
                        axes.set_axis_off()
                        plot.beachball.plot_beachball_mpl(
                                    mt_val,
                                    axes,
                                    beachball_type="'full'",
                                    size=60.,
                                    position=(0, 1),
                                    color_t=plot.mpl_color('scarletred2'),
                                    linewidth=1.0)
                        plot.beachball.plot_beachball_mpl(
                                    mt_pred,
                                    axes,
                                    beachball_type="'full'",
                                    size=60.,
                                    position=(0, 1),
                                    color_t=plot.mpl_color('scarletred2'),
                                    linewidth=1.0)
                        plt.show()
                else:
                    lons = []
                    lats = []
                    depths = []
                    for i, ev in enumerate(events):
                        lats.append(ev.lat)
                        lons.append(ev.lon)
                        depths.append(ev.depth)
                    lats = np.asarray(lats)
                    lats_max = np.max(lats)
                    lats_min = np.min(lats)
                    lons = np.asarray(lons)
                    lons_max = np.max(lons)
                    lons_min = np.min(lons)
                    depths = np.asarray(depths)
                    depths_max = np.max(depths)
                    depths_min = np.min(depths)
                    real_values = []
                    #lons = (lons-np.min(lons))/(np.max(lons)-np.min(lons))
                    #vals*(lons_max-lons_min)/lons_min
                    for i, vals in enumerate(y_val):
                        lat = (-vals[0]*lats_min)+(vals[0]*lats_max)+lats_min
                        lon =  (-vals[1]*lons_min)+(vals[1]*lons_max)+lons_min
                        depth = (-vals[2]*depths_min)+(vals[2]*depths_max)+depths_min

                        real_values.append([lat, lon, depth])
                    real_values_pred = []
                    diff_values = []
                    for i, vals in enumerate(pred):
                        lat = (-vals[0]*lats_min)+(vals[0]*lats_max)+lats_min
                        lon = (-vals[1]*lons_min)+(vals[1]*lons_max)+lons_min
                        depth = (-vals[2]*depths_min)+(vals[2]*depths_max)+depths_min
                        real_values_pred.append([lat, lon, depth])
                        diff_values.append([real_values[i][0]-real_values_pred[i][0], real_values[i][1]-real_values_pred[i][1], real_values[i][2]-real_values_pred[i][2]])

        #    print(real_values)
        #    print(real_values_pred)
        #    print(diff_values)

    plot_prescission(y_val, pred)
    if multilabel is True and mechanism is False:
        fig = plt.figure()
        ax.scatter(diff_values[:][0], diff_values[:][1])
        plt.show()
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        for i, vals in enumerate(pred):
            ax.scatter(real_values[i][0], real_values[i][1], real_values[i][2], c="k")
            ax.scatter(real_values_pred[i][0], real_values_pred[i][1], real_values_pred[i][2], c="r")
        plt.show()
    if detector_only is True:
        model.save('model_detector')
    elif mechanism is True:
        model.save('model_mechanism')
    else:
        model.save('model_locator')


def layer_activation(model):
    from keras import backend as K
    inp = model.input
    outputs = [layer.output for layer in model.layers]          # all layer outputs
    functors = [K.function([inp, K.learning_phase()], [out]) for out in outputs]    # evaluation functions
    test = x_val
    layer_outs = [func([test, 1.]) for func in functors]


def prepare_inputs(X_train, X_test):
    from sklearn.preprocessing import OrdinalEncoder
    oe = OrdinalEncoder()
    oe.fit(X_train)
    X_train_enc = oe.transform(X_train)
    X_test_enc = oe.transform(X_test)
    return X_train_enc, X_test_enc


def generate_arrays_from_file(path):
    while 1:
        f = open(path)
        for line in f:
            # create numpy arrays of input data
            # and labels, from each line in the file
            x, y = process_line(line)
            img = load_images(x)
            yield (img, y)
        f.close()


def plot_model(model):
    from keras.utils.vis_utils import plot_model
    plot_model(model, to_file='model_plot.png', show_shapes=True, show_layer_names=True)


def plot_acc_loss(history):
#    print(history.history)
    acc = history.history['accuracy']
    val_acc = history.history['accuracy']
    loss = history.history['loss']
    val_loss = history.history['loss']
    epochs = range(1, len(acc) + 1)
    plt.plot(epochs, acc, 'bo', label='Training acc')
    plt.plot(epochs, val_acc, 'b', label='Validation acc')
    plt.title('Training and validation accuracy')
    plt.legend()
    plt.figure()
    plt.plot(epochs, loss, 'bo', label='Training loss')
    plt.plot(epochs, val_loss, 'b', label='Validation loss')
    plt.title('Training and validation loss')
    plt.legend()
    plt.show()

def fit_from_file():
    model.fit_generator(generate_arrays_from_file('/my_file.txt'),
        samples_per_epoch=10000, nb_epoch=10)
