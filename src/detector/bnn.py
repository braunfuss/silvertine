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
from pyrocko import model, cake, orthodrome
from silvertine.util.ref_mods import landau_layered_model
from pyrocko import moment_tensor, util
from keras.layers import Conv1D, MaxPooling1D, Input
from keras import models
from PIL import Image
from pathlib import Path

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


def bnn_detector_data(waveforms, max_traces, events=True, multilabel=False):
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
            tr.lowpass(4, 13.)
            tr.highpass(4, 0.03)
            tr.ydata = tr.ydata/np.max(tr.ydata)
            nsamples = len(tr.ydata)
            data = tr.ydata

            if nsamples != maxsamples:
                data = np.pad(data, (0, maxsamples-nsamples), 'constant')
                nsamples = len(data)
                tr.ydata = data

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
                        labels.append([0, 0 , 0])

                    else:
                        tag = 1
                        labels.append([lats[i], lons[i], depths[i]])

                else:
                    tag = 1
                #labels.append([tag, ev.moment_tensor.strike1/360., ev.moment_tensor.dip1/90.])
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
    return data_traces, labels, len(data_traces)/nsamples, nsamples


def generate_test_data(store_id, nevents=50, noised=True):
    mod = landau_layered_model()
    engine = LocalEngine(store_superdirs=['gf_stores'])
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
            target = Target(
                    lat=st.lat,
                    lon=st.lon,
                    store_id=store_id,
                    interpolation='multilinear',
                    quantity='displacement',
                    codes=st.nsl() + (cha.name,))
            targets.append(target)
    for i in range(0, nevents):
        try:
            event = scenario.gen_random_tectonic_event(i, magmin=-0.5, magmax=3.)
            source_dc = DCSource(
                lat=scenario.randlat(49., 49.3),
                lon=scenario.rand(8.0, 8.3),
                depth=scenario.rand(100., 13000.),
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
                                tr.chop(arrival.t-2, arrival.t+2)
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
        except:
            pass
    # same number of non-events
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


def rgb2gray(rgb):
    return np.dot(rgb[...,:3], [0.2989, 0.5870, 0.1140])


def load_data(data_dir, store_id):
    mod = landau_layered_model()
    engine = LocalEngine(store_superdirs=['/home/asteinbe/gf_stores'])
    cake_phase = cake.PhaseDef("P")
    phase_list = [cake_phase]
    from pathlib import Path
    events = []
    waveforms = []
    pathlist = Path(data_dir).glob('scenario*/')
    for path in sorted(pathlist):
        try:
            targets = []
            path = str(path)+"/"
            traces_event = []
            event = model.load_events(path+"event.txt")[0]
            traces = io.load(path+"traces.mseed")
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
            if len(event.tags) > 0:
                if event.tags[0] == "no_event":
                    for tr in traces:
                        nsamples = len(tr.ydata)

                        tr.chop(tr.tmin, tr.tmin+4)
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
                                    tr.chop(tr.tmin+arrival.t-2, tr.tmin+arrival.t+2)
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


def load_data_archieve(validation_data, gf_freq, duration=4,
                       wanted_start=None, wanted_end=None):
    folder = validation_data
    pathlist = Path(folder).glob('day*')
    waveforms = []
    stations = []
    print(wanted_end, wanted_start)
    if wanted_start is not None:
        wanted_start = util.stt(wanted_start)
        wanted_end = util.stt(wanted_end)
    print(wanted_end, wanted_start)

    from pyrocko import pile
    paths = []
    safecon = 0
    for path in sorted(pathlist):
        path = str(path)
        d2 = float(str(path)[-12:])
        d1 = float(str(path)[-25:-13])
        if wanted_start is not None:
            if (d1 >= wanted_start and d2 <= wanted_end) or (d2-wanted_end<86400. and d2-wanted_end>0. and safecon == 0):
                #traces = io.load(path+"/waveforms/rest/traces.mseed")
                st = model.load_stations(path+"/waveforms/stations.raw.txt")

                d_diff = d2 - d1
                tr_packages = int(d_diff/duration)
                #for tr in traces:
                #    tr.downsample_to(gf_freq)
        #        if safecon == 0:
                print(path)

                pathlist_waveform_files = Path(path+"/waveforms/rest/").glob('*.mseed')
                for path_wave in sorted(pathlist_waveform_files):
        #                if

        #        if safecon != 0:
                    paths.append(str(path_wave))
                safecon += 1

    p = pile.make_pile(paths)
    for traces in p.chopper(tmin=wanted_start, tinc=duration):
        if traces:
            if traces[0].tmax < wanted_end:
            #    for i in range(0, tr_packages):
            #        traces = traces
                for tr in traces:
            #    tr.chop(tr.tmin+i*duration,
            #            tr.tmin+i*duration+duration)
                    tr.downsample_to(gf_freq)
                waveforms.append(traces)
                stations.append(st)
    #    else:
    #        traces = io.load(path+"/waveforms/rest/traces.mseed")
    #        st = model.load_stations(path+"/waveforms/stations.raw.txt")
    #        for tr in traces:
    #            tr.downsample_to(gf_freq)
    #        waveforms.append(traces)
    #        stations.append(st)
    return waveforms, stations


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
                 wanted_end=None):
    import _pickle as pickle
    if detector_only is True:
        multilabel = False

    if data_dir is not None:
        try:
            f = open("data_unseen_waveforms_bnn_gt_loaded", 'rb')
            waveforms_events, nsamples, nstations, events = pickle.load(f)
            f.close()
        except:
            waveforms_events, nsamples, nstations, events = load_data(data_dir, "landau_100hz")
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
            waveforms_events, nsamples, nstations, events = load_data(data_dir, "landau_100hz")
            f = open("data_waveforms_bnn_gt_loaded", 'wb')
            pickle.dump([waveforms_events, nsamples, nstations, events], f)
            f.close()
    else:
        try:
            f = open("data_waveforms_bnn_gt", 'rb')
            waveforms_events, waveforms_noise, nsamples, nstations, events = pickle.load(f)
            f.close()
        except:

            waveforms_events, waveforms_noise, nsamples, nstations, events = generate_test_data("landau_100hz", nevents=1200)
            f = open("data_waveforms_bnn_gt", 'wb')
            pickle.dump([waveforms_events, waveforms_noise, nsamples, nstations,
                         events], f)
            f.close()

    if validation_data is None:
        max_traces = 0.
        for traces in waveforms_events:
            for tr in traces:
                if np.max(tr.ydata) > max_traces:
                    max_traces = np.max(tr.ydata)

        data_events, labels_events, nstations, nsamples = bnn_detector_data(waveforms_events, max_traces, events=events, multilabel=multilabel)
        if data_dir is not None:
            data_events_unseen, labels_events_unseen, nstations_unseen, nsamples_unseen = bnn_detector_data(waveforms_unseen, max_traces, events=events_unseen, multilabel=multilabel)
        if detector_only is True:
            data_noise, labels_noise, nstations, nsamples = bnn_detector_data(waveforms_noise, max_traces, events=None, multilabel=multilabel)
            x_data = np.concatenate((data_events, data_noise), axis=0)
            y_data = np.concatenate((labels_events, labels_noise), axis= 0)
            #x_data = data_events
            #y_data = labels_events
            from keras.utils import to_categorical
            y_array = None
        #    nlabels = 3
            #for k in range(0, nlabels):
            #    print(np.shape(y_data))
            #    lst2 = [[item[k]] for item in y_data]
            #    print(np.shape(lst2))
            #    X_train_enc, X_test_enc_1 = prepare_inputs(lst2, lst2)
            #    y_data_vec = to_categorical(X_train_enc)
            #    y_data_vec = np.asarray(y_data_vec)
            #    if y_array is None:
        #            y_array = y_data_vec
            #    else:
            #        y_array = np.concatenate((y_array, y_data_vec), axis=0)
        #    y_data = y_array
        else:
            x_data = data_events
            y_data = labels_events

    else:
        # hardcoded for bgr envs
        trace_comp_event = waveforms_events[0][0]
        gf_freq = trace_comp_event.deltat
        waveforms_unseen, stations_unseen = load_data_archieve(validation_data,
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
        data_events_unseen, labels_events_unseen, nstations_unseen, nsamples_unseen = bnn_detector_data(waveforms_unseen, max_traces, events=None, multilabel=multilabel)
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

    np.random.seed(42)  # Set a random seed for reproducibility

    headline_data = dat
    headline_labels = labels
    additional_labels = labels
    additional_data = labels
    from sklearn.model_selection import train_test_split
    from keras.models import Sequential
    from keras.layers import Dense, Activation
    from keras.layers import Dense, Dropout, Activation
# For a single-input model with 2 classes (binary classi    fication):
    if train_model is True:

        model = Sequential()
        model.add(Activation('relu'))
    #    model.add(Dense(6056, activation='relu', input_dim=nsamples))

    #    model.add(Dense(2056, activation='relu', input_dim=nsamples))

        model.add(Dense(1060, activation='relu', input_dim=nsamples))
    #    model.add(Dense(256, activation='relu', input_dim=nsamples))
        model.add(Dense(64, activation='relu', input_dim=nsamples))
        model.add(Dense(32, activation='relu', input_dim=nsamples))
    #    model.add(Dense(16, activation='relu', input_dim=nsamples))
    #    model.add(Dense(8, activation='relu', input_dim=nsamples))
    #    model.add(Dense(6, activation='relu', input_dim=nsamples))
    #    model.add(Dense(4, activation='relu', input_dim=nsamples))
    #    model.add(Dense(64, activation='relu', input_dim=nsamples))

    #    model.add(Dropout(0.5))
        model.add(Dense(nlabels, activation='sigmoid'))
        # adadelta
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
        # Train the model, iterating on the data in batches of 32 samples
        from keras.callbacks import ModelCheckpoint
        checkpointer = ModelCheckpoint(filepath="best_weights.hdf5",
                                       monitor = 'val_accuracy',
                                       verbose=1,
                                       save_best_only=True)

        if train_model is True:
            if detector_only is True:
                history = model.fit(train, y_train, epochs=5, batch_size=20,
                                    callbacks=[checkpointer])
            else:
                history = model.fit(train, y_train, epochs=20, batch_size=1,
                                    callbacks=[checkpointer])

            plot_model(model)
            layer_outputs = [layer.output for layer in model.layers[:]]
            #  Extracts the outputs of the top 12 layers
            activation_model = models.Model(inputs=model.input, outputs=layer_outputs)
            plot_acc_loss(history)
        else:
            if detector_only is True:
                model = keras.models.load_model('model_detector')
            else:
                model = keras.models.load_model('model_locator')
        if data_dir is not None or validation_data is not None:
            pred = model.predict(data_events_unseen)
        else:
            pred = model.predict(x_val)
        print(np.shape(pred))
        print("here", pred)

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
    print(y_val, pred)
    print(abs(y_val)-abs(pred))
    print(np.sum(abs(y_val)-abs(pred)))
    recover_real_value = False
    if multilabel is True:
        lons = []
        lats = []
        depths = []
        for i, ev in enumerate(events):
            lats.append(ev.lat)
            lons.append(ev.lon)
            depths.append(ev.depth)
        lats = np.asarray(lats)
        lats_max = np.max(lats)
        lons = np.asarray(lons)
        lons_max = np.max(lons)
        depths = np.asarray(depths)
        depths_max = np.max(depths)
        real_values = []
        for i, vals in enumerate(y_val):
            real_values.append([vals[0]*lats_max, vals[1]*lons_max, vals[2]*depths_max])
        real_values_pred = []
        diff_values = []
        for i, vals in enumerate(pred):
            real_values_pred.append([vals[0]*lats_max, vals[1]*lons_max, vals[2]*depths_max])
            diff_values.append([real_values[i][0]-real_values_pred[i][0], real_values[i][1]-real_values_pred[i][1], real_values[i][2]-real_values_pred[i][2]])

        print(real_values)
        print(real_values_pred)
        print(diff_values)

    plot_prescission(y_val, pred)
    if multilabel is True:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(real_values[:][0], real_values[:][1], real_values[:][2], c="k")
        ax.scatter(real_values_pred[:][0], real_values_pred[:][1], real_values_pred[:][2], c="r")
        plt.show()
    if detector_only is True:
        model.save('model_detector')
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
    print(history.history)
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
