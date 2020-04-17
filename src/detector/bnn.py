import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from pyrocko.gf import LocalEngine, Target, DCSource, ws
from pyrocko import trace
from pyrocko.marker import PhaseMarker
from silvertine.util.waveform import plot_waveforms_raw
from pyrocko.gf import LocalEngine, Target, DCSource, ws
from pyrocko import trace
from pyrocko.marker import PhaseMarker
from silvertine import scenario
from pyrocko import model, cake, orthodrome
from silvertine.util.ref_mods import landau_layered_model
from silvertine.locate.locate1D import get_phases_list
from keras.layers import Input, Embedding, LSTM, Dense
from keras.models import Model
import keras
import numpy as np
#tf.disable_eager_execution()
from keras.layers import Conv2D, MaxPooling2D, Input
from PIL import Image
import os

from pyrocko.gf import LocalEngine, Target, DCSource, ws
from pyrocko import trace
from pyrocko.marker import PhaseMarker
def rgb2gray(rgb):
    return np.dot(rgb[...,:3], [0.2989, 0.5870, 0.1140])

from keras.layers import Input, Embedding, LSTM, Dense
from keras.models import Model
import keras
import numpy as np
#tf.disable_eager_execution()
from keras.layers import Conv2D, MaxPooling2D, Input
from PIL import Image
import os

from pyrocko.gf import LocalEngine, Target, DCSource, ws
from pyrocko import trace
from pyrocko.marker import PhaseMarker


def bnn_detector_data(waveforms, max_traces, events=True, multilabel=False):
    data_traces = []
    for traces in waveforms:
        traces_coll = None
        for tr in traces:
            tr.lowpass(4, 13.)
            tr.highpass(4, 0.03)
            #tr.ydata = tr.ydata/max_traces
            #tr.ydata = np.power(np.abs(tr.ydata), 1./2.)
            tr.ydata = tr.ydata/np.max(tr.ydata)
            nsamples = len(tr.ydata)
            data = tr.ydata
            if traces_coll is None:
                traces_coll = tr.ydata
            else:
                traces_coll = np.concatenate((traces_coll, data),  axis=0)
        nsamples = len(traces_coll)
        data_traces.append(traces_coll)

    if events is not None:
        if multilabel is True:
            labels = []
            for i, ev in enumerate(events):
                print(ev)
                labels.append([ev.moment_tensor.strike1/360., ev.moment_tensor.dip1/90.])
            labels = np.asarray(labels)
        else:
            labels = np.ones(len(data_traces), dtype=np.int32)*1
    else:
        if multilabel is True:
            labels = np.ones((len(data_traces),2), dtype=np.int32)*0
        else:
            labels = np.ones(len(data_traces), dtype=np.int32)*0
    data_traces = np.asarray(data_traces)

    return data_traces, labels, len(data_traces)/nsamples, nsamples


def generate_test_data(store_id, nevents=50, noised=False):
    from silvertine import scenario
    from pyrocko import model, cake, orthodrome
    from silvertine.util.ref_mods import landau_layered_model
    from silvertine.locate.locate1D import get_phases_list
    from pyrocko import moment_tensor
    mod = landau_layered_model()
    engine = LocalEngine(store_superdirs=['/home/asteinbe/gf_stores'])
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
            event = scenario.gen_random_tectonic_event(i, magmin=-1., magmax=3.)
            source_dc = DCSource(
                lat=scenario.randlat(49., 49.2),
                lon=scenario.rand(8.1, 8.2),
                depth=scenario.rand(100., 3000.),
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
                randdata = np.random.normal(size=nsamples)*scale
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



def bnn_detector(waveforms_events=None, waveforms_noise=None):
    import _pickle as pickle

    try:
        f = open("data_waveforms_bnn_gt", 'rb')
        waveforms_events, waveforms_noise, nsamples, nstations, events = pickle.load(f)
        f.close()
    except:

        waveforms_events, waveforms_noise, nsamples, nstations, events = generate_test_data("landau6", nevents=20)
        f = open("data_waveforms_bnn_gt", 'wb')
        pickle.dump([waveforms_events, waveforms_noise, nsamples, nstations, events], f)
        f.close()
    ncomponents = 3
    nstations = nstations*ncomponents
    max_traces = 0.
    for traces in waveforms_events:
        for tr in traces:
            if np.max(tr.ydata) > max_traces:
                max_traces = np.max(tr.ydata)
    data_events, labels_events, nstations, nsamples = bnn_detector_data(waveforms_events, max_traces, events=events, multilabel=True)
    data_noise, labels_noise, nstations, nsamples = bnn_detector_data(waveforms_noise, max_traces, events=None, multilabel=True)

    x_data = np.concatenate((data_events, data_noise), axis=0)
    nlabels = 1
    dat = x_data
    y_data = np.concatenate((labels_events, labels_noise), axis= 0)
    labels = y_data



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

# For a single-input model with 2 classes (binary classi    fication):
    model = Sequential()
    model.add(Dense(32, activation='relu', input_dim=nsamples))
    model.add(Dense(2, activation='sigmoid'))
    model.compile(optimizer='rmsprop',
                  loss='binary_crossentropy',
                  metrics=['accuracy'])

    # Generate dummy data
    data = dat


    # Train the model, iterating on the data in batches of 32 samples
    model.fit(data, labels, epochs=10, batch_size=32)
    pred = model.predict(data)

    bayesian = False
    if bayesian is True:
        train, x_val, y_train, y_val = train_test_split(headline_data, headline_labels,
                                                          test_size=0.2,
                                                          stratify=y_data,
                                                          random_state=10)

        num_epochs = 5
        batchsize = 32
        num_monte_carlo = 5
        for epoch in range(num_epochs):
            epoch_accuracy, epoch_loss = [], []
            for i in range(len(y_train) // batchsize):
                batch_x = train[i * batchsize: (i + 1) * batchsize]
                batch_y = y_train[i * batchsize: (i + 1) * batchsize]

                c = model.fit([batch_x, batch_y], [batch_y, batch_y])
                print(c1)

                probs = tf.stack([model.predict({'main_input': x_val, 'aux_input': y_val})
                                  for _ in range(num_monte_carlo)], axis=0)
                mean_probs = tf.reduce_mean(probs, axis=0)
                heldout_log_prob = tf.reduce_mean(tf.math.log(mean_probs))

    print(labels, pred)


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

#model.fit_generator(generate_arrays_from_file('/my_file.txt'),
#        samples_per_epoch=10000, nb_epoch=10)
