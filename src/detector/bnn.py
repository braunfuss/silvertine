import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow.compat.v1 as tf
import tensorflow_probability as tfp

from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, MaxPooling2D, Flatten, Dropout
tf.disable_eager_execution()
from PIL import Image

from keras.datasets import mnist
import os

from pyrocko.gf import LocalEngine, Target, DCSource, ws
from pyrocko import trace
from pyrocko.marker import PhaseMarker
from silvertine.util.waveform import plot_waveforms_raw
tfk = tf.keras
#tf.keras.backend.set_floatx("float64")
import tensorflow_probability as tfp
tfd = tfp.distributions
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest# Define helper functions.
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


def bnn_detector_data(waveforms, max_traces, events=True, multilabel=False):
    data_traces = []
    for traces in waveforms:
        data_event = []
        for tr in traces:
            tr.lowpass(4, 13.)
            tr.highpass(4, 0.03)
            #ff, nn = tr.spectrum()
        #    img = Image.fromarray(np.asarray([tr.ydata, tr.ydata]), 'RGB')
        #    img = rgb2gray(np.asarray(img))[0]
        #    img.reshape((len(tr.ydata)))
            #ff, nn = tr.spectrum()
        #    data_event.append(img)
        #    tr.ydata = tr.ydata/max_traces
            tr.ydata = np.power(np.abs(tr.ydata), 1./2.)

            data_event.append(tr.ydata)

        #fig = plot_waveforms_raw(traces, ".")
        #data_event = fig2data ( fig )
    #    print(data_event)
        #print(np.shape(data_event))
        #img = Image.fromarray(np.asarray(data_event), 'RGB')
        #img = rgb2gray(np.asarray(data_event))
        #print(np.shape(img))
    #    plt.figure()
    #    plt.imshow(data_event)
    #    plt.show()
        #img.reshape((len(data_event)))
        data_traces.append(np.asarray(data_event))

    data_traces = np.expand_dims(data_traces, axis=3)
    if events is True:
        if multilabel is True:
            for i, tr in enumerate(traces):
                labels.append(i+1)
        else:
            labels = np.ones(len(data_traces), dtype=np.int32)*1
    else:
        labels = np.ones(len(data_traces), dtype=np.int32)*0

    return data_traces, labels, np.shape(data_event)[0], np.shape(data_event)[1]


def generate_test_data(store_id, nevents=50, noised=False):
    from silvertine import scenario
    from pyrocko import model, cake, orthodrome
    from silvertine.util.ref_mods import landau_layered_model
    from silvertine.locate.locate1D import get_phases_list
    mod = landau_layered_model()
    engine = LocalEngine(store_superdirs=['/home/asteinbe/gf_stores'])
    scale = 2e-14
    cake_phase = cake.PhaseDef("P")
    phase_list = [cake_phase]
    waveforms_events = []
    waveforms_noise = []
    stations = model.load_stations("scenarios/stations.raw.txt")
    targets = []
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
            source_dc = DCSource(
                lat=scenario.randlat(49., 49.2),
                lon=scenario.rand(8.1, 8.2),
                depth=scenario.rand(100., 3000.),
                strike=scenario.rand(0., 360.),
                dip=scenario.rand(0., 90.),
                rake=scenario.rand(-180., 180.),
                magnitude=scenario.rand(0.2, 3))
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


                nsamples = len(tr.ydata)
                randdata = np.random.normal(size=nsamples)*scale
                white_noise = trace.Trace(deltat=tr.deltat, tmin=tr.tmin,
                                          ydata=randdata)
                if noised is True:
                    tr.add(white_noise)

            #    tr.lowpass(4, 0.02)
        #        tr.highpass(4, 3.)
            waveforms_events.append(traces)
        except:
            pass

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
            #    tr.lowpass(4, 0.02)
        #        tr.highpass(4, 3.)
                nsamples = len(tr.ydata)
                randdata = np.random.normal(size=nsamples)*scale
                white_noise = trace.Trace(deltat=tr.deltat, tmin=tr.tmin,
                                          ydata=randdata)
                tr.add(white_noise)
            waveforms_noise.append(traces)

        except:
            pass

    return waveforms_events, waveforms_noise, nsamples, len(stations)


def rgb2gray(rgb):
    return np.dot(rgb[...,:3], [0.2989, 0.5870, 0.1140])



def bnn_detector(waveforms_events=None, waveforms_noise=None):
    import _pickle as pickle

    try:
        f = open("data_waveforms_bnn", 'rb')
        waveforms_events, waveforms_noise, nsamples, nstations = pickle.load(f)
        f.close()
    except:

        waveforms_events, waveforms_noise, nsamples, nstations = generate_test_data("landau6", nevents=2000)
        f = open("data_waveforms_bnn", 'wb')
        pickle.dump([waveforms_events, waveforms_noise, nsamples, nstations], f)
        f.close()

    nstations = nstations*3
    max_traces = 0.
    for traces in waveforms_events:
        for tr in traces:
            if np.max(tr.ydata) > max_traces:
                max_traces = np.max(tr.ydata)
    data_events, labels_events, nstations, nsamples = bnn_detector_data(waveforms_events, max_traces,  events=True)
    data_noise, labels_noise, nstations, nsamples = bnn_detector_data(waveforms_noise, max_traces, events=False)
    x_data = np.concatenate((data_events, data_noise), axis=0)
    y_data = np.concatenate((labels_events, labels_noise), axis= 0)




    print('shape of x_data: ', x_data.shape)
    print('shape of y_data: ', y_data.shape)

    x_train, x_val, y_train, y_val = train_test_split(x_data, y_data,
                                                      test_size=0.2,
                                                      stratify=y_data,
                                                      random_state=10)

    np.random.seed(0)  # Set a random seed for reproducibility

    # Headline input: meant to receive sequences of 100 integers, between 1 and 10000.
    # Note that we can name any layer by passing it a "name" argument.
    main_input = Input(shape=(300,), dtype='float64', name='main_input')

    # This embedding layer will encode the input sequence
    # into a sequence of dense 512-dimensional vectors.
    x = Embedding(output_dim=1, input_dim=720, input_length=300)(main_input)

    # A LSTM will transform the vector sequence into a single vector,
    # containing information about the entire sequence
    lstm_out = LSTM(32)(x)
    auxiliary_output = Dense(1, activation='sigmoid', name='aux_output')(lstm_out)

    auxiliary_input = Input(shape=(1,), name='aux_input')
    x = keras.layers.concatenate([lstm_out, auxiliary_input])

    # We stack a deep densely-connected network on top
    x = Dense(64, activation='relu')(x)
    x = Dense(64, activation='relu')(x)
    x = Dense(64, activation='relu')(x)
    #x = Conv2D(64, (5, 5), padding='same', activation='relu')(x)

    #x = MaxPooling2D((3, 3), strides=(1, 1), padding='same')(x)

    # And finally we add the main logistic regression layer
    main_output = Dense(1, activation='sigmoid', name='main_output')(x)

    model = Model(inputs=[main_input, auxiliary_input], outputs=[main_output, auxiliary_output])

    model.compile(optimizer='rmsprop', loss='binary_crossentropy',
                  loss_weights=[1., 0.2])

    headline_data = dat
    print(np.shape(headline_data))
    print(np.shape(labels))
    headline_labels = labels
    additional_labels = np.ones(len(labels))
    additional_data = labels

    model.fit([headline_data, additional_data], [headline_labels, additional_labels],
              epochs=5, batch_size=32)

    print(headline_data, additional_data, headline_labels, additional_labels)

    pred = model.predict({'main_input': headline_data, 'aux_input': additional_data})
