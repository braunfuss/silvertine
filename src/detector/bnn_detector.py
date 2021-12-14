# Copyright 2018 The TensorFlow Probability Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
"""Trains a Bayesian neural network to classify MNIST digits.

The architecture is LeNet-5 [1].

#### References

[1]: Yann LeCun, Leon Bottou, Yoshua Bengio, and Patrick Haffner.
     Gradient-based learning applied to document recognition.
     _Proceedings of the IEEE_, 1998.
     http://yann.lecun.com/exdb/publis/pdf/lecun-01a.pdf
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import warnings

# Dependency imports
from absl import app
from absl import flags
import matplotlib
from matplotlib import pyplot as plt
#matplotlib.use('Agg')
from matplotlib import figure  # pylint: disable=g-import-not-at-top
from matplotlib.backends import backend_agg
import numpy as np
import tensorflow.compat.v2 as tf
import tensorflow_probability as tfp
from PIL import Image
from pyrocko import gf
import pandas as pd
import numpy as np
tfk = tf.keras
tf.keras.backend.set_floatx("float64")
import tensorflow_probability as tfp
tfd = tfp.distributions
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest# Define helper functions.
scaler = StandardScaler()
detector = IsolationForest(n_estimators=1000, behaviour="deprecated", contamination="auto", random_state=0)
neg_log_likelihood = lambda x, rv_x: -rv_x.log_prob(x)
tf.enable_v2_behavior()

import os

from pyrocko.gf import LocalEngine, Target, DCSource, ws
from pyrocko import trace
from pyrocko.marker import PhaseMarker

# The store we are going extract data from:
store_id = 'landau_100hz'

# First, download a Greens Functions store. If you already have one that you
# would like to use, you can skip this step and point the *store_superdirs* in
# the next step to that directory.


# We need a pyrocko.gf.Engine object which provides us with the traces
# extracted from the store. In this case we are going to use a local
# engine since we are going to query a local store.
engine = LocalEngine(store_superdirs=['/home/asteinbe/gf_stores'])



from silvertine import scenario
from pyrocko import model, cake, orthodrome
from silvertine.util.ref_mods import landau_layered_model
from silvertine.locate.locate1D import get_phases_list
mod = landau_layered_model()
scale = 2e-14
cake_phase = cake.PhaseDef("P")
phase_list = [cake_phase]
waveforms_events = []
waveforms_noise = []
stations = model.load_stations("stations.raw.txt")
nstations = len(stations)*3
noised = True
nevents = 1200
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

import _pickle as pickle

try:
    f = open("data_waveforms", 'rb')
    waveforms_events, nsamples = pickle.load(f)
    f.close()
except:
    f = open("data_waveforms", 'wb')

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

    pickle.dump([waveforms_events, nsamples], f)
    f.close()
# TODO(b/781nsamples93): Integration tests currently fail with seaborn imports.
warnings.simplefilter(action='ignore')



try:
  import seaborn as sns  # pylint: disable=g-import-not-at-top
  HAS_SEABORN = True
except ImportError:
  HAS_SEABORN = False

tfd = tfp.distributions

IMAGE_SHAPE = [nstations, nsamples, 1]
NUM_TRAIN_EXAMPLES = len(waveforms_events)*10
NUM_HELDOUT_EXAMPLES = len(waveforms_events)*10
NUM_CLASSES = 2


flags.DEFINE_float('learning_rate',
                   default=0.001,
                   help='Initial learning rate.')
flags.DEFINE_integer('num_epochs',
                     default=1,
                     help='Number of training steps to run.')
flags.DEFINE_integer('batch_size',
                     default=12,
                     help='Batch size.')
flags.DEFINE_string('data_dir',
                    default=os.path.join(os.getenv('TEST_TMPDIR', '/tmp'),
                                         'bayesian_neural_network/data'),
                    help='Directory where data is stored (if using real data).')
flags.DEFINE_string(
    'model_dir',
    default=os.path.join(os.getenv('TEST_TMPDIR', '.'),
                         'bayesian_neural_network/'),
    help="Directory to put the model's fit.")
flags.DEFINE_integer('viz_steps',
                     default=400,
                     help='Frequency at which save visualizations.')
flags.DEFINE_integer('num_monte_carlo',
                     default=5,
                     help='Network draws to compute predictive probabilities.')
flags.DEFINE_bool('fake_data',
                  default=False,
                  help='If true, uses fake data. Defaults to real data.')

FLAGS = flags.FLAGS


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
    #ax.scatter(input_vals[i, :].reshape(1,nsamples), np.arange(0, nsamples))
    ax.imshow(input_vals[i, :].reshape(nstations,nsamples))

    ax = fig.add_subplot(n, 3, 3*i + 2)
    for prob_sample in probs:
      sns.barplot(np.arange(NUM_CLASSES), prob_sample[i, :], alpha=0.1, ax=ax)
      ax.set_ylim([0, 1])
    ax.set_title('posterior samples')

    ax = fig.add_subplot(n, 3, 3*i + 3)
    sns.barplot(np.arange(NUM_CLASSES), tf.reduce_mean(probs[:, i, :], axis=0), ax=ax)
    ax.set_ylim([0, 1])
    ax.set_title('predictive probs')
  fig.suptitle(title)
  fig.tight_layout()

  canvas.print_figure(fname, format='png')
  print('saved {}'.format(fname))


def create_model():
  """Creates a Keras model using the LeNet-5 architecture.

  Returns:
      model: Compiled Keras model.
  """
  # KL divergence weighted by the number of training samples, using
  # lambda function to pass as input to the kernel_divergence_fn on
  # flipout layers.
  kl_divergence_function = (lambda q, p, _: tfd.kl_divergence(q, p) /  # pylint: disable=g-long-lambda
                            tf.cast(NUM_TRAIN_EXAMPLES, dtype=tf.float64))

  # Define a LeNet-5 model using three convolutional (with max pooling)
  # and two fully connected dense layers. We use the Flipout
  # Monte Carlo estimator for these layers, which enables lower variance
  # stochastic gradients than naive reparameterization.
  model = tf.keras.models.Sequential([
      tfp.layers.Convolution2DFlipout(
          6, kernel_size=5, padding='SAME',
          kernel_divergence_fn=kl_divergence_function,
          activation=tf.nn.relu),
      tf.keras.layers.MaxPooling2D(
          pool_size=[2, 2], strides=[2, 2],
          padding='SAME'),
      tfp.layers.Convolution2DFlipout(
          16, kernel_size=5, padding='SAME',
          kernel_divergence_fn=kl_divergence_function,
          activation=tf.nn.relu),
      tf.keras.layers.MaxPooling2D(
          pool_size=[2, 2], strides=[2, 2],
          padding='SAME'),
      tfp.layers.Convolution2DFlipout(
          12, kernel_size=5, padding='SAME',
          kernel_divergence_fn=kl_divergence_function,
          activation=tf.nn.relu),
      tf.keras.layers.Flatten(),
      tfp.layers.DenseFlipout(
          84, kernel_divergence_fn=kl_divergence_function,
          activation=tf.nn.relu),
      tfp.layers.DenseFlipout(
          NUM_CLASSES, kernel_divergence_fn=kl_divergence_function,
          activation=tf.nn.softmax)
  ])

  # Model compilation.
  optimizer = tf.keras.optimizers.Adam(lr=FLAGS.learning_rate)
  # We use the categorical_crossentropy loss since the MNIST dataset contains
  # ten labels. The Keras API will then automatically add the
  # Kullback-Leibler divergence (contained on the individual layers of
  # the model), to the cross entropy loss, effectively
  # calcuating the (negated) Evidence Lower Bound Loss (ELBO)
 # model.compile(optimizer, loss='categorical_crossentropy',
#                metrics=['accuracy'], experimental_run_tf_function=False)
  model.compile(optimizer, loss='categorical_crossentropy',
                metrics=['accuracy'], experimental_run_tf_function=False)
  return model


def rgb2gray(rgb):
    return np.dot(rgb[...,:3], [0.2989, 0.5870, 0.1140])


class PyrockoSequence(tf.keras.utils.Sequence):

    def __init__(self, data=None, batch_size=128, fake_data_size=None, syn=True):
        """Initializes the sequence.

        Args:
          data: Tuple of numpy `array` instances, the first representing images and
                the second labels.
          batch_size: Integer, number of elements in each training batch.
          fake_data_size: Optional integer number of fake datapoints to generate.
        """
        if syn is not False:
            images, labels = PyrockoSequence.__generate_waveform_data(
              num_images=NUM_TRAIN_EXAMPLES, num_classes=NUM_CLASSES)
        else:
            images, labels = PyrockoSequence.__generate_waveform_data(
              num_images=NUM_HELDOUT_EXAMPLES, num_classes=NUM_CLASSES, syn=False)
        self.images, self.labels = PyrockoSequence.__preprocessing(
            images, labels)
        self.batch_size = batch_size
    @staticmethod
    def __generate_waveform_data(num_images, num_classes, syn=True):
        """Generates fake data in the shape of the MNIST dataset for unittest.

        Args:
          num_images: Integer, the number of fake images to be generated.
          num_classes: Integer, the number of classes to be generate.
        Returns:
          images: Numpy `array` representing the fake image data. The
                  shape of the array will be (num_images, 28, 28).
          labels: Numpy `array` of integers, where each entry will be
                  assigned a unique integer.
        """
        images = []
        labels = []
        for i in range(0, num_images):
            traces = waveforms_events[i]
            if (i % 2) == 0:
                labels.append(1)
                tr_image = []
                for tr in traces:
                    #ff, nn = tr.spectrum()
                    for k in range(0, nstations):
                    #    img = Image.new("RGB", (nsamples, 1), "#FF0000")
                    #    for sam in range(0, nsamples):
                    #        img.paste((255,255,0),(0,0,1,sam))
                    #    img = rgb2gray(np.asarray(img))
                        nsamples = len(tr.ydata)
                        scale = np.mean(tr.ydata)
                        randdata = np.random.normal(size=nsamples)*scale
                        white_noise = trace.Trace(deltat=tr.deltat, tmin=tr.tmin,
                                                  ydata=randdata)
                    #    if syn is False:
                    #        tr.ydata = tr.ydata*0.
                        tr.add(white_noise)
                        img = Image.fromarray(np.asarray([tr.ydata, tr.ydata]), 'RGB')

                        img = rgb2gray(np.asarray(img))[0]

                        img.reshape((nsamples))

                        tr_image.append(img)
                images.append(np.asarray(tr_image))
            else:
                tr_image = []
                for tr in traces
                    ff, nn = tr.spectrum()
                    for k in range(0,nstations):
                    #    img = Image.new("RGB", (nsamples, 1), "#FF0000")
                        scale = np.mean(tr.ydata)
                        tr.ydata = tr.ydata*0.
                        randdata = np.random.normal(size=nsamples)*scale
                        white_noise = trace.Trace(deltat=tr.deltat, tmin=tr.tmin,
                                                  ydata=randdata)
                        tr.add(white_noise)
                        img = Image.fromarray(np.asarray([tr.ydata, tr.ydata]), 'RGB')
                    #    for sam in range(0, nsamples):
                    #        img.paste((155,155,0),(0,0,1,sam))
                        img = rgb2gray(np.asarray(img))[0]
                        img.reshape((nsamples))
                        tr_image.append(img)
                images.append(np.asarray(tr_image))
                labels.append(0)

        print("shape")
        print(np.shape(images), np.shape(labels))
        return np.asarray(images), np.asarray(labels)

    @staticmethod
    def __preprocessing(images, labels):
        """Preprocesses image and labels data.

        Args:
          images: Numpy `array` representing the image data.
          labels: Numpy `array` representing the labels data (range 0-9).

        Returns:
          images: Numpy `array` representing the image data, normalized
                  and expanded for convolutional network input.
          labels: Numpy `array` representing the labels data (range 0-9),
                  as one-hot (categorical) values.
        """
        images = 2 * (images / 255.) - 1.
        images = images[..., tf.newaxis]

        labels = tf.keras.utils.to_categorical(labels)
        print(labels)
        return images, labels

    def __len__(self):
        return int(tf.math.ceil(len(self.images) / self.batch_size))

    def __getitem__(self, idx):
        batch_x = self.images[idx * self.batch_size: (idx + 1) * self.batch_size]
        batch_y = self.labels[idx * self.batch_size:(idx + 1) * self.batch_size]
        return batch_x, batch_y


class MNISTSequence(tf.keras.utils.Sequence):
  """Produces a sequence of MNIST digits with labels."""

  def __init__(self, data=None, batch_size=128, fake_data_size=None, syn=True):
    """Initializes the sequence.

    Args:
      data: Tuple of numpy `array` instances, the first representing images and
            the second labels.
      batch_size: Integer, number of elements in each training batch.
      fake_data_size: Optional integer number of fake datapoints to generate.
    """
    if syn is not False:
        images, labels = MNISTSequence.__generate_real_data(
          num_images=NUM_TRAIN_EXAMPLES, num_classes=NUM_CLASSES)
    else:
        images, labels = MNISTSequence.__generate_real_data(
          num_images=NUM_HELDOUT_EXAMPLES, num_classes=NUM_CLASSES)
    self.images, self.labels = MNISTSequence.__preprocessing(
        images, labels)
    self.batch_size = batch_size

  @staticmethod
  def __generate_fake_data(num_images, num_classes):
    """Generates fake data in the shape of the MNIST dataset for unittest.

    Args:
      num_images: Integer, the number of fake images to be generated.
      num_classes: Integer, the number of classes to be generate.
    Returns:
      images: Numpy `array` representing the fake image data. The
              shape of the array will be (num_images, 28, 28).
      labels: Numpy `array` of integers, where each entry will be
              assigned a unique integer.
    """
    images = np.random.randint(low=0, high=256,
                               size=(num_images, IMAGE_SHAPE[0],
                                     IMAGE_SHAPE[1]))
    labels = np.random.randint(low=0, high=num_classes,
                               size=num_images)

    return images, labels

  @staticmethod
  def __generate_real_data(num_images, num_classes):
    """Generates fake data in the shape of the MNIST dataset for unittest.

    Args:
      num_images: Integer, the number of fake images to be generated.
      num_classes: Integer, the number of classes to be generate.
    Returns:
      images: Numpy `array` representing the fake image data. The
              shape of the array will be (num_images, 28, 28).
      labels: Numpy `array` of integers, where each entry will be
              assigned a unique integer.
    """
    images = []
    labels = []
    for i in range(0, num_images):
        if (i % 2) == 0:
            img = np.asarray(Image.new("RGB", (IMAGE_SHAPE[0], IMAGE_SHAPE[1]), "#FF0000"))
            img = rgb2gray(img)
            images.append(img)
            labels.append(1)
        else:
            img = Image.new("RGB", (IMAGE_SHAPE[0], IMAGE_SHAPE[1]), "#FF0000")
            img.paste((256,256,0),(0,0,1,3))
            img = rgb2gray(np.asarray(img))
            images.append(img)
            labels.append(0)
#    labels = np.random.randint(low=0, high=1,
#                               size=num_images)
#    images = np.random.randint(low=0, high=256,
#                               size=(num_images, IMAGE_SHAPE[0],
#                                     IMAGE_SHAPE[1]))
    #labels = np.random.randint(low=0, high=num_classes,
    #                           size=num_images)

    return np.asarray(images), np.asarray(labels)


  @staticmethod
  def __preprocessing(images, labels):
    """Preprocesses image and labels data.

    Args:
      images: Numpy `array` representing the image data.
      labels: Numpy `array` representing the labels data (range 0-9).

    Returns:
      images: Numpy `array` representing the image data, normalized
              and expanded for convolutional network input.
      labels: Numpy `array` representing the labels data (range 0-9),
              as one-hot (categorical) values.
    """
    images = 2 * (images / 255.) - 1.
    images = images[..., tf.newaxis]

    labels = tf.keras.utils.to_categorical(labels)

    return images, labels

  def __len__(self):
    return int(tf.math.ceil(len(self.images) / self.batch_size))

  def __getitem__(self, idx):
    batch_x = self.images[idx * self.batch_size: (idx + 1) * self.batch_size]
    batch_y = self.labels[idx * self.batch_size:(idx + 1) * self.batch_size]
    return batch_x, batch_y


def main(argv):
  del argv  # unused
  if tf.io.gfile.exists(FLAGS.model_dir):
    tf.compat.v1.logging.warning(
        'Warning: deleting old log directory at {}'.format(FLAGS.model_dir))
    tf.io.gfile.rmtree(FLAGS.model_dir)
  tf.io.gfile.makedirs(FLAGS.model_dir)

    #train_set, heldout_set = tf.keras.datasets.mnist.load_data()
  train_seq = PyrockoSequence(data=True, batch_size=FLAGS.batch_size, syn=True)
  heldout_seq = PyrockoSequence(data=True, batch_size=FLAGS.batch_size,
                                fake_data_size=NUM_HELDOUT_EXAMPLES, syn=False)



#    n_samples = nsamples
#
#    dat = []
#    labels = []
#    for tr in synthetic_traces_event:
#        ff, nn = tr.spectrum()
#        for k in range(0,1200):
#            data = tr.ydata
#            dat.append(data)
#            labels.append(1.)
#            dat.append(-1.*data)
#            labels.append(0.)
#    dat = np.asarray(dat)
#    labels = np.asarray(labels)
#

    # Define some hyperparameters.
#    n_epochs = 1
#    n_samples = nsamples
#    n_batches = 1
#    batch_size = np.floor(n_samples/n_batches)
#    buffer_size = n_samples# Define training and test data sizes.
#    n_train = int(0.7*nsamples)# Define dataset instance.

#    data = tf.data.Dataset.from_tensor_slices((dat, labels))
#    data = data.shuffle(n_samples, reshuffle_each_iteration=True)# Define train and test data instances.
#    train_seq  = data.take(n_train).batch(batch_size).repeat(n_epochs)
#    heldout_seq = data.skip(n_train).batch(1).repeat(n_epochs)
  ##from keras.layers import Input, Embedding, LSTM, Dense
  #from keras.models import Model
  #main_input = Input(shape=(nsamples,), dtype='float64', name='main_input')
  #auxiliary_input = Input(shape=(5,), name='aux_input')
  #main_output = Dense(1, activation='sigmoid', name='main_output')(x)
  model = create_model()
  # TODO(b/149259388): understand why Keras does not automatically build the
  # model correctly.
  model.build(input_shape=[None, 12, nsamples, 1])
 # model.build(input_shape=[None, nsamples, 1, 1])
  #prior = tfd.Independent(tfd.Normal(loc=tf.zeros(len(labels), dtype=tf.float64), scale=1.0), reinterpreted_batch_ndims=1)# Define model instance.

 # model = tfk.Sequential([
#tfk.layers.InputLayer(input_shape=(nsamples,), name="input"),
#tfk.layers.Dense(10, activation="relu", name="dense_1"),
#tfk.layers.Dense(tfp.layers.MultivariateNormalTriL.params_size(
#len(labels)), activation=None, name="distribution_weights"),
#tfp.layers.MultivariateNormalTriL(len(labels), activity_regularizer=tfp.layers.KLDivergenceRegularizer(prior, weight=1/n_batches), name="output")
#], name="model")
#  model.compile(optimizer="adam", loss=neg_log_likelihood)# Run training session.

  print(' ... Training convolutional neural network')
  for epoch in range(FLAGS.num_epochs):
    epoch_accuracy, epoch_loss = [], []
    for step, (batch_x, batch_y) in enumerate(train_seq):
#      print(np.shape(batch_x), np.shape(batch_y))
#      batch_x = batch_x.reshape((75,nsamples))
      batch_loss, batch_accuracy = model.train_on_batch(
          batch_x, batch_y)
      epoch_accuracy.append(batch_accuracy)
      epoch_loss.append(batch_loss)

      if step % 100 == 0:
        print('Epoch: {}, Batch index: {}, '
              'Loss: {:.3f}, Accuracy: {:.3f}'.format(
                  epoch, step,
                  tf.reduce_mean(epoch_loss),
                  tf.reduce_mean(epoch_accuracy)))

      if (step+1) % FLAGS.viz_steps == 0:
        # Compute log prob of heldout set by averaging draws from the model:
        # p(heldout | train) = int_model p(heldout|model) p(model|train)
        #                   ~= 1/n * sum_{i=1}^n p(heldout | model_i)
        # where model_i is a draw from the posterior p(model|train).
        print(' ... Running monte carlo inference')
        probs = tf.stack([model.predict(heldout_seq, verbose=1)
                          for _ in range(FLAGS.num_monte_carlo)], axis=0)
        mean_probs = tf.reduce_mean(probs, axis=0)
        heldout_log_prob = tf.reduce_mean(tf.math.log(mean_probs))
        print(' ... Held-out nats: {:.3f}'.format(heldout_log_prob))

        if HAS_SEABORN:
          names = [layer.name for layer in model.layers
                   if 'flipout' in layer.name]
          qm_vals = [layer.kernel_posterior.mean()
                     for layer in model.layers
                     if 'flipout' in layer.name]
          qs_vals = [layer.kernel_posterior.stddev()
                     for layer in model.layers
                     if 'flipout' in layer.name]
          plot_weight_posteriors(names, qm_vals, qs_vals,
                                 fname=os.path.join(
                                     FLAGS.model_dir,
                                     'epoch{}_step{:05d}_weights.png'.format(
                                         epoch, step)))
          plot_heldout_prediction(heldout_seq.images, probs,
                                  fname=os.path.join(
                                      FLAGS.model_dir,
                                      'epoch{}_step{}_pred.png'.format(
                                          epoch, step)),
                                  title='mean heldout logprob {:.2f}'
                                  .format(heldout_log_prob))

  model.save('my_model')
if __name__ == '__main__':
  app.run(main)
