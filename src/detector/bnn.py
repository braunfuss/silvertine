import matplotlib.pyplot as plt
from pyrocko.gf import LocalEngine, Target, DCSource, ws, MTSource
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
from pyrocko import model, cake, orthodrome, gf
from pyrocko.gf import meta
from pyrocko.gf.seismosizer import Source
from silvertine.util.ref_mods import landau_layered_model
from pyrocko import moment_tensor, util
from beat.utility import get_rotation_matrix
from keras.layers import Conv1D, MaxPooling1D, Input, Conv2D, LSTM, MaxPooling2D, Flatten
from keras import models
from PIL import Image
from pyrocko.guts import Float
from pathlib import Path
from silvertine.util import waveform
import scipy
from mtpar import cmt2tt, cmt2tt15, tt2cmt, tt152cmt
from mtpar.basis import change_basis
from mtpar.util import PI, DEG
from pyrocko import moment_tensor as mtm
import ray
import math
import copy
import psutil
import pyrocko
import _pickle as pickle
from sklearn.utils import shuffle
from pyrocko import trace as trd
from matplotlib import image
from keras.layers import Layer
from keras import backend as K
from keras import activations, initializers
from keras.layers import Layer
import tensorflow_probability as tfp
tfd = tfp.distributions
num_cpus = psutil.cpu_count(logical=False)
negloglik = lambda y, rv_y: -rv_y.log_prob(y)


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

import tensorflow as tf
init = tf.global_variables_initializer()
from matplotlib import pyplot as plt
from matplotlib.patches import Rectangle, FancyArrow
from matplotlib.collections import PatchCollection
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.ticker as tick


def neg_log_likelihood_1(y_obs, y_pred, sigma=0.1):
    dist = tfp.distributions.Normal(loc=y_pred, scale=sigma)
    return K.sum(-dist.log_prob(y_obs))


def neg_log_likelihood(y_true, y_pred):
    return -y_pred.log_prob(y_true)

def get_neg_log_likelihood_fn(bayesian=False):
    """
    Get the negative log-likelihood function
    # Arguments
        bayesian(bool): Bayesian neural network (True) or point-estimate neural network (False)

    # Returns
        a negative log-likelihood function
    """
    if bayesian:
        def neg_log_likelihood_bayesian(y_true, y_pred):
            labels_distribution = tfp.distributions.Categorical(logits=y_pred)
            log_likelihood = labels_distribution.log_prob(tf.argmax(input=y_true, axis=1))
            loss = -tf.reduce_mean(input_tensor=log_likelihood)
            return loss
        return neg_log_likelihood_bayesian
    else:
        def neg_log_likelihood(y_true, y_pred):
            y_pred_softmax = keras.layers.Activation('softmax')(y_pred)  # logits to softmax
            loss = keras.losses.log_prob(y_true, y_pred_softmax)
            return loss
        return neg_log_likelihood


# Specify the surrogate posterior over `keras.layers.Dense` `kernel` and `bias`.
def posterior_mean_field(kernel_size, bias_size=0, dtype=None):
  n = kernel_size + bias_size
  c = np.log(np.expm1(1.))
  return tf.keras.Sequential([
      tfp.layers.VariableLayer(2 * n, dtype=dtype),
      tfp.layers.DistributionLambda(lambda t: tfd.Independent(
          tfd.Normal(loc=t[..., :n],
                     scale=1e-5 + tf.nn.softplus(c + t[..., n:])),
          reinterpreted_batch_ndims=1)),
  ])


# Specify the prior over `keras.layers.Dense` `kernel` and `bias`.
def prior_trainable(kernel_size, bias_size=0, dtype=None):
  n = kernel_size + bias_size
  return tf.keras.Sequential([
      tfp.layers.VariableLayer(n, dtype=dtype),
      tfp.layers.DistributionLambda(lambda t: tfd.Independent(
          tfd.Normal(loc=t, scale=1),
          reinterpreted_batch_ndims=1)),
  ])


import numpy as num
pi = num.pi
pi4 = pi / 4.
km = 1000.
d2r = pi / 180.
r2d = 180. / pi

sqrt3 = num.sqrt(3.)
sqrt2 = num.sqrt(2.)
sqrt6 = num.sqrt(6.)



class MTQTSource(gf.SourceWithMagnitude):
    """
    A moment tensor point source.
    Notes
    -----
    Following Q-T parameterization after Tape & Tape 2015
    """

    discretized_source_class = meta.DiscretizedMTSource

    w = Float.T(
        default=0.,
        help='Lune latitude delta transformed to grid. '
             'Defined: -3/8pi <= w <=3/8pi. '
             'If fixed to zero the MT is deviatoric.')

    v = Float.T(
        default=0.,
        help='Lune co-longitude transformed to grid. '
             'Definded: -1/3 <= v <= 1/3. '
             'If fixed to zero together with w the MT is pure DC.')

    kappa = Float.T(
        default=0.,
        help='Strike angle equivalent of moment tensor plane.'
             'Defined: 0 <= kappa <= 2pi')

    sigma = Float.T(
        default=0.,
        help='Rake angle equivalent of moment tensor slip angle.'
             'Defined: -pi/2 <= sigma <= pi/2')

    h = Float.T(
        default=0.,
        help='Dip angle equivalent of moment tensor plane.'
             'Defined: 0 <= h <= 1')

    def __init__(self, **kwargs):
        n = 1000
        self._beta_mapping = num.linspace(0, pi, n)
        self._u_mapping = \
            (3. / 4. * self._beta_mapping) - \
            (1. / 2. * num.sin(2. * self._beta_mapping)) + \
            (1. / 16. * num.sin(4. * self._beta_mapping))

        self.lambda_factor_matrix = num.array(
            [[sqrt3, -1., sqrt2],
             [0., 2., sqrt2],
             [-sqrt3, -1., sqrt2]], dtype='float64')

        self.R = get_rotation_matrix()
        self.roty_pi4 = self.R['y'](-pi4)
        self.rotx_pi = self.R['x'](pi)

        self._lune_lambda_matrix = num.zeros((3, 3), dtype='float64')

        Source.__init__(self, **kwargs)

    @property
    def u(self):
        """
        Lunar co-latitude(beta), dependend on w
        """
        return (3. / 8.) * num.pi - self.w

    @property
    def gamma(self):
        """
        Lunar longitude, dependend on v
        """
        return v_to_gamma(self.v)

    @property
    def beta(self):
        """
        Lunar co-latitude, dependend on u
        """
        return w_to_beta(
            self.w, u_mapping=self._u_mapping, beta_mapping=self._beta_mapping)

    def delta(self):
        """
        From Tape & Tape 2012, delta measures departure of MT being DC
        Delta = Gamma = 0 yields pure DC
        """
        return (pi / 2.) - self.beta

    @property
    def rho(self):
        return mtm.magnitude_to_moment(self.magnitude) * sqrt2

    @property
    def theta(self):
        return num.arccos(self.h)

    @property
    def rot_theta(self):
        return self.R['x'](self.theta)

    @property
    def rot_kappa(self):
        return self.R['z'](-self.kappa)

    @property
    def rot_sigma(self):
        return self.R['z'](self.sigma)

    @property
    def lune_lambda(self):
        sin_beta = num.sin(self.beta)
        cos_beta = num.cos(self.beta)
        sin_gamma = num.sin(self.gamma)
        cos_gamma = num.cos(self.gamma)
        vec = num.array([sin_beta * cos_gamma, sin_beta * sin_gamma, cos_beta])
        return 1. / sqrt6 * self.lambda_factor_matrix.dot(vec) * self.rho

    @property
    def lune_lambda_matrix(self):
        num.fill_diagonal(self._lune_lambda_matrix, self.lune_lambda)
        return self._lune_lambda_matrix

    @property
    def rot_V(self):
        return self.rot_kappa.dot(self.rot_theta).dot(self.rot_sigma)

    @property
    def rot_U(self):
        return self.rot_V.dot(self.roty_pi4)

    @property
    def m9_nwu(self):
        """
        MT orientation is in NWU
        """
        return self.rot_U.dot(
            self.lune_lambda_matrix).dot(num.linalg.inv(self.rot_U))

    @property
    def m9(self):
        """
        Pyrocko MT in NED
        """
        return self.rotx_pi.dot(self.m9_nwu).dot(self.rotx_pi.T)

    @property
    def m6(self):
        return mtm.to6(self.m9)

    @property
    def m6_astuple(self):
        return tuple(self.m6.ravel().tolist())

    def base_key(self):
        return Source.base_key(self) + self.m6_astuple

    def discretize_basesource(self, store, target=None):
        times, amplitudes = self.effective_stf_pre().discretize_t(
            store.config.deltat, self.time)
        return meta.DiscretizedMTSource(
            m6s=self.m6[num.newaxis, :] * amplitudes[:, num.newaxis],
            **self._dparams_base_repeated(times))

    def pyrocko_moment_tensor(self):
        return mtm.MomentTensor(m=mtm.symmat6(*self.m6_astuple) * self.moment)

    def pyrocko_event(self, **kwargs):
        mt = self.pyrocko_moment_tensor()
        return Source.pyrocko_event(
            self,
            moment_tensor=self.pyrocko_moment_tensor(),
            magnitude=float(mt.moment_magnitude()),
            **kwargs)

    @classmethod
    def from_pyrocko_event(cls, ev, **kwargs):
        d = {}
        mt = ev.moment_tensor
        if mt:
            logger.warning(
                'From event will ignore MT components initially. '
                'Needs mapping from NED to QT space!')
            # d.update(m6=list(map(float, mt.m6())))

        d.update(kwargs)
        return super(MTQTSource, cls).from_pyrocko_event(ev, **d)

    def __getstate__(self):
        state = self.__dict__.copy()
        state['R'] = None
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        self.R = get_rotation_matrix()


class DenseVariational(Layer):
    def __init__(self,
                 units=72*2,
                 kl_weight=1/72,
                 activation=None,
                 prior_sigma_1=1.5,
                 prior_sigma_2=0.1,
                 prior_pi=0.5, **kwargs):
        self.units = units
        self.kl_weight = kl_weight
        self.activation = activations.get(activation)
        self.prior_sigma_1 = prior_sigma_1
        self.prior_sigma_2 = prior_sigma_2
        self.prior_pi_1 = prior_pi
        self.prior_pi_2 = 1.0 - prior_pi
        self.init_sigma = np.sqrt(self.prior_pi_1 * self.prior_sigma_1 ** 2 +
                                  self.prior_pi_2 * self.prior_sigma_2 ** 2)

        super().__init__(**kwargs)

    def compute_output_shape(self, input_shape):
        return input_shape[0], self.units

    def build(self, input_shape):
        self.kernel_mu = self.add_weight(name='kernel_mu',
                                         shape=(input_shape[1], self.units),
                                         initializer=initializers.normal(stddev=self.init_sigma),
                                         trainable=True)
        self.bias_mu = self.add_weight(name='bias_mu',
                                       shape=(self.units,),
                                       initializer=initializers.normal(stddev=self.init_sigma),
                                       trainable=True)
        self.kernel_rho = self.add_weight(name='kernel_rho',
                                          shape=(input_shape[1], self.units),
                                          initializer=initializers.constant(0.0),
                                          trainable=True)
        self.bias_rho = self.add_weight(name='bias_rho',
                                        shape=(self.units,),
                                        initializer=initializers.constant(0.0),
                                        trainable=True)
        super().build(input_shape)

    def call(self, inputs, **kwargs):
        kernel_sigma = tf.math.softplus(self.kernel_rho)
        kernel = self.kernel_mu + kernel_sigma * tf.random.normal(self.kernel_mu.shape)

        bias_sigma = tf.math.softplus(self.bias_rho)
        bias = self.bias_mu + bias_sigma * tf.random.normal(self.bias_mu.shape)

        self.add_loss(self.kl_loss(kernel, self.kernel_mu, kernel_sigma) +
                      self.kl_loss(bias, self.bias_mu, bias_sigma))

        return self.activation(K.dot(inputs, kernel) + bias)

    def kl_loss(self, w, mu, sigma):
        variational_dist = tfp.distributions.Normal(mu, sigma)
        return self.kl_weight * K.sum(variational_dist.log_prob(w) - self.log_prior_prob(w))

    def log_prior_prob(self, w):
        comp_1_dist = tfp.distributions.Normal(0.0, self.prior_sigma_1)
        comp_2_dist = tfp.distributions.Normal(0.0, self.prior_sigma_2)
        return K.log(self.prior_pi_1 * comp_1_dist.prob(w) +
                     self.prior_pi_2 * comp_2_dist.prob(w))


def vonmises_fisher(lats, lons, lats0, lons0, sigma=1.):
    """
    Von-Mises Fisher distribution function.
    Parameters
    ----------
    lats : float or array_like
        Spherical-polar latitude [deg][-pi/2 pi/2] to evaluate function at.
    lons : float or array_like
        Spherical-polar longitude [deg][-pi pi] to evaluate function at
    lats0 : float or array_like
        latitude [deg] at the center of the distribution (estimated values)
    lons0 : float or array_like
        longitude [deg] at the center of the distribution (estimated values)
    sigma : float
        Width of the distribution.
    Returns
    -------
    float or array_like
        log-probability of the VonMises-Fisher distribution.
    Notes
    -----
    Wikipedia:
        https://en.wikipedia.org/wiki/Von_Mises-Fisher_distribution
        modified from: https://github.com/williamjameshandley/spherical_kde
    """

    def logsinh(x):
        """ Compute log(sinh(x)), stably for large x.<
        Parameters
        ----------
        x : float or numpy.array
            argument to evaluate at, must be positive
        Returns
        -------
        float or numpy.array
            log(sinh(x))
        """
        if num.any(x < 0):
            raise ValueError("logsinh only valid for positive arguments")
        return x + num.log(1. - num.exp(-2. * x)) - num.log(2.)

    # transform to [0-pi, 0-2pi]
    lats_t = 90. + lats
    lons_t = 180. + lons
    lats0_t = 90. + lats0
    lons0_t = 180. + lons0

    x = cartesian_from_polar(
        phi=num.deg2rad(lons_t), theta=num.deg2rad(lats_t))
    x0 = cartesian_from_polar(
        phi=num.deg2rad(lons0_t), theta=num.deg2rad(lats0_t))

    norm = -num.log(4. * num.pi * sigma ** 2) - logsinh(1. / sigma ** 2)
    return norm + num.tensordot(x, x0, axes=[[0], [0]]) / sigma ** 2


def vonmises_std(lons, lats):
    """
    Von-Mises sample standard deviation.
    Parameters
    ----------
    phi, theta : array-like
        Spherical-polar coordinate samples to compute mean from.
    Returns
    -------
        solution for
        ..math:: 1/tanh(x) - 1/x = R,
        where
        ..math:: R = || \sum_i^N x_i || / N
    Notes
    -----
    Wikipedia:
        https://en.wikipedia.org/wiki/Von_Mises-Fisher_distribution#Estimation_of_parameters
        but re-parameterised for sigma rather than kappa.
    modidied from: https://github.com/williamjameshandley/spherical_kde
    """
    from scipy.optimize import brentq

    x = cartesian_from_polar(phi=num.deg2rad(lons), theta=num.deg2rad(lats))
    S = num.sum(x, axis=-1)

    R = S.dot(S) ** 0.5 / x.shape[-1]

    def f(s):
        return 1. / num.tanh(s) - 1. / s - R


    kappa = brentq(f, 1e-8, 1e8)
    sigma = kappa ** -0.5
    return sigma



def cartesian_from_polar(phi, theta):
    """
    Embedded 3D unit vector from spherical polar coordinates.
    Parameters
    ----------
    phi, theta : float or numpy.array
        azimuthal and polar angle in radians.
        (phi-longitude, theta-latitude)
    Returns
    -------
    nhat : numpy.array
        unit vector(s) in direction (phi, theta).
    """
    x = num.sin(theta) * num.cos(phi)
    y = num.sin(theta) * num.sin(phi)
    z = num.cos(theta)
    return num.array([x, y, z])



def cartesian_from_polar(phi, theta):
    """
    Embedded 3D unit vector from spherical polar coordinates.
    Parameters
    ----------
    phi, theta : float or numpy.array
        azimuthal and polar angle in radians.
        (phi-longitude, theta-latitude)
    Returns
    -------
    nhat : numpy.array
        unit vector(s) in direction (phi, theta).
    """
    x = num.sin(theta) * num.cos(phi)
    y = num.sin(theta) * num.sin(phi)
    z = num.cos(theta)
    return num.array([x, y, z])

def kde2plot(x, y, grid=200, ax=None, **kwargs):
    if ax is None:
        _, ax = plt.subplots(1, 1, squeeze=True)
    kde2plot_op(ax, x, y, grid, **kwargs)
    return ax


def spherical_kde_op(
        lats0, lons0, lats=None, lons=None, grid_size=(200, 200), sigma=None):

    from scipy.special import logsumexp


    if sigma is None:

        sigmahat = vonmises_std(lats=lats0, lons=lons0)
        sigma = 1.06 * sigmahat * lats0.size ** -0.2

    if lats is None and lons is None:
        lats_vec = num.linspace(-90., 90, grid_size[0])
        lons_vec = num.linspace(-180., 180, grid_size[1])

        lons, lats = num.meshgrid(lons_vec, lats_vec)

    if lats is not None:
        assert lats.size == lons.size

    vmf = vonmises_fisher(
        lats=lats, lons=lons,
        lats0=lats0, lons0=lons0, sigma=sigma)
    kde = num.exp(logsumexp(vmf, axis=-1)).reshape(   # , b=self.weights)
        (grid_size[0], grid_size[1]))
    return kde, lats, lons


def v_to_gamma(v):
    """
    Converts from v parameter (Tape2015) to lune longitude [rad]
    """
    return (1. / 3.) * num.arcsin(3. * v)


def w_to_beta(w, u_mapping=None, beta_mapping=None, n=1000):
    """
    Converts from  parameter w (Tape2015) to lune co-latitude
    """
    if beta_mapping is None:
        beta_mapping = num.linspace(0, pi, n)

    if u_mapping is None:
        u_mapping = (
            3. / 4. * beta_mapping) - (
            1. / 2. * num.sin(2. * beta_mapping)) + (
            1. / 16. * num.sin(4. * beta_mapping))
    return num.interp(3. * pi / 8. - w, u_mapping, beta_mapping)


def w_to_delta(w, n=1000):
    """
    Converts from parameter w (Tape2015) to lune latitude
    """
    beta = w_to_beta(w)
    return pi / 2. - beta


def get_gmt_config(gmtpy, h=20., w=20.):

    if gmtpy.is_gmt5(version='newest'):
        gmtconfig = {
            'MAP_GRID_PEN_PRIMARY': '0.1p',
            'MAP_GRID_PEN_SECONDARY': '0.1p',
            'MAP_FRAME_TYPE': 'fancy',
            'FONT_ANNOT_PRIMARY': '14p,Helvetica,black',
            'FONT_ANNOT_SECONDARY': '14p,Helvetica,black',
            'FONT_LABEL': '14p,Helvetica,black',
            'FORMAT_GEO_MAP': 'D',
            'GMT_TRIANGULATE': 'Watson',
            'PS_MEDIA': 'Custom_%ix%i' % (w * gmtpy.cm, h * gmtpy.cm),
        }
    else:
        gmtconfig = {
            'MAP_FRAME_TYPE': 'fancy',
            'GRID_PEN_PRIMARY': '0.01p',
            'ANNOT_FONT_PRIMARY': '1',
            'ANNOT_FONT_SIZE_PRIMARY': '12p',
            'PLOT_DEGREE_FORMAT': 'D',
            'GRID_PEN_SECONDARY': '0.01p',
            'FONT_LABEL': '14p,Helvetica,black',
            'PS_MEDIA': 'Custom_%ix%i' % (w * gmtpy.cm, h * gmtpy.cm),
        }
    return gmtconfig

def lune_plot(v_tape=None, w_tape=None):

    from pyrocko import gmtpy

    if len(gmtpy.detect_gmt_installations()) < 1:
        raise gmtpy.GmtPyError(
            'GMT needs to be installed for lune_plot!')

    fontsize = 14
    font = '1'

    def draw_lune_arcs(gmt, R, J):

        lons = [30., -30., 30., -30.]
        lats = [54.7356, 35.2644, -35.2644, -54.7356]

        gmt.psxy(
            in_columns=(lons, lats), N=True, W='1p,black', R=R, J=J)

    def draw_lune_points(gmt, R, J, labels=True):

        lons = [0., -30., -30., -30., 0., 30., 30., 30., 0.]
        lats = [-90., -54.7356, 0., 35.2644, 90., 54.7356, 0., -35.2644, 0.]
        annotations = [
            '-ISO', '', '+CLVD', '+LVD', '+ISO', '', '-CLVD', '-LVD', 'DC']
        alignments = ['TC', 'TC', 'RM', 'RM', 'BC', 'BC', 'LM', 'LM', 'TC']

        gmt.psxy(in_columns=(lons, lats), N=True, S='p6p', W='1p,0', R=R, J=J)

        rows = []
        if labels:
            farg = ['-F+f+j']
            for lon, lat, text, align in zip(
                    lons, lats, annotations, alignments):

                rows.append((
                    lon, lat,
                    '%i,%s,%s' % (fontsize, font, 'black'),
                    align, text))

            gmt.pstext(
                in_rows=rows,
                N=True, R=R, J=J, D='j5p', *farg)

    def draw_lune_kde(
            gmt, v_tape, w_tape, grid_size=(200, 200), R=None, J=None):

        def check_fixed(a, varname):
            if a.std() == 0:
                a += num.random.normal(loc=0., scale=0.25, size=a.size)

        print(v_tape)
        gamma = num.rad2deg(v_to_gamma(v_tape))   # lune longitude [rad]
        delta = num.rad2deg(w_to_delta(w_tape))   # lune latitude [rad]

        check_fixed(gamma, varname='v')
        check_fixed(delta, varname='w')

        lats_vec, lats_inc = num.linspace(
            -90., 90., grid_size[0], retstep=True)
        lons_vec, lons_inc = num.linspace(
            -30., 30., grid_size[1], retstep=True)
        lons, lats = num.meshgrid(lons_vec, lats_vec)

        kde_vals, _, _ = spherical_kde_op(
            lats0=delta, lons0=gamma,
            lons=lons, lats=lats, grid_size=grid_size)
        Tmin = num.min([0., kde_vals.min()])
        Tmax = num.max([0., kde_vals.max()])

        cptfilepath = '/tmp/tempfile.cpt'
        gmt.makecpt(
            C='white,yellow,orange,red,magenta,violet',
            Z=True, D=True,
            T='%f/%f' % (Tmin, Tmax),
            out_filename=cptfilepath, suppress_defaults=True)

        grdfile = gmt.tempfilename()
        gmt.xyz2grd(
            G=grdfile, R=R, I='%f/%f' % (lons_inc, lats_inc),
            in_columns=(lons.ravel(), lats.ravel(), kde_vals.ravel()),  # noqa
            out_discard=True)

        gmt.grdimage(grdfile, R=R, J=J, C=cptfilepath)

        # gmt.pscontour(
        #    in_columns=(lons.ravel(), lats.ravel(),  kde_vals.ravel()),
        #    R=R, J=J, I=True, N=True, A=True, C=cptfilepath)
        # -Ctmp_$out.cpt -I -N -A- -O -K >> $ps

    h = 20.
    w = h / 1.9

    gmtconfig = get_gmt_config(gmtpy, h=h, w=w)
    bin_width = 15  # tick increment

    J = 'H0/%f' % (w - 5.)
    R = '-30/30/-90/90'
    B = 'f%ig%i/f%ig%i' % (bin_width, bin_width, bin_width, bin_width)
    # range_arg="-T${zmin}/${zmax}/${dz}"

    gmt = gmtpy.GMT(config=gmtconfig)

    draw_lune_kde(
        gmt, v_tape=v_tape, w_tape=w_tape, grid_size=(300, 300), R=R, J=J)
    gmt.psbasemap(R=R, J=J, B=B)
    draw_lune_arcs(gmt, R=R, J=J)
    draw_lune_points(gmt, R=R, J=J)
    return gmt



def _parse_function(proto):
    # define your tfrecord again. Remember that you saved your image as a string.
    keys_to_features = {'image': tf.FixedLenFeature([], tf.string),
                        "label": tf.FixedLenFeature([], tf.int64)}

    # Load one example
    parsed_features = tf.parse_single_example(proto, keys_to_features)

    # Turn your saved image string into an array
    parsed_features['image'] = tf.decode_raw(
        parsed_features['image'], tf.uint8)

    return parsed_features['image'], parsed_features["label"]


def create_dataset(filepath):

    # This works with arrays as well
    dataset = tf.data.TFRecordDataset(filepath)

    # Maps the parser on every filepath in the array. You can set the number of parallel loaders here
    dataset = dataset.map(_parse_function, num_parallel_calls=8)

    # This dataset will go on forever
    dataset = dataset.repeat()

    # Set the number of datapoints you want to load and shuffle
    dataset = dataset.shuffle(SHUFFLE_BUFFER)

    # Set the batchsize
    dataset = dataset.batch(BATCH_SIZE)

    # Create an iterator
    iterator = dataset.make_one_shot_iterator()

    # Create your tf representation of the iterator
    image, label = iterator.get_next()

    # Bring your picture back in shape
    image = tf.reshape(image, [-1, 256, 256, 1])

    # Create a one hot array for your labels
    label = tf.one_hot(label, NUM_CLASSES)

    return image, label

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
                      mechanism=False, sources=None, source_type="MTQT2",
                      mtqt_ps=None, parallel=True, min_depth=None,
                      max_depth=None, max_lat=None, min_lat=None,
                      max_lon=None, min_lon=None, norm=True, con_line=False,
                      max_rho=0, mechanism_and_location=False, as_image=False):

    add_spectrum = False
    data_traces = []
    maxsamples = 0
    max_traces = 1e-02
    min_traces = None
    max_trace = None
    max_traces_block = None
    for traces in waveforms:
        for tr in traces:
        #    tr.ydata = np.abs(tr.ydata)
            if max_traces_block is None:
                max_traces_block = np.max(tr.ydata)
            else:
                if max_traces_block < np.max(tr.ydata):
                    max_traces_block = np.max(tr.ydata)
            if min_traces is None:
                min_traces = np.min(tr.ydata)
            else:
                if min_traces > np.min(tr.ydata):
                    min_traces = np.min(tr.ydata)
            if len(tr.ydata) > maxsamples:
                maxsamples = len(tr.ydata)
    for k, traces in enumerate(waveforms):
        if con_line is True:
            traces_coll = None
        else:
            traces_coll = []

        for tr in traces:
            tr.lowpass(4, 5.4)
            #tr.highpass(4, 0.03)
            #tr.lowpass(4, 20.2)
            tr.highpass(4, 1.)

        traces_orig = copy.deepcopy(traces)
    #    traces = normalize_by_std_deviation(traces)

    #    traces = normalize_by_tracemax(traces)
        #traces = normalize(traces)

#        traces = normalize_chunk(traces)
    #    traces = normalize_all(traces, min_traces, max_traces_block)

        #    tr.highpass(4, 1)
        #    tr.lowpass(4, 10)
        traces_rel_max_values = num.zeros(len(traces))
        traces_rel_max_values_stations = []
        for i, tr_or in enumerate(traces_orig):
            for tr in traces_orig:
                if tr.station == tr_or.station:
                    if traces_rel_max_values[i] == 0:
                        traces_rel_max_values[i] = np.max(abs(tr.ydata))
                    else:
                        if np.max(abs(tr.ydata)) > traces_rel_max_values[i]:
                            traces_rel_max_values[i] = np.max(abs(tr.ydata))
        count_traces = 0
        for tr, tr_orig in zip(traces, traces_orig):

            tr_orig.ydata = tr_orig.ydata/max_traces

            if norm is True:
            #    traces = normalize_by_std_deviation(traces)
            #    tr.ydata = tr.ydata/max_traces_block
            #    tr.ydata = tr.ydata/np.max(tr.ydata)
            #    tr.ydata = tr.ydata/max_traces
            #    tr.ydata =  (tr.ydata/traces_rel_max_values[count_traces])
            #    tr.ydata = 0.5 - (tr.ydata/(np.max(abs(max_traces)))) * 0.5

            #    tr.ydata = 0.5 - (tr.ydata/(np.max(abs(tr.ydata)))) * 0.5
            #    tr.ydata = 0.5 - (tr.ydata/max_traces) * 0.5
                tr.ydata = 0.5 - (tr.ydata/traces_rel_max_values[count_traces]) * 0.5

            count_traces = count_traces + 1

            nsamples = len(tr.ydata)
            data = tr.ydata
            nsamples = len(data)
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
            if con_line is True:
                if traces_coll is None:
                    traces_coll = tr.ydata
                else:
                    traces_coll = np.concatenate((traces_coll, data),  axis=0)

            else:
            #    data = np.append(tr.ydata, tr_orig.ydata)
                data = tr.ydata
                traces_coll.append(data)
                nsamples = len(data)

                #traces_coll.append(tr.ydata)
        #nsamples = len(traces_coll)
        data_traces.append(traces_coll)

        #trd.snuffle(traces)
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
        if parallel is False:
            lats = (lats-np.min(lats))/(np.max(lats)-np.min(lats))
            lons = np.asarray(lons)
            lons = (lons-np.min(lons))/(np.max(lons)-np.min(lons))
            depths = np.asarray(depths)
            depths = (depths-np.min(depths))/(np.max(depths)-np.min(depths))
        else:
            lats = (lats-min_lat)/(max_lat-min_lat)
            lons = np.asarray(lons)
            lons = (lons-min_lon)/(max_lon-min_lon)
            depths = np.asarray(depths)
            depths = (depths-min_depth)/(max_depth-min_depth)

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

                        if source_type == "MTQT" or source_type=="MTQT2":
                            mtqt_p = mtqt_ps[i]
                            if mechanism_and_location is True:
                                labels.append([
                                               0.5-(mtqt_p[1]/(1/3))*0.5,
                                               0.5-(mtqt_p[2]/((3/8)*pi))*0.5,
                                               mtqt_p[3]/(360),
                                               0.5-(mtqt_p[4]/(90))*0.5,
                                               mtqt_p[5]/1,
                                               lats[i], lons[i], depths[i]])
                                event.moment_tensor = mt
                                p = mtqt_p
                                rho, v, w, kappa, sigma, h = p[0], p[1], p[2], p[3], p[4], p[5]
                                #v, w, kappa, sigma, h = 0.5, 0.5, p[0], p[1], p[2]
                                rho = rho*max_rho
                                v = (1/3)-(((1/3)*2)*v)
                                w = ((3/8)*pi)-((((3/8)*pi)*2)*w)
                            #    kappa = kappa*360.
                            #    sigma = 90-((180)*sigma)
                                h = h
                                print(mt)

                                M2 = tt152cmt(rho, v, w, kappa, sigma, h)

                                M2 = change_basis(M2, 1, 2)
                                mt = moment_tensor.MomentTensor(mnn=M2[0], mee=M2[1], mdd=M2[2], mne=M2[3], mnd=M2[4], med=M2[5])
                                print(rho, v, w, kappa, sigma, h, mt)


                            else:
                            #    labels.append([mtqt_p[0]/max_rho,
                            #                   0.5-(mtqt_p[1]/(1/3))*0.5,
                            #                   0.5-(mtqt_p[1]/((3/8)*pi))*0.5,
                            #                   mtqt_p[3]/(360),
                            #                   0.5-(mtqt_p[4]/(90))*0.5,
                            #                   mtqt_p[5]/1])
                                # p = [
                                #            0.5-(mtqt_p[1]/(1/3))*0.5,
                                #            0.5-(mtqt_p[2]/((3/8)*pi))*0.5,
                                #            mtqt_p[3]/(360),
                                #            0.5-(mtqt_p[4]/(90))*0.5,
                                #            mtqt_p[5]/1]
                            #     mt = ev.moment_tensor
                            # #    p = mtqt_p
                            #     rho, v, w, kappa, sigma, h = 1, p[0], p[1], p[2], p[3], p[4]
                            #     rho = rho*max_rho
                            #     v = (1/3)-(((1/3)*2)*v)
                            #     w = ((3/8)*pi)-((((3/8)*pi)*2)*w)
                            #     kappa = kappa*360.
                            #     sigma = 90-((180)*sigma)
                            #     h = h
                            #     print(mtqt_p)
                            #     print(rho, v, w, kappa, sigma, h, mt)
                            #
                            #     M2 = tt152cmt(rho, v, w, kappa, sigma, h)
                            #
                            #     M2 = change_basis(M2, 1, 2)
                            #     mt = moment_tensor.MomentTensor(mnn=M2[0], mee=M2[1], mdd=M2[2], mne=M2[3], mnd=M2[4], med=M2[5])
                            #     print(rho, v, w, kappa, sigma, h, mt)
                            #     print(kill)
                        #        labels.append([
                        #                       0.5-(mtqt_p[1]/(1/3))*0.5,
                        #                       0.5-(mtqt_p[2]/((3/8)*pi))*0.5,
                        #                       mtqt_p[3]/(360),
                        #                       0.5-(mtqt_p[4]/(90))*0.5,
                        #                       mtqt_p[5]/1])

                                labels.append([
                                               0.5-(mtqt_p[1]/(1/3))*0.5,
                                               0.5-(mtqt_p[2]/((3/8)*pi))*0.5,
                                               mtqt_p[3]/(2.*pi),
                                               0.5-(mtqt_p[4]/(pi/2))*0.5,
                                               mtqt_p[5]])
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
    if con_line is False:
        data_traces = data_traces[0]
    data_traces = np.asarray(data_traces)
    if as_image is True:
    #    for event_traces in data_events:
    #        traces = []
    #        for spur in event_traces:
    #            t1 = trd.Trace(
    #            station='TEST', channel='Z', deltat=0.5, tmin=0, ydata=spur)
    #            traces.append(t1)
        plot_waveforms_raw(traces, "images/", iter=str(labels[0][0])+"_"+str(labels[0][1])+"_"+str(labels[0][2])+"_"+str(labels[0][3])+"_"+str(labels[0][4])+"_"+str(labels[0][5])+"_"+str(labels[0][6])+"_"+str(labels[0][7])+"_"+str(labels[0][8]))
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
                       real_noise_traces=None, post=2., pre=0.5,
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
    mechs = np.loadtxt("ridgecrest/scn_test.mech", dtype="str")
    dates = []
    strikes = []
    rakes = []
    dips = []
    depths = []
    lats = []
    lons = []
    events = []
    for i in mechs:
        dates.append(i[1][0:4]+"-"+i[1][5:7]+"-"+i[1][8:]+" "+i[2])
        strikes.append(float(i[16]))
        dips.append(float(i[17]))
        rakes.append(float(i[18]))
        lats.append(float(i[7]))
        lons.append(float(i[8]))
        depths.append(float(i[9]))
        mt = moment_tensor.MomentTensor(strike=float(i[16]), dip=float(i[17]), rake=float(i[18]),
                                        magnitude=float(i[5]))
        event = model.event.Event(lat=float(i[7]), lon=float(i[8]), depth=float(i[9]),
                                  moment_tensor=mt, magnitude=float(i[5]),
                                  time=util.str_to_time(i[1][0:4]+"-"+i[1][5:7]+"-"+i[1][8:]+" "+i[2]))
        events.append(event)
    return events


@ray.remote
def get_parallel_dc(i, targets, store_id, noised, real_noise_traces, post, pre, no_events, stations, mod, params, strikes, dips, rakes,
                    source_type="DC", mechanism=True, multilabel=True, maxvals=None, batch_loading=50, npm=0,
                    dump_full=False, seiger1f=False, path_count=0, paths_disks=None, con_line=True):
    engine = LocalEngine(store_superdirs=['gf_stores'])
    store = engine.get_store(store_id)
    lat, lon, depth = params
    depth_max, depth_min, lat_max, lat_min, lon_max, lon_min = maxvals
    traces_uncuts = []
    traces = []
    sources = []
    events = []
    data_events = []
    labels_events = []
    mag = 5
    count = 0
    npm_rem = npm
    if seiger1f is True:
        current_path = paths_disks[path_count]
    for strike in strikes:
        for dip in dips:
            for rake in rakes:
                traces = []
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
                traces_syn = response.pyrocko_traces()
                event.lat = source_dc.lat
                event.lon = source_dc.lon
                event.depth = source_dc.depth
                mt = moment_tensor.MomentTensor(strike=source_dc.strike, dip=source_dc.dip, rake=source_dc.rake,
                                                magnitude=source_dc.magnitude)
                event.moment_tensor = mt
                if dump_full is True:
                    traces_uncut = copy.deepcopy(traces)
                    traces_uncuts.append(traces_uncut)
                for tr in traces_syn:
                    for st in stations:
                        if st.station == tr.station:
                            processed = False
                            dist = (orthodrome.distance_accurate50m(source_dc.lat,
                                                                 source_dc.lon,
                                                                 st.lat,
                                                                 st.lon)+st.elevation)#*cake.m2d
                            while processed is False:
                                processed = False
                                depth = source_dc.depth
                                arrival = store.t('begin', (depth, dist))
                                if processed is False:
                                #    data_zeros = np.zeros(int(30*(1/tr.deltat)))
                                #    t1 = trace.Trace(
                                #        station=tr.station, channel=tr.channel, deltat=tr.deltat, tmin=event.time, ydata=data_zeros)
                                    tr.chop(arrival-pre, arrival+post)
                                #    tr.shift(tr.tmin+event.time)

                                    #t1.add(tr)
                                    traces.append(tr)
                                #    io.save(t1, "kills")
                                    #tr.resample(1./2.)
                                    #tr.chop(event.time+1,event.time+30)
                                    processed = True
                                else:
                                    pass
                    nsamples = len(tr.ydata)
                    randdata = np.random.normal(size=nsamples)*np.min(tr.ydata)
                    white_noise = trace.Trace(deltat=tr.deltat, tmin=tr.tmin,
                                              ydata=randdata)

                    if noised is True:
                        tr.add(white_noise)
                max_traces = 1e-02

                data_event, label_event, nstations, nsamples = bnn_detector_data([traces], max_traces, events=[event], multilabel=multilabel, mechanism=mechanism, sources=[source_dc],
                                                                                    parallel=True, min_depth=depth_min, max_depth=depth_max, max_lat=lat_max, min_lat=lat_min, max_lon=lon_max, min_lon=lon_min, con_line=con_line,
                                                                                    source_type=source_type)
                data_events.append(data_event)
                labels_events.append(label_event)
                events.append(event)

                if batch_loading == 1:
                    print("stop")
                    if dump_full is True:
                        f = open("grids/grid_%s_SDR%s_%s_%s" % (i, strike, dip, rake), 'wb')
                        pickle.dump([traces, event, source_dc, nsamples, traces_uncut], f)
                        f.close()
                    f = open("grid_ml/grid_%s_SDR%s_%s_%s" % (i, strike, dip, rake), 'wb')
                    pickle.dump([data_events, labels_events, nstations, nsamples, events], f)
                    count = 0
                    data_events = []
                    labels_events = []
                    events = []
                    traces_uncuts = []

                else:
                    if count == batch_loading or npm_rem<batch_loading:
                        npm_rem = npm_rem - batch_loading
                        if dump_full is True:
                            f = open("grids/batch_%s_grid_%s_SDR%s_%s_%s" % (count, i, strike, dip, rake), 'wb')
                            pickle.dump([traces, event, source_dc, nsamples, traces_uncut], f)
                            f.close()
                        if seiger1f is True:
                            free = os.statvfs(current_path)[0]*os.statvfs(current_path)[4]
                            if free < 40000:
                                current_path = paths_disks[path_count+1]
                                path_count = path_count + 1
                            f = open(current_path+"/batch_%s_grid_%s_SDR%s_%s_%s" % (count, i, strike, dip, rake), 'wb')
                            pickle.dump([data_events, labels_events, nstations, nsamples, events], f)
                            f.close()
                        else:
                            f = open("grid_ml/batch_%s_grid_%s_SDR%s_%s_%s" % (count, i, strike, dip, rake), 'wb')
                            pickle.dump([data_events, labels_events, nstations, nsamples, events], f)
                            f.close()
                        count = 0
                        data_events = []
                        labels_events = []
                        events = []
                        traces_uncuts = []
                        mtqt_ps = []
                    else:
                        count = count+1
    return []

@ray.remote
def get_parallel_mtqt(i, targets, store_id, noised, real_noise_traces, post, pre, no_events, stations, mod, params, strikes, dips, rakes,
                    source_type="MTQT2", mechanism=True, multilabel=True, maxvals=None, batch_loading=500, npm=0,
                    dump_full=False, seiger1f=False, path_count=0, paths_disks=None, con_line=True, max_rho=0.):
    if seiger1f is True:
        engine = LocalEngine(store_superdirs=['/diska/home/steinberg/gf_stores'])
    else:
        engine = LocalEngine(store_superdirs=['gf_stores'])
    store = engine.get_store(store_id)
    lat, lon, depth = params
    depth_max, depth_min, lat_max, lat_min, lon_max, lon_min = maxvals
    traces_uncuts = []
    tracess = []
    sources = []
    mtqt_ps = []
    mag = -6
    count = 0
    npm = 72
    # npm = 243
    npm_rem = npm
    data_events = []
    labels_events = []
    events = []
    if seiger1f is True:
        current_path = paths_disks[path_count]
    vs = [0, -0.1]
    ws = [0, -0.1]
#    vs = np.linspace(-1/3, 1/3, 5)
#    ws = np.linspace(-(3/8)*pi,-(3/8)*pi, 5)
    print(len(strikes)*len(dips)*len(rakes)*len(vs)*len(ws))
#    for strike, dip, rake, v, w in zip(strikes, dips, rakes, vs, ws):
    for strike in strikes:
        for dip in dips:
            for rake in rakes:
                for v in vs:
                    for w in ws:
                #for vs in np.arange(-(1/3), (1/3), 0.3):
                #    for ws in np.arange(-(3. / 8.)*num.pi, (3. / 8.)*num.pi, 0.1):

            #                                   0.5-(mtqt_p[1]/(1/3))*0.5,
            #                                   0.5-(mtqt_p[2]/((3/8)*pi))*0.5,
    #            try:
                            event = scenario.gen_random_tectonic_event(i, magmin=-0.5,
                                                                          magmax=3.)
                        #    mt = moment_tensor.MomentTensor(strike=strike, dip=dip,
                        ##                                    rake=rake,
                        #                                    magnitude=mag)
                        #    mt_use = mt.m6_up_south_east()
                    #        mt_input = []
                        #    for mt_comp in mt_use:
                        #        if mt_comp == 0:
                    #                mt_comp += 1e-32
                            #    else:
                            #        mt_comp = mt_comp/mt.moment

                    #            mt_input.append(mt_comp)
                            #rho, v, w, kappa, sigma, h = cmt2tt15(np.array(mt_input))
                            #try:
                            #    kappa = kappa[0]
                            #    sigma = sigma[0]
                        #        h = h[0]
                        #    except:
                        #        kappa = kappa
                        #        sigma = sigma
                        #        h = h
                        #    v = 0
                        #    w = 0
                            kappa = strike
                            sigma = rake
                            h = dip
                            # source_mtqt = MTSource(
                            #      lat=lat,
                            #      lon=lon,
                            #      depth=depth,
                            #      mnn=mt.mnn,
                            #      mee=mt.mee,
                            #      mdd=mt.mdd,
                            #      mne=mt.mne,
                            #      mnd=mt.mnd,
                            #      med=mt.med)
                            #
                            # source_mtqt = DCSource(
                            #       lat=lat,
                            #       lon=lon,
                            #       depth=depth,
                            #       strike=strike,
                            #       dip=dip,
                            #       rake=rake,
                            #      magnitude=mt.magnitude
                            #       )
                        #    u = 0
                        #    v = 0
                            source_mtqt = MTQTSource(
                                lon=lon,
                                lat=lat,
                                depth=depth,
                                w=w,
                                v=v,
                                kappa=kappa,
                                sigma=sigma,
                                h=h,
                                magnitude=mag
                            )

                            response = engine.process(source_mtqt, targets)
                            traces_synthetic = response.pyrocko_traces()

                            event.lat = source_mtqt.lat
                            event.lon = source_mtqt.lon
                            event.depth = source_mtqt.depth
                            event.moment_tensor = source_mtqt.pyrocko_moment_tensor()

                            if dump_full is True:
                                traces_uncut = copy.deepcopy(traces)
                                traces_uncuts.append(traces_uncut)
                            traces = []
                            for tr in traces_synthetic:
                                for st in stations:
                                    if st.station == tr.station:
                                        processed = False
                                        dist = (orthodrome.distance_accurate50m(source_mtqt.lat,
                                                                             source_mtqt.lon,
                                                                             st.lat,
                                                                             st.lon)+st.elevation)#*cake.m2d
                                        while processed is False:
                                            processed = False
                                            depth = source_mtqt.depth
                                            #arrival = store.t('begin', (depth, dist))
                                            arrival = store.t('P', (depth, dist))
        #                                    if arrival_p < arrival:
        #                                        arrival =
                                            if processed is False:
                                            #    data_zeros = np.zeros(int(30*(1/tr.deltat)))
                                            #    data_zeros = np.zeros(int(8*(1/tr.deltat)))

                                                #t1 = trace.Trace(
                                                #    station=tr.station, channel=tr.channel, deltat=tr.deltat, tmin=event.time, ydata=data_zeros)
                                                tr.chop(arrival-pre, arrival+post)
                                                #tr.shift(tr.tmin+event.time)

                                                #t1.add(tr)
                                                traces.append(tr)
                                                #traces.append(t1)
                                                processed = True
                                            else:
                                                pass

                                nsamples = len(tr.ydata)
                                randdata = np.random.normal(size=nsamples)*np.min(tr.ydata)
                                white_noise = trace.Trace(deltat=tr.deltat, tmin=tr.tmin,
                                                          ydata=randdata)
                                if noised is True:
                                    tr.add(white_noise)
                        #    max_traces = 1e-3
                            max_traces = 1e-02
                            rho = 1
                            mtqt_ps = [[rho, v, w, kappa, sigma, h]]

                            data_event, label_event, nstations, nsamples = bnn_detector_data([traces], max_traces, events=[event], multilabel=multilabel, mechanism=mechanism, sources=[source_mtqt],
                                                                                                parallel=True, min_depth=depth_min, max_depth=depth_max, max_lat=lat_max, min_lat=lat_min, max_lon=lon_max, min_lon=lon_min, source_type=source_type, mtqt_ps=mtqt_ps, con_line=con_line, max_rho=max_rho)
                            data_events.append(data_event)
                            labels_events.append(label_event)
                            events.append(event)
                            mtqt_ps.append([rho, v, w, kappa, sigma, h])
                            if batch_loading == 1:
                                if dump_full is True:
                                    f = open("grids/grid_%s_SDR%s_%s_%s" % (i, strike, dip, rake), 'wb')
                                    pickle.dump([traces, event, source_dc, nsamples, traces_uncut], f)
                                    f.close()
                                f = open("grid_ml/grid_%s_SDR%s_%s_%s" % (i, strike, dip, rake), 'wb')
                                pickle.dump([data_events, labels_events, nstations, nsamples, events], f)
                                count = 0
                                data_events = []
                                labels_events = []
                                events = []
                                traces_uncuts = []

                            else:
                                if count == batch_loading or npm_rem<batch_loading:
                                    print(npm_rem, batch_loading, count)
                                    npm_rem = npm_rem - batch_loading
                                    if dump_full is True:
                                        f = open("grids/batch_%s_grid_%s_SDR%s_%s_%s" % (count, i, strike, dip, rake), 'wb')
                                        pickle.dump([traces, event, source_dc, nsamples, traces_uncut], f)
                                        f.close()
                                    if seiger1f is True:
                                        free = os.statvfs(current_path)[0]*os.statvfs(current_path)[4]
                                        if free < 80000:
                                            current_path = paths_disks[path_count+1]
                                            path_count = path_count + 1
                                        f = open(current_path+"/batch_%s_grid_%s_SDR%s_%s_%s_%s_%s" % (count, i, strike, dip, rake, v, w), 'wb')
                                        pickle.dump([data_events, labels_events, nstations, nsamples, events], f)
                                        f.close()
                                    else:
                                        f = open("grid_ml/batch_%s_grid_%s_SDR%s_%s_%s_%s_%s" % (count, i, strike, dip, rake, v, w), 'wb')
                                        pickle.dump([data_events, labels_events, nstations, nsamples, events], f)
                                        f.close()
                                    count = 0
                                    data_events = []
                                    labels_events = []
                                    events = []
                                    traces_uncuts = []
                                    mtqt_ps = []


                                else:
                                    count = count+1
                #        except:
                #            pass
    return []


def generate_test_data_grid(store_id, nevents=50, noised=False,
                            real_noise_traces=None, strike_min=180.,
                            strike_max=360., strike_step=30,
                            dip_min=0., dip_max=90., dip_step=30.,
                            rake_min=-170., rake_max=0., rake_step=30.,
                            mag_min=4.4, mag_max=4.5, mag_step=0.1,
                            depth_step=200., zmin=7000., zmax=7400.,
                            dimx=0.04, dimy=0.04, center=None, source_type="MTQT2",
                            kappa_min=0, kappa_max=2*pi, kappa_step=0.05,
                            sigma_min=-pi/2, sigma_max=pi/2, sigma_step=0.05,
                            h_min=0, h_max=1, h_step=0.05,
                            v_min=-1/3, v_max=1/3, v_step=0.05,
                            u_min=0, u_max=(3/4)*pi, u_step=0.05,
                            pre=0.5, post=3., no_events=False,
                            parallel=True, batch_loading=50,
                            con_line=True):

    mod = landau_layered_model()
    paths_disks = ["/data1/steinberg/grid_ml/", "/data2/steinberg/grid_ml/", "/data3/steinberg/grid_ml/", "/diskb/steinberg/grid_ml/", "/dev/shm/steinberg/grid_ml/"]
    if source_type == "MTQT2":
        rake_min = rake_min+0.0001
        rake_max = rake_max-0.0001

    engine = LocalEngine(store_superdirs=['gf_stores'])
    store = engine.get_store(store_id)
    mod = store.config.earthmodel_1d
    scale = 2e-14
    cake_phase = cake.PhaseDef("P")
    phase_list = [cake_phase]
    waveforms_events = []
    waveforms_events_uncut = []
    waveforms_noise = []
    sources = []
    #change stations

    latmin=35.81
    latmax=35.964
    lonmin=-117.771
    lonmax=-117.61

    latmin=35.905
    latmax=35.910
    lonmin=-117.704
    lonmax=-117.708

#    latmin = 35.7076667
#    latmax = 35.9976667
#    lonmin = -117.9091667
#    lonmax = -117.5091667
    center = [np.mean([latmin, latmax]), np.mean([lonmin, lonmax])]
    dim_step = 0.01
    use_grid = True
    use_coords_from_input = False
    use_coords_from_scn = False
    if use_grid is True:
        lats, lons, depths = make_grid(center, dimx, dimy, zmin, zmax, dim_step, depth_step)
        lats = [35.908]
        lons = [-117.709]
        depths = [4900.]
    else:
        lats = []
        lons = []
        depths = []
        events = []
        params = []
        strikes = []
        rakes = []
        dips = []
        magnitudes = []

    stations_unsorted = model.load_stations("ridgecrest/data/events/stations.pf")
    for st in stations_unsorted:
        st.dist = orthodrome.distance_accurate50m(st.lat, st.lon, lats[0], lons[0])
        st.azi = orthodrome.azimuth(st.lat, st.lon, lats[0], lons[0])
    stations = sorted(stations_unsorted, key=lambda x: x.dist, reverse=True)

    #stations = sorted(stations_unsorted, key=lambda x: x.azi, reverse=True)

#    stations = model.load_stations("/home/asteinbe/src/seiger-detector/landau/events_test/stations.prepared.txt")

    targets = []
    events = []
    mean_lat = []
    mean_lon = []
    max_rho = 0.
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

    if use_coords_from_input is True:
        data_dir = "ridgecrest/data/events/"
    #    data_dir = "/home/asteinbe/src/seiger-detector/landau/events_test/"

        pathlist = Path(data_dir).glob('ev*/')
        print(data_dir)
        for path in sorted(pathlist):

            path = str(path)
            events.append(model.load_events(path+"/event.txt")[0])
        times = []

        strikes = np.arange(180, 360, 10)
        dips = np.arange(0, 90, 10)
        rakes = np.arange(-179, 179, 20)
        magnitudes = np.arange(4.2, 4.5, 0.2)

        for event_scn in events:
            print(event_scn)

            lats.append(event_scn.lat)
            lons.append(event_scn.lon)
        #    depths.append(event_scn.depth*1000.)
            depths.append(event_scn.depth)

            times.append(event_scn.time)
    #        strikes.append(event_scn.moment_tensor.strike1)
    #        dips.append(event_scn.moment_tensor.dip1)
    #        rakes.append(event_scn.moment_tensor.rake1)
    #        magnitudes.append(event_scn.magnitude)
        #events_scn = get_scn_mechs()
        #for event_scn in events_scn:

        #    if event_scn.time in times:
            #if event_scn.magnitude > 4.1:
        #        strikes.append(event_scn.moment_tensor.strike1)
        #        dips.append(event_scn.moment_tensor.dip1)
        #        rakes.append(event_scn.moment_tensor.rake1)
        #        magnitudes.append(event_scn.magnitude)

        #    params.append([event_scn.lat, event_scn.lon, event_scn.depth*1000.])
            params.append([event_scn.lat, event_scn.lon, event_scn.depth])


    if use_coords_from_scn is True:
        events_scn = get_scn_mechs()
        for event_scn in events_scn:
            lats.append(event_scn.lat)
            lons.append(event_scn.lon)
            depths.append(event_scn.depth)
            params.append([event_scn.lat, event_scn.lon, event_scn.depth])
            strikes.append(event_scn.moment_tensor.strike1)
            dips.append(event_scn.moment_tensor.dip1)
            rakes.append(event_scn.moment_tensor.rake1)
            magnitudes.append(event_scn.magnitude)
    if use_coords_from_input is False:

        strikes = np.arange(strike_min, strike_max, strike_step)
        dips = np.arange(dip_min, dip_max, dip_step)
        rakes = np.arange(rake_min, rake_max, rake_step)
        magnitudes = np.arange(mag_min, mag_max, mag_step)
        strikes = [1*pi, 0.5*pi, 0.25*pi]
        dips = [0., 0.25]
        rakes = [-1, -0.5, 0]
    #    strikes = np.linspace(0, 1*pi, 5)
    #    dips = np.linspace(0., 0.5, 5)
    #    rakes = np.linspace(-pi/2., pi/2., 5)
        magnitudes = [4.4]

    kappas = np.arange(kappa_min, kappa_max, kappa_step)
    sigmas = np.arange(sigma_min, sigma_max, sigma_step)
    vs = np.arange(v_min, v_max, v_step)
    us = np.arange(u_min, u_max, u_step)
    hs = np.arange(h_min, h_max, h_step)

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
        if parallel is True:
            if use_grid is True:
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
            npm_geom = len(strikes)*len(dips)*len(rakes)

            maxvals = [np.max(depths), np.min(depths), np.max(lats), np.min(lats), np.max(lons), np.min(lons)]
            f = open("maxvals", 'wb')
            pickle.dump([np.max(depths), np.min(depths), np.max(lats), np.min(lats), np.max(lons), np.min(lons)], f)
            f.close()
            mt_max = moment_tensor.MomentTensor(strike=360,rake=-179.9,dip=90, magnitude=mag_max)
            mt_use_max = mt_max.m6_up_south_east()
            max_rho, v, u, kappa, sigma, h = cmt2tt15(np.array(mt_use_max))
            results = ray.get([get_parallel_mtqt.remote(i, targets, store_id, noised, real_noise_traces, post, pre, no_events, stations, mod, params[i], strikes, dips, rakes, maxvals=maxvals, batch_loading=batch_loading, npm=npm_geom, paths_disks=paths_disks, con_line=con_line, max_rho=max_rho) for i in range(len(params))])
        #    print(results)
        #    for rests in results:
                #print(rests)
        #        for res inpmn rests:
        #            print(kill)

        # load data directly:
    #        pathlist = Path('grids').glob('*')
    #        for path in sorted(pathlist):
    #            f = open(path, 'rb')
    #            tracess, eventss, sourcess, nsamples, uncut = pickle.load(f)
    #            for i, traces in enumerate(tracess):
    ##                events.append(eventss[i])
    #                waveforms_events.append(traces)
    #                nsamples = nsamples
    #                sources.append(sourcess[i])
                #    waveforms_events_uncut.append(res[4])
            del results, params
        else:
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
            npm_geom = len(strikes)*len(dips)*len(rakes)

            maxvals = [np.max(depths), np.min(depths), np.max(lats), np.min(lats), np.max(lons), np.min(lons)]
            f = open("maxvals", 'wb')
            pickle.dump([np.max(depths), np.min(depths), np.max(lats), np.min(lats), np.max(lons), np.min(lons)], f)
            f.close()
            results = ray.get([get_parallel_dc.remote(i, targets, store_id, noised, real_noise_traces, post, pre, no_events, stations, mod, params[i], strikes, dips, rakes, maxvals=maxvals, batch_loading=batch_loading, npm=npm_geom, paths_disks=paths_disks, con_line=con_line) for i in range(len(params))])
        #    print(results)
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
    if parallel is True:
        return waveforms_events
    else:
        return waveforms_events, waveforms_noise, nsamples, len(stations), events, sources, mtqt_ps


def m6_ridgecrest():
    return [-0.25898825,  0.61811539, -0.35912714, -0.67312731,  0.35961476,  0.35849677]


def rgb2gray(rgb):
    return np.dot(rgb[..., :3], [0.2989, 0.5870, 0.1140])


def load_data(data_dir, store_id, scenario=False, stations=None, pre=0.5,
              post=3, gf_freq=None):
    mod = landau_layered_model()
    engine = LocalEngine(store_superdirs=['/home/asteinbe/gf_stores'])
    store = engine.get_store(store_id)
    mod = store.config.earthmodel_1d
    cake_phase = cake.PhaseDef("P")
    phase_list = [cake_phase]
    from pathlib import Path
    events = []
    waveforms = []
    if scenario is True:
        pathlist = Path(data_dir).glob('scenario*/')
    else:
        pathlist = Path(data_dir).glob('ev_0/')
    for path in sorted(pathlist):
        try:
            targets = []
            path = str(path)+"/"
            traces_event = []
            event = model.load_events(path+"event.txt")[0]
            traces_loaded = io.load(path+"traces.mseed")
            #trace.snuffle(traces_loaded)
            if scenario is True:
                stations_unsorted = model.load_stations(path+"stations.pf")
            else:
                stations_unsorted = model.load_stations(data_dir+"stations.pf")
            lats = [35.908]
            lons = [-117.709]
            depths = [4900.]
            for st in stations_unsorted:
                st.dist = orthodrome.distance_accurate50m(st.lat, st.lon, lats[0], lons[0])
                st.azi = orthodrome.azimuth(st.lat, st.lon, lats[0], lons[0])
        #    stations = sorted(stations_unsorted, key=lambda x: x.dist, reverse=True)
        #    stations = sorted(stations_unsorted, key=lambda x: x.azi, reverse=True)
            stations = sorted(stations_unsorted, key=lambda x: x.dist, reverse=True)

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
            traces = []
            tracess = []
            for st in stations:
                found = False

                for tr in traces_loaded:
                    if st.station == tr.station:
                        if len(tr.ydata) > 420:
                            traces.append(tr)
                            found = True
                if found == False:
                    traces.append()
                    data_zeros = np.zeros(int(30*(1/tr.deltat)))
                    t1 = trace.Trace(
                        station=st.station, channel=tr.channel, deltat=tr.deltat, tmin=event.time, ydata=data_zeros)

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
                for st in stations:
                    for tr in traces:
                        nsamples = len(tr.ydata)
                        if st.station == tr.station:
                            dists = (orthodrome.distance_accurate50m(event.lat,
                                                                     event.lon,
                                                                     st.lat,
                                                                     st.lon)+st.elevation)*cake.m2d
                            processed = False

                            for i, arrival in enumerate(mod.arrivals([dists],
                                                        phases=["p", "P"],
                                                        zstart=event.depth)):
                                if processed is False:

                                #        if gf_freq is not None:
                                #            tr.resample(1./gf_freq)
                                        try:

                                        #    tr.chop(event.time+arrival.t-pre, event.time+arrival.t+post)
                                        #    tr.chop(event.time+arrival.t-pre, event.time+arrival.t+post)
                                        #    data_zeros = np.zeros(int(30*(1/tr.deltat)))
                                        #    t1 = trace.Trace(
                                        #        station=tr.station, channel=tr.channel, deltat=tr.deltat, tmin=event.time, ydata=data_zeros)
                                            tr.chop(event.time+arrival.t-pre, event.time+arrival.t+post)
                                            #tr.shift(-event.time)
                                        #    t1.add(tr)
                                            tracess.append(tr)
                                            processed = True

                                        except:
                                            traces.remove(tr)
                                        # deal with empty traces
                                        # deal with shift around
                                #        pass
                                else:
                                    pass
                    nsamples = len(tr.ydata)
                traces_event.append(tracess)
            nsamples = len(tr.ydata)
            events.append(event)
            waveforms.append(tracess)
        except:
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


def read_wavepickle(path):
    f = open(path, 'rb')
    data_events, labels_events, nstations, nsamples, events = pickle.load(f)
    f.close()
    return data_events, labels_events



class WaveformImageGenerator(keras.utils.Sequence):

    def __init__(self, filenames, batch_size):
        self.filenames = filenames
        self.batch_size = batch_size

    def __len__(self):
        return (np.ceil(len(self.filenames) / float(self.batch_size))).astype(np.int)

    def __getitem__(self, idx) :
        batch_x = self.filenames[idx * self.batch_size : (idx+1) * self.batch_size]
        data = []
        labels = []
        for filename in batch_x:
            img = image.imread(filename)
            data.append(np.asarray(img))
            label = str(filename).split("_")
            label_float = []
            for item in label[1:-1]:
                label_float.append(float(item))

            labels.append(label_float)

        return np.array(data), np.array(labels)

    def getitem(filenames, batch_size, idx) :
        batch_x = filenames[idx * batch_size : (idx+1) * batch_size]
        data = []
        labels = []
        for filename in batch_x:
            data_events, labels_events = read_wavepickle(filename)
            data.append(data_events)
            labels.append(labels_events)
        return np.array(data), np.array(labels)


class WaveformGenerator(keras.utils.Sequence):

    def __init__(self, filenames, batch_size):
        self.filenames = filenames
        self.batch_size = batch_size

    def __len__(self):
        return (np.ceil(len(self.filenames) / float(self.batch_size))).astype(np.int)

    def __getitem__(self, idx) :
        batch_x = self.filenames[idx * self.batch_size : (idx+1) * self.batch_size]
        data = []
        labels = []
        for filename in batch_x:
            data_events, labels_events = read_wavepickle(filename)
            data.append(data_events)
            labels.append(labels_events)

        return np.array(data), np.array(labels)

    def getitem(filenames, batch_size, idx) :
        batch_x = filenames[idx * batch_size : (idx+1) * batch_size]
        data = []
        labels = []
        for filename in batch_x:
            data_events, labels_events = read_wavepickle(filename)
            data.append(data_events)
            labels.append(labels_events)
        return np.array(data), np.array(labels)


class WaveformGenerator_SingleBatch(keras.utils.Sequence):

    def __init__(self, filenames, batch_size):
        self.filenames = filenames
        self.batch_size = batch_size

    #def __len__(self):
    #    return (np.ceil(len(self.filenames) / float(self.batch_size))).astype(np.int)
    def __len__(self):
        return (np.ceil(len(self.filenames))).astype(np.int)

    def __getitem__(self, idx):
        batch_x = self.filenames[idx]
        f = open(batch_x, 'rb')
        data_events, labels_events, nstations, nsamples, events = pickle.load(f)
        data = []
        labels = []
        f.close()
        for d, l in zip(data_events, labels_events):
            data.append(d)
            labels.append(l)
    #    data = np.asarray(data)
    #    data = data.reshape(data.shape+(1,)) # here comp as dimensions?

        return np.array(data), np.array(labels)

    def getitem(filenames, batch_size, idx):
        batch_x = filenames[idx]
        f = open(batch_x, 'rb')
        data_events, labels_events, nstations, nsamples, events = pickle.load(f)
        f.close()
        return np.array(data_events), np.array(labels_events), events

    def getitem_perturbed(filenames, batch_size, idx):
        batch_x = filenames[idx]
        f = open(batch_x, 'rb')
        data_events, labels_events, nstations, nsamples, events = pickle.load(f)
        f.close()


        return np.array(data_events), np.array(labels_events), events


import numpy as np
import keras

class DDataGenerator(keras.utils.Sequence):
    'Generates data for Keras'
    def __init__(self, filenames, batch_size=51, dim=(32,32,32), n_channels=1,
                 n_classes=10, shuffle=True):
        'Initialization'
        self.filenames = filenames
        self.dim = dim
        self.batch_size = batch_size
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.shuffle = shuffle
        self.on_epoch_end()

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.filenames)))

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch

        batch_x = self.filenames[index]
        f = open(batch_x, 'rb')
        data_events, labels_events, nstations, nsamples, events = pickle.load(f)
        data = []
        labels = []
        f.close()
        for d, l in zip(data_events, labels_events):
            labels.append(l[0])
            d = np.asarray(d)
            #d = d.reshape((1,)+d.shape+(1,))
            d = d.reshape(d.shape+(1,))
            data.append(d)
        return np.array(data), np.array(labels)

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.filenames))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __data_generation(self, list_IDs_temp):
        'Generates data containing batch_size samples' # X : (n_samples, *dim, n_channels)
        # Initialization
        X = np.empty((self.batch_size, *self.dim, self.n_channels))
        y = np.empty((self.batch_size), dtype=int)

        # Generate data
        for i, ID in enumerate(list_IDs_temp):
            # Store sample
            X[i,] = np.load('data/' + ID + '.npy')

            # Store class
            y[i] = self.labels[ID]

        return X, keras.utils.to_categorical(y, num_classes=self.n_classes)

def my2dGenerator(filenames, batchsize=72, npm=72):
    batchsize = batchsize
    while 1:
        s = 0
        data = []
        labels = []
        npm_rem = npm
        for i in range(len(filenames)): # 1875 * 32 = 60000 -> # of training samples
            batch_x = filenames[i]
            f = open(batch_x, 'rb')
            data_events, labels_events, nstations, nsamples, events = pickle.load(f)
            f.close()
            for d, l in zip(data_events, labels_events):
                labels.append(l[0])
                d = np.asarray(d)
                #d = d.reshape((1,)+d.shape+(1,))
                d = d.reshape(d.shape+(1,))
                data.append(d)
                npm_rem = npm_rem - 1
                if len(labels) == batchsize:
                    yield np.array(data), np.array(labels)
                    data = []
                    labels = []

class WaveformGenerator_SingleBatch2d(keras.utils.Sequence):

    def __init__(self, filenames, batch_size):
        self.filenames = filenames
        self.batch_size = batch_size
    def __len__(self):
        return 51
#    def __len__(self):
#        return (np.ceil(len(self.filenames) / float(self.batch_size))).astype(np.int)
#    def __len__(self):
#        return (np.ceil(len(self.filenames))).astype(np.int)

    def __getitem__(self, idx):
        batch_x = self.filenames[idx]
        f = open(batch_x, 'rb')
        data_events, labels_events, nstations, nsamples, events = pickle.load(f)
        data = []
        labels = []
        f.close()
        for d, l in zip(data_events, labels_events):
            labels.append(l)
            d = np.asarray(d)
            d = d.reshape((1,)+d.shape) # here comp as dimensions?
            data.append(d)
        print(np.shape(data))
        print(kill)
        return np.array(data), np.array(labels)

    def getitem(filenames, batch_size, idx):
        batch_x = filenames[idx]
        f = open(batch_x, 'rb')
        data_events, labels_events, nstations, nsamples, events = pickle.load(f)
        f.close()
        return np.array(data_events), np.array(labels_events), events

    def getitem_perturbed(filenames, batch_size, idx):
        batch_x = filenames[idx]
        f = open(batch_x, 'rb')
        data_events, labels_events, nstations, nsamples, events = pickle.load(f)
        f.close()


        return np.array(data_events), np.array(labels_events), events


class WaveformGenerator_SingleBatch_to_line(keras.utils.Sequence):

    def __init__(self, filenames, batch_size):
        self.filenames = filenames
        self.batch_size = batch_size

    def __len__(self):
        return (np.ceil(len(self.filenames))).astype(np.int)

    def __getitem__(self, idx):
        batch_x = self.filenames[idx]
        f = open(batch_x, 'rb')
        data_events, labels_events, nstations, nsamples, events = pickle.load(f)
        data = []
        labels = []
        f.close()
        for d, l in zip(data_events, labels_events):
            d = d.ravel()
            data.append(d)
            labels.append(l)
        return np.array(data), np.array(labels)

    def getitem(filenames, batch_size, idx):
        batch_x = filenames[idx]
        f = open(batch_x, 'rb')
        data_events, labels_events, nstations, nsamples, events = pickle.load(f)
        f.close()
        data = []
        labels = []
        f.close()
        for d, l in zip(data_events, labels_events):
            d.ravel()
            data.append(d)
            l.ravel()
            labels.append(l)
    #    data = np.asarray(data)
    #    data = data.reshape((1,)+(1,)+data.shape)
        return np.array(data), np.array(labels), events

    def getitem_perturbed(filenames, batch_size, idx):
        batch_x = filenames[idx]
        f = open(batch_x, 'rb')
        data_events, labels_events, nstations, nsamples, events = pickle.load(f)
        f.close()

        return np.array(data_events), np.array(labels_events), events


class WaveformGenerator_SingleBatch_line(keras.utils.Sequence):

    def __init__(self, filenames, batch_size):
        self.filenames = filenames
        self.batch_size = batch_size

    def __len__(self):
        return (np.ceil(len(self.filenames))).astype(np.int)

    def __getitem__(self, idx):
        batch_x = self.filenames[idx]
        f = open(batch_x, 'rb')
        data_events, labels_events, nstations, nsamples, events = pickle.load(f)
        data = []
        labels = []
        f.close()
        for d, l in zip(data_events, labels_events):
            data.append(d)
            labels.append(l)
        return np.array(data), np.array(labels)

    def getitem(filenames, batch_size, idx):
        batch_x = filenames[idx]
        f = open(batch_x, 'rb')
        data_events, labels_events, nstations, nsamples, events = pickle.load(f)
        f.close()
        data = []
        labels = []
        f.close()
        for d, l in zip(data_events, labels_events):
            d.ravel()
            data.append(d)
            l.ravel()
            labels.append(l)
    #    data = np.asarray(data)
    #    data = data.reshape((1,)+(1,)+data.shape))
        return np.array(data), np.array(labels), events

    def getitem_perturbed(filenames, batch_size, idx):
        batch_x = filenames[idx]
        f = open(batch_x, 'rb')
        data_events, labels_events, nstations, nsamples, events = pickle.load(f)
        f.close()

        return np.array(data_events), np.array(labels_events), events


class WaveformGenerator_SingleBatch_line_to2d(keras.utils.Sequence):

    def __init__(self, filenames, batch_size):
        self.filenames = filenames
        self.batch_size = batch_size

    def __len__(self):
        return (np.ceil(len(self.filenames))).astype(np.int)

    def __getitem__(self, idx):
        batch_x = self.filenames[idx]
        f = open(batch_x, 'rb')
        data_events, labels_events, nstations, nsamples, events = pickle.load(f)
        data = []
        labels = []
        f.close()
        comp_count = 0
        comp = []
        for d, l in zip(data_events, labels_events):
            for i in range(0, nstations):
                if comp_count < 2:
                    comp_count = comp_count + 1
                    comp.append(d[50*i])
                else:
                    data.append(comp)
                    labels.append(l)
                    comp_count = 0
                    comp = []
        return np.array(data), np.array(labels)

    def getitem(filenames, batch_size, idx):
        batch_x = filenames[idx]
        f = open(batch_x, 'rb')
        data_events, labels_events, nstations, nsamples, events = pickle.load(f)
        f.close()
        data = []
        labels = []
        f.close()
        for d, l in zip(data_events, labels_events):
            d.ravel()
            data.append(d)
            l.ravel()
            labels.append(l)
    #    data = np.asarray(data)
    #    data = data.reshape((1,)+(1,)+data.shape)
        return np.array(data), np.array(labels), events

    def getitem_perturbed(filenames, batch_size, idx):
        batch_x = filenames[idx]
        f = open(batch_x, 'rb')
        data_events, labels_events, nstations, nsamples, events = pickle.load(f)
        f.close()

        return np.array(data_events), np.array(labels_events), events


def plot_mechanisms_on_map(events, stations, center):
    from pyrocko.plot.automap import Map
    from pyrocko.example import get_example_data
    from pyrocko import model, gmtpy
    from pyrocko import moment_tensor as pmt

    gmtpy.check_have_gmt()

    # Generate the basic map
    m = Map(
        lat=center[0],
        lon=center[1],
        radius=150000.,
        width=30., height=30.,
        show_grid=False,
        show_topo=True,
        color_dry=(238, 236, 230),
        topo_cpt_wet='light_sea_uniform',
        topo_cpt_dry='light_land_uniform',
        illuminate=True,
        illuminate_factor_ocean=0.15,
        show_rivers=False,
        show_plates=True)

    # Draw some larger cities covered by the map area
    m.draw_cities()

    # Generate with latitute, longitude and labels of the stations
    lats = [s.lat for s in stations]
    lons = [s.lon for s in stations]
    labels = ['.'.join(s.nsl()) for s in stations]

    # Stations as black triangles. Genuine GMT commands can be parsed by the maps'
    # gmt attribute. Last argument of the psxy function call pipes the maps'
    # pojection system.
    m.gmt.psxy(in_columns=(lons, lats), S='t20p', G='black', *m.jxyr)

    # Station labels
    for i in range(len(stations)):
        m.add_label(lats[i], lons[i], labels[i])


    # Load events from catalog file (generated using catalog.GlobalCMT()
    # download from www.globalcmt.org)
    # If no moment tensor is provided in the catalogue, the event is plotted
    # as a red circle. Symbol size relative to magnitude.

    beachball_symbol = 'd'
    factor_symbl_size = 5.0
    for ev in events:
        mag = ev.magnitude
        if ev.moment_tensor is None:
            ev_symb = 'c'+str(mag*factor_symbl_size)+'p'
            m.gmt.psxy(
                in_rows=[[ev.lon, ev.lat]],
                S=ev_symb,
                G=gmtpy.color('scarletred2'),
                W='1p,black',
                *m.jxyr)
        else:
            devi = ev.moment_tensor.deviatoric()
            beachball_size = mag*factor_symbl_size
            mt = devi.m_up_south_east()
            mt = mt / ev.moment_tensor.scalar_moment() \
                * pmt.magnitude_to_moment(5.0)
            m6 = pmt.to6(mt)
            data = (ev.lon, ev.lat, 10) + tuple(m6) + (1, 0, 0)

            if m.gmt.is_gmt5():
                kwargs = dict(
                    M=True,
                    S='%s%g' % (beachball_symbol[0], (beachball_size) / gmtpy.cm))
            else:
                kwargs = dict(
                    S='%s%g' % (beachball_symbol[0],
                                (beachball_size)*2 / gmtpy.cm))

            m.gmt.psmeca(
                in_rows=[data],
                G=gmtpy.color('chocolate1'),
                E='white',
                W='1p,%s' % gmtpy.color('chocolate3'),
                *m.jxyr,
                **kwargs)

    m.save('automap_area.png')



def plot_mechanisms_on_map_fuzzy(events_list, stations, center):
    from pyrocko.plot.automap import Map
    from pyrocko.example import get_example_data
    from pyrocko import model, gmtpy
    from pyrocko import moment_tensor as pmt

    gmtpy.check_have_gmt()

    # Generate the basic map
    m = Map(
        lat=center[0],
        lon=center[1],
        radius=50000.,
        width=30., height=30.,
        show_grid=False,
        show_topo=True,
        color_dry=(238, 236, 230),
        topo_cpt_wet='light_sea_uniform',
        topo_cpt_dry='light_land_uniform',
        illuminate=True,
        illuminate_factor_ocean=0.15,
        show_rivers=False,
        show_plates=True)

    # Draw some larger cities covered by the map area
    m.draw_cities()

    # Generate with latitute, longitude and labels of the stations
    lats = [s.lat for s in stations]
    lons = [s.lon for s in stations]
    labels = ['.'.join(s.nsl()) for s in stations]

    # Stations as black triangles. Genuine GMT commands can be parsed by the maps'
    # gmt attribute. Last argument of the psxy function call pipes the maps'
    # pojection system.
    m.gmt.psxy(in_columns=(lons, lats), S='t20p', G='black', *m.jxyr)

    # Station labels
    for i in range(len(stations)):
        m.add_label(lats[i], lons[i], labels[i])

    beachball_symbol = 'd'
    factor_symbl_size = 5.0
    for events in events_list:
        for ev in events:
                mts.append(mtm.MomentTensor.from_values(
                (ev.moment_tensor.strike1, ev.moment_tensor.dip1, ev.moment_tensor.rake1)))

        mag = ev.magnitude

        devi = ev.moment_tensor.deviatoric()
        beachball_size = mag*factor_symbl_size
        mt = devi.m_up_south_east()
        mt = mt / ev.moment_tensor.scalar_moment() \
            * pmt.magnitude_to_moment(5.0)
        m6 = pmt.to6(mt)
        data = (ev.lon, ev.lat, 10) + tuple(m6) + (1, 0, 0)

        if m.gmt.is_gmt5():
            kwargs = dict(
                M=True,
                S='%s%g' % (beachball_symbol[0], (beachball_size) / gmtpy.cm))
        else:
            kwargs = dict(
                S='%s%g' % (beachball_symbol[0],
                            (beachball_size)*2 / gmtpy.cm))

        m.gmt.psmeca(
            in_rows=[data],
            G=gmtpy.color('chocolate1'),
            E='white',
            W='1p,%s' % gmtpy.color('chocolate3'),
            *m.jxyr,
            **kwargs)

    m.save('automap__fuzzy_area.png')

def bnn_detector(waveforms_events=None, waveforms_noise=None, load=True,
                 multilabel=True, data_dir=None, train_model=True,
                 detector_only=False, validation_data=None, wanted_start=None,
                 wanted_end=None, mode="detector_only", parallel=True,
                 batch_loading=50, source_type="MTQT2",
                 perturb_val=True, con_line=False, to_line=False,
                 store_id="mojave_large_ml",
                 mechanism_and_location=False,
                 as_image=False, conv1d=False):

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
            if parallel is False:
                try:
                    f = open("data_unseen_waveforms_bnn_gt_loaded", 'rb')
                    waveforms_events, nsamples, nstations, events = pickle.load(f)
                    f.close()
                except:
                    waveforms_events, nsamples, nstations, events = load_data(data_dir,
                                                                              store_id)
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
                waveforms_events, nsamples, nstations, events = load_data(data_dir, store_id)
                f = open("data_waveforms_bnn_gt_loaded", 'wb')
                pickle.dump([waveforms_events, nsamples, nstations, events], f)
                f.close()
        else:
            try:
                f = open("data_waveforms_bnn_gt", 'rb')
                waveforms_events, waveforms_noise, nsamples, nstations, events = pickle.load(f)
                f.close()
            except:

                waveforms_events, waveforms_noise, nsamples, nstations, events = generate_test_data(store_id, nevents=1200)
                f = open("data_waveforms_bnn_gt", 'wb')
                pickle.dump([waveforms_events, waveforms_noise, nsamples, nstations,
                             events], f)
                f.close()
            sources = None
    else:
        if parallel is True:
            if len(os.listdir('grid_ml/')) == 0:
                batch_loading = 50
                waveforms_events = generate_test_data_grid(store_id, nevents=1200, batch_loading=batch_loading, source_type=source_type, con_line=con_line, parallel=parallel)
            else:
                print("Grid already calculated")

        else:
            try:
                print("loading")
                f = open("data_waveforms_bnn_mechanism", 'rb')
                waveforms_events, waveforms_noise, nsamples, nstations, events, sources, mtqt_ps = pickle.load(f)
                f.close()
            except:

                waveforms_events, waveforms_noise, nsamples, nstations, events, sources, mtqt_ps = generate_test_data_grid(store_id, nevents=1200, con_line=con_line, parallel=parallel)
                f = open("data_waveforms_bnn_mechanism", 'wb')
                print("dump")
                pickle.dump([waveforms_events, waveforms_noise, nsamples, nstations,
                             events, sources, mtqt_ps], f)
                f.close()
    print("prep done")
    pr.disable()
    filename = 'profile_bnn.prof'
    pr.dump_stats(filename)
    if validation_data is None:
        max_traces = 0.
        if parallel is False:
            for traces in waveforms_events:
                for tr in traces:
                    if np.max(tr.ydata) > max_traces:
                        max_traces = np.max(tr.ydata)

            data_events, labels_events, nstations, nsamples = bnn_detector_data(waveforms_events, max_traces, events=events, multilabel=multilabel, mechanism=mechanism, sources=sources, con_line=con_line,source_type=source_type)
    #        print(len(data_events))
    #        print(labels_events)
        if data_dir is not None:
            if parallel is False:
                data_events_unseen, labels_events_unseen, nstations_unseen, nsamples_unseen = bnn_detector_data(waveforms_unseen, max_traces, events=events_unseen, multilabel=multilabel,
                                                                                                        mechanism=mechanism, con_line=con_line,source_type=source_type)
        if detector_only is True:
            data_noise, labels_noise, nstations, nsamples = bnn_detector_data(waveforms_noise, max_traces, events=None, multilabel=multilabel, mechanism=mechanism, con_line=con_line,source_type=source_type)
            x_data = np.concatenate((data_events, data_noise), axis=0)
            y_data = np.concatenate((labels_events, labels_noise), axis= 0)
            #x_data = data_events
            #y_data = labels_events
            from keras.utils import to_categorical
            y_array = None
        else:
            if parallel is False:
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
        data_events_unseen, labels_events_unseen, nstations_unseen, nsamples_unseen = bnn_detector_data(waveforms_unseen, max_traces, events=None, multilabel=multilabel, mechanism=mechanism, con_line=con_line,source_type=source_type)

        x_data = data_events_unseen
        y_data = labels_events_unseen

    if parallel is False:
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


        headline_data = dat
        headline_labels = labels
        additional_labels = labels
        additional_data = labels
    else:
            retrieved_meta = False
            pathlist = Path('grid_ml').glob('*')
            while retrieved_meta is False:
                for path in sorted(pathlist):
                    f = open(path, 'rb')
                    data_events, labels_events, nstations, nsamples, events = pickle.load(f)
                    f.close()
                    retrieved_meta = True
                    break
                    print("retrieved")
            if source_type == "DC":
                if mechanism_and_location is True:
                    nlabels = 9
                else:
                    nlabels = 6
            if source_type == "MTQT" or source_type == "MTQT2":
                if mechanism_and_location is True:
                    nlabels = 9
                else:
                    nlabels = 5
    np.random.seed(42)  # Set a random seed for reproducibility

    from sklearn.model_selection import train_test_split
#    from keras.models import Sequential
#    from keras.layers import Dense, Activation, Masking
#    from keras.layers import Dense, Dropout, Activation
    from tensorflow.keras import Sequential
    from tensorflow.keras.layers import Dense, Activation, Masking, Dense, Dropout, Activation, Conv2D, Flatten
# For a single-input model with 2 classes (binary classi    fication):
    if train_model is True:
        #print(nstations)
        model = Sequential(name="MT")
        if parallel is True and con_line is False:

            #plt.show()
            #plt.figure()
            #plt.plot(data_events[0][0])
            #plt.show()
            #plt.imshow(data_events[0])
            #plt.show()
        #    print((data_events.shape[0],),(1,),data_events.shape[1:])
            #plt.imshow(data_events[0])
            #plt.show()
        #    model.add(Activation('relu', input_shape=np.shape(data_events)))
            print(np.shape(data_events))
        #    print(np.shape(data_events))
            print(nlabels)
            print(nstations)
            print(nsamples)
            if conv1d is False:
                data_events = np.asarray(data_events)
                data_events = data_events.reshape((data_events.shape[0],)+data_events.shape[1:]+(1,))
                print(np.shape(data_events))
                #model.add(Conv2D(filters=1, kernel_size=(1,1), activation="relu", input_shape=np.shape(data_events)[1:]))
                model.add(Activation('relu', input_shape=np.shape(data_events)[1:]))
            else:
                model.add(Activation('relu', input_shape=np.shape(data_events)[1:]))

#                model.add(Conv2D(1, 1, 1,
#                                        border_mode='valid',
#                                        input_shape=(123, 70, 1)))
            if as_image is False:
            #    model.add(Activation('relu', input_shape=np.shape(data_events)[1:]))
                print("none")
            else:
                image_data = image.imread("images/_0.581866733418902_0.3678746675226514_0.5502585990584465_0.6977426346063564_0.4155393144092137_0.4459477392022609_0.0_0.0_0.0_.png")
                data_events = np.asarray(image_data)
                #model.add(Activation('relu', input_shape=np.shape(data_events)))

        elif parallel is True and con_line is True:
            print(np.shape(data_events))
            print(np.shape(data_events)[1]*np.shape(data_events)[2])
            #model.add(Activation('relu', input_shape=(np.shape(data_events)[1]*np.shape(data_events)[2],)))
            model.add(Activation('relu', input_shape=np.shape(data_events)[1:]))
            print(nsamples)

        #else:
        #    model.add(Activation('relu'))
        if con_line is False:
        #    model.add(Conv1D(int(nsamples), int(nstations/3), activation="relu"))
        #    model.add(Dropout(0.1))
        #    model.add(MaxPooling1D(2))
        #    model.add(Conv1D(int(nsamples/2), int(nstations/4), activation="relu"))
        #    model.add(Dropout(0.1))
        #    model.add(Conv1D(int(nsamples/6), int(nstations/8), activation="relu"))
        #    model.add(Conv1D(35, 1, activation="relu"))
        #    model.add(Conv1D(20, 41, activation="relu"))
        #    model.add(Conv1D(15, 20, activation="relu"))
        #    model.add(Conv1D(12, 12, activation="relu"))
        #    model.add(Conv1D(11, 53, activation="relu"))
        #    model.add(Conv1D(35, 61, activation="relu"))
        #    model.add(Conv1D(15, 63, activation="relu"))
            if conv1d is True:
             model.add(Conv1D(int(nsamples), int(nstations/3), activation="relu"))
             model.add(Dropout(0.1))
             model.add(MaxPooling1D(2))
             model.add(Conv1D(int(nsamples/2), int(nstations/4), activation="relu"))
             model.add(Dropout(0.1))
             model.add(MaxPooling1D(12))

        #    model.add(Conv2D(filters=1, kernel_size=(123,1), activation="relu"))
            ##model.add(Conv2D(filters=1, kernel_size=(3,3), activation="relu"))
            #model.add(Conv2D(filters=10, kernel_size=(300,300), activation="relu"))
            #model.add(Conv2D(filters=20, kernel_size=(30,30), activation="relu"))
            else:
        #        model.add(Conv2D(filters=64, kernel_size=(3,3), activation="relu"))
            #    model.add(MaxPooling2D(41,1))
        #        model.add(Dropout(0.3))
        #        model.add(MaxPooling2D(pool_size=2))
                #model.add(Conv2D(filters=64, kernel_size=(3,3), activation="relu"))
                #model.add(MaxPooling2D(pool_size=2))
                #model.add(Dropout(0.3))

                #model.add(Conv2D(filters=32, kernel_size=(3,3), activation="relu"))
        #        model.add(Conv2D(filters=64, kernel_size=(3,3), activation="relu", name='block1_conv1'))
                #model.add(MaxPooling2D(pool_size=2))
                #model.add(Dropout(0.3))

                kl_weight = 1.0 / 72.
                prior_params = {
                    'prior_sigma_1': 1.5,
                    'prior_sigma_2': 0.1,
                    'prior_pi': 0.5
                }
                kl_divergence_function = (lambda q, p, _: tfd.kl_divergence(q, p) /  # pylint: disable=g-long-lambda
                                        tf.cast(72, dtype=tf.float32))
        #         model.add(Conv2D(filters=16, kernel_size=(3,3), activation="relu"))
        # #        model.add(Conv2D(filters=64, kernel_size=(3,3), activation="relu", name='block1_conv1'))
        #         model.add(MaxPooling2D(pool_size=2))
        #         model.add(Dropout(0.3))
        #
        #         model.add(Conv2D(filters=8, kernel_size=(3,3), activation="relu"))
        # #        model.add(Conv2D(filters=64, kernel_size=(3,3), activation="relu", name='block1_conv1'))
        #         model.add(MaxPooling2D(pool_size=2))
        #         model.add(Dropout(0.3))
        #
        #    #    model.add(MaxPooling2D(41,1))
        #         model.add(Dropout(0.3))
        #         model.add(MaxPooling2D(pool_size=2))
        #         model.add(Conv2D(filters=64, kernel_size=(3,3), activation="relu"))
        #     #    model.add(MaxPooling2D(41,1))
        #         model.add(Dropout(0.3))
        #         model.add(MaxPooling2D(pool_size=(1,2)))
        #
        #         model.add(Conv2D(filters=32, kernel_size=(3,3), activation="relu"))
        # #        model.add(Conv2D(filters=64, kernel_size=(3,3), activation="relu", name='block1_conv1'))
        #         model.add(MaxPooling2D(pool_size=(1,2)))
        #         model.add(Dropout(0.3))
        #
        #         model.add(Conv2D(filters=8, kernel_size=(3,3), activation="relu", name='block1_conv1'))
        #     #    model.add(MaxPooling2D(pool_size=2))
        #         model.add(Dropout(0.3))

                model.add(Conv2D(filters=2, kernel_size=(1,2), activation="relu"))
                model.add(Conv2D(filters=3, kernel_size=(3,1), activation="relu"))
            #    model.add(Conv2D(filters=2, kernel_size=(3,3), activation="relu"))
            #    model.add(Conv2D(filters=2, kernel_size=(3,3), activation="relu"))

            #    model.add(tfp.layers.Convolution2DReparameterization(2, kernel_size=(1,2),  padding="SAME", activation=tf.nn.relu))
            #    model.add(tfp.layers.Convolution2DReparameterization(3, kernel_size=(3,1),  padding="SAME", activation=tf.nn.relu))
            #    model.add(tfp.layers.Convolution2DReparameterization(3, kernel_size=(3,3),  padding="SAME", activation=tf.nn.relu))
            #    model.add(tfp.layers.Convolution2DReparameterization(3, kernel_size=(3,3),  padding="SAME", activation=tf.nn.relu))

            #    model.add(tfp.layers.Convolution2DFlipout(filters=2, kernel_size=(1,2), activation="relu", kernel_divergence_fn=kl_divergence_function))
            #    model.add(tfp.layers.Convolution2DFlipout(filters=3, kernel_size=(3,1), activation="relu", kernel_divergence_fn=kl_divergence_function))
            #    model.add(tfp.layers.Convolution2DFlipout(filters=3, kernel_size=(3,3), activation="relu", kernel_divergence_fn=kl_divergence_function))
            #    model.add(tfp.layers.Convolution2DFlipout(filters=3, kernel_size=(3,3), activation="relu", kernel_divergence_fn=kl_divergence_function))

        #        model.add(Dropout(0.1))

            #    model.add(Conv2D(filters=2, kernel_size=(1,2), activation="relu"))
            #    model.add(Dropout(0.3))

            #    model.add(Conv2D(filters=2, kernel_size=(1,2), activation="relu"))
            #    model.add(Conv2D(filters=3, kernel_size=(1,3), activation="relu"))
            #    model.add(Conv2D(filters=2, kernel_size=(1,3), activation="relu"))
            #    model.add(Conv2D(filters=2, kernel_size=(1,10), activation="relu"))

        #        model.add(Conv2D(filters=4, kernel_size=(2,2), activation="relu"))
        #        model.add(Dropout(0.3))

        #        model.add(Dropout(0.1))
            #    model.add(Conv2D(filters=4, kernel_size=(9,9), activation="relu"))
        #        model.add(Dropout(0.1))

            #    model.add(Conv2D(filters=2, kernel_size=(3,3), activation="relu"))
            #    model.add(Conv2D(filters=2, kernel_size=(3,3), activation="relu"))

        #        model.add(Dropout(0.1))
            #    model.add(Conv2D(filters=8, kernel_size=(9,9), activation="relu"))
            #    model.add(Conv2D(filters=8, kernel_size=(3,3), activation="relu"))
            #    model.add(Conv2D(filters=8, kernel_size=(3,3), activation="relu"))
            #    model.add(Conv2D(filters=8, kernel_size=(3,3), activation="relu"))

            #    model.add(Conv2D(filters=8, kernel_size=(3,3), activation="relu"))
        #        model.add(Conv2D(filters=2, kernel_size=(9,9), activation="relu"))
        #        model.add(Conv2D(filters=2, kernel_size=(18,18), activation="relu"))

            #    model.add(MaxPooling2D(41,1))
            #    model.add(Dropout(0.3))
            #    model.add(MaxPooling2D(pool_size=(1,2)))

            #    model.add(Conv2D(filters=8, kernel_size=(3,3), activation="relu"))
        #        model.add(Conv2D(filters=64, kernel_size=(3,3), activation="relu", name='block1_conv1'))
            #    model.add(MaxPooling2D(pool_size=(1,2)))
            #    model.add(Dropout(0.3))

            #    model.add(Conv2D(filters=8, kernel_size=(3,3), activation="relu", name='block1_conv1'))
            #    model.add(MaxPooling2D(pool_size=2))
            #    model.add(Dropout(0.3))

        #        model.add(Conv2D(filters=8, kernel_size=(3,3), activation="relu", name='block1_conv1'))
            #    model.add(MaxPooling2D(pool_size=2))
        #        model.add(Dropout(0.3))

    #        model.add(Conv2D(filters=32, kernel_size=(3,3), activation="relu"))

        #    model.add(MaxPooling2D(20,20))
        #    model.add(LSTM(2))

            #model.add(Conv2D(filters=1, kernel_size=(1,3), activation="relu"))

                model.add(Flatten())

            #    model.add(Dense(nstations*9, activation='relu'))

        #        model.add(Dense(nstations*6, activation='relu'))
        #        model.add(Dense(nstations*3, activation='relu'))

        #        model.add(Dense(nstations, activation='relu'))
            #    model.add(Dense(32*32, activation='relu'))
            #    model.add(Dense(3*3*3, activation='relu'))
            #    model.add(Dense(nlabels*10, activation='relu'))
            #    model.add(Dense(nlabels*10*2, activation='relu'))
                #model.add(Dense(72, activation='relu'))
            #    model.add(tfp.layers.DenseFlipout(72*72,
            #                    activation="relu"))


            #    model.add(tfp.layers.DenseVariational(72, posterior_mean_field, prior_trainable))
            #    model.add(DenseVariational(72, kl_weight, **prior_params, activation='relu'))
            #    model.add(tfp.layers.DenseVariational(1, posterior_mean_field, prior_trainable, kl_weight=1/72.))
        #        model.add(tfp.layers.DistributionLambda(lambda t: tfd.Normal(loc=t, scale=1)))
                model.add(Dense(72*72, activation='relu'))
            #    model.add(Dense(nlabels*10, activation='relu'))

#                model.add(Dense(nstations, activation='softmax'))
            #    model.add(Dense(int(nstations/3), activation='relu'))

            #    model.add(Dense(256, activation='softmax'))
            #    model.add(Dense(40, activation='softmax'))

        else:
        #    model.add(Dense(512, activation='relu', input_dim=nsamples))

            model.add(Dense(256, activation='relu', input_dim=nsamples))
#        model.add(Dropout(0.5))
    #    model.add(Dense(64, activation='relu', input_dim=nsamples))
    #    model.add(Dropout(0.5))
    #    model.add(Dense(120, activation='relu', input_dim=nsamples))
    #    model.add(Dense(120, activation='relu', input_dim=nsamples))
            model.add(Dense(40, activation='relu', input_dim=nsamples))
        #    model.add(Dense(nstations, activation='relu'))
        #    model.add(Conv1D(filters=1, kernel_size=(3), activation="relu"))
    #        model.add(Conv2D(filters=1, kernel_size=(3,3), activation="relu"))
        #    model.add(Flatten())
        #    model.add(LSTM(50))

            #model.add(Dropout(0.2))
        #    model.add(MaxPooling1D(20))
    #        model.add(Conv1D(int(nsamples/3), 83, activation="relu"))

        #    model.add(Dropout(0.2))
        #    model.add(MaxPooling1D(2))
    #        model.add(MaxPooling2D((2,2)))
        #    model.add(LSTM(int(nstations/3)))
        #    model.add(Dense(nstations/3, activation='relu'))

        #    model.add(Conv1D(nsamples, 83, activation="relu"))
        #    model.add(Dropout(0.2))
            #model.add(MaxPooling1D())
        #    model.add(MaxPooling1D(2))
        #    model.add(Conv1D(int(nsamples/4), int(nstations/6), activation="relu"))
        #    model.add(Dropout(0.2))
        #    model.add(MaxPooling1D(2))
            #model.add(LSTM((1, 1, 1)))
        #    model.add(Dense(nsamples, activation='relu', input_dim=nsamples))
            #model.add(Conv1D(int(nsamples/4), int(nstations/6), activation="relu"))
#        model.add(layers.Conv2D(32, 3, activation="relu"))

#        model.add(Dense(2056, activation='relu', input_dim=nsamples))

    #    model.add(Dense(1060, activation='relu', input_dim=nsamples))
    #    model.add(Dropout(0.5))
        #model.add(Masking(mask_value=-99999999))
    #    model.add(Dense(3600, activation='relu', input_dim=nsamples))
#        model.add(Dropout(0.5))
    #    model.add(Dense(64, activation='relu', input_dim=nsamples))
    #    model.add(Dropout(0.5))
    #        model.add(Dense(300, activation='relu'))
    #    model.add(Dense(30, activation='relu', input_dim=nsamples))
    #    model.add(Dense(21, activation='relu', input_dim=nsamples))

        #model.add(Dense(11664, activation='relu', input_dim=nsamples))
    #    model.add(Dense(36, activation='relu', input_dim=nsamples))
    #    model.add(Dense(12, activation='relu', input_dim=nsamples))

#        model.add(Dense(38, activation='relu', input_dim=nsamples))
    #    model.add(Dense(3600, activation='relu', input_dim=nsamples))
    #    model.add(Dense(36, activation='relu', input_dim=nsamples))
    #    model.add(Conv1D(121,1, activation="relu"))

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
    #    model.add(Dense(nlabels, activation='sigmoid', name="predictions"))

    #    model.add(Dense(nlabels, activation='sigmoid', name="predictions"))
        neg_log_likelihood = lambda x, rv_x: -rv_x.log_prob(x)
        prior = tfd.Independent(tfd.Normal(loc=tf.zeros(5, dtype=tf.float64), scale=1.0),
        reinterpreted_batch_ndims=1)
    #    model.add(Dense(nlabels, name="predictions"))
        model.add(Dense(tfp.layers.MultivariateNormalTriL.params_size(
                5), activation=None, name="distribution_weights"))
        model.add(tfp.layers.MultivariateNormalTriL(5, name="output"))
    #    model.add(tfp.layers.DenseVariational(3, posterior_mean_field, prior_trainable))
    #    model.add(tfp.layers.DenseFlipout(5))

    #    model.add(tfp.layers.DistributionLambda(lambda t: tfd.Normal(loc=t, scale=1)))
        # adadelta
        #opt = keras.optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=True)
    #    model.compile(optimizer='rmsprop',
    #                  loss='binary_crossentropy',
    #                  metrics=['accuracy'])

    #    from keras.optimizers import Adam
        from tensorflow.keras.optimizers import Adam
        #lr=0.0001
    #    negloglik = lambda y, rv_y: -rv_y.log_prob(y)

    #    model.compile(optimizer=Adam(lr=0.001),
    #                  loss='mean_squared_error',
    #                  metrics=['accuracy'])

        model.compile(optimizer=Adam(lr=0.001),
                      loss=neg_log_likelihood,
                      metrics=['accuracy'])
        model.summary()
    #    model.compile(optimizer='adam',
    #                  loss='binary_crossentropy',
    #                  metrics=['accuracy'])
        print("compile")
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

    viz_steps = 2
    #pred = model.predict(train)
    bayesian = False

    if bayesian is False:
        if parallel is False:
            data = dat
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
                if parallel is False:
                    history = model.fit(train, y_train, epochs=5, batch_size=400,
                                        callbacks=[checkpointer])
                else:
                    batch_size = 51

                    my_training_batch_generator = WaveformGenerator(X_train_filenames, y_train, batch_size)
                    my_validation_batch_generator = WaveformGenerator(X_train_filenames, y_val, batch_size)
                    model.fit_generator(generator=my_training_batch_generator,
                                       steps_per_epoch = int(3800 // batch_size),
                                       epochs = 10,
                                       verbose = 1,
                                       validation_data = my_validation_batch_generator,
                                       validation_steps = int(950 // batch_size))
            else:
                if parallel is False:
                    history = model.fit(dat, labels, epochs=1000, batch_size=400,
                                        callbacks=[checkpointer])
                else:
                    batch_size = 51
                    paths = []
                    pathlist = Path('grid_ml').glob('*')
                    for path in sorted(pathlist):
                        paths.append(path)
                #    model.build()
                    model.summary()
                    if batch_loading == 1:
                        batch_size = batch_loading
                        my_training_batch_generator = WaveformGenerator(paths, batch_size)
                        my_validation_batch_generator = WaveformGenerator(paths, batch_size)
                    else:
                        batch_size = batch_loading+1
                        if perturb_val is True:
                            from sklearn.utils import shuffle
                            filenames_shuffled = shuffle(paths)
                            train_paths, val_paths, y_train, y_val = train_test_split(
                                filenames_shuffled, np.ones(len(filenames_shuffled)), test_size=0.2, random_state=1)
                        else:
                            train_paths = filenames
                            val_paths = filenames
                    #    train_paths = filenames_shuffled
                    #    val_paths = filenames_shuffled



                        if con_line is False and to_line is False:
                            if conv1d is True:
                                my_training_batch_generator = WaveformGenerator_SingleBatch(train_paths, batch_size)
                                my_validation_batch_generator = WaveformGenerator_SingleBatch(val_paths, batch_size)
        #                        my_validation_batch_generator = WaveformGenerator_SingleBatch(val_paths, batch_size)
                                my_validation_batch_generator = WaveformGenerator_SingleBatch(train_paths, batch_size)
                            else:
                                my_training_batch_generator = WaveformGenerator_SingleBatch2d(train_paths, batch_size)
                                my_validation_batch_generator = WaveformGenerator_SingleBatch2d(val_paths, batch_size)
        #                        my_validation_batch_generator = WaveformGenerator_SingleBatch(val_paths, batch_size)
                                my_validation_batch_generator = WaveformGenerator_SingleBatch2d(train_paths, batch_size)

                                my_training_batch_generator = my2dGenerator(filenames_shuffled)
                                my_validation_batch_generator = my2dGenerator(filenames_shuffled)
        #                        my_validation_batch_generator = WaveformGenerator_SingleBatch(val_paths, batch_size)
                        #        my_training_batch_generator = DDataGenerator(filenames_shuffled)
                        #        my_validation_batch_generator = DDataGenerator(filenames_shuffled)

                        elif con_line is True:
                            my_training_batch_generator = WaveformGenerator_SingleBatch_line(train_paths, batch_size)
                            #my_validation_batch_generator = WaveformGenerator_SingleBatch_to_line(val_paths, batch_size)
    #                        my_validation_batch_generator = WaveformGenerator_SingleBatch(val_paths, batch_size)
                            my_validation_batch_generator = WaveformGenerator_SingleBatch_line(val_paths, batch_size)

                        else:
                            my_training_batch_generator = WaveformGenerator_SingleBatch_to_line(train_paths, batch_size)
                            #my_validation_batch_generator = WaveformGenerator_SingleBatch_to_line(val_paths, batch_size)
    #                        my_validation_batch_generator = WaveformGenerator_SingleBatch(val_paths, batch_size)
                            my_validation_batch_generator = WaveformGenerator_SingleBatch_to_line(train_paths, batch_size)

                    if as_image is True:
                            paths = []
                            pathlist = Path('images/').glob('*')
                            for path in sorted(pathlist):
                                paths.append(path)
                            my_validation_batch_generator = WaveformImageGenerator(paths, batch_size)
                            my_training_batch_generator = WaveformImageGenerator(paths, batch_size)
                    variational_bayesian = False
                    if variational_bayesian is True:
                        train_seq = []
                        heldout_seq = []
                        for i in range(len(filenames_shuffled)): # 1875 * 32 = 60000 -> # of training samples
                            batch_x = filenames_shuffled[i]
                            f = open(batch_x, 'rb')
                            data_events, labels_events, nstations, nsamples, events = pickle.load(f)
                            f.close()
                            labels = []
                            data = []
                            for d, l in zip(data_events, labels_events):
                                labels.append(l[0])
                                d = np.asarray(d)
                                #d = d.reshape((1,)+d.shape+(1,))
                                d = d.reshape(d.shape+(1,))
                                data.append(d)
                                heldout_seq.append([np.array(d)])

                            train_seq.append([np.array(data), np.array(labels)])
                        #train_seq = np.array(data), np.array(labels)
                        num_monte_carlo = 10
                        for epoch in range(100):
                            epoch_accuracy, epoch_loss = [], []
                            for step, (batch_x, batch_y) in enumerate(train_seq):
                              batch_loss, batch_accuracy = model.train_on_batch(
                                  batch_x, batch_y)
                              epoch_accuracy.append(batch_accuracy)
                              epoch_loss.append(batch_loss)
                              probs = tf.stack([model.predict(heldout_seq, verbose=1)
                                              for _ in range(num_monte_carlo)], axis=0)
                              mean_probs = tf.reduce_mean(probs, axis=0)
                              heldout_log_prob = tf.reduce_mean(tf.math.log(mean_probs))
                    else:
                        model.fit_generator(generator=my_training_batch_generator,
                                           steps_per_epoch = 2,
                                          verbose=1,
                                           epochs = 100, callbacks=[checkpointer])
                #    model.fit_generator(generator=my_training_batch_generator,
                #                       steps_per_epoch = 2,
                #                      verbose=1,
                #                       epochs = 100, callbacks=[checkpointer])
                    #x_val = dat
                    #y_val = labelsw

            #plot_model(model)
        #    layer_outputs = [layer.output for layer in model.layers[:]]
            #  Extracts the outputs of the top 12 layers
        #    activation_model = models.Model(inputs=model.input, outputs=layer_outputs)
        #    plot_acc_loss(history)
        else:
            if detector_only is True:
                model = keras.models.load_model('model_detector')
            if mode == "mechanism_mode":
            #    model = keras.models.load_model('model_mechanism.tf')
                 model = tf.keras.models.load_model('model_mechanism.tf')

            #    kl_divergence_function = (lambda q, p, _: tfd.kl_divergence(q, p) /  # pylint: disable=g-long-lambda
            #                            tf.cast(72, dtype=tf.float32))
            #    model = tf.saved_model.load('model_mechanism.tf', tags=None, export_dir=None)
                # file = h5py.File('models/{}.h5'.format(model_name), 'r')
                # weight = []
                # for i in range(len(file.keys())):
                #    weight.append(file['weight' + str(i)][:])
                # model.set_weights(weight)
            else:
                model = keras.models.load_model('model_locator')
        if data_dir is not None or validation_data is not None:
            if parallel is False:
                pred = model.predict(data_events_unseen)
            else:
                engine = LocalEngine(store_superdirs=['/home/asteinbe/gf_stores'])
                store = engine.get_store(store_id)
                waveforms_events, nsamples, nstations, events = load_data(data_dir,
                                                                          store_id,
                                                                          gf_freq=store.config.sample_rate)

                #f = open("data_unseen_waveforms_bnn_gt_loaded", 'wb')
                #pickle.dump([waveforms_events, nsamples, nstations, events], f)
                #f.close()
                data_events, label_event, nstations, nsamples = bnn_detector_data(waveforms_events, max_traces, events=None, multilabel=multilabel, mechanism=mechanism,
                                                                                parallel=True, con_line=con_line)

                print(data_events)
                data = []
                data.append(data_events)
                data_events = np.asarray(data)
                print(np.shape(data_events))
                print(label_event)
            #    data_events.append(data_event)
                if conv1d is False:
                    data_events = np.asarray(data_events)
                    data_events = data_events.reshape((data_events.shape[0],)+data_events.shape[1:]+(1,))
                pred = model.predict(data_events)
        else:
            if parallel is False:
                pred = model.predict(x_val)
            else:
                if data_dir is None:
                    paths = []
                    pathlist = Path('grid_ml').glob('*')
                    for path in sorted(pathlist):
                        paths.append(path)

                if batch_loading == 1:
                    batch_size = batch_loading
                    my_training_batch_generator = WaveformGenerator(paths, batch_size)
                    my_validation_batch_generator = WaveformGenerator(paths, batch_size)
                else:
                    batch_size = batch_loading+1
                    if perturb_val is True:
                        from sklearn.utils import shuffle
                        filenames_shuffled = shuffle(paths)
                        train_paths, val_paths, y_train, y_val = train_test_split(
                            filenames_shuffled, np.ones(len(filenames_shuffled)), test_size=0.2, random_state=1)
                    else:
                        train_paths = filenames
                        val_paths = filenames
                #    train_paths = filenames_shuffled
                #    val_paths = filenames_shuffled


                    if conv1d is False:
                        my_validation_batch_generator = my2dGenerator(train_paths, batchsize=10)
                    #    my_validation_batch_generator = DDataGenerator(train_paths)

                    elif con_line is False and to_line is False:
                        my_training_batch_generator = WaveformGenerator_SingleBatch(train_paths, batch_size)
                        my_validation_batch_generator = WaveformGenerator_SingleBatch(val_paths, batch_size)
#                        my_validation_batch_generator = WaveformGenerator_SingleBatch(val_paths, batch_size)
                        my_validation_batch_generator = WaveformGenerator_SingleBatch(train_paths, batch_size)

                    elif con_line is True:
                        my_training_batch_generator = WaveformGenerator_SingleBatch_line(train_paths, batch_size)
                        #my_validation_batch_generator = WaveformGenerator_SingleBatch_to_line(val_paths, batch_size)
#                        my_validation_batch_generator = WaveformGenerator_SingleBatch(val_paths, batch_size)
                        my_validation_batch_generator = WaveformGenerator_SingleBatch_line(val_paths, batch_size)



                    else:
                        my_training_batch_generator = WaveformGenerator_SingleBatch_to_line(train_paths, batch_size)
                        #my_validation_batch_generator = WaveformGenerator_SingleBatch_to_line(val_paths, batch_size)
#                        my_validation_batch_generator = WaveformGenerator_SingleBatch(val_paths, batch_size)
                        my_validation_batch_generator = WaveformGenerator_SingleBatch_to_line(train_paths, batch_size)
                if conv1d is False:
                    print("pred")
                    pred = model.predict_generator(my_validation_batch_generator, 1)
                else:
                    pred = model.predict_generator(my_validation_batch_generator, 1)


    else:
        # implement for pickle loading
        # save data not in pickle but native?
        train, x_val, y_train, y_val = train_test_split(dat,
                                                        labels,
                                                        test_size=0.2,
                                                        random_state=10)

        num_epochs = 5
        batchsize = 51
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

    if parallel is True and data_dir is None:
        idxs = [0]
        y_val = []
        for idx in idxs:
            if con_line is False and to_line is False:
                data, values, events = WaveformGenerator_SingleBatch.getitem(paths, batch_size, idx)
            #    print(len(events), len(values), "here")
            #    print(values)
            elif con_line is True:
                data, values, events = WaveformGenerator_SingleBatch_line.getitem(paths, batch_size, idx)
            else:
                data, values = WaveformGenerator_SingleBatch_line.getitem(paths, batch_size, idx)
            #data, values = WaveformGenerator.getitem(paths, batch_size, idx)
            for val in values:
            #    print(val)
                y_val.append(val[0])
            #for val in values[0]:
            #    y_val.append(val[0])
        preds_new = []
        vals_new = []

        mag_max = 5.2
        mt_max = moment_tensor.MomentTensor(strike=340,rake=-179.9,dip=90, magnitude=mag_max)
        mt_use_max = mt_max.m6_up_south_east()
        max_rho, v, u, kappa, sigma, h = cmt2tt15(np.array(mt_use_max))
        pred_ms = []
        real_ms = []

    #    import keract
    #    model = tf.keras.models.load_model('model_mechanism')
        ##data_events = np.asarray(data_events)
    #    input = data_events.reshape((data_events.shape[0],)+data_events.shape[1:])
    #    input = data_events[0]
    #    input = input.reshape((1,)+input.shape)

    #    activations = keract.get_activations(model, input)
    #    keract.display_heatmaps(activations, input, save=True)
        if conv1d is False:
            pred = [pred]
        if source_type == "MTQT" or source_type=="MTQT2":

            for p in pred:
            #    if conv1d is False:
            #        p = p[0]
                p = p[0]

#                rho, v, w, kappa, sigma, h = p[0], p[1], p[2], p[3], p[4], p[5]
                v, w, kappa, sigma, h = p[0], p[1], p[2], p[3], p[4]
            #    v, w, kappa, sigma, h = 0.5, 0.5, p[0], p[1], p[2]

                rho = 1
#                rho = rho*max_rho
                v = (1/3)-(((1/3)*2)*v)
                w = ((3/8)*pi)-((((3/8)*pi)*2)*w)
            #    kappa = kappa*360.
            #    sigma = 90-((180)*sigma)
                kappa = kappa*2.*pi
                sigma = (pi/2)-(2*(pi/2)*sigma)
                h = h
                if h > 1.:
                    h = 1.
                print(rho, v, w, kappa, sigma, h)
                #M2 = tt152cmt(rho, v, w, kappa, sigma, h)

                #M2 = change_basis(M2, 1, 2)
                mtqt_source = MTQTSource(v=v, w=w, kappa=kappa, sigma=sigma, h=h)
                mt = mtqt_source.pyrocko_moment_tensor()
                M2 = mtqt_source.m6

                #mt = moment_tensor.MomentTensor(mnn=M2[0], mee=M2[1], mdd=M2[2], mne=M2[3], mnd=M2[4], med=M2[5])
                pred_ms.append(mt)

            #    M2 = M2.tolist()
                if mechanism_and_location is True:
                    M2.append(p[6])
                    M2.append(p[7])
                    M2.append(p[8])
                preds_new.append(M2)

            for p in y_val:
                if source_type == "MTQT" or source_type=="MTQT2":
        #            rho, v, w, kappa, sigma, h = p[0], p[1], p[2], p[3], p[4], p[5]
                    rho = rho*max_rho
                    v, w, kappa, sigma, h = p[0], p[1], p[2], p[3], p[4]
                #    v, w, kappa, sigma, h = 0.5, 0.5, p[0], p[1], p[2]
                    rho = 1
                    v = (1/3)-(((1/3)*2)*v)
                    w = ((3/8)*pi)-((((3/8)*pi)*2)*w)
                #    kappa = kappa*360.
                #    sigma = 90-((180)*sigma)
                    kappa = kappa*2.*pi
                    sigma = (pi/2)-(2*(pi/2)*sigma)
                    h = h
                #    M2 = tt152cmt(rho, v, w, kappa, sigma, h)
                #    M2 = change_basis(M2, 1, 2)
                #    mt = moment_tensor.MomentTensor(mnn=M2[0], mee=M2[1], mdd=M2[2], mne=M2[3], mnd=M2[4], med=M2[5])
                    mtqt_source = MTQTSource(v=v, w=w, kappa=kappa, sigma=sigma, h=h)
                    mt = mtqt_source.pyrocko_moment_tensor()
                    real_ms.append(mt)
                    M2 = mtqt_source.m6
                    #M2 = M2.tolist()
                    if mechanism_and_location is True:
                        M2.append(p[6])
                        M2.append(p[7])
                        M2.append(p[8])
                    vals_new.append(M2)
            vals = vals_new
            for pred_m, real_m in zip(pred_ms, real_ms):
                print("pred", pred_m)
                print(real_m)
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

        for p in pred:
            print(p)
            if source_type == "MTQT" or source_type=="MTQT2":
                p = p[0]
            else:
                preds_new.append(p[0])
        pred = preds_new
    #print(y_val[-3:-1], pred[-3:-1])
    #print(abs(y_val)-abs(pred))
#    print(np.sum(abs(y_val)-abs(pred)))
    recover_real_value = False
    sdr = False
    if multilabel is True:
        if mechanism is True:
            if sdr is False:
                real_values = []
                real_ms = []
                if data_dir is None:

                    for i, vals in enumerate(y_val):
                        if source_type == "DC":
                            real_value = [(1.-(2.*vals[0]))*events[i].moment_tensor.moment*2, (1.-(2.*vals[1]))*events[i].moment_tensor.moment*2,
                                                (1.-(2.*vals[2]))*events[i].moment_tensor.moment*2, (1.-(2.*vals[3]))*events[i].moment_tensor.moment*2,
                                                (1.-(2.*vals[4]))*events[i].moment_tensor.moment*2, (1.-(2.*vals[5]))*events[i].moment_tensor.moment*2]
                            real_values.append(real_value)
                            m = moment_tensor.MomentTensor(mnn=(1.-(2.*vals[0]))*events[i].moment_tensor.moment*2, mee=(1.-(2.*vals[1]))*events[i].moment_tensor.moment*2,
                                                mdd=(1.-(2.*vals[2]))*events[i].moment_tensor.moment*2, mne=(1.-(2.*vals[3]))*events[i].moment_tensor.moment*2,
                                                mnd=(1.-(2.*vals[4]))*events[i].moment_tensor.moment*2, med=(1.-(2.*vals[5]))*events[i].moment_tensor.moment*2)
                            #print(vals)
                        #    print(m.both_strike_dip_rake())
                        #    print(events[i])
                        #    print(sources[i])
                            real_ms.append(m)
                real_values_pred = []
                diff_values = []
                pred_ms = []
                if conv1d is False and data_dir is not None:
                    pred = [pred]
                print(len(pred), len(events))
                if data_dir is None:
                    if len(events)< batch_size:
                        pred = pred[0:len(events)]
                for i, vals in enumerate(pred):
                    print(i)
                    print(events[i])
                    print(vals)
                    if data_dir is None:
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
                    else:
                        if source_type == "DC":
                            vals = vals[0]
                            real_value = [(1.-(2.*vals[0])), (1.-(2.*vals[1])),
                                                (1.-(2.*vals[2])), (1.-(2.*vals[3])),
                                                (1.-(2.*vals[4])), (1.-(2.*vals[5]))]
                            real_values_pred.append(real_value)
                            print(real_value)
                            m = moment_tensor.MomentTensor(mnn=(1.-(2.*vals[0])), mee=(1.-(2.*vals[1])),
                                                mdd=(1.-(2.*vals[2])), mne=(1.-(2.*vals[3])),
                                                mnd=(1.-(2.*vals[4])), med=(1.-(2.*vals[5])))
                            print(m.both_strike_dip_rake())
                        #    print(real_value)
                            from pyrocko import plot
                            from pyrocko.plot import beachball
                            fig = plt.figure()
                            axes = fig.add_subplot(1, 1, 1)
                            axes.set_xlim(-2., 4.)
                            axes.set_ylim(-2., 2.)
                            axes.set_axis_off()
                            plot.beachball.plot_beachball_mpl(
                                        m,
                                        axes,
                                        beachball_type='deviatoric',
                                        size=60.,
                                        position=(0, 1),
                                        color_t=plot.mpl_color('scarletred2'),
                                        linewidth=1.0)
                            plt.show()
                            pred_ms.append(m)
                            f = open("maxvals", 'rb')
                            depths_max, depths_min, lats_max, lats_min, lons_max, lons_min = pickle.load(f)
                            f.close()

                            data = data_events[i]
                            predss = model.predict(data_events)
                            data = data.reshape(123,70)
                            print(np.shape((data)))
                            print(np.max(data))
                            plt.imshow(data)
                            plt.show()
                            print(model)
                            import keract

                            activations = keract.get_activations(model, data_events)
                            keract.display_heatmaps(activations, data_events[i], save=True)

                            lat = (-vals[6]*lats_min)+(vals[6]*lats_max)+lats_min
                            lon = (-vals[7]*lons_min)+(vals[7]*lons_max)+lons_min
                            depth = (-vals[8]*depths_min)+(vals[8]*depths_max)+depths_min
                            print(lat,lon, depth)
                        else:

                            p = vals[0]
                            print(model)
                            import keract

                    #        activations = keract.get_activations(model, data_events)
                    #        keract.display_heatmaps(activations, data_events[i], save=True)

                        #    rho, v, w, kappa, sigma, h = p[0], p[1], p[2], p[3], p[4], p[5]
                            v, w, kappa, sigma, h = p[0], p[1], p[2], p[3], p[4]
                        #    v, w, kappa, sigma, h = 0.5, 0.5, p[0], p[1], p[2]
                            rho = 1
                            #rho = rho*max_rho
                            v = (1/3)-(((1/3)*2)*v)
                            w = ((3/8)*pi)-((((3/8)*pi)*2)*w)
                        #    kappa = kappa*360.
                        #    sigma = 90-((180)*sigma)
                            kappa = kappa*2.*pi
                            sigma = (pi/2)-(2*(pi/2)*sigma)
                            h = h
                            print("pred", p)
                            import keract

                            activations = keract.get_activations(model, data_events)
                            keract.display_heatmaps(activations, data_events[i], save=True)
                        #    M2 = tt152cmt(rho, v, w, kappa, sigma, h)
                        #    M2 = change_basis(M2, 1, 2)
                            if h > 1.:
                                h = 1.
                            mtqt_source = MTQTSource(v=v, w=w, kappa=kappa, sigma=sigma, h=h)
                        #    M2 = mtqt_source.m6
                        #    m = moment_tensor.MomentTensor(mnn=M2[0], mee=M2[1], mdd=M2[2], mne=M2[3], mnd=M2[4], med=M2[5])
                            m = mtqt_source.pyrocko_moment_tensor()
                            print(m)

                            from pyrocko import plot
                            from pyrocko.plot import beachball
                            fig = plt.figure()
                            axes = fig.add_subplot(1, 1, 1)
                            axes.set_xlim(-2., 4.)
                            axes.set_ylim(-2., 2.)
                            axes.set_axis_off()
                            plot.beachball.plot_beachball_mpl(
                                        m,
                                        axes,
                                        beachball_type='deviatoric',
                                        size=60.,
                                        position=(0, 1),
                                        color_t=plot.mpl_color('scarletred2'),
                                        linewidth=1.0)
                            plt.show()
                            pred_ms.append(m)
                            f = open("maxvals", 'rb')
                            depths_max, depths_min, lats_max, lats_min, lons_max, lons_min = pickle.load(f)
                            f.close()
                            print(v, w)
                            gmt = lune_plot(            # this explodes for large sample sizes ...
                                v_tape=num.asarray([v, v+0.1]),
                                w_tape=num.asarray([w, w+0.1]))


                            gmt.save("lune.pdf", resolution=300, size=5000)
                        #    lat = (-p[6]*lats_min)+(p[6]*lats_max)+lats_min
                        #    lon = (-p[7]*lons_min)+(p[7]*lons_max)+lons_min
                        #    depth = (-p[8]*depths_min)+(p[8]*depths_max)+depths_min
                        #    print(lat,lon, depth)
                        #    print(kill)
                        #real_values_pred.append([lat, lon, depth])

                    #diff_values.append([real_values[i][0]-real_values_pred[i][0], real_values[i][1]-real_values_pred[i][1], real_values[i][2]-real_values_pred[i][2]])
            print(len(pred_ms), len(real_ms))
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
                for i, vals in enumerate(y_val[0::20]):
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
                for i, vals in enumerate(pred[0::20]):
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
            if data_dir is None:
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
                    if data_dir is None:
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
                    print("here")
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
    if detector_only is True:
        model.save('model_detector')
    elif mechanism is True:
        model.save('model_mechanism.tf')
    #    tf.keras.models.save_model(model, 'model_mechanism.tf')
#        tf.saved_model.save(model, 'model_mechanism.tf')
        #file = h5py.File('models/{}.h5'.format(model_mechanism), 'w')
        #weight = model.get_weights()
        #for i in range(len(weight)):
         #  file.create_dataset('weight' + str(i), data=weight[i])
        #file.close()
    else:
        model.save('model_locator')

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


def plot_grid_writeout_sparrow(lats, lons, depths):
    events = []
    for lat in lats:
        for lon in lons:
            for depth in depths:
                events.append(model.event.Event(lat=lat, lon=lon, depth=depth))
    model.dump_events(events, "event_grid.pf")

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
