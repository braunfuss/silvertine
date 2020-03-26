import time
import logging
import shutil
import os

from silvertine.config import read_config, write_config
from silvertine import meta, run_info

op = os.path

logger = logging.getLogger('silvertine.environment')


class silvertineEnvironmentError(meta.silvertineError):
    pass


class EventSelectionFailed(silvertineEnvironmentError):
    pass


class NoCurrentEventAvailable(silvertineEnvironmentError):
    def __init__(self, message='no current event available'):
        silvertineEnvironmentError.__init__(self, message)


class NoEventSelectionAvailable(silvertineEnvironmentError):
    def __init__(self, message='no event selection available'):
        silvertineEnvironmentError.__init__(self, message)


class NoRundirAvailable(silvertineEnvironmentError):
    def __init__(self, message='no rundir available'):
        silvertineEnvironmentError.__init__(self, message)


class NoPlotCollectionManagerAvailable(silvertineEnvironmentError):
    def __init__(self, message='no plot collection manager available'):
        silvertineEnvironmentError.__init__(self, message)


class Environment(object):

    def __init__(self, args=None, config=None, event_names=None):

        self._current_event_name = None
        self._selected_event_names = None
        self._config = None
        self._plot_collection_manager = None

        if isinstance(args, str):
            args = [args]

        if not args and not config:
            raise silvertineEnvironmentError('missing arguments')

        if config and event_names:
            self._config_path = None
            self._rundir_path = None
            self._config = config

            if isinstance(event_names, str):
                event_names = [event_names]
            self.set_selected_event_names(event_names)

        elif op.isdir(args[0]):
            self._rundir_path = args[0]
            self._config_path = op.join(self._rundir_path, 'config.yaml')

        else:
            self._rundir_path = None
            self._config_path = args[0]
            self.set_selected_event_names(args[1:])

        self.reset()

    @classmethod
    def discover(cls, rundir, wait=20.):
        start_watch = time.time()
        while (time.time() - start_watch) < wait:
            try:
                cls.verify_rundir(rundir)
                return cls([rundir])
            except silvertineEnvironmentError:
                time.sleep(.25)

    @staticmethod
    def verify_rundir(rundir_path):
        files = [
            'config.yaml',
            'problem.yaml',
            'optimiser.yaml',
            'misfits'
            ]
        for fn in files:
            if not op.exists(op.join(rundir_path, fn)):
                raise silvertineEnvironmentError('inconsistent rundir')

    def copy(self, destination, force=False):
        ''' Copy the environment and return it '''
        files = [
            'config.yaml',
            'problem.yaml',
            'optimiser.yaml',
            'misfits',
            'models',
            'choices',
            'chains'
            ]

        if op.exists(destination) and not force:
            raise OSError('Directory %s already exists' % destination)

        destination = op.abspath(destination)
        os.makedirs(destination, exist_ok=True)

        for file in files:
            src = op.join(self._rundir_path, file)
            dest = op.join(destination, file)

            if not op.isfile(src):
                logger.debug('Cannot find file %s', src)
                continue
            logger.debug('Copying %s to %s', src, dest)

            shutil.copy(src, dest)

        cls = self.__class__
        return cls(destination)

    def reset(self):
        self._histories = {}
        self._dataset = None
        self._optimiser = None
        self._problem = None

    def get_config(self):
        if self._config is None:
            self._config = read_config(self._config_path)

        return self._config

    def write_config(self):
        write_config(self.get_config(), self.get_config_path())

    def get_available_event_names(self):
        return self.get_config().get_event_names()

    def set_current_event_name(self, event_name):
        self._current_event_name = event_name
        self.reset()

    def get_current_event_name(self):
        if self._current_event_name is None:
            try:
                self.get_rundir_path()
                self._current_event_name = self.get_rundir_path()
            except NoRundirAvailable:
                try:
                    event_names = self.get_selected_event_names()
                    if len(event_names) == 1:
                        self._current_event_name = event_names[0]
                    else:
                        raise NoCurrentEventAvailable()

                except NoEventSelectionAvailable:
                    raise NoCurrentEventAvailable()

        return self._current_event_name

    def set_selected_event_names(self, args):
        event_names = self.get_available_event_names()
        if len(args) == 0:
            if len(event_names) == 1:
                self._selected_event_names = event_names
            else:
                if not event_names:
                    raise EventSelectionFailed(
                        'No event file found, check your config!')
                raise EventSelectionFailed(
                    'Ambiguous event selection. Select from available events:'
                    '\n    %s\n  or \'all\' to use all available events'
                    % '\n    '.join(event_names))

        elif len(args) == 1 and args[0] == 'all':
            self._selected_event_names = event_names

        else:
            self._selected_event_names = []
            for event_name in args:
                if event_name not in event_names:
                    self._selected_event_names = None
                    raise EventSelectionFailed(
                        'No such event: %s' % event_name)

                self._selected_event_names.append(event_name)

    @property
    def nevents_selected(self):
        return len(self.get_selected_event_names())

    def get_selected_event_names(self):
        if self._selected_event_names is None:
            raise NoEventSelectionAvailable()

        return self._selected_event_names

    def get_dataset(self):
        if self._dataset is None:
            event_name = self.get_current_event_name()
            self._dataset = self.get_config().get_dataset(event_name)

        return self._dataset

    def set_rundir_path(self, path):
        self._rundir_path = path

    def get_rundir_path(self):
        if self._rundir_path is None:
            raise NoRundirAvailable()

        return self._rundir_path

    def have_rundir(self):
        return self._rundir_path is not None

    def get_run_info_path(self):
        return op.join(self.get_rundir_path(), 'run_info.yaml')

    def get_run_info(self):
        run_info_path = self.get_run_info_path()
        if not op.exists(run_info_path):
            info = run_info.RunInfo()
            return info
        else:
            return run_info.read_info(run_info_path)

    def set_run_info(self, info):
        run_info_path = self.get_run_info_path()
        run_info.write_info(info, run_info_path)

    def get_optimiser(self):
        if self._optimiser is None:
            try:
                self._optimiser = load_optimiser_info(self.get_rundir_path())
            except NoRundirAvailable:
                self._optimiser = \
                    self.get_config().optimiser_config.get_optimiser()

        return self._optimiser

    def get_problem(self):
        if self._problem is None:
            try:
                self._problem = load_problem_info(self.get_rundir_path())
            except NoRundirAvailable:
                self._problem = \
                    self.get_config().get_problem(
                        self.get_dataset().get_event())

        return self._problem

    def get_history(self, subset=None):
        if subset not in self._histories:
            self._histories[subset] = \
                ModelHistory(
                    self.get_problem(),
                    nchains=self.get_optimiser().nchains,
                    path=meta.xjoin(self.get_rundir_path(), subset))

            self._histories[subset].ensure_bootstrap_misfits(
                self.get_optimiser())

        return self._histories[subset]

    def set_plot_collection_manager(self, pcm):
        self._plot_collection_manager = pcm

    def get_plot_collection_manager(self):
        if self._plot_collection_manager is None:
            raise NoPlotCollectionManagerAvailable()

        return self._plot_collection_manager

    def setup_modelling(self):
        '''Must be called before any modelling can be done.'''
        logger.debug('Setting up modelling...')
        self.get_config().setup_modelling_environment(self.get_problem())
        ds = self.get_dataset()
        for target in self.get_problem().targets:
            target.set_dataset(ds)

    def get_plot_classes(self):
        '''Discover all plot classes relevant for the setup.'''

        plots = set()
        plots.update(get_plot_classes_cls())
        return sorted(list(plots), key=lambda plot: plot.name)

    def get_plots_path(self):
        try:
            return op.join(self.get_rundir_path(), 'plots')
        except NoRundirAvailable:
            return 'plots'

    def get_config_path(self):
        return self._config_path


import math
import logging
import random

import numpy as num
from matplotlib import cm, patches, colors as mcolors

from pyrocko.guts import Tuple, Float, Int, String, List, Bool, StringChoice

from pyrocko.plot import mpl_margins, mpl_color, mpl_init, mpl_graph_color
from pyrocko.plot import beachball, hudson

from grond.plot.config import PlotConfig
from grond.plot.section import SectionPlotConfig, SectionPlot
from grond.plot.collection import PlotItem
from grond import meta, core, stats
from matplotlib import pyplot as plt

logger = logging.getLogger('grond.problem.plot')

guts_prefix = 'grond'


def cluster_label(icluster, perc):
    if icluster == -1:
        return 'Unclust. (%.0f%%)' % perc
    else:
        return 'Cluster %i (%.0f%%)' % (icluster, perc)


def cluster_color(icluster):
    if icluster == -1:
        return mpl_color('aluminium3')
    else:
        return mpl_graph_color(icluster)


def fixlim(lo, hi):
    if lo == hi:
        return lo - 1.0, hi + 1.0
    else:
        return lo, hi


def eigh_sorted(mat):
    evals, evecs = num.linalg.eigh(mat)
    iorder = num.argsort(evals)
    return evals[iorder], evecs[:, iorder]


class JointparPlot(PlotConfig):
    '''
    Source problem parameter's tradeoff plots.
    '''

    name = 'jointpar'
    size_cm = Tuple.T(2, Float.T(), default=(20., 20.))
    misfit_cutoff = Float.T(optional=True)
    ibootstrap = Int.T(optional=True)
    color_parameter = String.T(default='misfit')
    exclude = List.T(String.T())
    include = List.T(String.T())
    show_ellipses = Bool.T(default=False)
    nsubplots = Int.T(default=6)
    show_ticks = Bool.T(default=False)
    show_reference = Bool.T(default=True)

    def make(self, environ):
        cm = environ.get_plot_collection_manager()
        history = environ.get_history(subset='harvest')
        optimiser = environ.get_optimiser()
        sref = 'Dark gray boxes mark reference solution.' \
            if self.show_reference else ''

        mpl_init(fontsize=self.font_size)
        cm.create_group_mpl(
            self,
            self.draw_figures(history, optimiser),
            title=u'Jointpar Plot',
            section='solution',
            feather_icon='crosshair',
            description=u'''
Source problem parameter's scatter plots, to evaluate the resolution of source
parameters and possible trade-offs between pairs of model parameters.

A subset of model solutions (from harvest) is shown in two dimensions for all
possible parameter pairs as points. The point color indicates the misfit for
the model solution with cold colors (blue) for high misfit models and warm
colors (red) for low misfit models. The plot ranges are defined by the given
parameter bounds and shows the model space of the optimsation. %s''' % sref)

    def draw_figures(self, history, optimiser):

        color_parameter = self.color_parameter
        exclude = self.exclude
        include = self.include
        nsubplots = self.nsubplots
        figsize = self.size_inch
        ibootstrap = 0 if self.ibootstrap is None else self.ibootstrap
        misfit_cutoff = self.misfit_cutoff
        show_ellipses = self.show_ellipses
        msize = 1.5
        cmap = 'coolwarm'

        problem = history.problem
        if not problem:
            return []

        models = history.models

        exclude = list(exclude)
        bounds = problem.get_combined_bounds()
        for ipar in range(problem.ncombined):
            par = problem.combined[ipar]
            lo, hi = bounds[ipar]
            if lo == hi:
                exclude.append(par.name)

        xref = problem.get_reference_model()

        isort = history.get_sorted_misfits_idx(chain=ibootstrap)[::-1]
        models = history.get_sorted_models(chain=ibootstrap)[::-1]
        nmodels = history.nmodels

        gms = history.get_sorted_misfits(chain=ibootstrap)[::-1]
        if misfit_cutoff is not None:
            ibest = gms < misfit_cutoff
            gms = gms[ibest]
            models = models[ibest]

        kwargs = {}

        if color_parameter == 'dist':
            mx = num.mean(models, axis=0)
            cov = num.cov(models.T)
            mdists = core.mahalanobis_distance(models, mx, cov)
            icolor = meta.ordersort(mdists)

        elif color_parameter == 'misfit':
            iorder = num.arange(nmodels)
            icolor = iorder

        elif color_parameter in problem.parameter_names:
            ind = problem.name_to_index(color_parameter)
            icolor = problem.extract(models, ind)

        elif color_parameter in history.attribute_names:
            icolor = history.get_attribute(color_parameter)[isort]
            icolor_need = num.unique(icolor)

            colors = []
            for i in range(icolor_need[-1]+1):
                colors.append(mpl_graph_color(i))

            cmap = mcolors.ListedColormap(colors)
            cmap.set_under(mpl_color('aluminium3'))
            kwargs.update(dict(vmin=0, vmax=icolor_need[-1]))
        else:
            raise meta.GrondError(
                'Invalid color_parameter: %s' % color_parameter)

        smap = {}
        iselected = 0
        for ipar in range(problem.ncombined):
            par = problem.combined[ipar]
            if exclude and par.name in exclude or \
                    include and par.name not in include:
                continue

            smap[iselected] = ipar
            iselected += 1

        nselected = iselected

        if nselected < 2:
            logger.warn('Cannot draw joinpar figures with less than two '
                        'parameters selected.')
            return []

        nfig = (nselected - 2) // nsubplots + 1

        figs = []
        for ifig in range(nfig):
            figs_row = []
            for jfig in range(nfig):
                if ifig >= jfig:
                    item = PlotItem(name='fig_%i_%i' % (ifig, jfig))
                    item.attributes['parameters'] = []
                    figs_row.append((item, plt.figure(figsize=figsize)))
                else:
                    figs_row.append(None)

            figs.append(figs_row)

        for iselected in range(nselected):
            ipar = smap[iselected]
            ypar = problem.combined[ipar]
            for jselected in range(iselected):
                jpar = smap[jselected]
                xpar = problem.combined[jpar]

                ixg = (iselected - 1)
                iyg = jselected

                ix = ixg % nsubplots
                iy = iyg % nsubplots

                ifig = ixg // nsubplots
                jfig = iyg // nsubplots

                aind = (nsubplots, nsubplots, (ix * nsubplots) + iy + 1)

                item, fig = figs[ifig][jfig]

                tlist = item.attributes['parameters']
                if xpar.name not in tlist:
                    tlist.append(xpar.name)
                if ypar.name not in tlist:
                    tlist.append(ypar.name)

                axes = fig.add_subplot(*aind)

                axes.axvline(0., color=mpl_color('aluminium3'), lw=0.5)
                axes.axhline(0., color=mpl_color('aluminium3'), lw=0.5)
                for spine in axes.spines.values():
                    spine.set_edgecolor(mpl_color('aluminium5'))
                    spine.set_linewidth(0.5)

                xmin, xmax = fixlim(*xpar.scaled(bounds[jpar]))
                ymin, ymax = fixlim(*ypar.scaled(bounds[ipar]))

                if ix == 0 or jselected + 1 == iselected:
                    for (xpos, xoff, x) in [
                            (0.0, 10., xmin),
                            (1.0, -10., xmax)]:

                        axes.annotate(
                            '%.3g%s' % (x, xpar.get_unit_suffix()),
                            xy=(xpos, 1.05),
                            xycoords='axes fraction',
                            xytext=(xoff, 5.),
                            textcoords='offset points',
                            verticalalignment='bottom',
                            horizontalalignment='left',
                            rotation=45.)

                if iy == nsubplots - 1 or jselected + 1 == iselected:
                    for (ypos, yoff, y) in [
                            (0., 10., ymin),
                            (1.0, -10., ymax)]:

                        axes.annotate(
                            '%.3g%s' % (y, ypar.get_unit_suffix()),
                            xy=(1.0, ypos),
                            xycoords='axes fraction',
                            xytext=(5., yoff),
                            textcoords='offset points',
                            verticalalignment='bottom',
                            horizontalalignment='left',
                            rotation=45.)

                axes.set_xlim(xmin, xmax)
                axes.set_ylim(ymin, ymax)

                if not self.show_ticks:
                    axes.get_xaxis().set_ticks([])
                    axes.get_yaxis().set_ticks([])
                else:
                    axes.tick_params(length=4, which='both')
                    axes.get_yaxis().set_ticklabels([])
                    axes.get_xaxis().set_ticklabels([])

                if iselected == nselected - 1 or ix == nsubplots - 1:
                    axes.annotate(
                        xpar.get_label(with_unit=False),
                        xy=(0.5, -0.05),
                        xycoords='axes fraction',
                        verticalalignment='top',
                        horizontalalignment='right',
                        rotation=45.)

                if iy == 0:
                    axes.annotate(
                        ypar.get_label(with_unit=False),
                        xy=(-0.05, 0.5),
                        xycoords='axes fraction',
                        verticalalignment='top',
                        horizontalalignment='right',
                        rotation=45.)

                fx = problem.extract(models, jpar)
                fy = problem.extract(models, ipar)

                axes.scatter(
                    xpar.scaled(fx),
                    ypar.scaled(fy),
                    c=icolor,
                    s=msize, alpha=0.5, cmap=cmap, edgecolors='none', **kwargs)

                if show_ellipses:
                    cov = num.cov((xpar.scaled(fx), ypar.scaled(fy)))
                    evals, evecs = eigh_sorted(cov)
                    evals = num.sqrt(evals)
                    ell = patches.Ellipse(
                        xy=(
                            num.mean(xpar.scaled(fx)),
                            num.mean(ypar.scaled(fy))),
                        width=evals[0] * 2,
                        height=evals[1] * 2,
                        angle=num.rad2deg(
                            num.arctan2(evecs[1][0], evecs[0][0])))

                    ell.set_facecolor('none')
                    axes.add_artist(ell)

                if self.show_reference:
                    fx = problem.extract(xref, jpar)
                    fy = problem.extract(xref, ipar)

                    ref_color = mpl_color('aluminium6')
                    ref_color_light = 'none'
                    axes.plot(
                        xpar.scaled(fx), ypar.scaled(fy), 's',
                        mew=1.5, ms=5, mfc=ref_color_light, mec=ref_color)

        figs_flat = []
        for figs_row in figs:
            figs_flat.extend(
                item_fig for item_fig in figs_row if item_fig is not None)

        return figs_flat


class HistogramPlot(PlotConfig):
    '''
    Histograms or Gaussian kernel densities (default) of all parameters
    (marginal distributions of model parameters).

    The histograms (by default shown as Gaussian kernel densities) show (red
    curved solid line) the distributions of the parameters (marginals) along
    with some characteristics: The red solid vertical line gives the median of
    the distribution and the dashed red vertical line the mean value. Dark gray
    vertical lines show reference values (given in the event.txt file). The
    overlapping red-shaded areas show the 68% confidence intervals (innermost
    area), the 90% confidence intervals (middle area) and the minimum and
    maximum values (widest area). The plot ranges are defined by the given
    parameter bounds and show the model space. Well resolved model parameters
    show peaked distributions.
    '''

    name = 'histogram'
    size_cm = Tuple.T(2, Float.T(), default=(12.5, 7.5))
    exclude = List.T(String.T())
    include = List.T(String.T())
    method = StringChoice.T(
        choices=['gaussian_kde', 'histogram'],
        default='gaussian_kde')
    show_reference = Bool.T(default=True)

    def make(self, environ):
        cm = environ.get_plot_collection_manager()
        history = environ.get_history(subset='harvest')

        mpl_init(fontsize=self.font_size)
        cm.create_group_mpl(
            self,
            self.draw_figures(history),
            title=u'Histogram',
            section='solution',
            feather_icon='bar-chart-2',
            description=u'''
Distribution of the problem's parameters.

The histograms are shown either as Gaussian kernel densities (red curved solid
line) or as bar plots the distributions of the parameters (marginals) along
with some characteristics:

The red solid vertical line gives the median of the distribution and the dashed
red vertical line the mean value. Dark gray vertical lines show reference
parameter values if given in the event.txt file. The overlapping red-shaded
areas show the 68% confidence intervals (innermost area), the 90% confidence
intervals (middle area) and the minimum and maximum values (widest area). The
plot ranges are defined by the given parameter bounds and show the model
space.
''')

    def draw_figures(self, history):

        import scipy.stats
        from grond.core import make_stats

        exclude = self.exclude
        include = self.include
        figsize = self.size_inch
        fontsize = self.font_size
        method = self.method
        ref_color = mpl_color('aluminium6')
        stats_color = mpl_color('scarletred2')
        bar_color = mpl_color('scarletred1')
        stats_color3 = mpl_color('scarletred3')

        problem = history.problem

        models = history.models

        bounds = problem.get_combined_bounds()
        exclude = list(exclude)
        for ipar in range(problem.ncombined):
            par = problem.combined[ipar]
            vmin, vmax = bounds[ipar]
            if vmin == vmax:
                exclude.append(par.name)

        xref = problem.get_reference_model()

        smap = {}
        iselected = 0
        for ipar in range(problem.ncombined):
            par = problem.combined[ipar]
            if exclude and par.name in exclude or \
                    include and par.name not in include:
                continue

            smap[iselected] = ipar
            iselected += 1

        nselected = iselected
        del iselected

        pnames = [
            problem.combined[smap[iselected]].name
            for iselected in range(nselected)]

        rstats = make_stats(problem, models,
                            history.get_primary_chain_misfits(),
                            pnames=pnames)

        for iselected in range(nselected):
            ipar = smap[iselected]
            par = problem.combined[ipar]
            vs = problem.extract(models, ipar)
            vmin, vmax = bounds[ipar]

            fig = plt.figure(figsize=figsize)
            labelpos = mpl_margins(
                fig, nw=1, nh=1, w=7., bottom=5., top=1, units=fontsize)

            axes = fig.add_subplot(1, 1, 1)
            labelpos(axes, 2.5, 2.0)
            axes.set_xlabel(par.get_label())
            axes.set_ylabel('PDF')
            axes.set_xlim(*fixlim(*par.scaled((vmin, vmax))))

            if method == 'gaussian_kde':
                try:
                    kde = scipy.stats.gaussian_kde(vs)
                except Exception:
                    logger.warn(
                        'Cannot create plot histogram with gaussian_kde: '
                        'possibly all samples have the same value.')
                    continue

                vps = num.linspace(vmin, vmax, 600)
                pps = kde(vps)

                axes.plot(
                    par.scaled(vps), par.inv_scaled(pps), color=stats_color)

            elif method == 'histogram':
                pps, edges = num.histogram(
                    vs,
                    bins=num.linspace(vmin, vmax, num=40),
                    density=True)
                vps = 0.5 * (edges[:-1] + edges[1:])

                axes.bar(par.scaled(vps), par.inv_scaled(pps),
                         par.scaled(2.*(vps - edges[:-1])),
                         color=bar_color)

            pstats = rstats.parameter_stats_list[iselected]

            axes.axvspan(
                par.scaled(pstats.minimum),
                par.scaled(pstats.maximum),
                color=stats_color, alpha=0.1)
            axes.axvspan(
                par.scaled(pstats.percentile16),
                par.scaled(pstats.percentile84),
                color=stats_color, alpha=0.1)
            axes.axvspan(
                par.scaled(pstats.percentile5),
                par.scaled(pstats.percentile95),
                color=stats_color, alpha=0.1)

            axes.axvline(
                par.scaled(pstats.median),
                color=stats_color3, alpha=0.5)
            axes.axvline(
                par.scaled(pstats.mean),
                color=stats_color3, ls=':', alpha=0.5)

            if self.show_reference:
                axes.axvline(
                    par.scaled(problem.extract(xref, ipar)),
                    color=ref_color)

            item = PlotItem(name=par.name)
            item.attributes['parameters'] = [par.name]
            yield item, fig


class MTDecompositionPlot(PlotConfig):
    '''
    Moment tensor decomposition plot.
    '''

    name = 'mt_decomposition'
    size_cm = Tuple.T(2, Float.T(), default=(15., 5.))
    cluster_attribute = meta.StringID.T(
        optional=True,
        help='name of attribute to use as cluster IDs')
    show_reference = Bool.T(default=True)

    def make(self, environ):
        cm = environ.get_plot_collection_manager()
        history = environ.get_history(subset='harvest')
        mpl_init(fontsize=self.font_size)
        cm.create_group_mpl(
            self,
            self.draw_figures(history),
            title=u'MT Decomposition',
            section='solution',
            feather_icon='sun',
            description=u'''
Moment tensor decomposition of the best-fitting solution into isotropic,
deviatoric and best double couple components.

Shown are the ensemble best, the ensemble mean%s and, if available, a reference
mechanism. The symbol size indicates the relative strength of the components.
The inversion result is consistent and stable if ensemble mean and ensemble
best have similar symbol size and patterns.
''' % (', cluster results' if self.cluster_attribute else ''))

    def draw_figures(self, history):

        fontsize = self.font_size

        fig = plt.figure(figsize=self.size_inch)
        axes = fig.add_subplot(1, 1, 1, aspect=1.0)
        fig.subplots_adjust(left=0., right=1., bottom=0., top=1.)

        problem = history.problem
        models = history.models

        if models.size == 0:
            logger.warn('Empty models vector.')
            return []

        # gms = problem.combine_misfits(history.misfits)
        # isort = num.argsort(gms)
        # iorder = num.empty_like(isort)
        # iorder[isort] = num.arange(iorder.size)[::-1]

        ref_source = problem.base_source

        mean_source = stats.get_mean_source(
            problem, history.models)

        best_source = history.get_best_source()

        nlines_max = int(round(self.size_cm[1] / 5. * 4. - 1.0))

        if self.cluster_attribute:
            cluster_sources = history.mean_sources_by_cluster(
                self.cluster_attribute)
        else:
            cluster_sources = []

        def get_deco(source):
            mt = source.pyrocko_moment_tensor()
            return mt.standard_decomposition()

        lines = []
        lines.append(
            ('Ensemble best', get_deco(best_source), mpl_color('aluminium5')))

        lines.append(
            ('Ensemble mean', get_deco(mean_source), mpl_color('aluminium5')))

        for (icluster, perc, source) in cluster_sources:
            if len(lines) < nlines_max - int(self.show_reference):
                lines.append(
                    (cluster_label(icluster, perc),
                     get_deco(source),
                     cluster_color(icluster)))
            else:
                logger.warn(
                    'Skipping display of cluster %i because figure height is '
                    'too small. Figure height should be at least %g cm.' % (
                        icluster, (3 + len(cluster_sources)
                                   + int(self.show_reference)) * 5/4.))

        if self.show_reference:
            lines.append(
                ('Reference', get_deco(ref_source), mpl_color('aluminium3')))

        moment_full_max = max(deco[-1][0] for (_, deco, _) in lines)

        for xpos, label in [
                (0., 'Full'),
                (2., 'Isotropic'),
                (4., 'Deviatoric'),
                (6., 'CLVD'),
                (8., 'DC')]:

            axes.annotate(
                label,
                xy=(1 + xpos, nlines_max),
                xycoords='data',
                xytext=(0., 0.),
                textcoords='offset points',
                ha='center',
                va='center',
                color='black',
                fontsize=fontsize)

        for i, (label, deco, color_t) in enumerate(lines):
            ypos = nlines_max - i - 1.0

            [(moment_iso, ratio_iso, m_iso),
             (moment_dc, ratio_dc, m_dc),
             (moment_clvd, ratio_clvd, m_clvd),
             (moment_devi, ratio_devi, m_devi),
             (moment_full, ratio_full, m_full)] = deco

            size0 = moment_full / moment_full_max

            axes.annotate(
                label,
                xy=(-2., ypos),
                xycoords='data',
                xytext=(0., 0.),
                textcoords='offset points',
                ha='left',
                va='center',
                color='black',
                fontsize=fontsize)

            for xpos, mt_part, ratio, ops in [
                    (0., m_full, ratio_full, '-'),
                    (2., m_iso, ratio_iso, '='),
                    (4., m_devi, ratio_devi, '='),
                    (6., m_clvd, ratio_clvd, '+'),
                    (8., m_dc, ratio_dc, None)]:

                if ratio > 1e-4:
                    try:
                        beachball.plot_beachball_mpl(
                            mt_part, axes,
                            beachball_type='full',
                            position=(1. + xpos, ypos),
                            size=0.9 * size0 * math.sqrt(ratio),
                            size_units='data',
                            color_t=color_t,
                            linewidth=1.0)

                    except beachball.BeachballError as e:
                        logger.warn(str(e))

                        axes.annotate(
                            'ERROR',
                            xy=(1. + xpos, ypos),
                            ha='center',
                            va='center',
                            color='red',
                            fontsize=fontsize)

                else:
                    axes.annotate(
                        'N/A',
                        xy=(1. + xpos, ypos),
                        ha='center',
                        va='center',
                        color='black',
                        fontsize=fontsize)

                if ops is not None:
                    axes.annotate(
                        ops,
                        xy=(2. + xpos, ypos),
                        ha='center',
                        va='center',
                        color='black',
                        fontsize=fontsize)

        axes.axison = False
        axes.set_xlim(-2.25, 9.75)
        axes.set_ylim(-0.5, nlines_max+0.5)

        item = PlotItem(name='main')
        return [[item, fig]]


class MTLocationPlot(SectionPlotConfig):
    ''' MT location plot of the best solutions in three cross-sections. '''
    name = 'location_mt'
    beachball_type = StringChoice.T(
        choices=['full', 'deviatoric', 'dc'],
        default='dc')
    normalisation_gamma = Float.T(
        default=3.,
        help='Normalisation of colors and alpha as :math:`x^\gamma`.'
             'A linear colormap/alpha with :math:`\gamma=1`.')

    def make(self, environ):
        environ.setup_modelling()
        cm = environ.get_plot_collection_manager()
        history = environ.get_history(subset='harvest')
        mpl_init(fontsize=self.font_size)
        self._to_be_closed = []
        cm.create_group_mpl(
            self,
            self.draw_figures(history),
            title=u'MT Location',
            section='solution',
            feather_icon='target',
            description=u'''
Location plot of the ensemble of best solutions in three cross-sections.

The coordinate range is defined by the search space given in the config file.
Symbols show best double-couple mechanisms, and colors indicate low (red) and
high (blue) misfit.
''')
        for obj in self._to_be_closed:
            obj.close()

    def draw_figures(self, history, color_p_axis=False):
        from matplotlib import colors

        color = 'black'
        fontsize = self.font_size
        markersize = fontsize * 1.5
        beachballsize_small = markersize * 0.5
        beachball_type = self.beachball_type

        problem = history.problem
        sp = SectionPlot(config=self)
        self._to_be_closed.append(sp)

        fig = sp.fig
        axes_en = sp.axes_xy
        axes_dn = sp.axes_zy
        axes_ed = sp.axes_xz

        bounds = problem.get_combined_bounds()

        models = history.get_sorted_primary_models()[::-1]

        iorder = num.arange(history.nmodels)

        for parname, set_label, set_lim in [
                ['east_shift', sp.set_xlabel, sp.set_xlim],
                ['north_shift', sp.set_ylabel, sp.set_ylim],
                ['depth', sp.set_zlabel, sp.set_zlim]]:

            ipar = problem.name_to_index(parname)
            par = problem.combined[ipar]
            set_label(par.get_label())
            xmin, xmax = fixlim(*par.scaled(bounds[ipar]))
            set_lim(xmin, xmax)

        if 'volume_change' in problem.parameter_names:
            volumes = models[:, problem.name_to_index('volume_change')]
            volume_max = volumes.max()
            volume_min = volumes.min()

        def scale_size(source):
            if not hasattr(source, 'volume_change'):
                return beachballsize_small

            volume_change = source.volume_change
            fac = (volume_change - volume_min) / (volume_max - volume_min)
            return markersize * .25 + markersize * .5 * fac

        for axes, xparname, yparname in [
                (axes_en, 'east_shift', 'north_shift'),
                (axes_dn, 'depth', 'north_shift'),
                (axes_ed, 'east_shift', 'depth')]:

            ixpar = problem.name_to_index(xparname)
            iypar = problem.name_to_index(yparname)

            xpar = problem.combined[ixpar]
            ypar = problem.combined[iypar]

            xmin, xmax = fixlim(*xpar.scaled(bounds[ixpar]))
            ymin, ymax = fixlim(*ypar.scaled(bounds[iypar]))

            try:
                axes.set_facecolor(mpl_color('aluminium1'))
            except AttributeError:
                axes.patch.set_facecolor(mpl_color('aluminium1'))

            rect = patches.Rectangle(
                (xmin, ymin), xmax-xmin, ymax-ymin,
                facecolor=mpl_color('white'),
                edgecolor=mpl_color('aluminium2'))

            axes.add_patch(rect)

            # fxs = xpar.scaled(problem.extract(models, ixpar))
            # fys = ypar.scaled(problem.extract(models, iypar))

            # axes.set_xlim(*fixlim(num.min(fxs), num.max(fxs)))
            # axes.set_ylim(*fixlim(num.min(fys), num.max(fys)))

            cmap = cm.ScalarMappable(
                norm=colors.PowerNorm(
                    gamma=self.normalisation_gamma,
                    vmin=iorder.min(),
                    vmax=iorder.max()),

                cmap=plt.get_cmap('coolwarm'))

            for ix, x in enumerate(models):

                source = problem.get_source(x)
                mt = source.pyrocko_moment_tensor(
                    store=problem.get_gf_store(problem.targets[0]),
                    target=problem.targets[0])
                fx = problem.extract(x, ixpar)
                fy = problem.extract(x, iypar)
                sx, sy = xpar.scaled(fx), ypar.scaled(fy)

                # TODO: Add rotation in cross-sections
                color = cmap.to_rgba(iorder[ix])

                alpha = (iorder[ix] - iorder.min()) / \
                    float(iorder.max() - iorder.min())
                alpha = alpha**self.normalisation_gamma

                try:
                    beachball.plot_beachball_mpl(
                        mt, axes,
                        beachball_type=beachball_type,
                        position=(sx, sy),
                        size=scale_size(source),
                        color_t=color,
                        color_p=color if color_p_axis else 'white',
                        alpha=alpha,
                        zorder=1,
                        linewidth=0.25)

                except beachball.BeachballError as e:
                    logger.warn(str(e))

        item = PlotItem(name='main')
        return [[item, fig]]


class ShakemapPlot(PlotConfig):

    '''
    Illustration of the solution distribution of decomposed moment tensor.
    '''

    name = 'shakemap'
    size_cm = Tuple.T(2, Float.T(), default=(17.5, 17.5*(3./4.)))

    def make(self, environ):
        cm = environ.get_plot_collection_manager()
        mpl_init(fontsize=self.font_size)
        cm.create_group_mpl(
            self,
            self.draw_figures(history),
            title=u'shakemap',
            section='shakemap',
            feather_icon='box',
            description=u'''
Shakemap
''')

    def draw_figures(self):
        item = PlotItem(
            name='gf_shakemap')
        return [[item]]


def get_plot_classes_cls():
    return [
        ShakemapPlot
        ]


__all__ = [
    'silvertineEnvironmentError',
    'EventSelectionFailed',
    'NoCurrentEventAvailable',
    'NoRundirAvailable',
    'NoPlotCollectionManagerAvailable',
    'Environment',
]
