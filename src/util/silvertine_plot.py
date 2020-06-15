from pyrocko.cake_plot import str_to_mpl_color as scolor
from pyrocko.cake_plot import light
from pyrocko.plot import mpl_papersize, mpl_init, mpl_graph_color, mpl_margins
import numpy as num

from matplotlib import pyplot as plt
from matplotlib.patches import Rectangle, FancyArrow
from matplotlib.collections import PatchCollection
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.ticker as tick
from pyrocko import cake_plot as cp


def bayesian_model_plot(models, axes=None, draw_bg=True, highlightidx=[]):
    """
    Plot cake layered earth models.
    """
    fontsize = 10
    if axes is None:
        mpl_init(fontsize=fontsize)
        fig, axes = plt.subplots(
            nrows=1, ncols=1, figsize=mpl_papersize('a6', 'portrait'))
        labelpos = mpl_margins(
            fig, left=6, bottom=4, top=1.5, right=0.5, units=fontsize)
        labelpos(axes, 2., 1.5)

    def plot_profile(mod, axes, vp_c, vs_c, lw=0.5):
        z = mod.profile('z')
        vp = mod.profile('vp')
        vs = mod.profile('vs')
        axes.plot(vp, z, color=vp_c, lw=lw)
        axes.plot(vs, z, color=vs_c, lw=lw)

    cp.labelspace(axes)
    cp.labels_model(axes=axes)
    if draw_bg:
        cp.sketch_model(models[0], axes=axes)
    else:
        axes.spines['right'].set_visible(False)
        axes.spines['top'].set_visible(False)

    ref_vp_c = scolor('aluminium5')
    ref_vs_c = scolor('aluminium5')
    vp_c = scolor('scarletred2')
    vs_c = scolor('skyblue2')

    for i, mod in enumerate(models):
        plot_profile(
            mod, axes, vp_c=light(vp_c, 0.3), vs_c=light(vs_c, 0.3), lw=1.)

    for count, i in enumerate(sorted(highlightidx)):
        if count == 0:
            vpcolor = ref_vp_c
            vscolor = ref_vs_c
        else:
            vpcolor = vp_c
            vscolor = vs_c

        plot_profile(
            models[i], axes, vp_c=vpcolor, vs_c=vscolor, lw=2.)

    ymin, ymax = axes.get_ylim()
    xmin, xmax = axes.get_xlim()
    xmin = 0.
    my = (ymax - ymin) * 0.05
    mx = (xmax - xmin) * 0.2
    axes.set_ylim(ymax, ymin - my)
    axes.set_xlim(xmin, xmax + mx)
    return fig, axes
