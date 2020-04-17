import numpy as num
from pyrocko import io
import matplotlib.pyplot as plt
from pyrocko import trace, model, util
from pyrocko.cake_plot import str_to_mpl_color as scolor
from pyrocko.cake_plot import light
from pyrocko import plot
fontsize = 9
ntickmarks_max = 2
label_pad = 25
km = 1000.


def str_dist(dist):
    """
    Return string representation of distance.
    """
    if dist < 10.0:
        return '%g m' % dist
    elif 10. <= dist < 1. * km:
        return '%.0f m' % dist
    elif 1. * km <= dist < 10. * km:
        return '%.1f km' % (dist / km)
    else:
        return '%.0f km' % (dist / km)


def str_duration(t):
    """
    Convert time to str representation.
    """
    s = ''
    if t < 0.:
        s = '-'

    t = abs(t)

    if t < 10.0:
        return s + '%.2g s' % t
    elif 10.0 <= t < 3600.:
        return s + util.time_to_str(t, format='%M:%S min')
    elif 3600. <= t < 24 * 3600.:
        return s + util.time_to_str(t, format='%H:%M h')
    else:
        return s + '%.1f d' % (t / (24. * 3600.))


def plot_taper(axes, t, taper, mode='geometry', **kwargs):
    y = num.ones(t.size) * 0.9
    taper(y, t[0], t[1] - t[0])
    y2 = num.concatenate((y, -y[::-1]))
    t2 = num.concatenate((t, t[::-1]))
    axes.fill(t2, y2, **kwargs)


def plot_trace(axes, tr, **kwargs):
    return axes.plot(tr.get_xdata(), tr.get_ydata(), **kwargs)


def plot_cc(axes, tr, space, mi, ma, **kwargs):
    t = tr.get_xdata()
    y = tr.get_ydata()
    t = num.linspace(tr.tmin, tr.tmax, len(y))
#    mi = num.min(y)
#    ma = num.max(y)
    y2 = (num.concatenate((y, num.zeros(y.size))) - mi) / \
        (ma - mi) * space - (1.0 + space)
    t2 = num.concatenate((t, t[::-1]))
    axes.fill(
        t2, y2,
        clip_on=False,
        **kwargs)


def numpy_correlate_fixed(a, b, mode='valid', use_fft=False):
    '''
    Call :py:func:`numpy.correlate` with fixes.
        c[k] = sum_i a[i+k] * conj(b[i])
    Note that the result produced by newer numpy.correlate is always flipped
    with respect to the formula given in its documentation (if ascending k
    assumed for the output).
    '''

    if use_fft:
        if a.size < b.size:
            c = signal.fftconvolve(b[::-1], a, mode=mode)
        else:
            c = signal.fftconvolve(a, b[::-1], mode=mode)
        return c

    else:
        buggy = numpy_has_correlate_flip_bug()

        a = num.asarray(a)
        b = num.asarray(b)

        if buggy:
            b = num.conj(b)

        c = num.correlate(a, b, mode=mode)

        if buggy and a.size < b.size:
            return c[::-1]
        else:
            return c


def plot_dtrace(axes, tr, space, mi, ma, **kwargs):
    t = tr.get_xdata()
    y = tr.get_ydata()
    spec, ff = tr.spectrum()
    y = ff
    t = num.linspace(tr.tmin, tr.tmax, len(y))
    mi = num.min(y)
    ma = num.max(y)

    y2 = (num.concatenate((y, num.zeros(y.size))) - mi) / \
        (ma - mi) * space - (1.0 + space)
    t2 = num.concatenate((t, t[::-1]))
    axes.fill(
        t2, y2,
        clip_on=False,
        **kwargs)


def plot_waveforms_raw(traces, savedir):
    fig = plt.figure()

    tap_color_annot = (0.35, 0.35, 0.25)
    tap_color_edge = (0.85, 0.85, 0.80)
    waveform_color = scolor('aluminium5')
    misfit_color = scolor('scarletred1')
    for i, tr in enumerate(traces):
        comp = tr.channel
        dtrace = tr

        axes2 = fig.add_subplot(len(traces)/6, len(traces)/6, i+1)

        space = 0.5
        space_factor = 1.0 + space
        axes2.set_axis_off()
        axes2.set_ylim(-1.05 * space_factor, 1.05)

        axes = axes2.twinx()
        axes.set_axis_off()
        plot_trace(
            axes, dtrace,
            color=waveform_color, lw=0.5, zorder=5)
    fig.savefig(savedir+"waveforms.png")
#    plt.show()
    plt.close()
    return fig


def plot_waveforms(traces, event, stations, savedir, show=True):
    fig = plt.figure(figsize=plot.mpl_papersize('a4', 'landscape'))

    tap_color_annot = (0.35, 0.35, 0.25)
    tap_color_edge = (0.85, 0.85, 0.80)
    waveform_color = scolor('aluminium5')
    misfit_color = scolor('scarletred1')
    for i, st in enumerate(stations):
        for tr in traces:
            if tr.station == st.station:
                comp = tr.channel
                tr.downsample_to(0.05)
                tr.highpass(4, 0.01)
                tr.lowpass(4, 0.2)
                dtrace = tr
        target = st

        tmin_fit = dtrace.tmin
        tmax_fit = dtrace.tmax

        tfade_taper = 1./0.2

        taper = trace.CosTaper(
            tmin_fit - 10,
            tmin_fit,
            tmax_fit,
            tmax_fit + 10)

        axes2 = fig.add_subplot(len(stations)/3, len(stations)/3, i+1)

        space = 0.5
        space_factor = 1.0 + space
        axes2.set_axis_off()
        axes2.set_ylim(-1.05 * space_factor, 1.05)

        axes = axes2.twinx()
        axes.set_axis_off()

        bw_filter = trace.ButterworthResponse(
                                              corner=2,
                                              order=4,
                                              type='low')

        setup = trace.MisfitSetup(
            description='setup',
            norm=2,
            taper=taper,
            filter=bw_filter,
            domain='time_domain')

        abs_tr = dtrace.copy()
        abs_tr.set_ydata(abs(dtrace.get_ydata()))

        plot_cc(
            axes2, abs_tr, space, 0., num.max(abs_tr.get_ydata()),
            fc=light(misfit_color, 0.3),
            ec=misfit_color, zorder=4)

        plot_trace(
            axes, dtrace,
            color=waveform_color, lw=0.5, zorder=5)

        tmarks = [
            dtrace.tmin,
            dtrace.tmax]

        for tmark in tmarks:
            axes2.plot(
                [tmark, tmark], [-0.9, 0.1], color=tap_color_annot)

        for tmark, text, ha, va in [
                (tmarks[0],
                 '$\,$ ' + str_duration(tmarks[0]),
                 'left',
                 'bottom'),
                (tmarks[1],
                 '$\Delta$ ' + str_duration(tmarks[1] - tmarks[0]),
                 'right',
                 'bottom')]:
                            axes2.annotate(
                                text,
                                xy=(tmark, -0.9),
                                xycoords='data',
                                xytext=(
                                    fontsize * 0.4 * [-1, 1][ha == 'left'],
                                    fontsize * 0.2),
                                textcoords='offset points',
                                ha=ha,
                                va=va,
                                color=tap_color_annot,
                                fontsize=fontsize, zorder=10)

        infos = []

        infos.append(target.network+"."+target.station+"."+dtrace.channel)
        dist = event.distance_to(target)
        azi = event.azibazi_to(target)[0]
        infos.append(str_dist(dist))
        infos.append(u'%.0f\u00B0' % azi)

        axes2.annotate(
            '\n'.join(infos),
            xy=(0., 1.),
            xycoords='axes fraction',
            xytext=(2., 2.),
            textcoords='offset points',
            ha='left',
            va='top',
            fontsize=fontsize,
            fontstyle='normal')
    fig.savefig(savedir+"waveforms.png")
    if show is True:
        plt.show()
    else:
        plt.close()
