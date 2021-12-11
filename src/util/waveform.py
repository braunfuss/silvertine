import numpy as num
from pyrocko import io
import matplotlib.pyplot as plt
from pyrocko import trace, model, util
from pyrocko.cake_plot import str_to_mpl_color as scolor
from pyrocko.cake_plot import light
from pyrocko import plot
from pathlib import Path
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


def plot_waveforms_raw(traces, savedir, iter=None):
    fig = plt.figure()

    tap_color_annot = (0.35, 0.35, 0.25)
    tap_color_edge = (0.85, 0.85, 0.80)
    waveform_color = scolor('aluminium5')
    misfit_color = scolor('scarletred1')
    for i, tr in enumerate(traces):
        comp = tr.channel
        dtrace = tr

        axes2 = fig.add_subplot(len(traces)/8, len(traces)/3, i+1)

        space = 1.5
        space_factor = 1.0 + space
        axes2.set_axis_off()
        axes2.set_ylim(-1.05 * space_factor, 1.05)

        axes = axes2.twinx()
        axes.set_axis_off()
        plot_trace(
            axes, dtrace,
            color=waveform_color, lw=0.5, zorder=5)
    if iter is None:
        fig.savefig(savedir+"waveforms.png")
    else:
        fig.savefig(savedir+"_%s_.png" %iter)
#    plt.show()
    plt.close()
    return fig


def plot_waveforms(traces, event, stations, savedir, picks, show=True):
    fig = plt.figure(figsize=plot.mpl_papersize('a4', 'landscape'))
    tap_color_annot = (0.35, 0.35, 0.25)
    tap_color_edge = (0.85, 0.85, 0.80)
    waveform_color = scolor('aluminium5')
    misfit_color = scolor('scarletred1')
    ncomps = 3
    k = 0
    nstations = len(stations)
    ntraces = nstations*ncomps
    i = 0
    for st in stations:
        for comp in st.channels:
            for tr in traces:
                if tr.station == st.station:
                    if comp.name == tr.channel:
                #    tr.downsample_to(0.05)
                #    tr.highpass(4, 0.01)
                #    tr.lowpass(4, 0.2)
                        dtrace = tr
                        i = i+1
            target = st

            tmin_fit = dtrace.tmin
            tmax_fit = dtrace.tmax

            tfade_taper = 1./0.2

            taper = trace.CosTaper(
                tmin_fit - 20,
                tmin_fit,
                tmax_fit,
                tmax_fit + 30)
            k = k + 1
            axes2 = fig.add_subplot(nstations/3, nstations/3, k)
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

            if picks is not None:
                for stp in picks["phases"]:
                    phases_station = []
                    picks_station = []
                    if st.station == stp["station"]:
                        phases_station.append(str(stp["phase"]))
                        picks_station.append(event.time + float(stp["pick"]))
                        picks_station.append(event.time)

                    tmarks = picks_station

                    for tmark in tmarks:
                        axes2.plot(
                            [tmark, tmark], [-1, 1.], color="blue")

                    for tmark, text, ha, va in [
                            (tmarks,
                             phases_station,
                             'left',
                             'bottom')]:
                                    try:
                                        axes2.annotate(
                                            text[0],
                                            xy=(tmark[0], -1.2),
                                            xycoords='data',
                                            xytext=(
                                                8 * 0.4 * [-1, 1][ha == 'left'],
                                                8 * 0.2),
                                            textcoords='offset points',
                                            ha=ha,
                                            va=va,
                                            color=tap_color_annot,
                                            fontsize=8, zorder=10)
                                    except:
                                        pass
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
            if i/nstations == 1 or i/nstations == 2 or i/nstations ==3:
                fig.savefig(savedir+"waveforms_%s.png" % str(int(i/nstations)), dpi=100)

                if show is True:
                    plt.show()
                else:
                    plt.close()
                fig = plt.figure(figsize=plot.mpl_papersize('a4', 'landscape'))
                k = 0


def load_data_archieve(validation_data, gf_freq, duration=4,
                       wanted_start=None, wanted_end=None):
    folder = validation_data
    pathlist = Path(folder).glob('day*')
    waveforms = []
    stations = []
    if wanted_start is not None:
        try:
            wanted_start = util.stt(wanted_start)
            wanted_end = util.stt(wanted_end)
        except:
            pass

    from pyrocko import pile
    paths = []
    safecon = 0
    for path in sorted(pathlist):
        path = str(path)
        d2 = float(str(path)[-12:])
        d1 = float(str(path)[-25:-13])
        if wanted_start is not None:
            do_safety_files = False
            if (d1 >= wanted_start and d2 <= wanted_end) or (d2-wanted_end<86400. and d2-wanted_end>0. and safecon == 0):
                st = model.load_stations(path+"/waveforms/stations.raw.txt")

                d_diff = d2 - d1
                tr_packages = int(d_diff/duration)
                #for tr in traces:
                #    tr.downsample_to(gf_freq)
        #        if safecon == 0:

                pathlist_waveform_files = Path(path+"/waveforms/rest/").glob('*.mseed')
                wanted_start_str = util.tts(wanted_start)[14:16]
                diff_to_full = float(wanted_start_str)
                max_diff = 55.
                min_diff = 5.
                if diff_to_full > max_diff or diff_to_full < min_diff:
                    do_safety_files = True
                for path_wave in sorted(pathlist_waveform_files):
                    path_wave = str(path_wave)
                    p1 = path_wave[-25:-15]
                    p2 = path_wave[-14:-12]
                    p3 = path_wave[-11:-9]
                    p4 = path_wave[-8:-6]
                    try:
                        file_time = util.stt(p1+" "+p2+":"+p3+":"+p4)
                        tdiff = file_time - wanted_start
                        if do_safety_files is True:
                            if float(p2)-float(util.tts(wanted_start)[11:13]) == 0:
                                paths.append(str(path_wave))
                            if diff_to_full > max_diff and float(p2)-float(util.tts(wanted_start)[11:13]) == 1.:
                                paths.append(str(path_wave))
                            if diff_to_full < min_diff and float(p2)-float(util.tts(wanted_start)[11:13]) == -1.:
                                paths.append(str(path_wave))

                        else:
                            if float(p2)-float(util.tts(wanted_start)[11:13]) == 0:
                                paths.append(str(path_wave))
                    except:
                        pass

                safecon += 1

    p = pile.make_pile(paths)
    for traces in p.chopper(tmin=wanted_start, tinc=duration):
        if traces:
            if traces[0].tmax < wanted_end:
            #    for i in range(0, tr_packages):
            #        traces = traces
                #for tr in traces:
            #    tr.chop(tr.tmin+i*duration,
            #            tr.tmin+i*duration+duration)
                    #tr.downsample_to(gf_freq)
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
