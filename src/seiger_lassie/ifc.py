from __future__ import print_function
import logging
import numpy as num
from collections import defaultdict
from scipy.signal import fftconvolve
from pyrocko.guts import Object, String, Float, Bool, StringChoice, List, Dict
from pyrocko import trace, autopick, util, model
from pyrocko.gui import util as gui_util
from pyrocko import marker as pmarker
from silvertine.seiger_lassie import shifter, common, geo

logger = logging.getLogger('lassie_seiger.ifc')

guts_prefix = 'seiger_lassie'


def downsample(tr, deltat):
    try:
        tr.downsample_to(
            deltat, demean=False, snap=True,
            allow_upsample_max=4)

    except util.UnavailableDecimation:
        logger.warn('using resample instead of decimation')
        tr.resample(deltat)


class TraceSelector(Object):
    '''
    Filter traces used in an IFC by NSLC-id lists and/or lists of regular
    expressions.
    '''

    white_list = List.T(
        optional=True,
        default=[],
        help='list of NSLC ids'
    )

    white_list_regex = List.T(
        String.T(
            default=[],
            optional=True,
        ),
        help='list of regular expressions'
    )

    def __call__(self, trs):
        matched = []
        for tr in trs:
            nslc = '.'.join(tr.nslc_id)

            if self.white_list and nslc in self.white_list:
                matched.append(tr)
                continue
            if self.white_list_regex and util.match_nslc(
                    self.white_list_regex, nslc):
                matched.append(tr)
        return matched

    def __str__(self):
        return '%s:\n wl: %s\n wlre: %s\n' % (
            self.__class__.__name__, self.white_list, self.white_list_regex)


class IFC(common.HasPaths):
    '''Image function contribution.'''

    name = String.T()
    weight = Float.T(
        default=1.0,
        help='global weight for this IFC')

    weights = Dict.T(
                String.T(help='NSL regular expression identifying stations'),
                Float.T(help='weighting factor'),
                optional=True,
                help='weight selected traces')

    fmin = Float.T()
    fmax = Float.T()
    shifter = shifter.Shifter.T(optional=True)

    trace_selector = TraceSelector.T(
        optional=True, help='select traces to be treated by this IFC')

    def __init__(self, *args, **kwargs):
        common.HasPaths.__init__(self, *args, **kwargs)
        self.shifter.t_tolerance = 1./(self.fmax * 2.)

    def setup(self, config):
        if self.shifter:
            self.shifter.setup(config)

    def get_table(self, grid, receivers):
        return self.shifter.get_table(grid, receivers)

    def get_tpad(self):
        return 4. / self.fmin

    def get_fsmooth(self):
        return self.fmin

    def prescan(self, p):
        pass

    def deltat_cf_is_available(self, deltat_cf):
        return False

    def preprocess(self, trs, wmin, wmax, tpad_new, deltat_cf):
        pass

    def get_weights(self, nsls):
        if self.weights is None:
            return num.ones(len(nsls), dtype=num.float)

        else:
            weights = num.empty(len(nsls))
            selectors = self.weights.keys()
            for insl, nsl in enumerate(nsls):
                weights[insl] = 1.
                for selector in selectors:
                    if util.match_nslc(selector, nsl):
                        weights[insl] = self.weights[selector]
                        break
            return weights


class WavePacketIFC(IFC):

    '''
    An image function useful as a simple fast general purpose event detector.

    This detector type image function is sensible to transient wave packets
    traveling with a given velocity through the network, e.g. P phase together
    with P coda or S together with S coda or surface waves packets.
    '''

    fsmooth = Float.T(optional=True)
    fsmooth_factor = Float.T(default=0.1)

    def get_tpad(self):
        return 4. / self.get_fsmooth()

    def get_fsmooth(self):
        if self.fsmooth is not None:
            return self.fsmooth
        else:
            return self.fmin * self.fsmooth_factor

    def deltat_cf_is_available(self, deltat_cf):
        return deltat_cf < 0.025 / self.get_fsmooth()

    def preprocess(self, trs, wmin, wmax, tpad_new, deltat_cf):
        fsmooth = self.get_fsmooth()

        if self.trace_selector:
            trs = self.trace_selector(trs)

        if not trs:
            return []

        trs_filt = []
        for orig_tr in trs:
            tr = orig_tr.copy()
            tr.bandpass(4, self.fmin, self.fmax, demean=True)

            tr.ydata = tr.ydata**2

            n = int(num.round(1./fsmooth / tr.deltat))
            taper = num.hanning(n)
            tr.set_ydata(fftconvolve(tr.get_ydata(), taper))
            tr.set_ydata(num.maximum(tr.ydata, 0.0))
            tr.shift(-(n/2.*tr.deltat))
            downsample(tr, deltat_cf)
            trs_filt.append(tr)

        trs_by_nsl = common.grouped_by(trs_filt, lambda tr: tr.nslc_id[:3])

        dataset = []
        for nsl in sorted(trs_by_nsl.keys()):
            sumtr = None
            for tr in trs_by_nsl[nsl]:
                if sumtr is None:
                    sumtr = tr.copy()
                    sumtr.set_codes(channel='CF_%s' % self.name)
                else:
                    sumtr.add(tr)

            navg = 5./fsmooth / deltat_cf

            yavg = trace.moving_avg(sumtr.ydata, navg)

            if num.any(yavg == 0.0):
                continue

            sumtr.ydata /= yavg
            # sumtr.ydata -= 1.0
            sumtr.chop(wmin - tpad_new, wmax + tpad_new)

            dataset.append((nsl, sumtr))

        return dataset


class OnsetIFC(IFC):

    '''
    An image function sensible to sharp onsets in the signal.

    This image function is based on an STA/LTA.
    '''

    short_window = Float.T()
    window_ratio = Float.T(default=5.0)
    fsmooth = Float.T(optional=True)
    fsmooth_factor = Float.T(optional=True)
    fnormalize = Float.T(optional=True)
    fnormalize_factor = Float.T(optional=True)

    def get_tpad(self):
        return 2. / self.get_fnormalize()

    def get_fsmooth(self):
        if self.fsmooth is None and self.fsmooth_factor is None:
            raise common.LassieError('must use fsmooth or fsmooth_factor')

        if self.fsmooth is not None:
            return self.fsmooth
        else:
            return self.fmin * self.fsmooth_factor

    def get_fnormalize(self):
        if self.fnormalize is None and self.fnormalize_factor is None:
            raise common.LassieError(
                'must use fnormalize or fnormalize_factor')

        if self.fnormalize is not None:
            return self.fnormalize
        else:
            return self.fmin * self.fnormalize_factor

    def deltat_cf_is_available(self, deltat_cf):
        return deltat_cf < 0.025 / self.get_fsmooth()

    def preprocess(self, trs, wmin, wmax, tpad_new, deltat_cf):
        fsmooth = self.get_fsmooth()
        fnormalize = self.get_fnormalize()

        if self.trace_selector:
            trs = self.trace_selector(trs)

        if not trs:
            return []

        trs_filt = []
        for orig_tr in trs:
            tr = orig_tr.copy()
            tr.highpass(4, self.fmin, demean=True)
            tr.lowpass(4, self.fmax, demean=False)
            tr.ydata = tr.ydata**2

            trs_filt.append(tr)

        trs_by_nsl = common.grouped_by(trs_filt, lambda tr: tr.nslc_id[:3])

        swin = self.short_window
        lwin = self.window_ratio * swin

        dataset = []
        for nsl in sorted(trs_by_nsl.keys()):
            sumtr = None
            for tr in trs_by_nsl[nsl]:
                if sumtr is None:
                    sumtr = tr.copy()
                    sumtr.set_codes(channel='CF_%s' % self.name)
                else:
                    sumtr.add(tr)

            sumtr.set_ydata(sumtr.get_ydata().astype(num.float32))
            autopick.recursive_stalta(swin, lwin, 1.0, 4.0, 3.0, sumtr)

            sumtr.shift(-swin/2.)

            normtr = sumtr.copy()

            ntap_smooth = int(num.round(1./fsmooth / tr.deltat))
            ntap_normalize = int(num.round(1./fnormalize / tr.deltat))
            taper_smooth = num.hanning(ntap_smooth)
            taper_normalize = num.hanning(ntap_normalize)

            sumtr.shift(-(ntap_smooth / 2. * tr.deltat))
            normtr.shift(-(ntap_normalize / 2. * tr.deltat))

            sumtr.set_ydata(
                fftconvolve(sumtr.get_ydata()/len(trs_by_nsl), taper_smooth))

            normtr.set_ydata(
                fftconvolve(
                    normtr.get_ydata()/len(trs_by_nsl),
                    taper_normalize))

            normtr.set_codes(channel='normtr')

            tmin = max(sumtr.tmin, normtr.tmin)
            tmax = min(sumtr.tmax, normtr.tmax)
            sumtr.chop(tmin, tmax)
            normtr.chop(tmin, tmax)

            sumtr.set_ydata(sumtr.get_ydata() / normtr.get_ydata())

            downsample(sumtr, deltat_cf)

            sumtr.chop(wmin - tpad_new, wmax + tpad_new)

            dataset.append((nsl, sumtr))

        return dataset


class TemplateMatchingIFC(IFC):

    template_event_path = common.Path.T(
        help='Event parameters of the template')

    template_markers_path = common.Path.T(
        optional=False,
        help='File with markers defining the template')

    sum_square = Bool.T(
        default=False,
        help='Sum square of correlation')

    normalization = StringChoice.T(
        default='gliding',
        choices=['off', 'normal', 'gliding'])

    downsample_rate = Float.T(
        optional=True,
        help='If set, downsample to this sampling rate before processing [Hz]')

    use_fft = Bool.T(
        optional=True,
        default=False,
        help='If set, correlate traces in the spectral domain')

    def get_tpad(self):
        tmin_masters = min(tr.tmin for tr in self.masters.values())
        tmax_masters = max(tr.tmax for tr in self.masters.values())
        tmaster = tmax_masters - tmin_masters
        return tmaster

    def get_template_origin(self):

        event = model.load_one_event(self.expand_path(
            self.template_event_path))

        origin = geo.Point(
            lat=event.lat,
            lon=event.lon,
            z=event.depth)

        return origin

    def get_table(self, grid, receivers):
        origin = self.get_template_origin()
        return self.shifter.get_offset_table(grid, receivers, origin)

    def extract_template(self, p):

        markers = gui_util.load_markers(self.expand_path(
            self.template_markers_path))

        def trace_selector_global(tr):
            return True

        period_highpass = 1./self.fmin
        tpad = 2 * period_highpass

        master_traces = []
        for marker in markers:
            if marker.tmin == marker.tmax:
                logger.warn('tmin == tmax in template marker %s' % marker)

            if not marker.nslc_ids:
                trace_selector = trace_selector_global
            else:
                def trace_selector(tr):
                    return (
                        marker.match_nslc(tr.nslc_id) and
                        trace_selector_global(tr))

            master_traces.extend(p.all(
                tmin=marker.tmin,
                tmax=marker.tmax,
                trace_selector=trace_selector,
                tpad=tpad))

        masters = {}
        for xtr in master_traces:
            tr = xtr.copy()
            if self.downsample_rate is not None:
                downsample(tr, 1./self.downsample_rate)

            tr.highpass(4, self.fmin, demean=False)
            tr.lowpass(4, self.fmax, demean=False)
            smin = round(xtr.wmin / tr.deltat) * tr.deltat
            smax = round(xtr.wmax / tr.deltat) * tr.deltat
            tr.chop(smin, smax)
            if tr.nslc_id in masters:
                raise common.LassieError(
                    'more than one waveform selected on trace with id "%s"'
                    % '.'.join(tr.nslc_id))

            masters[tr.nslc_id] = tr

        return masters

    def prescan(self, p):
        self.masters = self.extract_template(p)

    def deltat_cf_is_available(self, deltat_cf):
        return False

    def preprocess(self, trs, wmin, wmax, tpad_new, deltat_cf):

        tmin_masters = min(tr.tmin for tr in self.masters.values())
        tmax_masters = max(tr.tmax for tr in self.masters.values())
        tmaster = tmax_masters - tmin_masters
        tref = tmin_masters

        nsl_to_traces = defaultdict(list)
        for orig_b in trs:

            b = orig_b.copy()

            nslc = b.nslc_id
            a = self.masters.get(nslc, False)
            if not a:
                continue

            if self.downsample_rate is not None:
                downsample(b, 1./self.downsample_rate)

            b.highpass(4, self.fmin, demean=False)
            b.lowpass(4, self.fmax, demean=False)
            smin = round((wmin - tmaster) / b.deltat) * b.deltat
            smax = round((wmax + tmaster) / b.deltat) * b.deltat
            b.chop(smin, smax)

            normalization = self.normalization
            if normalization == 'off':
                normalization = None

            c = trace.correlate(
                a, b, mode='valid', normalization=normalization,
                use_fft=self.use_fft)

            c.shift(-c.tmin + b.tmin - (a.tmin - tref))
            c.meta = {'tabu': True}
            if self.sum_square:
                c.ydata = c.ydata**2

            c.chop(wmin - tpad_new, wmax + tpad_new)

            nsl_to_traces[nslc[:3]].append(c)

        dataset = []
        for nsl, cs in nsl_to_traces.items():
            csum = cs[0]
            for c in cs[1:]:
                csum.add(c)

            dataset.append((nsl, csum))

        return dataset


class ManualPickIFC(IFC):

    '''
    An image function based on manual picks.
    '''

    fsmooth = Float.T(default=0.1)
    picks_path = common.Path.T()
    picks_phasename = String.T()

    def __init__(self, *args, **kwargs):
        IFC.__init__(self, *args, **kwargs)
        self._picks_data = None

    def get_picks_data(self):
        if not self._picks_data:
            markers = pmarker.load_markers(self.expand_path(self.picks_path))
            nsl_to_index = {}
            picked_index = []
            picked_times = []
            index = -1
            for marker in markers:
                if isinstance(marker, pmarker.PhaseMarker) \
                        and marker.get_phasename() == self.picks_phasename:

                    nsl = marker.one_nslc()[:3]
                    if nsl not in nsl_to_index:
                        index += 1
                        nsl_to_index[nsl] = index

                    ind = nsl_to_index[nsl]

                    picked_index.append(ind)
                    picked_times.append((marker.tmin + marker.tmax) * 0.5)

            self._picks_data = (
                nsl_to_index,
                num.array(picked_index, dtype=num.int64),
                num.array(picked_times, dtype=num.float))

        return self._picks_data

    def get_fsmooth(self):
        return self.fsmooth

    def deltat_cf_is_available(self, deltat_cf):
        return deltat_cf < 0.025 / self.get_fsmooth()

    def preprocess(self, trs, wmin, wmax, tpad_new, deltat_cf):
        fsmooth = self.get_fsmooth()

        nsl_to_index, picked_index, picked_times = self.get_picks_data()

        if not trs:
            return []

        mask = num.logical_and(
            wmin <= picked_times,
            wmax >= picked_times)

        picked_times = picked_times[mask]
        picked_index = picked_index[mask]

        trs_filt = []
        for orig_tr in trs:
            tr = orig_tr.copy()

            nsl = tr.nslc_id[:3]
            try:
                index = nsl_to_index[nsl]
                ts = picked_times[picked_index == index]
                its = (num.round((ts - tr.tmin) / tr.deltat)).astype(num.int64)
                its = its[num.logical_and(0 <= its, its < tr.data_len())]
                ydata = num.zeros(tr.data_len())
                ydata[its] = 1.0
                tr.set_ydata(ydata)
                trs_filt.append(tr)

            except KeyError:
                pass

        trs_by_nsl = common.grouped_by(trs_filt, lambda tr: tr.nslc_id[:3])

        dataset = []
        for nsl in sorted(trs_by_nsl.keys()):
            sumtr = None
            sumn = 0
            for tr in trs_by_nsl[nsl]:
                if sumtr is None:
                    sumtr = tr.copy()
                    sumtr.set_codes(channel='CF_%s' % self.name)
                    sumn = 1
                else:
                    sumtr.add(tr)
                    sumn += 1

            sumtr.ydata /= sumn

            ntap = int(num.round(1./fsmooth / tr.deltat))
            taper = num.hanning(ntap)
            sumtr.shift(-(ntap/2.*tr.deltat))

            sumtr.set_ydata(
                num.convolve(sumtr.get_ydata()/len(trs_by_nsl), taper))

            downsample(sumtr, deltat_cf)

            sumtr.chop(wmin - tpad_new, wmax + tpad_new)
            if num.any(sumtr != 0.):
                dataset.append((nsl, sumtr))

        return dataset


__all__ = [
    'IFC',
    'WavePacketIFC',
    'OnsetIFC',
    'TemplateMatchingIFC',
    'TraceSelector',
]
