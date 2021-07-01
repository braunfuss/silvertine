import copy
import os
import numpy as num
import logging

from pyrocko import util, model, orthodrome, pile
from pyrocko.gui import snuffler
from pyrocko.gui import marker as pmarker
from pyrocko.gui import util as gui_util
from pyrocko.gui.snuffling import Snuffling, Param, Choice, Switch
from lassie import geo, ifc


logger = logging.getLogger('lassie.snuffling')


kind_default = '1 (green)'


def detections_to_event_markers(fn_detections):
    markers = []
    if fn_detections:
        with open(fn_detections, 'r') as f:
            for line in f.readlines():
                data = line.split()
                i, t_d, t_t, apeak, latpeak, lonpeak, xpeak, ypeak, zpeak = \
                    data
                lat, lon = orthodrome.ne_to_latlon(
                    float(latpeak), float(lonpeak), float(xpeak), float(ypeak))
                t = util.str_to_time("%s %s" % (t_d, t_t))
                label = "%s-%s" % (apeak, i)
                e = model.Event(lat=lat, lon=lon, depth=float(zpeak),
                                name=label, time=t)
                m = gui_util.EventMarker(e, kind=int(kind_default[0]))
                markers.append(m)

    return markers


class LassieSnuffling(Snuffling):

    @property
    def __doc__(self):
        s = '''
        <html>
        <head>
        <style type="text/css">
            body { margin-left:10px };
        </style>
        </head>
        <body>
        <h2 style="test-align:center"> Scrutinize Lassie Performance and
        Re-Detect</h2>
        <p>
        Adjust the detector <i>threshold</i>, press <i>run</i>. From every
        instant, where the signal rises above <i>threshold</i>, a time length
        of <i>tsearch</i> seconds is searched for a maximum. Detections are
        added as event markers to the viewer.  </p>

        <p>
        If you want to save the updated detections, it might be helpful to use
        the marker table
        (see <a href="http://pyrocko.org/v0.3/snuffler_tutorial.html"> Snuffler
        Tutorial</a> at the bottom) to sort all markers by their kind.
        </p>
        <h3 style="test-align:center"> Compare Lassie Detections with Reference
        Catalog</h3>
        <p>
        %s
        </p>
        </body>
        </html>

        ''' % self.show_comparison.__doc__
        return s

    def __init__(self):
        Snuffling.__init__(self)
        self.config = None
        self.detections = []

    def setup(self):
        if self.config:
            detector_default = self.config.detector_threshold
        else:
            detector_default = 100.

        self.set_name('Lassie investigate')
        self.add_parameter(Param('Tsearch', 'tsearch', 20., 0.01, 100))
        self.add_parameter(Param(
            'Detector threshold', 'detector_threshold', detector_default, 1.,
            10000.))
        self.add_parameter(Switch('Level Trace', 'level_trace', False))
        self.add_parameter(Switch(
            'Hold comparison figure', 'hold_figure', False))
        self.add_parameter(Choice(
            'new marker kind', 'marker_kind', kind_default,
            ['1 (green)', '2 (blue)', '3 (orange)', '4 (purple)', '5 (brown)',
             '0 (red)']))
        self.add_trigger('load reference', self.load_comparison)
        self.add_trigger('show comparison', self.show_comparison)
        self.add_trigger('remove comparison', self.remove_comparison)
        # self.add_trigger('read Lassie config', self.fail)
        self.set_live_update(True)
        self.markers_compare = []
        self.detections = []
        self.fig = None
        self.fframe = None
        self.grid = self.config.get_grid()

    def mycleanup(self):
        viewer = self.get_viewer()
        viewer.release_data(self._tickets)
        viewer.remove_markers(self.detections)
        self._tickets = []
        self._markers = []

    def call(self):
        self.mycleanup()
        self.detections = []
        i_detection = 0
        zpeak = 0.
        lat = 0.
        lon = 0.
        for traces in self.chopper_selected_traces(
                mode='all',
                trace_selector=lambda x: x.station == "SMAX",
                fallback=True):
            tr_smax = [tr for tr in traces if tr.location == '']
            tr_i = [tr for tr in traces if tr.location == 'i']
            if not tr_i:
                tr_i = [None] * len(tr_smax)

            for tr_i, tr_stackmax in zip(tr_i, tr_smax):
                tpeaks, apeaks = tr_stackmax.peaks(
                    self.detector_threshold, self.tsearch)
                if self.level_trace:
                    ltrace = tr_stackmax.copy(data=False)
                    ltrace.set_ydata(
                        num.ones(
                            tr_stackmax.data_len()) * self.detector_threshold)
                    self.add_trace(ltrace)
                for t, a in zip(tpeaks, apeaks):
                    if tr_i:
                        lat, lon, xpeak, ypeak, zpeak = \
                            self.grid.index_to_location(tr_i(t)[1])

                        lat, lon = orthodrome.ne_to_latlon(
                            lat, lon, xpeak, ypeak)

                    e = model.Event(
                        time=t, name="%s-%s" % (i_detection, a), lat=lat,
                        lon=lon, depth=zpeak)
                    self.detections.append(
                        gui_util.EventMarker(
                            event=e, kind=int(self.marker_kind[0])))
                    i_detection += 1
        self.add_markers(self.detections)

        if self.hold_figure:
            self.show_comparison()

    def load_comparison(self):
        '''
        For comparison in synthetic tests.
        '''
        fn = self.input_filename(caption='Select an event catalog')
        kind_compare = 4
        compare_events = model.load_events(fn)
        markers = [gui_util.EventMarker(event=e, kind=kind_compare) for e in
                   compare_events]

        self.markers_compare = markers
        self.add_markers(self.markers_compare)

    def remove_comparison(self):
        '''Remove comparison markers from viewer.'''
        self.get_viewer().remove_markers(self.markers_compare)

    def filter_visible(self, markers):
        vtmin, vtmax = self.get_viewer().get_time_range()
        return [x for x in markers if vtmin < x.tmin < vtmax]

    def show_comparison(self):
        '''
        Iterates through reference catalog and searches for lassie detection
        candidates in a time window of +- 1.5 seconds around the reference.

        If multiple candidates are available selects the first as the matching
        lassie detection for this reference.

        This option requires the catalog to contain only pyrocko.model.Event
        instances.
        '''
        scan_time = 3.
        # select_by = 'first'

        if not self.markers_compare:
            self.fail('No catalog to compare to')

        markers_compare = self.filter_visible(self.markers_compare)
        not_detected = []
        detections_success = []
        detections = copy.deepcopy(self.filter_visible(self.detections))
        for i_m, mcompare in enumerate(markers_compare):
            detection_times = num.array([d.tmin for d in detections])
            i_want = num.where(num.abs(detection_times - mcompare.tmin)
                               < (scan_time / 2.))[0]
            if len(i_want) == 0:
                not_detected.append(mcompare)
                continue

            candidates = [detections[i] for i in i_want]

            # if select_by == 'first':
            matched_marker = min(
                candidates, key=lambda x: x.get_event().time)

            # elif select_by == 'strongest':
            #     matched_marker = max(
            #         candidates, key=lambda x: float(x.get_event().name))

            detections_success.append((matched_marker, i_m))

            for c in candidates:
                detections.remove(c)

        if self.hold_figure and self.fframe and not self.fframe.closed:
            self.fig.clf()
        else:
            self.fframe = self.pylab('Lassie', get='figure_frame')
            self.fig = self.fframe.gcf()

        ax = self.fig.add_subplot(111)
        compare_events = [x.get_event() for x in markers_compare]
        associated_events = [compare_events[a[1]] for a in detections_success]
        magnitudes = [e.get_event().magnitude for e in markers_compare]
        detected_magnitudes = [e.magnitude for e in associated_events]
        bins = num.linspace(-1, max(magnitudes), 30)
        ax.hist([detected_magnitudes, magnitudes], bins,
                label=['Lassie', 'Reference'], alpha=0.7)
        n_leftover_detections = len(detections)
        n_undetected = len(not_detected)

        ax.text(
            0.05, 0.95, 'Other detections: %s\nNot detected: %s (%1.1f %%)' %
            (n_leftover_detections, n_undetected,
             (float(n_undetected)/len(markers_compare)*100.)),
            transform=ax.transAxes)

        ax.set_xlabel('Magnitude')
        ax.set_ylabel('N detections')
        ax.legend()
        self.fig.canvas.draw()


def __snufflings__():
    return [LassieSnuffling()]


def snuffle(config):
    global _lassie_config
    _lassie_config = copy.deepcopy(config)
    for _ifc in _lassie_config.image_function_contributions:
        _ifc.setup(config)

    def load_snuffling(win):
        s = LassieSnuffling()
        s.config = _lassie_config
        s.setup()
        win.pile_viewer.viewer.add_snuffling(s, reloaded=True)
        win.pile_viewer.viewer.add_blacklist_pattern('*.SMAX.i.*')
        for bl in _lassie_config.blacklist:
            win.pile_viewer.viewer.add_blacklist_pattern('%s.*' % bl)

        detections_path = _lassie_config.get_detections_path()

        if os.path.exists(detections_path):
            s.detections = detections_to_event_markers(detections_path)
            s.add_markers(s.detections)

        for _ifc in s.config.image_function_contributions:
            if isinstance(_ifc, ifc.ManualPickIFC):
                markers_path_extra = _ifc.picks_path
            elif isinstance(_ifc, ifc.TemplateMatchingIFC):
                markers_path_extra = _ifc.template_markers_path
            else:
                continue

            if os.path.exists(markers_path_extra):
                s.add_markers(pmarker.load_markers(markers_path_extra))
            else:
                logger.warn('No such file: %s (referenced in %s, named %s)' % (
                    markers_path_extra, _ifc.__class__.__name__, _ifc.name))

    receivers = config.get_receivers()
    stations = set()
    lats, lons = geo.points_coords(receivers, system='latlon')
    for ir, (lat, lon) in enumerate(zip(lats, lons)):
        n, s, l = receivers[ir].codes[:3]
        stations.add(model.Station(
            lat=lat, lon=lon, network=n, station=s, location=l))

    paths = config.expand_path(config.data_paths)
    paths.append(config.get_ifm_dir_path())

    p = pile.make_pile(paths=paths, fileformat='detect')

    meta = {'tabu': True}
    for tr in p.iter_traces(trace_selector=lambda x: x.station == 'SMAX'):
        if tr.meta:
            tr.meta.update(meta)
        else:
            tr.meta = meta

    snuffler.snuffle(p, stations=stations,
                     launch_hook=load_snuffling)


__all__ = [
    'snuffle']
