from __future__ import print_function
from builtins import str
import os
import copy
import numpy as num
from collections import defaultdict
from pyrocko.gui.snuffling import Snuffling, Param, PhaseMarker, Switch, Choice, \
    EventMarker
from pyrocko import guts, orthodrome, trace, pile, model
from pyrocko.gui.util import to01
from pyrocko.plot import graph_colors
from matplotlib import pyplot as plt

km = 1000.
wood_anderson_response = trace.PoleZeroResponse(
    zeros=[0., 0.],
    poles=[(-5.49779 - 5.60886j), (-5.49779 + 5.60886j)],
    constant=1.
)

vmin = 1500.
vmax = 6000.
 
'''
Local Magnitude Estimation
--------------------------

Main Control High- and Lowpass filters are applied to the data before
simulating the Wood-Anderson receiver.

The suggested default values for the geometrical spreading, anelastic
attenuation and static magnification are those recommended by IASPEI.

For correct estimates these values have to be calibrated for the region
under investigation.

For further information:
http://gfzpublic.gfz-potsdam.de/pubman/item/escidoc:816929:1/component/escidoc:816928/IS_3.3_rev1.pdf

Waveform data either need to have unit 'meters' or if instument
responses have to be removed select *Needs restitution* and use the file
browser to read responses.

References:
- Bormann,  P. and Dewey J., 2012. The new IASPEI standards for determining
        magnitudes from digital data and their relation to classical
        magnitudes. doi:
- Hutton, K.L. and Boore D.M., 1987. The ML scale in southern California.
        Bull. seism. Soc. Am., 77, 2074-2094
- Richter C.F., 1935. An instrumental earthquake magnitude scale, Bull.
        seism. Soc. Am., 25, 1-32.
_responses = None
self.add_parameter(Param(
    'geom. spreading', 'const_a', 1.11, 1., 2.))
self.add_parameter(Param(
    'anelastic attenuation', 'const_b', 0.00189, 0., 1.))
self.add_parameter(Param(
    'static magnification', 'const_c', -2.09, -5., 5.))
self.add_parameter(Param(
    'Duration for "fixed" time window',
    'duration_fixed', 200., 1., 500.))
self.add_parameter(Choice(
    'Time window', 'time_window', 'visible / selected',
    ['visible / selected', 'fixed', 'distance dependent']))
self.add_parameter(Choice(
    'Apply to', 'apply_to', 'active event',
    ['active event', 'selected events', 'all events']))
self.add_parameter(Switch(
    'Show restituted traces', 'show_restituded_traces', False))
self.add_parameter(Switch(
    'Mark readings', 'show_markers', False))
self.add_parameter(Switch(
    'Show plot', 'show_plot', False))
self.add_parameter(Switch(
    'Show message', 'do_show_message', True))
self.add_parameter(Switch(
    'Needs restitution', 'needs_restitution', False))
self.add_parameter(Switch(
    'Set event magnitude', 'modify_inplace', False))
'''




       

def read_responses(dirname):
    responses = {}
    entries = os.listdir(dirname)
    for entry in entries:
        if entry.endswith('.pf'):
            key = tuple(entry[:-3].split('.'))
            fn = os.path.join(dirname, entry)
            resp = guts.load(filename=fn)
            responses[key] = resp

    return responses


def local_magnitude(distance, amplitude, const_a=1.11, const_b=0.00189, const_c=-2.09):

    return num.log10(amplitude*1.0e9) + \
        const_a*num.log10(distance/km) + \
        const_b*distance/km + const_c

def get_response(nslc, input_directory):
    _responses = read_responses(input_directory)
    n, s, l, c = nslc
    for k in [(n, s, l, c), (n, s, c), (s, c), (s,)]:
        if k in _responses:
            return _responses[k]

def get_traces(p, event, stations, trace_selector, tpad, time_window="distance_dependant", vmin=1500, vmax=6000, duration_fixed=200):

   # trace_selector_viewer = self.get_viewer_trace_selector('visible')
    if time_window == 'distance_dependant':
        for station in stations:
            distance = orthodrome.distance_accurate50m(event, station)
            tmin = distance / vmax
            tmax = (distance + event.depth) / vmin

            for trs in p.chopper(
                    tmin=event.time + tmin,
                    tmax=event.time + tmax,
                    tpad=tpad,
                    trace_selector=lambda tr: (
                        trace_selector(tr) and
                        tr.nslc_id[:3] == station.nsl())):

                for tr in trs:
                    yield tr

    elif time_window == 'fixed': # put here chopper
        tmin = 0. 
        tmax = duration_fixed
        for trs in p.chopper(
                tmin=event.time + tmin,
                tmax=event.time + tmax,
                tpad=tpad,
                trace_selector=lambda tr: (
                    trace_selector(tr))):

            for tr in trs:
                yield tr

    #else: # put here chopper
     #   for trs in self.chopper_selected_traces(
      #          fallback=True, tpad=tpad,
       #         trace_selector=trace_selector, mode='inview'):

        #    for tr in trs:
         #       yield tr

def call(events, data_paths, fmin=1, fmax=5, show_restituded_traces=False, stations=None,
         needs_restitution=True, make_markers=True, show_plot=True,
         modify_inplace=True):


    events.sort(key=lambda ev: ev.time)

    stations_dict = dict((s.nsl(), s) for s in stations)


    markers = []
    local_magnitudes = []
    p = pile.make_pile(data_paths, fileformat="mseed", show_progress=False)

    for event in events:
        mags = defaultdict(list)
        tpad = 2./fmin

        def trace_selector(tr):
            c = tr.channel.upper()
            return c.endswith('E') or c.endswith('N') or \
                tr.location.endswith('_rest')

        distances = {}
        rest_traces = []

        event2 = copy.deepcopy(event)

        for tr in get_traces(p,
                event, stations_dict.values(), trace_selector, tpad):

            nslc = tr.nslc_id

            try:
                tr.highpass(4, fmin, nyquist_exception=True)
                tr.lowpass(4, fmax, nyquist_exception=True)
            except:
                pass

            try:
                station = stations_dict[nslc[:3]]
            except KeyError as e:
                print(e)
                continue

            if needs_restitution is True:
                resp = get_response(nslc)
                try:
                    tr_vel = tr.transfer(
                        tfade=tpad,
                        freqlimits=(
                            fmin*0.5, fmin,
                            fmax, fmax*2.0),
                        transfer_function=resp,
                        invert=True)
                except trace.TraceTooShort as e:
                    continue

            else:
                try:
                    tr_vel = tr.transfer(
                        tfade=tpad,
                        freqlimits=(
                            fmin*0.5, fmin,
                            fmax, fmax*2.0),
                        transfer_function=wood_anderson_response,
                        invert=False)
                except trace.TraceTooShort as e:
                    continue

            distance = orthodrome.distance_accurate50m(event, station)

            tr_vel.set_codes(location=tr_vel.location+'_rest')
            tr_vel.meta = dict(tabu=True)
            t_of_max, amplitude = tr_vel.absmax()

            if show_restituded_traces:
                rest_traces.append(tr_vel)
                m_nslc = tr_vel.nslc_id
            else:
                m_nslc = tr.nslc_id

            mag = local_magnitude(distance, amplitude)
            if make_markers is True:
                markers.append(PhaseMarker(
                    [m_nslc],
                    t_of_max, t_of_max, 1, phasename='%3.1f' % mag,
                    event=event2))

            mags[nslc[:2]].append(mag)
            distances[nslc[:2]] = distance

        if not mags:
            continue

        for k in mags:
            mags[k] = max(mags[k])

        local_magnitude = round(num.median(list(mags.values())), 1)

        if show_plot is True:
            data = []
            for k in mags:
                data.append((distances[k], mags[k]))

            dists, mags_arr = num.array(data).T

            dists /= km
            fig = plt.figure()
            axes = fig.add_subplot(1, 1, 1)
            axes.plot(dists, mags_arr, 'o', color=to01(graph_colors[0]))
            for x, y, label in zip(dists, mags_arr, mags.keys()):
                axes.text(x, y, '.'.join(label))

            axes.axhline(local_magnitude, color=to01(graph_colors[0]))
            mag_std = num.std(list(mags.values()))

            msg = 'local magnitude: %s, std: %s' % \
                (round(local_magnitude, 1),
                    round(mag_std, 1))
            axes.text(max(dists), local_magnitude, msg,
                      verticalalignment='bottom',
                      horizontalalignment='right')

            axes.axhspan(
                    local_magnitude-mag_std,
                    local_magnitude+mag_std,
                    alpha=0.1)

            axes.set_xlabel('Distance [km]')
            axes.set_ylabel('Local Magnitude')
            plt.savefig("ml_%s.png" % event.name)

        local_magnitudes.append(local_magnitude)

    if modify_inplace is True:
        event.magnitude = local_magnitude


def main():
    events = model.load_events("events.pf")
    call(events)