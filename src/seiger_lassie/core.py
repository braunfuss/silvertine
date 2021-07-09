import math
import logging
import os.path as op
import shutil
import time

from collections import defaultdict

import numpy as num

from pyrocko import pile, trace, util, io, model
from pyrocko.parstack import parstack, argmax as pargmax

from pyrocko.guts import Object, Timestamp, String, Float


from silvertine.seiger_lassie import common, plot, grid as gridmod, geo
from silvertine.seiger_lassie.config import write_config
from obspy.core.event import read_events
from obspy.core.event import Catalog
from glob import glob
import os
logger = logging.getLogger('seiger_lassie.core')


class Detection(Object):
    id = String.T()
    time = Timestamp.T()
    location = geo.Point.T()
    ifm = Float.T()

    def get_event(self):
        lat, lon = geo.point_coords(self.location, system='latlon')

        event = model.Event(
            lat=lat,
            lon=lon,
            depth=self.location.z,
            time=self.time,
            name='%s (%g)' % (self.id, self.ifm))

        return event


def check_data_consistency(p, config):
    receivers = config.get_receivers()
    nsl_ids = [nslc_id[:3] for nslc_id in p.nslc_ids.keys()]
    r_ids = [r.codes for r in receivers]

    r_not_in_p = []
    t_not_in_r = []
    to_be_blacklisted = []

    for r in receivers:
        if r.codes[:3] not in nsl_ids:
            r_not_in_p.append(r.codes)
        if '.'.join(r.codes[:3]) in config.blacklist:
            to_be_blacklisted.append('.'.join(r.codes))

    for nsl_id in nsl_ids:
        if nsl_id not in r_ids:
            t_not_in_r.append(nsl_id)

    if len(r_not_in_p) != 0.:
        logger.warn('Following receivers have no traces in data set:')
        for nsl_id in r_not_in_p:
            logger.warn('  %s' % '.'.join(nsl_id))
        logger.warn('-' * 40)

    if len(t_not_in_r) != 0.:
        logger.warn('Following traces have no associated receivers:')
        for nsl_id in t_not_in_r:
            logger.warn('  %s' % '.'.join(nsl_id))
        logger.warn('-' * 40)

    if len(to_be_blacklisted):
        logger.info('Blacklisted receivers:')
        for code in to_be_blacklisted:
            logger.info('  %s' % code)
        logger.info('-' * 40)

    if len(config.blacklist) and\
            len(to_be_blacklisted) != len(config.blacklist):
        logger.warn('Blacklist NSL codes that did not match any receiver:')
        for code in config.blacklist:
            if code not in to_be_blacklisted:
                logger.warn('  %s' % code)
        logger.info('-' * 40)


def zero_fill(trs, tmin, tmax):
    trs = trace.degapper(trs)

    d = defaultdict(list)
    for tr in trs:
        d[tr.nslc_id].append(tr)

    trs_out = []
    for nslc, trs_group in d.items():
        if not all(tr.deltat == trs_group[0].deltat for tr in trs_group):
            logger.warn('inconsistent sample rate, cannot merge traces')
            continue

        if not all(tr.ydata.dtype == trs_group[0].ydata.dtype
                   for tr in trs_group):

            logger.warn('inconsistent data type, cannot merge traces')
            continue

        tr_combi = trs_group[0].copy()
        tr_combi.extend(tmin, tmax, fillmethod='zeros')
        for tr in trs_group[1:]:
            tr.ydata = tr.ydata.astype(num.int32)
            tr_combi.add(tr)

        trs_out.append(tr_combi)

    return trs_out


def search(
        config,
        override_tmin=None,
        override_tmax=None,
        show_detections=False,
        show_movie=False,
        show_window_traces=False,
        force=False,
        stop_after_first=False,
        nparallel=None,
        bark=False):

    fp = config.expand_path

    run_path = fp(config.run_path)

    if op.exists(run_path):
        if force:
            shutil.rmtree(run_path)
        else:
            raise common.LassieError(
                'run directory already exists: %s' %
                run_path)

    util.ensuredir(run_path)

    write_config(config, op.join(run_path, 'config.yaml'))

    ifm_path_template = config.get_ifm_path_template()
    detections_path = config.get_detections_path()
    events_path = config.get_events_path()
    figures_path_template = config.get_figures_path_template()

    config.setup_image_function_contributions()
    ifcs = config.image_function_contributions

    grid = config.get_grid()
    receivers = config.get_receivers()

    norm_map = gridmod.geometrical_normalization(grid, receivers)

    data_paths = fp(config.data_paths)
    for data_path in fp(data_paths):
        if not op.exists(data_path):
            raise common.LassieError(
                'waveform data path does not exist: %s' % data_path)

    p = pile.make_pile(data_paths, fileformat='detect')
    if p.is_empty():
        raise common.LassieError('no usable waveforms found')

    for ifc in ifcs:
        ifc.prescan(p)

    shift_tables = []
    tshift_minmaxs = []
    for ifc in ifcs:
        shift_tables.append(ifc.get_table(grid, receivers))
        tshift_minmaxs.append(num.nanmin(shift_tables[-1]))
        tshift_minmaxs.append(num.nanmax(shift_tables[-1]))

    fsmooth_min = min(ifc.get_fsmooth() for ifc in ifcs)

    tshift_min = min(tshift_minmaxs)
    tshift_max = max(tshift_minmaxs)

    if config.detector_tpeaksearch is not None:
        tpeaksearch = config.detector_tpeaksearch
    else:
        tpeaksearch = (tshift_max - tshift_min) + 1.0 / fsmooth_min

    tpad = max(ifc.get_tpad() for ifc in ifcs) + \
        (tshift_max - tshift_min) + tpeaksearch

    tinc = (tshift_max - tshift_min) * 10. + 3.0 * tpad
    tavail = p.tmax - p.tmin
    tinc = min(tinc, tavail - 2.0 * tpad)

    if tinc <= 0:
        raise common.LassieError(
            'available waveforms too short \n'
            'required: %g s\n'
            'available: %g s\n' % (2.*tpad, tavail))

    blacklist = set(tuple(s.split('.')) for s in config.blacklist)
    whitelist = set(tuple(s.split('.')) for s in config.whitelist)

    distances = grid.distances(receivers)
    distances_to_grid = num.min(distances, axis=0)

    distance_min = num.min(distances)
    distance_max = num.max(distances)

    station_index = dict(
        (rec.codes, i) for (i, rec) in enumerate(receivers)
        if rec.codes not in blacklist and (
            not whitelist or rec.codes in whitelist) and (
            config.distance_max is None or
                distances_to_grid[i] <= config.distance_max))

    check_data_consistency(p, config)

    deltat_cf = max(p.deltats.keys())
    assert deltat_cf > 0.0

    while True:
        if not all(ifc.deltat_cf_is_available(deltat_cf * 2) for ifc in ifcs):
            break

        deltat_cf *= 2
    logger.info('CF lassie sampling interval (rate): %g s (%g Hz)' % (
        deltat_cf, 1.0/deltat_cf))

    ngridpoints = grid.size()

    logger.info('number of grid points: %i' % ngridpoints)
    logger.info('minimum source-receiver distance: %g m' % distance_min)
    logger.info('maximum source-receiver distance: %g m' % distance_max)
    logger.info('minimum travel-time: %g s' % tshift_min)
    logger.info('maximum travel-time: %g s' % tshift_max)

    idetection = 0

    tmin = override_tmin or config.tmin or p.tmin + tpad
    tmax = override_tmax or config.tmax or p.tmax - tpad

    events = config.get_events()
    twindows = []
    if events is not None:
        for ev in events:
            if tmin <= ev.time <= tmax:
                twindows.append((
                    ev.time + tshift_min - (tshift_max - tshift_min) *
                    config.event_time_window_factor,
                    ev.time + tshift_min + (tshift_max - tshift_min) *
                    config.event_time_window_factor))

    else:
        twindows.append((tmin, tmax))

    for iwindow_group, (tmin_win, tmax_win) in enumerate(twindows):

        nwin = int(math.ceil((tmax_win - tmin_win) / tinc))

        logger.info('start processing time window group %i/%i: %s - %s' % (
            iwindow_group + 1, len(twindows),
            util.time_to_str(tmin_win),
            util.time_to_str(tmax_win)))

        logger.info('number of time windows: %i' % nwin)
        logger.info('time window length: %g s' % (tinc + 2.0*tpad))
        logger.info('time window payload: %g s' % tinc)
        logger.info('time window padding: 2 x %g s' % tpad)
        logger.info('time window overlap: %g%%' % (
            100.0*2.0*tpad / (tinc+2.0*tpad)))

        iwin = -1

        for trs in p.chopper(tmin=tmin_win, tmax=tmax_win, tinc=tinc, tpad=tpad,
                want_incomplete=config.fill_incomplete_with_zeros,
                trace_selector=lambda tr: tr.nslc_id[:3] in station_index):
            iwin += 1
            trs_ok = []
            for tr in trs:
                if tr.ydata.size == 0:
                    logger.warn(
                        'skipping empty trace: %s.%s.%s.%s' % tr.nslc_id)

                    continue

                if not num.all(num.isfinite(tr.ydata)):
                    logger.warn(
                        'skipping trace because of invalid values: '
                        '%s.%s.%s.%s' % tr.nslc_id)

                    continue

                trs_ok.append(tr)

            trs = trs_ok

            if not trs:
                continue

            logger.info('processing time window %i/%i: %s - %s' % (
                iwin + 1, nwin,
                util.time_to_str(trs[0].wmin),
                util.time_to_str(trs[0].wmax)))

            wmin = trs[0].wmin
            wmax = trs[0].wmax

            if config.fill_incomplete_with_zeros:
                trs = zero_fill(trs, wmin - tpad, wmax + tpad)

            t0 = math.floor(wmin / deltat_cf) * deltat_cf
            iwmin = int(round((wmin-tpeaksearch-t0) / deltat_cf))
            iwmax = int(round((wmax+tpeaksearch-t0) / deltat_cf))
            lengthout = iwmax - iwmin + 1

            pdata = []
            trs_debug = []
            parstack_params = []
            for iifc, ifc in enumerate(ifcs):
                dataset = ifc.preprocess(
                    trs, wmin-tpeaksearch, wmax+tpeaksearch,
                    tshift_max - tshift_min, deltat_cf)
                if not dataset:
                    continue

                nstations_selected = len(dataset)

                nsls_selected, trs_selected = zip(*dataset)

                for tr in trs_selected:
                    tr.meta = {'tabu': True}

                trs_debug.extend(trs + list(trs_selected))

                istations_selected = num.array(
                    [station_index[nsl] for nsl in nsls_selected],
                    dtype=num.int)

                arrays = [tr.ydata.astype(num.float) for tr in trs_selected]

                offsets = num.array(
                    [int(round((tr.tmin-t0) / deltat_cf))
                     for tr in trs_selected],
                    dtype=num.int32)

                w = ifc.get_weights(nsls_selected)

                weights = num.ones((ngridpoints, nstations_selected))
                weights *= w[num.newaxis, :]
                weights *= ifc.weight

                shift_table = shift_tables[iifc]

                ok = num.isfinite(shift_table[:, istations_selected])
                bad = num.logical_not(ok)

                shifts = -num.round(
                    shift_table[:, istations_selected] /
                    deltat_cf).astype(num.int32)

                weights[bad] = 0.0
                shifts[bad] = num.max(shifts[ok])

                pdata.append((list(trs_selected), shift_table, ifc))
                parstack_params.append((arrays, offsets, shifts, weights))

            if config.stacking_blocksize is not None:
                ipstep = config.stacking_blocksize
                frames = None
            else:
                ipstep = lengthout
                frames = num.zeros((ngridpoints, lengthout))

            twall_start = time.time()
            frame_maxs = num.zeros(lengthout)
            frame_argmaxs = num.zeros(lengthout, dtype=num.int)
            ipmin = iwmin
            while ipmin < iwmin+lengthout:
                ipsize = min(ipstep, iwmin + lengthout - ipmin)
                if ipstep == lengthout:
                    frames_p = frames
                else:
                    frames_p = num.zeros((ngridpoints, ipsize))

                for (arrays, offsets, shifts, weights) in parstack_params:
                    frames_p, _ = parstack(
                        arrays, offsets, shifts, weights, 0,
                        offsetout=ipmin,
                        lengthout=ipsize,
                        result=frames_p,
                        nparallel=nparallel,
                        impl='openmp')

                if config.sharpness_normalization:
                    frame_p_maxs = frames_p.max(axis=0)
                    frame_p_means = num.abs(frames_p).mean(axis=0)
                    frames_p *= (frame_p_maxs / frame_p_means)[num.newaxis, :]
                    frames_p *= norm_map[:, num.newaxis]

                if config.ifc_count_normalization:
                    frames_p *= 1.0 / len(ifcs)

                frame_maxs[ipmin-iwmin:ipmin-iwmin+ipsize] = \
                    frames_p.max(axis=0)
                frame_argmaxs[ipmin-iwmin:ipmin-iwmin+ipsize] = \
                    pargmax(frames_p)

                ipmin += ipstep
                del frames_p

            twall_end = time.time()

            logger.info('wallclock time for stacking: %g s' % (
                twall_end - twall_start))

            tmin_frames = t0 + iwmin * deltat_cf

            tr_stackmax = trace.Trace(
                '', 'SMAX', '', '',
                tmin=tmin_frames,
                deltat=deltat_cf,
                ydata=frame_maxs)

            tr_stackmax.meta = {'tabu': True}

            trs_debug.append(tr_stackmax)

            if show_window_traces:
                trace.snuffle(trs_debug)

            ydata_window = tr_stackmax.chop(
                wmin, wmax, inplace=False).get_ydata()

            logger.info('CF stats: min %g, max %g, median %g' % (
                num.min(ydata_window),
                num.max(ydata_window),
                num.median(ydata_window)))

            tpeaks, apeaks = list(zip(*[(tpeak, apeak) for (tpeak, apeak) in zip(
                *tr_stackmax.peaks(config.detector_threshold, tpeaksearch)) if
                wmin <= tpeak and tpeak < wmax])) or ([], [])

            tr_stackmax_indx = tr_stackmax.copy(data=False)
            tr_stackmax_indx.set_ydata(frame_argmaxs.astype(num.int32))
            tr_stackmax_indx.set_location('i')

            for (tpeak, apeak) in zip(tpeaks, apeaks):

                iframe = int(round((tpeak - tmin_frames) / deltat_cf))
                imax = frame_argmaxs[iframe]

                latpeak, lonpeak, xpeak, ypeak, zpeak = \
                    grid.index_to_location(imax)

                idetection += 1

                detection = Detection(
                    id='%06i' % idetection,
                    time=tpeak,
                    location=geo.Point(
                        lat=float(latpeak),
                        lon=float(lonpeak),
                        x=float(xpeak),
                        y=float(ypeak),
                        z=float(zpeak)),
                    ifm=float(apeak))

                if bark:
                    common.bark()

                logger.info('detection found: %s' % str(detection))

                f = open(detections_path, 'a')
                f.write('%06i %s %g %g %g %g %g %g\n' % (
                    idetection,
                    util.time_to_str(
                        tpeak,
                        format='%Y-%m-%d %H:%M:%S.6FRAC'),
                    apeak,
                    latpeak, lonpeak, xpeak, ypeak, zpeak))

                f.close()

                ev = detection.get_event()
                f = open(events_path, 'a')
                model.dump_events([ev], stream=f)
                f.close()

                if show_detections or config.save_figures:
                    fmin = min(ifc.fmin for ifc in ifcs)
                    fmax = min(ifc.fmax for ifc in ifcs)

                    fn = figures_path_template % {
                        'id': util.tts(t0).replace(" ", "T"),
                        'format': 'png'}

                    util.ensuredirs(fn)

                    if frames is not None:
                        frames_p = frames
                        tmin_frames_p = tmin_frames
                        iframe_p = iframe

                    else:
                        iframe_min = max(
                            0,
                            int(round(iframe - tpeaksearch/deltat_cf)))
                        iframe_max = min(
                            lengthout-1,
                            int(round(iframe + tpeaksearch/deltat_cf)))

                        ipsize = iframe_max - iframe_min + 1
                        frames_p = num.zeros((ngridpoints, ipsize))
                        tmin_frames_p = tmin_frames + iframe_min*deltat_cf
                        iframe_p = iframe - iframe_min

                        for (arrays, offsets, shifts, weights) \
                                in parstack_params:

                            frames_p, _ = parstack(
                                arrays, offsets, shifts, weights, 0,
                                offsetout=iwmin+iframe_min,
                                lengthout=ipsize,
                                result=frames_p,
                                nparallel=nparallel,
                                impl='openmp')

                        if config.sharpness_normalization:
                            frame_p_maxs = frames_p.max(axis=0)
                            frame_p_means = num.abs(frames_p).mean(axis=0)
                            frames_p *= (
                                frame_p_maxs/frame_p_means)[num.newaxis, :]
                            frames_p *= norm_map[:, num.newaxis]

                        if config.ifc_count_normalization:
                            frames_p *= 1.0 / len(ifcs)
                    try:
                        plot.plot_detection(
                            grid, receivers, frames_p, tmin_frames_p,
                            deltat_cf, imax, iframe_p, xpeak, ypeak,
                            zpeak,
                            tr_stackmax, tpeaks, apeaks,
                            config.detector_threshold,
                            wmin, wmax,
                            pdata, trs, fmin, fmax, idetection,
                            tpeaksearch,
                            movie=show_movie,
                            show=show_detections,
                            save_filename=fn,
                            event=ev)
                    except:
                        pass

                    del frames_p

                if stop_after_first:
                    return

            tr_stackmax.chop(wmin, wmax)
            tr_stackmax_indx.chop(wmin, wmax)

            io.save([tr_stackmax, tr_stackmax_indx], ifm_path_template)

            del frames
        logger.info('end processing time window group: %s - %s' % (
            util.time_to_str(tmin_win),
            util.time_to_str(tmax_win)))
    cat = Catalog()
    files = glob("%s/figures/*qml*" % run_path)
    files.sort(key=os.path.getmtime)
    for file in files:
        cat_read = read_events(file)
        for event in cat_read:
            cat.append(event)
    cat.write("%s/all_events_qml.xml" % run_path, format="QUAKEML")

__all__ = [
    'search',
]
