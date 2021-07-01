from builtins import range
import numpy as num

from pyrocko.plot import automap
from pyrocko import plot, util
from lassie import grid as gridmod, geo

km = 1000.


def map_gmt(
        receivers, grid, center_lat, center_lon, radius, output_path,
        width=25., height=25.,
        show_station_labels=False):

    station_lats = num.array([r.lat for r in receivers])
    station_lons = num.array([r.lon for r in receivers])

    map = automap.Map(
        width=width,
        height=height,
        lat=center_lat,
        lon=center_lon,
        radius=radius,
        show_rivers=False,
        show_topo=False,
        illuminate_factor_land=0.35,
        color_dry=(240, 240, 235),
        topo_cpt_wet='white_sea_land',
        topo_cpt_dry='white_sea_land')

    map.gmt.psxy(
        in_columns=(station_lons, station_lats),
        S='t8p',
        G='black',
        *map.jxyr)

    if grid:
        surf_points = grid.surface_points()
        map.gmt.psxy(
            in_columns=(surf_points[1], surf_points[0]),
            S='c1p',
            G='black',
            *map.jxyr)

    if show_station_labels:
        for r in receivers:
            map.add_label(r.lat, r.lon, '%s' % r.station)

    map.save(output_path)


def map_geometry(config, output_path):
    receivers = config.get_receivers()
    grid = config.get_grid()

    lat0, lon0, north, east, depth = geo.bounding_box_square(
        *geo.points_coords(receivers),
        scale=config.autogrid_radius_factor)

    radius = max((north[1] - north[0]), (east[1] - east[0]))

    radius *= config.autogrid_radius_factor * 1.5

    map_gmt(receivers, grid, lat0, lon0, radius, output_path,
            show_station_labels=False)


def plot_receivers(axes, receivers, system='latlon', units=1.0, style=dict(
        color='black', ms=10.)):
    xs, ys = geo.points_coords(receivers, system=system)
    artists = axes.plot(ys/units, xs/units, '^', **style)
    names = ['.'.join(x for x in rec.codes if x) for rec in receivers]
    for x, y, name in zip(xs, ys, names):
        artists.append(axes.annotate(
            name,
            xy=(y/units, x/units),
            xytext=(10., 0.),
            textcoords='offset points',
            va='bottom',
            ha='left',
            alpha=0.25,
            color='black'))

    return artists


def plot_geometry_carthesian(grid, receivers):

    from matplotlib import pyplot as plt

    plot.mpl_init()

    plt.figure(figsize=(9, 9))
    axes = plt.subplot2grid((1, 1), (0, 0), aspect=1.0)
    plot.mpl_labelspace(axes)

    grid.plot_points(axes, system=('ne', grid.lat, grid.lon))
    plot_receivers(axes, receivers, system=('ne', grid.lat, grid.lon))

    distances = grid.distances(receivers)
    delta_grid = max(grid.dx, grid.dy, grid.dz)
    norm_map = gridmod.geometrical_normalization(distances, delta_grid)
    grid.plot(axes, norm_map, system=('ne', grid.lat, grid.lon))

    plt.show()


def plot_detection(
        grid, receivers, frames, tmin_frames, deltat_cf, imax, iframe,
        xpeak, ypeak, zpeak, tr_stackmax, tpeaks, apeaks,
        detector_threshold, wmin, wmax, pdata, trs_raw, fmin, fmax,
        idetection, tpeaksearch,
        movie=False, save_filename=None, show=True):

    from matplotlib import pyplot as plt
    from matplotlib import cm
    from matplotlib.animation import FuncAnimation

    nsls = set(tr.nslc_id[:3] for tr in trs_raw)
    receivers_on = [r for r in receivers if r.codes in nsls]
    receivers_off = [r for r in receivers if r.codes not in nsls]

    distances = grid.distances(receivers)

    plot.mpl_init(fontsize=9)

    fig = plt.figure(figsize=plot.mpl_papersize('a4', 'landscape'))

    axes = plt.subplot2grid((2, 3), (0, 2), aspect=1.0)
    plot.mpl_labelspace(axes)

    axes2 = plt.subplot2grid((2, 3), (1, 2))
    plot.mpl_labelspace(axes2)

    axes3 = plt.subplot2grid((2, 3), (0, 1), rowspan=2)
    axes4 = plt.subplot2grid((2, 3), (0, 0), rowspan=2)

    if grid.distance_max() > km:
        dist_units = km
        axes.set_xlabel('Easting [km]')
        axes.set_ylabel('Northing [km]')
        axes4.set_ylabel('Distance [km]')
    else:
        dist_units = 1.0
        axes.set_xlabel('Easting [m]')
        axes.set_ylabel('Northing [m]')
        axes4.set_ylabel('Distance [m]')

    axes.locator_params(nbins=6, tight=True)

    axes2.set_xlabel('Time [s]')
    axes2.set_ylabel('Detector level')

    axes3.set_xlabel('Time [s]')
    for el in axes3.get_yticklabels():
        el.set_visible(False)

    axes4.set_xlabel('Time [s]')

    tpeak_current = tmin_frames + deltat_cf * iframe
    t0 = tpeak_current
    tduration = 2.0*tpeaksearch

    axes2.axvspan(
        tr_stackmax.tmin - t0, wmin - t0,
        color=plot.mpl_color('aluminium2'))

    axes2.axvspan(
        wmax - t0, tr_stackmax.tmax - t0,
        color=plot.mpl_color('aluminium2'))

    axes2.axvspan(
        tpeak_current-0.5*tduration - t0,
        tpeak_current+0.5*tduration - t0,
        color=plot.mpl_color('scarletred2'),
        alpha=0.3,
        lw=0.)

    axes2.set_xlim(tr_stackmax.tmin - t0, tr_stackmax.tmax - t0)

    axes2.axhline(
        detector_threshold,
        color=plot.mpl_color('aluminium6'),
        lw=2.)

    t = tr_stackmax.get_xdata()
    amp = tr_stackmax.get_ydata()
    axes2.plot(t - t0, amp, color=plot.mpl_color('scarletred2'), lw=1.)

    for tpeak, apeak in zip(tpeaks, apeaks):
        axes2.plot(
            tpeak-t0, apeak, '*',
            ms=20.,
            mfc='white',
            mec='black')

    station_index = dict(
        (rec.codes, i) for (i, rec) in enumerate(receivers))

    dists_all = []
    amps = []
    shifts = []
    pdata2 = []
    for trs, shift_table, shifter in pdata:
        trs = [tr.copy() for tr in trs]
        dists = []
        for tr in trs:
            istation = station_index[tr.nslc_id[:3]]
            shift = shift_table[imax, istation]
            tr2 = tr.chop(
                tpeak_current - 0.5*tduration + shift,
                tpeak_current + 0.5*tduration + shift,
                inplace=False)

            dists.append(distances[imax, istation])
            amp = tr2.get_ydata() * shifter.weight
            amps.append(num.max(num.abs(amp)))
            shifts.append(shift)

        pdata2.append((trs, dists, shift_table, shifter))
        dists_all.extend(dists)

    dist_min = min(dists_all)
    dist_max = max(dists_all)

    shift_min = min(shifts)
    shift_max = max(shifts)

    amp_max = max(amps)

    scalefactor = (dist_max - dist_min) / len(trs) * 0.5

    axes3.set_xlim(-0.5*tduration + shift_min, 0.5*tduration + shift_max)
    axes3.set_ylim(
        (dist_min - scalefactor)/dist_units,
        (dist_max + scalefactor)/dist_units)

    axes4.set_xlim(-0.5*tduration + shift_min, 0.5*tduration + shift_max)
    axes4.set_ylim(
        (dist_min - scalefactor)/dist_units,
        (dist_max + scalefactor)/dist_units)

    axes3.axvline(
        0.,
        color=plot.mpl_color('aluminium3'),
        lw=2.)

    nsl_have = set()
    for ishifter, (trs, dists, shift_table, shifter) in enumerate(pdata2):
        color = plot.mpl_graph_color(ishifter)

        for tr, dist in zip(trs, dists):
            tr = tr.chop(
                tpeak_current - 0.5*tduration + shift_min,
                tpeak_current + 0.5*tduration + shift_max, inplace=False)

            nsl = tr.nslc_id[:3]
            istation = station_index[nsl]
            shift = shift_table[imax, istation]
            axes3.plot(
                shift, dist/dist_units, '|',
                mew=2, mec=color, ms=10, zorder=2)

            t = tr.get_xdata()
            amp = tr.get_ydata() * shifter.weight
            amp /= amp_max
            axes3.plot(
                t-t0,
                (dist + scalefactor*amp + ishifter*scalefactor*0.1)/dist_units,
                color=color,
                zorder=1)

            if nsl not in nsl_have:
                axes3.annotate(
                    '.'.join(nsl),
                    xy=(t[0]-t0, dist/dist_units),
                    xytext=(10., 0.),
                    textcoords='offset points',
                    verticalalignment='top')

                nsl_have.add(nsl)

        for tr in trs_raw:
            istation = station_index[tr.nslc_id[:3]]
            dist = distances[imax, istation]
            shift = shift_table[imax, istation]

            tr = tr.copy()

            tr.highpass(4, fmin, demean=True)
            tr.lowpass(4, fmax, demean=False)

            tr.chop(
                tpeak_current - 0.5*tduration + shift_min,
                tpeak_current + 0.5*tduration + shift_max)

            t = tr.get_xdata()
            amp = tr.get_ydata().astype(num.float)
            amp = amp / num.max(num.abs(amp))

            axes4.plot(
                t-t0, (dist + scalefactor*amp)/dist_units,
                color='black', alpha=0.5, zorder=1)

            axes4.plot(
                shift, dist/dist_units, '|',
                mew=2, mec=color, ms=10, zorder=2)

    nframes = frames.shape[1]

    iframe_min = max(0, int(round(iframe - 0.5*tduration/deltat_cf)))
    iframe_max = min(nframes-1, int(round(iframe + 0.5*tduration/deltat_cf)))

    amax = frames[imax, iframe]

    axes.set_xlim(grid.ymin/dist_units, grid.ymax/dist_units)
    axes.set_ylim(grid.xmin/dist_units, grid.xmax/dist_units)

    cmap = cm.YlOrBr
    system = ('ne', grid.lat, grid.lon)

    static_artists = []
    static_artists.extend(plot_receivers(
        axes, receivers_on, system=system, units=dist_units, style=dict(
            mfc='black',
            ms=5.0)))

    static_artists.extend(plot_receivers(
        axes, receivers_off, system=system, units=dist_units, style=dict(
            mfc='none',
            ms=5.0)))

    static_artists.extend(axes.plot(
        ypeak/dist_units, xpeak/dist_units, '*',
        ms=20.,
        mec='black',
        mfc='white'))

    static_artists.append(fig.suptitle(
        '%06i - %s' % (idetection, util.time_to_str(t0))))

    frame_artists = []
    progress_artists = []

    def update(iframe):
        if iframe is not None:
            frame = frames[:, iframe]
            if not progress_artists:
                progress_artists[:] = [axes2.axvline(
                    tmin_frames - t0 + deltat_cf * iframe,
                    color=plot.mpl_color('scarletred3'),
                    alpha=0.5,
                    lw=2.)]

            else:
                progress_artists[0].set_xdata(
                    tmin_frames - t0 + deltat_cf * iframe)

        else:
            frame = num.max(frames[:, iframe_min:iframe_max+1], axis=1)

        frame_artists[:] = grid.plot(
            axes, frame,
            amin=0.0,
            amax=amax,
            cmap=cmap,
            system=system,
            artists=frame_artists,
            units=dist_units,
            shading='gouraud')

        return frame_artists + progress_artists + static_artists

    if movie:
        ani = FuncAnimation(
            fig, update,
            frames=list(range(iframe_min, iframe_max+1))[::10] + [None],
            interval=20.,
            repeat=False,
            blit=True)

    else:
        ani = None
        update(None)

    if save_filename:
        fig.savefig(save_filename)

    if show:
        plt.show()
    else:
        plt.close()

    del ani


__all__ = [
    'map_geometry',
    'map_gmt',
    'plot_detection',
    'plot_geometry_carthesian',
    'plot_receivers',
]
