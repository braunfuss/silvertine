from collections import defaultdict

import numpy as num
from matplotlib import pyplot as plt

from pyrocko.guts import Float

from pyrocko import gf, trace, plot, beachball, util, orthodrome
from pyrocko import moment_tensor as pmt

km = 1000.

util.setup_logging('gf_shakemap')


def make_shakemap(engine, source, store_id, folder, stations=None):
    targets, norths, easts, stf_spec = get_scenario(engine,
                                                    source,
                                                    store_id)
    response = engine.process(source, targets)
    values = post_process(response, norths, easts, stf_spec)
    if stations is not None:
        targets_stations, norths_stations, easts_stations, stf_spec = get_scenario(engine,
                                                                                   source,
                                                                                   store_id,
                                                                                   stations=stations)
        response_stations = engine.process(source, targets_stations)
        values_stations = post_process(response_stations, norths_stations,
                                       easts_stations,
                                       stf_spec, stations=True)
        values_stations = values_stations[0][0:len(stations)]

        plot_shakemap(source, norths, easts, values, 'gf_shakemap.d100.png', folder,
                      values_stations=values_stations,
                      norths_stations=norths_stations,
                      easts_stations=easts_stations)

    else:
        plot_shakemap(source, norths, easts, values, 'gf_shakemap.d100.png', folder)


def get_scenario(engine, source, store_id, extent=30, ngrid=50,
                 stations=None):
    '''
    Setup scenario with source model, STF and a rectangular grid of targets.
    '''

    # physical grid size in [m]
    grid_extent = extent*km
    lat, lon = source.lat, source.lon
    # number of grid points
    nnorth = neast = ngrid
    try:
        stf_spec = BruneResponse(duration=source.duration)
    except AttributeError:
        stf_spec = BruneResponse(duration=0.5)
    store = engine.get_store(store_id)

    if stations is None:
        # receiver grid
        r = grid_extent / 2.0

        norths = num.linspace(-r, r, nnorth)
        easts = num.linspace(-r, r, neast)
        norths2, easts2 = coords_2d(norths, easts)
        targets = []
        for i in range(norths2.size):

            for component in 'ZNE':
                target = gf.Target(
                    quantity='displacement',
                    codes=('', '%04i' % i, '', component),
                    lat=lat,
                    lon=lon,
                    north_shift=float(norths2[i]),
                    east_shift=float(easts2[i]),
                    store_id=store_id,
                    interpolation='nearest_neighbor')
                # in case we have not calculated GFs for zero distance
                if source.distance_to(target) >= store.config.distance_min:
                    targets.append(target)
    else:
        targets = []
        norths = []
        easts = []
        # here maybe use common ne frame?
        for i, st in enumerate(stations):#
            north, east = orthodrome.latlon_to_ne_numpy(
                lat,
                lon,
                st.lat,
                st.lon,
                )
            norths.append(north[0])
            easts.append(east[0])
            norths2, easts2 = coords_2d(north, east)
            for cha in st.channels:
                target = gf.Target(
                    quantity='displacement',
                    codes=(str(st.network), i, str(st.location),
                           str(cha.name)),
                    lat=lat,
                    lon=lon,
                    north_shift=float(norths2),
                    east_shift=float(easts2),
                    store_id=store_id,
                    interpolation='nearest_neighbor')
                # in case we have not calculated GFs for zero distance
                if source.distance_to(target) >= store.config.distance_min:
                    targets.append(target)
        norths = num.asarray(norths)
        easts = num.asarray(easts)
    return targets, norths, easts, stf_spec


def post_process(response, norths, easts, stf_spec, stations=False):
    nnorth = norths.size
    neast = easts.size

    norths2, easts2 = coords_2d(norths, easts)

    by_i = defaultdict(list)
    for source, target, tr in response.iter_results():
        tr = tr.copy()
        trans = trace.DifferentiationResponse(2)

        trans = trace.MultiplyResponse(
            [trans, stf_spec])

        tr = tr.transfer(transfer_function=trans)
        # tr = tr.transfer(tfade=10., freqlimits=(0., 0.05, 10, 100.),
        #                  transfer_function=trans)

        tr.highpass(4, 0.5)
        tr.lowpass(4, 4.0)
        tr_resamp = tr.copy()

        # uncomment to active resampling to get a smooth image (slow):
        tr_resamp.resample(tr.deltat*0.25)
        by_i[int(target.codes[1])].append(tr_resamp)

    values = num.zeros(nnorth*neast)

    plot_trs = []
    for i in range(norths2.size):
        trs = by_i[i]
        if trs:
            ysum = num.sqrt(sum(tr.ydata**2 for tr in trs))
            ymax = num.max(ysum)
            values[i] = ymax
            if norths2[i] == easts2[i]:
                plot_trs.extend(trs)
    values = values.reshape((norths.size, easts.size))
    return values


def plot_shakemap(source, norths, easts, values, filename, folder,
                  values_stations=None, easts_stations=None,
                  norths_stations=None):
    plot.mpl_init()
    fig = plt.figure(figsize=plot.mpl_papersize('a5', 'landscape'))
    axes = fig.add_subplot(1, 1, 1, aspect=1.0)

    axes.set_xlim(easts.min()/km, easts.max()/km)
    axes.set_ylim(norths.min()/km, norths.max()/km)

    axes.set_xlabel('Easting [km]')
    axes.set_ylabel('Northing [km]')

    _, vmax = num.min(values), num.max(values)

    im = axes.contourf(
        easts/km, norths/km, values,
        vmin=0., vmax=vmax,
        cmap=plt.get_cmap('YlOrBr'))

    fig.colorbar(im, label='Acceleration [m/s^2]')

    mt = source.pyrocko_moment_tensor()

    beachball.plot_beachball_mpl(
        mt, axes,
        position=(0., 0.),
        color_t='black',
        zorder=2,
        size=20.)
    if values_stations is not None:
        plt.scatter(easts_stations/km, norths_stations/km, c=values_stations, s=36, cmap=plt.get_cmap('YlOrBr'), vmin=0., vmax=vmax, edgecolor="k")

    fig.savefig(folder+filename)
    plt.show()


def coords_2d(norths, easts):
    norths2 = num.repeat(norths, easts.size)
    easts2 = num.tile(easts, norths.size)
    return norths2, easts2


class BruneResponse(trace.FrequencyResponse):

    duration = Float.T()

    def evaluate(self, freqs):
        return 1.0 / (1.0 + (freqs*self.duration)**2)
