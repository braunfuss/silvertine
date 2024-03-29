from collections import defaultdict

import numpy as num
from matplotlib import pyplot as plt
from pyrocko.guts import Float
from pyrocko import gf, trace, plot, beachball, util, orthodrome, model
from pyrocko import moment_tensor as pmt
import _pickle as pickle
from mpl_toolkits.basemap import Basemap
import matplotlib.cm as cm
import copy
import os
import urllib
from silvertine.locate import locate1D
from silvertine.util import waveform
km = 1000.

util.setup_logging('gf_shakemap')


def make_shakemap(engine, source, store_id, folder, stations=None, save=True,
                  stations_corrections_file=None, pertub_mechanism=False,
                  pertub_degree=20, measured=None, n_pertub=0,
                  value_level=0.004, pertub_velocity_model=False,
                  scenario=False, picks=None,
                  folder_waveforms="/home/asteinbe/bgr_data/acquisition",
                  get_measured=True):

    if n_pertub != 0:
        pertub_mechanism = True
    targets, norths, easts, stf_spec = get_scenario(engine,
                                                    source,
                                                    store_id)
#    try:
    if measured is True:
        event = model.Event(lat=source.lat, lon=source.lon,
                            time=source.time)
        measured = get_max_pga([event], folder_waveforms, stations=stations)
    if measured is True and get_measured is False:
        measured = num.genfromtxt(folder+"measured_pgv",
                                  delimiter=',', dtype=None)
#    except Exception:
#        measured = None

    if pertub_mechanism is True:
        sources = []
        sources.append(source)
        for i in range(0, n_pertub):
            source_pert = copy.deepcopy(source)
            mts = source_pert.pyrocko_moment_tensor()
            strike, dip, rake = mts.both_strike_dip_rake()[0]
            strike = num.random.uniform(strike-pertub_degree,
                                        strike+pertub_degree)
            dip = num.random.uniform(dip-pertub_degree,
                                     dip+pertub_degree)
            rake = num.random.uniform(rake-pertub_degree,
                                      rake+pertub_degree)
            mtm = pmt.MomentTensor.from_values((strike, dip, rake))
            mtm.moment = mts.moment
            source_pert.mnn = mtm.mnn
            source_pert.mee = mtm.mee
            source_pert.mdd = mtm.mdd
            source_pert.mne = mtm.mne
            source_pert.mnd = mtm.mnd
            source_pert.med = mtm.med
            sources.append(source_pert)
    else:
        sources = [source]
    values_pertubed = []
    values_stations_pertubed = []
    for source in sources:
        if pertub_velocity_model is True:
            response = engine.process(source, targets)
            values = post_process(response, norths, easts, stf_spec, savedir=folder,
                                  save=save)
            values_pertubed.append(values)
        else:
            response = engine.process(source, targets)
            values = post_process(response, norths, easts, stf_spec, savedir=folder,
                                  save=save)
            values_pertubed.append(values)
        if stations is not None:
            targets_stations, norths_stations, easts_stations, stf_spec = get_scenario(engine,
                                                                                       source,
                                                                                       store_id,
                                                                                       stations=stations)

            response_stations = engine.process(source, targets_stations)
            values_stations = post_process(response_stations, norths_stations,
                                           easts_stations,
                                           stf_spec, stations=True,
                                           savedir=folder, save=save)
            values_stations = values_stations[0][0:len(stations)]
            if stations_corrections_file is not None:
                stations_corrections_file = num.genfromtxt(stations_corrections_file,
                                                        delimiter=",", dtype=None)
                stations_corrections_value = []
                for i_st, st in enumerate(stations):
                    for stc in stations_corrections_file:
                        if st.station == stc[0].decode():
                            stations_corrections_value.append(float(stc[1]))
                            values_stations[i_st] = values_stations[i_st] + (values_stations[i_st]*float(stc[1]))
                # values_stations = values_stations * num.asarray(stations_corrections_value)
            values_stations_pertubed.append(values_stations)

    if stations is not None:
        plot_shakemap(sources, norths, easts, values_pertubed,
                      'gf_shakemap.png', folder,
                      stations,
                      values_stations_list=values_stations_pertubed,
                      norths_stations=norths_stations,
                      easts_stations=easts_stations,
                      value_level=value_level, measured=measured,
                      engine=engine, plot_values=True,
                      store_id=store_id)
        if measured is not None:
            plot_shakemap(sources, norths, easts, values_pertubed,
                          'gf_shakemap_residuals.png', folder,
                          stations,
                          values_stations_list=values_stations_pertubed,
                          norths_stations=norths_stations,
                          easts_stations=easts_stations,
                          measured=measured,
                          value_level=value_level, engine=engine,
                          store_id=store_id)
    else:
        plot_shakemap(sources, norths, easts, values_pertubed,
                      'gf_shakemap.png', folder,
                      stations, value_level=value_level, engine=engine,
                      store_id=store_id)


def get_scenario(engine, source, store_id, extent=30, ngrid=40,
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
        for i, st in enumerate(stations):
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


def post_process(response, norths, easts, stf_spec, stations=False,
                 show=True, savedir=None, save=True, quantity="velocity"):
    nnorth = norths.size
    neast = easts.size

    norths2, easts2 = coords_2d(norths, easts)

    by_i = defaultdict(list)
    traces = []
    for source, target, tr in response.iter_results():
        tr = tr.copy()

        if quantity == "velocity":
            trans = trace.DifferentiationResponse(1)
        if quantity == "acceleration":
            trans = trace.DifferentiationResponse(2)

        if quantity is not "displacement":
            trans = trace.MultiplyResponse(
                [trans, stf_spec])

            tr = tr.transfer(transfer_function=trans)

        #tr.highpass(4, 0.5)
        #tr.lowpass(4, 4.0)
        tr_resamp = tr.copy()

        # uncomment to active resampling to get a smooth image (slow):
        tr_resamp.resample(tr.deltat*0.25)
        by_i[int(target.codes[1])].append(tr_resamp)
        traces.append(tr)
    values = num.zeros(nnorth*neast)
    from pyrocko import trace as trd
#    if stations is True:

#        trd.snuffle(traces)
#    if stations is True:
#        from pyrocko import io
#        io.save(traces, "traces_pgv.mseed")
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
    if save is True:
        path = savedir + '/shakemap.pkl'
        f = open(path, 'wb')
        pickle.dump([values, easts, norths], f)
        f.close()

    return values


def load_shakemap(path):
    path = savedir + '/shakemap.pkl'
    f = open(path, 'rb')
    values, easts, norths = pickle.load(f)
    f.close()
    return values, easts, norths


def get_max_pga(events, folder_waveforms, gf_freq=10., duration=30,
                forerun=10., quantity="pgv", stations=None,
                absolute_max=False):
    event = events[0]
    tmin = event.time-forerun
    tmax = event.time+duration
    waveforms, stations_list = waveform.load_data_archieve(folder_waveforms,
                                                           gf_freq,
                                                           duration=duration,
                                                           wanted_start=tmin,
                                                           wanted_end=tmax)
    pgas_waveforms = []
    for i, pile_data in enumerate(waveforms):
        if stations is None:
            stations = stations_list[i]
        pga = []
        for st in stations:
            max_value = 0
            value_E = 0
            value_N = 0
            value_Z = 0
            for tr in pile_data:
                if absolute_max is True:
                    if st.station == tr.station:
                        if quantity == "pga":
                            value = num.max(num.diff(num.diff(tr.ydata)))
                        if quantity == "pgv":
                            value = num.max(num.abs(num.diff(tr.ydata)))
                        if value > max_value:
                            max_value = value
                else:
                    if st.station == tr.station:
                        if tr.channel[-1] == "E" or tr.channel == "R":
                            if quantity == "pga":
                                value_E = num.max(num.diff(num.diff(tr.ydata)))
                            if quantity == "pgv":
                                value_E = num.max(num.abs(num.diff(tr.ydata)))
                        if tr.channel[-1] == "N" or tr.channel == "T":
                            if quantity == "pga":
                                value_N = num.max(num.diff(num.diff(tr.ydata)))
                            if quantity == "pgv":
                                value_N = num.max(num.abs(num.diff(tr.ydata)))
                        if tr.channel[-1] == "Z":
                            if quantity == "pga":
                                value_Z = num.max(num.diff(num.diff(tr.ydata)))
                            if quantity == "pgv":
                                value_Z = num.max(num.abs(num.diff(tr.ydata)))
            max_value = num.sqrt((value_E**2)+(value_N)**2+(value_Z**2))
            pgas_waveforms.append(max_value)
    return pgas_waveforms


def plot_shakemap(sources, norths, easts, values_list, filename, folder,
                  stations,
                  values_stations_list=None, easts_stations=None,
                  norths_stations=None, latlon=True, show=False,
                  plot_background_map=True, measured=None,
                  value_level=0.001, quantity="velocity",
                  scale="mm", plot_values=False, vs30=True,
                  engine=None, type_factors=None, store_id=None,
                  plot_snr=False):
    plot.mpl_init()
    fig = plt.figure(figsize=plot.mpl_papersize('a5', 'landscape'))
    axes = fig.add_subplot(1, 1, 1, aspect=1.0)
    mts = []
    plot_kwargs = {
        'size': 4000,
        'edgecolor': 'black'
        }

    for i, source in enumerate(sources):
        mts.append(source.pyrocko_moment_tensor())
        if i == 0:
            best_mt = source.pyrocko_moment_tensor()
    for i, values_pertubed in enumerate(values_list):
        if i == 0:
            values = values_pertubed
            values_cum = num.zeros(num.shape(values))
            values_cum = values_cum + values
        else:
            values_cum = values_cum + values_pertubed
    vales_cum = values_cum/float(len(values_list))
    if scale == "mm":
        values = values*1000.
    if values_stations_list is not None:
        values_stations = values_stations_list[0]
        if scale == "mm":
            values_stations = values_stations*1000.

    if latlon is False:
        axes.set_xlim(easts.min()/km, easts.max()/km)
        axes.set_ylim(norths.min()/km, norths.max()/km)

        axes.set_xlabel('Easting [km]')
        axes.set_ylabel('Northing [km]')

        im = axes.contourf(
            easts/km, norths/km, values,
            vmin=0., vmax=vmax,
            cmap=plt.get_cmap('YlOrBr'))

        if quantity == "velocity":
            if scale == "mm":
                fig.colorbar(im, label='Velocity [mm/s]')
            else:
                fig.colorbar(im, label='Velocity [m/s]')
        if quantity == "acceleration":
            if scale == "mm":
                fig.colorbar(im, label='Velocity [mm/s]')
            else:
                fig.colorbar(im, label='Velocity [m/s]')

        if source.base_key()[6] is not "ExplosionSource":
            beachball.plot_fuzzy_beachball_mpl_pixmap(
                mts, axes, best_mt,
                position=(0, 0.),
                color_t='black',
                **plot_kwargs)
        if values_stations is not None:
            plt.scatter(easts_stations/km, norths_stations/km,
                        c=values_stations, s=36, cmap=plt.get_cmap('YlOrBr'),
                        vmin=0., vmax=vmax, edgecolor="k")
    else:
        lats = []
        lons = []
        for east, north in zip(easts, norths):
            lat, lon = orthodrome.ne_to_latlon(source.lat, source.lon,
                                               north, east)
            lats.append(lat)
            lons.append(lon)

        if vs30 is True:
            from silvertine.shakemap import vs30
            values_vs30 = vs30.extract_rectangle(num.min(lons), num.max(lons), num.min(lats), num.max(lats))
            from scipy import ndimage
            factor_x = num.shape(values)[0]/num.shape(values_vs30)[0]
            factor_y = num.shape(values)[1]/num.shape(values_vs30)[1]
            values_vs30_resam = ndimage.zoom(values_vs30, (factor_x, factor_y))

            store = engine.get_store(store_id)
            if type_factors is None:
                layer0 = store.config.earthmodel_1d.layer(0)
                base_velocity = layer0.mtop.vs
                base_velocity = 400.
                type_factors = 1
                amp_factor = (base_velocity/values_vs30_resam)**type_factors
                values = values*amp_factor
                values = values/15.87
                #values = values/10.

        if plot_background_map is True:
            map = Basemap(projection='merc',
                          llcrnrlon=num.min(lons),
                          llcrnrlat=num.min(lats),
                          urcrnrlon=num.max(lons),
                          urcrnrlat=num.max(lats),
                          resolution='h', epsg=3395)
            ratio_lat = num.max(lats)/num.min(lats)
            ratio_lon = num.max(lons)/num.min(lons)

            map.drawmapscale(num.min(lons)+0.05,
                             num.min(lats)+0.05,
                             num.min(lons), num.min(lats), 10)
            parallels = num.arange(num.around(num.min(lats), decimals=2),
                                   num.around(num.max(lats), decimals=2), 0.1)
            meridians = num.arange(num.around(num.min(lons), decimals=2),
                                   num.around(num.max(lons), decimals=2), 0.2)
            map.drawparallels(parallels, labels=[1, 0, 0, 0], fontsize=22)
            map.drawmeridians(meridians, labels=[1, 1, 0, 1], fontsize=22)
            xpixels = 1000
            try:
                map.arcgisimage(service='World_Shaded_Relief', xpixels=xpixels,
                                verbose=False, zorder=1, cmap="gray")
            except urllib.error.URLError:
                pass
    #    axes.set_xlim(lats.min()/km, lats.max()/km)
    #    axes.set_ylim(norths.min()/km, norths.max()/km)
        if plot_background_map is True:
            pos1, pos2 = map(source.lon, source.lat)
        else:
            pos1 = source.lat
            pos2 = source.lon
        if source.base_key()[6] is not "ExplosionSource":
            beachball.plot_fuzzy_beachball_mpl_pixmap(
                mts, axes, best_mt,
                position=(pos1, pos2),
                color_t='black',
                zorder=2,
                **plot_kwargs)

        if plot_background_map is True:
            lats_map, lons_map = map(lons, lats)
            values[values == 0] = 'nan'
            alpha = 0.5
        else:
            lats_map, lons_map = lats, lons
            alpha = 1.
        _, vmax = num.min(values), num.max(values)

        if values_stations_list is not None:
            st_lats, st_lons = [], []
            for st in stations:
                st_lats.append(st.lat)
                st_lons.append(st.lon)
            if plot_background_map is True:
                st_lats, st_lons = map(st_lons, st_lats)
            if plot_values is True and measured is not None:
                measured_values = []
            try:
                for k, st in enumerate(stations):
                    for data in measured:
                            if data[0].decode() == st.station:
                                if scale == "mm":
                                    measured_values.append(data[1]*1000.)
                _, vmax = num.min(measured_values), num.max(measured_values)

            except:
                measured_values = []
                if measured is not None:
                    for data in measured:
                        if scale == "mm":
                        #    measured_values.append(data*1000.*100)
                            measured_values.append(data*1000.)

                    _, vmax = num.min(measured_values), num.max(measured_values)

            if plot_values is True and measured is None:
                plt.scatter(st_lats, st_lons,
                            c=num.asarray(values_stations), s=36,
                            cmap=plt.get_cmap('YlOrBr'),
                            vmin=0., vmax=vmax, edgecolor="k", alpha=alpha)
                for k, st in enumerate(stations):
                    plt.text(st_lats[k], st_lons[k], str(st.station))
            if plot_values is True and measured is not None:
                plt.scatter(st_lats, st_lons,
                            c=num.asarray(measured_values), s=36,
                            cmap=plt.get_cmap('YlOrBr'),
                            vmin=0., vmax=vmax, edgecolor="k", alpha=alpha)
                for k, st in enumerate(stations):
                    plt.text(st_lats[k], st_lons[k], str(st.station))
            if measured is not None and plot_values is False:
                residuals = []
                stations_write = []
                try:
                    for k, st in enumerate(stations):
                        for data in measured:
                            # case for manual input
                            if data[0].decode() == st.station:
                                residuals.append(values_stations[k]-data[1])
                                stations_write.append(st.station)
                except:
                    if measured is not None:
                        for data in measured:
                            # case for measured input
                            residuals.append(values_stations[k]-data)
                            stations_write.append(st.station)
                fobj = open(os.path.join(folder, 'residuals.txt'), 'w')
                for i in range(0, len(residuals)):
                    fobj.write('%s %.20f\n' % (stations_write[i],
                                               residuals[i]))
                fobj.close()

                fobj = open(os.path.join(folder, 'synthetic_pga.txt'), 'w')
                for i in range(0, len(residuals)):
                    fobj.write('%s %.20f\n' % (stations_write[i],
                                               values_stations[i]))
                fobj.close()

                plt.scatter(st_lats, st_lons,
                            c=residuals, s=36, cmap=plt.get_cmap('YlOrBr'),
                            vmin=0., vmax=vmax, edgecolor="k", alpha=alpha)
        #values = values*77.66
        im = axes.contourf(
            lats_map, lons_map, values.T,
            vmin=0., vmax=vmax,
            cmap=plt.get_cmap('YlOrBr'),
            alpha=alpha)
        if quantity == "velocity":
            if scale == "mm":
                #fig.colorbar(im, label='Velocity [mm/s]')
                m = plt.cm.ScalarMappable(cmap=plt.get_cmap('YlOrBr'))
                m.set_array(values)
                m.set_clim(0., vmax)
                plt.colorbar(m, boundaries=num.linspace(0, vmax, 6))
            else:
                fig.colorbar(im, label='Velocity [m/s]')
        if quantity == "acceleration":
            if scale == "mm":
                fig.colorbar(im, label='Velocity [mm/s]')
            else:
                fig.colorbar(im, label='Velocity [m/s]')


        axes.contour(lats_map, lons_map, vales_cum.T, cmap='brg',
                     levels=[value_level])

    fig.savefig(folder+filename)
    if show is True:
        plt.show()
    else:
        plt.close()
    if vs30 is not False:
        fig = plt.figure(figsize=plot.mpl_papersize('a5', 'landscape'))
        axes = fig.add_subplot(1, 1, 1, aspect=1.0)

        map = Basemap(projection='merc',
                      llcrnrlon=num.min(lons),
                      llcrnrlat=num.min(lats),
                      urcrnrlon=num.max(lons),
                      urcrnrlat=num.max(lats),
                      resolution='h', epsg=3395)
        ratio_lat = num.max(lats)/num.min(lats)
        ratio_lon = num.max(lons)/num.min(lons)

        parallels = num.arange(num.around(num.min(lats), decimals=2),
                               num.around(num.max(lats), decimals=2), 0.1)
        meridians = num.arange(num.around(num.min(lons), decimals=2),
                               num.around(num.max(lons), decimals=2), 0.2)
        map.drawparallels(parallels, labels=[1, 0, 0, 0], fontsize=22)
        map.drawmeridians(meridians, labels=[1, 1, 0, 1], fontsize=22)
        im = map.imshow(num.rot90(values_vs30.T))
        fig.colorbar(im, label='Velocity [m/s]')
        fig.savefig(folder+"vs30.png")



def coords_2d(norths, easts):
    norths2 = num.repeat(norths, easts.size)
    easts2 = num.tile(easts, norths.size)
    return norths2, easts2


class BruneResponse(trace.FrequencyResponse):

    duration = Float.T()

    def evaluate(self, freqs):
        return 1.0 / (1.0 + (freqs*self.duration)**2)
