import logging
import os.path as op

from pyrocko.guts import String, Float, Timestamp, List, Bool, Int
from pyrocko import model, guts
from pyrocko.io import stationxml
from pyrocko.gf import TPDef

from silvertine.seiger_lassie import receiver, ifc, grid, geo
from silvertine.seiger_lassie.common import Earthmodel, HasPaths, Path, LassieError, \
    expand_template

guts_prefix = 'seiger_lassie'

logger = logging.getLogger('seiger_lassie.config')


class Config(HasPaths):

    stations_path = Path.T(
        optional=True,
        help='stations file in Pyrocko format')

    stations_stationxml_path = Path.T(
        optional=True,
        help='stations file in StationXML format')

    receivers = List.T(
        receiver.Receiver.T(),
        help='receiver coordinates if not read from file')

    data_paths = List.T(
        Path.T(),
        help='list of directories paths to search for data')

    events_path = Path.T(
        optional=True,
        help='limit processing to time windows around given events')

    event_time_window_factor = Float.T(
        default=2.,
        help='controls length of time windows for event-wise processing')

    blacklist = List.T(
        String.T(),
        help='codes in the form NET.STA.LOC of receivers to be excluded')

    whitelist = List.T(
        String.T(),
        help='codes in the form NET.STA.LOC of receivers to be included')

    distance_max = Float.T(
        optional=True,
        help='receiver maximum distance from grid')

    tmin = Timestamp.T(
        optional=True,
        help='beginning of time interval to be processed')

    tmax = Timestamp.T(
        optional=True,
        help='end of time interval to be processed')

    run_path = Path.T(
        optional=True,
        help='output is saved to this directory')

    save_figures = Bool.T(
        default=False,
        help='flag to activate saving of detection figures')

    grid = grid.Grid.T(
        optional=True,
        help='definition of search grid, if not given, a default grid is '
             'chosen')

    autogrid_radius_factor = Float.T(
        default=1.5,
        help='size factor to use when automatically choosing a grid size')

    autogrid_density_factor = Float.T(
        default=10.0,
        help='grid density factor used when automatically choosing a grid '
             'spacing')

    image_function_contributions = List.T(
        ifc.IFC.T(),
        help='image function contributions')

    sharpness_normalization = Bool.T(
        default=True,
        help='whether to divide image function frames by their mean value')

    ifc_count_normalization = Bool.T(
        default=False,
        help='whether to divide image function by number of contributors')

    detector_threshold = Float.T(
        default=70.,
        help='threshold on detector function')

    detector_tpeaksearch = Float.T(
        optional=True,
        help='search time span for peak detection')

    fill_incomplete_with_zeros = Bool.T(
        default=True,
        help='fill incomplete trace time windows with zeros '
             '(and let edge effects ruin your day)')

    earthmodels = List.T(
        Earthmodel.T(),
        help='list of earthmodels usable in shifters')

    tabulated_phases = List.T(
        TPDef.T(),
        help='list of tabulated phase definitions usable shifters')

    cache_path = Path.T(
        default='lassie_phases.cache',
        help='directory where lassie stores tabulated phases etc.')

    stacking_blocksize = Int.T(
        optional=True,
        help='enable chunked stacking to reduce memory usage. Setting this to '
             'e.g. 64 will use ngridpoints * 64 * 8 bytes of memory to hold '
             'the stacking results, instead of computing the whole processing '
             'time window in one shot. Setting this to a very small number '
             'may lead to bad performance. If this is enabled together with '
             'plotting, the cutout of the image function seen in the map '
             'image must be stacked again just for plotting (redundantly and '
             'memory greedy) because it may intersect more than one '
             'processing chunk.')

    def __init__(self, *args, **kwargs):
        HasPaths.__init__(self, *args, **kwargs)
        self._receivers = None
        self._grid = None
        self._events = None
        self._config_name = 'untitled'

    def setup_image_function_contributions(self):
        '''
        Post-init setup of image function contributors.
        '''
        for ifc_ in self.image_function_contributions:
            ifc_.setup(self)

    def set_config_name(self, config_name):
        self._config_name = config_name

    def expand_path(self, path):
        def extra(path):
            return expand_template(path, dict(
                config_name=self._config_name))

        return HasPaths.expand_path(self, path, extra=extra)

    def get_events_path(self):
        run_path = self.expand_path(self.run_path)
        return op.join(run_path, 'events.list')

    def get_ifm_dir_path(self):
        run_path = self.expand_path(self.run_path)
        return op.join(run_path, 'ifm')

    def get_ifm_path_template(self):
        return op.join(
            self.get_ifm_dir_path(),
            '%(station)s_%(tmin_ms)s.mseed')

    def get_detections_path(self):
        run_path = self.expand_path(self.run_path)
        return op.join(run_path, 'detections.list')

    def get_figures_path_template(self):
        run_path = self.expand_path(self.run_path)
        return op.join(run_path, 'figures', 'detection_%(id)s.%(format)s')

    def get_receivers(self):
        '''Aggregate receivers from different sources.'''

        fp = self.expand_path

        if self._receivers is None:
            self._receivers = list(self.receivers)
            if self.stations_path:
                for station in model.load_stations(fp(self.stations_path)):
                    self._receivers.append(
                        receiver.Receiver(
                            codes=station.nsl(),
                            lat=station.lat,
                            lon=station.lon,
                            z=station.depth))

            if self.stations_stationxml_path:
                sx = stationxml.load_xml(filename=fp(self.stations_stationxml_path))
                for station in sx.get_pyrocko_stations():
                    self._receivers.append(
                        receiver.Receiver(
                            codes=station.nsl(),
                            lat=station.lat,
                            lon=station.lon,
                            z=station.depth))

        return self._receivers

    def get_events(self):
        if self.events_path is None:
            return None

        if self._events is None:
            self._events = model.load_events(self.expand_path(
                self.events_path))

        return self._events

    def get_grid(self):
        '''Get grid or make default grid.'''

        self.setup_image_function_contributions()

        if self._grid is None:

            if not self.grid:
                receivers = self.get_receivers()

                fsmooth_max = max(
                    ifc.get_fsmooth() for ifc in
                    self.image_function_contributions)

                vmin = min(ifc.shifter.get_vmin() for ifc in
                           self.image_function_contributions)

                spacing = vmin / fsmooth_max / self.autogrid_density_factor

                lat0, lon0, north, east, depth = geo.bounding_box_square(
                    *geo.points_coords(receivers),
                    scale=self.autogrid_radius_factor)

                self._grid = grid.Carthesian3DGrid(
                    lat=lat0,
                    lon=lon0,
                    xmin=north[0],
                    xmax=north[1],
                    dx=spacing,
                    ymin=east[0],
                    ymax=east[1],
                    dy=spacing,
                    zmin=depth[0],
                    zmax=depth[1],
                    dz=spacing)

                logger.info('automatic grid:\n%s' % self._grid)

            else:
                self._grid = self.grid

            self._grid.update()

        return self._grid


def read_config(path):
    config = guts.load(filename=path)
    if not isinstance(config, Config):
        raise LassieError('invalid Lassie configuration in file "%s"' % path)

    config.set_basepath(op.dirname(path) or '.')
    config.set_config_name(op.splitext(op.basename(path))[0])

    return config


def write_config(config, path):
    basepath = config.get_basepath()
    dirname = op.dirname(path) or '.'
    config.change_basepath(dirname)
    guts.dump(config, filename=path)
    config.change_basepath(basepath)


__all__ = [
    'Config',
    'read_config',
]
