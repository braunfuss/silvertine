#!/usr/bin/env python

import sys
import logging
from optparse import OptionParser

from pyrocko import util
import multiprocessing
from silvertine import seiger_lassie as lassie

logger = logging.getLogger('main')


def d2u(d):
    if isinstance(d, dict):
        return dict((k.replace('-', '_'), v) for (k, v) in d.items())
    else:
        return d.replace('-', '_')


def str_to_time(s):
    try:
        return util.str_to_time(s)
    except util.TimeStrError as e:
        raise lassie.LassieError(str(e))


subcommand_descriptions = {
    'init': 'create initial configuration file',
    'search': 'detect seismic events',
    'map-geometry': 'make station map',
    'snuffle': 'snuffle'
}

subcommand_usages = {
    'init': 'init',
    'search': 'search <configfile> [options]',
    'map-geometry': 'map-geometry <configfile> [options] <output.(png|pdf)',
    'snuffle': 'snuffle <configfile>',
}

subcommands = subcommand_descriptions.keys()

program_name = 'seiger-lassie'

usage_tdata = d2u(subcommand_descriptions)
usage_tdata['program_name'] = program_name

usage = program_name + ''' <subcommand> [options] [--] <arguments> ...

Subcommands:

    init          %(init)s
    search        %(search)s
    map-geometry  %(map_geometry)s
    snuffle       %(snuffle)s

To get further help and a list of available options for any subcommand run:

    %(program_name)s <subcommand> --help

''' % usage_tdata


def add_common_options(parser):
    parser.add_option(
        '--loglevel',
        action='store',
        dest='loglevel',
        type='choice',
        choices=('critical', 'error', 'warning', 'info', 'debug'),
        default='info',
        help='set logger level to '
             '"critical", "error", "warning", "info", or "debug". '
             'Default is "%default".')


def process_common_options(options):
    util.setup_logging(program_name, options.loglevel)


def cl_parse(command, args, setup=None):
    usage = subcommand_usages[command]
    descr = subcommand_descriptions[command]

    if isinstance(usage, str):
        usage = [usage]

    susage = '%s %s' % (program_name, usage[0])
    for s in usage[1:]:
        susage += '\n%s%s %s' % (' '*7, program_name, s)

    parser = OptionParser(
        usage=susage,
        description=descr[0].upper() + descr[1:] + '.')

    if setup:
        setup(parser)

    add_common_options(parser)
    (options, args) = parser.parse_args(args)
    process_common_options(options)
    return parser, options, args


def die(message, err=''):
    if err:
        sys.exit('%s: error: %s \n %s' % (program_name, message, err))
    else:
        sys.exit('%s: error: %s' % (program_name, message))


def help_and_die(parser, message):
    parser.print_help(sys.stderr)
    sys.stderr.write('\n')
    die(message)


def escape(s):
    return s.replace("'", "\\'")


def command_init(args):
    def setup(parser):
        parser.add_option(
            '--stations', dest='stations_path',
            metavar='PATH',
            help='stations file')

        parser.add_option(
            '--data', dest='data_paths',
            default=[],
            action='append',
            metavar='PATH',
            help='data directory (this option may be repeated)')

    parser, options, args = cl_parse('init', args, setup=setup)

    if options.data_paths:
        s_data_paths = '\n'.join(
            "- '%s'" % escape(x) for x in options.data_paths)
    else:
        s_data_paths = "- 'DATA'"

    if options.stations_path:
        stations_path = options.stations_path
    else:
        stations_path = 'STATIONS_PATH'

    print('''%%YAML 1.1
--- !lassie.Config

## Configuration file for Lassie, your friendly earthquake detector
##
## Receiver coordinates can be read from a stations file in Pyrocko format:
stations_path: '%(stations_path)s'

## Receivers can also be listed in the config file, lat/lon and carthesian
## (x/y/z) = (North/East/Down) coordinates are supported and may be combined
## (interpreted as reference + offset). Omitted values are treated as zero.
# receivers:
# - !lassie.Receiver
#   codes: ['', 'ACC13', '']
#   lat: 10.
#   lon: 12.
#   x: 2397.56
#   y: 7331.94
#   z: -404.1

## List of data directories. Lassie will recurse into subdirectories to find
## all contained waveform files.
data_paths:
%(s_data_paths)s

## name template for Lassie's output directory. The placeholder
## "${config_name}" will be replaced with the basename of the config file.
run_path: '${config_name}.turd'

## Processing time interval (default: use time interval of available data)
# tmin: '2012-02-06 04:20:00'
# tmax: '2012-02-06 04:30:00'

## Search grid; if not given here (if commented), a default grid will be chosen
# grid: !lassie.Carthesian3DGrid
#   lat: 38.7
#   lon: -7.9
#   xmin: -70e3  # in [m]
#   xmax: 70e3
#   ymin: -70e3
#   ymax: 70e3
#   zmin: 0.0
#   zmax: 0.0
#   dx: 2.5e3
#   dy: 2.5e3
#   dz: 2.5e3

## Size factor to use when automatically choosing a grid size
autogrid_radius_factor: 1.5

## Grid density factor used when automatically choosing a grid
autogrid_density_factor: 10.0

## Composition of image function
image_function_contributions:

- !lassie.OnsetIFC
  name: 'P'
  weight: 30.0
  fmin: 1.0
  fmax: 15.0
  short_window: 1.0
  window_ratio: 8.0
  fsmooth: 0.2
  fnormalize: 0.02
  #shifter: !lassie.VelocityShifter
  #  velocity: 6000.
  shifter: !lassie.CakePhaseShifter
    timing: '{stored:p}'
    earthmodel_id: 'swiss'

- !lassie.WavePacketIFC
  name: 'S'
  weight: 1.0
  fmin: 1.0
  fmax: 8.0
  fsmooth: 0.05
  #shifter: !lassie.VelocityShifter
  #  velocity: 3300.
  shifter: !lassie.CakePhaseShifter
    # factor: 1.0
    # offset: 1.0
    timing: '{stored:s}'
    earthmodel_id: 'swiss'

## Whether to divide image function frames by their mean value
sharpness_normalization: false

## Threshold on detector function
detector_threshold: 150.

## Whether to create a figure for every detection and save it in the output
## directory
save_figures: true

## Mapping of phase ID to phase definition in cake syntax (used e.g. in the
## CakePhaseShifter config sections)
tabulated_phases:
- !pf.TPDef
  id: 'p'
  definition: 'P,p'
- !pf.TPDef
  id: 's'
  definition: 'S,s'

## Mapping of earthmodel ID  to the actual earth model in nd format (used in
## the CakePhaseShifter config sections)
earthmodels:
- !lassie.CakeEarthmodel
  id: 'swiss'
  earthmodel_1d: |2
    0.0 5.53 3.10  2.75
    2.0 5.53 3.10  2.75
    2.0 5.80 3.25  2.75
    5.0 5.80 3.25  2.75
    5.0 5.83 3.27  2.75
    8.0 5.83 3.27  2.75
    8.0 5.95 3.34  2.8
    13.0 5.95 3.34  2.8
    13.0 5.96 3.34  2.8
    22.0 5.96 3.34  2.8
    22.0 6.53 3.66  2.8
    30.0 6.53 3.66  2.8
    30.0 7.18 4.03 3.3
    40.0 7.18 4.03 3.3
    40.0 7.53 4.23 3.3
    50.0 7.53 4.23 3.3
    50.0 7.83 4.39 3.3
    60.0 7.83 4.39 3.3
    60.0 8.15 4.57 3.3
    120.0 8.15 4.57 3.3
''' % dict(
        stations_path=stations_path,
        s_data_paths=s_data_paths))


def command_search(args):
    def setup(parser):
        parser.add_option(
            '--force', dest='force', action='store_true',
            help='overwrite existing files')

        parser.add_option(
            '--show-detections', dest='show_detections', action='store_true',
            help='show plot for every detection found')

        parser.add_option(
            '--show-movie', dest='show_movie', action='store_true',
            help='show movie when showing detections')

        parser.add_option(
            '--show-window-traces', dest='show_window_traces',
            action='store_true',
            help='show preprocessed traces for every processing time window')

        parser.add_option(
            '--stop-after-first', dest='stop_after_first', action='store_true',
            help='show plot for every detection found')

        parser.add_option(
            '--tmin', dest='tmin', metavar="'YYYY-MM-DD HH:MM:SS.XXX'",
            help='beginning of processing time window '
                 '(overrides config file settings)')

        parser.add_option(
            '--tmax', dest='tmax', metavar="'YYYY-MM-DD HH:MM:SS.XXX'",
            help='end of processing time window '
                 '(overrides config file settings)')

        parser.add_option(
            '--nworkers', dest='nworkers', metavar="N",
            help='use N cpus in parallel')

        parser.add_option(
            '--speak', dest='bark', action='store_true',
            help='alert on detection of events')

    parser, options, args = cl_parse('search', args, setup=setup)
    if len(args) != 1:
        help_and_die(parser, 'missing argument')

    config_path = args[0]
    config = lassie.read_config(config_path)
    try:
        tmin = tmax = None

        if options.tmin:
            tmin = str_to_time(options.tmin)

        if options.tmax:
            tmax = str_to_time(options.tmax)

        if options.nworkers:
            nparallel = int(options.nworkers)
        else:
            nparallel = multiprocessing.cpu_count()

        lassie.search(
            config,
            override_tmin=tmin,
            override_tmax=tmax,
            force=options.force,
            show_detections=options.show_detections,
            show_movie=options.show_movie,
            show_window_traces=options.show_window_traces,
            stop_after_first=options.stop_after_first,
            nparallel=nparallel,
            bark=options.bark)

    except lassie.LassieError as e:
        die(str(e))


def command_map_geometry(args):
    parser, options, args = cl_parse('map-geometry', args)
    if len(args) != 2:
        help_and_die(parser, 'missing arguments')

    config_path = args[0]
    output_path = args[1]
    config = lassie.read_config(config_path)
    lassie.map_geometry(config, output_path)


def command_snuffle(args):
    parser, options, args = cl_parse('snuffle', args)
    if len(args) != 1:
        help_and_die(parser, 'missing arguments')

    config_path = args[0]
    config = lassie.read_config(config_path)

    lassie.snuffle(config)


if __name__ == '__main__':
    main()


def main():
    usage_sub = 'fomosto %s [options]'
    if len(sys.argv) < 2:
        sys.exit('Usage: %s' % usage)

    args = list(sys.argv)
    args.pop(0)
    command = args.pop(0)

    if command in subcommands:
        globals()['command_' + d2u(command)](args)

    elif command in ('--help', '-h', 'help'):
        if command == 'help' and args:
            acommand = args[0]
            if acommand in subcommands:
                globals()['command_' + acommand](['--help'])

        sys.exit('Usage: %s' % usage)

    else:
        die('no such subcommand: %s' % command)
