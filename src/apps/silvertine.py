from __future__ import print_function, absolute_import
import sys
from os.path import join as pjoin
import os.path as op
import os
from multiprocessing import Pool
from multiprocessing import Process
import logging
from optparse import OptionParser, OptionValueError, IndentedHelpFormatter
from io import StringIO
import tempfile
import time
from pathlib import Path
import shutil
import ray
from silvertine.setup_info import version as __version__
import silvertine
from silvertine.util.parser_setup import *
from silvertine.util import ref_mods
from obspy import read_events
import obspy.io.seiscomp.event as sc3
from obspy.io.quakeml.core import _write_quakeml
from tensorflow.keras import backend as K
import gc
import threading
from silvertine.locate.grond import locate as glocate

# from gevent import monkey
# monkey.patch_all()
try:
    from pyrocko import util, marker, model
    from pyrocko import pile as pile_mod
    from pyrocko import orthodrome as ort
    from pyrocko.gui.snuffler_app import *
    from pyrocko.io import quakeml
    from pyrocko.gui.pile_viewer import PhaseMarker, EventMarker
    from pyrocko.gui import marker as pm
except ImportError:
    print(
        "Pyrocko is required for silvertine!"
        "Go to https://pyrocko.org/ for installation instructions."
    )


logger = logging.getLogger("silvertine.main")
km = 1e3

def add_common_options(parser):
    parser.add_option(
        "--loglevel",
        action="store",
        dest="loglevel",
        type="choice",
        choices=("critical", "error", "warning", "info", "debug"),
        default="info",
        help="set logger level to "
        '"critical", "error", "warning", "info", or "debug". '
        'Default is "%default".',
    )

    parser.add_option("--docs", dest="rst_docs", action="store_true")


def print_docs(command, parser):
    class DocsFormatter(IndentedHelpFormatter):
        def format_heading(self, heading):
            return "%s\n%s\n\n" % (heading, "." * len(heading))

        def format_usage(self, usage):
            lines = usage.splitlines()
            return self.format_heading(
                "Usage"
            ) + ".. code-block:: none\n\n%s" % "\n".join(
                "    " + line.strip() for line in lines
            )

        def format_option(self, option):
            if not option.help:
                return ""

            result = []
            opts = self.option_strings[option]
            result.append("\n.. describe:: %s\n\n" % opts)

            help_text = self.expand_default(option)
            result.append("    %s\n\n" % help_text)

            return "".join(result)

    parser.formatter = DocsFormatter()
    parser.formatter.set_parser(parser)

    def format_help(parser):
        formatter = parser.formatter
        result = []

        result.append(parser.format_description(formatter) + "\n")

        if parser.usage:
            result.append(parser.get_usage() + "\n")

        result.append("\n")

        result.append(parser.format_option_help(formatter))

        result.append("\n")

        result.append(parser.format_epilog(formatter))
        return "".join(result)

    print(command)
    print("-" * len(command))
    print()
    print(".. program:: %s" % program_name)
    print()
    print(".. option:: %s" % command)
    print()
    print(format_help(parser))


def process_common_options(command, parser, options):
    util.setup_logging(program_name, options.loglevel)
    if options.rst_docs:
        print_docs(command, parser)
        exit(0)


def cl_parse(command, args, setup=None, details=None):
    usage = subcommand_usages[command]
    descr = subcommand_descriptions[command]

    if isinstance(usage, str):
        usage = [usage]

    susage = "%s %s" % (program_name, usage[0])
    for s in usage[1:]:
        susage += "\n%s%s %s" % (" " * 7, program_name, s)

    description = descr[0].upper() + descr[1:] + "."

    if details:
        description = description + "\n\n%s" % details

    parser = OptionParser(usage=susage, description=description)

    if setup:
        setup(parser)

    add_common_options(parser)
    (options, args) = parser.parse_args(args)
    process_common_options(command, parser, options)
    return parser, options, args


def die(message, err="", prelude=""):
    if prelude:
        prelude = prelude + "\n"

    if err:
        err = "\n" + err

    sys.exit("%s%s failed: %s%s" % (prelude, program_name, message, err))


def help_and_die(parser, message):
    sio = StringIO()
    parser.print_help(sio)
    die(message, prelude=sio.getvalue())


def multiple_choice(option, opt_str, value, parser, choices):
    options = value.split(",")
    for opt in options:
        if opt not in choices:
            raise OptionValueError(
                "Invalid option %s - valid options are: %s" % (opt, ", ".join(choices))
            )
    setattr(parser.values, option.dest, options)

subcommand_descriptions = {
    "init": "initialise new project structure or print configuration",
    "scenario": "create a forward-modelled scenario project",
    "locate": "locate a single or a set of earthquakes",
    "post_shakemap": "post_shakemap a single or a set of earthquakes",
    "plot_prod": "plot_prod a single or a set of earthquakes",
    "pertub_earthmodels": "Pertub earthmodels",
    "plot_mods": "plot_mods a single or a set of earthquakes",
    "analyse_statistics": "command_analyse_statistics",
    "optimize": "optimize",
    "monitor": "Monitor stations",
    "beam": "beamform for a single or a set of earthquakes",
    "beam_process": "beam_process for a single or a set of earthquakes",
    "events": "print available event names for given configuration",
    "check": "check data and configuration",
    "detect": "detect",
    "go": "run silvertine optimisation",
    "supply": "deliver waveforms",
    "download_raw": "download waveforms",
    "report": "create result report",
    "shmconv": "Seismic handler compat",
    "diff": "compare two configs or other normalized silvertine YAML files",
    "upgrade-config": "upgrade config file to the latest version of silvertine",
    "version": "print version number of silvertine and its main dependencies",
}

subcommand_usages = {
    "init": (
        "init list [options]",
        "init <example> [options]",
        "init <example> <projectdir> [options]",
    ),
    "scenario": "scenario [options] <projectdir>",
    "locate": "locate [options] <projectdir>",
    "post_shakemap": "post_shakemap [options] <projectdir>",
    "plot_prod": "plot_prod [options] <projectdir>",
    "pertub_earthmodels": "pertub_earthmodels [options]",
    "plot_mods": "plot_mods [options] <projectdir>",
    "analyse_statistics": "command_analyse_statistics [options] <projectdir>",
    "detect": "detect [options] <projectdir>",
    "optimize": "optimize [options] <projectdir>",
    "monitor": "monitor [options] <projectdir>",
    "events": "events <configfile>",
    "supply": "tmin tmax",
    "download_raw": "tmin tmax",
    "shmconv": "task file",
    "beam": "beams [options] <projectdir>",
    "beam_process": "beam_process [options] <projectdir>",
    "check": "check <configfile> <eventnames> ... [options]",
    "report": ("report <rundir> ... [options]", "report <configfile> <eventnames> ..."),
    "diff": "diff <left_path> <right_path>",
    "upgrade-config": "upgrade-config <configfile>",
    "version": "version",
}

subcommands = subcommand_descriptions.keys()

program_name = "silvertine"

usage_tdata = d2u(subcommand_descriptions)
usage_tdata["program_name"] = program_name

usage_tdata["version_number"] = __version__


usage = (
    """%(program_name)s <subcommand> [options] [--] <arguments> ...

silvertine is a framework for handling seiger related projects.

This is silvertine version %(version_number)s.

Subcommands:

    scenario        %(scenario)s
    plot_prod     %(plot_prod)s
    plot_mods     %(plot_mods)s
    pertub_earthmodels     %(pertub_earthmodels)s
    analyse_statistics     %(analyse_statistics)s
    locate        %(locate)s
    post_shakemap %(post_shakemap)s
    detect        %(detect)s
    optimize      %(optimize)s
    monitor        %(monitor)s
    beam        %(beam)s
    supply        %(supply)s
    download_raw        %(download_raw)s
    beam_process        %(beam_process)s
    init            %(init)s
    events          %(events)s
    check           %(check)s
    go              %(go)s
    shmconv         %(shmconv)s
    report          %(report)s
    diff            %(diff)s
    upgrade-config  %(upgrade_config)s
    version         %(version)s

To get further help and a list of available options for any subcommand run:

    %(program_name)s <subcommand> --help

What do you want to bust today?!
"""
    % usage_tdata
)


class CLIHints(object):
    init = """
We created a folder structure in {project_dir}.
Check out the YAML configuration in {config} and start the optimisation by:

    silvertine go {config}
"""
    scenario = """
To start the scenario's optimisation, change to folder

    cd {project_dir}

Check out the YAML configuration in {config} and start the optimisation by:

    silvertine go {config}
"""
    report = """
To open the report in your web browser, run

    silvertine report -s --open {config}
"""
    check = """
To start the optimisation, run

    silvertine go {config}
"""
    go = """
To look at the results, run

    silvertine report -so {rundir}
"""

    def __new__(cls, command, **kwargs):
        return "{c.BOLD}Hint{c.END}\n".format(c=Color) + getattr(cls, command).format(
            **kwargs
        )


def main(args=None):
    if not args:
        args = sys.argv

    args = list(args)
    if len(args) < 2:
        sys.exit("Usage: %s" % usage)

    args.pop(0)
    command = args.pop(0)

    if command in subcommands:
        globals()["command_" + d2u(command)](args)

    elif command in ("--help", "-h", "help"):
        if command == "help" and args:
            acommand = args[0]
            if acommand in subcommands:
                globals()["command_" + acommand](["--help"])

        sys.exit("Usage: %s" % usage)

    else:
        die("No such subcommand: %s" % command)


def magnitude_range(option, opt_str, value, parser):
    mag_range = value.split("-")
    if len(mag_range) != 2:
        raise OptionValueError(
            "Invalid magnitude %s - valid range is e.g. 6-7." % value
        )
    try:
        mag_range = tuple(map(float, mag_range))
    except ValueError:
        raise OptionValueError("Magnitudes must be numbers.")

    if mag_range[0] > mag_range[1]:
        raise OptionValueError(
            "Minimum magnitude must be larger than" " maximum magnitude."
        )
    setattr(parser.values, option.dest, mag_range)


def command_shmconv(args):
    def setup(parser):
        parser.add_option(
            "--ttt", dest="ttt", type=str, default=False, help="Convert ttt"
        )

    parser, options, args = cl_parse("shmconv", args, setup)
    fname = args[0]

    if options.ttt is not False:
        options.ttt = True
    else:
        from silvertine import seismic_handler

        events, phase_markers = seismic_handler.convert_locsat_output.locsat2pyrocko(
            fname
        )


def command_monitor(args):
    def setup(parser):
        parser.add_option(
            "--addresses",
            dest="adresses",
            type=str,
            default=["eida.bgr.de", "eida.bgr.de"],
            help="Adresses",
        )
        parser.add_option(
            "--paths",
            dest="paths",
            type=str,
            default=["GR.INS*.*.EH*", "GR.TMO*.*.EH*"],
            help="Path of monitored stations",
        )
        parser.add_option(
            "--save",
            dest="save",
            type=str,
            default=False,
            help="Save waveforms",
        )
        parser.add_option(
            "--delay",
            dest="delay",
            type=float,
            default=50,
            help="Delay for pulling data",
        )
    parser, options, args = cl_parse("monitor", args, setup)

    from silvertine.monitoring import stream

    stream.live_steam(adresses=options.adresses, paths=options.paths,
                      delay=options.delay,
                      save=options.save)


def command_locate(args):
    def setup(parser):
        parser.add_option(
            "--show",
            dest="show",
            type=str,
            default=False,
            help="Display progress of localisation for each event in browser",
        )
        parser.add_option(
            "--nevents",
            dest="nevents",
            type=int,
            default=1,
            help="Number of events to locate (default: %default)",
        )
        parser.add_option(
            "--scenario",
            dest="scenario",
            type=str,
            default=True,
            help="Synthetic scenario",
        )
        parser.add_option(
            "--data_folder",
            dest="data_folder",
            type=str,
            default="data",
            help="Data folder for real data",
        )
        parser.add_option(
            "--parallel",
            dest="parallel",
            type=str,
            default=False,
            help="parallel location",
        )
        parser.add_option(
            "--singular",
            dest="singular",
            type=str,
            default=False,
            help="parallel location",
        )
        parser.add_option(
            "--adress",
            dest="adress",
            type=str,
            default=None,
            help="Adress of ray instance for cluster computation",
        )
        parser.add_option(
            "--model",
            dest="model",
            type=str,
            default="insheim",
            help="Name of the refrence model, if crust the appropiate crust\
                  model will be used",
        )
        parser.add_option(
            "--nboot",
            dest="nboot",
            type=int,
            default=1,
            help="Number of bootstrap results, based on different\
                  velocity models",
        )
        parser.add_option(
            "--minimum_vel",
            dest="minimum_vel",
            type=str,
            default=False,
            help="minimum 1d model",
        )
        parser.add_option(
            "--station_dropout",
            dest="t_station_dropout",
            type=str,
            default=False,
            help="t_station_dropout",
        )
        parser.add_option(
            "--reference",
            dest="reference",
            type=str,
            default="catalog",
            help="Use reference events, either catalog or hyposat",
        )
        parser.add_option(
            "--start",
            dest="start",
            type=int,
            default=0,
            help="Which event is the first?",
        )
        parser.add_option(
            "--hybrid",
            dest="hybrid",
            type=str,
            default=False,
            help="associate_waveforms",
        )

    parser, options, args = cl_parse("locate", args, setup)

    from silvertine.locate import locate1D

    project_dir = args[0]
    if options.show is not False:
        options.show = True
    if options.singular is not False:
        options.singular = True
    if options.parallel is not False:
        options.parallel = True
    if options.nevents == 0:
        options.nevents = None
    if options.scenario is not True:
        options.scenario = False
    if options.minimum_vel is not False:
        options.minimum_vel = True
    if options.t_station_dropout is not False:
        options.t_station_dropout = True
    if options.hybrid is not False:
        options.hybrid = True
    result, best_model = silvertine.locate.locate1D.solve(
        scenario_folder=project_dir,
        show=options.show,
        n_tests=options.nevents,
        scenario=options.scenario,
        data_folder=options.data_folder,
        parallel=options.parallel,
        adress=options.adress,
        singular=options.singular,
        mod_name=options.model,
        nboot=options.nboot,
        minimum_vel=options.minimum_vel,
        reference=options.reference,
        nstart=options.start,
        hybrid=options.hybrid,
    )


def process_event_data(args):
    try:
        event = model.load_events(args[0])[0]
        tmin = event.time - 30
        tmax = event.time + 30
        time = event.time
    except:
        time = args[0]
        tmin = time - 30
        tmax = time + 30

    subprocess.run(
        [
            "python3",
            "seiger_down.py",
            "--window='%s,%s'" % (tmin, tmax),
            "49.1",
            "8.1",
            "50.",
            "0.001",
            "50.",
            "event_%s" % (time),
            "--force",
        ]
    )


def str2bool(v):
    import argparse
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("Boolean value expected.")


def command_supply(args):
    def setup(parser):
        parser.add_option("--tmin", dest="tmin", type=str, default=None, help="begin")
        parser.add_option("--tmax", dest="tmax", type=str, default=None, help="end")

    parser, options, args = cl_parse("supply", args, setup)

    project_dir = args[0]
    from silvertine.util import download_raw

    download_raw.supply(watch_folder=project_dir, tmin=options.tmin, tmax=options.tmax)


def command_download_raw(args):
    def setup(parser):
        parser.add_option("--tmin", dest="tmin", type=str, default=None, help="begin")
        parser.add_option("--tmax", dest="tmax", type=str, default=None, help="end")
        parser.add_option("--freq", dest="freq", type=float, default=None, help="end")
        parser.add_option("--tinc", dest="tinc", type=float, default=None, help="end")
        parser.add_option(
            "--detector", dest="detector", type=str, default=False, help="end"
        )

    parser, options, args = cl_parse("download_raw", args, setup)

    project_dir = args[0]
    from silvertine.util import download_raw

    detector = str2bool(options.detector)
    if detector is True:
        clean = False
    else:
        clean = True
    if options.tinc is not None:
        tinc = float(options.tinc)
        iter = 0
        if int(int(util.stt(options.tmax) - util.stt(options.tmin)) / int(tinc)) == 0:
            nwindow = 1
        else:
            nwindow = int(
                int(util.stt(options.tmax) - util.stt(options.tmin)) / int(tinc)
            )
        for i in range(nwindow):
            twin_start = util.stt(options.tmin) + iter * tinc
            twin_end = util.stt(options.tmin) + tinc + iter * tinc
            try:
                download_raw.download_raw(
                    path=project_dir,
                    tmint=twin_start,
                    tmaxt=twin_end,
                    common_f=options.freq,
                    tinc=options.tinc,
                    detector=detector,
                    clean=clean,
                )
            except:
                pass
            iter = iter + 1
    else:
        download_raw.download_raw(
            path=project_dir,
            tmint=options.tmin,
            tmaxt=options.tmax,
            common_f=options.freq,
            detector=detector,
            clean=clean,
        )


def command_detect_run(args):
    def setup(parser):
        parser.add_option(
            "--detector",
            dest="detector",
            type=str,
            default="both",
            help="Detect with both the EQT and the stacking based approaches\
                  or only one")


def command_collect_detections(args):
    def setup(parser):

        parser.add_option(
            "--mode",
            dest="detector_modes",
            type=str,
            default="both",
            help="From which detector to collect information",
        )
        from pyrocko import model
        import glob

        events_collected = []
        for filename in glob.iglob(args[0] + "**/events.pf", recursive=True):
            events = model.load_events(filename)
            events_collected.extend(events)


def command_locate_with_grond(marker_file, gf_stores_path, scenario_dir, config_path, stations_path, event_name, evqml, qml, event_stack, event_marker, savedir, store_path_reader):
    best = glocate(marker_file, gf_stores_path, scenario_dir, config_path, stations_path, event_name)

    lat, lon = ort.ne_to_latlon(best.lat, best.lon, best.north_shift, best.east_shift)
    evqml.preferred_origin.latitude = quakeml.RealQuantity(value=float(lat))
    evqml.preferred_origin.longitude = quakeml.RealQuantity(value=float(lon))
    evqml.preferred_origin.time = quakeml.TimeQuantity(value=best.time)
    evqml.preferred_origin.depth = quakeml.RealQuantity(value=best.depth)
    public_id = "quakeml:seiger/" + str(int(float(event_stack.name.split()[1].replace(')','').replace('(',''))))+"_"+str(best.time)
    event_marker.lat = lat
    event_marker.lon = lon
    event_marker.depth = best.depth
    event_marker.time = best.time
    model.dump_events([event_marker], savedir+"/event.pf")

    evqml.public_id = public_id
    for pick in evqml.pick_list:
        rand = pick.public_id[-5]
        pick.public_id = public_id+"_"+rand+"_"+pick.phase_hint.value
        arrival = quakeml.Arrival(public_id = pick.public_id+"_arrival", pick_id = pick.public_id, phase=quakeml.Phase(value=pick.phase_hint.value))
        evqml.origin_list[0].arrival_list.append(arrival)
    evqml.preferred_origin_id = public_id+"_"+rand+"O"
    evqml.origin_list[0].public_id = public_id+"_"+rand+"O"
    qml.event_parameters.event_list = [evqml]

    qml.dump_xml(filename=savedir+"event_combined.qml")
    evs = read_events(savedir+"event_combined.qml")
    for ev in evs:
        sc3._write_sc3ml(evs, store_path_reader+"/LI_SC_%s.qml" % best.time)
        _write_quakeml(evs, store_path_reader+"/LI_%s.qml" % best.time)


def command_detect(args):
    from silvertine import detector, locate
    from silvertine import seiger_lassie as lassie
    from silvertine.util import waveform
    from pyrocko import model

    def setup(parser):
        parser.add_option(
            "--download_method",
            dest="download_method",
            type=str,
            default="stream",
            help="Choose download method",
        )
        parser.add_option(
            "--mode",
            dest="mode",
            type=str,
            default="both",
            help="Mode to to use"
        )
        parser.add_option(
            "--download",
            dest="download",
            type=str,
            default=False,
            help="Download data.",)
        parser.add_option(
            "--tinc",
            dest="tinc",
            type=float,
            default=None,
            help="Increment.")
        parser.add_option(
            "--path",
            dest="path",
            type=str,
            default=None,
            help="path")
        parser.add_option(
            "--data_dir",
            dest="data_dir",
            type=str,
            default=None,
            help="data_dir")
        parser.add_option(
            "--tmin",
            dest="tmin",
            type=str,
            default=None,
            help="begin")
        parser.add_option(
            "--sds",
            dest="sds",
            type=str,
            default=None,
            help="sds")
        parser.add_option(
            "--tmax",
            dest="tmax",
            type=str,
            default=None,
            help="end")
        parser.add_option(
            "--ngpu",
            help="number of GPUs to use")
        parser.add_option(
            "--gpu-no",
            help="GPU number to use",
            type=int)
        parser.add_option(
            "--config",
            help="Load a grond configuration file")
        parser.add_option(
            "--ttt_path",
            help="Path of ttt stores")
        parser.add_option(
            "--config_grond",
            help="Load a configuration file")
        parser.add_option(
            "--configs",
            help="load a comma separated list of configs and process them"
        )
        parser.add_option(
            "--debug",
            help="enable logging level DEBUG",
            action="store_true")
        parser.add_option(
            "--force",
            action="store_true")
        parser.add_option(
            "--freq",
            dest="freq",
            type=float,
            default=None,
            help="Frequency value")
        parser.add_option(
            "--apply_residuals",
            dest="apply_residuals",
            type=str,
            default=True,
            help="apply_residuals",)
        parser.add_option(
            "--seiger",
            dest="seiger",
            type=str,
            default=True,
            help="seiger",)
        parser.add_option(
            "--hf",
            dest="hf",
            type=float,
            default=50,
            help="High frequency filter")
        parser.add_option(
            "--lf",
            dest="lf",
            type=float,
            default=2,
            help="end")
        parser.add_option(
            "--store_path",
            dest="store_path",
            type=str,
            default="",
            help="Store path")
        parser.add_option(
            "--waveform_path",
            dest="waveform_path",
            type=str,
            default="",
            help="waveform path")
        parser.add_option(
            "--ref_catalog",
            dest="ref_catalog",
            type=str,
            default=None,
            help="Reference catalog path")
        parser.add_option(
            "--store_path_reader",
            dest="store_path_reader",
            type=str,
            default=None,
            help="store_path_reader")
        parser.add_option(
            "--store_interval",
            dest="store_interval",
            type=float,
            default=50,
            help="store_interval")
        parser.add_option(
            "--wait_period",
            dest="wait_period",
            type=float,
            default=130,
            help="wait_period")
        parser.add_option(
            "--sources_list",
            dest="sources_list",
            default=["seedlink://eida.bgr.de/GR.INS*.*.EH*",
                     "seedlink://eida.bgr.de/LE.*.*.*",
                     "seedlink://eida.bgr.de/GR.TMO*.*.EH*"],
            help="Pyrocko format station file")
        parser.add_option(
            "--stations_file",
            dest="stations_file",
            type=str,
            default="stations_landau.txt",
            help="Pyrocko format station file")
    parser, options, args = cl_parse("detect", args, setup)
    if options.mode == "both" or options.mode == "eqt" or options.mode == "lassie":
        from silvertine.util import eqt_util
        model_eqt = eqt_util.load_eqt_model()
    else:
        model_eqt = []
    if options.apply_residuals is True:
        residuals = ref_mods.insheim_i1_layered_model_residuals()
    if options.mode == "both" or options.mode == "eqt" or options.mode == "lassie":
            piled = pile_mod.make_pile()
            sources_list = options.sources_list  # seedlink sources
            pollinjector = None
            tempdir = None
            if options.store_path is not None:
                if options.sds is None:
                    store_path_base_down = options.waveform_path+"download-tmp"
                    store_path_base = options.store_path
                else:
                    store_path_base_down = options.waveform_path
                    store_path_base = options.store_path
            else:
                store_path_base_down = "."
                store_path_base = "."
            if options.store_path_reader is None:
                store_path_reader = options.store_path+"read"
            else:
                store_path_reader = options.store_path_reader

            stations = model.load_stations(store_path_base+options.stations_file)

            if sources_list:
                if store_path_base_down is None:
                    tempdir = tempfile.mkdtemp("", "snuffler-tmp-")
                    store_path_stream = pjoin(
                        tempdir,
                        "trace-%(network)s.%(station)s.%(location)s.%(channel)s."
                        "%(tmin_ms)s.mseed",
                    )
                util.ensuredir(store_path_base_down)
                if os.path.isdir(store_path_base_down):
                    store_path_stream = pjoin(
                        store_path_base_down,
                        "trace-%(network)s.%(station)s.%(location)s.%(channel)s."
                        "%(tmin_ms)s.mseed",
                    )
                _injector = pile_mod.Injector(
                    piled,
                    path=store_path_stream,
                    fixation_length=options.store_interval,
                    forget_fixed=True)

                # Data is downloaded continously after starting the stream
                if options.download_method is "stream":
                    sources = setup_acquisition_sources(sources_list)
                    for source in sources:
                        source.start()
                if options.ref_catalog is not None:
                    ref_catalog = model.load_events(options.ref_catalog)

                diff = 0 # for keeping track of time between data saving
                models = []
                events_eqt = []
                events_stacking = []
                process_in_progress = True
                config_path = options.config
                config = lassie.read_config(config_path)
                fine_detection = False

                if options.tmin is not None:
                    tmin_override = util.stt(options.tmin)
                else:
                    tmin_override = None
                if options.tmax is not None:
                    tmax_override = util.stt(options.tmax)
                else:
                    tmax_override = None
                while process_in_progress is True:
                    try:
                        if options.download_method is "stream":
                            time.sleep(options.wait_period-diff)
                    except:
                        print("Alarm! Processing takes to long!")
                        for source in sources:
                            source.stop()
                        pass

                    if options.download_method is "stream":
                        for source in sources:
                            trs = source.poll()
                            for tr in trs:
                                _injector.inject(tr)
                    start = time.time()
                    if options.seiger is True:
                        store_path_LE = "/diskb/steinberg/seiger-data"
                        if options.sds is not None:
                            list_sds = options.sds.split(",")
                            path_waveforms = []
                            for sd in list_sds:
                                for st in stations:
                                    for cha in st.channels:
                                        path_waveforms.append(store_path_base_down+"%s/%s/%s/%s.D/" %(sd, st.network, st.station, cha.name))
                                if int(sd) <= 2020:
                                    path_waveforms.append(store_path_LE+"/%s/" %(sd))

                        config.data_paths = path_waveforms

                    else:
                        path_waveforms = store_path_base_down
                        config.data_paths = [path_waveforms]
                    if options.mode == "both" or options.mode == "lassie":
                        target = lassie.search(config,
                                               override_tmin=tmin_override,
                                               override_tmax=tmax_override,
                                               force=True,
                                               show_detections=True,
                                               nparallel=10)
                        gc.collect()
                    if options.mode == "both" or options.mode == "eqt":
                        detector.picker.main(
                            store_path_base,
                            tmin=options.tmin,
                            tmax=options.tmax,
                            minlat=49.0,
                            maxlat=49.979,
                            minlon=7.9223,
                            maxlon=8.9723,
                            channels=["EH" + "[ZNE]"],
                            client_list=[
                                "http://eida.bgr.de",
                                "http://ws.gpi.kit.edu",
                                ],
                            path_waveforms=path_waveforms,
                            download=options.download,
                            tinc=options.tinc,
                            freq=options.freq,
                            hf=options.hf,
                            lf=options.lf,
                            models=[model_eqt],
                        )
                        gc.collect()
                    end = time.time()
                    diff = end - start
                    try:
                        try:
                            events_stacking = model.load_events(store_path_base+"stacking_events.pf")
                        except:
                            events_stacking = []
                        events_stacking.extend(model.load_events(config.get_events_path()))
                        model.dump_events(events_stacking, store_path_base+"stacking_events.pf")
                    except:
                        pass

                    # for event in events_stacking:
                    #     savedir = store_path_base + '/stacking_detections/' + str(event.time) + '/'
                    #     if not os.path.exists(savedir):
                    #         os.makedirs(savedir)
                    #         os.makedirs(savedir+"figures_stacking")
                    #     for item in Path(config.path_prefix+"/"+config.run_path+"/figures").glob("*"):
                    #         try:
                    #             time_item = util.stt(str(item.absolute())[-27:-17]+" "+str(item.absolute())[-16:-4])
                    #             if event.time-6 < time_item and event.time+6 > time_item:
                    #                 os.system("cp %s %s" % (item.absolute(), savedir+"figures_stacking"))
                    #                 plot_waveforms = False
                    #                 if plot_waveforms is True:
                    #                     pile_event = pile.make_pile(store_path_base_down+"download-tmp", show_progress=False)
                    #                     for traces in pile_event.chopper(tmin=event.time-10,
                    #                                                      tmax=event.time+10,
                    #                                                      keep_current_files_open=False,
                    #                                                      want_incomplete=True):
                    #                         waveform.plot_waveforms(traces, event, stations,
                    #                                                 savedir, None)
                    #         except:
                    #             pass
                    try:
                        events_eqt = model.load_events(store_path_base+"eqt_events.pf")
                    except:
                        events_eqt = []
                    if options.mode == "both" or options.mode == "eqt":

                        for item in Path(store_path_base).glob("asociation_*/events.pf"):
                            events_eqt.extend(model.load_events(item.absolute()))
                        model.dump_events(events_eqt, store_path_base+"eqt_events.pf")
                        phase_markers_collected = []
                        for event in events_eqt:
                            savedir = store_path_base + '/eqt_detections/' + str(event.time) + '/'
                            if not os.path.exists(savedir):
                                os.makedirs(savedir)
                                os.makedirs(savedir+"figures_eqt")
                            for item in Path(store_path_base+"/").glob("asociation_*/associations.xml"):

                                qml = quakeml.QuakeML.load_xml(filename=item)
                                events_qml = qml.get_pyrocko_events()
                                components = ["*"]
                                for i, eq in enumerate(events_qml):
                                    if event.time == eq.time:
                                        phase_markers = []
                                        evqml = qml.get_events()[i]
                                        phase_markers_qml = evqml.get_pyrocko_phase_markers()
                                        for phase_marker in phase_markers_qml:
                                            for component in components:
                                                d = phase_marker.copy()
                                                nsl = list(d.nslc_ids[0])

                                                nsl[3] = component
                                                nsl[2] = ""
                                                nsl = tuple(nsl)
                                                d.set([nsl], d.tmin, d.tmax)

                                                phase_markers.append(d)
                                                phase_markers_collected.append(d)
                                        PhaseMarker.save_markers(phase_markers, savedir+"/events_eqt.pym", fdigits=3)

                                qml.dump_xml(filename=savedir+"event_eqt.qml")
                            PhaseMarker.save_markers(phase_markers_collected, store_path_base+"/events_eqt_collected.pym", fdigits=3)

                            for item in Path(store_path_base+"/").glob("detections_*/*/figures/*"):
                                try:
                                    time_item = util.stt(str(item.absolute())[-31:-21]+" "+str(item.absolute())[-20:-5])
                                    if event.time-options.wait_period < time_item and event.time+options.wait_period > time_item:
                                        os.system("cp %s %s" % (item.absolute(), savedir+"figures_eqt"))
                                except:
                                    pass
                    catalog = read_events(store_path_base+"/LI_catalog.qml")
                    event_name = None
                    if options.mode == "both":
                        for event_stack in events_stacking:
                            for event_eqt in events_eqt:
                                phase_markers = []
                                event_marker = []
                                if event_stack.time-6 < event_eqt.time and event_stack.time+6 > event_eqt.time:
                                    savedir = store_path_base + '/combined_detections/' + util.tts(event_stack.time) + '/'
                                    if not os.path.exists(savedir):
                                        os.makedirs(savedir)
                                    for item in Path(store_path_base+"/").glob("asociation_*/associations.xml"):
                                        qml = quakeml.QuakeML.load_xml(filename=item)
                                        events_qml = qml.get_pyrocko_events()
                                        components = ["*"]

                                        for ieqml, eq in enumerate(events_qml):
                                            if event_eqt.time == eq.time:
                                                phase_markers = []
                                                ieqml_found = ieqml
                                                evqml = qml.get_events()[ieqml]
                                                event_marker = evqml.get_pyrocko_event()
                                                if options.seiger is True:
                                                # seiger center coordinates
                                                    event_marker.lat = 49.16512600149505
                                                    event_marker.lon = 8.133401103618198
                                                event_marker.name = str(int(event.time))
                                                event_name = event_marker.name
                                                phase_markers_qml = evqml.get_pyrocko_phase_markers()
                                                #event_marker.hash = phase_markers[0].get_event_hash()
                                                phase_markers.append(EventMarker(event_marker))
                                                model.dump_events([event_marker], savedir+"/event.pf")
                                                if options.mode is not "stacking":
                                                for phase_marker in phase_markers_qml:
                                                    phase_marker.set_event_hash(EventMarker(event_marker).get_event_hash())
                                                    if phase_marker.get_phasename() == 'P':
                                                        phase_marker.set_phasename("any_P")
                                                    if phase_marker.get_phasename() == 'S':
                                                        phase_marker.set_phasename("any_S")
                                                    for component in components:

                                                        d = phase_marker.copy()
                                                        nsl = list(d.nslc_ids[0])

                                                        nsl[3] = component
                                                        nsl[2] = ""
                                                        nsl = tuple(nsl)

                                                        if options.apply_residuals:
                                                            try:
                                                                P_res = residuals["%s" % nsl[1]][0]
                                                                S_res = residuals["%s" % nsl[1]][1]
                                                                if phase_marker.get_phasename() == 'any_P':
                                                                    d.set([nsl], d.tmin+P_res, d.tmax)
                                                                if phase_marker.get_phasename() == 'any_S':
                                                                    d.set([nsl], d.tmin+S_res, d.tmax)
                                                            except:
                                                                d.set([nsl], d.tmin, d.tmax)
                                                        else:
                                                            d.set([nsl], d.tmin, d.tmax)

                                                        phase_markers.append(d)
                                                phase_markers = list(set(phase_markers))
                                                for i, p in enumerate(phase_markers):
                                                    for k, ps in enumerate(phase_markers):
                                                        try:
                                                            if k != i:
                                                                if p.get_phasename() == ps.get_phasename() and p.nslc_ids[0] == ps.nslc_ids[0]:
                                                                    phase_markers.remove(p)
                                                        except:
                                                            pass
                                                PhaseMarker.save_markers(phase_markers, savedir+"/phases_res.pym", fdigits=3)
                                    if event_name is not None:
                                        marker_file = savedir+'/phases_res.pym'
                                    #    gf_stores_path = "/media/asteinbe/aki/seiger-data/data_single/grond/gf_stores"
                                        scenario_dir = savedir
                                    #    config_path = '/media/asteinbe/aki/seiger-data/data_single/grond/grond.conf'
                                    #    stations_path = '/media/asteinbe/aki/seiger-data/stations_landau.txt'
                                        gf_stores_path = options.ttt_path
                                        config_path = options.config_grond
                                        stations_path = store_path_base+options.stations_file
                                        if options.download_method is "stream":
                                            thread = threading.Thread(target=command_locate_with_grond, args=(marker_file, gf_stores_path, scenario_dir, config_path, stations_path, event_name, evqml, qml, event_stack, event_marker, savedir, store_path_reader))
                                            thread.start()
                                        else:
                                            best = glocate(marker_file, gf_stores_path, scenario_dir, config_path, stations_path, event_name)

                                            lat, lon = ort.ne_to_latlon(best.lat, best.lon, best.north_shift, best.east_shift)
                                            evqml.preferred_origin.latitude = quakeml.RealQuantity(value=float(lat))
                                            evqml.preferred_origin.longitude = quakeml.RealQuantity(value=float(lon))
                                            evqml.preferred_origin.time = quakeml.TimeQuantity(value=best.time)
                                            evqml.preferred_origin.depth = quakeml.RealQuantity(value=best.depth)
                                            public_id = "quakeml:seiger/" + str(int(float(event_stack.name.split()[1].replace(')','').replace('(',''))))+"_"+str(best.time)
                                            event_marker.lat = lat
                                            event_marker.lon = lon
                                            event_marker.depth = best.depth
                                            event_marker.time = best.time
                                            model.dump_events([event_marker], savedir+"/event.pf")

                                            evqml.public_id = public_id
                                            for pick in evqml.pick_list:
                                                rand = pick.public_id[-5]
                                                pick.public_id = public_id+"_"+rand+"_"+pick.phase_hint.value
                                                arrival = quakeml.Arrival(public_id = pick.public_id+"_arrival", pick_id = pick.public_id, phase=quakeml.Phase(value=pick.phase_hint.value))
                                                evqml.origin_list[0].arrival_list.append(arrival)
                                            evqml.preferred_origin_id = public_id+"_"+rand+"O"
                                            evqml.origin_list[0].public_id = public_id+"_"+rand+"O"
                                            qml.event_parameters.event_list = [evqml]

                                            qml.dump_xml(filename=savedir+"event_combined.qml")
                                            evs = read_events(savedir+"event_combined.qml")
                                            for ev in evs:
                                                if len(ev.picks) > 2:
                                                    catalog.append(ev)
                                                sc3._write_sc3ml(evs, store_path_reader+"/LI_SC_%s.qml" % best.time)
                                                _write_quakeml(evs, store_path_reader+"/LI_%s.qml" % best.time)

                        sc3._write_sc3ml(catalog, store_path_base+"/LI_catalog_SC.qml")
                        _write_quakeml(catalog, store_path_base+"/LI_catalog.qml")
                    gc.collect()
                    piled = pile_mod.make_pile()

                    if options.download_method is "stream":
                        remove_outdated_wc(store_path_base+"/download-tmp",
                                           2.5,
                                           wc="*")
                        remove_outdated_wc(store_path_base,
                                           1,
                                           wc="detections_*")
                        remove_outdated_wc(store_path_base,
                                           1,
                                           wc="asociation_*")

                    for item in Path(store_path_base+"/downloads/").glob("*"):
                        try:
                            shutil.rmtree(item.absolute())
                        except:
                            pass
                    if options.download_method == "stream_sim":
                        process_in_progress = False

                if options.download_method == "stream":
                    for source in sources:
                        source.stop()


def command_optimize(args):
    def setup(parser):
        parser.add_option(
            "--show",
            dest="show",
            type=str,
            default=False,
            help="Display progress of localisation for each event in browser",
        )
        parser.add_option(
            "--all", dest="all", type=str, default=False, help="Optimize all in folder"
        )
        parser.add_option(
            "--domain",
            dest="domain",
            type=str,
            default="time_domain",
            help="Optimize all in folder",
        )
        parser.add_option(
            "--problem",
            dest="problem",
            type=str,
            default="CMTProblem",
            help="Optimize all in folder",
        )

    from silvertine import mechanism
    from pyrocko import model

    parser, options, args = cl_parse("locate", args, setup)
    if options.all is False:
        project_dir = args[0]
        rundir = project_dir + "grun"

        event = model.load_events(project_dir + "event.txt")[0]
        mechanism.run_grond(
            rundir,
            project_dir,
            event.name,
            "landau_100hz",
            problem_type=options.problem,
            domain=options.domain,
        )
    else:
        from pathlib import Path

        pathlist = Path(args[0]).glob("scenario*/")
        for path in sorted(pathlist):
            project_dir = str(path) + "/"
            rundir = project_dir + "grun"
            try:
                # event.txt for real data?
                try:
                    event = model.load_events(project_dir + "event.txt")[0]
                    eventname = event.name
                except:
                    eventname = project_dir

                mechanism.run_grond(rundir, project_dir, eventname, "landau_100hz")
            except:
                pass


def command_plot_mods(args):
    def setup(parser):
        parser.add_option(
            "--show",
            dest="show",
            type=str,
            default=False,
            help="overwrite existing project folder.",
        )
        parser.add_option(
            "--ref_models",
            dest="ref_models",
            type=str,
            default=False,
            help="Plot reference models only.",
        )
        parser.add_option(
            "--folder", dest="folder", type=str, default="data/", help="model."
        )
    parser, options, args = cl_parse("plot_mods", args, setup)

    from silvertine.util import silvertine_plot
    from pyrocko import cake
    from matplotlib import pyplot as plt
    from silvertine.util import ref_mods

    mod_insheim = ref_mods.insheim_layered_model()
    mod_landau = ref_mods.landau_layered_model()
    mod_vsp = ref_mods.vsp_layered_model()
    insheim_i1_layered_model = ref_mods.insheim_i1_layered_model()
    landau_l1_layered_model = ref_mods.landau_l1_layered_model()

    mods = [
        mod_vsp,
        mod_insheim,
        mod_landau,
        insheim_i1_layered_model,
        landau_l1_layered_model,
    ]
    if options.ref_models is not False:
        fig, axes = silvertine_plot.bayesian_model_plot(
            mods, axes=None, highlightidx=[0, 1, 2, 3, 4]
        )
    else:
        from pathlib import Path

        pathlist = Path(options.folder).glob("*_*")
        for path in pathlist:
            mod = cake.load_model(str(path))
            mods.append(mod)
        fig, axes = silvertine_plot.bayesian_model_plot(
            mods, axes=None, highlightidx=[0, 1, 2, 3, 4]
        )

    if options.show is not False:
        plt.show()


def command_pertub_earthmodels(args):
    def setup(parser):
        parser.add_option(
            "--folder", dest="folder", type=str, default="data/", help="model."
        )
        parser.add_option("--model", dest="mod", type=str, default="vsp", help="model.")
        parser.add_option(
            "--nboot", dest="nboot", type=int, default=1, help="Number of boots."
        )
        parser.add_option(
            "--error_depth",
            dest="error_depth",
            type=float,
            default=0.2,
            help="Error in depth.",
        )
        parser.add_option(
            "--error_velocities",
            dest="error_velocities",
            type=float,
            default=0.2,
            help="Error in depth.",
        )
        parser.add_option(
            "--depth_variation",
            dest="depth_variation",
            type=float,
            default=16000.0,
            help="Max. error in depth.",
        )
        parser.add_option(
            "--gf_store", dest="gf_store", type=str, default=None, help="model."
        )

    parser, options, args = cl_parse("pertub_earthmodels", args, setup)
    from silvertine.util import store_variation, ref_mods

    if options.mod == "insheim":
        mod = ref_mods.insheim_layered_model()
    elif options.mod == "landau":
        mod = ref_mods.landau_layered_model()
    elif options.mod == "vsp":
        mod = ref_mods.vsp_layered_model()
    else:
        from pyrocko import cake

        mod = cake.load_model(options.folder + options.mod)

    pertubed_mods = store_variation.ensemble_earthmodel(
        mod,
        num_vary=options.nboot,
        error_depth=options.error_depth,
        error_velocities=options.error_velocities,
        depth_limit_variation=options.depth_variation,
    )

    store_variation.save_varied_models(pertubed_mods, options.folder, name=options.mod)

    if options.gf_store is not None:
        for k, mod in enumerate(pertubed_mods):
            store_variation.create_gf_store(
                mod, options.gf_store, name=options.mod + "_pertubation_%s" % (k)
            )


def command_scenario(args):
    def setup(parser):
        parser.add_option(
            "--magnitude-range",
            dest="magnitude_range",
            type=str,
            action="callback",
            callback=magnitude_range,
            default=[6.0, 7.0],
            help="Magnitude range min_mag-max_mag (default: %default)",
        )
        parser.add_option(
            "--nstations",
            dest="nstations",
            type=int,
            default=20,
            help="number of seismic stations to create (default: %default)",
        )
        parser.add_option(
            "--nevents",
            dest="scenarios",
            type=int,
            default=10,
            help="number of events to create (default: %default)",
        )
        parser.add_option(
            "--latmin",
            dest="latmin",
            type=float,
            default=49.09586,
            help="min latitude of the scenario (default: %default)",
        )
        parser.add_option(
            "--latmax",
            dest="latmax",
            type=float,
            default=49.25,
            help="max latitude of the scenario (default: %default)",
        )
        parser.add_option(
            "--lonmin",
            dest="lonmin",
            type=float,
            default=8.0578,
            help="min longititude of the scenario (default: %default)",
        )
        parser.add_option(
            "--lonmax",
            dest="lonmax",
            type=float,
            default=8.2078,
            help="max longititude of the scenario (default: %default)",
        )
        parser.add_option(
            "--depth_min",
            dest="depmin",
            type=int,
            default=3,
            help="minimum depth (default: %default)",
        )
        parser.add_option(
            "--depth_max",
            dest="depmax",
            type=int,
            default=10,
            help="maximum depth (default: %default)",
        )
        parser.add_option(
            "--mag_min",
            dest="magmin",
            type=int,
            default=0.1,
            help="minimum depth (default: %default)",
        )
        parser.add_option(
            "--mag_max",
            dest="magmax",
            type=int,
            default=3,
            help="maximum depth (default: %default)",
        )
        parser.add_option(
            "--stations_file",
            dest="stations_file",
            type=str,
            default="stations.raw.txt",
            help="maximum depth (default: %default)",
        )
        parser.add_option(
            "--force",
            dest="force",
            action="store_true",
            help="overwrite existing project folder.",
        )
        parser.add_option(
            "--gf-store-superdirs",
            dest="gf_store_superdirs",
            help="Comma-separated list of directories containing GF stores",
        )
        parser.add_option(
            "--shakemap",
            dest="shakemap",
            type=str,
            default=False,
            help="Generate synthetic shakemaps for the scenario",
        )
        parser.add_option(
            "--station_dropout",
            dest="station_dropout",
            type=str,
            default=False,
            help="t_station_dropout",
        )
        parser.add_option(
            "--scenario_type",
            dest="scenario_type",
            type=str,
            default="full",
            help="Type of scenario to be generated",
        )
        parser.add_option(
            "--event_list",
            dest="event_list",
            type=str,
            default=False,
            help="Use pre-determined event list as basis",
        )

    parser, options, args = cl_parse("scenario", args, setup)

    gf_store_superdirs = None
    if options.gf_store_superdirs:
        gf_store_superdirs = options.gf_store_superdirs.split(",")
    else:
        gf_store_superdirs = None
    if options.shakemap is not False:
        options.shakemap = True
    if options.station_dropout is not False:
        options.station_dropout = True
    from silvertine import scenario as silvertine_scenario

    project_dir = args[0]
    scenario = silvertine.scenario.silvertineScenario(
        project_dir,
        scenarios=options.scenarios,
        magmin=options.magmin,
        magmax=options.magmax,
        latmin=options.latmin,
        latmax=options.latmax,
        lonmin=options.lonmin,
        lonmax=options.lonmax,
        depmin=options.depmin,
        depmax=options.depmax,
        stations_file=options.stations_file,
        shakemap=options.shakemap,
        gf_store_superdirs=options.gf_store_superdirs,
        scenario_type=options.scenario_type,
        event_list=options.event_list,
    )


def command_post_shakemap(args):
    def setup(parser):

        parser.add_option(
            "--wanted_start",
            dest="wanted_start",
            type=int,
            default=0,
            help="number of events to create (default: %default)",
        )
        parser.add_option(
            "--wanted_end",
            dest="wanted_end",
            type=int,
            default=1,
            help="number of events to create (default: %default)",
        )
        parser.add_option(
            "--stations_file",
            dest="stations_file",
            type=str,
            default="stations.raw.txt",
            help="maximum depth (default: %default)",
        )
        parser.add_option(
            "--store_id",
            dest="store_id",
            type=str,
            default="insheim_100hz",
            help="maximum depth (default: %default)",
        )
        parser.add_option(
            "--measured",
            dest="measured",
            type=str,
            default=None,
            help="maximum depth (default: %default)",
        )
        parser.add_option(
            "--scenario",
            dest="scenario",
            type=str,
            default=False,
            help="maximum depth (default: %default)",
        )
        parser.add_option(
            "--force",
            dest="force",
            action="store_true",
            help="overwrite existing project folder.",
        )
        parser.add_option(
            "--pertub_velocity_model",
            dest="pertub_velocity_model",
            type=str,
            default=False,
            help="pertub_velocity_model.",
        )
        parser.add_option(
            "--gf-store-superdirs",
            dest="gf_store_superdirs",
            help="Comma-separated list of directories containing GF stores",
        )
        parser.add_option(
            "--n_pertub",
            dest="n_pertub",
            type=int,
            default=0,
            help="number of pertubations to create (default: %default)",
        )
        parser.add_option(
            "--pertub_degree",
            dest="pertub_degree",
            type=float,
            default=20,
            help="number of pertubations to create (default: %default)",
        )
        parser.add_option(
            "--pgv_outline",
            dest="value_level",
            type=float,
            default=0.005,
            help="Outline of certain PGV value (default: %default)",
        )
        parser.add_option(
            "--strike",
            dest="strike",
            type=float,
            default=None,
            help="Outline of certain PGV value (default: %default)",
        )
        parser.add_option(
            "--dip",
            dest="dip",
            type=float,
            default=None,
            help="Outline of certain PGV value (default: %default)",
        )
        parser.add_option(
            "--rake",
            dest="rake",
            type=float,
            default=None,
            help="Outline of certain PGV value (default: %default)",
        )
        parser.add_option(
            "--moment",
            dest="moment",
            type=float,
            default=None,
            help="Outline of certain PGV value (default: %default)",
        )
        parser.add_option(
            "--depth",
            dest="depth",
            type=float,
            default=None,
            help="Outline of certain PGV value (default: %default)",
        )
        parser.add_option(
            "--source_type",
            dest="source_type",
            type=str,
            default="MT",
            help="Source Type (default: %default)",
        )
        parser.add_option(
            "--stations_corrections_file",
            dest="stations_corrections_file",
            type=str,
            default=None,
            help="stations_corrections_file",
        )

    parser, options, args = cl_parse("post_shakemap", args, setup)

    gf_store_superdirs = None
    if options.gf_store_superdirs:
        gf_store_superdirs = options.gf_store_superdirs.split(",")
    else:
        gf_store_superdirs = None
    if options.measured is not None:
        options.measured = True
    if options.scenario is not False:
        options.scenario = True
    from silvertine import scenario as silvertine_scenario

    project_dir = args[0]
    scenario = silvertine.scenario.fwd_shakemap_post(
        project_dir,
        wanted_start=options.wanted_start,
        wanted_end=options.wanted_end,
        store_id=options.store_id,
        gf_store_superdirs=options.gf_store_superdirs,
        pertub_degree=options.pertub_degree,
        n_pertub=options.n_pertub,
        value_level=options.value_level,
        pertub_velocity_model=options.pertub_velocity_model,
        measured=options.measured,
        scenario_run=options.scenario,
        strike=options.strike,
        dip=options.dip,
        rake=options.rake,
        moment=options.moment,
        depth=options.depth,
        source_type=options.source_type,
        stations_corrections_file=options.stations_corrections_file,
    )



def make_report(env_args, event_name, conf, update_without_plotting, nthreads):
    from silvertine.environment import Environment
    from silvertine.report import report

    try:
        env = Environment(env_args)
        if event_name:
            env.set_current_event_name(event_name)

        report(
            env,
            conf,
            update_without_plotting=update_without_plotting,
            make_index=False,
            make_archive=False,
            nthreads=nthreads,
        )

        return True

    except silvertine.silvertineError as e:
        logger.error(str(e))
        return False


def command_report(args):

    import matplotlib

    matplotlib.use("Agg")

    from pyrocko import parimap

    from silvertine.report import (
        report_index,
        report_archive,
        serve_ip,
        serve_report,
        read_config,
        write_config,
        ReportConfig,
    )

    def setup(parser):
        parser.add_option(
            "--index-only",
            dest="index_only",
            action="store_true",
            help="create index only",
        )
        parser.add_option(
            "--serve",
            "-s",
            dest="serve",
            action="store_true",
            help="start http service",
        )
        parser.add_option(
            "--serve-external",
            "-S",
            dest="serve_external",
            action="store_true",
            help="shortcut for --serve --host=default --fixed-port",
        )
        parser.add_option(
            "--host",
            dest="host",
            default="localhost",
            help="<ip> to start the http server on. Special values for "
            '<ip>: "*" binds to all available interfaces, "default" '
            'to default external interface, "localhost" to "127.0.0.1".',
        )
        parser.add_option(
            "--port",
            dest="port",
            type=int,
            default=8383,
            help="set default http server port. Will count up if port is "
            "already in use unless --fixed-port is given.",
        )
        parser.add_option(
            "--fixed-port",
            dest="fixed_port",
            action="store_true",
            help="fail if port is already in use",
        )
        parser.add_option(
            "--open",
            "-o",
            dest="open",
            action="store_true",
            help="open report in browser",
        )
        parser.add_option(
            "--all",
            "-a",
            dest="all",
            action="store_true",
            help="Make report for all in directory",
        )
        parser.add_option(
            "--config",
            dest="config",
            metavar="FILE",
            help="report configuration file to use",
        )
        parser.add_option(
            "--write-config",
            dest="write_config",
            metavar="FILE",
            help="write configuration (or default configuration) to FILE",
        )
        parser.add_option(
            "--update-without-plotting",
            dest="update_without_plotting",
            action="store_true",
            help="quick-and-dirty update parameter files without plotting",
        )
        parser.add_option(
            "--parallel",
            dest="nparallel",
            type=int,
            default=1,
            help="set number of runs to process in parallel, "
            "If set to more than one, --status=quiet is implied.",
        )
        parser.add_option(
            "--scenario", dest="scenario", type=str, default="False",
                          help="Scenario."
        )
        parser.add_option(
            "--threads",
            dest="nthreads",
            type=int,
            default=1,
            help="set number of threads per process (default: 1)."
            "Set to 0 to use all available cores.",
        )
        parser.add_option(
            "--no-archive",
            dest="no_archive",
            action="store_true",
            help="don't create archive file.",
        )

    parser, options, args = cl_parse("report", args, setup)
    if options.scenario is not "False":
        options.scenario = True
    s_conf = ""
    if options.config:
        try:
            conf = read_config(options.config)
        except silvertine.silvertineError as e:
            die(str(e))

        s_conf = ' --config="%s"' % options.config
    else:
        from silvertine import plot

        conf = ReportConfig(plot_config_collection=plot.get_plot_config_collection())
        conf.set_basepath(".")

    if options.write_config:
        try:
            write_config(conf, options.write_config)
            sys.exit(0)

        except silvertine.silvertineError as e:
            die(str(e))

    # commandline options that can override config values
    if options.no_archive:
        conf.make_archive = False

    if len(args) == 1 and op.exists(op.join(args[0], "index.html")):
        conf.report_base_path = conf.rel_path(args[0])
        s_conf = " %s" % args[0]
        args = []

    report_base_path = conf.expand_path(conf.report_base_path)

    if options.index_only:
        report_index(conf)
        report_archive(conf)
        args = []

    entries_generated = False
    payload = []
    try:
        if options.all is True:
            from pathlib import Path
            if options.scenario is True:
                pathlist = Path(args[0]).glob("scenario*/")
            else:
                pathlist = Path(args[0]).glob("event*/")
            for path in sorted(pathlist):
                rundir = str(path) + "/"
                payload.append(
                    (
                        [rundir],
                        None,
                        conf,
                        options.update_without_plotting,
                        options.nthreads,
                    )
                )
        else:
            from pathlib import Path
            rundir = str(path) + "/"
            payload.append(
                (
                    [rundir],
                    None,
                    conf,
                    options.update_without_plotting,
                    options.nthreads,
                )
            )
        if all is False and open is False:
            if args and all(op.isdir(rundir) for rundir in args):
                rundirs = args
                all_failed = True
                for rundir in rundirs:
                    payload.append(
                        (
                            [rundir],
                            None,
                            conf,
                            options.update_without_plotting,
                            options.nthreads,
                        )
                    )
    except:
        pass
    if payload:
        entries_generated = []
        for result in parimap.parimap(
            make_report, *zip(*payload), nprocs=options.nparallel
        ):

            entries_generated.append(result)

        all_failed = not any(entries_generated)
        entries_generated = any(entries_generated)

        if all_failed:
            die("no report entries generated")

        report_index(conf)
        report_archive(conf)

    if options.serve or options.serve_external:
        if options.serve_external:
            host = "default"
        else:
            host = options.host

        addr = serve_ip(host), options.port

        serve_report(
            addr,
            report_config=conf,
            fixed_port=options.fixed_port or options.serve_external,
            open=options.open,
        )

    elif options.open:
        import webbrowser

        url = "file://%s/index.html" % op.abspath(report_base_path)
        webbrowser.open(url)

    else:
        if not entries_generated and not options.index_only:
            logger.info("Nothing to do, see: silvertine report --help")

    if entries_generated and not (options.serve or options.serve_external):
        logger.info(CLIHints("report", config=s_conf))


def command_qc_polarization(args):
    def setup(parser):
        parser.add_option(
            "--time-factor-pre",
            dest="time_factor_pre",
            type=float,
            metavar="NUMBER",
            default=0.5,
            help="set duration to extract before synthetic P phase arrival, "
            "relative to 1/fmin. fmin is taken from the selected target "
            "group in the config file (default=%default)",
        )
        parser.add_option(
            "--time-factor-post",
            dest="time_factor_post",
            type=float,
            metavar="NUMBER",
            default=0.5,
            help="set duration to extract after synthetic P phase arrival, "
            "relative to 1/fmin. fmin is taken from the selected target "
            "group in the config file (default=%default)",
        )
        parser.add_option(
            "--distance-min",
            dest="distance_min",
            type=float,
            metavar="NUMBER",
            help="minimum event-station distance [m]",
        )
        parser.add_option(
            "--distance-max",
            dest="distance_max",
            type=float,
            metavar="NUMBER",
            help="maximum event-station distance [m]",
        )
        parser.add_option(
            "--depth-min",
            dest="depth_min",
            type=float,
            metavar="NUMBER",
            help="minimum station depth [m]",
        )
        parser.add_option(
            "--depth-max",
            dest="depth_max",
            type=float,
            metavar="NUMBER",
            help="maximum station depth [m]",
        )
        parser.add_option(
            "--picks",
            dest="picks_filename",
            metavar="FILENAME",
            help="add file with P picks in Snuffler marker format",
        )
        parser.add_option(
            "--save",
            dest="output_filename",
            metavar="FILENAME.FORMAT",
            help="save output to file FILENAME.FORMAT",
        )
        parser.add_option(
            "--dpi",
            dest="output_dpi",
            type=float,
            default=120.0,
            metavar="NUMBER",
            help="DPI setting for raster formats (default=120)",
        )

    parser, options, args = cl_parse("qc-polarization", args, setup)
    if len(args) != 3:
        help_and_die(parser, "missing arguments")

    if options.output_filename:
        import matplotlib

        matplotlib.use("Agg")

    import silvertine.qc

    config_path, event_name, target_group_path = args

    try:
        config = silvertine.read_config(config_path)
    except silvertine.silvertineError as e:
        die(str(e))

    ds = config.get_dataset(event_name)

    engine = config.engine_config.get_engine()

    nsl_to_time = None
    if options.picks_filename:
        markers = marker.load_markers(options.picks_filename)
        marker.associate_phases_to_events(markers)

        nsl_to_time = {}
        for m in markers:
            if isinstance(m, marker.PhaseMarker):
                ev = m.get_event()
                if ev is not None and ev.name == event_name:
                    nsl_to_time[m.one_nslc()[:3]] = m.tmin

        if not nsl_to_time:
            help_and_die(
                parser,
                'no markers associated with event "%s" found in file "%s"'
                % (event_name, options.picks_filename),
            )

    target_group_paths_avail = []
    for target_group in config.target_groups:
        name = target_group.path
        if name == target_group_path:
            imc = target_group.misfit_config
            fmin = imc.fmin
            fmax = imc.fmax
            ffactor = imc.ffactor

            store = engine.get_store(target_group.store_id)
            timing = "{cake:P|cake:p|cake:P\\|cake:p\\}"

            silvertine.qc.polarization(
                ds,
                store,
                timing,
                fmin=fmin,
                fmax=fmax,
                ffactor=ffactor,
                time_factor_pre=options.time_factor_pre,
                time_factor_post=options.time_factor_post,
                distance_min=options.distance_min,
                distance_max=options.distance_max,
                depth_min=options.depth_min,
                depth_max=options.depth_max,
                nsl_to_time=nsl_to_time,
                output_filename=options.output_filename,
                output_dpi=options.output_dpi,
            )

            return

        target_group_paths_avail.append(name)

        die(
            'no target group with path "%s" found. Available: %s'
            % (target_group_path, ", ".join(target_group_paths_avail))
        )


def command_upgrade_config(args):
    def setup(parser):
        parser.add_option(
            "--diff",
            dest="diff",
            action="store_true",
            help="create diff between normalized old and new versions",
        )

    parser, options, args = cl_parse("upgrade-config", args, setup)
    if len(args) != 1:
        help_and_die(parser, "missing argument <configfile>")

    from silvertine import upgrade

    upgrade.upgrade_config_file(args[0], diff=options.diff)


def command_diff(args):
    def setup(parser):
        pass

    parser, options, args = cl_parse("diff", args, setup)
    if len(args) != 2:
        help_and_die(parser, "requires exactly two arguments")

    from silvertine.config import diff_configs

    diff_configs(*args)


def command_version(args):
    def setup(parser):
        parser.add_option(
            "--short",
            dest="short",
            action="store_true",
            help="only print silvertine's version number",
        )
        parser.add_option(
            "--failsafe",
            dest="failsafe",
            action="store_true",
            help="do not get irritated when some dependencies are missing",
        )

    parser, options, args = cl_parse("version", args, setup)

    if options.short:
        print(silvertine.__version__)
        return

    elif not options.failsafe:
        from silvertine import info
        print(info.version_info())
        return

    print("silvertine: %s" % silvertine.__version__)

    try:
        import pyrocko

        print("pyrocko: %s" % pyrocko.long_version)
    except ImportError:
        print("pyrocko: N/A")

    try:
        import numpy

        print("numpy: %s" % numpy.__version__)
    except ImportError:
        print("numpy: N/A")

    try:
        import scipy

        print("scipy: %s" % scipy.__version__)
    except ImportError:
        print("scipy: N/A")

    try:
        import matplotlib

        print("matplotlib: %s" % matplotlib.__version__)
    except ImportError:
        print("matplotlib: N/A")

    try:
        from pyrocko.gui.qt_compat import Qt

        print("PyQt: %s" % Qt.PYQT_VERSION_STR)
        print("Qt: %s" % Qt.QT_VERSION_STR)
    except ImportError:
        print("PyQt: N/A")
        print("Qt: N/A")

    import sys

    print("python: %s.%s.%s" % sys.version_info[:3])

    if not options.failsafe:
        die("fell back to failsafe version printing")


if __name__ == "__main__":
    main()
