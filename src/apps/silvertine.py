from __future__ import print_function, absolute_import
import sys
from os.path import join as pjoin
import os.path as op
import os
from multiprocessing import Pool
import logging
from optparse import OptionParser, OptionValueError, IndentedHelpFormatter
from io import StringIO
import tempfile
import time
from pathlib import Path
import arrow
import shutil

from silvertine.setup_info import version as __version__
import silvertine
try:
    from pyrocko import util, marker, model
    from pyrocko import pile as pile_mod
    from pyrocko.gui.snuffler_app import *
    from pyrocko.io import quakeml

except ImportError:
    print(
        "Pyrocko is required for silvertine!"
        "Go to https://pyrocko.org/ for installation instructions."
    )


logger = logging.getLogger("silvertine.main")
km = 1e3


class Color:
    PURPLE = "\033[95m"
    CYAN = "\033[96m"
    DARKCYAN = "\033[36m"
    BLUE = "\033[94m"
    GREEN = "\033[92m"
    YELLOW = "\033[93m"
    RED = "\033[91m"
    BOLD = "\033[1m"
    UNDERLINE = "\033[4m"
    END = "\033[0m"


def remove_outdated_wc(path, factor, scale="minutes", wc="*", factor2=None):
    if scale == "minutes":
        criticalTime = arrow.now().shift(minutes=-factor)
    elif scale == "seconds":
        criticalTime = arrow.now().shift(seconds=-factor)
    elif scale == "minutes-seconds":
        criticalTime = arrow.now().shift(minutes=-factor).shift(seconds=-factor2)

    for item in Path(path).glob(wc):
        itemTime = arrow.get(item.stat().st_mtime)
        if itemTime < criticalTime:
            try:
                os.remove(item.absolute())
            except:
                shutil.rmtree(item.absolute())


def d2u(d):
    if isinstance(d, dict):
        return dict((k.replace("-", "_"), v) for (k, v) in d.items())
    else:
        return d.replace("-", "_")

def check_options(options):
    if options.load is not False:
        options.load = True
    if options.train_model is not True:
        options.train_model = False
    if options.detector_only is not False:
        options.detector_only = True
    if options.mode is "detector_only":
        options.detector_only = True
    if options.detector_only is not False:
        options.detector_only = True
    if options.download is not False:
        options.download = True
    return options


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
    "forward": "run forward modelling",
    "harvest": "manually run harvesting",
    "supply": "deliver waveforms",
    "download_raw": "download waveforms",
    "cluster": "run cluster analysis on result ensemble",
    "plot": "plot optimisation result",
    "movie": "visualize optimiser evolution",
    "export": "export results",
    "tag": "add user-defined label to run directories",
    "report": "create result report",
    "shmconv": "Seismic handler compat",
    "diff": "compare two configs or other normalized silvertine YAML files",
    "qc-polarization": "check sensor orientations with polarization analysis",
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
    "go": "go <configfile> <eventnames> ... [options]",
    "forward": (
        "forward <rundir> [options]",
        "forward <configfile> <eventnames> ... [options]",
    ),
    "harvest": "harvest <rundir> [options]",
    "cluster": (
        "cluster <method> <rundir> [options]",
        "cluster <clusteringconfigfile> <rundir> [options]",
    ),
    "plot": (
        "plot <plotnames> ( <rundir> | <configfile> <eventname> ) [options]",
        "plot all ( <rundir> | <configfile> <eventname> ) [options]",
        "plot <plotconfigfile> ( <rundir> | <configfile> <eventname> ) [options]",  # noqa
        "plot list ( <rundir> | <configfile> <eventname> ) [options]",
        "plot config ( <rundir> | <configfile> <eventname> ) [options]",
    ),
    "movie": "movie <rundir> <xpar> <ypar> <filetemplate> [options]",
    "export": "export (best|mean|ensemble|stats) <rundirs> ... [options]",
    "tag": ("tag add <tag> <rundir>", "tag remove <tag> <rundir>", "tag list <rundir>"),
    "report": ("report <rundir> ... [options]", "report <configfile> <eventnames> ..."),
    "diff": "diff <left_path> <right_path>",
    "qc-polarization": "qc-polarization <configfile> <eventname> "
    "<target_group_path> [options]",
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

silvertine is a framework for handling seiger.

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
    forward         %(forward)s
    harvest         %(harvest)s
    cluster         %(cluster)s
    plot            %(plot)s
    movie           %(movie)s
    export          %(export)s
    tag             %(tag)s
    report          %(report)s
    diff            %(diff)s
    qc-polarization %(qc_polarization)s
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


def command_detect(args):
    def setup(parser):
        parser.add_option(
            "--download_method",
            dest="download_method",
            type=str,
            default="stream",
            help="Choose download method",
        )
        parser.add_option(
            "--on_run",
            dest="on_run",
            type=str,
            default=False,
            help="Detect on live stream.",
        )
        parser.add_option(
            "--show",
            dest="show",
            type=str,
            default=False,
            help="Display progress of localisation for each event in browser",
        )
        parser.add_option(
            "--mode",
            dest="mode",
            type=str,
            default="both",
            help="Mode to to use"
        )
        parser.add_option(
            "--load",
            dest="load",
            type=str,
            default=False,
            help="Load data"
        )
        parser.add_option(
            "--train_model",
            dest="train_model",
            type=str,
            default=True,
            help="train_model",
        )
        parser.add_option(
            "--detector_only",
            dest="detector_only",
            type=str,
            default=False,
            help="Detector only mode.")
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
            "--tmax",
            dest="tmax",
            type=str,
            default=None,
            help="end")
        parser.add_option(
            "--on_stream",
            dest="on_stream",
            type=str,
            default=False,
            help="end")
        parser.add_option(
            "--config",
            help="Load a configuration file")
        parser.add_option(
            "--configs",
            help="load a comma separated list of configs and process them"
        )
        parser.add_option(
            "--train",
            action="store_true")
        parser.add_option(
            "--evaluate",
            action="store_true",
            help="Predict from input of `evaluation_data_generator` in config.",
        )
        parser.add_option(
            "--evaluate-errors",
            action="store_true",
            help="Predict errors input of `evaluation_data_generator` in config.",
        )
        parser.add_option(
            "--annotate",
            action="store_true",
            help="Add labels in error evaluation plots.",
        )
        parser.add_option(
            "--predict",
            action="store_true",
            help="Predict from input of `predict_data_generator` in config.",
        )
        parser.add_option(
            "--detect",
            action="store_true",
            help="Detect earthquakes")
        parser.add_option(
            "--optimize",
            metavar="FILENAME",
            help="use optimizer defined in FILENAME"
        )
        parser.add_option(
            "--write-tfrecord",
            metavar="FILENAME",
            help="write data_generator out to FILENAME",
        )
        parser.add_option(
            "--from-tfrecord",
            metavar="FILENAME",
            help="read tfrecord")
        parser.add_option("--new-config")
        parser.add_option(
            "--clear",
            help="delete remaints of former runs",
            action="store_true"
        )
        parser.add_option(
            "--show-data",
            type=int,
            metavar="N",
            help="show N data examples. Call with `--debug` to get plot figures with additional information.",
        )
        parser.add_option(
            "--nskip",
            type=int,
            help="For plotting. Examples to skip.")
        parser.add_option(
            "--ngpu",
            help="number of GPUs to use")
        parser.add_option(
            "--gpu-no",
            help="GPU number to use",
            type=int)
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
            "--pre_load",
            dest="pre_load",
            type=str,
            default=True,
            help="Pre-load the EQT model")
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
            "--stations_file",
            dest="stations_file",
            type=str,
            default="/stations_landau.txt",
            help="Pyrocko format station file")


    from silvertine import detector, locate
    from silvertine import seiger_lassie as lassie
    from silvertine.util import waveform
    from pyrocko import model
    parser, options, args = cl_parse("detect", args, setup)

    options = check_options(options)

    if options.pre_load is True:
        from silvertine.util import eqt_util
        model_eqt = eqt_util.load_eqt_model()
    else:
        model_eqt = []
    if options.on_run is False:
        if options.mode == "transformer":
            detector.picker.main(
                options.path,
                tmin=options.tmin,
                tmax=options.tmax,
                minlat=49.0,
                maxlat=49.979,
                minlon=7.9223,
                maxlon=8.9723,
                channels=["EH" + "[ZNE]"],
                client_list=[
                    "http://192.168.11.220:8080",
                    "http://ws.gpi.kit.edu",
                    "http://eida.gfz-potsdam.de",
                ],
                path_waveforms=options.data_dir,
                download=options.download,
                tinc=options.tinc,
                freq=options.freq,
                hf=options.hf,
                lf=options.lf,
            )

        if options.mode == "detect" or options.mode == "locate":
            detector.locator.locator.main()

        if options.mode == "BNN":
            detector.bnn.bnn_detector(
                load=options.load,
                train_model=options.train_model,
                detector_only=options.detector_only,
                data_dir=options.data_dir,
                validation_data=options.validation_data,
                wanted_start=options.wanted_start,
                wanted_end=options.wanted_end,
                mode=options.mode,
            )
    else:
        if options.mode == "both":
            piled = pile_mod.make_pile()
            sources_list = [
                "seedlink://eida.bgr.de/GR.INS*.*.EH*",
                "seedlink://eida.bgr.de/GR.TMO*.*.EH*",
            ]  # seedlink sources
            pollinjector = None
            tempdir = None
            if options.store_path is not None:
                store_path_base_down = options.store_path+"/download-tmp"
                store_path_base = options.store_path
            else:
                store_path_base_down = "."
                store_path_base = "."

            stations = model.load_stations(store_path_base+options.stations_file)

            if sources_list:
                if store_path_base_down is None:
                    tempdir = tempfile.mkdtemp("", "snuffler-tmp-")
                    store_path = pjoin(
                        tempdir,
                        "trace-%(network)s.%(station)s.%(location)s.%(channel)s."
                        "%(tmin_ms)s.mseed",
                    )
                elif os.path.isdir(store_path_base_down):
                    store_path = pjoin(
                        store_path_base_down,
                        "trace-%(network)s.%(station)s.%(location)s.%(channel)s."
                        "%(tmin_ms)s.mseed",
                    )
                _injector = pile_mod.Injector(
                    piled,
                    path=store_path,
                    fixation_length=options.store_interval,
                    forget_fixed=True)

                # Data is downloaded continously after starting the stream
                if options.download_method is "stream":
                    sources = setup_acquisition_sources(sources_list)
                    for source in sources:
                        source.start()

                diff = 0 # for keeping track of time between data saving
                models = []
                events_eqt = []
                events_stacking = []
                process_in_progress = True

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
                    config_path = options.config
                    config = lassie.read_config(config_path)
                    pool = Pool(processes=2)
                    pool.apply_async(lassie.search(config,
                                                   override_tmin=options.tmin,
                                                   override_tmax=options.tmax,
                                                   force=True,
                                                   show_detections=True,
                                                   nparallel=10))
                    pool.apply_async(detector.picker.main(
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
                        path_waveforms=store_path_base_down,
                        download=options.download,
                        tinc=options.tinc,
                        freq=options.freq,
                        hf=options.hf,
                        lf=options.lf,
                        models=[model_eqt],
                    ))
                    pool.close()
                    pool.join()
                    end = time.time()
                    diff = end - start
                    # if detection make fine location and output here

                    try:
                        try:
                            events_stacking = model.load_events(store_path_base+"stacking_events.pf")
                        except:
                            events_stacking = []
                        events_stacking.extend(model.load_events(config.get_events_path()))
                        model.dump_events(events_stacking, store_path_base+"stacking_events.pf")
                    except:
                        pass
                    try:
                        events_eqt = model.load_events(store_path_base+"eqt_events.pf")
                    except:
                        events_eqt = []
                    for item in Path(store_path_base).glob("asociation_*/events.pf"):
                        events_eqt.extend(model.load_events(item.absolute()))
                    model.dump_events(events_eqt, store_path_base+"eqt_events.pf")
                    for event in events_stacking:
                        savedir = store_path_base + '/stacking_detections/' + str(event.time) + '/'
                        if not os.path.exists(savedir):
                            os.makedirs(savedir)
                            os.makedirs(savedir+"figures_stacking")
                        for item in Path(config.path_prefix+"/"+config.run_path+"/figures").glob("*"):
                            try:
                                time_item = util.stt(str(item.absolute())[-27:-17]+" "+str(item.absolute())[-16:-4])
                                if event.time-10 < time_item and event.time+10 > time_item:
                                    os.system("cp %s %s" % (item.absolute(), savedir+"figures_stacking"))
                                    plot_waveforms = False
                                    if plot_waveforms is True:
                                        pile_event = pile.make_pile(store_path_base_down+"download-tmp", show_progress=False)
                                        for traces in pile_event.chopper(tmin=event.time-10,
                                                                         tmax=event.time+10,
                                                                         keep_current_files_open=False,
                                                                         want_incomplete=True):
                                            waveform.plot_waveforms(traces, event, stations,
                                                                    savedir, None)
                            except:
                                pass

                    for event in events_eqt:
                        savedir = store_path_base + '/eqt_detections/' + str(event.time) + '/'
                        if not os.path.exists(savedir):
                            os.makedirs(savedir)
                            os.makedirs(savedir+"figures_eqt")
                        for item in Path(store_path_base+"/").glob("asociation_*/associations.xml"):
                            qml = quakeml.QuakeML.load_xml(filename=item)
                            events_qml = qml.get_pyrocko_events()
                            for i, eq in enumerate(events_qml):
                                if event.time == eq.time:
                                    evqml = events_qml.get_events()[i]
                                    evqml.dump_xml(filename=savedir+"phases_eqt.qml")
                        for item in Path(store_path_base+"/").glob("detections_*/*/figures/*"):
                            try:
                                time_item = util.stt(str(item.absolute())[-31:-21]+" "+str(item.absolute())[-20:-5])
                                if event.time-options.wait_period < time_item and event.time+options.wait_period > time_item:
                                    os.system("cp %s %s" % (item.absolute(), savedir+"figures_eqt"))
                            except:
                                pass

                    for event_stack in events_stacking:
                        for event_eqt in events_eqt:
                            if event_stack.time-10 < event_eqt.time and event_stack.time+10 > event_eqt.time:
                                savedir = store_path_base + '/combined_detections/' + util.tts(event_stack.time) + '/'
                                if not os.path.exists(savedir):
                                    os.makedirs(savedir)

                                # lassie.search(config_fine,
                                #                override_tmin=options.tmin,
                                #                override_tmax=options.tmax,
                                #                force=True,
                                #                show_detections=True,
                                #                nparallel=10)

                    remove_outdated_wc(store_path_base+"/download-tmp",
                                       3.5,
                                       wc="*")
                    remove_outdated_wc(store_path_base,
                                       1,
                                       wc="detections_*")
                    remove_outdated_wc(store_path_base,
                                       1,
                                       wc="asociation_*")

                    for item in Path(store_path_base+"/downloads/").glob("*"):
                        shutil.rmtree(item.absolute())
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


def command_monitor(args):
    def setup(parser):
        parser.add_option(
            "--show",
            dest="show",
            type=str,
            default=False,
            help="Display progress of localisation for each event in browser",
        )

    parser, options, args = cl_parse("monitor", args, setup)

    from silvertine.monitoring import stream

    stream.live_steam()


def command_beam(args):
    def setup(parser):
        parser.add_option(
            "--show",
            dest="show",
            type=str,
            default=False,
            help="overwrite existing project folder.",
        )
        parser.add_option(
            "--nevents",
            dest="nevents",
            type=int,
            default=1,
            help="Number of events to locate (default: %default)",
        )

    parser, options, args = cl_parse("locate", args, setup)

    from silvertine.beam_depth import abedeto

    project_dir = args[0]
    if options.show is not False:
        options.show = True
    silvertine.beam_depth.abedeto.beam(
        project_dir, show=options.show, n_tests=options.nevents
    )


def command_plot_prod(args):
    def setup(parser):
        parser.add_option(
            "--show",
            dest="show",
            type=str,
            default=False,
            help="overwrite existing project folder.",
        )

    parser, options, args = cl_parse("plot_prod", args, setup)

    from silvertine.util import prod_data

    if options.show is not False:
        options.show = True
    prod_data.plot_insheim_prod_data()


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
        #    try:
        from pathlib import Path

        pathlist = Path(options.folder).glob("*_*")
        for path in pathlist:
            mod = cake.load_model(str(path))
            mods.append(mod)
        #    except:
        #        pass
        fig, axes = silvertine_plot.bayesian_model_plot(
            mods, axes=None, highlightidx=[0, 1, 2, 3, 4]
        )

    if options.show is not False:
        plt.show()


def command_analyse_statistics(args):
    def setup(parser):
        parser.add_option(
            "--show",
            dest="show",
            type=str,
            default=False,
            help="Show plots or only save them.",
        )
        parser.add_option(
            "--catalog",
            dest="catalog",
            type=str,
            default="ler",
            help="Analyse statistics of fixed catalog.",
        )
        parser.add_option(
            "--data_folder",
            dest="data_folder",
            type=str,
            default=False,
            help="Data folder.",
        )
        parser.add_option("--start", dest="start", type=int, default=0, help="start.")
        parser.add_option("--end", dest="end", type=int, default=None, help="start.")

    parser, options, args = cl_parse("analyse_statistics", args, setup)

    from silvertine.util import prod_data, silvertine_meta, stats
    from pyrocko import model

    if options.show is not False:
        options.show = True
    if options.data_folder is False:
        options.data_folder = "data"
    if options.catalog is "geres":
        (
            events,
            pyrocko_stations,
            ev_dict_list,
            ev_list_picks,
        ) = silvertine_meta.load_data(options.data_folder, nevent=options.end)
    if options.catalog is "ler":
        events = model.load_events("data/events_ler.pf")

    distances, time_rel, msd = stats.calcuate_msd(events)
    # prod_data.plot_insheim_prod_data()


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

        mod = cake.load_model(folder + options.mod)

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


def command_beam_process(args):
    def setup(parser):
        parser.add_option(
            "--show",
            dest="show",
            type=str,
            default=False,
            help="overwrite existing project folder.",
        )
        parser.add_option(
            "--nevents",
            dest="nevents",
            type=int,
            default=1,
            help="Number of events to locate (default: %default)",
        )

    parser, options, args = cl_parse("locate", args, setup)

    from silvertine.beam_depth import abedeto

    project_dir = args[0]
    if options.show is not False:
        options.show = True
    silvertine.beam_depth.abedeto.beam(
        project_dir, show=options.show, n_tests=options.nevents
    )

    import argparse

    parser = argparse.ArgumentParser("What was the depth, again?", add_help=False)
    parser.add_option("--log", required=False, default="INFO")

    sp = parser.add_subparsers(dest="cmd")

    process_parser = sp.add_parser("beam_process", help="Create images")
    process_parser.add_option(
        "projects", help='default "all available"', nargs="*", default="."
    )
    process_parser.add_option(
        "--array-id",
        dest="array_id",
        help="array-id to process",
        required=False,
        default=False,
    )
    process_parser.add_option(
        "--settings", help="settings file", default=False, required=False
    )
    process_parser.add_option(
        "--cc_align", help="dummy argument at the moment", required=False
    )
    process_parser.add_option(
        "--store-superdirs",
        help="super directory where to look for stores",
        dest="store_superdirs",
        nargs="*",
        default=["stores"],
        required=False,
    )
    process_parser.add_option(
        "--store", help="name of store id", dest="store_id", required=False
    )
    process_parser.add_option(
        "--depth", help="assumed source depth [km]", required=False
    )
    process_parser.add_option(
        "--depths",
        help="testing depths in km. zstart:zstop:delta, default 0:15:1",
        default="0:15:1",
        required=False,
    )
    process_parser.add_option(
        "--quantity",
        help="velocity|displacement",
        choices=["velocity", "displacement", "restituted"],
        required=False,
    )
    process_parser.add_option(
        "--filter", help='4th order butterw. default: "0.7:4.5"', required=False
    )
    process_parser.add_option(
        "--correction", required=False, help="a global correction in time [s]"
    )
    process_parser.add_option(
        "--gain", required=False, help="gain factor", default=1.0, type=float
    )
    process_parser.add_option(
        "--zoom",
        required=False,
        help="time window to look at. default -7:15",
        default="-7:15",
    )

    process_parser.add_option(
        "--normalize", help="normalize traces to 1", action="store_true", required=False
    )
    process_parser.add_option(
        "--skip-true",
        help="if true, do not plot recorded and the assigned synthetic trace on top of each other",
        dest="skip_true",
        action="store_true",
        required=False,
    )
    process_parser.add_option(
        "--show",
        help="show matplotlib plots after each step",
        action="store_true",
        required=False,
    )
    process_parser.add_option(
        "--force-nearest-neighbor",
        help="handles OOB",
        dest="force_nearest_neighbor",
        default=False,
        action="store_true",
        required=False,
    )
    process_parser.add_option(
        "--auto-caption",
        help="Add a caption to figure with basic info",
        dest="auto_caption",
        default=False,
        action="store_true",
        required=False,
    )
    process_parser.add_option(
        "--out-filename", help="file to store image", dest="save_as", required=False
    )
    process_parser.add_option(
        "--print-parameters",
        dest="print_parameters",
        help="creates a text field giving the used parameters",
        required=False,
    )
    process_parser.add_option(
        "--title", dest="title", help="template for title.", required=False
    )
    process_parser.add_option(
        "--overwrite-settings",
        dest="overwrite_settings",
        help="overwrite former settings files",
        default=False,
        action="store_true",
        required=False,
    )
    args = parser.parse_args()
    silvertine.beam_depth.abedeto.process(
        args, project_dir, show=options.show, n_tests=options.nevents
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


def command_init(args):

    from .cmd_init import silvertineInit

    silvertine_init = silvertineInit()

    def print_section(entries):
        if len(entries) == 0:
            return "\tNone available."

        padding = max([len(n) for n in entries.keys()])
        rstr = []
        lcat = None
        for name, desc in entries.items():

            cat = name.split("_")[0]
            if lcat is not None and lcat != cat:
                rstr.append("")
            lcat = cat

            rstr.append(
                "    {c.BOLD}{name:<{padding}}{c.END} : {desc}".format(
                    name=name, desc=desc, padding=padding, c=Color
                )
            )
        return "\n".join(rstr)

    help_text = """Available configuration examples for silvertine.

{c.BOLD}Example Projects{c.END}

    Deploy a full project structure into a directory.

    usage: silvertine init <example> <projectdir>

    where <example> is any of the following:

{examples_list}

{c.BOLD}Config Sections{c.END}

    Print out configuration snippets for various components.

    usage: silvertine init <section>

    where <section> is any of the following:

{sections_list}
""".format(
        c=Color,
        examples_list=print_section(silvertine_init.get_examples()),
        sections_list=print_section(silvertine_init.get_sections()),
    )

    def setup(parser):
        parser.add_option("--force", dest="force", action="store_true")

    parser, options, args = cl_parse(
        "init", args, setup, "Use silvertine init list to show available examples."
    )

    if len(args) not in (1, 2):
        help_and_die(parser, "1 or 2 arguments required")

    if args[0] == "list":
        print(help_text)
        return

    if args[0].startswith("example_"):
        if len(args) == 1:
            config = silvertine_init.get_content_example(args[0])
            if not config:
                help_and_die(parser, "Unknown example: %s" % args[0])

            sys.stdout.write(config + "\n\n")

            logger.info(
                "Hint: To create a project, use: silvertine init <example> "
                "<projectdir>".format(c=Color, example=args[0])
            )

        elif op.exists(op.abspath(args[1])) and not options.force:
            help_and_die(
                parser,
                'Directory "%s" already exists! Use --force to overwrite.' % args[1],
            )
        else:
            try:
                silvertine_init.init_example(args[0], args[1], force=options.force)
            except OSError as e:
                print(str(e))

    else:
        sec = silvertine_init.get_content_snippet(args[0])
        if not sec:
            help_and_die(parser, "Unknown snippet: %s" % args[0])

        sys.stdout.write(sec)


def command_init_old(args):

    from . import cmd_init as init

    def setup(parser):
        parser.add_option(
            "--targets",
            action="callback",
            dest="targets",
            type=str,
            callback=multiple_choice,
            callback_kwargs={"choices": ("waveforms", "gnss", "insar", "all")},
            default="waveforms",
            help="select from:"
            " waveforms, gnss and insar. "
            "(default: --targets=%default,"
            " multiple selection by --targets=waveforms,gnss,insar)",
        )
        parser.add_option(
            "--problem",
            dest="problem",
            type="choice",
            choices=["cmt", "rectangular"],
            help="problem to generate: 'dc' (double couple)"
            " or'rectangular' (rectangular finite fault)"
            " (default: '%default')",
        )
        parser.add_option(
            "--force",
            dest="force",
            action="store_true",
            help="overwrite existing project folder",
        )

    parser, options, args = cl_parse("init", args, setup)

    try:
        project = init.silvertineProject()

        if "all" in options.targets:
            targets = ["waveforms", "gnss", "insar"]
        else:
            targets = options.targets

        if not options.problem:
            if "insar" in targets or "gnss" in targets:
                problem = "rectangular"
            else:
                problem = "cmt"
        else:
            problem = options.problem

        if problem == "rectangular":
            project.set_rectangular_source()
        elif problem == "cmt":
            project.set_cmt_source()

        if "waveforms" in targets:
            project.add_waveforms()

        if "insar" in targets:
            project.add_insar()

        if "gnss" in targets:
            project.add_gnss()

        if len(args) == 1:
            project_dir = args[0]
            project.build(project_dir, options.force)
            logger.info(
                CLIHints(
                    "init",
                    project_dir=project_dir,
                    config=op.join(project_dir, "config", "config.gronf"),
                )
            )
        else:
            sys.stdout.write(project.dump())

    except silvertine.silvertineError as e:
        die(str(e))


def command_events(args):
    def setup(parser):
        pass

    parser, options, args = cl_parse("events", args, setup)
    if len(args) != 1:
        help_and_die(parser, "missing arguments")

    config_path = args[0]
    try:
        config = silvertine.read_config(config_path)

        for event_name in silvertine.get_event_names(config):
            print(event_name)

    except silvertine.silvertineError as e:
        die(str(e))


def command_check(args):

    from silvertine.environment import Environment

    def setup(parser):
        parser.add_option(
            "--target-ids",
            dest="target_string_ids",
            metavar="TARGET_IDS",
            help="process only selected targets. TARGET_IDS is a "
            "comma-separated list of target IDs. Target IDs have the "
            "form SUPERGROUP.GROUP.NETWORK.STATION.LOCATION.CHANNEL.",
        )

        parser.add_option(
            "--waveforms",
            dest="show_waveforms",
            action="store_true",
            help="show raw, restituted, projected, and processed waveforms",
        )

        parser.add_option(
            "--nrandom",
            dest="n_random_synthetics",
            metavar="N",
            type=int,
            default=10,
            help="set number of random synthetics to forward model (default: "
            "10). If set to zero, create synthetics for the reference "
            "solution.",
        )

        parser.add_option(
            "--save-stations-used",
            dest="stations_used_path",
            metavar="FILENAME",
            help="aggregate all stations used by the setup into a file",
        )

    parser, options, args = cl_parse("check", args, setup)
    if len(args) < 1:
        help_and_die(parser, "missing arguments")

    try:
        env = Environment(args)
        config = env.get_config()

        target_string_ids = None
        if options.target_string_ids:
            target_string_ids = options.target_string_ids.split(",")

        silvertine.check(
            config,
            event_names=env.get_selected_event_names(),
            target_string_ids=target_string_ids,
            show_waveforms=options.show_waveforms,
            n_random_synthetics=options.n_random_synthetics,
            stations_used_path=options.stations_used_path,
        )

        logger.info(CLIHints("check", config=env.get_config_path()))

    except silvertine.silvertineError as e:
        die(str(e))


def command_go(args):

    from silvertine.environment import Environment

    def setup(parser):
        parser.add_option(
            "--force",
            dest="force",
            action="store_true",
            help="overwrite existing run directory",
        )
        parser.add_option(
            "--preserve",
            dest="preserve",
            action="store_true",
            help="preserve old rundir",
        )
        parser.add_option(
            "--status",
            dest="status",
            default="state",
            type="choice",
            choices=["state", "quiet"],
            help="status output selection (choices: state, quiet, default: " "state)",
        )
        parser.add_option(
            "--parallel",
            dest="nparallel",
            type=int,
            default=1,
            help="set number of events to process in parallel, "
            "if set to more than one, --status=quiet is implied.",
        )
        parser.add_option(
            "--threads",
            dest="nthreads",
            type=int,
            default=1,
            help="set number of threads per process (default: 1). "
            "Set to 0 to use all available cores.",
        )

    parser, options, args = cl_parse("go", args, setup)

    try:
        env = Environment(args)

        status = options.status
        if options.nparallel != 1:
            status = "quiet"

        silvertine.go(
            env,
            force=options.force,
            preserve=options.preserve,
            status=status,
            nparallel=options.nparallel,
            nthreads=options.nthreads,
        )
        if len(env.get_selected_event_names()) == 1:
            logger.info(CLIHints("go", rundir=env.get_rundir_path()))

    except silvertine.silvertineError as e:
        die(str(e))


def command_forward(args):

    from silvertine.environment import Environment

    def setup(parser):
        pass

    parser, options, args = cl_parse("forward", args, setup)
    if len(args) < 1:
        help_and_die(parser, "missing arguments")

    try:
        env = Environment(args)
        silvertine.forward(env)
    except silvertine.silvertineError as e:
        die(str(e))


def command_harvest(args):
    def setup(parser):
        parser.add_option(
            "--force",
            dest="force",
            action="store_true",
            help="overwrite existing harvest directory",
        )
        parser.add_option(
            "--neach",
            dest="neach",
            type=int,
            default=10,
            help="take NEACH best samples from each chain (default: %default)",
        )
        parser.add_option(
            "--weed",
            dest="weed",
            type=int,
            default=0,
            help="weed out bootstrap samples with bad global performance. "
            "0: no weeding (default), "
            "1: only bootstrap chains where all NEACH best samples "
            "global misfit is less than the global average misfit of all "
            "NEACH best in all chains plus one standard deviation are "
            "included in the harvest ensemble, "
            "2: same as 1 but additionally individual samples are "
            "removed if their global misfit is greater than the global "
            "average misfit of all NEACH best in all chains, "
            "3: harvesting is done on the global chain only, bootstrap "
            "chains are excluded",
        )

    parser, options, args = cl_parse("harvest", args, setup)
    if len(args) != 1:
        help_and_die(parser, "no rundir")

    (run_path,) = args
    silvertine.harvest(
        run_path, force=options.force, nbest=options.neach, weed=options.weed
    )


def command_cluster(args):
    from silvertine import Clustering
    from silvertine.clustering import metrics, methods, read_config, write_config

    def setup(parser):
        parser.add_option(
            "--metric",
            dest="metric",
            metavar="METRIC",
            default="kagan_angle",
            choices=metrics.metrics,
            help="metric to measure model distances. Choices: [%s]. Default: "
            "kagan_angle" % ", ".join(metrics.metrics),
        )

        parser.add_option(
            "--write-config",
            dest="write_config",
            metavar="FILE",
            help="write configuration (or default configuration) to FILE",
        )

    method = args[0] if args else ""
    try:
        parser, options, args = cl_parse(
            "cluster",
            args[1:],
            setup=Clustering.cli_setup(method, setup),
            details="Available clustering methods: [%s]. Use "
            '"silvertine cluster <method> --help" to get list of method'
            "dependent options." % ", ".join(methods),
        )

        if method not in Clustering.name_to_class and not op.exists(method):
            help_and_die(
                parser,
                "no such clustering method: %s" % method
                if method
                else "no clustering method specified",
            )

        if op.exists(method):
            clustering = read_config(method)
        else:
            clustering = Clustering.cli_instantiate(method, options)

        if options.write_config:
            write_config(clustering, options.write_config)
        else:
            if len(args) != 1:
                help_and_die(parser, "no rundir")
            (run_path,) = args

            silvertine.cluster(run_path, clustering, metric=options.metric)

    except silvertine.silvertineError as e:
        die(str(e))


def command_plot(args):
    def setup(parser):
        parser.add_option(
            "--show",
            dest="show",
            action="store_true",
            help="show plot for interactive inspection",
        )

    details = ""

    parser, options, args = cl_parse("plot", args, setup, details)

    if not options.show:
        import matplotlib

        matplotlib.use("Agg")

    from silvertine.environment import Environment

    if len(args) not in (1, 2, 3):
        help_and_die(parser, "1, 2 or 3 arguments required")

    if len(args) > 1:
        env = Environment(args[1:])
    else:
        env = None

    from silvertine import plot

    if args[0] == "list":

        def get_doc_title(doc):
            for ln in doc.split("\n"):
                ln = ln.strip()
                if ln != "":
                    ln = ln.strip(".")
                    return ln
            return "Undocumented."

        if env:
            plot_classes = env.get_plot_classes()
        else:
            plot_classes = plot.get_all_plot_classes()

        plot_names, plot_doc = zip(*[(pc.name, pc.__doc__) for pc in plot_classes])

        plot_descs = [get_doc_title(doc) for doc in plot_doc]
        left_spaces = max([len(pn) for pn in plot_names])

        for name, desc in zip(plot_names, plot_descs):
            print("{name:<{ls}} - {desc}".format(ls=left_spaces, name=name, desc=desc))

    elif args[0] == "config":
        plot_config_collection = plot.get_plot_config_collection(env)
        print(plot_config_collection)

    elif args[0] == "all":
        if env is None:
            help_and_die(parser, "two or three arguments required")
        plot_names = plot.get_plot_names(env)
        plot.make_plots(env, plot_names=plot_names, show=options.show)

    elif op.exists(args[0]):
        if env is None:
            help_and_die(parser, "two or three arguments required")
        plots = plot.PlotConfigCollection.load(args[0])
        plot.make_plots(env, plots, show=options.show)

    else:
        if env is None:
            help_and_die(parser, "two or three arguments required")
        plot_names = [name.strip() for name in args[0].split(",")]
        plot.make_plots(env, plot_names=plot_names, show=options.show)


def command_movie(args):

    import matplotlib

    matplotlib.use("Agg")

    def setup(parser):
        pass

    parser, options, args = cl_parse("movie", args, setup)

    if len(args) != 4:
        help_and_die(parser, "four arguments required")

    run_path, xpar_name, ypar_name, movie_filename_template = args

    from silvertine import plot

    movie_filename = movie_filename_template % {"xpar": xpar_name, "ypar": ypar_name}

    try:
        plot.make_movie(run_path, xpar_name, ypar_name, movie_filename)

    except silvertine.silvertineError as e:
        die(str(e))


def command_export(args):
    def setup(parser):
        parser.add_option(
            "--type",
            dest="type",
            metavar="TYPE",
            choices=("event", "event-yaml", "source", "vector"),
            help="select type of objects to be exported. Choices: "
            '"event" (default), "event-yaml", "source", "vector".',
        )

        parser.add_option(
            "--parameters",
            dest="parameters",
            metavar="PLIST",
            help="select parameters to be exported. PLIST is a "
            "comma-separated list where each entry has the form "
            '"<parameter>[.<measure>]". Available measures: "best", '
            '"mean", "std", "minimum", "percentile16", "median", '
            '"percentile84", "maximum".',
        )

        parser.add_option(
            "--selection",
            dest="selection",
            metavar="EXPRESSION",
            help="only export data for runs which match EXPRESSION. "
            'Example expression: "tags_contains:excellent,good"',
        )

        parser.add_option(
            "--output", dest="filename", metavar="FILE", help="write output to FILE"
        )

    parser, options, args = cl_parse("export", args, setup)
    if len(args) < 2:
        help_and_die(parser, "arguments required")

    what = args[0]

    dirnames = args[1:]

    what_choices = ("best", "mean", "ensemble", "stats")

    if what not in what_choices:
        help_and_die(
            parser,
            "invalid choice: %s (choose from %s)"
            % (repr(what), ", ".join(repr(x) for x in what_choices)),
        )

    if options.parameters:
        pnames = options.parameters.split(",")
    else:
        pnames = None

    try:
        silvertine.export(
            what,
            dirnames,
            filename=options.filename,
            type=options.type,
            pnames=pnames,
            selection=options.selection,
        )

    except silvertine.silvertineError as e:
        die(str(e))


def command_tag(args):
    def setup(parser):
        parser.add_option(
            "-d",
            "--dir-names",
            dest="show_dirnames",
            action="store_true",
            help="show directory names instead of run names",
        )

    parser, options, args = cl_parse("tag", args, setup)
    if len(args) < 2:
        help_and_die(parser, "two or more arguments required")

    action = args.pop(0)

    if action not in ("add", "remove", "list"):
        help_and_die(parser, "invalid action: %s" % action)

    if action in ("add", "remove"):
        if len(args) < 2:
            help_and_die(parser, "three or more arguments required")

        tag = args.pop(0)

        rundirs = args

    if action == "list":
        rundirs = args

    from silvertine.environment import Environment

    errors = False
    for rundir in rundirs:
        try:
            env = Environment([rundir])
            if options.show_dirnames:
                name = rundir
            else:
                name = env.get_problem().name

            info = env.get_run_info()
            if action == "add":
                info.add_tag(tag)
                env.set_run_info(info)
            elif action == "remove":
                info.remove_tag(tag)
                env.set_run_info(info)
            elif action == "list":
                print("%-60s : %s" % (name, ", ".join(info.tags)))

        except silvertine.silvertineError as e:
            errors = True
            logger.error(e)

    if errors:
        die("Errors occurred, see log messages above.")


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
