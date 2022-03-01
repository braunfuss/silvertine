from __future__ import print_function, absolute_import
import arrow
from pathlib import Path
import sys
from os.path import join as pjoin
import os.path as op
import os
import subprocess
import logging
from optparse import OptionParser, OptionValueError, IndentedHelpFormatter
from io import StringIO
import shutil
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
                subprocess.call(['rm','-r'] + item.absolute())
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
