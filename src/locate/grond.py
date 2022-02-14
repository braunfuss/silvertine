from __future__ import absolute_import
import os
import shutil

from pyrocko import guts, model

from grond import config
from grond import Environment

from silvertine.locate import common
from silvertine.locate.common import grond, chdir
from pyrocko.io import quakeml as q
from pyrocko.util import str_to_time as stt


_multiprocess_can_split = True


def locate(marker_file, gf_stores_path, scenario_dir, config_path, stations_path, event_name):
    playground_dir = common.get_playground_dir()

    with chdir(playground_dir):

        with chdir(scenario_dir):
            quick_config_path = scenario_dir + 'scenario_quick.gronf'
            event_names = grond('events', config_path).strip().split('\n')
            env = Environment([config_path] + event_names)
            conf = env.get_config()
            event_names = [event_name]

            mod_conf = conf.clone()
            target_groups = guts.load_all(string='''
--- !grond.PhasePickTargetGroup
normalisation_family: 'pick'
path: 'pick.p'
distance_min: 0e3
distance_max: 102000.0
pick_synthetic_traveltime: '{stored:any_P}'
pick_phasename: 'any_P'
store_id: 'landau_100hz_noweak'
depth_max: 1
--- !grond.PhasePickTargetGroup
normalisation_family: 'pick'
path: 'pick.s'
distance_min: 0e3
distance_max: 102000.0
pick_synthetic_traveltime: '{stored:any_S}'
pick_phasename: 'any_S'
store_id: 'landau_100hz_noweak'
depth_max: 1
--- !grond.PhasePickTargetGroup
normalisation_family: 'pick'
path: 'moer.p'
distance_min: 0e3
distance_max: 102000.0
pick_synthetic_traveltime: '{stored:any_P}'
pick_phasename: 'any_P'
store_id: 'moer_1hz_no_weak'
depth_min: 69
depth_max: 71
--- !grond.PhasePickTargetGroup
normalisation_family: 'pick'
path: 'moer.s'
distance_min: 0e3
distance_max: 102000.0
pick_synthetic_traveltime: '{stored:any_S}'
pick_phasename: 'any_S'
store_id: 'moer_1hz_no_weak'
depth_min: 69
depth_max: 71
--- !grond.PhasePickTargetGroup
normalisation_family: 'pick'
path: 'lde.p'
distance_min: 0e3
distance_max: 102000.0
pick_synthetic_traveltime: '{stored:any_P}'
pick_phasename: 'any_P'
store_id: 'lde_100hz_no_weak'
depth_min: 149
depth_max: 151
--- !grond.PhasePickTargetGroup
normalisation_family: 'pick'
path: 'lde.s'
distance_min: 0e3
distance_max: 102000.0
pick_synthetic_traveltime: '{stored:any_S}'
pick_phasename: 'any_S'
store_id: 'lde_100hz_no_weak'
depth_min: 149
depth_max: 151
--- !grond.PhasePickTargetGroup
normalisation_family: 'pick'
path: 'rott.p'
distance_min: 0e3
distance_max: 102000.0
pick_synthetic_traveltime: '{stored:any_P}'
pick_phasename: 'any_P'
store_id: 'rott_100hz_no_weak'
depth_min: 304
depth_max: 306
--- !grond.PhasePickTargetGroup
normalisation_family: 'pick'
path: 'rott.s'
distance_min: 0e3
distance_max: 102000.0
pick_synthetic_traveltime: '{stored:any_S}'
pick_phasename: 'any_S'
store_id: 'rott_100hz_no_weak'
depth_min: 304
depth_max: 306
''')

            mod_conf.dataset_config.picks_paths = [marker_file]
            mod_conf.target_groups = target_groups
            mod_conf.path_prefix = scenario_dir
            mod_conf.dataset_config.stations_path = stations_path
            mod_conf.event_names = [event_name]
            # mod_conf.set_elements(
            #     'analyser_configs[:].niterations', 100)
            # mod_conf.set_elements(
            #     'optimiser_config.sampler_phases[:].niterations', 100)
            # mod_conf.set_elements(
            #     'optimiser_config.nbootstrap', 5)

            mod_conf.optimiser_config.sampler_phases[-1].niterations = 10000

            mod_conf.set_basepath(conf.get_basepath())
            config.write_config(mod_conf, quick_config_path)

            grond('go', quick_config_path, *event_names)
            rundir_paths = common.get_rundir_paths(
                quick_config_path, event_names)
            best = grond('export', 'best', *rundir_paths, '--output=best.pf')
            best = model.load_events(scenario_dir+"/best.pf")[0]

            #grond('report', *rundir_paths)
            return best
            # rundir_paths = common.get_rundir_paths(
            #     quick_config_path, event_names)
            # grond('report', *rundir_paths)
