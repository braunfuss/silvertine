import math
import os
import os.path as op
import sys
import logging
import time
import copy
import sys
import numpy as num

from grond.monitor import GrondMonitor
from pyrocko import gf, util, guts
from pyrocko.parimap import parimap
from grond.optimisers.highscore import UniformSamplerPhase, DirectedSamplerPhase
from pyrocko.guts import String, Float, Dict, StringChoice, Int

import grond
logger = logging.getLogger('grond.silvertine')
#import solverscipy


class STFType(StringChoice):
    choices = ['HalfSinusoidSTF', 'ResonatorSTF']

    cls = {
        'HalfSinusoidSTF': gf.HalfSinusoidSTF,
        'ResonatorSTF': gf.ResonatorSTF}

    @classmethod
    def base_stf(cls, name):
        return cls.cls[name]()


class MTType(StringChoice):
    choices = ['full', 'deviatoric', 'dc']


def run_grond(rundir, datafolder, eventname, store_id, domain="time_domain"):
    km = 1000.
    ds = grond.Dataset(event_name=eventname)
    ds.add_waveforms(paths=['%s' % datafolder])
    ds.add_events(filename='%s/event.txt' % datafolder)
    ds.add_stations(stationxml_filenames=['%s/stations.xml' % datafolder])
    ds.add_responses(stationxml_filenames=['%s/stations.xml' % datafolder])

    ds._event_name = eventname
    print(eventname)
    ds.add_blacklist([])
    ds.empty_cache()

    quantity = 'displacement'
    group = 'all'
    tmin = '{stored:begin}'
    tmax = '{stored:begin}+0.5'
    fmin = 1
    fmax = 13.
    ffactor = 1.
    engine = gf.LocalEngine(store_superdirs=['/home/asteinbe/gf_stores'])
    gf_interpolation = 'nearest_neighbor'
    imc_Z = grond.WaveformMisfitConfig(
        fmin=fmin,
        fmax=fmax,
        ffactor=ffactor,
        tmin=tmin,
        tmax=tmax,
        norm_exponent=1,
        domain=domain,
        quantity=quantity)
    cha_Z = 'Z'

    imc_T = grond.WaveformMisfitConfig(
        fmin=fmin,
        fmax=fmax,
        ffactor=ffactor,
        tmin=tmin,
        tmax=tmax,
        norm_exponent=1,
        domain=domain,
        quantity=quantity,)
    cha_T = 'T'

    imc_R = grond.WaveformMisfitConfig(
        fmin=fmin,
        fmax=fmax,
        ffactor=ffactor,
        tmin=tmin,
        tmax=tmax,
        norm_exponent=1,
        domain=domain,
        quantity=quantity,)
    cha_R = 'R'

    event = ds.get_events()[0]
    event_origin = gf.Source(
        lat=event.lat,
        lon=event.lon)


    if event.depth is None:
        event.depth = 7*km

    distance_min = None
    distance_max = 30000.
    targets = []
    for st in ds.get_stations():
        for cha in cha_Z:
            target = grond.WaveformMisfitTarget(
                interpolation=gf_interpolation,
                store_id=store_id,
                codes=st.nsl() + (cha,),
                lat=st.lat,
                lon=st.lon,
                quantity=quantity,
                misfit_config=imc_Z,
                normalisation_family="td",
                path="Z")
            _, bazi = event_origin.azibazi_to(target)
            if cha == 'Z':
                target.azimuth = 0.
                target.dip = -90.
            target.set_dataset(ds)
            targets.append(target)
    for st in ds.get_stations():
        for cha in cha_T:
            target = grond.WaveformMisfitTarget(
                codes=st.nsl() + (cha,),
                lat=st.lat,
                lon=st.lon,
                interpolation=gf_interpolation,
                store_id=store_id,
                normalisation_family="td",
                path="T",
                quantity=quantity,
                misfit_config=imc_T)
            _, bazi = event_origin.azibazi_to(target)
            if cha == 'T':
                target.azimuth = bazi - 90.
                target.dip = 0.
            target.set_dataset(ds)
            targets.append(target)
    for st in ds.get_stations():
        for cha in cha_R:
            target = grond.WaveformMisfitTarget(
                codes=st.nsl() + (cha,),
                lat=st.lat,
                lon=st.lon,
                interpolation=gf_interpolation,
                store_id=store_id,
                normalisation_family="td",
                path="R",
                quantity=quantity,
                misfit_config=imc_R)
            _, bazi = event_origin.azibazi_to(target)
            if cha == 'R':
                target.azimuth = bazi - 180.
                target.dip = 0.
            target.set_dataset(ds)
            targets.append(target)
    base_source = gf.MTSource.from_pyrocko_event(event)
    stf_type = 'HalfSinusoidSTF'
    base_source.set_origin(event_origin.lat, event_origin.lon)
    base_source.set_depth = event_origin.depth
    stf = STFType.base_stf(stf_type)
    stf.duration = event.duration or 0.0
    base_source.stf = stf
    print(base_source)
    ranges = dict(
        time=gf.Range(-0.1, 0.1, relative='add'),
        north_shift=gf.Range(-1*km, 1*km),
        east_shift=gf.Range(-1*km, 1*km),
        depth=gf.Range(3400, 12000),
        magnitude=gf.Range(1.7, 3.1),
        duration=gf.Range(0., 0.2),
        rmnn=gf.Range(-1.4, 1.4),
        rmee=gf.Range(-1.4, 1.4),
        rmdd=gf.Range(-1.4, 1.4),
        rmne=gf.Range(-1.4, 1.4),
        rmnd=gf.Range(-1.4, 1.4),
        rmed=gf.Range(-1.4, 1.4))

    problem = grond.problems.CMTProblem(
        name=event.name,
        base_source=base_source,
        distance_min=600.,
        mt_type='deviatoric',
        ranges=ranges,
        targets=targets,
        norm_exponent=1,
        stf_type=stf_type,
        )
    problem.set_engine(engine)
    monitor = GrondMonitor.watch(rundir)
    print("analysing")
    analyser_iter = 100
    analyser = grond.analysers.target_balancing.TargetBalancingAnalyser(
                niter=analyser_iter,
                use_reference_magnitude=False,
                cutoff=None)
    analyser.analyse(
        problem,
        ds)

    problem.dump_problem_info(rundir)

    from grond import config
    from grond import Environment

    config_path = '%s/scenario.gronf' % datafolder
    quick_config_path = '%sconfig/config.yaml' % datafolder
    util.ensuredir("%sconfig" % datafolder)
    configs_dir = os.path.dirname(os.path.abspath(__file__))+"/configs"
    os.system("cp -r %s/config.yaml* %s" % (configs_dir, quick_config_path))
    from grond.config import read_config
    conf = read_config(quick_config_path)
    uniform_iter = 2000
    directed_iter = 50
    mod_conf = conf.clone()
#    mod_conf.set_elements(
#        'path_prefix', ".")

    mod_conf.set_elements(
        'analyser_configs[:].niterations', analyser_iter)
    mod_conf.set_elements(
        'optimiser_config.sampler_phases[0].niterations', uniform_iter)
    mod_conf.set_elements(
        'optimiser_config.sampler_phases[0].niterations', directed_iter)
    mod_conf.set_elements(
        'optimiser_config.nbootstrap', 100)
    mod_conf.set_elements(
        'target_groups[:].misfit_config.tmin', "%s" % tmin)
    mod_conf.set_elements(
        'target_groups[:].misfit_config.tmax', "%s" % tmax)
    mod_conf.set_elements(
        'target_groups[:].misfit_config.fmin', "%s" % fmin)
    mod_conf.set_elements(
        'target_groups[:].misfit_config.fmax', "%s" % fmax)
    mod_conf.set_elements(
        'target_groups[:].misfit_config.domain', "%s" % domain)
    mod_conf.set_elements(
        'target_groups[:].misfit_config.ffactor', "%s" % ffactor)
    mod_conf.set_basepath(conf.get_basepath())

    config.write_config(mod_conf, quick_config_path)
    config.write_config(mod_conf, rundir+"/config.yaml")

    sampler_phases = [UniformSamplerPhase(niterations=uniform_iter),
                      DirectedSamplerPhase(niterations=directed_iter)]
    optimiser = grond.optimisers.highscore.HighScoreOptimiser(sampler_phases=list(sampler_phases),
                                                              chain_length_factor=8,
                                                              nbootstrap=100)
    optimiser.set_nthreads(1)

    optimiser.init_bootstraps(problem)
    tstart = time.time()
    optimiser.optimise(problem, rundir=rundir)
    grond.harvest(rundir, problem, force=True)
    #solverscipy.solve(problem, quiet=False, niter_explorative=2000, niter=10000)
    tstop = time.time()
    #os.system("grond report %s" % rundir)
    os.system("grond plot fits_waveform %s" % rundir)
    os.system("grond plot hudson %s" % rundir)
    os.system("grond plot fits_waveform_ensemble %s" % rundir)
    os.system("grond plot location_mt %s" % rundir)
    os.system("grond plot seismic_stations %s" % rundir)
    os.system("grond plot sequence %s" % rundir)
    ds.empty_cache()
    print("done with grond")
