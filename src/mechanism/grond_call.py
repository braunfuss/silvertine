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
from pyrocko import gf, util
from pyrocko.parimap import parimap
from grond.optimisers.highscore import UniformSamplerPhase, DirectedSamplerPhase

import grond
logger = logging.getLogger('grond.silvertine')
#import solverscipy

def run_grond(rundir, datafolder, eventname, store_id, domain="time_domain"):
    km = 1000.
    print(datafolder)
    ds = grond.Dataset(event_name=eventname)
    ds.add_waveforms(paths=['%s/waveforms' % datafolder])
    ds.add_events(filename='%s/event.txt' % datafolder)
    ds.add_stations(stationxml_filenames=['%s/waveforms/stations.xml' % datafolder])
    ds.add_responses(stationxml_filenames=['%s/waveforms/stations.xml' % datafolder])

    ds._event_name = eventname

    ds.add_blacklist([])
    ds.empty_cache()

    quantity = 'displacement'
    group = 'all'

    engine = gf.LocalEngine(store_superdirs=['/home/asteinbe/gf_stores'])
    gf_interpolation = 'nearest_neighbor'
    imc_P = grond.WaveformMisfitConfig(
        fmin=1.,
        fmax=20.,
        ffactor=1.5,
        tmin='{stored:begin}+2',
        tmax='{stored:begin}+3',
        norm_exponent=1,
        domain=domain,
        quantity=quantity)
    cha_P ='Z'

    imc_S = grond.WaveformMisfitConfig(
        fmin=1.,
        fmax=20.,
        ffactor=1.5,
        tmin='{stored:begin}+2',
        tmax='{stored:begin}+3',
        norm_exponent=1,
        domain=domain,
        quantity=quantity,)
    cha_S ='T'

    event = ds.get_events()[0]
    event_origin = gf.Source(
        lat=event.lat,
        lon=event.lon)

    ##
    if event.depth is None:
        event.depth = 7*km

    # define distance minimum
    distance_min = None
    distance_max = 30000.
    targets = []
    ## first for P phases
    for st in ds.get_stations():
        for cha in cha_P:
            target = grond.WaveformMisfitTarget(
                interpolation=gf_interpolation,
                store_id=store_id,
                codes=st.nsl() + (cha,),
                lat=st.lat,
                lon=st.lon,
                quantity=quantity,
                misfit_config=imc_P,
                normalisation_family="td",
    	    path="P")
            _, bazi = event_origin.azibazi_to(target)
            if cha == 'R':
                target.azimuth = bazi - 180.
                target.dip = 0.
            elif cha == 'T':
                target.azimuth = bazi - 90.
                target.dip = 0.
            elif cha == 'Z':
                target.azimuth = 0.
                target.dip = -90.
            target.set_dataset(ds)
            targets.append(target)
    # for S phases
    for st in ds.get_stations():
        for cha in cha_S:
            target = grond.WaveformMisfitTarget(
                codes=st.nsl() + (cha,),
                lat=st.lat,
                lon=st.lon,
                interpolation=gf_interpolation,
                store_id=store_id,
                normalisation_family="td",
        	    path="S",
                quantity=quantity,
                misfit_config=imc_S)
            _, bazi = event_origin.azibazi_to(target)
            if cha == 'R':
                target.azimuth = bazi - 180.
                target.dip = 0.
            elif cha == 'T':
                target.azimuth = bazi - 90.
                target.dip = 0.
            elif cha == 'Z':
                target.azimuth = 0.
                target.dip = -90.
            target.set_dataset(ds)
            targets.append(target)

    base_source = gf.MTSource.from_pyrocko_event(event)
    base_source.set_origin(event_origin.lat, event_origin.lon)

    ranges=dict(
        time=gf.Range(0, 2.0, relative='add'),
        north_shift=gf.Range(-1*km, 1*km),
        east_shift=gf.Range(-1*km, 1*km),
        depth=gf.Range(400, 3000),
        magnitude=gf.Range(1.1, 1.8),
        duration=gf.Range(0., 0.5),
        rmnn=gf.Range(-1., 1.),
        rmee=gf.Range(-1., 1.),
        rmdd=gf.Range(-1., 1.),
        rmne=gf.Range(-1., 1.0),
        rmnd=gf.Range(-1., 1),
        rmed=gf.Range(-1., 1.))

    problem = grond.problems.CMTProblem(
        name=event.name,
        base_source=base_source,
        distance_min=600.,
        mt_type='deviatoric',
        ranges=ranges,
        targets=targets,
        norm_exponent=1,
        )

    problem.set_engine(engine)
    monitor = GrondMonitor.watch(rundir)
    print("analysing")
    analyser = grond.analysers.target_balancing.TargetBalancingAnalyser(
                niter=100,
                use_reference_magnitude=False,
                cutoff=None)
    analyser.analyse(
        problem,
        ds)

    problem.dump_problem_info(rundir)

    print("solving")
    sampler_phases =[UniformSamplerPhase(niterations=1000),
                 DirectedSamplerPhase(niterations=5000)]
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

    print("done with grond")
