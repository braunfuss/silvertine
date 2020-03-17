import numpy as num
from pyrocko import orthodrome as ortho
import ray
import psutil
import time
from pyrocko import spit, cake
from pyrocko.gf import meta
import sys



xrange = range
km = 1000.
class GridElem(object):
    def __init__(self, lat, lon, depth, tt, delta):
        self.lat = lat
        self.lon = lon
        self.depth = depth
        self.tt = tt
        self.delta = delta


class TTTGrid(object):
    def __init__(self, dimZ, Latul, Lonul, GridArray):
        self.dimZ = dimZ
        self.Latul = Latul
        self.Lonul = Lonul
        self.GridArray = GridArray


class MinTMaxT(object):
    def __init__(self, mint, maxt):
        self.mint = mint
        self.maxt = maxt


@ray.remote
def calculate_ttt(stations, mod, phase_defs, mod_name, gridspacing=1000.,
                  dimx=50000., dimy=50000., dimz=10000., o_lat=49.2,
                  o_lon=8.4, folder="scenarios/ttt/"):

    t_tolerance = 0.01                           # in seconds
    x_tolerance = num.array((10., 10.))       # in meters
    # Boundaries of the grid.
    xmin = 0.
    xmax = 360000.
    zmin = 0.
    zmax = 24000.
    x_bounds = num.array(((xmin, xmax), (zmin, zmax)))
    # In this example the receiver is located at the surface.
    receiver_depth = 0.

    interpolated_tts = {}
    try:
        if len(phase_defs) >= 1:
            pass
    except TypeError:
        phase_defs = meta.TPDef(id='%s' % phase_defs.definition(), definition='%s' % phase_defs.definition())
        phase_defs = [phase_defs]
    for phase_def in phase_defs:
        v_horizontal = phase_def.horizontal_velocities

        def evaluate(args):
            '''Calculate arrival using source and receiver location
            defined by *args*. To be evaluated by the SPTree instance.'''
            x, source_depth = args
            t = []

            # Calculate arrivals
            rays = mod.arrivals(
                phases=phase_def.phases,
                distances=[x*cake.m2d],
                zstart=source_depth,
                zstop=receiver_depth)

            for ray in rays:
                t.append(ray.t)

            for v in v_horizontal:
                t.append(x/(v*1000.))
            if t:
                return min(t)
            else:
                return None

        # Creat a :py:class:`pyrocko.spit.SPTree` interpolator.
        sptree = spit.SPTree(
            f=evaluate,
            ftol=t_tolerance,
            xbounds=x_bounds,
            xtols=x_tolerance)

        # Store the result in a dictionary which is later used to retrieve an
        # SPTree (value) for each phase_id (key).
        interpolated_tts[phase_def.id] = sptree

        # Dump the sptree for later reuse:
        sptree.dump(filename=folder+'/'+mod_name+'sptree_%s.yaml' % phase_def.id)


def calculate_ttt_parallel(stations, mod, phase_list, mod_name, gridspacing=1000., dimx=50000., dimy=50000., dimz=10000., o_lat=49.2, o_lon=8.4, adress=None):
    num_cpus = psutil.cpu_count(logical=False)
    if adress is not None:
        ray.init(adress=adress)
    else:
        ray.init(num_cpus=num_cpus)
    stations = stations[0]
    ray.get([calculate_ttt.remote(stations, mod, phase_list[i], mod_name) for i in range(len(phase_list))])


def load_sptree(phase_defs, mod_name, folder="scenarios/ttt/"):
    interpolated_tts = {}
    missing = []
    for phase_def in phase_defs:
        try:
            spt = spit.SPTree(filename=folder+'/'+mod_name+'sptree_%s.yaml' % phase_def.definition())
            interpolated_tts[phase_def.definition()] = spt
        except:
            missing.append(phase_def)

    return interpolated_tts, missing
