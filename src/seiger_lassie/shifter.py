import logging
import hashlib
from io import BytesIO
import os.path as op
import numpy as num
from pyrocko.guts import Object, Float, String   # noqa
from pyrocko import cake, spit, util
from pyrocko.gf import meta
from silvertine.seiger_lassie import geo
from silvertine.seiger_lassie.common import LassieError, CakeEarthmodel

guts_prefix = 'seiger_lassie'
logger = logging.getLogger('lassie_seiger.config')


class Shifter(Object):

    def setup(self, config):
        pass


class VelocityShifter(Shifter):
    velocity = Float.T()
    offset = Float.T(default=0.0)

    def get_table(self, grid, receivers):
        distances = grid.distances(receivers)
        return self.offset + distances / self.velocity

    def get_offset_table(self, grid, receivers, origin):
        distances = grid.distances(receivers)

        x, y = geo.points_coords([origin], system=('ne', grid.lat, grid.lon))
        x = x[0]
        y = y[0]
        z = origin.z

        rx, ry = geo.points_coords(
            receivers, system=('ne', grid.lat, grid.lon))

        rz = num.array([r.z for r in receivers], dtype=num.float)

        distances_origin = num.sqrt((x - rx)**2 + (y - ry)**2 + (z - rz)**2)

        na = num.newaxis
        offset_distances = distances - distances_origin[na, :]

        return offset_distances / self.velocity

    def get_vmin(self):
        return self.velocity


class CakePhaseShifter(Shifter):
    earthmodel_id = String.T()
    timing = meta.Timing.T()
    offset = Float.T(default=0.0)
    factor = Float.T(default=1.0)

    def setup(self, config):
        Shifter.setup(self, config)
        self._earthmodels = config.earthmodels
        self._earthmodels.extend(
            [
                CakeEarthmodel(
                    id=fn,
                    earthmodel_1d=cake.load_model(
                        cake.builtin_model_filename(fn)))
                for fn in cake.builtin_models()
            ]
        )
        self._tabulated_phases = config.tabulated_phases

        if not self._tabulated_phases:
            raise LassieError('missing tabulated phases in config')

        self._cache_path = config.expand_path(config.cache_path)

    def get_earthmodel(self):
        for earthmodel in self._earthmodels:
            if isinstance(earthmodel, CakeEarthmodel):
                if earthmodel.id == self.earthmodel_id:
                    return earthmodel.earthmodel_1d

        raise LassieError(
            'no cake earthmodel with id "%s" found' % self.earthmodel_id)

    def get_vmin(self):
        vs = self.get_earthmodel().profile('vs')
        vp = self.get_earthmodel().profile('vp')
        v = num.concatenate((vs, vp))
        vmin = num.min(v[v != 0.0])
        return vmin

    def ttt_path(self, ehash):
        return op.join(self._cache_path, ehash + '.spit')

    def ttt_hash(self, earthmodel, phases, x_bounds, x_tolerance, t_tolerance):
        f = BytesIO()
        earthmodel.profile('z').dump(f)
        earthmodel.profile('vp').dump(f)
        earthmodel.profile('vs').dump(f)
        earthmodel.profile('rho').dump(f)

        f.write(b','.join(phase.definition().encode() for phase in phases))
        x_bounds.dump(f)
        x_tolerance.dump(f)
        f.write(str(t_tolerance).encode())
        s = f.getvalue()
        h = hashlib.sha1(s).hexdigest()
        f.close()
        return h

    def get_table(self, grid, receivers):
        distances = grid.lateral_distances(receivers)
        r_depths = num.array([r.z for r in receivers], dtype=num.float)
        s_depths = grid.depths()
        x_bounds = num.array(
            [[num.min(r_depths), num.max(r_depths)],
             [num.min(s_depths), num.max(s_depths)],
             [num.min(distances), num.max(distances)]], dtype=num.float)

        x_tolerance = num.array((grid.dz/2., grid.dz/2., grid.dx/2.))
        t_tolerance = grid.max_delta()/(self.get_vmin()*5.)
        earthmodel = self.get_earthmodel()

        interpolated_tts = {}

        for phase_def in self._tabulated_phases:
            ttt_hash = self.ttt_hash(
                earthmodel, phase_def.phases, x_bounds, x_tolerance,
                t_tolerance)

            fpath = self.ttt_path(ttt_hash)

            if not op.exists(fpath):
                def evaluate(args):
                    receiver_depth, source_depth, x = args
                    t = []
                    rays = earthmodel.arrivals(
                        phases=phase_def.phases,
                        distances=[x*cake.m2d],
                        zstart=source_depth,
                        zstop=receiver_depth)

                    for ray in rays:
                        t.append(ray.t)

                    if t:
                        return min(t)
                    else:
                        return None

                logger.info(
                    'prepare tabulated phases: %s [%s]' % (
                        phase_def.id, ttt_hash))

                sptree = spit.SPTree(
                    f=evaluate,
                    ftol=t_tolerance,
                    xbounds=x_bounds,
                    xtols=x_tolerance)

                util.ensuredirs(fpath)
                sptree.dump(filename=fpath)
            else:
                sptree = spit.SPTree(filename=fpath)

            interpolated_tts["stored:"+str(phase_def.id)] = sptree

        arrivals = num.zeros(distances.shape)

        def interpolate(phase_id):
            return interpolated_tts[phase_id].interpolate_many

        for i_r, r in enumerate(receivers):
            r_depths = num.zeros(distances.shape[0]) + r.z
            coords = num.zeros((distances.shape[0], 3))
            coords[:, 0] = r_depths
            coords[:, 1] = s_depths
            coords[:, 2] = distances[:, i_r]
            arr = self.timing.evaluate(interpolate, coords)
            arrivals[:, i_r] = arr

        return arrivals * self.factor + self.offset


__all__ = [
    'Shifter',
    'VelocityShifter',
    'CakePhaseShifter',
]
