from builtins import range
import math
import numpy as num
from pyrocko.guts import Object, Float
from pyrocko import orthodrome as od

from silvertine.seiger_lassie import geo

guts_prefix = 'seiger_lassie'


class Grid(Object):
    pass


class Carthesian3DGrid(Grid):
    lat = Float.T(default=0.0)
    lon = Float.T(default=0.0)
    xmin = Float.T()
    xmax = Float.T()
    ymin = Float.T()
    ymax = Float.T()
    zmin = Float.T()
    zmax = Float.T()
    dx = Float.T()
    dy = Float.T()
    dz = Float.T()

    def __init__(self, **kwargs):
        for k in kwargs.keys():
            kwargs[k] = float(kwargs[k])

        Grid.__init__(self, **kwargs)
        self.update()

    def size(self):
        nx, ny, nz = self._shape()
        return nx * ny * nz

    def max_delta(self):
        return max(self.dx, self.dy, self.dz)

    def update(self):
        self._coords = None

    def _shape(self):
        nx = int(round(((self.xmax - self.xmin) / self.dx))) + 1
        ny = int(round(((self.ymax - self.ymin) / self.dy))) + 1
        nz = int(round(((self.zmax - self.zmin) / self.dz))) + 1
        return nx, ny, nz

    def _get_coords(self):
        if self._coords is None:
            nx, ny, nz = self._shape()
            x = num.linspace(self.xmin, self.xmax, nx)
            y = num.linspace(self.ymin, self.ymax, ny)
            z = num.linspace(self.zmin, self.zmax, nz)
            self._coords = x, y, z

        return self._coords

    def depths(self):
        nx, ny, _ = self._shape()
        _, _, z = self._get_coords()
        return num.repeat(z, nx*ny)

    def index_to_location(self, i):
        nx, ny, nz = self._shape()
        iz, iy, ix = num.unravel_index(i, (nz, ny, nx))
        x, y, z = self._get_coords()
        return self.lat, self.lon, x[ix], y[iy], z[iz]

    def lateral_distances(self, receivers):
        nx, ny, nz = self._shape()
        x, y, z = self._get_coords()

        rx, ry = geo.points_coords(
            receivers, system=('ne', self.lat, self.lon))

        na = num.newaxis

        nr = len(receivers)

        distances = num.sqrt(
            (x[na, na, :, na] - rx[na, na, na, :])**2 +
            (y[na, :, na, na] - ry[na, na, na, :])**2).reshape((nx*ny*nr))

        return num.tile(distances, nz).reshape((nx*ny*nz, nr))

    def distances(self, receivers):
        nx, ny, nz = self._shape()
        x, y, z = self._get_coords()

        rx, ry = geo.points_coords(
            receivers, system=('ne', self.lat, self.lon))

        rz = num.array([r.z for r in receivers], dtype=num.float)

        na = num.newaxis

        nr = len(receivers)

        distances = num.sqrt(
            (x[na, na, :, na] - rx[na, na, na, :])**2 +
            (y[na, :, na, na] - ry[na, na, na, :])**2 +
            (z[:, na, na, na] - rz[na, na, na, :])**2).reshape((nx*ny*nz, nr))

        return distances

    def distance_max(self):
        return math.sqrt(
            (self.xmax - self.xmin)**2 +
            (self.ymax - self.ymin)**2 +
            (self.zmax - self.zmin)**2)

    def surface_points(self, system='latlon'):
        x, y, z = self._get_coords()
        xs = num.tile(x, y.size)
        ys = num.repeat(y, x.size)
        if system == 'latlon':
            return od.ne_to_latlon(self.lat, self.lon, xs, ys)
        elif system[0] == 'ne':
            lat0, lon0 = system[1:]
            if lat0 == self.lat and lon0 == self.lon:
                return xs, ys
            else:
                elats, elons = od.ne_to_latlon(self.lat, self.lon, xs, ys)
                return od.latlon_to_ne_numpy(lat0, lon0, elats, elons)

    def plot_points(self, axes, system='latlon'):
        x, y = self.surface_points(system=system)
        axes.plot(y, x, '.', color='black', ms=1.0)

    def plot(
            self, axes, a,
            amin=None,
            amax=None,
            z_slice=None,
            cmap=None,
            system='latlon',
            shading='gouraud',
            units=1.,
            artists=[]):

        if system == 'latlon':
            assert False, 'not implemented yet'

        if not (system[1] == self.lat and system[2] == self.lon):
            assert False, 'not implemented yet'

        nx, ny, nz = self._shape()
        x, y, z = self._get_coords()

        a3d = a.reshape((nz, ny, nx))

        if z_slice is not None:
            iz = num.argmin(num.abs(z-z_slice))
            a2d = a3d[iz, :, :]
        else:
            a2d = num.max(a3d, axis=0)

        if artists:
            if shading == 'gouraud':
                artists[0].set_array(a2d.T.ravel())
            elif shading == 'flat':
                artists[0].set_array(a2d.T[:-1, :-1].ravel())
            else:
                assert False, 'unknown shading option'

            return artists

        else:
            return [
                axes.pcolormesh(
                    y/units, x/units, a2d.T,
                    vmin=amin, vmax=amax, cmap=cmap, shading=shading)]


def geometrical_normalization(grid, receivers):
    distances = grid.distances(receivers)

    delta_grid = grid.max_delta()

    delta_ring = delta_grid * 3.0
    ngridpoints, nstations = distances.shape
    norm_map = num.zeros(ngridpoints)

    for istation in range(nstations):
        dists_station = distances[:, istation]

        dist_min = num.floor(num.min(dists_station) / delta_ring) * delta_ring
        dist_max = num.ceil(num.max(dists_station) / delta_ring) * delta_ring

        dist = dist_min
        while dist < dist_max:
            indices = num.where(num.logical_and(
                dist <= dists_station,
                dists_station < dist + delta_ring))[0]

            nexpect = math.pi * ((dist + delta_ring)**2 - dist**2) /\
                delta_grid**2

            # nexpect = math.pi * ((dist + delta_ring)**3 - dist**3) /\
            #     delta_grid**3

            norm_map[indices] += indices.size / nexpect

            dist += delta_ring

    norm_map /= nstations

    return norm_map


__all__ = [
    'Grid',
    'Carthesian3DGrid',
]
