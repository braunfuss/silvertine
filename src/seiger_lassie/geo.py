import numpy as num
from pyrocko import orthodrome as od
from pyrocko.guts import Object, Float


class Point(Object):
    lat = Float.T(default=0.0)
    lon = Float.T(default=0.0)
    x = Float.T(default=0.0)
    y = Float.T(default=0.0)
    z = Float.T(default=0.0)

    @property
    def vec(self):
        return self.lat, self.lon, self.x, self.y, self.z


def points_coords(points, system=None):
    a = num.zeros((len(points), 5))
    for ir, r in enumerate(points):
        a[ir, :] = r.vec

    lats, lons, xs, ys, zs = a.T

    if system is None:
        return (lats, lons, xs, ys, zs)

    elif system == 'latlon':
        return od.ne_to_latlon(lats, lons, xs, ys)

    elif system[0] == 'ne':
        lat0, lon0 = system[1:3]
        if num.all(lats == lat0) and num.all(lons == lon0):
            return xs, ys
        else:
            elats, elons = od.ne_to_latlon(lats, lons, xs, ys)
            return od.latlon_to_ne_numpy(lat0, lon0, elats, elons)


def point_coords(point, system=None):
    coords = points_coords([point], system=system)
    return [x[0] for x in coords]


def float_array_broadcast(*args):
    return num.broadcast_arrays(*[
        num.asarray(x, dtype=num.float) for x in args])


def surface_distance(alat, alon, anorth, aeast, blat, blon, bnorth, beast):
    args = float_array_broadcast(
        alat, alon, anorth, aeast, blat, blon, bnorth, beast)

    want_scalar = False
    if len(args[0].shape) == 0:
        want_scalar = True
        args = [num.atleast_1d(x) for x in args]

    (alat, alon, anorth, aeast, blat, blon, bnorth, beast) = args

    eqref = num.logical_and(alat == blat, alon == blon)
    neqref = num.logical_not(eqref)

    dist = num.empty_like(alat)
    dist[eqref] = num.sqrt(
        (bnorth[eqref] - anorth[eqref])**2 + (beast[eqref] - aeast[eqref])**2)

    aalat, aalon = od.ne_to_latlon(
        alat[neqref], alon[neqref], anorth[neqref], aeast[neqref])
    bblat, bblon = od.ne_to_latlon(
        blat[neqref], blon[neqref], bnorth[neqref], beast[neqref])

    dist[neqref] = od.distance_accurate50m_numpy(aalat, aalon, bblat, bblon)

    if want_scalar:
        return dist[0]
    else:
        return dist


def bounding_box(lat, lon, north, east, depth, scale=1.0):
    lat, lon, north, east, depth = float_array_broadcast(
        lat, lon, north, east, depth)

    if num.all(lat[0] == lat) and num.all(lon[0] == lon):

        return _scaled_bb(
            lat[0], lon[0],
            (num.min(north), num.max(north)),
            (num.min(east), num.max(east)),
            (num.min(depth), num.max(depth)), scale)

    else:
        elat, elon = od.ne_to_latlon(lat, lon, north, east)
        enorth, eeast = od.latlon_to_ne_numpy(elat[0], elon[0], elat, elon)
        enorth_min = num.min(enorth)
        enorth_max = num.max(enorth)
        eeast_min = num.min(eeast)
        eeast_max = num.max(eeast)

        mnorth = 0.5*(enorth_min + enorth_max)
        meast = 0.5*(eeast_min + eeast_max)

        lat0, lon0 = od.ne_to_latlon(elat[0], elon[0], mnorth, meast)
        return _scaled_bb(
            lat0, lon0,
            (enorth_min - mnorth, enorth_max - mnorth),
            (eeast_min - meast, eeast_max - meast),
            (num.min(depth), num.max(depth)),
            scale)


def bounding_box_square(lat, lon, north, east, depth, scale=1.0):

    lat, lon, (north_min, north_max), (east_min, east_max), \
        (depth_min, depth_max) = bounding_box(lat, lon, north, east, depth)

    dnorth = north_max - north_min
    mnorth = 0.5 * (north_min + north_max)
    deast = east_max - east_min
    meast = 0.5 * (east_min + east_max)
    dmax = max(dnorth, deast)

    if dnorth < dmax:
        north_min = mnorth - 0.5*dmax
        north_max = mnorth + 0.5*dmax

    if deast < dmax:
        east_min = meast - 0.5*dmax
        east_max = meast + 0.5*dmax

    return _scaled_bb(
        lat, lon,
        (north_min, north_max),
        (east_min, east_max),
        (depth_min, depth_max),
        scale)


def _scaled_range(ra, scale):
    mi, ma = ra
    d = ma - mi
    m = 0.5 * (ma + mi)
    return m - 0.5*scale*d, m + 0.5*scale*d


def _scaled_bb(lat, lon, north, east, depth, scale):

    return (
        lat, lon,
        _scaled_range(north, scale),
        _scaled_range(east, scale),
        _scaled_range(depth, scale))
