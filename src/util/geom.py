from pyrocko.model import event
from pyrocko import orthodrome, util
from pyrocko.guts import dump
import sys
from pathlib import Path
import numpy as num

km = 1000.

global evpath


def duplicate_property(array):
    ndims = len(array.shape)
    if ndims == 1:
        return num.hstack((array, array))
    elif ndims == 2:
        return num.vstack((array, array))
    else:
        raise TypeError('Only 1-2d data supported!')


def to_cartesian(items, reflatlon):
    res = defaultdict()
    for i, item in enumerate(items):

        y, x = ortho.latlon_to_ne(reflatlon, item)
        depth = item.depth
        elevation = item.elevation
        dz = elevation - depth
        lat = item.lat/180.*num.pi
        z = r_earth+dz*num.sin(lat)
        res[item.nsl()[:2]] = (x, y, z)
    return res


def cmp(a, b):
    return (a > b) - (a < b)


def well_geometry_sparrow_export(file, folder, name):
    import utm
    from pyproj import Proj
    from pyrocko.model import Geometry
    data = num.loadtxt(file, delimiter=",")
    utmx = data[:, 0]
    utmy = data[:, 1]
    z = data[:, 2]

    proj_gk4 = Proj(init="epsg:31467")  # GK-projection
    lons, lats = proj_gk4(utmx, utmy, inverse=True)
    ev = event.Event(lat=num.mean(lats), lon=num.mean(lons), depth=num.mean(z),
                     time=util.str_to_time("2000-01-01 01:01:01"))

    ncorners = 4
    verts = []
    xs = []
    ys = []
    dist = 200.
    for i in range(0, len(z)):
        try:
            x = lats[i]
            y = lons[i]
            x1 = lats[i+1]
            y1 = lons[i+1]
            depth = z[i]
            depth1 = z[i+1]
            xyz = ([dist/2.8, dist/2.8, depth], [dist/2.8, dist/2.8, depth1],
                   [dist/2.8, -dist/2.8, depth1], [dist/2.8, -dist/2.8, depth])
            latlon = ([x, y], [x1, y1], [x1, y1], [x, y])
            patchverts = num.hstack((latlon, xyz))
            verts.append(patchverts)
        except Exception:
            pass

    vertices = num.vstack(verts)

    npatches = int(len(vertices))
    faces1 = num.arange(ncorners * npatches, dtype='int64').reshape(
        npatches, ncorners)
    faces2 = num.fliplr(faces1)
    faces = num.hstack((faces2, faces1))
    srf_semblance_list = []

    srf_semblance = num.ones(num.shape(data[:, 2]))
    srf_semblance = duplicate_property(srf_semblance)
    srf_semblance_list.append(srf_semblance)

    srf_semblance = num.asarray(srf_semblance_list).T
    srf_times = num.linspace(0, 1, 1)
    geom = Geometry(times=srf_times, event=ev)
    geom.setup(vertices, faces)
    sub_headers = tuple([str(i) for i in srf_times])
    geom.add_property((('semblance', 'float64', sub_headers)), srf_semblance)
    dump(geom, filename='geom_gtla1.yaml')
