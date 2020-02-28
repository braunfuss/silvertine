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
