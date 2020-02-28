iiter = 0

phase = 'P'  # Phase to fit
tmin_fit = 15.  # [s] to fit before synthetic phase onset (from GFStore)
tmax_fit = 35.  # [s] ... after

bounds = OrderedDict([
    ('north_shift', (-0.2*km, 0.2*km)),
    ('east_shift', (-0.2*km, 0.2*km)),
    ('depth', (5.*km, 8.*km)),
    ('p_vel', (1.5, 2.)),
    ('s_vel', (0.5, 1.)),
    ('timeshift', (-0.1, 0.1))])


stations_list = stations_inversion.copy()
for s in stations_list:
    s.set_channels_by_name(*component.split())


targets = []
for s in stations_list:
        target = Target(
        lat=s.lat,
        lon=s.lon,
        store_id="placeholder",   # The gf-store to be used for this target,
        interpolation='multilinear',  # Interpolation method between GFStore nodes
        quantity='displacement',
        codes=s.nsl() + ('BH' + component,))
        targets.append(target)


source = gf.DCSource(
    lat=source_dc.lat,
    lon=source_dc.lon)

Pg = cake.PhaseDef('P<(moho)')
pg = cake.PhaseDef('p<(moho)')
p = cake.PhaseDef('p')
P = cake.PhaseDef('P')

Phase_S = cake.PhaseDef('S')
Sg = cake.PhaseDef('S<(moho)')
sg = cake.PhaseDef('s<(moho)')
phase_list = [P, p, Sg, sg, pg, Phase_S]


def update_source(params):
    s = source
    s.north_shift = float(params[0])
    s.east_shift = float(params[1])
    s.depth = float(params[2])
    s.time = float(source_dc.time - params[3])
    return source


def update_layered_model(params):
    mod = cake.LayeredModel.from_scanlines(cake.read_nd_model_str('''
  0.             %s            %s           2.7         1264.           600.
  0.54           2.25           1.11           2.7         1264.           600.
  0.54           2.25           1.11           2.7         1264.           600.
  0.77           2.85           1.44           2.7         1264.           600.
  0.77           2.85           1.44           2.7         1264.           600.
  1.07           3.18           1.61           2.7         1264.           600.
  1.07           3.18           1.61           2.7         1264.           600.
  2.25           3.81           2.01           2.7         1264.           600.
  2.25           3.81           2.01           2.7         1264.           600.
  2.40           5.12           2.65           2.7         1264.           600.
  2.40           5.12           2.65           2.7         1264.           600.
  2.55           4.53           2.39           2.7         1264.           600.
  2.55           4.53           2.39           2.7         1264.           600.
  3.28           5.14           2.91           2.7         1264.           600.
  3.28           5.14           2.91           2.7         1264.           600.
  3.550          5.688          3.276           2.7         1264.           600.
  3.550          5.688          3.276           2.7         1264.           600.
  5.100          5.98          3.76           2.7         1264.           600.
  5.100          5.98          3.76           2.7         1264.           600.
  15.00          6.18          3.57           2.7         1264.           600.
  15.00          6.18          3.57           2.7         1264.           600.
  20.00          6.25          3.61           2.7         1264.           600.
  20.00          6.25          3.61           2.7         1264.           600.
  21.00          6.88          3.97           2.7         1264.           600.
  21.00          6.88          3.97           2.7         1264.           600.
 24.             8.1            4.69           2.7         1264.           600.
mantle
 24.             8.1            4.69           2.7         1264.           600.'''.lstrip() % (params[3], params[4])))

    return mod

def picks_fit(params, line=None):
    global iiter
    update_source(params)
    # mod = update_layered_model(params)

    misfits = 0.
    norms = 0.
    dists = []

    for ev in ev_dict_list:
        for st in ev["phases"]:
            dists.append(ortho.distance_accurate50m(source.lat, source.lon, st["lat"], st["lon"])*cake.m2d)
    #    ray = timing.t(mod, (source.depth, dist), get_ray=True)
    ev_iter = 0
    dist = None
    for i, arrival in enumerate(mod.arrivals(dists, phases=phase_list, zstart=source.depth)):
        if dist == None:
            dist = arrival.x*cake.d2m
        if dist == arrival.x*cake.d2m:
            ev = ev_dict_list[ev_iter]
            dist = arrival.x*cake.d2m
        else:
            ev_iter = ev_iter+1
            ev = ev_dict_list[ev_iter]
            dist = arrival.x*cake.d2m

        for p in ev["phases"]:
            tdiff = source.time - p[2]
            phase = p[1]
            if p[1] == "Pg":
                phase = "P<(moho)"
            if p[1] == "pg":
                phase = "p<(moho)"
            used_phase = arrival.used_phase()
            print(used_phase.given_name(), phase)
            if phase == used_phase.given_name():

                misfits += num.sqrt(num.sum((tdiff - arrival.t)**2))
                print(misfits)
                norms += num.sqrt(num.sum(arrival.t**2))
    misfit = num.sqrt(misfits**2 / norms**2)
    print(misfit)

    iiter += 1

    if line:
        data = {
            'y': [misfit],
            'x': [iiter],
        }
        line.data_source.stream(data)

    return misfit


def solve():
    t = timesys.time()

    result = scipy.optimize.differential_evolution(
        picks_fit,
        args=[plot],
        bounds=tuple(bounds.values()),
        maxiter=2,
        tol=0.01,
        callback=lambda a, convergence: push_notebook())

    source = update_source(result.x)
    source.regularize()

    print("Time elapsed: %.1f s" % (timesys.time() - t))
    print("Best model:\n - Misfit %f" % trace_fit(result.x))
    print(source)
    return result, source


# Start the optimisation
result, best_source = solve()
