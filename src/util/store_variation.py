import copy
import numpy as num
from .ref_mods import *
from pyrocko.plot import cake_plot as plot
from pyrocko.fomosto import qseis, qssp
from pyrocko import util, trace, gf, cake  # noqa

km = 1000.


def ensemble_earthmodel(ref_earthmod, num_vary=10, error_depth=0.4,
                        error_velocities=0.2, depth_limit_variation=600):
    """
    Create ensemble of earthmodels that vary around a given input earth model
    by a Gaussian of 2 sigma (in Percent 0.1 = 10%) for the depth layers
    and for the p and s wave velocities. Vp / Vs is kept unchanged
    Parameters
    ----------
    ref_earthmod : :class:`pyrocko.cake.LayeredModel`
        Reference earthmodel defining layers, depth, velocities, densities
    num_vary : scalar, int
        Number of variation realisations
    error_depth : scalar, float
        3 sigma error in percent of the depth for the respective layers
    error_velocities : scalar, float
        3 sigma error in percent of the velocities for the respective layers
    depth_limit_variation : scalar, float
        depth threshold [m], layers with depth > than this are not varied
    Returns
    -------
    List of Varied Earthmodels :class:`pyrocko.cake.LayeredModel`
    """

    earthmods = []
    i = 0
    while i < num_vary:
        new_model, cost = vary_model(
            ref_earthmod,
            error_depth,
            error_velocities,
            depth_limit_variation)

        if cost > 20:
            print('Skipped unlikely model %f' % cost)
        else:
            i += 1
            earthmods.append(new_model)

    return earthmods


def vary_model(
        earthmod, error_depth=0.4, error_velocities=0.4,
        depth_limit_variation=600):
    """
    Vary depths and velocities in the given source model by Gaussians with
    given 2-sigma errors [percent]. Ensures increasing velocity with depth.
    Stops variating the input model at the given depth_limit_variation [m].
    Mantle discontinuity uncertainties are hardcoded based on
    Mooney et al. 1981 and Woodward et al.1991
    Parameters
    ----------
    earthmod : :class:`pyrocko.cake.LayeredModel`
        Earthmodel defining layers, depth, velocities, densities
    error_depth : scalar, float
        2 sigma error in percent of the depth for the respective layers
    error_velocities : scalar, float
        2 sigma error in percent of the velocities for the respective layers
    depth_limit_variations : scalar, float
        depth threshold [m], layers with depth > than this are not varied
    Returns
    -------
    Varied Earthmodel : :class:`pyrocko.cake.LayeredModel`
    Cost : int
        Counts repetitions of cycles to ensure increasing layer velocity,
        unlikely velocities have high Cost
        Cost of up to 20 are ok for crustal profiles.
    """

    new_earthmod = copy.deepcopy(earthmod)
    layers = new_earthmod.layers()

    last_l = None
    cost = 0
    deltaz = 0

    # uncertainties in discontinuity depth after Shearer 1991
    discont_unc = {
        '410': 3 * km,
        '520': 4 * km,
        '660': 8 * km}

    # uncertainties in velocity for upper and lower mantle from Woodward 1991
    # and Mooney 1989
    mantle_vel_unc = {
        '100': 0.05,     # above 100
        '200': 0.03,     # above 200
        '400': 0.01}     # above 400

    for layer in layers:
        # stop if depth_limit_variation is reached
        if depth_limit_variation:
            if layer.ztop >= depth_limit_variation:
                layer.ztop = last_l.zbot
                # assign large cost if previous layer has higher velocity
                if layer.mtop.vp < last_l.mtop.vp or \
                   layer.mtop.vp > layer.mbot.vp:
                    cost = 1000
                # assign large cost if layer bottom depth smaller than top
                if layer.zbot < layer.ztop:
                    cost = 1000
                break
        repeat = 1
        count = 0
        while repeat:
            if count > 1000:
                break

            # vary layer velocity
            # check for layer depth and use hardcoded uncertainties
            for l_depth, vel_unc in mantle_vel_unc.items():
                if float(l_depth) * km < layer.ztop:
                    error_velocities = vel_unc
                    print('Velocity error: %f ', error_velocities)

            deltavp = float(
                num.random.normal(
                    0, layer.mtop.vp * error_velocities / 3., 1))

            if layer.ztop == 0:
                layer.mtop.vp += deltavp
                layer.mbot.vs += (deltavp / layer.mbot.vp_vs_ratio())

            # ensure increasing velocity with depth
            if last_l:
                # gradient layer without interface
                if layer.mtop.vp == last_l.mbot.vp:
                    if layer.mbot.vp + deltavp < layer.mtop.vp:
                        count += 1
                    else:
                        layer.mbot.vp += deltavp
                        layer.mbot.vs += (deltavp /
                                          layer.mbot.vp_vs_ratio())
                        repeat = 0
                        cost += count
                elif layer.mtop.vp + deltavp < last_l.mbot.vp:
                    count += 1
                else:
                    layer.mtop.vp += deltavp
                    layer.mtop.vs += (deltavp / layer.mtop.vp_vs_ratio())

                    if isinstance(layer, cake.GradientLayer):
                        layer.mbot.vp += deltavp
                        layer.mbot.vs += (deltavp / layer.mbot.vp_vs_ratio())
                    repeat = 0
                    cost += count
            else:
                repeat = 0

        # vary layer depth
        layer.ztop += deltaz
        repeat = 1

        # use hard coded uncertainties for mantle discontinuities
        if '%i' % (layer.zbot / km) in discont_unc:
            factor_d = discont_unc['%i' % (layer.zbot / km)] / layer.zbot
        else:
            factor_d = error_depth

        while repeat:
            # ensure that bottom of layer is not shallower than the top
            deltaz = float(
                num.random.normal(
                    0, layer.zbot * factor_d / 3., 1))  # 3 sigma
            layer.zbot += deltaz
            if layer.zbot < layer.ztop:
                layer.zbot -= deltaz
                count += 1
            else:
                repeat = 0
                cost += count

        last_l = copy.deepcopy(layer)

    return new_earthmod, cost


def vary_insheim():
    insheim_mod = insheim_layered_model()
    mod_varied, cost = vary_model(insheim_mod)
    return mod_varied


def vary_insheim_ensemble():
    insheim_mod = insheim_layered_model()
    mods_varied = ensemble_earthmodel(insheim_mod)
    return mods_varied


def vary_landau():
    landau_mod = landau_layered_model()
    mod_varied, cost = vary_model(landau_mod)
    return mod_varied


def vary_landau_ensemble():
    landau_mod = landau_layered_model()
    mods_varied = ensemble_earthmodel(landau_mod)
    return mods_varied


def vary_vsp_ensemble():
    vsp_mod = vsp_layered_model()
    mods_varied = ensemble_earthmodel(vsp_mod)
    return mods_varied


def create_gf_store(mod, path, name="model"):
    store_dir = path+"/"+name+"/"
    qsconf = qseis.QSeisConfig()
    qsconf.qseis_version = '2006a'

    qsconf.time_region = (
        gf.meta.Timing('{stored:begin}-5'),
        gf.meta.Timing('{stored:end}+20'))

    qsconf.cut = (
        gf.meta.Timing('{stored:begin}-5'),
        gf.meta.Timing('{stored:end}+20'))

    qsconf.wavelet_duration_samples = 0.001
    qsconf.sw_flat_earth_transform = 0

    config = gf.meta.ConfigTypeA(
        id='qseis_%s' %name,
        sample_rate=100,
        receiver_depth=0.*km,
        source_depth_min=0*km,
        source_depth_max=15*km,
        source_depth_delta=0.4*km,
        distance_min=0.5,
        distance_max=40*km,
        distance_delta=0.4*km,
        modelling_code_id='qseis.2006a',
        earthmodel_1d=mod,
        tabulated_phases=[
            gf.meta.TPDef(
                id='any_P',
                definition='p,P'),
            gf.meta.TPDef(
                id='end',
                definition='1.5'),
            gf.meta.TPDef(
                id='begin',
                definition='10'),
        ])

    config.validate()
    gf.store.Store.create_editables(
        store_dir, config=config, extra={'qseis': qsconf})

    store = gf.store.Store(store_dir, 'r')
    store.make_ttt()
    store.close()

    try:
        qseis.build(store_dir, nworkers=1)
    except qseis.QSeisError as e:
        if str(e).find('could not start qseis') != -1:
            logger.warn('qseis not installed; '
                        'skipping test_pyrocko_gf_vs_qseis')
            return
        else:
            raise


def save_varied_models(mods, folder, plot=False, name="model"):
    for i, mod in enumerate(mods):
        if plot is True:
            plot.my_model_plot(mod)
        cake.write_nd_model(mod, folder+"/%s_%s" % (name, i))


def load_varied_models(folder, nmodels=1):
    mods = []
    for i in range(0, nmodels):
        mods.append(cake.load_model(folder+"/model_%s" % i))
    return mods
