import numpy as num
from pyrocko import cake
from silvertine.util.ref_mods import *
km = 1000.


def convert2shm_ttt(mod):

    # define cake phase to be used
    phases = cake.PhaseDef.classic('Pg')

    # define distances and depth (both in km)
    distances = num.linspace(1, 50, 10)
    distances = distances*km * cake.m2d
    depths = num.linspace(1.*km, 15.*km, 10)

    fobj = open('Landau_Pg.TTT', 'w')

    # make header
    fobj.write('TTT\n')
    fobj.write('distance bounds\n')
    fobj.write('%s %s \n' % (num.min(distances), num.max(distances)))
    fobj.write('depth steps\n')
    for depth in depths:
        fobj.write('%02s ' % num.round(depth/1000, 1))
    fobj.write('\n')

    # write out time per distance/depth
    data = []
    for distance in distances:
        fobj.write(' %02s ' % distance)

        for depth in depths:
            rays = mod.arrivals(
                phases=phases, distances=[distance], zstart=depth)

            for ray in rays[:1]:
                data.append((distance, ray.t))
                ttt = ray.t
            fobj.write('%s ' % num.round(ttt, 2))
        fobj.write('\n')
    fobj.close()


def convert2locsat_ttt(mod):

    km = 1000.

    # define Earth structure model in cake format to be used

    # define cake phase to be used
    phases_print = ["Pg", "Sg"]
    for phase in phases_print:
        phases = cake.PhaseDef.classic(phase)

        # define distances and depth (both in km)
        distances = num.linspace(0, 150, 150)
        distances = distances*km * cake.m2d
        depths = num.linspace(0.*km, 15.*km, 15)

        fobj = open('lan.%s' % phase, 'w')

        # make header
        fobj.write('n # %s     travel-time (and amplitude) tables \n' % phase)
        fobj.write(' %s # number of depth samples\n' % len(depths))
        fobj.write('    ')

        for depth in depths:
            depth = num.round(depth/1000, 2)
            fobj.write('%s   ' % (depth))
        fobj.write('\n')
        fobj.write('%s # number of distance samples\n' % len(distances))
        fobj.write('    ')
        for distance in distances:
            fobj.write('%s   ' % num.round(distance, 2))
        # write out time per distance/depth
        data = []
        fobj.write('\n')

        for depth in depths:
            depth_print = num.round(depth/1000, 2)
            fobj.write('# Travel-time/amplitude for z =    %s \n' % depth_print)

            for distance in distances:

                rays = mod.arrivals(
                    phases=phases, distances=[distance], zstart=depth)

                for ray in rays[:1]:
                    data.append((distance, ray.t))
                    ttt = ray.t
                if ttt is None:
                    ttt = 0
                fobj.write('     %s ' % num.round(ttt, 3))
                fobj.write('\n')
        fobj.close()
