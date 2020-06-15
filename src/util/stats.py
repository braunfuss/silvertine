from pyrocko import orthodrome
import numpy as num
import math


def calcuate_msd(events, start=0, end=None, distance_3d=True):
    base_event = events[start]
    distances = []
    time_rel = []
    if end is None:
        end = len(events)
    for i in range(0, end):
        if i == start:
            pass
        else:
            surface_dist = orthodrome.distance_accurate50m(base_event,
                                                           events[i])
            if distance_3d:
                dd = base_event.depth - events[i].depth
                dist_m = math.sqrt(dd**2 + surface_dist**2)
                distances.append(dist_m)
        time_rel.append(base_event.time - events[i].time)
    msd = (1./(end-start)) * (num.sum(distances))
    return distances, time_rel, msd
