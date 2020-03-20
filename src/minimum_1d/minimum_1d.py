import random
import math
import numpy as num
from os.path import join as pjoin
import os.path as op
from pyrocko.moment_tensor import MomentTensor
from collections import OrderedDict
from pyrocko import util, model, io, trace, config
from pyrocko.gf import Target, DCSource, RectangularSource
from pyrocko import gf
from pyrocko.fdsn import ws
from pyrocko.fdsn import station as fs
from pyrocko.guts import Object, Int, String
import os
from pyrocko import orthodrome as ortho
from pyrocko import cake
from pyrocko import model
import time as timesys
import scipy
from pyrocko.gui.pile_viewer import PhaseMarker, EventMarker
from ..util.ref_mods import *


def update_layered_model_insheim(params, nevents):
    mod = cake.LayeredModel.from_scanlines(cake.read_nd_model_str('''
  0.             %s            %s           2.7         1264.           600.
  0.54           %s           %s           2.7         1264.           600.
  0.54           %s           %s            2.7         1264.           600.
  0.77           %s           %s            2.7         1264.           600.
  0.77           %s           %s            2.7         1264.           600.
  1.07           %s           %s            2.7         1264.           600.
  1.07           %s            %s            2.7         1264.           600.
  2.25           %s            %s            2.7         1264.           600.
  2.25           %s            %s            2.7         1264.           600.
  2.40           %s            %s            2.7         1264.           600.
  2.40           %s            %s            2.7         1264.           600.
  2.55           %s            %s            2.7         1264.           600.
  2.55           %s            %s            2.7         1264.           600.
  3.28           %s            %s            2.7         1264.           600.
  3.28           %s            %s            2.7         1264.           600.
  3.550          %s           %s            2.7         1264.           600.
  3.550          %s           %s            2.7         1264.           600.
  5.100          %s           %s            2.7         1264.           600.
  5.100          %s           %s            2.7         1264.           600.
  15.00          6.18          3.57           2.7         1264.           600.
  15.00          6.18          3.57           2.7         1264.           600.
  20.00          6.25          3.61           2.7         1264.           600.
  20.00          6.25          3.61           2.7         1264.           600.
  21.00          6.88          3.97           2.7         1264.           600.
  21.00          6.88          3.97           2.7         1264.           600.
 24.             8.1            4.69           2.7         1264.           600.
mantle
 24.             8.1            4.69           2.7         1264.           600.'''.lstrip() % (params[1+4*nevents], params[2+4*nevents],
                                                                                               params[3+4*nevents], params[4+4*nevents], params[3+4*nevents], params[4+4*nevents],
                                                                                               params[5+4*nevents], params[6+4*nevents], params[5+4*nevents], params[6+4*nevents],
                                                                                               params[7+4*nevents], params[8+4*nevents], params[7+4*nevents], params[8+4*nevents],
                                                                                               params[9+4*nevents], params[10+4*nevents], params[9+4*nevents], params[10+4*nevents],
                                                                                               params[11+4*nevents], params[12+4*nevents], params[11+4*nevents], params[12+4*nevents],
                                                                                               params[13+4*nevents], params[14+4*nevents], params[13+4*nevents], params[14+4*nevents],
                                                                                               params[15+4*nevents], params[16+4*nevents], params[15+4*nevents], params[16+4*nevents],
                                                                                               params[17+4*nevents], params[18+4*nevents], params[17+4*nevents], params[18+4*nevents],
                                                                                               params[19+4*nevents], params[20+4*nevents], params[19+4*nevents], params[20+4*nevents])))

    return mod
