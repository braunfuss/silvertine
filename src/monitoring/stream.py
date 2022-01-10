import subprocess
import os
from os.path import join as pjoin
import sys
import logging
import gc
import tempfile
import shutil
import glob

from pyrocko import pile as pile_mod
from pyrocko import util
from pyrocko import model
from pyrocko import config
from pyrocko import io
from pyrocko.gui import marker
from pyrocko.io import stationxml
from pyrocko.gui.snuffler_app import *


class PollInjector(qc.QObject):

    def __init__(self, *args, **kwargs):
        qc.QObject.__init__(self)
        self._injector = pile.Injector(*args, **kwargs)
        self._sources = []
        self.startTimer(1000.)

    def add_source(self, source):
        self._sources.append(source)

    def remove_source(self, source):
        self._sources.remove(source)

    def timerEvent(self, ev):
        for source in self._sources:
            trs = source.poll()
            for tr in trs:
                self._injector.inject(tr)

    # following methods needed because mulitple inheritance does not seem
    # to work anymore with QObject in Python3 or PyQt5

    def set_fixation_length(self, length):
        return self._injector.set_fixation_length(length)

    def set_save_path(
            self,
            path='dump_%(network)s.%(station)s.%(location)s.%(channel)s_'
                 '%(tmin)s_%(tmax)s.mseed'):

        return self._injector.set_save_path(path)

    def fixate_all(self):
        return self._injector.fixate_all()

    def free(self):
        return self._injector.free()


def live_steam(adresses=["eida.bgr.de"], paths=["GR.BFO.*.BHZ"], delay=50,
               save=False):

    combined_adress = " "
    for adress, path in zip(adresses, paths):
        combined_adress = combined_adress + "seedlink://"+adress+"/"+path+" "
    if save is False:
        os.system("snuffler " + combined_adress + " --follow="+str(delay))
    else:
        os.system("snuffler " + combined_adress + " --follow="+str(delay)+ " --store-interval=60 --store-path='stream_download/%(network)s.%(station)s.%(location)s.%(channel)s.%(tmin)s.mseed'")
