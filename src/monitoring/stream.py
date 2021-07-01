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


def live_steam(adress="eida.gfz-potsdam.de", path="GR.BFO.*.BHZ", delay=50,
               seiger=False, save=False, delete_delay=None):
    if seiger is True:
        if save is False:
            os.system("snuffler seedlink://"+"eida.bgr.de"+"/"+"GR.INS*.*.*"+" --follow="+str(delay))
        else:
            os.system("snuffler seedlink://"+"eida.bgr.de"+"/"+"GR.INS*.*.*"+" --follow="+str(delay)+" --store-path='datadump/%(network)s.%(station)s.%(location)s.%(channel)s.%(tmin)s.mseed'")

    else:
        if save is False:
            os.system("snuffler seedlink://"+adress+"/"+path+" --follow="+str(delay))
        else:
            os.system("snuffler seedlink://"+adress+"/"+path+" --follow="+str(delay)+ " --store-interval=30 --store-path='datadump/%(network)s.%(station)s.%(location)s.%(channel)s.%(tmin)s.mseed'")


def poll(store_path=none, store_interval=None, sources=['seedlink://eida.bgr.de/GR.INS*.*.*',
                                                        'seedlink://eida.bgr.de/GR.TMO*.*.*',
                                                        'seedlink://eida.bgr.de/GR.TMO*.*.*']):

    piled = pile_mod.make_pile()
    sources = []
    pollinjector = None
    tempdir = None
    store_path = "temp"
    store_interval = 10
    wait_period = 10
    sources.extend(setup_acquisition_sources)
    if sources:
        if store_path is None:
            tempdir = tempfile.mkdtemp('', 'snuffler-tmp-')
            store_path = pjoin(
                tempdir,
                'trace-%(network)s.%(station)s.%(location)s.%(channel)s.'
                '%(tmin_ms)s.mseed')
        elif os.path.isdir(store_path):
            store_path = pjoin(
                store_path,
                'trace-%(network)s.%(station)s.%(location)s.%(channel)s.'
                '%(tmin_ms)s.mseed')

        _injector = pile.Injector(piled, path=store_path, fixation_length=10,
                                  forget_fixed=True)

        for source in sources:
            source.start()
        while True:
            time.sleep(10.3)
            for source in sources:
                trs = source.poll()
                for tr in trs:
                    _injector.inject(tr)

        for source in sources:
            source.stop()
