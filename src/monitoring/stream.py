import subprocess
import os


def live_steam(adress="eida.gfz-potsdam.de", path="GR.BFO.*.BHZ", delay=200, seiger=True, save=False):
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
