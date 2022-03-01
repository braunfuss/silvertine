from pyrocko.client import fdsn
from pyrocko import util, io, trace, model
import subprocess
from subprocess import DEVNULL, STDOUT, check_call
import os
import time as timemod
import numpy as num
fdsn.g_timeout = 60.


def get_seiger_stations(tmin, tmax):
    # Stations in the Resarch network "south palatinate"
    selection = [
        ('GR', 'TMO*', '*', 'EH*', tmin, tmax),
        ('GR', 'INS*', '*', 'EH*', tmin, tmax),
        ('*', 'LDE*', '*', 'EH*', tmin, tmax),
        ('*', 'LDO*', '*', 'EH*', tmin, tmax),
        ('*', 'ROTT*', '*', 'EH*', tmin, tmax),
        ('*', 'MOER*', '*', 'EH*', tmin, tmax),
     ]
    return selection


def get_time_format_eq(time):
    time = util.tts(time)
    time_year = time[0:4]
    time_month = time[5:7]
    time_day = time[8:10]
    time_hour = time[11:13]
    time_minute = time[14:16]
    time_seconds = time[17:19]
    date = time_year+time_month+time_day+"T"+time_hour+time_minute+time_seconds+"Z"
    return date


def supply(watch_folder, tmin, tmax, seiger=True, selection=None, duration=3,
           package_length=86400,
           providers=["bgr", "http://ws.gpi.kit.edu"],
           clean=True):

    try:
        tmin = util.stt(tmin)
        tmax = util.stt(tmax)
    except:
        pass
    iter = 0
    vorhalten = True
    while vorhalten is True:
        timemod.sleep(5)
        totalDir = 0
        for base, dirs, files in os.walk(watch_folder):
            for directories in dirs:
                totalDir += 1

        if totalDir < duration:
            twin_start = tmin + iter*package_length
            twin_end = tmin + package_length + iter*package_length
            download_raw(watch_folder, twin_start, twin_end, seiger=seiger,
                         selection=selection,
                         providers=providers, clean=clean)
            iter = iter+ 1

        if twin_start > tmax:
            vorhalten = False


def download_raw(path, tmint, tmaxt, seiger=True, selection=None,
                 providers=["bgr", "http://ws.gpi.kit.edu"],
                 clean=True,
                 detector=False, common_f=80, tinc=None):
    try:
        tmin = util.stt(tmint)
        tmax = util.stt(tmaxt)
    except:
        tmin = tmint
        tmax = tmaxt
    util.ensuredir(path+"/downloads")
    for provider in providers:
        if clean is True and detector is True:
            subprocess.run(['rm -r %s*' % (path+'/downloads/')], shell=True)
        if seiger is True:
            selection = get_seiger_stations(tmin, tmax)

        request_waveform = fdsn.dataselect(site=provider, selection=selection)
        # write the incoming data stream to 'traces.mseed'
        if provider == "http://ws.gpi.kit.edu":
            provider = "kit"
        if provider == "http://192.168.11.220:8080":
            provider = "bgr"
        download_basepath = os.path.join(path, "traces_%s.mseed" % provider)

        with open(download_basepath, 'wb') as file:
            file.write(request_waveform.read())

        traces = io.load(download_basepath)
        if common_f is not None:
            for tr in traces:
                if tr.deltat != common_f:
                    tr.downsample_to(1/common_f)
                    tr.ydata = tr.ydata.astype(num.int32)
        if detector is True:
            for tr in traces:
                tr.chop(tmin, tmax)
                date_min = get_time_format_eq(tr.tmin)
                date_max = get_time_format_eq(tr.tmax)
                io.save(tr, "%sdownloads/%s/%s.%s..%s__%s__%s.mseed" % (path,
                                                                        tr.station,
                                                                        tr.network,
                                                                        tr.station,
                                                                        tr.channel,
                                                                        date_min,
                                                                        date_max))
        else:
            util.ensuredir("%s/downloads/" % path)
            window_start = traces[0].tmin
            window_end = traces[0].tmax
            timestring = util.time_to_str(window_start, format='%Y-%m')
            io.save(traces, "%s/%s/%s_%s_%s.mseed" % (path, timestring,
                                                      provider,
                                                      tmin,
                                                      tmax))
    if clean is True:
        for provider in providers:
            if provider == "http://192.168.11.220:8080":
                provider = "bgr"
            if provider == "http://ws.gpi.kit.edu":
                provider = "kit"
            subprocess.run(['rm -r %s*' % (path+'/traces_%s.mseed' % provider)], shell=True, stdout=DEVNULL, stderr=STDOUT)
