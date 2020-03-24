import subprocess
import os


def live_steam(adress="eida.gfz-potsdam.de", path="GR.*.*.BHZ", delay=200, seiger=True):
    if seiger is True:
        os.system("snuffler seedlink://"+"/eida.bgr.de"+"/"+"GR.INS*.*.*"+" --follow="+str(delay))

    else:
        os.system("snuffler seedlink://"+adress+"/"+path+" --follow="+str(delay))
