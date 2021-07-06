
import os
from silvertine.detector.utils.downloader import makeStationList
from silvertine.detector.utils.hdf5_maker import preprocessor
from silvertine.detector.core.predictor import predictor
from silvertine.util import download_raw
from pyrocko import pile, io, util
import subprocess


def make_station_json(path, tmin="2016-02-12 06:20:03.800",
                      tmax="2016-02-12 06:30:03.800",
                      minlat=49.1379, maxlat=49.1879, minlon=8.1223,
                      maxlon=8.1723,
                      channels=["EH"+"[ZNE]"], client_list=["BGR"],
                      seiger=True):
    # CHANLIST =["HH[ZNE]", "HH[Z21]", "BH[ZNE]", "EH[ZNE]", "SH[ZNE]", "HN[ZNE]", "HN[Z21]", "DP[ZNE]"]
    if tmin is None:
        tmin = util.stt("2021-01-22 14:05:03.00")
        tmax = util.stt("2021-12-22 14:05:03.00")
    json_basepath = os.path.join(path, "json/station_list.json")
    if seiger is False:
        makeStationList(json_path=json_basepath,
                        client_list=client_list,
                        min_lat=minlon,
                        max_lat=maxlat,
                        min_lon=minlon,
                        max_lon=maxlon,
                        start_time=tmin,
                        end_time=tmax,
                        channel_list=channels,
                        filter_network=[],
                        filter_station=[])
    else:
        json_path = os.path.dirname(os.path.abspath(__file__))+"/json/station_list.json"
        util.ensuredir(os.path.join(path, "json"))
        os.system("cp  %s %s" % (json_path, json_basepath))

    json_basepath = os.path.join(os.getcwd(), "json/station_list.json")


def preprocess(path, tmin="2016-02-12 06:20:03.800",
               tmax="2016-02-12 06:30:03.800",
               minlat=49.1379, maxlat=49.1879, minlon=8.1223, maxlon=8.1723):

    pre_proc_basepath = os.path.join(path, "prepoc")
    downloads_basepath = os.path.join(path, "downloads")
    json_basepath = os.path.join(path, "json/station_list.json")

    preprocessor(preproc_dir=pre_proc_basepath,
                 mseed_dir=downloads_basepath,
                 stations_json=json_basepath,
                 overlap=0.0,
                 n_processor=2)


def predict(path, tmin="2016-02-12 06:20:03.800",
            tmax="2016-02-12 06:20:03.800", minlat=49.1379, maxlat=49.1879,
            minlon=8.1223,
            maxlon=8.1723, model_path=None, model=None, iter=None):

    if iter is None:
        out_basepath = os.path.join(path, 'detections')
    else:
        out_basepath = os.path.join(path, 'detections_%s_%s' % (tmin, tmax))


    if model_path is None:
        model_path = os.path.dirname(os.path.abspath(__file__))+"/model/EqT_model.h5"
    downloads_basepath = os.path.join(path, "downloads")
    downloads_processed_path = os.path.join(path, "downloads_processed_hdfs")

    predictor(input_dir=downloads_processed_path,
              input_model=model_path,
              output_dir=out_basepath,
              estimate_uncertainty=False,
              output_probabilities=False,
              number_of_sampling=1,
              loss_weights=[0.03, 0.40, 0.58],
              detection_threshold=0.0003,
              P_threshold=0.0003,
              S_threshold=0.0003,
              number_of_plots=0,
              plot_mode='time',
              batch_size=500,
              number_of_cpus=6,
              keepPS=False,
              model=model,
              spLimit=3)


def associate(path, tmin, tmax, minlat=49.1379, maxlat=49.1879, minlon=8.1223,
              maxlon=8.1723,
              channels=["EH"+"[ZNE]"], client_list=["BGR"], iter=None,
              pair_n, moving_window=30):

    import shutil
    import os
    from silvertine.detector.utils.associator import run_associator
    if iter is None:
        out_basepath = os.path.join(path, 'detections')
        out_dir = os.path.join(path, 'asociation')
    else:
        out_basepath = os.path.join(path, 'detections_%s_%s' % (tmin, tmax))
        out_dir = os.path.join(path, 'asociation_%s_%s' % (tmin, tmax))

    try:
        shutil.rmtree(out_dir)
    except Exception:
        pass
    os.makedirs(out_dir)

    run_associator(input_dir=out_basepath,
                   start_time=util.tts(tmin),
                   end_time=util.tts(tmax),
                   moving_window=moving_window,
                   pair_n=pair_n,
                   output_dir=out_dir,
                   consider_combination=False)


def reject_blacklisted(tr, blacklist):
    '''returns `False` if nslc codes of `tr` match any of the blacklisting
    patters. Otherwise returns `True`'''
    return not util.match_nslc(blacklist, tr.nslc_id)


def iter_chunked(tinc, path, data_pile, tmin=None,
                 tmax=None, minlat=49.1379,
                 maxlat=49.1879,
                 minlon=8.1223,
                 maxlon=8.1723,
                 channels=["EH"+"[ZNE]"], client_list=["BGR"],
                 download=True, seiger=True, selection=None,
                 path_waveforms=None,
                 stream=False,
                 reject_blacklisted=None, tpad=0,
                 tstart=None, tstop=None,
                 hf=10, lf=2):
    try:
        tstart = util.stt(tmin) if tmin else None
        tstop = util.stt(tmax) if tmax else None
    except:
        pass
    model_path = os.path.dirname(os.path.abspath(__file__))+"/model/EqT_model.h5"
    from tensorflow.keras.models import load_model
    from tensorflow.keras.optimizers import Adam

    from silvertine.detector.core.EqT_utils import f1, SeqSelfAttention, FeedForward, LayerNormalization

    model = load_model(model_path,
                       custom_objects={'SeqSelfAttention': SeqSelfAttention,
                                       'FeedForward': FeedForward,
                                       'LayerNormalization': LayerNormalization,
                                       'f1': f1
                                        })
    model.compile(loss=['binary_crossentropy', 'binary_crossentropy',
                        'binary_crossentropy'],
                  loss_weights=[0.02, 0.40, 0.58],
                  optimizer=Adam(lr=0.001),
                  metrics=[f1])
    for i, trs in enumerate(data_pile.chopper(tinc=tinc, tmin=tstart,
                                              tmax=tstop, tpad=tpad,
                                            #  keep_current_files_open=False,
                                              want_incomplete=True)):
        tminc = None
        tmaxc = None
        for tr in trs:
            if tminc is None:
                tminc = tr.tmin
                tmaxc = tr.tmax
            else:
                if tminc < tr.tmin:
                    tminc = tr.tmin
                if tmaxc > tr.tmax:
                    tmaxc = tr.tmax
        for tr in trs:
            tr.highpass(4, lf)
            tr.lowpass(4, hf)

            tr.chop(tminc, tmaxc)
            date_min = download_raw.get_time_format_eq(tminc)
            date_max = download_raw.get_time_format_eq(tmaxc)
            io.save(tr, "%s/downloads/%s/%s.%s..%s__%s__%s.mseed" % (path,
                                                                    tr.station,
                                                                    tr.network,
                                                                    tr.station,
                                                                    tr.channel,
                                                                    date_min,
                                                                    date_max))

        process(path, tmin=tminc, tmax=tmaxc, minlat=minlat, maxlat=maxlat,
                minlon=minlon, maxlon=maxlon, channels=channels,
                client_list=client_list, download=download, seiger=seiger,
                selection=selection, path_waveforms=path_waveforms,
                stream=stream, model=model, iter=i)
        for tr in trs:
            subprocess.run(["rm -r %s/downloads/%s/%s.%s..%s__%s__%s.mseed" %(path, tr.station, tr.network, tr.station, tr.channel, date_min, date_max)], shell=True)


def load_eqt_folder(data_paths, tinc, path, tmin="2021-05-26 06:20:03.800",
                    tmax="2016-02-12 06:20:03.800", minlat=49.1379,
                    maxlat=49.1879,
                    minlon=8.1223,
                    maxlon=8.1723,
                    channels=["EH"+"[ZNE]"], client_list=["BGR"],
                    download=True, seiger=True, selection=None,
                    path_waveforms=None,
                    stream=False,
                    data_format='mseed', deltat_want=100,
                    tstart=None, tstop=None, hf=8, lf=2):

    data_pile = pile.make_pile(data_paths, fileformat=data_format, show_progress=False)
    iter_chunked(tinc, path, data_pile, tmin=tmin, tmax=tmax, minlat=minlat,
                 maxlat=maxlat,
                 minlon=minlon, maxlon=maxlon, channels=channels,
                 client_list=client_list, download=download, seiger=seiger,
                 selection=selection, path_waveforms=path_waveforms,
                 stream=stream,
                 reject_blacklisted=None, tpad=0,
                 tstart=None, tstop=None, hf=hf,
                 lf=lf)


def process(path, tmin="2021-05-26 06:20:03.800",
            tmax="2016-02-12 06:20:03.800", minlat=49.1379, maxlat=49.1879,
            minlon=8.1223,
            maxlon=8.1723,
            channels=["EH"+"[ZNE]"], client_list=["BGR"],
            download=True, seiger=True, selection=None,
            path_waveforms=None, model=None,
            stream=False, iter=None):

    make_station_json(path, tmin=tmin,
                      tmax=tmax,
                      minlat=minlat, maxlat=maxlat, minlon=minlon,
                      maxlon=maxlon,
                      channels=channels, client_list=client_list,
                      seiger=seiger)
    preprocess(path, tmin=tmin,
               tmax=tmax,
               minlat=minlat, maxlat=maxlat, minlon=minlon,
               maxlon=maxlon)

    predict(path, tmin=tmin,
            tmax=tmax,
            minlat=minlat, maxlat=maxlat, minlon=minlon,
            maxlon=maxlon, model=model, iter=iter)
    associate(path, tmin=tmin,
              tmax=tmax,
              minlat=minlat, maxlat=maxlat, minlon=minlon,
              maxlon=maxlon, iter=iter)


def main(path, tmin="2021-05-26 06:20:03.800",
         tmax="2016-02-12 06:20:03.800", minlat=49.1379, maxlat=49.1879,
         minlon=8.1223,
         maxlon=8.1723,
         channels=["EH"+"[ZNE]"], client_list=["BGR",
                                               "http://ws.gpi.kit.edu/"],
         download=True, seiger=True, selection=None, clean=False,
         stream=False, path_waveforms=None, tinc=None,
         lf=5, hf=40):

    if download is True:
        if tinc is None:
            download_raw.download_raw(path, tmin, tmax, seiger=seiger,
                                      selection=selection,
                                      providers=client_list, clean=True,
                                      detector=True)
        else:
            tmin = util.stt(tmin)
            tmax = util.stt(tmax)
            iter = 0
            vorhalten = True
            tinc = float(tinc)
            for i in range(int(int(tmax-tmin)/int(tinc))):
                twin_start = tmin + iter*tinc
                twin_end = tmin + tinc + iter*tinc
                download_raw.download_raw(path, twin_start, twin_end, seiger=seiger,
                             selection=selection,
                             providers=client_list, clean=clean,
                             detector=True)
                load_eqt_folder(path, tinc, path, tmin=twin_start, tmax=twin_end,
                                minlat=minlat, maxlat=maxlat,
                                minlon=minlon, maxlon=maxlon, channels=channels,
                                client_list=client_list, download=download,
                                seiger=seiger,
                                selection=selection,
                                stream=stream, hf=hf, lf=lf)

            #    process(path, tmin=tmin, tmax=tmax, minlat=minlat, maxlat=maxlat,
            #            minlon=minlon, maxlon=maxlon, channels=channels,
            #            client_list=client_list, download=download, seiger=seiger,
            #            selection=selection, path_waveforms=path_waveforms,
            #            stream=stream)
                subprocess.run(['rm -r %s*' % (path+'/downloads*')], shell=True)


                iter =+ 1

            #    if twin_start > tmax:
            #        vorhalten = False

    if path_waveforms is not None:
        load_eqt_folder(path_waveforms, tinc, path, tmin=tmin, tmax=tmax,
                        minlat=minlat, maxlat=maxlat,
                        minlon=minlon, maxlon=maxlon, channels=channels,
                        client_list=client_list, download=download,
                        seiger=seiger,
                        selection=selection,
                        stream=stream, hf=hf, lf=lf)
    else:
        process(path, tmin=tmin, tmax=tmax, minlat=minlat, maxlat=maxlat,
                minlon=minlon, maxlon=maxlon, channels=channels,
                client_list=client_list, download=download, seiger=seiger,
                selection=selection, path_waveforms=path_waveforms,
                stream=stream)
