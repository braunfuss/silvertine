
import os
from silvertine.detector.utils.downloader import makeStationList
from silvertine.detector.utils.hdf5_maker import preprocessor
from silvertine.detector.core.predictor import predictor
from silvertine.util import download_raw


def make_station_json(path, tmin="2016-02-12 06:20:03.800",
                      tmax="2016-02-12 06:30:03.800",
                      minlat=49.1379, maxlat=49.1879, minlon=8.1223,
                      maxlon=8.1723,
                      channels=["EH"+"[ZNE]"], client_list=["BGR"]):
    # CHANLIST =["HH[ZNE]", "HH[Z21]", "BH[ZNE]", "EH[ZNE]", "SH[ZNE]", "HN[ZNE]", "HN[Z21]", "DP[ZNE]"]

    json_basepath = os.path.join(path, "json/station_list.json")

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
                 overlap=0.3,
                 n_processor=2)


def predict(path, tmin="2016-02-12 06:20:03.800",
            tmax="2016-02-12 06:20:03.800", minlat=49.1379, maxlat=49.1879,
            minlon=8.1223,
            maxlon=8.1723):

    out_basepath = os.path.join(path, 'detections')
    model_path = os.path.join(path, 'ModelsAndSampleData/EqT_model.h5')

    downloads_basepath = os.path.join(path, "downloads")
    downloads_processed_path = os.path.join(path, "downloads_processed_hdfs")

    predictor(input_dir=downloads_processed_path,
             input_model=model_path,
             output_dir=out_basepath,
             estimate_uncertainty=False,
             output_probabilities=False,
             number_of_sampling=5,
             loss_weights=[0.02, 0.40, 0.58],
             detection_threshold=0.001,
             P_threshold=0.01,
             S_threshold=0.01,
             number_of_plots=10,
             plot_mode='time',
             batch_size=500,
             number_of_cpus=4,
             keepPS=False,
             spLimit=3)


def associate(path, tmin, tmax, minlat=49.1379, maxlat=49.1879, minlon=8.1223,
              maxlon=8.1723,
              channels=["EH"+"[ZNE]"], client_list=["BGR"]):

    import shutil
    import os
    from silvertine.detector.utils.associator import run_associator
    out_basepath = os.path.join(path, 'detections')
    out_dir = os.path.join(path, 'asociation')

    try:
        shutil.rmtree(out_dir)
    except Exception:
        pass
    os.makedirs(out_dir)

    run_associator(input_dir=out_basepath,
                   start_time=tmin,
                   end_time=tmax,
                   moving_window = 15,
                   pair_n = 2,
                   output_dir=out_dir,
                   consider_combination=False)


def main(path, tmin="2021-05-26 06:20:03.800",
         tmax="2016-02-12 06:20:03.800", minlat=49.1379, maxlat=49.1879,
         minlon=8.1223,
         maxlon=8.1723,
         channels=["EH"+"[ZNE]"], client_list=["BGR"],
         download=True, seiger=True, selection=None):

    if download is True:
        download_raw.download_raw(path, tmin, tmax, seiger=seiger, selection=selection,
                     provider=client_list, clean=True)

    make_station_json(path, tmin=tmin,
                          tmax=tmax,
                          minlat=minlat, maxlat=maxlat, minlon=minlon,
                          maxlon=maxlon,
                          channels=channels, client_list=client_list)

    preprocess(path, tmin=tmin,
                          tmax=tmax,
                          minlat=minlat, maxlat=maxlat, minlon=minlon,
                          maxlon=maxlon)

    predict(path, tmin=tmin,
                          tmax=tmax,
                          minlat=minlat, maxlat=maxlat, minlon=minlon,
                          maxlon=maxlon)
    associate(path, tmin=tmin,
                          tmax=tmax,
                          minlat=minlat, maxlat=maxlat, minlon=minlon,
                          maxlon=maxlon)
