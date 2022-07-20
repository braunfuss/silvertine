from pyrocko.io import quakeml
from pyrocko import util, marker, model, orthodrome
import obspy.io.seiscomp.event as sc3
import string
import random
import csv

insheim_lat = 49.15665945780025
insheim_lon = 8.155952709819513
landau_lat = 49.18762179618079
landau_lon = 8.12753212452453

qml_new = quakeml.QuakeML.load_xml(filename="qml_base.yaml")

item = "events_ORG_mag.qml"
qml_ml = quakeml.QuakeML.load_xml(filename=item)
events_qml_ml = qml_ml.get_pyrocko_events()

item = "events_ORG.qml"
qml_mw = quakeml.QuakeML.load_xml(filename=item)
events_qml_mw = qml_mw.get_pyrocko_events()

qml_blank = quakeml.QuakeML.load_xml(filename=item)
event_params = []
event_params_ins = []
event_params_land = []
ins_events = []
land_events = []
for i, emw in enumerate(events_qml_mw):
    for k, eml in enumerate(events_qml_ml):
        if emw.time-0.2<= eml.time and emw.time+0.2>= eml.time:
            evq_ml = qml_ml.get_events()[k]
            evq_mw = qml_mw.get_events()[i]

            magnitude_list = evq_ml.magnitude_list
            magnitude_list.append(quakeml.Magnitude(
                    public_id=evq_ml.origin_list[0].public_id+"%s_mw" %random.choice(string.ascii_letters),
                    origin_id=evq_ml.origin_list[0].public_id,
                    type="Mw",
                    mag=quakeml.RealQuantity(value=emw.magnitude)))

            evq_ml.magnitude_list = magnitude_list
            qml_blank.event_parameters.event_list.append(evq_ml)

            dist_ins = orthodrome.distance_accurate50m(insheim_lat, insheim_lon, eml.lat, eml.lon)
            dist_land = orthodrome.distance_accurate50m(landau_lat, landau_lon, eml.lat, eml.lon)
            if dist_ins < 4500.:
                ins_events.append(eml)
                event_params_ins.append([str(util.tts(eml.time)), str(eml.time), str(eml.lat), str(eml.lon), str(eml.depth), str(eml.magnitude), str(emw.magnitude)])

            if dist_land < 4500.:
                land_events.append
                event_params_land.append([str(util.tts(eml.time)), str(eml.time), str(eml.lat), str(eml.lon), str(eml.depth), str(eml.magnitude), str(emw.magnitude)])

            event_params.append([str(util.tts(eml.time)), str(eml.time), str(eml.lat), str(eml.lon), str(eml.depth), str(eml.magnitude), str(emw.magnitude)])
model.dump_events(land_events, "landau_events.pf")
model.dump_events(ins_events, "ins_events.pf")

with open('seiger_catalog_alpha.csv', 'w', newline='') as csvfile:
    spamwriter = csv.writer(csvfile, delimiter=',',
                            quotechar='"', quoting=csv.QUOTE_MINIMAL)
    spamwriter.writerow(["Time", "Time(s)unix time", "Lat", "Lon", "Tiefe", "Ml", "Mw"])
    for p in event_params:
        spamwriter.writerow(p)

with open('seiger_catalog_alpha_landau.csv', 'w', newline='') as csvfile:
    spamwriter = csv.writer(csvfile, delimiter=',',
                            quotechar='"', quoting=csv.QUOTE_MINIMAL)
    spamwriter.writerow(["Time", "Time(s)unix time", "Lat", "Lon", "Tiefe", "Ml", "Mw"])
    for p in event_params_land:
        spamwriter.writerow(p)

with open('seiger_catalog_alpha_insheim.csv', 'w', newline='') as csvfile:
    spamwriter = csv.writer(csvfile, delimiter=',',
                            quotechar='"', quoting=csv.QUOTE_MINIMAL)
    spamwriter.writerow(["Time", "Time(s)unix time", "Lat", "Lon", "Tiefe", "Ml", "Mw"])
    for p in event_params_ins:
        spamwriter.writerow(p)

# qml_blank.dump_xml(filename="mw_ml.qml")
# from obspy import read_events
#
# evs = read_events("mw_ml.qml")
# for i, ev in enumerate(evs):
#
#     ev.write("qml_out/ev_%s.qml" %i, format="SC3ML")
