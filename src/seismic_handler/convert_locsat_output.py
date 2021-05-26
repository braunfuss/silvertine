from pyrocko import model
import numpy as np
from pyrocko.gui.pile_viewer import PhaseMarker, EventMarker
from pyrocko import cake
from pyrocko.util import stt


def monthToNum(shortMonth):
    return {
            'JAN': "01",
            'FEB': "02",
            'MAR': "03",
            'APR': "04",
            'MAY': "05",
            'JUN': "06",
            'JUL': "07",
            'AUG': "08",
            'SEP': "09",
            'OCT': "10",
            'NOV': "11",
            'DEC': "12"
    }[shortMonth]


def convert_phase(phase):
    if phase == "Pg":
        phase = "P<(moho)"
    if phase == "pg":
        phase = "p<(moho)"
    if phase == "Sg":
        phase = "S<(moho)"
    if phase == "sg":
        phase = "s<(moho)"
    if phase == "PG":
        phase = "P>(moho)"
    if phase == "pG":
        phase = "p>(moho)"
    if phase == "SG":
        phase = "S>(moho)"
    if phase == "sG":
        phase = "s>(moho)"
    if phase == "P*":
        phase = "P"
    if phase == "S*":
        phase = "S"
    if phase == "SmS":
        phase = 'Sv(moho)s'
    if phase == "PmP":
        phase = 'Pv(moho)p'
    if phase == "Pn":
        phase = 'Pv_(moho)p'
    if phase == "Sn":
        phase = 'Sv_(moho)s'
    cake_phase = cake.PhaseDef(phase)

    return cake_phase


def convertlocsat2pyrocko(fname):
    f = open(fname, "r")
    file = f.readlines()
    events = []
    events_data = []
    phase_markers = []
    for line in file:
        phase = None
        idx = line.find("Event ID               :")
        if idx != -1:
            idx = line.find(":")

            event_id = line[idx+1:]
            if event_id in events:
                idx_event = events.index(event_id)
                event_data = events_data[idx_event]
            else:
                events.append(event_id)
                event_data = []
        idx = line.find("Station code           :")
        if idx != -1:
            idx = line.find(":")

            station = line[idx+1:]
        idx = line.find("Phase name             :")
        if idx != -1:
            idx = line.find(":")

            phase = line[idx+1:]
        idx = line.find("Component              :")
        if idx != -1:
            idx = line.find(":")
            component = line[idx+1:]
        idx = line.find("Magnitude ml           :")
        if idx != -1:
            idx = line.find(":")
            ml = line[idx+1:]
        idx = line.find("Onset time             :")
        if idx != -1:
            idx = line.find(":")
            time_full = str(line[idx+1:])
            idx = time_full.find("_")
            time = time_full[idx+1:]
            idxs = time_full.find("-")
            day = time_full[:idxs]
            month = time_full[idxs+1:idxs+4]
            month = monthToNum(month)
            year = time_full[idxs+5:idxs+9]
            t = year+"-"+str(month)+"-"+str(day)+" "+ str(time)
            t = stt(t)
            if phase is not None:
                phase_markers.append(PhaseMarker(["0", station],
                                                 tmin=t,
                                                 tmax=t,
                                                 phasename=phase,
                                                 event_hash=ev_id))

        idx = line.find("Latitude               :")
        if idx != -1:
            idx = line.find("+")
            lat = line[idx+1:]
        idx = line.find("Longitude              :")
        if idx != -1:
            idx = line.find("+")
            long = line[idx+1:]
            events_data.append([lat, long])
