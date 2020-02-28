def write_hypoDD_dat(ev_dict_list):
    f = open("phase_hypo.dat", 'w')
    for ev in ev_dict_list:
        time = ev["time"]
        date = util.time_to_str(time)
        print(date)
        yr = date[0:4]
        mo = date[5:7]
        dy = date[8:10]
        hr = date[11:13]
        mi = date[14:16]
        sc = date[17:]

        s = "#  %s  %s  %s  %s  %s  %s  %s  %s  %s  %s  %s  %s  %s  %s" % (yr,mo, dy, hr, mi, sc, ev["lat"], ev["lon"], ev["depth"], ev["mag"], ev["error_h"], ev["error_z"], ev["rms"], ev["id"])
        f.write(s+'\n')
        for phase in ev["phases"]:
            tdiff = phase["pick"]-time
            s = "%s \t %s \t %s \t %s" % (phase["station"], tdiff, 1., phase["phase"])
            f.write(s+'\n')

    f.close()
