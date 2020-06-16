import numpy as np
import matplotlib.pyplot as plt
from pyrocko import util, model


def get_kuperkoch_data(time=None, lag=120):
    rate = []
    temp = []
    pressure = []
    times = []
    if time is None:
        fname = "rest_data/opdata_Insheim_2012.txt"
        data = np.loadtxt(fname, skiprows=1)
        for log in data:
        	pressure.append(log[-1])
        	temp.append(log[-2])
        	rate.append(log[-3])
        	times.append(util.str_to_time(str(int(log[2]))+"-"+str(int(log[1]))+"-"+str(int(log[0]))+" "+str(int(log[3]))+":"+str(int(log[4]))+":"+str(int(log[5]))))
        fname = "rest_data/opdata_Insheim_2013.txt"
        data = np.loadtxt(fname, skiprows=1)
        for log in data:
        	pressure.append(log[-1])
        	temp.append(log[-2])
        	rate.append(log[-3])
        	times.append(util.str_to_time(str(int(log[2]))+"-"+str(int(log[1]))+"-"+str(int(log[0]))+" "+str(int(log[3]))+":"+str(int(log[4]))+":"+str(int(log[5]))))
        fname = "rest_data/opdata_Insheim_2014.txt"
        data = np.loadtxt(fname, skiprows=1)
        for log in data:
        	pressure.append(log[-1])
        	temp.append(log[-2])
        	rate.append(log[-3])
        	times.append(util.str_to_time(str(int(log[2]))+"-"+str(int(log[1]))+"-"+str(int(log[0]))+" "+str(int(log[3]))+":"+str(int(log[4]))+":"+str(int(log[5]))))
        fname = "rest_data/opdata_Insheim_2015.txt"
        data= np.loadtxt(fname, skiprows=1)
        for log in data:
        	pressure.append(log[-1])
        	temp.append(log[-2])
        	rate.append(log[-3])
        	times.append(util.str_to_time(str(int(log[2]))+"-"+str(int(log[1]))+"-"+str(int(log[0]))+" "+str(int(log[3]))+":"+str(int(log[4]))+":"+str(int(log[5]))))
        fname = "rest_data/opdata_Insheim_2016.txt"
        data = np.loadtxt(fname, skiprows=1)
        for log in data:
            pressure.append(log[-1])
            temp.append(log[-2])
            rate.append(log[-3])
            times.append(util.str_to_time(str(int(log[2]))+"-"+str(int(log[1]))+"-"+str(int(log[0]))+" "+str(int(log[3]))+":"+str(int(log[4]))+":"+str(int(log[5]))))

    else:
        fname = "rest_data/opdata_Insheim_%s.txt" % util.time_to_str(time)[0:4]
        data = np.loadtxt(fname, skiprows=1)
        for log in data:
            time_data = util.str_to_time(str(int(log[2]))+"-"+str(int(log[1]))+"-"+str(int(log[0]))+" "+str(int(log[3]))+":"+str(int(log[4]))+":"+str(int(log[5])))
            if time_data > time - lag and time_data < time + lag:
                times.append(util.str_to_time(str(int(log[2]))+"-"+str(int(log[1]))+"-"+str(int(log[0]))+" "+str(int(log[3]))+":"+str(int(log[4]))+":"+str(int(log[5]))))
                pressure.append(log[-1])
                temp.append(log[-2])
                rate.append(log[-3])

    return times, pressure, temp, rate


def plot_insheim_prod_data(reference="catalog", time=None, savedir=None,
                           source=None):
    times, pressure, temp, rate = get_kuperkoch_data(time)
    if reference == "catalog":
        reference_events = model.load_events("data/events_ler.pf")
    fig, ax1 = plt.subplots()
    ax1.set_xlabel('time (s)')
    line1 = ax1.plot(times, pressure, "k")
    line2 = ax1.plot(times, temp, "r")
    line3 = ax1.plot(times, rate, "b")
    plt.legend(('pressure [bar]', 'temperature [c]', 'rate [l/s]'), loc="upper left")
    ax2 = ax1.twinx()

    if time is not None:
        ax2.scatter(source.time, source.magnitude, c="k")
        ax2.set_ylabel('magnitude')
        ax2.tick_params(axis='y')
        fig.savefig(savedir+'production_data.png')
        plt.close()
    else:
        for event in reference_events:
            ax2.scatter(event.time, event.magnitude, c="k")
        ax2.set_ylabel('magnitude')
        ax2.tick_params(axis='y')
        fig.tight_layout()
        plt.show()
