import numpy as np
import matplotlib.pyplot as plt
from pyrocko import util, model


def get_kuperkoch_data():
    rate = []
    temp = []
    pressure = []
    times = []
    fname = "rest_data/opdata_Insheim_2012.txt"
    data = np.loadtxt(fname, skiprows=1)
    for log in data:
    	pressure.append(log[-1])
    	temp.append(log[-2])
    	rate.append(log[-3])
    	times.append(util.str_to_time(str(int(log[2]))+"-"+str(int(log[1]))+"-"+str(int(log[0]))+" "+str(int(log[3]))+":"+str(int(log[4]))+":"+str(int(log[5]))))
    fname = "rest_data/opdata_Insheim_2013.txt"
    data= np.loadtxt(fname, skiprows=1)
    for log in data:
    	pressure.append(log[-1])
    	temp.append(log[-2])
    	rate.append(log[-3])
    	times.append(util.str_to_time(str(int(log[2]))+"-"+str(int(log[1]))+"-"+str(int(log[0]))+" "+str(int(log[3]))+":"+str(int(log[4]))+":"+str(int(log[5]))))
    fname = "rest_data/opdata_Insheim_2014.txt"
    data= np.loadtxt(fname, skiprows=1)
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
    data= np.loadtxt(fname, skiprows=1)
    for log in data:
    	pressure.append(log[-1])
    	temp.append(log[-2])
    	rate.append(log[-3])
    	times.append(util.str_to_time(str(int(log[2]))+"-"+str(int(log[1]))+"-"+str(int(log[0]))+" "+str(int(log[3]))+":"+str(int(log[4]))+":"+str(int(log[5]))))

    return times, pressure, temp, rate


def plot_insheim_prod_data(reference="catalog"):
    times, pressure, temp, rate = get_kuperkoch_data()
    if reference == "catalog":
        reference_events = model.load_events("data/events_ler.pf")
    fig, ax1 = plt.subplots()
    ax1.set_xlabel('time (s)')
    line1 = ax1.plot(times, pressure, "k")
    line2 = ax1.plot(times, temp, "r")
    line3 = ax1.plot(times, rate, "b")
    plt.legend(('pressure [bar]', 'temperature [c]', 'rate [l/s]'), loc="upper left")
    ax2 = ax1.twinx()
    for event in reference_events:
        ax2.scatter(event.time, event.magnitude, c="k")
    ax2.set_ylabel('magnitude')
    ax2.tick_params(axis='y')
    fig.tight_layout()
    plt.show()
