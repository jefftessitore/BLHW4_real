from netCDF4 import Dataset
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from netCDF4 import Dataset

import cmocean
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np

import cmocean.cm as cm

import os

cmaps = {
    'w':  {'cm': 'seismic',   'label': 'vertical velocity [m/s]'},
    'wspd': {'cm': 'gist_stern_r',              'label': 'windspeed [m/s]'},
    'wdir': {'cm': cmocean.cm.phase,   'label': 'wind direction [deg]'},
    'pt': {'cm': cmocean.cm.thermal, 'label': 'potential temperature [C]'},
    't': {'cm': cmocean.cm.thermal, 'label': 'temperature [C]'},
    'q':  {'cm': cmocean.cm.haline_r,  'label': 'q [g/kg]'},
    'dp': {'cm': cmocean.cm.haline_r,  'label': 'dewpoint [C]'},
    'rh': {'cm': cmocean.cm.haline_r,  'label': 'RH [%]'},
    'std': {'cm': cmocean.cm.thermal,  'label': 'Standard Deviation'}
}


def timeheight(time, height, data, field, ax, datemin=None, datemax=None,
                datamin=None, datamax=None, zmin=None, zmax=None, cmap=None, **kwargs):
    '''
    Produces a time height plot of a 2-D field
    :param time: Array of times (1-D or 2-D but must have same dimenstions as height)
    :param height: Array of heights (1-D or 2-D but must have same dimensions as time)
    :param data: Array of the data to plot (2-D)
    :param field: Field being plotted. Currently supported:
        'w': Vertical Velocity
        'ws': Wind Speed
        'wd': Wind Direction
        'pt': Potential Temperature
        'q':  Specific Humidity
        'dp': Dewpoint
        'rh': Relative Humidity
        'std': Standard Deviation
    :param ax: Axis to plot the data to
    :param datemin: Datetime object
    :param datemax: Datetime object
    :param datamin: Minimum value of data to plot
    :param datamax: Maximum value of data to plot
    :param zmin: Minimum height to plot
    :param zmax: Maximum height to plot
    :return:
    '''

    # Get the colormap and label of the data
    if cmap is None:
        cm, cb_label = cmaps[field]['cm'], cmaps[field]['label']
    else:
        cm, cb_label = cmap, cmaps[field]['label']

    # Convert the dates to matplolib format if not done already
    if time.ndim == 1 and height.ndim == 1:
        time = mdates.date2num(time)
        time, height = np.meshgrid(time, height)

    # Create the plot
    c = ax.pcolormesh(time, height, data, vmin=datamin, vmax=datamax, cmap=cm, **kwargs)

    # Format the colorbar
    # c.cmap.set_bad('grey', 1.0)
    cb = plt.colorbar(c, ax=ax)
    cb.set_label(cb_label)

    # Format the limits
    ax.xaxis.set_major_locator(mdates.HourLocator())
    ax.xaxis.set_minor_locator(mdates.HourLocator())
    # ax.xaxis.set_major_formatter(mdates.DateFormatter('%m/%d'))
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%H%M'))
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)

    if zmin is not None and zmax is not None:
        ax.set_ylim(zmin, zmax)
    if datemin is not None and datemax is not None:
        ax.set_xlim(mdates.date2num(np.array([datemin, datemax])))

    # Set the labels
    ax.set_ylabel('Height [m]')
    ax.set_xlabel('Time [UTC]')

    return ax



# Grab the times, heights, and data we're interested in 
def parse_data(nc):
    '''
    Parses given dataset for times, heights, windspeeds, & wind directions
    :param nc: The NetCDF Dataset object to parse
    :return times: Times (in UTC) of each datapoint, formatted as 'HH:MM'
    :return heights: Heights (in m) of each datapoint
    :return wspd: Wind speed (in m/s) of each datapoint (equiv. magnitude of vector v, the horizontal wind vector)
    :return wdir: Wind direction (in degrees) of each datapoint
    :return w: Vertical velocity (in m/s) of each datapoint
    '''
    times = np.array([datetime.utcfromtimestamp(d) for d in nc['base_time'][0]+nc['time_offset'][:]])
    heights = nc['height'][:] * 1e3
    wspd = nc['wspd'][:]
    wdir = nc['wdir'][:]
    w = nc['w'][:]
    nc.close()
    return times,heights,wspd,w

filename180 = 'P1/dltruckdlcsmwindsDL1.a3.20241028.000000.cdf'
nc180 = Dataset(filename180, 'r')
# Print out the netCDF header so you know what's in there
print (nc180)
times180,heights180,wspd180,w180 = parse_data(nc180)

filename10 = 'P1/dltruckdlcsmwindsDL1.a2.20241028.000000.cdf'
nc10 = Dataset(filename10,'r')
print(nc10)
times10,heights10,wspd10,w10 = parse_data(nc10)

filename3 = 'P1/dltruckdlcsmwindsDL1.a1.20241028.000000.cdf'
nc3 = Dataset(filename3,'r')
print(nc3)
times3,heights3,wspd3,w3 = parse_data(nc3)

# Make the figure with the time_height function above. Feel free to not use this if you want to make it in your own style
def plot_series(times,heights,wspd,w,filename,title_base):
    #fig, (ax1, ax2) = plt.subplots(2, 1)
    fig = plt.figure(figsize=(9,9))
    plt.rcParams.update({'font.size': 15}) # stolen from the internet: https://www.geeksforgeeks.org/change-font-size-in-matplotlib/
    ax1 = plt.subplot(111)
    
    ax1 = timeheight(times, heights, wspd.T, 'wspd', ax1, datamin=0, datamax=25, zmin=0, zmax = 2000) 
    
    #ax2 = timeheight(times, heights, wdir.T, 'wdir', ax2, datamin=0, datamax=360, zmin=0, zmax = 2000) 
    sTitle = 'Horizontal Windspeeds' + title_base
    plt.title(sTitle)
    plt.savefig(filename.replace('.cdf', '_v.png'))

    figw = plt.figure(figsize=(8,8))
    plt.rcParams.update({'font.size': 15})
    axw = plt.subplot(111)
    axw = timeheight(times, heights, w.T, 'w', axw, datamin=-6, datamax=6, zmin=0, zmax=2000) # experimentally-derived datamin/datamax, discounting bad scans
    sTitle = 'Vertical Velocities' + title_base
    plt.title(sTitle)
    plt.savefig(filename.replace('.cdf','_w.png'))
    
    #return fig,(ax1,ax2)
    return fig,ax1,figw,axw

#fig180,(ax180_1,ax180_2) = plot_series(times180,heights180,wspd180,wdir180,filename180)
#fig10,(ax10_1,ax10_2) = plot_series(times10,heights10,wspd10,wdir10,filename10)
#fig3,(ax3_1,ax3_2) = plot_series(times3,heights3,wspd3,wdir3,filename3)

# Homework problem only asks for horizontal wind speed, not direction, so I'll eliminate the expectation of the second axis:
fig180_v,ax180_v,fig180_w,ax180_w = plot_series(times180,heights180,wspd180,w180,filename180,' (Full Scan)')
fig10_v,ax10_v,fig30_w,ax30_w = plot_series(times10,heights10,wspd10,w10,filename10,' (10-sec. Integration)')
fig3_v,ax3_v,fig3_w,ax3_w = plot_series(times3,heights3,wspd3,w3,filename3,' (Max Resolution)')
