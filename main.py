import matplotlib
import matplotlib.pyplot as plt

from datetime import datetime, timedelta
from future.builtins import *  # NOQA
from datetime import timedelta
from obspy.core import read
from obspy.core.utcdatetime import UTCDateTime
from obspy.core.inventory import read_inventory
import numpy as np
from obspy.clients.fdsn.client import Client
import pandas as pd
from moonquake_libs.utils import linear_interpolation, timing_correction
from moonquake_libs.plot_timing_divergence import plot_timing


plt.style.use('ggplot')
plt.rcParams['figure.figsize'] = 10, 4
plt.rcParams['lines.linewidth'] = 0.5
plt.rcParams['font.size'] = 12
SECONDS_PER_DAY=3600.*24

def view_Apollo(starttime, endtime, station, outfile, channel='MH1', network='XA', location='*',plot_response=False):
    client = Client("IRIS")

    # get the response file (wildcards allowed)
    try:

        inv = client.get_stations(starttime=starttime, endtime=endtime,
        network=network, sta=station, loc=location, channel=channel,
        level="response")
        stream = client.get_waveforms(network=network, station=station, channel=channel, location=location, starttime=starttime, endtime=endtime)
        for tr in stream:
            # interpolate across the gaps of one sample 
            linear_interpolation(tr,interpolation_limit=1)
        stream.merge()
        
        for tr in stream:
            # optionally interpolate across any gap 
            # for removing the instrument response from a seimogram, 
            # it is useful to get a mask, then interpolate across the gaps, 
            # then mask the trace again. 
            if tr.stats.channel in ['MH1', 'MH2', 'MHZ']:

                # add linear interpolation but keep the original mask
                original_mask = linear_interpolation(tr,interpolation_limit=None)
                # remove the instrument response
                pre_filt = [0.1,0.3,0.9,1.1]
                tr.remove_response(inventory=inv, pre_filt=pre_filt, output="DISP",
                        water_level=None, plot=plot_response)
                if plot_response:
                    plt.show()
                # apply the mask back to the trace 
                tr.data = np.ma.masked_array(tr, mask=original_mask)

            elif tr.stats.channel in ['SHZ']:

                # add linear interpolation but keep the original mask
                original_mask = linear_interpolation(tr,interpolation_limit=None)
                # remove the instrument response
                pre_filt = [1,2,11,13] 
                tr.remove_response(inventory=inv, pre_filt=pre_filt, output="DISP",
                        water_level=None, plot=plot_response)
                if plot_response:
                    plt.show()
                
                # apply the mask back to the trace 
                tr.data = np.ma.masked_array(tr, mask=original_mask)

        stream.plot(equal_scale=False,size=(1000,600),method='full', outfile=outfile)
    except:
        print("no such data, skipping this")
     

    


class DateConverter:
    def __init__(self, year, day_of_year, hour, minute, second, gap, is_start):
        if is_start:
            self._date = datetime(year, 1, 1) + timedelta(days=day_of_year-1, hours=hour, minutes=minute, seconds=second) - timedelta(hours = gap)
        else:
            self._date = datetime(year, 1, 1) + timedelta(days=day_of_year-1, hours=hour, minutes=minute, seconds=second) + timedelta(hours = gap)


    @property
    def year(self):
        return self._date.year

    @property
    def month(self):
        return self._date.month

    @property
    def day(self):
        return self._date.day
    
    @property
    def hour(self):
        return self._date.hour
    
    @property
    def minute(self):
        return self._date.minute
    
    @property
    def second(self):
        return self._date.second

    def __str__(self):
        return self._date.strftime('%Y-%m-%d-%H-%M-%S')

df = pd.read_csv('nakamura_1979_sm_locations.csv')
stations = ["S12", "S14", "S15", "S16"]
for index, row in df.iterrows():
    half_gap = int(row['Magnitude']) / 6
    start_date = DateConverter(int(row['Year']),int(row['Day']), int(row['H']),int(row['M']),int(row['S']), half_gap, True)
    end_date = DateConverter(int(row['Year']),int(row['Day']), int(row['H']),int(row['M']),int(row['S']), half_gap, False)
    start_time = UTCDateTime(start_date.year, start_date.month, start_date.day,start_date.hour, start_date.minute, start_date.second)
    end_time = UTCDateTime(end_date.year, end_date.month, end_date.day ,end_date.hour, end_date.minute, end_date.second)
    for station in stations:
        outfile = "./" + str(station) + "/" + str(row['Year']) + str(row['Day']) + str(row['H']) + str(row['M']) + str(row['S'])+'.png'
        view_Apollo(start_time, end_time, station, outfile)




