# libraries for data handling/manipulation
import pandas as pd
import numpy as np

# plotting libraries
import matplotlib.pyplot as plt
import matplotlib
import matplotlib.dates as dates
import matplotlib.font_manager
from matplotlib.ticker import MaxNLocator
from itertools import cycle, islice
from matplotlib import cm

# mathematical libraries (integration, fourier transform, etc.)
from scipy import integrate
from scipy.fftpack import fft, ifft
from scipy.optimize import curve_fit
from scipy import asarray as ar, exp
from scipy.interpolate import interp1d
from scipy.stats.stats import pearsonr
import matplotlib.mlab as mlab

# Calendar, date, time handling libraries
import datetime
import calendar


class DisplayData:
    """ Proivdes extensive range of plotting functions to analyse and plot the processed covariance data
    e.g. to plot fluxes as a function of time, integrated fluxes, averaged fluxes by hour of the day, meteorological data
    such as windspeed, temperature and relative humidity histograms, calculations of correlation between water and CO2
    fluxes, etc.
    Due to high degree of customization, there are many redundancies in the code """

    def __init__(self):
        """ provides functions to extract and to plot data from eddy pro output files"""

        DIR = 'C:\Users\herrmann\Documents\Herrmann Lars\Licor\corrected\eddy_out\_tl_analysis_impact\\complete'
        FILE = 'full.csv'

        # -------------------------- PLOT METEOROLOGICAL DATA ----------------------------------
        # self.plot_windspeed_histograms()
        # self.windspeeds_by_hour()
        # self.plot_monthly_averages()
        # self.plot_meteo_histograms_biomet()

        # -------------------------- PLOT CORRELATION ANALYSIS ----------------------------------
        # self.plot_correlation_by_hour()
        # self.plot_correlation_entire_year()

        # -------------------------- PLOT FLUX ANALYSIS ----------------------------------
        # self.plot_h2o_flux_by_hour()
        # self.plot_co2_flux_by_hour()
        # self.plot_fluxes_entire_year()
        # self.plot_h2o_flux_vs_wind_speed()
        # self.plot_co2_flux_vs_wind_speed()
        # st = datetime.datetime(2017, 3, 27, 0, 0, 0)
        # self.plot_flux_of_day(st)

        # -------------------------- PLOT INTEGRATED DATA ----------------------------------
        # self.plot_custom_integrate_eddy_data_entire_year_water()
        # self.plot_custom_integrate_eddy_data_entire_year_co2()
        # self.integrate_fluxes(DIR, FILE)
        # self.extract_eddy_data(DIR, FILE)
        # self.compare_eddy_data(DIR)

        # -------------------------- PLOT FOOTPRINTS ----------------------------------
        # self.plot_footprints(DIR, FILE)

        # -------------------------- INTERNAL TIME LAG AND QUALITY FLAG ANALYSIS ----------------------------------
        # self.analyse_time_lags(DIR, FILE)
        # self.compute_fourier_transform(DIR, FILE)
        # self.plot_data_with_min_quality(DIR, FILE)
        # self.plot_qc_histograms(DIR, FILE)
        # self.plot_timelag_integrated_flux_comparison()

        # -------------------------- SAMPLING RATE ANALYSIS ----------------------------------
        # self.plot_daily_variance_sampling_rate()
        # self.plot_fluxes_sampling_rate()
        # self.plot_integrated_instant_sampling_rate()
        # self.plot_fluxes_sampling_rate()
        # self.compare_sampling_rate_histogram()
        # self.plot_fluxes_sampling_rate()

    def plot_windspeed_histograms(self):
        """ plots histograms of wind speed"""
        DIR1 = 'C:\Users\herrmann\Documents\Herrmann Lars\Licor\corrected\eddy_out\_tl_analysis_impact\\complete'
        FILE1 = 'full.csv'

        path = DIR1 + '\\' + FILE1
        df = pd.read_csv(path, sep=',', parse_dates=[['date', 'time']],
                         names=['date', 'time', 'co2flow', 'co2qc', 'h2oflow', 'temp', 'rh', 'u', 'v', 'w',
                                'windspeed'],
                         usecols=[1, 2, 13, 14, 15, 34, 44, 47, 48, 49, 53], header=2)

        # convert temp from K to C
        df['temp'] = df['temp'] - 273.15

        df.ix[df['u'] <= -25, 'u'] = np.nan
        df.ix[df['u'] >= 25, 'u'] = np.nan
        df.ix[df['v'] <= -25, 'v'] = np.nan
        df.ix[df['v'] >= 25, 'v'] = np.nan
        df.ix[df['temp'] <= -25, 'temp'] = np.nan
        df.ix[df['temp'] >= 50, 'temp'] = np.nan
        df.ix[df['rh'] < 0, 'rh'] = np.nan
        df.ix[df['rh'] > 100, 'rh'] = np.nan
        df.ix[df['windspeed'] <= -25, 'windspeed'] = np.nan
        df.ix[df['windspeed'] >= 25, 'windspeed'] = np.nan
        df = df.set_index(pd.DatetimeIndex(df['date_time']))

        fig, axs = plt.subplots(2, 6, figsize=(10, 4), facecolor='w', edgecolor='k', sharey=True, sharex=True)
        fig.subplots_adjust(hspace=0, wspace=0)

        axs = axs.ravel()
        st = datetime.date(2016, 4, 1)
        titles = ['Apr 16', 'May 16', 'Jun 16', 'Jul 16', 'Aug 16', 'Sep 16', 'Oct 16', 'Nov 16',
                  'Dec 16', 'Jan 17', 'Feb 17', 'Mar 17']

        k = 0
        for m in range(12):
            # specify analysis


            en = self.add_months(st, 1)
            dftmp = df[st:en].copy()
            st = en

            bounds = np.arange(0, 10, 1)
            ap = 0.8 # alpha value for histogram
            data = dftmp['v'].copy().dropna().tolist()
            weights = np.ones_like(data) / float(len(data))

            axs[m].hist(data, bins=bounds, color='royalblue', alpha=ap, weights=weights)

            # elif m<18:
            #     axs[m].hist(dftmp['u'].dropna().tolist(), bins=bounds, normed=True, color='royalblue', alpha=ap)
            # elif m<24:
            #     axs[m].hist(dftmp['v'].dropna().tolist(), bins=bounds, normed=True, color='deepskyblue', alpha=ap)


            from matplotlib.ticker import MultipleLocator, FormatStrFormatter

            majorLocator = MultipleLocator(10)
            axs[m].xaxis.set_major_locator(majorLocator)
            axs[m].xaxis.grid(True, which="major")
            axs[m].set_ylim([0, 0.6])
            axs[m].set_yticks([0, 0.2, 0.4])
            axs[m].set_xticks([0, 4, 8])

            #axs[m].yaxis.grid(False, which="major")


            axs[m].set_title(titles[k], y=0.80, x=0.4, fontsize=12)
            k += 1

            # if m == 9:
            #     axs[m].set_xlabel('')
            # if m == 4:
            #     axs[m].set_ylabel('relative frequency')

        #plt.tight_layout()
        outputpath = 'C:\Users\herrmann\Documents\Herrmann Lars\Documentation\\figs\\'
        filename = 'vertical_windspeed_histogram_monthly.png'
        plt.savefig(outputpath + filename, dpi=300)
        plt.show()

    def plot_meteo_histograms_internal(self):
        """ plots histograms of temperature and relative humidity of internal measures derived from anemometer"""
        DIR1 = 'C:\Users\herrmann\Documents\Herrmann Lars\Licor\corrected\eddy_out\_tl_analysis_impact\\complete'
        FILE1 = 'full.csv'

        path = DIR1 + '\\' + FILE1
        df = pd.read_csv(path, sep=',', parse_dates=[['date', 'time']],
                         names=['date', 'time', 'co2flow', 'co2qc', 'h2oflow', 'temp', 'rh', 'u', 'v', 'w',
                                'windspeed'],
                         usecols=[1, 2, 13, 14, 15, 34, 44, 47, 48, 49, 53], header=2)

        # convert temp from K to C
        df['temp'] = df['temp'] - 273.15

        df.ix[df['u'] <= -25, 'u'] = np.nan
        df.ix[df['u'] >= 25, 'u'] = np.nan
        df.ix[df['v'] <= -25, 'v'] = np.nan
        df.ix[df['v'] >= 25, 'v'] = np.nan
        df.ix[df['temp'] <= -25, 'temp'] = np.nan
        df.ix[df['temp'] >= 50, 'temp'] = np.nan
        df.ix[df['rh'] < 0, 'rh'] = np.nan
        df.ix[df['rh'] > 100, 'rh'] = np.nan
        df.ix[df['windspeed'] <= -25, 'windspeed'] = np.nan
        df.ix[df['windspeed'] >= 25, 'windspeed'] = np.nan
        df = df.set_index(pd.DatetimeIndex(df['date_time']))

        fig, axs = plt.subplots(3, 8, figsize=(10, 6), facecolor='w', edgecolor='k', sharey=True, sharex=True)
        fig.subplots_adjust(hspace=0, wspace=0)

        axs = axs.ravel()
        st = datetime.date(2016, 4, 1)
        titles = ['Apr 16', 'May 16', 'Jun 16', 'Jul 16', 'Aug 16', 'Sep 16', 'Oct 16', 'Nov 16',
                  'Dec 16', 'Jan 17', 'Feb 17', 'Mar 17']

        k = 0
        for m in range(24):
            # specify analysis

            if m%2 == 0:
                en = self.add_months(st, 1)
                dftmp = df[st:en].copy()
                st = en

            bounds = np.arange(0, 32.5, 2.5)

            if m%2==0:
                axs[m].hist(dftmp['temp'].dropna().tolist(), bins=bounds, normed=True, color='darkgoldenrod', alpha=0.5)
            else:
                axs[m].hist(dftmp['rh'].dropna().tolist(), bins=bounds, normed=True, color='purple', alpha=0.5)



            from matplotlib.ticker import MultipleLocator, FormatStrFormatter

            majorLocator = MultipleLocator(10)
            axs[m].xaxis.set_major_locator(majorLocator)
            axs[m].xaxis.grid(True, which="major")
            axs[m].set_ylim([0, 0.25])
            axs[m].set_yticks([0, 0.08, 0.16])
            axs[m].set_xticks([0, 10, 20])

            #axs[m].yaxis.grid(False, which="major")

            if m%2 == 0:
                axs[m].set_title(titles[k], y=0.82, x=0.4, fontsize=12)
                k += 1

            # if m == 9:
            #     axs[m].set_xlabel('')
            # if m == 4:
            #     axs[m].set_ylabel('relative frequency')

        #plt.tight_layout()
        # outputpath = 'C:\Users\herrmann\Documents\Herrmann Lars\Documentation\\figs\\'
        # filename = 'temperature_rh_histograms_monthly.png'
        # plt.savefig(outputpath + filename, dpi=300)
        plt.show()

    def plot_meteo_histograms_biomet(self):
        """ plots histograms of temperature and relative humidity of biomet data"""
        DIR1 = 'C:\Users\herrmann\Documents\Herrmann Lars\Licor\corrected\\biomet'
        FILE1 = 'biomet.csv'

        path = DIR1 + '\\' + FILE1
        df = pd.read_csv(path, sep=',', parse_dates=['date_time'],
                         names=['date_time', 'P', 'RH', 'T'],
                         usecols=[0, 1, 2, 3], header=2)

        # convert temp from K to C
        df['T'] = df['T'] - 273.15

        df = df.set_index(pd.DatetimeIndex(df['date_time']))

        fig, axs = plt.subplots(3, 8, figsize=(10, 6), facecolor='w', edgecolor='k', sharey=True, sharex=True)
        fig.subplots_adjust(hspace=0, wspace=0)

        axs = axs.ravel()
        st = datetime.date(2016, 4, 1)
        titles = ['Apr 16', 'May 16', 'Jun 16', 'Jul 16', 'Aug 16', 'Sep 16', 'Oct 16', 'Nov 16',
                  'Dec 16', 'Jan 17', 'Feb 17', 'Mar 17']

        k = 0
        for m in range(24):
            # specify analysis

            if m%2 == 0:
                en = self.add_months(st, 1)
                dftmp = df[st:en].copy()
                st = en



            if m%2==0:
                bounds = np.arange(-7.5, 32.5, 2.5)
                axs[m].hist(dftmp['T'].dropna().tolist(), bins=bounds, normed=True, color='darkgoldenrod', alpha=0.5)
                axs[m].set_xlim([-10, 30])
                axs[m].set_xticks([0, 10, 20])
            else:
                bounds = np.arange(0, 100, 5)
                axs[m].hist(dftmp['RH'].dropna().tolist(), bins=bounds, normed=True, color='purple', alpha=0.5)
                axs[m].set_xlim([-20, 100])
                axs[m].set_xticks([0, 40, 80])
            axs[m].xaxis.grid(True, which="major")
            axs[m].set_ylim([0, 0.15])
            axs[m].set_yticks([0, 0.05, 0.1])

            if m%2 == 0:
                axs[m].set_title(titles[k], y=0.82, x=0.4, fontsize=12)
                k += 1

        plt.tight_layout()
        outputpath = 'C:\Users\herrmann\Documents\Herrmann Lars\Documentation\\figs\\'
        filename = 'temperature_rh_histograms_monthly_external_meteo.png'
        plt.savefig(outputpath + filename, dpi=300)
        plt.show()

    def plot_monthly_meteo_averages(self):
        """ plots monthly averages of temperature and humidity
        both internal and external data is t used """

        # ---------Load internal anemometer data
        DIR1 = 'C:\Users\herrmann\Documents\Herrmann Lars\Licor\corrected\eddy_out\_tl_analysis_impact\\complete'
        FILE1 = 'full.csv'

        path = DIR1 + '\\' + FILE1
        df = pd.read_csv(path, sep=',', parse_dates=[['date', 'time']],
                         names=['date', 'time', 'temp', 'rh'],
                         usecols=[1, 2, 34, 44], header=2)

        # convert temp from K to C
        df['temp'] = df['temp'] - 273.15
        df.ix[df['temp'] <= -25, 'temp'] = np.nan
        df.ix[df['temp'] >= 50, 'temp'] = np.nan
        df.ix[df['rh'] < 0, 'rh'] = np.nan
        df.ix[df['rh'] > 100, 'rh'] = np.nan
        df = df.set_index(pd.DatetimeIndex(df['date_time']))

        # ---------Load external biomet data
        DIR1 = 'C:\Users\herrmann\Documents\Herrmann Lars\Licor\corrected\\biomet'
        FILE1 = 'biomet.csv'

        path = DIR1 + '\\' + FILE1
        dfext = pd.read_csv(path, sep=',', parse_dates=['date_time'],
                         names=['date_time', 'P', 'RH', 'T'],
                         usecols=[0, 1, 2, 3], header=2)

        # convert temp from K to C
        dfext['T'] = dfext['T'] - 273.15
        dfext = dfext.set_index(pd.DatetimeIndex(dfext['date_time']))

        st = datetime.date(2016, 4, 1)
        x=range(12)
        av_temp = []
        av_temp_err1 = []
        av_temp_err2 = []
        av_rh = []
        av_rh_err1 = []
        av_rh_err2 = []

        av_tempext = []
        av_temp_errext = []
        av_temp_err1ext = []
        av_temp_err2ext = []
        av_rhext = []
        av_rh_err1ext = []
        av_rh_err2ext = []

        titles = ['Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov',
                  'Dec', 'Jan', 'Feb', 'Mar']
        for m in range(12):
            # specify analysis

            en = self.add_months(st, 1)
            dftmp = df[st:en].copy()
            dftmpext = dfext[st:en].copy()
            st = en
            print dftmp['temp'].mean()

            av_temp.append(dftmp['temp'].mean())
            av_temp_err1.append(dftmp['temp'].mean() + dftmp['temp'].std())
            av_temp_err2.append(dftmp['temp'].mean() - dftmp['temp'].std())

            av_tempext.append(dftmpext['T'].mean())
            av_temp_errext.append(dftmpext['T'].std())
            av_temp_err1ext.append(dftmpext['T'].astype(float).mean() + dftmpext['T'].std())
            av_temp_err2ext.append(dftmpext['T'].mean() - dftmpext['T'].std())

            av_rh.append(dftmp['rh'].mean())
            av_rh_err1.append(dftmp['rh'].mean() + dftmp['rh'].std())
            av_rh_err2.append(dftmp['rh'].mean() - dftmp['rh'].std())

            av_rhext.append(dftmpext['RH'].mean())
            av_rh_err1ext.append(dftmpext['RH'].mean() + dftmpext['RH'].std())
            av_rh_err2ext.append(dftmpext['RH'].mean() - dftmpext['RH'].std())

        fig = plt.figure(figsize=(7,4))
        ax1 = fig.add_subplot(111)
        ax2 = fig.add_subplot(111, sharex=ax1, frameon=False)

        plt.xticks(x, titles)
        plt.xticks(range(12), titles, rotation=0)  # writes strings with 45 degree angle
        ax1.set_ylabel('Temperature ($^\circ$C)')

        ax1.plot(x, av_temp, color='orange', alpha=0.7)
        #ax1.fill_between(x, y1=av_temp_err1, y2=av_temp_err2, color='darkgoldenrod', alpha=0.3, linewidth=0)

        ax1.plot(x, av_tempext, color='darkgoldenrod')
        #ax1.fill_between(x, y1=av_temp_err1ext, y2=av_temp_err2ext, color='darkgoldenrod', alpha=0.3, linewidth=0)
        ax1.set_ylim([-20, 60])

        ax2.plot(x, av_rh, color='violet')
        #ax2.fill_between(x, y1=av_rh_err1, y2=av_rh_err2, color='purple', alpha=0.3, linewidth=0)

        ax2.plot(x, av_rhext, color='purple')
        #ax2.fill_between(x, y1=av_rh_err1ext, y2=av_rh_err2ext, color='purple', alpha=0.3, linewidth=0)

        ax2.yaxis.tick_right()
        ax2.yaxis.set_label_position("right")
        ax2.set_ylabel('Relative humidity (%)', rotation=-90, labelpad=20)
        ax2.set_ylim([0, 120])
        ax1.xaxis.grid(linestyle='--')
        plt.tight_layout()
        outputpath = 'C:\Users\herrmann\Documents\Herrmann Lars\Documentation\\figs\\'
        filename = 'temperature_humidity_monthly_with_ext_and_int.png'
        plt.savefig(outputpath + filename, dpi=300)
        plt.show()

    def plot_windspeeds_by_hour(self):
        """ This averages the horizontal and vertical wind speed over a certain daily hour over a month
        """
        """ plots the co2 flux vs the wind speed for the entire data set or possibly for subsets (e.g. daytime, nighttime, etc.)"""
        DIR1 = 'C:\Users\herrmann\Documents\Herrmann Lars\Licor\corrected\eddy_out\_tl_analysis_impact\\complete'
        FILE1 = 'full.csv'

        path = DIR1 + '\\' + FILE1
        df = pd.read_csv(path, sep=',', parse_dates=[['date', 'time']],
                         names=['date', 'time', 'co2flow', 'co2qc', 'h2oflow', 'temp', 'rh', 'u', 'v', 'w', 'windspeed'],
                         usecols=[1, 2, 13, 14, 15, 34, 44, 47, 48, 49, 53], header=2)

        # convert temp from K to C
        df['temp'] = df['temp'] - 273.15

        df.ix[df['u'] <= -25, 'u'] = np.nan
        df.ix[df['u'] >= 25, 'u'] = np.nan
        df.ix[df['v'] <= -25, 'v'] = np.nan
        df.ix[df['v'] >= 25, 'v'] = np.nan
        df.ix[df['temp'] <= -25, 'temp'] = np.nan
        df.ix[df['temp'] >= 50, 'temp'] = np.nan
        df.ix[df['rh'] < 0, 'rh'] = np.nan
        df.ix[df['rh'] > 100, 'rh'] = np.nan
        df.ix[df['windspeed'] <= -25, 'windspeed'] = np.nan
        df.ix[df['windspeed'] >= 25, 'windspeed'] = np.nan
        df = df.set_index(pd.DatetimeIndex(df['date_time']))

        df['horizontal'] = np.sqrt(df['u'] ** 2 + df['v'] ** 2)

        fig, axs = plt.subplots(3, 4, figsize=(10, 6), facecolor='w', edgecolor='k', sharey=True,sharex=True)
        fig.subplots_adjust(hspace=0, wspace=0)

        axs = axs.ravel()
        st = datetime.date(2016, 4, 1)
        titles = ['Apr 2016', 'May 2016', 'Jun 2016', 'Jul 2016', 'Aug 2016', 'Sep 2016', 'Oct 2016', 'Nov 2016',
                  'Dec 2016', 'Jan 2017', 'Feb 2017', 'Mar 2017']
        for m in range(12):
            # specify analysis

            en = self.add_months(st, 1)
            dftmp = df[st:en].copy()
            st = en

            hour = dftmp.index.hour
            res = []
            err1 = []
            err2 = []
            resv = []
            errv = []
            errv1 = []
            errv2 = []
            rest= []
            errt = []
            errt1 = []
            errt2 = []
            time = range(24)
            time = [k+1 for k in time]
            for i in range(24):
                selector = (i == hour)
                df2 = dftmp[selector].copy()

                res.append(df2['windspeed'].mean())
                err1.append(df2['windspeed'].mean()+df2['windspeed'].std())
                err2.append(df2['windspeed'].mean()-df2['windspeed'].std())

                resv.append(df2['rh'].mean())
                errv.append(df2['rh'].std())
                errv1.append(df2['rh'].mean() + df2['rh'].std())
                errv2.append(df2['rh'].mean() - df2['rh'].std())

                rest.append(df2['temp'].mean())
                errt.append(df2['temp'].std())
                errt1.append(df2['temp'].mean() + df2['temp'].std())
                errt2.append(df2['temp'].mean() - df2['temp'].std())

            # plot hourly wind speeds
            # axs[m].errorbar(time, res, marker='.', color='royalblue')
            # axs[m].fill_between(time, y1=err1, y2=err2, color='royalblue', alpha=0.3)

            # plot hourly relative humidity
            #axs[m].errorbar(time, resv, marker='.', color='purple')
            #axs[m].fill_between(time, y1=errv1, y2=errv2, color='purple', alpha=0.3)

            # plot hourly temperature
            axs[m].errorbar(time, rest, marker='.', color='darkgoldenrod')
            axs[m].fill_between(time, y1=errt1, y2=errt2, color='darkgoldenrod', alpha=0.3)

            axs[m].set_ylim([15, 30])
            axs[m].set_xlim([1, 24])

            from matplotlib.ticker import MultipleLocator, FormatStrFormatter

            majorLocator = MultipleLocator(6)
            axs[m].xaxis.set_major_locator(majorLocator)
            axs[m].xaxis.grid(True, which="major")
            axs[m].yaxis.set_major_locator(MaxNLocator(4))
            axs[m].yaxis.grid(True, which="major")

            axs[m].set_title(titles[m], y=0.84, x=0.7, fontsize=12)

            if m == 9:
                axs[m].set_xlabel('')
            if m == 4:
                axs[m].set_ylabel('temperature ($^\circ$C)')

        #plt.tight_layout()
        # outputpath = 'C:\Users\herrmann\Documents\Herrmann Lars\Documentation\\figs\\'
        # filename = 'temperature_hourly_average.png'
        # plt.savefig(outputpath + filename, dpi=300)
        plt.show()

    def plot_co2_flux_by_hour(self):
        """ This averages the co2 flux over a certain daily hour over a certain range (here month)
        """
        """ plots the co2 flux vs the wind speed for the entire data set or possibly for subsets (e.g. daytime, nighttime, etc.)"""
        DIR1 = 'C:\Users\herrmann\Documents\Herrmann Lars\Licor\corrected\eddy_out\_tl_analysis_impact\\complete'
        FILE1 = 'full.csv'

        path = DIR1 + '\\' + FILE1
        df = pd.read_csv(path, sep=',', parse_dates=[['date', 'time']],
                         names=['date', 'time', 'co2flow', 'co2qc', 'h2oflow', 'windspeed'],
                         usecols=[1, 2, 13, 14, 15, 53], header=2)

        # replace all values with missing data with the error flag code -9999
        df.replace(to_replace=np.nan, value=-9999, inplace=True)
        df.replace(to_replace=0, value=-9999, inplace=True)
        # replace data where quality control flag is 2 (bad) with np.nan
        df['orig'] = df['co2flow']
        df['bad'] = df.ix[df['co2qc'] == 2.0, 'co2flow']
        df.ix[df['co2qc'] == 2.0, 'co2flow'] = np.nan

        # now interpolate all nan values
        df.interpolate(method='linear', inplace=True)
        df.ix[df['co2flow'] <= -50, 'co2flow'] = np.nan
        df.ix[df['co2flow'] >= 50, 'co2flow'] = np.nan
        df = df.set_index(pd.DatetimeIndex(df['date_time']))

        fig, axs = plt.subplots(3, 4, figsize=(10, 10), facecolor='w', edgecolor='k', sharey=True,sharex=True)
        fig.subplots_adjust(hspace=.05, wspace=.05)

        axs = axs.ravel()
        st = datetime.date(2016, 4, 1)
        titles = ['Apr 2016', 'May 2016', 'Jun 2016', 'Jul 2016', 'Aug 2016', 'Sep 2016', 'Oct 2016', 'Nov 2016',
                  'Dec 2016', 'Jan 2017', 'Feb 2017', 'Mar 2017']
        for m in range(12):
            # specify analysis

            en = self.add_months(st, 1)
            dftmp = df[st:en].copy()
            st = en

            hour = dftmp.index.hour
            res = []
            err = []
            time = range(24)
            time = [k+1 for k in time]
            sem_a = []
            sem_b = []
            from scipy import stats
            for i in range(24):
                selector = (i == hour)
                df2 = dftmp[selector].copy()
                res.append(df2['co2flow'].mean())
                err.append(df2['co2flow'].std())
                div = calendar.monthrange(st.year, st.month)[1]
                sem_a.append(df2['co2flow'].mean()-df2['co2flow'].mean()/np.sqrt(div))
                sem_b.append(df2['co2flow'].mean() + df2['co2flow'].mean() / np.sqrt(div))


            axs[m].errorbar(time, res, yerr=err, marker='.', color='royalblue')
            axs[m].fill_between(time, y1=sem_a, y2=sem_b, color='royalblue', alpha=0.8)
            axs[m].set_ylim([-20, 20])
            axs[m].set_xlim([1,24])

            from matplotlib.ticker import MultipleLocator, FormatStrFormatter
            #minorLocator = MultipleLocator(3)
            majorLocator = MultipleLocator(6)
            #axs[m].xaxis.set_minor_locator(minorLocator)
            axs[m].xaxis.set_major_locator(majorLocator)
            axs[m].xaxis.grid(True, which="major")
            axs[m].yaxis.set_major_locator(MaxNLocator(6))
            axs[m].yaxis.grid(True, which="major")

            axs[m].set_title(titles[m], y=0.87, x=0.7, fontsize=12)

            if m == 9:
                axs[m].set_xlabel('')
            if m == 4:
                axs[m].set_ylabel('average CO$_2$ flux ($\mu$mol m$^{-2}$s$^{-1}$)')

            # plt.figure(figsize=(5,5))
            # plt.plot(time, res, marker='.', color='royalblue')
            # plt.xlabel('hour')
            # plt.ylabel('average CO$_2$ flow ($\mu$mol m$^{-2}$s$^{-1}$)')
            # plt.tight_layout()
            # plt.show()
        plt.tight_layout()
        outputpath = 'C:\Users\herrmann\Documents\Herrmann Lars\Documentation\\figs\\'
        filename = 'flux_hourly_average_tmp.png'
        plt.savefig(outputpath + filename, dpi=300)
        plt.show()

    def plot_h2o_flux_by_hour(self):
        """ This averages the h2o flux over a certain daily hour over a certain range (here month)
        """
        DIR1 = 'C:\Users\herrmann\Documents\Herrmann Lars\Licor\corrected\eddy_out\_tl_analysis_impact\\complete'
        FILE1 = 'full.csv'

        path = DIR1 + '\\' + FILE1
        df = pd.read_csv(path, sep=',', parse_dates=[['date', 'time']],
                         names=['date', 'time', 'co2flow', 'co2qc', 'h2oflow', 'h2oqc', 'windspeed'],
                         usecols=[1, 2, 13, 14, 15, 16, 53], header=2)

        # replace all values with missing data with the error flag code -9999
        df.replace(to_replace=np.nan, value=-9999, inplace=True)
        df.replace(to_replace=0, value=-9999, inplace=True)
        # replace data where quality control flag is 2 (bad) with np.nan
        df['orig'] = df['h2oflow']
        df['bad'] = df.ix[df['h2oqc'] == 2.0, 'h2oflow']
        df.ix[df['h2oqc'] == 2.0, 'h2oflow'] = np.nan

        # now interpolate all nan values
        df.interpolate(method='linear', inplace=True)
        df.ix[df['h2oflow'] <= -5, 'h2oflow'] = np.nan
        df.ix[df['h2oflow'] >= 5, 'h2oflow'] = np.nan
        df = df.set_index(pd.DatetimeIndex(df['date_time']))

        fig, axs = plt.subplots(3, 4, figsize=(10, 10), facecolor='w', edgecolor='k', sharey=True,sharex=True)
        fig.subplots_adjust(hspace=.05, wspace=.05)

        axs = axs.ravel()
        st = datetime.date(2016, 4, 1)
        titles = ['Apr 2016', 'May 2016', 'Jun 2016', 'Jul 2016', 'Aug 2016', 'Sep 2016', 'Oct 2016', 'Nov 2016',
                  'Dec 2016', 'Jan 2017', 'Feb 2017', 'Mar 2017']
        for m in range(12):
            # specify analysis

            en = self.add_months(st, 1)
            dftmp = df[st:en].copy()
            st = en

            hour = dftmp.index.hour
            res = []
            err = []
            sem_a=[]
            sem_b=[]
            time = range(24)
            time = [k+1 for k in time]
            for i in range(24):
                selector = (i == hour)
                df2 = dftmp[selector].copy()
                res.append(df2['h2oflow'].mean())
                err.append(df2['h2oflow'].std())
                div = calendar.monthrange(st.year, st.month)[1]
                sem_a.append(df2['h2oflow'].mean() - df2['h2oflow'].mean() / np.sqrt(div))
                sem_b.append(df2['h2oflow'].mean() + df2['h2oflow'].mean() / np.sqrt(div))
            #print res
            axs[m].errorbar(time, res, yerr=err, marker='.', color='seagreen')
            axs[m].fill_between(time, y1=sem_a, y2=sem_b, color='seagreen', alpha=0.8)
            axs[m].set_ylim([-1.5, 2.4])
            axs[m].set_xlim([1,24])

            from matplotlib.ticker import MultipleLocator, FormatStrFormatter
            #minorLocator = MultipleLocator(3)
            majorLocator = MultipleLocator(6)
            #axs[m].xaxis.set_minor_locator(minorLocator)
            axs[m].xaxis.set_major_locator(majorLocator)
            axs[m].xaxis.grid(True, which="major")
            axs[m].yaxis.set_major_locator(MaxNLocator(6))
            axs[m].yaxis.grid(True, which="major")

            axs[m].set_title(titles[m], y=0.87, x=0.7, fontsize=12)

            if m == 9:
                axs[m].set_xlabel('')
            if m == 4:
                axs[m].set_ylabel('average H$_2$O flux (10 mmol m$^{-2}$s$^{-1}$)')

            # plt.figure(figsize=(5,5))
            # plt.plot(time, res, marker='.', color='royalblue')
            # plt.xlabel('hour')
            # plt.ylabel('average CO$_2$ flow ($\mu$mol m$^{-2}$s$^{-1}$)')
            # plt.tight_layout()
            # plt.show()
        plt.tight_layout()
        outputpath = 'C:\Users\herrmann\Documents\Herrmann Lars\Documentation\\figs\\'
        filename = 'h2o_flux_hourly.png'
        plt.savefig(outputpath + filename, dpi=300)
        plt.show()



    def plot_co2_flux_vs_wind_speed(self):
        """ plots the co2 flux vs the wind speed for the entire data set or possibly for subsets (e.g. daytime, nighttime, etc.)"""
        DIR1 = 'C:\Users\herrmann\Documents\Herrmann Lars\Licor\corrected\eddy_out\_tl_analysis_impact\\complete'
        FILE1 = 'full.csv'

        path = DIR1 + '\\' + FILE1
        df = pd.read_csv(path, sep=',', parse_dates=[['date', 'time']],
                         names=['date', 'time', 'co2flow', 'co2qc', 'h2oflow', 'w', 'windspeed'],
                         usecols=[1, 2, 13, 14, 15, 49, 53], header=2)

        # replace all values with missing data with the error flag code -9999
        df.replace(to_replace=np.nan, value=-9999, inplace=True)
        df.replace(to_replace=0, value=-9999, inplace=True)
        # replace data where quality control flag is 2 (bad) with np.nan
        df['orig'] = df['co2flow']
        df['bad'] = df.ix[df['co2qc'] == 2.0, 'co2flow']
        df.ix[df['co2qc'] == 2.0, 'co2flow'] = np.nan

        # now interpolate all nan values
        df.interpolate(method='linear', inplace=True)
        df.ix[df['co2flow'] <= -100, 'co2flow'] = np.nan
        df = df.set_index(pd.DatetimeIndex(df['date_time']))
        hour = df.index.hour

        # night time
        selector = ((19 <= hour) & (hour <= 23)) | (( 00 <= hour) & (hour <= 05))
        # day time
        #selector = ((06 <= hour) & (hour <= 18))
        df = df[selector]

        df.plot(x='w', y='co2flow', linewidth=0, marker='.', markersize=2, color='black', figsize=(5,4), legend=False)
        plt.xlim([0, 1.5])
        plt.ylim([-30, 30])
        plt.ylabel('CO$_2$ flow ($\mu$mol m$^{-2}$s$^{-1}$)')
        plt.xlabel('mean wind speed (m/s)')
        plt.tight_layout()
        outputpath = 'C:\Users\herrmann\Documents\Herrmann Lars\Documentation\\figs\\'
        filename = 'co2_flux_vs_windspeed_vertical_night.png'
        plt.savefig(outputpath + filename, dpi=300)
        plt.show()

    def plot_h2o_flux_vs_wind_speed(self):
        """ plots the co2 flux vs the wind speed for the entire data set or possibly for subsets (e.g. daytime, nighttime, etc.)"""
        DIR1 = 'C:\Users\herrmann\Documents\Herrmann Lars\Licor\corrected\eddy_out\_tl_analysis_impact\\complete'
        FILE1 = 'full.csv'

        path = DIR1 + '\\' + FILE1
        df = pd.read_csv(path, sep=',', parse_dates=[['date', 'time']],
                         names=['date', 'time', 'co2flow', 'co2qc', 'h2oflow', 'h2oqc', 'w', 'windspeed'],
                         usecols=[1, 2, 13, 14, 15, 16, 49, 53], header=2)

        # replace all values with missing data with the error flag code -9999
        df.replace(to_replace=np.nan, value=-9999, inplace=True)
        df.replace(to_replace=0, value=-9999, inplace=True)
        # replace data where quality control flag is 2 (bad) with np.nan
        df['orig'] = df['h2oflow']
        df['bad'] = df.ix[df['h2oqc'] == 2.0, 'h2oflow']
        df.ix[df['co2qc'] == 2.0, 'h2oflow'] = np.nan

        # now interpolate all nan values
        df.interpolate(method='linear', inplace=True)
        df.ix[df['h2oflow'] <= -5, 'h2oflow'] = np.nan
        df = df.set_index(pd.DatetimeIndex(df['date_time']))
        hour = df.index.hour

        # night time
        #selector = ((19 <= hour) & (hour <= 23)) | (( 00 <= hour) & (hour <= 05))
        # day time
        #selector = ((06 <= hour) & (hour <= 18))
        #df = df[selector]

        df.plot(x='w', y='h2oflow', linewidth=0, marker='.', markersize=2, color='royalblue', figsize=(5,4), legend=False)
        plt.xlim([0, 1.5])
        plt.ylim([-5, 5])
        plt.ylabel('H$_2$O flow (mmol m$^{-2}$s$^{-1}$)')
        plt.xlabel('mean vertical wind speed (m/s)')
        plt.tight_layout()
        outputpath = 'C:\Users\herrmann\Documents\Herrmann Lars\Documentation\\figs\\'
        filename = 'h2o_flux_vs_windspeed_vertical_overall.png'
        plt.savefig(outputpath + filename, dpi=300)
        plt.show()


    def plot_fluxes_entire_year(self):
        DIR1 = 'C:\Users\herrmann\Documents\Herrmann Lars\Licor\corrected\eddy_out\_tl_analysis_impact\\complete'
        FILE1 = 'full.csv'

        path = DIR1 + '\\' + FILE1
        df = pd.read_csv(path, sep=',', parse_dates=[['date', 'time']],
                         names=['date', 'time', 'co2flow', 'co2qc', 'h2oflow', 'h2oqc'],
                         usecols=[1, 2, 13, 14, 15, 16], header=2)

        # replace all values with missing data with the error flag code -9999
        df.replace(to_replace=np.nan, value=-9999, inplace=True)
        df.replace(to_replace=0, value=-9999, inplace=True)
        # replace data where quality control flag is 2 (bad) with np.nan
        df['orig'] = df['co2flow']
        df['bad'] = df.ix[df['co2qc'] == 2.0, 'co2flow']

        df.ix[df['co2qc'] == 2.0, 'co2flow'] = np.nan
        df.ix[df['h2oqc'] == 2.0, 'h2oflow'] = np.nan
        # now interpolate all nan values
        df.interpolate(method='linear', inplace=True)

        #df.fillna(axis=0, method='ffill', inplace=True)
        df.ix[df['co2flow'] <= -100, 'shades'] = True
        df.ix[df['co2flow'] >= -100, 'shades'] = False

        df.ix[df['h2oflow'] <= -100, 'shades_h2o'] = True
        df.ix[df['h2oflow'] >= -100, 'shades_h2o'] = False

        # now bring back the -9999 values to np.nan
        df.replace(to_replace=-9999, value=np.nan, inplace=True)

        # ensure not to plot outliers
        df.ix[df['co2flow'] < -25.0, 'co2flow'] = np.nan
        df.ix[df['h2oflow'] < -10.0, 'h2oflow'] = np.nan

        df = df.set_index(pd.DatetimeIndex(df['date_time']))

        # WHAT TO PLOT
        average = True
        co2_flow = False

        # Apply rolling average
        if average:
            df['av1'] = df['co2flow'].rolling(window=6).mean()
            df['av2'] = df['h2oflow'].rolling(window=6).mean()
            df.interpolate(method='linear', inplace=True)
            if co2_flow:
                df.ix[df['shades']==True, 'av1'] = np.nan
            else:
                df.ix[df['shades_h2o'] == True, 'av2'] = np.nan

            df = df.set_index(pd.DatetimeIndex(df['date_time']))



        start = datetime.date(year=2016, month=4, day=1)
        stop = start
        titles = ['Apr 2016', 'May 2016', 'Jun 2016', 'Jul 2016', 'Aug 2016', 'Sep 2016', 'Oct 2016', 'Nov 2016',
                  'Dec 2016', 'Jan 2017', 'Feb 2017', 'Mar 2017']
        for t in range(12):
            fig, ax = plt.subplots(figsize=(10, 2.5))
            if co2_flow:
                if average:
                    ax.plot(df.index, df.av1, color='black')
                else:
                    ax.plot(df.index, df['co2flow'], color='black')
                ax.set_ylabel('CO$_2$ flux ($\mu$mol m$^{-2}$s$^{-1}$)')
                ax.set_ylim([-25, 25])
            else:
                if average:
                    ax.plot(df.index, df.av2, color='black')
                else:
                    ax.plot(df.index, df['h2oflow'], color='black')
                ax.set_ylabel('H$_2$O flux (10 mmol m$^{-2}$s$^{-1}$)')
                ax.set_ylim([-5, 5])
            start = stop
            stop = self.add_months(start, 1)
            ax.set_xlim(pd.Timestamp(start), pd.Timestamp(stop))

            ax.set_axis_bgcolor('white')

            if co2_flow:
                ax.fill_between(df.index, y1=100, y2=-100, where=df['shades'], color='purple', alpha=0.2, linewidth=0)
            else:
                ax.fill_between(df.index, y1=100, y2=-100, where=df['shades_h2o'], color='purple', alpha=0.2, linewidth=0)
            ax.xaxis.set_minor_locator(dates.DayLocator())
            ax.xaxis.grid(True, which="minor", linestyle='--')
            ax.yaxis.set_major_locator(MaxNLocator(4))
            ax.yaxis.grid(True, which="major", linestyle='--')
            ax.set_title(titles[t], y=0.88, x=0.05, fontsize=10, fontweight='bold')
            plt.tight_layout()
            outputpath = 'C:\Users\herrmann\Documents\Herrmann Lars\Documentation\\figs\\'
            if co2_flow:
                if average:
                    filename = 'v2_average_co2flux_analysis_' + str(t) + '.png'
                else:
                    filename = 'v2_co2flux_analysis_' + str(t) + '.png'
            else:
                if average:
                    filename = 'v2_average_h2oflux_analysis_' + str(t) + '.png'
                else:
                    filename = 'v2_h2oflux_analysis_' + str(t) + '.png'
            plt.savefig(outputpath + filename, dpi=300)

    def plot_timelag_integrated_flux_comparison(self):
        """ plots integrated fluxes for different set (constant) time lags in the EddyPro analysis"""
        month = 'aug16'
        DIR1 = 'C:\Users\herrmann\Documents\Herrmann Lars\Licor\corrected\eddy_out\_tl_analysis_impact\\' + month + '_t56'
        FILE1 = 'full.csv'
        DIR2 = 'C:\Users\herrmann\Documents\Herrmann Lars\Licor\corrected\eddy_out\_tl_analysis_impact\\' + month + '_t60'
        FILE2 = 'full.csv'
        DIR3 = 'C:\Users\herrmann\Documents\Herrmann Lars\Licor\corrected\eddy_out\_tl_analysis_impact\\' + month + '_t64'
        FILE3 = 'full.csv'
        DIR4 = 'C:\Users\herrmann\Documents\Herrmann Lars\Licor\corrected\eddy_out\_tl_analysis_impact\\' + month + '_t68'
        FILE4 = 'full.csv'

        fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(5,3))

        df1 = self.custom_integrate_eddy_data(DIR1, FILE1)
        df1.plot(x='date_time', y='integration', ax=axes, label='56 s', color='royalblue')
        df2 = self.custom_integrate_eddy_data(DIR2, FILE2)
        df2.plot(x='date_time', y='integration', ax=axes, label='60 s', color='darkgoldenrod')
        df3 = self.custom_integrate_eddy_data(DIR3, FILE3)
        df3.plot(x='date_time', y='integration', ax=axes, label='64 s', color='purple')
        df4 = self.custom_integrate_eddy_data(DIR4, FILE4)
        df4.plot(x='date_time', y='integration', ax=axes, label='68 s', color='seagreen')
        plt.xlim(pd.Timestamp('2016-08-01'), pd.Timestamp('2016-09-01'))
        plt.ylabel('CO$_2$ flux (mol m$^{-2}$)')
        plt.legend(frameon=False, prop={'size': 10})
        plt.xlabel('time')
        plt.tight_layout()
        outputpath = 'C:\Users\herrmann\Documents\Herrmann Lars\Documentation\\figs\\'
        filename = 'integrated_co2flux_august2016.pdf'
        plt.savefig(outputpath + filename, dpi=300)
        plt.show()

    def plot_min_max_flux_vs_timelag(self):
        """ plot sthe minimum and maximum flux for different set constant time lags in Eddy Pro covariance analysis"""
        month = 'dec16'
        DIR1 = 'C:\Users\herrmann\Documents\Herrmann Lars\Licor\corrected\eddy_out\_tl_analysis_impact\\' + month + '_t56'
        FILE1 = 'full.csv'
        DIR2 = 'C:\Users\herrmann\Documents\Herrmann Lars\Licor\corrected\eddy_out\_tl_analysis_impact\\' + month + '_t60'
        DIR3 = 'C:\Users\herrmann\Documents\Herrmann Lars\Licor\corrected\eddy_out\_tl_analysis_impact\\' + month + '_t64'
        DIR4 = 'C:\Users\herrmann\Documents\Herrmann Lars\Licor\corrected\eddy_out\_tl_analysis_impact\\' + month + '_t68'

        dlst = [DIR1, DIR2, DIR3, DIR4]
        df1 = pd.DataFrame()
        df2 = pd.DataFrame()
        df3 = pd.DataFrame()
        df4 = pd.DataFrame()
        for idx, i in enumerate(dlst):
            path = i + '\\' + FILE1
            df = pd.read_csv(path, sep=',', parse_dates=[['date', 'time']],
                             names=['date', 'time', 'co2flow', 'co2qc', 'h2oflow', 'h2oqc'],
                             usecols=[1, 2, 13, 14, 15, 16], header=2)

            # need to remove all values where qc is 2
            df.ix[df['co2qc'] == 2.0, 'co2flow'] = np.nan
            df.ix[df['h2oqc'] == 2.0, 'h2oflow'] = np.nan
            df.interpolate(method='linear', axis=0, inplace=True)

            df.replace(to_replace=-9999, value=np.nan, inplace=True)
            df.replace(to_replace=0.0, value=np.nan, inplace=True)



            df.ix[df['co2flow'] <= -25, 'co2flow'] = np.nan
            df.ix[df['co2flow'] >= 25, 'co2flow'] = np.nan
            df.ix[df['h2oflow'] <= -50, 'h2oflow'] = np.nan
            df.ix[df['h2oflow'] >= 50, 'h2oflow'] = np.nan

            df = df.set_index(pd.DatetimeIndex(df['date_time']))

            if idx == 0:
                df1 = df
            elif idx==1:
                df2 = df
            elif idx == 2:
                df3 = df
            elif idx == 3:
                df4 = df

        new = pd.concat([df1[['date_time', 'h2oflow']], df2[['h2oflow']], df3[['h2oflow']], df4[['h2oflow']]], axis=1)

        new.columns = ['date_time', 'i1', 'i2', 'i3', 'i4']

        # new = pd.concat([df1[['date_time', 'co2flow']], df2[['co2flow']], df3[['co2flow']], df4[['co2flow']]], axis=1)
        # new.columns = ['date_time', 'i1', 'i2', 'i3', 'i4']


        new['mean'] = new.mean(axis=1)
        new['d1'] = new['i1'].rolling(window=6).mean()
        new['d2'] = new['i2'].rolling(window=6).mean()
        new['d3'] = new['i3'].rolling(window=6).mean()
        new['d4'] = new['i4'].rolling(window=6).mean()

        new['mx'] = new[['d1', 'd2', 'd3', 'd4']].max(axis=1)
        new['mn'] = new[['d1', 'd2', 'd3', 'd4']].min(axis=1)

        new = new.set_index(pd.DatetimeIndex(new['date_time']))

        fig, ax = plt.subplots(figsize=(16, 4))
        new = new.ix[:-1]
        ax.fill_between(new.index, new.mn, new.mx, color='royalblue', linewidth=1)

        ax.set_xlim(pd.Timestamp('2016-12-01'), pd.Timestamp('2017-01-01'))
        ax.set_ylim([-5, 5])
        ax.set_xlabel('')
        ax.set_ylabel('H$_2$O flow (mmol m$^{-2}$s$^{-1}$)')
        # ax.set_ylabel('CO$_2$ flow ($\mu$mol m$^{-2}$s$^{-1}$)')
        ax.set_axis_bgcolor('white')

        ax.xaxis.set_minor_locator(dates.DayLocator())

        ax.xaxis.grid(True, which="minor")
        ax.yaxis.set_major_locator(MaxNLocator(4))
        ax.yaxis.grid(True, which="major")

        plt.tight_layout()

        outputpath = 'C:\Users\herrmann\Documents\Herrmann Lars\Documentation\\figs\\'
        filename = 'h2oflux_' + month +'_max_deviation.pdf'
        plt.savefig(outputpath + filename, dpi=300)

        plt.show()

    def analyse_time_lags(self, DIR, FILE):
        """ 
        plots time lag data
        :return: 
        """

        path = DIR + '\\' + FILE

        # load time lag data
        df = pd.read_csv(path, sep=',', parse_dates=[['date', 'time']], names=['date', 'time', 'co2qc', 'h2oqc', 'co2tl', 'h2otl'],
                         usecols=[1, 2, 14, 16, 26, 31], header=2)

        df.replace(to_replace=-9999, value=np.nan, inplace=True)
        df.replace(to_replace='nan', value=np.nan, inplace=True)
        df.dropna()

        colors = {0: 'purple', 1: 'green', 2: 'royalblue', np.NAN: 'gray'}
        time_lags = []
        qcdates = []
        cols = []
        for idx, i in enumerate(df['co2qc']):
            try:
                if i != 3.0:
                    cols.append(colors[i])
                    time_lags.append(df['co2tl'].iloc[idx])
                    qcdates.append(df['date_time'].iloc[idx])

            except KeyError:
                print("")
                # cols.append('gray')

        fig, ax = plt.subplots(figsize=(7,4))
        dates = np.array(qcdates)
        dates = [pd.to_datetime(d) for d in dates]
        ax.scatter(dates, time_lags, c=cols, linewidths=0, marker='.')

        # #ax = df.plot(x='date_time', y='co2tl', figsize=(6, 4), color='royalblue', marker='.', markersize=0.5, linewidth=0, legend=False)
        ax.set_ylabel('Time lag (s)')
        ax.set_xlabel('Time')
        #
        plt.tight_layout()
        outputpath = 'C:\Users\herrmann\Documents\Herrmann Lars\Documentation\\figs\\'
        filename = 'qc_timelag_30min-average_v2.png'
        plt.savefig(outputpath + filename, dpi=300)
        plt.show()

    def compute_fourier_transform(self, direc, filename):
        """ computers fourier transform of time lag data to identify possible periodic patterns/oscillations"""

        path = direc + '\\' + filename

        df = pd.read_csv(path, sep=',', parse_dates=[['date', 'time']],
                         names=['date', 'time', 'co2qc', 'h2oqc', 'co2tl', 'h2otl'],
                         usecols=[1, 2, 14, 16, 26, 31], header=2)

        df.replace(to_replace=-9999, value=np.nan, inplace=True)
        df.dropna()

        time_lags = []
        qcdates = []
        cols = []
        for idx, i in enumerate(df['co2qc']):
            try:
                if i==2.0 or i == 1.0 or i==0.0:
                    time_lags.append(df['co2tl'].iloc[idx])
                    #qcdates.append(df['date_time'].iloc[idx])
            except KeyError:
                print("")
                # cols.append('gray')

        ar = np.array(time_lags)
        plt.figure(figsize=(8, 5))
        #
        # sample points
        N = len(ar)
        # sample spacing
        T = 60  # spaced in 1 minute intervals

        yf = fft(ar)


        # test code
        #x = np.linspace(0.0, N * T, N)
        #y = np.sin(50.0 * 2.0 * np.pi * x) + 0.5 * np.sin(80.0 * 2.0 * np.pi * x)
        #yf = fft(y)

        xf = np.linspace(0.0, 1.0 / (2.0 * T), N // 2)
        plt.loglog(xf * 1000, 2.0 / N * np.abs(yf[0:N // 2]), marker='.', linewidth=1, markersize=1.5, color='royalblue')
        plt.ylabel("Relative intensity (arb. units)")
        plt.xlabel("Frequency (mHz)")
        plt.tight_layout()
        outputpath = 'C:\Users\herrmann\Documents\Herrmann Lars\Documentation\\figs\\'
        filename = 'qc_timelag_fourier.png'
        plt.savefig(outputpath + filename, dpi=300)

        plt.show()

    def apply_color(self, x):
        colors = {0: 'red', 1: 'blue', 2: 'green', np.NAN: 'white'}
        try:
            return colors[x]
        except KeyError:
            return colors[0]

    def plot_quality_flag(self):
        """ 
        plots the quality flags associated with the data

        :return: 
        """
        direc = 'C:\Users\herrmann\Documents\Herrmann Lars\Licor\corrected\eddy_out\\time_lag_tests\\tl_offset\\b_adv_full'
        filename = 'eddypro_LH_full_output_2017-05-18T172908_adv.csv'
        path = direc + '\\' + filename

        # load time lag data
        df = pd.read_csv(path, sep=',', parse_dates=[['date', 'time']], names=['date', 'time', 'co2qc', 'h2oqc'],
                         usecols=[1, 2, 14, 16], header=2)
        df.plot(x='date_time', y='co2qc', marker='.', markersize=0.5, linewidth=0)

        plt.show()

    def plot_data_with_min_quality(self, direc, filename):
        """ 
        only plots the data with a certain quality - forward fills the missing data

        :return: 
        """
        path = direc + '\\' + filename

        # load time lag data
        df = pd.read_csv(path, sep=',', parse_dates=[['date', 'time']], names=['date', 'time', 'co2flux', 'co2qc'],
                         usecols=[1, 2, 13, 14], header=2)


        #print df
        df.ix[df['co2qc'] == 2, 'co2flux'] = np.nan
        df.fillna(axis=0, method='ffill', inplace=True)

        df.replace(to_replace=-9999, value=np.nan, inplace=True, limit=None, regex=False, method='pad')
        df.dropna(axis=0, how='any', inplace=True)

        df.plot(x='date_time', y='co2flux', marker='.', linewidth=0.5, markersize=1.3, color='royalblue')
        #df.loc[df['co2qc'] == 2, 'FirstName'] = "Matt"
        #df.plot(x='date_time', y='co2qc', marker='.', markersize=0.5, linewidth=0)
        plt.show()

    def plot_custom_integrate_eddy_data_entire_year_water(self):
        """ custom integration of water flux data """
        dir = 'C:\Users\herrmann\Documents\Herrmann Lars\Licor\corrected\eddy_out\complete_meteo_2'
        file = 'full.csv'
        path = dir + '\\' + file

        # h2o flux
        # ----------------------------
        df = pd.read_csv(path, sep=',', parse_dates=[['date', 'time']],
                         names=['date', 'time', 'co2flow', 'co2qc', 'h2oflow', 'h2oqc'],
                         usecols=[1, 2, 13, 14, 15, 16], header=2)

        # need to remove all values where qc is 2
        # first put all initial np.nan to -19999
        df.replace(to_replace=np.nan, value=-1000000, inplace=True)
        df.replace(to_replace=0, value=-1000000, inplace=True)

        # then put all quality flag 2 and quality flag -9999 data points to nan
        df.ix[df['h2oqc'] == 2.0, 'h2oflow'] = np.nan
        df.ix[df['h2oqc'] == -1000000, 'h2oflow'] = np.nan
        df.ix[df['co2qc'] == 2.0, 'co2flow'] = np.nan
        df.ix[df['co2qc'] == -1000000, 'co2flow'] = np.nan
        # now interpolate all nan values
        df.interpolate(method='linear', inplace=True)

        # now put all -19999 values back to np.nan (and all other outliers)
        # df.ix[df['co2flow'] == 0, 'shades'] = True
        df.ix[df['h2oflow'] <= -25, 'shades'] = True
        df.ix[df['h2oflow'] >= -25, 'shades'] = False
        df.ix[df['h2oflow'] <= -25, 'h2oflow'] = np.nan
        df.ix[df['h2oflow'] >= 25, 'h2oflow'] = np.nan
        df.ix[df['co2flow'] <= -25, 'co2flow'] = np.nan
        df.ix[df['co2flow'] >= 25, 'co2flow'] = np.nan

        #---------- Load file 2
        dir = 'C:\Users\herrmann\Documents\Herrmann Lars\Licor\corrected\eddy_out\_tl_analysis_impact\\complete'
        file = 'full.csv'
        path = dir + '\\' + file
        # h2o flux
        # ----------------------------
        df2 = pd.read_csv(path, sep=',', parse_dates=[['date', 'time']],
                         names=['date', 'time', 'co2flow', 'co2qc', 'h2oflow', 'h2oqc'],
                         usecols=[1, 2, 13, 14, 15, 16], header=2)

        # need to remove all values where qc is 2
        # first put all initial np.nan to -19999
        df2.replace(to_replace=np.nan, value=-1000000, inplace=True)
        df2.replace(to_replace=0, value=-1000000, inplace=True)

        # then put all quality flag 2 and quality flag -9999 data points to nan
        df2.ix[df['h2oqc'] == 2.0, 'h2oflow'] = np.nan
        df2.ix[df['co2qc'] == 2.0, 'co2flow'] = np.nan
        df2.ix[df['h2oqc'] == -1000000, 'h2oflow'] = np.nan
        df2.ix[df['co2qc'] == -1000000, 'co2flow'] = np.nan
        # now interpolate all nan values
        df2.interpolate(method='linear', inplace=True)

        # now put all -19999 values back to np.nan (and all other outliers)
        # df.ix[df['co2flow'] == 0, 'shades'] = True
        df2.ix[df2['h2oflow'] <= -25, 'shades'] = True
        df2.ix[df2['h2oflow'] >= -25, 'shades'] = False
        df2.ix[df2['h2oflow'] <= -25, 'h2oflow'] = np.nan
        df2.ix[df2['h2oflow'] >= 25, 'h2oflow'] = np.nan
        df2.ix[df2['co2flow'] <= -25, 'co2flow'] = np.nan
        df2.ix[df2['co2flow'] >= 25, 'co2flow'] = np.nan

        # calculate histogram
        #bounds = np.arange(-2, 2, 0.0125)
        bounds = np.arange(-5, 5, 1)
        # n, bins, rectangles = plt.hist(df['h2oflow'].dropna().copy().tolist(), bins=bounds, normed=True)

        # calculate the center of each bin (middle_bound)
        middle_bound = []
        for idx, i in enumerate(bounds):
            if idx + 1 < len(bounds):
                middle_bound.append((bounds[idx] + bounds[idx + 1]) / 2.0)

        # norm n to 1 (such that highest probability has value 1
        # n = [i / max(n) for i in n]

        # integrate the flow, pass the nvals and avlength array to let the integration algorithm know
        # how to handle the ranges with missing data
        # arrs = []
        # for idx, i in enumerate(middle_bound):
        #     print ("Process integration " + str(idx))
        #     arr = self.custom_integrate_water_min_max(df, 'custom', i, 0, 0, 0, 0, middle_bound)
        #     arrs.append(arr)
        #
        # # now fit n values with Gaussian
        # k = len(middle_bound)  # the number of data
        # mean = 0
        # sigma = 0
        # for i in range(len(n)):
        #     mean += (middle_bound[i] * n[i]) / k  # note this correction
        # for i in range(len(n)):
        #     sigma += (n[i] * (middle_bound[i] - mean) ** 2) / k  # note this correction

        # popt, pcov = curve_fit(self.gauss, middle_bound, n, p0=[1, mean, sigma])

        # now calculate most probable line
        mostprob = self.custom_integrate_water_min_max(df, 'null', 0, 0, 0, 0, 0,
                                                     middle_bound)

        mostprob2 = self.custom_integrate_water_min_max(df2, 'null', 0, 0, 0, 0, 0,
                                                       middle_bound)

        mostprob_co2 = self.custom_integrate_co2_min_max(df, 'null', 0, 0, 0, 0, 0,
                                                       middle_bound)

        mostprob2_co2 = self.custom_integrate_co2_min_max(df2, 'null', 0, 0, 0, 0, 0,
                                                        middle_bound)

        df = df.set_index(pd.DatetimeIndex(df['date_time']))
        df2 = df2.set_index(pd.DatetimeIndex(df2['date_time']))

        fig = plt.figure(figsize=(14/2.54, 8/2.54))
        ax1 = fig.add_subplot(111)
        ax2 = fig.add_subplot(111, sharex=ax1, frameon=False)
        # for idx, a in enumerate(arrs):
        #     if idx + 1 < len(arrs):
        #         ax.fill_between(df.index, y1=arrs[idx], y2=arrs[idx + 1], color='seagreen',
        #                         alpha=n[idx], linewidth=0)
        # id = n.index(max(n))
        # ax.plot(df.index, arrs[id+1], color='black', linewidth=2)
        ax1.plot(df.index, mostprob_co2, color='red', label='CO$_2$ with biomet')
        ax1.plot(df2.index, mostprob2_co2, color='seagreen', label='CO$_2$ w/o biomet')
        ax2.plot(df.index, mostprob, color='royalblue', label='H$_2$O with biomet')
        ax2.plot(df2.index, mostprob2, color='orange', label='H$_2$O w/o biomet')

        ax1.set_xlim(pd.Timestamp('2016-03-31'), pd.Timestamp('2017-04-01'))
        ax1.set_ylabel('Integrated CO$_2$ flux (mol m$^{-2}$)', labelpad=0)

        ax1.legend(loc='lower left', fontsize=9, frameon=False)
        plt.legend(loc='lower right',fontsize=9, frameon=False)


        ax2.yaxis.tick_right()
        ax2.yaxis.set_label_position("right")
        ax2.set_ylabel('Integrated H$_2$O flux (kmol m$^{-2}$)', rotation=-90, labelpad=10)
        ax2.fill_between(df.index, y1=100, y2=-100, where=df['shades'], color='purple', alpha=0.2, linewidth=0)
        ax2.set_ylim([-.5, 2])


        outputpath = 'C:\Users\herrmann\Documents\Herrmann Lars\Documentation\\figs\\'
        filename = 'integrated_flux_entire_year_biomet_tmp.pdf'
        plt.savefig(outputpath + filename, dpi=300)

        plt.show()

    def plot_custom_integrate_eddy_data_entire_year_co2(self):
        """ custom integration of co2 flux data

        note that this function and the above function for water are very redundant
        """
        dir = 'C:\Users\herrmann\Documents\Herrmann Lars\Licor\corrected\eddy_out\_tl_analysis_impact\\complete'
        file = 'full.csv'
        path = dir + '\\' + file

        # co2 concentration
        # ----------------------------
        df = pd.read_csv(path, sep=',', parse_dates=[['date', 'time']], names=['date', 'time', 'co2flow', 'co2qc', 'h2oflow'],
                         usecols=[1, 2, 13, 14, 15], header=2)

        # need to remove all values where qc is 2
        # first put all initial np.nan to -19999
        df.replace(to_replace=np.nan, value=-1000000, inplace=True)
        df.replace(to_replace=0, value=-1000000, inplace=True)

        # then put all quality flag 2 and quality flag -9999 data points to nan
        df.ix[df['co2qc'] == 2.0, 'co2flow'] = np.nan
        df.ix[df['co2qc'] == -1000000, 'co2flow'] = np.nan
        # now interpolate all nan values
        df.interpolate(method='linear', inplace=True)

        # now put all -19999 values back to np.nan (and all other outliers)
        #df.ix[df['co2flow'] == 0, 'shades'] = True
        df.ix[df['co2flow'] <= -25, 'shades'] = True
        df.ix[df['co2flow'] >= -25, 'shades'] = False
        df.ix[df['co2flow'] <= -25, 'co2flow'] = np.nan
        df.ix[df['co2flow'] >= 25, 'co2flow'] = np.nan


        # find the positions of the missing data and determine the length of each
        nanpos = []
        nanlength = []
        c = 0 # counter
        for idx, i in df.iterrows():
            if i['shades']:     # is not nan
                if c == 0:
                    nanpos.append(idx)
                c+=1
            else:   # is nan
                if c > 0:
                    nanlength.append(c)
                c=0
        nansort = nanlength[:]
        nansort.sort()

        # now build histogram for each of the length averaged co2 flows
        # i.e. for each missing data interval we average the co2 flows over the length of the interval
        # of missing data
        # then we build a histogram out of it which will tell us the probability to encounter a certain flow over
        # a certain time period

        #plt.figure(figsize=(3, 5))
        current = 0 # current av length
        avlength = []
        nvals = []
        bounds = np.arange(-10, 11, 0.1)
        print ("Create histogram")
        for i in nansort:
            dftmp = df.copy()
            if i != current:
                dftmp.dropna(inplace=True)
                dftmp['co2flow'].rolling(window=i).mean()
                n, bins, rectangles = plt.hist(dftmp['co2flow'].copy().tolist(), bins=bounds, normed=1)
                avlength.append(i)
                nvals.append(n)

        # calculate the center of each bin (middle_bound)
        arrs = []
        middle_bound = []
        for idx, i in enumerate(bounds):
            if idx+1 < len(bounds):
                middle_bound.append((bounds[idx] + bounds [idx+1])/2.0)

        # norm n to 1 (such that highest probability has value 1
        n = [i / max(n) for i in n]

        # now fit n values with Gaussian
        k = len(middle_bound)  # the number of data
        mean = 0
        sigma = 0
        for i in range(len(n)):
            mean += (middle_bound[i] * n[i]) / k  # note this correction
        for i in range(len(n)):
            sigma += (n[i] * (middle_bound[i] - mean) ** 2) / k  # note this correction

        # plt.figure(figsize=(3, 5))
        popt, pcov = curve_fit(self.gauss, middle_bound, n, p0=[1, mean, sigma])
        # n, bins, rectangles = plt.hist(df['co2flow'].dropna().copy().tolist(), bins=bounds, normed=1)
        # plt.ylim([0,1])
        # plt.plot(middle_bound, self.gauss(middle_bound,*popt),'--',label='fit', color='black')
        # outputpath = 'C:\Users\herrmann\Documents\Herrmann Lars\Documentation\\figs\\'
        # filename = 'integrated_co2flux_entire_year_histo.png'
        # plt.savefig(outputpath + filename, dpi=300)
        # plt.show()

        # now calculate most probable line
        mostprob = self.custom_integrate_co2_min_max(df, 'custom', 0, nvals, avlength, nanpos, nanlength,
                                                     middle_bound)

        # integrate the flow, pass the nvals and avlength array to let the integration algorithm know
        # how to handle the ranges with missing data
        for idx, i in enumerate(middle_bound):
            print ("Process integration " + str(idx))
            arr= self.custom_integrate_co2_min_max(df, 'custom2', i, nvals, avlength, nanpos, nanlength, middle_bound)
            arrs.append(arr)
        #arrnull = self.custom_integrate_min_max(df, 'null', 0)



        df = df.set_index(pd.DatetimeIndex(df['date_time']))
        fig, ax = plt.subplots(figsize=(8, 5))
        for idx, a in enumerate(arrs):
            if idx+1 < len(arrs):
                ax.fill_between(df.index, y1=arrs[idx], y2=arrs[idx+1], color='royalblue', alpha=self.gauss(middle_bound[idx],*popt), linewidth=0)

        ax.plot(df.index, mostprob, color='black', linewidth=2)

        ax.set_xlim(pd.Timestamp('2016-03-31'), pd.Timestamp('2017-04-01'))

        ax.set_ylabel('Integrated CO$_2$ flux (mol m$^{-2}$)')
        ax.fill_between(df.index, y1=100, y2=-100, where=df['shades'], color='purple', alpha=0.2, linewidth=0)
        ax.set_ylim([-20, 5])
        plt.grid()

        outputpath = 'C:\Users\herrmann\Documents\Herrmann Lars\Documentation\\figs\\'
        filename = 'integrated_co2flux_entire_year_tmp.png'
        plt.savefig(outputpath + filename, dpi=300)

        plt.show()

    def gauss(self, x, a, x0, sigma):
        """ for gaussian fitting"""
        return a * exp(-(x - x0) ** 2 / (2 * sigma ** 2))

    def custom_integrate_co2_min_max(self, df, method, ival, nvals, avlength, nanpos, nanlength, middle_bound):
        """
        print
        :return:
        """
        int_list = []

        ma = df['co2flow'].max()
        mi = df['co2flow'].min()

        last = 0 # default value
        for idx, c in df['co2flow'].iteritems():

            if np.isnan(c):         # what to do if there is a np.nan value?!
                if idx == 0:        # very first value
                    int_list.append(0)
                if method == 'min':
                    step = (df['date_time'].iloc[idx] - df['date_time'].iloc[idx - 1]).total_seconds()
                    int_list.append(int_list[idx - 1] + ma * step)
                elif method == 'max':
                    step = (df['date_time'].iloc[idx] - df['date_time'].iloc[idx - 1]).total_seconds()
                    int_list.append(int_list[idx - 1] + mi * step)
                elif method == 'null':
                    int_list.append(int_list[idx-1])    # append the previous value (assume 0 flux)
                elif method == 'custom2':
                    step = (df['date_time'].iloc[idx] - df['date_time'].iloc[idx - 1]).total_seconds()
                    int_list.append(int_list[idx - 1] + ival * step)
                elif method == 'custom':

                    step = (df['date_time'].iloc[idx] - df['date_time'].iloc[idx - 1]).total_seconds()

                    if idx in nanpos:
                        # length of nan interval where current index is
                        lnan = nanlength[nanpos.index(idx)]

                        # maximum probability
                        maxn = nvals[avlength.index(lnan)].max()
                        # boundary id where maximum prob occurs
                        temparr = nvals[avlength.index(lnan)].tolist()

                        #print temparr.tolist()
                        bdid = temparr.index(maxn)

                        int_list.append(int_list[idx - 1] + middle_bound[bdid] * step)
                        last = middle_bound[bdid]
                    else:
                        int_list.append(int_list[idx - 1] + last * step)

            elif not np.isnan(c): # simply integrate
                if idx == 0:        # very first value
                    int_list.append(c)
                else:               # afterwards add the co2 flow multiplied by the time in seconds
                    step = (df['date_time'].iloc[idx]-df['date_time'].iloc[idx-1]).total_seconds()
                    #print int_list
                    int_list.append(int_list[idx-1] + c * step)

        for idx, i in enumerate(int_list):
            int_list[idx] = i / 1e6  # convert to mol
        arr = np.array(int_list)
        return arr

    def custom_integrate_water_min_max(self, df, method, ival, nvals, avlength, nanpos, nanlength, middle_bound):
        """
        print
        :return:
        """
        int_list = []

        last = 0 # default value
        for idx, c in df['h2oflow'].iteritems():

            if np.isnan(c):         # what to do if there is a np.nan value?!
                if idx == 0:        # very first value
                    int_list.append(0)
                elif method == 'null':
                    int_list.append(int_list[idx-1])    # append the previous value (assume 0 flux)
                elif method == 'custom':
                    step = (df['date_time'].iloc[idx] - df['date_time'].iloc[idx - 1]).total_seconds()
                    int_list.append(int_list[idx - 1] + ival * step)
            elif not np.isnan(c): # simply integrate
                if idx == 0:        # very first value
                    int_list.append(c)
                else:               # afterwards add the co2 flow multiplied by the time in seconds
                    step = (df['date_time'].iloc[idx]-df['date_time'].iloc[idx-1]).total_seconds()
                    int_list.append(int_list[idx-1] + c * step)

        for idx, i in enumerate(int_list):
            int_list[idx] = i / 1e6  # convert to kmol
        arr = np.array(int_list)
        return arr

    def custom_integrate_eddy_data(self, dir, filename):
        """ custom integration of co2 flux data """

        path = dir + '\\' + filename

        # averaging interval
        min = 30
        # co2 concentration
        # ----------------------------
        df = pd.read_csv(path, sep=',', parse_dates=[['date', 'time']], names=['date', 'time', 'co2flow', 'co2qc', 'h2oflow', 'h2oqc'],
                         usecols=[1, 2, 13, 14, 15, 16], header=2)



        # need to remove all values where qc is 2
        df.ix[df['co2qc'] == 2.0, 'co2flow'] = np.nan
        df.ix[df['h2oqc'] == 2.0, 'h2oflow'] = np.nan

        df.replace(to_replace=-9999, value=np.nan, inplace=True)
        df.replace(to_replace=0.0, value=np.nan, inplace=True)


        df.ix[df['co2flow'] <= -25, 'co2flow'] = np.nan
        df.ix[df['co2flow'] >= 25, 'co2flow'] = np.nan
        df.ix[df['h2oflow'] <= -5, 'h2oflow'] = np.nan
        df.ix[df['h2oflow'] >= 10, 'h2oflow'] = np.nan

        df = df.set_index(pd.DatetimeIndex(df['date_time']))

        df.interpolate(method='linear', axis=0, inplace=True)
        #df.fillna(axis=0, method='bfill', inplace=True)
        df = df.dropna(axis=0, how='any').reset_index(drop=True)

        conc = df['co2flow'].values.tolist()
        timeline = df['date_time'].values.tolist()
        int_list = []
        int_list.append(0)
        last_val = 0
        import time
        start = time.time()
        for idx, i in enumerate(conc):
            if i != -9999:
                if not np.isnan(int_list[idx]):  # if current value is not nan, then add the next interval
                    last_val = int_list[idx] + min*60*i
                    int_list.append(last_val)
                else:
                    last_val = last_val + min*60*i
                    int_list.append(last_val)
            else:
                int_list.append(np.nan)
        print ("This took: " + str((time.time()-start)) + " s")
        del int_list[-1]
        for idx, i in enumerate(int_list):
            int_list[idx] = i/1e6   # convert to mol
        arr = np.array(int_list)
        df["integration"] = arr[df.index]

        return df




    def integrate_fluxes(self, dir, filename):
        """ integrates the data over time to see the net flux"""

        path = dir + '\\' + filename

        # co2 concentration
        # ----------------------------
        df = pd.read_csv(path, sep=',', parse_dates=[['date', 'time']], names=['date', 'time', 'co2flow', 'co2qc', 'h2oflow'],
                         usecols=[1, 2, 13, 14, 15], header=2)

        # need to remove all values where qc is 2
        df.ix[df['co2qc'] == 2.0, 'co2flow'] = np.nan
        # print df

        # replace the incorrect values (missing data) with 0 - so will not have any influence on integration but
        # maintain continuous flow
        #df.replace(to_replace=-9999, value=0, inplace=True, limit=None, regex=False, method='pad')
        df.replace(to_replace=-9999, value=np.nan, inplace=True, limit=None, regex=False, method='pad')

        # also replace values which are extremely high
        df.ix[df['co2flow'] > 50, 'co2flow'] = 0

        df_orig = df

        #df.fillna(axis=0, method='ffill', inplace=True)
        df = df.dropna(axis=0, how='any').reset_index(drop=True)

        #print df

        # set index to date_time
        df = df.set_index(pd.DatetimeIndex(df['date_time']))
        # integrate over co2flow column
        df_int = integrate.cumtrapz(df['co2flow'], df.index.astype(np.int64) / 10 ** 9, initial=0)
        # convert result from umol to mol/m2
        df['integral'] = df_int/1e6  ## !! convert to mol/m2

        # remove values where there was nan

        df.ix[df_orig['co2flow'] > 20, 'integral'] = np.nan

        ax = df.plot(x='date_time', y='integral', color='purple', kind='line', linewidth=1.0)

        ax.set_ylabel("CO$_2$ net flux (mol m$^{-2}$)")
        # set y label
        ax.set_xlabel("time")
        #ax.set_ylim([-100, 100])
        ax.get_yaxis().set_label_coords(-0.07, 0.5)
        ax.legend(['integrated CO2 flux'], loc='best', frameon=False, prop={'size': 7})

        plt.tight_layout()

        outputpath = 'C:\Users\herrmann\Documents\Herrmann Lars\Documentation\\figs\\'
        filename = 'year_integral.pdf'
        plt.savefig(outputpath + filename, dpi=300)

        plt.show()

    def plot_qc_histograms(self, DIR, FILE):
        """ 
        plots histograms of quality flags

        :return: 
        """
        path = DIR + '\\' + FILE

        fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(8, 5))

        # load time lag data
        df = pd.read_csv(path, sep=',', parse_dates=[['date', 'time']], names=['date', 'time', 'CO$_2$', 'H$_2$O'],
                         usecols=[1, 2, 14, 16], header=2)
        df.replace(to_replace=-9999, value=np.nan, inplace=True)
        df.dropna()
        #df.hist(df, column='co2qc')
        bounds = [-0.25, 0.25, 0.75, 1.25, 1.75, 2.25]#np.linspace(0,2,11)
        ax = df.hist(bins=bounds, ax=axes, color='seagreen', grid=False)
        axes[0].set_xlabel('Quality flag')
        axes[1].set_xlabel('Quality flag')
        axes[0].set_ylabel('Counts')
        axes[0].set_xticks([0, 1, 2])
        axes[1].set_xticks([0, 1, 2])
        #df.plot(x='date_time', y='co2qc', marker='.', markersize=0.5, linewidth=0)
        #plt.suptitle('test')
        plt.tight_layout()
        outputpath = 'C:\Users\herrmann\Documents\Herrmann Lars\Documentation\\figs\\'
        filename = 'qc_histogram_v2.pdf'
        plt.savefig(outputpath + filename, dpi=300)
        plt.show()

    def plot_fluxes(self, dir, filename):
        """ extractes co2 and h2o flux data from eddy pro files and plots it"""
        path = dir + '\\' + filename

        # change the font style and family
        font = {'family': 'sans-serif',
                'weight': 'light',
                'size': 9}
        matplotlib.rc('font', **font)
        # change the axes thickness
        matplotlib.rc('axes', linewidth=0.5)
        fig, axes = plt.subplots(nrows=2, ncols=1)

        # co2 concentration
        # ----------------------------
        df = pd.read_csv(path, sep=',', parse_dates=[['date', 'time']], names=['date', 'time', 'co2flow', 'h2oflow'], usecols=[1,2, 13, 15], header=2)
        ax = df.plot(x='date_time', y='co2flow', kind='line', figsize=(7, 4), grid=False,
                      color='mediumseagreen', linewidth=0.5, marker='.', markersize=1.8, ax=axes[0], sharex=axes[1])

        # average the dataframe
        # -- alternative to consider df=df.resample('1h').mean()
        df_av = df.set_index('date_time').rolling(window=60, center=False, win_type='boxcar').mean()
        df_av = df_av * 10
        df_av.plot(ax=ax, y='co2flow', kind='line', use_index=True, figsize=(7, 4), grid=False, color='red', linewidth=1.0, markersize=1.0)


        ax.legend(['timelag corrected', 'moving average (x10)'], loc='best', frameon=False, prop={'size': 7})

        # common x axis with plot underneath, therefore remove x label
        ax.set_ylabel("CO$_2$ ($\mathrm{\mu}$mol s$^{-1}$ m$^{-2}$)")
        # set y label
        ax.set_xlabel("")
        ax.set_ylim([-100, 100])
        ax.get_yaxis().set_label_coords(-0.07, 0.5)
        ax.set_xlim(pd.Timestamp('2016-07-09'), pd.Timestamp('2016-07-10'))

        # h2o concentration
        # ----------------------------
        ax = df.plot(x='date_time', y='h2oflow', ax=axes[1], legend=True, color='royalblue', linewidth=0.5, marker='.',
                        markersize=1.8)
        df_av.plot(ax=axes[1], y='h2oflow', kind='line', use_index=True, figsize=(7, 4), grid=False, color='red',
                   linewidth=1.0, markersize=1.0)
        ax.set_xlabel("Time")
        ax.legend(['timelag corrected', 'moving average (x10)'], loc='best', frameon=False, prop={'size': 7})
        ax.set_ylim([-5, 5])
        ax.set_ylabel("H$_2$O (nmol s$^{-1}$ m$^{-2}$)")
        ax.get_yaxis().set_label_coords(-0.07, 0.5)
        axes[1].set_xlim(pd.Timestamp('2016-07-09'), pd.Timestamp('2016-07-10'))

        plt.tight_layout()

        #vertical spacing for subplots
        fig.subplots_adjust(hspace=0.2)

        outputpath = 'C:\Users\herrmann\Documents\Herrmann Lars\Documentation\\figs\\'
        filename = 'raw_flow_summer_day.pdf'
        plt.savefig(outputpath + filename, dpi=300)

        plt.show()

    def compare_eddy_data(self, dir):
        """ extractes co2 and h2o flux data from eddy pro files and plots it"""
        path1 = dir + '\\processing_comparison\\time_lags\\advanced_mode'
        path2 = dir + '\\processing_comparison\\time_lags\\express_mode'
        file1 = 'eddypro_LH_essentials_2017-05-10T095215_adv.csv'
        file2 = 'eddypro_LH_essentials_2017-05-10T093641_exp.csv'
        path1 = path1 + '\\' + file1
        path2 = path2 + '\\' + file2
        df1 = pd.read_csv(path1, sep=',', parse_dates=[['date', 'time']], names=['date', 'time', 'co2flow', 'h2oflow'],
                          usecols=[1,2, 12, 14], header=0)
        df2 = pd.read_csv(path2, sep=',', parse_dates=[['date', 'time']], names=['date', 'time', 'co2flow', 'h2oflow'],
                          usecols=[1, 2, 12, 14], header=0)

        # change the font style and family
        font = {'family': 'sans-serif',
                'weight': 'light',
                'size': 9}
        matplotlib.rc('font', **font)
        # change the axes thickness
        matplotlib.rc('axes', linewidth=0.5)

        fig, axes = plt.subplots(nrows=2, ncols=1)


        # SUBPLOT 1: CO2 FLUX
        ax = df1.plot(x='date_time', y='co2flow', kind='line', figsize=(7,4), grid=False, ax=axes[0], sharex=axes[1],
                        color = 'royalblue', linewidth = 0.75, marker = '.', markersize = 1.0)
        df2.plot(ax=ax, x='date_time', y='co2flow', kind='line', figsize=(7,4), grid=False,
                 color='chocolate', linewidth=0.75, marker='.', markersize=1.0)
        # specify legend and set font size
        ax.legend(['timelag corrected', 'raw'], loc='best', frameon=False, prop={'size':7})
        # common x axis with plot underneath, therefore remove x label
        ax.set_xlabel("")
        # set y label
        ax.set_ylabel("CO2 flux (??)")
        ax.set_ylim([-40, 40])
        ax.get_yaxis().set_label_coords(-0.07, 0.5)


        # SUBPLOT 2 -  ABSOLUTE DIFFERENCE
        dfrel = (-df1['co2flow'] + df2['co2flow'])
        dfrel = pd.concat([df1['date_time'], dfrel], axis=1)
        dfrel.columns = ['date_time', 'difference']
        dfrel['date_time'] = pd.to_datetime(dfrel['date_time'].astype(str), format='%Y-%m-%d %H:%M:%S')
        dfrel = dfrel.set_index(pd.DatetimeIndex(dfrel['date_time']))
        df2['date_time'] = pd.to_datetime(df2['date_time'].astype(str), format='%Y-%m-%d %H:%M:%S')
        df2 = df2.set_index(pd.DatetimeIndex(df2['date_time']))

        ax = dfrel.plot(x='date_time', y='difference', ax=axes[1], legend=False, color='k', linewidth=0.75, marker='.',
                        markersize=1.0)
        ax.set_xlabel("Time")
        ax.set_ylabel("CO2 flux difference")
        ax.get_yaxis().set_label_coords(-0.07, 0.5)
        axes[1].set_xlim(pd.Timestamp('2016-04-01'), pd.Timestamp('2016-04-02'))



        # set overall figure title
        #plt.suptitle('Influence of timelag correction')

        # adjust layout
        plt.tight_layout()
        # offset for title
        fig.subplots_adjust(top=0.9)
        # vertical spacing between subplots
        fig.subplots_adjust(hspace=0.2)

        outputpath='C:\Users\herrmann\Documents\Herrmann Lars\Documentation\\figs\\'
        filename = 'timelagcorrection.pdf'
        plt.savefig(outputpath + filename, dpi=300)

    def compare_sampling_rate_histogram(self):
        month = 'apr16'
        base = 'C:\Users\herrmann\Documents\Herrmann Lars\Licor\corrected\eddy_out\_tl_analysis_impact\\sampling\\'
        file = 'full.csv'
        #path0 = base + 'apr\\r50\\' + file
        path1 = base + 'nov\\r100\\' + file
        path2 = base + 'nov\\r200\\' + file
        path3 = base + 'nov\\r500\\' + file
        path4 = base + 'nov\\r1000\\' + file
        path5 = base + 'nov\\r5000\\' + file
        path6 = base + 'nov\\r10000\\' + file
        paths = [path1, path2, path3, path4, path5, path6]
        n = len(paths)

        hists_co2 = []
        hists_water = []

        samples_co2 = 11
        samples_water = 11
        range_co2 = 40
        range_water = 4
        bound_co2 = np.linspace(-20, 20, samples_co2)
        bound_water = np.linspace(-2, 2, samples_water)
        for p in paths:
            print("\tLoad data \r\n\t" + p)
            df = pd.read_csv(p, sep=',', parse_dates=[['date', 'time']],
                              names=['date', 'time', 'co2flow', 'co2qc', 'h2oflow', 'h2oqc'],
                              usecols=[1, 2, 13, 14, 15, 16], header=2)


            # replace all values with missing data with the error flag code -9999
            df.replace(to_replace=np.nan, value=-9999, inplace=True)
            df.replace(to_replace=0, value=-9999, inplace=True)
            # replace data where quality control flag is 2 (bad) with np.nan
            df.ix[df['co2qc'] == 2.0, 'co2flow'] = np.nan
            df.ix[df['h2oqc'] == 2.0, 'h2oflow'] = np.nan
            # now interpolate all nan values
            df.interpolate(method='linear', inplace=True)

            # df.fillna(axis=0, method='ffill', inplace=True)
            df.ix[df['co2flow'] <= -100, 'shades'] = True
            df.ix[df['co2flow'] >= -100, 'shades'] = False
            df.ix[df['co2flow'] <= -50, 'co2flow'] = np.nan
            df.ix[df['co2flow'] >= 50, 'co2flow'] = np.nan
            df.ix[df['h2oflow'] <= -5, 'h2oflow'] = np.nan
            df.ix[df['h2oflow'] >= 5, 'h2oflow'] = np.nan

            # now bring back the -9999 values to np.nan
            df.replace(to_replace=-9999, value=np.nan, inplace=True)
            df.dropna(inplace=True)

            print("\tCreate histogram \r\n\t" + p)

            weights_co2 = np.ones_like(df['co2flow']) / len(df['co2flow'])
            hist_co2, bin_edges = np.histogram(df['co2flow'], bins=bound_co2, weights=weights_co2)
            weights_w = np.ones_like(df['h2oflow']) / len(df['h2oflow'])
            hist_water, bin_edges = np.histogram(df['h2oflow'], bins=bound_water, weights=weights_w)

            hists_co2.append(hist_co2)
            hists_water.append(hist_water)

        middle_bound_co2 = []
        for idx, i in enumerate(bound_co2):
            if idx + 1 < len(bound_co2):
                middle_bound_co2.append((bound_co2[idx] + bound_co2[idx + 1]) / 2.0)

        middle_bound_water = []
        for idx, i in enumerate(bound_water):
            if idx + 1 < len(bound_water):
                middle_bound_water.append((bound_water[idx] + bound_water[idx + 1]) / 2.0)

        fig, ax = plt.subplots(nrows=2, ncols=1, figsize=(8,5))
        middle_bounds_co2 = []
        middle_bounds_water = []

        width_c = (range_co2 / (samples_co2 - 1))
        gap_c = 0.1 * width_c
        barwidth_c = (width_c - gap_c) / n
        offset_c = -(width_c - gap_c) / 2 + barwidth_c / 2
        for j in range(n):
            middle_bound_tmp_co2 = []
            for i in range(samples_co2-1):
                middle_bound_tmp_co2.append(middle_bound_co2[i] + offset_c + j*barwidth_c)
            middle_bounds_co2.append(middle_bound_tmp_co2)

        width_w = (float(range_water)/(samples_water-1))
        print width_w
        gap_w = 0.1*width_w
        barwidth_w = (float(width_w)-float(gap_w))/float(n)
        offset_w = -(width_w-gap_w)/2+barwidth_w/2
        for j in range(n):
            middle_bound_tmp_water = []
            for i in range(samples_water-1):
                middle_bound_tmp_water.append(middle_bound_water[i] + offset_w + j * barwidth_w)
            middle_bounds_water.append(middle_bound_tmp_water)

        hrz = [10, 5, 2, 1, 0.2, 0.1]

        my_colors = list(islice(cycle(['steelblue', 'royalblue', 'cornsilk', 'y', 'olive', 'green']), None, n+1))
        #colors = []
        for i in range(n):
            ax[0].bar(middle_bounds_co2[i], hists_co2[i]*100, width=barwidth_c, color=my_colors[i], align='center', edgecolor='black', linewidth=0.5,
                      label=str(hrz[i]) + ' Hz')
            ax[0].set_xticks(middle_bound_co2)
            ax[0].set_yticks([0, 20, 40])
            ax[0].set_xlabel('mean CO$_2$ flux ($\mu$mol m$^{-2}$s$^{-1}$)')
            ax[0].set_ylabel('relative frequency (%)')
            ax[0].set_yscale("log")
            ax[0].set_ylim([0.01, 100])
            ax[0].legend(fontsize=6, frameon=False, loc='upper right')
            ax[0].set_title('November 2016 - CO$_2$ fluxes', y=0.84, x=0.16, fontsize=10)


            ax[1].bar(middle_bounds_water[i], hists_water[i]*100, width=barwidth_w, color=my_colors[i], align='center', edgecolor='black',
                     linewidth=0.5, label=str(hrz[i]) + ' Hz')
            ax[1].set_xticks(middle_bound_water)
            ax[1].set_yticks([0, 20, 40])
            ax[1].set_xlabel('mean H$_2$O flux (mmol m$^{-2}$s$^{-1}$)')
            ax[1].set_ylabel('relative frequency (%)')
            ax[1].set_yscale("log")
            ax[1].set_ylim([0.01, 100])
            ax[1].legend(fontsize=6, frameon=False,loc='upper left')
            ax[1].set_title('November 2016 - H$_2$O fluxes', y=0.84, x=0.84, fontsize=10)

        mx = []
        mn = []
        for i in range(samples_co2-1):
            tmp=[]
            for j in range(4):
                h = hists_co2[j]
                tmp.append(h[i])
            print tmp
            mx.append(max(tmp))
            mn.append(min(tmp))
        diff = []
        for idx, i in enumerate(mx):
            diff.append(100*float(i)/mn[idx])
        print("Differences " + str(diff))



        plt.tight_layout()
        outputpath = 'C:\Users\herrmann\Documents\Herrmann Lars\Documentation\\figs\\'
        filename = 'sampling_rate_nov2016.pdf'
        plt.savefig(outputpath + filename, dpi=300)
        plt.show()

    def compare_sampling_rate_flow_difference(self):
        month = 'jul'
        base = 'C:\Users\herrmann\Documents\Herrmann Lars\Licor\corrected\eddy_out\_tl_analysis_impact\\sampling\\'
        file = 'full.csv'
        path0 = base + month + '\\r50\\' + file
        path1 = base + month + '\\r100\\' + file
        path2 = base + month + '\\r200\\' + file
        path3 = base + month + '\\r500\\' + file
        path4 = base + month + '\\r1000\\' + file
        path5 = base + month + '\\r5000\\' + file
        path6 = base + month + '\\r10000\\' + file
        paths = [path0, path1, path2, path3, path4, path5, path6]

        flows = []
        for idx, p in enumerate(paths):
            print("\tLoad data \r\n\t" + p)
            df = pd.read_csv(p, sep=',', parse_dates=[['date', 'time']],
                              names=['date', 'time', 'co2flow', 'co2qc', 'h2oflow', 'h2oqc'],
                              usecols=[1, 2, 13, 14, 15, 16], header=2)


            # replace all values with missing data with the error flag code -9999
            df.replace(to_replace=np.nan, value=-9999, inplace=True)
            df.replace(to_replace=0, value=-9999, inplace=True)
            # replace data where quality control flag is 2 (bad) with np.nan
            df.ix[df['co2qc'] == 2.0, 'co2flow'] = np.nan
            df.ix[df['h2oqc'] == 2.0, 'h2oflow'] = np.nan
            # now interpolate all nan values
            #df.interpolate(method='linear', inplace=True)

            # df.fillna(axis=0, method='ffill', inplace=True)
            df.ix[df['co2flow'] <= -100, 'shades'] = True
            df.ix[df['co2flow'] >= -100, 'shades'] = False
            df.ix[df['co2flow'] <= -100, 'co2flow'] = np.nan
            df.ix[df['co2flow'] >= 100, 'co2flow'] = np.nan
            df.ix[df['h2oflow'] <= -5, 'h2oflow'] = np.nan
            df.ix[df['h2oflow'] >= 5, 'h2oflow'] = np.nan


            # now bring back the -9999 values to np.nan
            df.replace(to_replace=-9999, value=np.nan, inplace=True)
            #df.dropna(inplace=True)

            df = df.set_index(pd.DatetimeIndex(df['date_time']))

            if idx == 0:
                dftmp = df.copy()
                del dftmp['co2qc']
                del dftmp['h2oflow']
                del dftmp['h2oqc']
            else:
                colname = 'co2flow' + str(idx)
                dftmp[colname] = df['co2flow']

        for i in range(len(paths)-1):
            colname = 'co2flow' + str(i+1)
            colnamediff = 'co2flowdiff' + str(i + 1)
            dftmp[colnamediff] = 100*(dftmp[colname]-dftmp['co2flow'])/dftmp['co2flow']
            dftmp.ix[dftmp[colnamediff] <= -50, colnamediff] = np.nan
            dftmp.ix[dftmp[colnamediff] >= 400, colnamediff] = np.nan

        fig, axs = plt.subplots(1, 1, figsize=(10, 4), facecolor='w', edgecolor='k', sharey=True, sharex=True)
        fig.subplots_adjust(hspace=0, wspace=0)

        dftmp.plot(x='date_time', y='co2flowdiff1', marker='.', linewidth=0, markersize=4, ax=axs, label='10 Hz',
                   color='royalblue')
        dftmp.plot(x='date_time', y='co2flowdiff4', marker='.', linewidth=0, markersize=4, ax=axs, label='1 Hz',
                   color='darkgoldenrod')
        dftmp.plot(x='date_time', y='co2flowdiff6', marker='.', linewidth=0, markersize=4, ax=axs, label='0.1 Hz',
                   color='seagreen')
        axs.legend(frameon=True)
        axs.set_ylabel('relative deviation (%)')
        plt.xlim(pd.Timestamp('2016-07-13'), pd.Timestamp('2016-07-19'))
        plt.ylim([-20,20])
        plt.tight_layout()
        outputpath = 'C:\Users\herrmann\Documents\Herrmann Lars\Documentation\\figs\\'
        filename = 'sampling_rate_jul2016_flow_diff.pdf'
        plt.savefig(outputpath + filename, dpi=300)
        plt.show()

    def plot_fluxes_sampling_rate(self):
        month = 'apr'
        base = 'C:\Users\herrmann\Documents\Herrmann Lars\Licor\corrected\eddy_out\_tl_analysis_impact\\sampling\\instant\\'
        file = 'full.csv'
        path0 = base + month + '\\r50\\' + file
        path1 = base + month + '\\r100\\' + file
        path2 = base + month + '\\r200\\' + file
        path3 = base + month + '\\r500\\' + file
        path4 = base + month + '\\r1000\\' + file
        path5 = base + month + '\\r5000\\' + file
        path6 = base + month + '\\r10000\\' + file
        paths = [path0, path1, path2, path3, path4, path5, path6]
        n = len(paths)

        fig, axs = plt.subplots(1, 1, figsize=(10, 7), facecolor='w', edgecolor='k')
        fig.subplots_adjust(hspace=0, wspace=0)

        hrz = ['20 Hz', '10 Hz', '5 Hz', '2 Hz', '1 Hz', '0.2 Hz', '0.1 Hz']
        my_colors = list(islice(cycle(['steelblue', 'royalblue', 'darkred', 'y', 'olive', 'green']), None, n + 3))
        for idx, p in enumerate(paths):
            if idx >-1:
                print("\tLoad data \r\n\t" + p)
                df = pd.read_csv(p, sep=',', parse_dates=[['date', 'time']],
                                  names=['date', 'time', 'co2flow', 'co2qc', 'h2oflow', 'h2oqc'],
                                  usecols=[1, 2, 13, 14, 15, 16], header=2)

                # replace all values with missing data with the error flag code -9999
                df.replace(to_replace=np.nan, value=-9999, inplace=True)
                df.replace(to_replace=0, value=-9999, inplace=True)
                # replace data where quality control flag is 2 (bad) with np.nan
                df.ix[df['co2qc'] == 2.0, 'co2flow'] = np.nan
                df.ix[df['h2oqc'] == 2.0, 'h2oflow'] = np.nan


                # now interpolate all nan values
                df.interpolate(method='linear', inplace=True)

                # df.fillna(axis=0, method='ffill', inplace=True)
                df.ix[df['co2flow'] <= -100, 'shades'] = True
                df.ix[df['co2flow'] >= -100, 'shades'] = False
                df.ix[df['co2flow'] <= -500, 'co2flow'] = np.nan
                df.ix[df['co2flow'] >= 500, 'co2flow'] = np.nan
                df.ix[df['h2oflow'] <= -5, 'h2oflow'] = np.nan
                df.ix[df['h2oflow'] >= 5, 'h2oflow'] = np.nan


                # now bring back the -9999 values to np.nan
                df.replace(to_replace=-9999, value=np.nan, inplace=True)
                #df.dropna(inplace=True)

                df = df.set_index(pd.DatetimeIndex(df['date_time']))

                df['co2flow'] = df['co2flow'] + 150-(idx-1)*50

                df.plot(x='date_time', y='co2flow', linewidth=1, alpha=1, ax=axs,
                           color=my_colors[idx], label=hrz[idx])

                print df['co2flow'].var()

        axs.fill_between(df.index, y1=-1000, y2=1000, where=df['shades'], color='purple', alpha=0.2, linewidth=0)
        axs.legend(frameon=True, loc='upper right')
        axs.set_ylim([-150, 200])
        axs.set_ylabel('CO$_2$ flux ($\mu$mol m$^{-2}$s$^{-1}$)')
        #axs.xaxis.set_minor_locator(dates.DayLocator())
        axs.xaxis.grid(True, which="minor", linestyle='--', alpha=0.8)
        plt.xlim(pd.Timestamp('2016-04-01'), pd.Timestamp('2016-05-01'))
        axs.yaxis.set_major_locator(MaxNLocator(4))
        axs.yaxis.grid(True, which="major",  linestyle='--', alpha=0.8)
        plt.tight_layout()
        outputpath = 'C:\Users\herrmann\Documents\Herrmann Lars\Documentation\\figs\\'
        filename = 'sampling_rate_instant_apr2016_flux_month_tmp.pdf'
        plt.savefig(outputpath + filename, dpi=300)
        plt.show()

    def plot_daily_variance_sampling_rate(self):
        """ plots the daily or hourly variance due to different sampling rates
        but right now this does not show a clear intcrease in variance on a daily/hourly basis
        """
        month = 'apr'
        base = 'C:\Users\herrmann\Documents\Herrmann Lars\Licor\corrected\eddy_out\_tl_analysis_impact\\sampling\\instant\\'
        file = 'full.csv'
        path0 = base + month + '\\r50\\' + file
        path1 = base + month + '\\r100\\' + file
        path2 = base + month + '\\r200\\' + file
        path3 = base + month + '\\r500\\' + file
        path4 = base + month + '\\r1000\\' + file
        path5 = base + month + '\\r5000\\' + file
        path6 = base + month + '\\r10000\\' + file
        paths = [path0, path1, path2, path3, path4, path5, path6]
        n = len(paths)

        fig, axs = plt.subplots(1, 1, figsize=(6, 4), facecolor='w', edgecolor='k')
        fig.subplots_adjust(hspace=0, wspace=0)

        hrz = ['20 Hz', '10 Hz', '5 Hz', '2 Hz', '1 Hz', '0.2 Hz', '0.1 Hz']
        my_colors = list(islice(cycle(['steelblue', 'royalblue', 'darkred', 'y', 'olive', 'green']), None, n + 3))
        var_res=[]
        for idx, p in enumerate(paths):
            if idx >-1:
                print("\tLoad data \r\n\t" + p)
                df = pd.read_csv(p, sep=',', parse_dates=[['date', 'time']],
                                  names=['date', 'time', 'co2flow', 'co2qc', 'h2oflow', 'h2oqc'],
                                  usecols=[1, 2, 13, 14, 15, 16], header=2)

                # replace all values with missing data with the error flag code -9999
                df.replace(to_replace=np.nan, value=-9999, inplace=True)
                df.replace(to_replace=0, value=-9999, inplace=True)
                # replace data where quality control flag is 2 (bad) with np.nan
                df.ix[df['co2qc'] == 2.0, 'co2flow'] = np.nan
                df.ix[df['h2oqc'] == 2.0, 'h2oflow'] = np.nan


                # now interpolate all nan values
                df.interpolate(method='linear', inplace=True)

                # df.fillna(axis=0, method='ffill', inplace=True)
                df.ix[df['co2flow'] <= -100, 'shades'] = True
                df.ix[df['co2flow'] >= -100, 'shades'] = False
                df.ix[df['co2flow'] <= -500, 'co2flow'] = np.nan
                df.ix[df['co2flow'] >= 500, 'co2flow'] = np.nan
                df.ix[df['h2oflow'] <= -5, 'h2oflow'] = np.nan
                df.ix[df['h2oflow'] >= 5, 'h2oflow'] = np.nan


                # now bring back the -9999 values to np.nan
                df.replace(to_replace=-9999, value=np.nan, inplace=True)
                #df.dropna(inplace=True)

                df = df.set_index(pd.DatetimeIndex(df['date_time']))
                dftmp = df.copy()
                day = dftmp.index.day
                hour = dftmp.index.hour
                var_tmp=[]
                for i in range(24):
                    selector = (i == hour)
                    df2 = dftmp[selector].copy()
                    var_tmp.append(df2['co2flow'].var())
                var_res.append(var_tmp)


        b=var_res[0]
        var_res_2 = []
        print var_res[0]
        print var_res[1]
        for i in range(len(var_res)):
            var_res_tmp =[]
            for k in range(len(var_res[i])):
                var_res_tmp.append(var_res[i][k]/var_res[0][k])
            var_res_2.append(var_res_tmp)

        for i in var_res_2:
            axs.plot(i)

        plt.tight_layout()
        outputpath = 'C:\Users\herrmann\Documents\Herrmann Lars\Documentation\\figs\\'
        filename = 'sampling_instant_var.pdf'
        plt.savefig(outputpath + filename, dpi=300)
        plt.show()


    def plot_integrated_instant_sampling_rate(self):
        """ integrates the co2 flux of the instant sampled datasets to see the estimate of sampling rate """
        month = 'apr'
        base = 'C:\Users\herrmann\Documents\Herrmann Lars\Licor\corrected\eddy_out\_tl_analysis_impact\\sampling\\instant\\'
        file = 'full.csv'
        path0 = base + month + '\\r50\\'
        path1 = base + month + '\\r100\\'
        path2 = base + month + '\\r200\\'
        path3 = base + month + '\\r500\\'
        path4 = base + month + '\\r1000\\'
        path5 = base + month + '\\r5000\\'
        path6 = base + month + '\\r10000\\'
        paths = [path0, path1, path2, path3, path4, path5, path6]
        n = len(paths)

        fig, axs = plt.subplots(1, 1, figsize=(8, 5), facecolor='w', edgecolor='k')
        fig.subplots_adjust(hspace=0, wspace=0)

        hrz = ['20 Hz', '10 Hz', '5 Hz', '2 Hz', '1 Hz', '0.2 Hz', '0.1 Hz']
        my_colors = list(islice(cycle(['steelblue', 'royalblue', 'darkred', 'y', 'olive', 'green', 'seagreen']), None, n + 3))

        df = self.custom_integrate_eddy_data(path1, file)
        df['i2'] = self.custom_integrate_eddy_data(path2, file)['integration']
        df['i3'] = self.custom_integrate_eddy_data(path3, file)['integration']
        df['i4'] = self.custom_integrate_eddy_data(path4, file)['integration']
        df['i5'] = self.custom_integrate_eddy_data(path5, file)['integration']
        df['i6'] = self.custom_integrate_eddy_data(path6, file)['integration']
        df['i1'] = df['integration']

        for i in range(7):
            if i>0:
                lab = 'i' + str(i)
                df.plot(x='date_time', y=lab, linewidth=1, alpha=1, ax=axs,
                                   color=my_colors[i], label=hrz[i])



        axs.legend(frameon=True, loc='upper right')
        axs.set_ylabel('CO$_2$ flux (mol m$^{-2}$s$^{-1}$)')
        #axs.xaxis.set_minor_locator(dates.DayLocator())
        axs.xaxis.grid(True, which="minor", linestyle='--', alpha=0.8)
        plt.xlim(pd.Timestamp('2016-04-01'), pd.Timestamp('2016-05-01'))
        axs.yaxis.set_major_locator(MaxNLocator(4))
        axs.yaxis.grid(True, which="major",  linestyle='--', alpha=0.8)
        plt.tight_layout()
        outputpath = 'C:\Users\herrmann\Documents\Herrmann Lars\Documentation\\figs\\'
        filename = 'intgrated_instant_sampling_rate.pdf'
        plt.savefig(outputpath + filename, dpi=300)
        plt.show()


    def add_months(self, sourcedate, months):
        """ custom helper function to add one month to given date 
        @:param sourcedate      the date for which to increase the month
        @:param months          number of months to increase
        @:return                sourcedate + 1 month"""
        month = sourcedate.month - 1 + months
        year = int(sourcedate.year + month / 12)
        month = month % 12 + 1
        day = min(sourcedate.day, calendar.monthrange(year, month)[1])
        return datetime.date(year, month, day)


    def plot_footprints(self, dir, filename):
        """ plots the estimated footprints as a function of time """
        path = dir + '\\' + filename



        # co2 concentration
        # ----------------------------
        df = pd.read_csv(path, sep=',', parse_dates=[['date', 'time']],
                         names=['date', 'time', 'co2flow', 'co2qc', 'h2oflow', 'h2oqc', 'fppk', 'fp90'],
                         usecols=[1, 2, 13, 14, 15, 16, 66, 72], header=2)

        # replace all values with missing data with the error flag code -9999
        df.replace(to_replace=np.nan, value=-9999, inplace=True)
        df.replace(to_replace=0, value=-9999, inplace=True)

        # now interpolate all nan values
        #df.interpolate(method='linear', inplace=True)

        # df.fillna(axis=0, method='ffill', inplace=True)
        df.ix[df['fp90'] <= -100, 'shades'] = True
        df.ix[df['fp90'] >= -100, 'shades'] = False
        df.ix[df['fp90'] <= 0, 'fp90'] = np.nan
        df.ix[df['fp90'] >= 40000, 'fp90'] = np.nan


        #df.interpolate(method='linear', inplace=True)

        # now bring back the -9999 values to np.nan
        #df.replace(to_replace=-9999, value=np.nan, inplace=True)
        # df.dropna(inplace=True)

        df = df.set_index(pd.DatetimeIndex(df['date_time']))
        print df.dtypes
        fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(14, 5))

        df['rolling'] = df['fppk'].astype(float)
        df.ix[df['rolling'] <= 0, 'rolling'] = np.nan
        df.ix[df['rolling'] >= 20000, 'rolling'] = np.nan

        #convert to km
        df['fp90'] = df['fp90'] / 1000
        df['rolling'] = df['rolling'] / 1000
        df['mean_fp90'] = df['fp90'].mean()
        df['std_fp90t'] = df['mean_fp90']+df['fp90'].std()
        df['std_fp90b'] = df['mean_fp90']-df['fp90'].std()
        df['mean_fppk'] = df['rolling'].mean()
        df['std_fppkt'] = df['mean_fppk'] + df['rolling'].std()
        df['std_fppkb'] = df['mean_fppk'] - df['rolling'].std()



        #df['fppk'] = df['fppk'].astype(float)
        df.plot(x='date_time', y='fp90', ax=ax, marker='.', markersize=1, linewidth=0, color='royalblue', label='' )
        #df.plot(x='date_time', y='mean_fp90', ax=ax, marker='.', markersize=0, linewidth=1, color='royalblue')
        ax.fill_between(df.index, y1=df['std_fp90b'], y2=df['std_fp90t'], color='royalblue', alpha=0.2, label='90%')

        df.plot(x='date_time', y='rolling', ax=ax, marker='.', markersize=1, linewidth=0, color='seagreen', label='')
        #df.plot(x='date_time', y='mean_fppk', ax=ax, marker='.', markersize=0, linewidth=1, color='seagreen')
        ax.fill_between(df.index, y1=df['std_fppkb'], y2=df['std_fppkt'], color='seagreen', alpha=0.2, label='peak')

        #ax.legend(['timelag corrected', 'moving average (x10)'], loc='best', frameon=False, prop={'size': 7})
        ax.set_ylim([0, 15])
        ax.set_ylabel("footprint (km)")
        ax.legend(loc='upper right', frameon=True)
        plt.tight_layout()

        outputpath = 'C:\Users\herrmann\Documents\Herrmann Lars\Documentation\\figs\\'
        filename = 'footprints.png'
        plt.savefig(outputpath + filename, dpi=300)

        plt.show()

    def plot_correlation_by_hour(self):
        """ This plots the correlation coefficient bewteen co2 and h2o fluxes for each months, resolved into the hours of the day
        """

        DIR1 = 'C:\Users\herrmann\Documents\Herrmann Lars\Licor\corrected\eddy_out\_tl_analysis_impact\\complete'
        FILE1 = 'full.csv'

        path = DIR1 + '\\' + FILE1
        df = pd.read_csv(path, sep=',', parse_dates=[['date', 'time']],
                         names=['date', 'time', 'co2flow', 'co2qc', 'h2oflow', 'h2oqc'],
                         usecols=[1, 2, 13, 14, 15, 16], header=2)

        # replace all values with missing data with the error flag code -9999
        df.replace(to_replace=np.nan, value=-9999, inplace=True)
        df.replace(to_replace=0, value=-9999, inplace=True)
        # replace data where quality control flag is 2 (bad) with np.nan
        df['orig'] = df['co2flow']
        df['bad'] = df.ix[df['co2qc'] == 2.0, 'co2flow']
        df.ix[df['co2qc'] == 2.0, 'co2flow'] = np.nan
        df.ix[df['co2qc'] == 2.0, 'h2oflow'] = np.nan
        df.ix[df['h2oqc'] == 2.0, 'h2oflow'] = np.nan
        df.ix[df['h2oqc'] == 2.0, 'co2flow'] = np.nan

        # now interpolate all nan values
        df.interpolate(method='linear', inplace=True)
        df.ix[df['co2flow'] <= -50, 'co2flow'] = np.nan
        df.ix[df['co2flow'] >= 50, 'co2flow'] = np.nan
        df.ix[df['h2oflow'] <= -50, 'h2oflow'] = np.nan
        df.ix[df['h2oflow'] >= 50, 'h2oflow'] = np.nan
        df = df.set_index(pd.DatetimeIndex(df['date_time']))

        fig, axs = plt.subplots(3, 4, figsize=(8, 8), facecolor='w', edgecolor='k', sharey=True,sharex=True)
        fig.subplots_adjust(hspace=.05, wspace=.05)

        axs = axs.ravel()
        st = datetime.date(2016, 4, 1)
        titles = ['Apr 2016', 'May 2016', 'Jun 2016', 'Jul 2016', 'Aug 2016', 'Sep 2016', 'Oct 2016', 'Nov 2016',
                  'Dec 2016', 'Jan 2017', 'Feb 2017', 'Mar 2017']


        for m in range(12):
            # specify analysis

            en = self.add_months(st, 1)
            dftmp = df[st:en].copy()
            st = en

            hour = dftmp.index.hour
            res = []
            err = []
            time = range(24)
            time = [k+1 for k in time]

            for i in range(24):
                selector = (i == hour)
                df2 = dftmp[selector].copy()
                # now calculate correlation
                res.append(df2['co2flow'].corr(df2['h2oflow']))
            # interpolate

            f = interp1d(time, res)
            x = np.linspace(1, 24, num=2400, endpoint=True)
            axs[m].scatter(x, f(x), c=cm.seismic(np.abs((f(x)+1)/2.0)), edgecolor='none', s=2)
            axs[m].scatter(time, res, color='black', s=4)

            axs[m].set_ylim([-1, 1])
            axs[m].set_xlim([1,24])

            from matplotlib.ticker import MultipleLocator, FormatStrFormatter
            #minorLocator = MultipleLocator(3)
            majorLocator = MultipleLocator(6)
            #axs[m].xaxis.set_minor_locator(minorLocator)
            axs[m].xaxis.set_major_locator(majorLocator)
            axs[m].xaxis.grid(True, which="major", linestyle='--')
            axs[m].yaxis.set_major_locator(MaxNLocator(4))
            axs[m].yaxis.grid(True, which="major", linestyle='--')

            axs[m].set_title(titles[m], y=0.02, x=0.28, fontsize=10)

            if m == 9:
                axs[m].set_xlabel('')
            if m == 4:
                axs[m].set_ylabel('Pearson correlation coefficient')

            # plt.figure(figsize=(5,5))
            # plt.plot(time, res, marker='.', color='royalblue')
            # plt.xlabel('hour')
            # plt.ylabel('average CO$_2$ flow ($\mu$mol m$^{-2}$s$^{-1}$)')
            # plt.tight_layout()
            # plt.show()
        plt.tight_layout()
        plt.grid(linestyle='--')
        outputpath = 'C:\Users\herrmann\Documents\Herrmann Lars\Documentation\\figs\\'
        filename = 'hourly_correlation.png'
        plt.savefig(outputpath + filename, dpi=300)
        plt.show()

    def plot_correlation_entire_year(self):
        """ this plots the correlation coefficient between h2o and co2 over the entire year"""
        DIR1 = 'C:\Users\herrmann\Documents\Herrmann Lars\Licor\corrected\eddy_out\_tl_analysis_impact\\complete'
        FILE1 = 'full.csv'

        path = DIR1 + '\\' + FILE1
        df = pd.read_csv(path, sep=',', parse_dates=[['date', 'time']],
                         names=['date', 'time', 'co2flow', 'co2qc', 'h2oflow', 'h2oqc'],
                         usecols=[1, 2, 13, 14, 15, 16], header=2)

        # replace all values with missing data with the error flag code -9999
        df.replace(to_replace=np.nan, value=-9999, inplace=True)
        df.replace(to_replace=0, value=-9999, inplace=True)
        # replace data where quality control flag is 2 (bad) with np.nan
        df['orig'] = df['co2flow']
        df['bad'] = df.ix[df['co2qc'] == 2.0, 'co2flow']

        df.ix[df['co2qc'] == 2.0, 'co2flow'] = np.nan
        df.ix[df['co2qc'] == 2.0, 'h2oflow'] = np.nan
        df.ix[df['h2oqc'] == 2.0, 'h2oflow'] = np.nan
        df.ix[df['h2oqc'] == 2.0, 'co2flow'] = np.nan

        # --- NOTE: do not interpolate original dataframe on which correlation will be calculated
        # since this can increase correlation --------
        dfo = df.copy()
        dfo.interpolate(method='linear', inplace=True)
        # df.fillna(axis=0, method='ffill', inplace=True)
        dfo.ix[dfo['co2flow'] <= -100, 'shades'] = True
        dfo.ix[dfo['co2flow'] >= -100, 'shades'] = False
        df.ix[df['co2flow'] <= -50, 'co2flow'] = np.nan
        df.ix[df['co2flow'] >= 50, 'co2flow'] = np.nan
        df.ix[df['h2oflow'] <= -20, 'h2oflow'] = np.nan
        df.ix[df['h2oflow'] >= 20, 'h2oflow'] = np.nan



        # now bring back the -9999 values to np.nan
        df.replace(to_replace=-9999, value=np.nan, inplace=True)

        df = df.set_index(pd.DatetimeIndex(df['date_time']))
        dfo = dfo.set_index(pd.DatetimeIndex(dfo['date_time']))
        st = datetime.datetime(2017, 1, 1, 0, 0, 0)
        percentage_high = []
        percentage_low =[]

        titles = ['Apr 2016', 'May 2016', 'Jun 2016', 'Jul 2016', 'Aug 2016', 'Sep 2016',  'Oct 2016', 'Nov 2016',
                  'Dec 2016', 'Jan 2017', 'Feb 2017', 'Mar 2017']
        for i in range(12):
            if i < 9:
                fin = datetime.datetime(2016, i+4, 1, 0, 0, 0)
                st = fin
                startstamp = '2016-' + str(i+4) + '-01'
                if i ==8:
                    stopstamp = '2017-01-01'
                else:
                    stopstamp = '2016-' + str(i+5) + '-01'
            else:
                fin = datetime.datetime(2017, i-8, 1, 0, 0, 0)
                st = fin
                startstamp = '2017-' + str(i-8) + '-01'
                stopstamp = '2017-' + str(i-7) + '-01'
            time = []
            res = []
            p_value = []
            rg = int((fin-st).total_seconds()/3600)
            print rg
            for t in range(120):
                if t % 500 == 0:
                    print("Step " + str(t) + "/" + str(rg))
                en = (st + datetime.timedelta(hours=24))
                time.append((st + datetime.timedelta(hours=12)).strftime("%Y-%m-%d %H:%M:%S"))
                dftmp = df[st:en].copy()

                # count nan values in range and if too high (>p where p i ratio), append nan instead of correlation
                p = 0.66  # rejection ratio
                try:
                    if float(dftmp['co2flow'].isnull().sum())/float(len(dftmp['co2flow'])) > p:
                        res.append(np.nan)
                    else:
                        res.append(dftmp['co2flow'].corr(dftmp['h2oflow']))
                except ZeroDivisionError:
                        res.append(np.nan)
                # compute p value of correlation
                p_value.append(pearsonr(dftmp['co2flow'], dftmp['h2oflow'])[1])
                st = en

            fig, ax = plt.subplots(figsize=(10, 2.75))

            dfs = pd.DataFrame()
            dfs['dat'] = time
            dfs['cor'] = res
            dfs['pval'] = p_value
            dfs = dfs.set_index(pd.DatetimeIndex(dfs['dat']))
            df_reindexed = dfs.reindex(pd.date_range(start=df.index.min(),
                                                     end=df.index.max(),
                                                     freq='1 min'))
            df_reindexed.interpolate(method='linear', inplace=True)
            ax.scatter(df_reindexed.index, df_reindexed['cor'], c=cm.seismic(np.abs((df_reindexed['cor'] + 1) / 2.0)), edgecolor='none', s=2)
            ax.scatter(dfs.index, dfs['cor'], s=4, color='black')
            #dfs.plot(x='dat', y='cor', color='black', ax=ax)
            ax.set_ylim([-1, 1])
            ax.set_xlabel('')
            ax.set_ylabel('Pearson correlation coefficient')
            #
            ax.set_title(titles[i], y=0.88, x=0.05, fontsize=10, fontweight='bold')

            ax.fill_between(dfo.index, y1=0.75, y2=1,  color='green', alpha=0.4, linewidth=0)
            ax.fill_between(dfo.index, y1=0.5, y2=0.75, color='green', alpha=0.2, linewidth=0)
            ax.fill_between(dfo.index, y1=0.25, y2=0.5, color='green', alpha=0.1, linewidth=0)
            ax.fill_between(dfo.index, y1=-0.25, y2=0.25, color='grey', alpha=0.2, linewidth=0)
            ax.fill_between(dfo.index, y1=-0.25, y2=-0.5, color='orange', alpha=0.1, linewidth=0)
            ax.fill_between(dfo.index, y1=-0.5, y2=-0.75, color='orange', alpha=0.2, linewidth=0)
            ax.fill_between(dfo.index, y1=-0.75, y2=-1, color='orange', alpha=0.4, linewidth=0)
            ax.fill_between(dfo.index, y1=100, y2=-100, where=dfo['shades'], color='thistle', alpha=1, linewidth=0)
            ax.xaxis.set_minor_locator(dates.DayLocator())
            ax.xaxis.grid(True, which="minor", linestyle='--')
            # #ax.yaxis.set_major_locator(MaxNLocator(4))
            # #ax.yaxis.grid(True, which="major")
            plt.tight_layout()
            plt.xlim(pd.Timestamp(startstamp), pd.Timestamp(stopstamp))

            outputpath = 'C:\Users\herrmann\Documents\Herrmann Lars\Documentation\\figs\\'
            filename = 'correlation_analysis_daily_'+ str(i) + '.png'
            plt.savefig(outputpath + filename, dpi=300)

            # perform statistical analysis

            # count how many data points above 0.9
            counter_tot = 0
            counter_high = 0
            counter_low = 0
            for index, row in dfs.iterrows():
                if row['cor'] >= 0.9:
                    counter_high +=1
                elif row['cor'] <= -0.9:
                    counter_low +=1
                counter_tot+=1
            percentage_high.append(100.0*counter_high/counter_tot)
            percentage_low.append(100.0 * counter_low / counter_tot)

        # fig, ax = plt.subplots(figsize=(8, 3))
        # x = range(12)
        # x2 = [0, 2, 4, 6, 8, 10]
        # ax.plot(x, percentage_high, color='royalblue', label='$C>0.9$')
        # ax.plot(x, percentage_low, color='seagreen', label='$C<0.9$')
        # ax.set_ylabel('Percentage (%)')
        # ax.legend(frameon=False, loc='upper left', fontsize=9)
        # plt.xticks(x2, titles)
        # outputpath = 'C:\Users\herrmann\Documents\Herrmann Lars\Documentation\\figs\\'
        # filename = 'correlation_percentage.png'
        # plt.savefig(outputpath + filename, dpi=300)
        # plt.show()

        # fig, ax = plt.subplots(figsize=(8, 3))
        # x = range(12)
        # x2 = [0, 2, 4, 6, 8, 10]
        # ax.plot(dfs.index, dfs['pval'], color='royalblue')
        #
        # ax.set_ylabel('p-value (%)')
        # #ax.legend(frameon=False, loc='upper left', fontsize=9)
        # #plt.xticks(x2, titles)
        # outputpath = 'C:\Users\herrmann\Documents\Herrmann Lars\Documentation\\figs\\'
        # filename = 'correlation_p_value.png'
        # plt.savefig(outputpath + filename, dpi=300)
        # plt.show()

    def plot_flux_of_day(self, st):
        """ plots co2 and h2o flux of specified day
        @param: st      start date, date object
        """

        DIR1 = 'C:\Users\herrmann\Documents\Herrmann Lars\Licor\corrected\eddy_out\_tl_analysis_impact\\complete'
        FILE1 = 'full.csv'
        print("Loading data")
        path = DIR1 + '\\' + FILE1
        df = pd.read_csv(path, sep=',', parse_dates=[['date', 'time']],
                         names=['date', 'time', 'co2flow', 'co2qc', 'h2oflow', 'h2oqc'],
                         usecols=[1, 2, 13, 14, 15, 16], header=2)
        print("\tData loaded")
        # replace all values with missing data with the error flag code -9999
        df.replace(to_replace=np.nan, value=-9999, inplace=True)
        df.replace(to_replace=0, value=-9999, inplace=True)

        df.ix[df['co2qc'] == 2.0, 'co2flow'] = np.nan
        df.ix[df['co2qc'] == 2.0, 'h2oflow'] = np.nan
        df.ix[df['h2oqc'] == 2.0, 'h2oflow'] = np.nan
        df.ix[df['h2oqc'] == 2.0, 'co2flow'] = np.nan
        dforig = df.copy()
        # now interpolate all nan values
        df.interpolate(method='linear', inplace=True)

        # now bring back the -9999 values to np.nan
        df.replace(to_replace=-9999, value=np.nan, inplace=True)

        # ensure not to plot outliers
        df.ix[df['co2flow'] <= -50, 'co2flow'] = np.nan
        df.ix[df['co2flow'] >= 50, 'co2flow'] = np.nan
        df.ix[df['h2oflow'] <= -20, 'h2oflow'] = np.nan
        df.ix[df['h2oflow'] >= 20, 'h2oflow'] = np.nan

        df = df.set_index(pd.DatetimeIndex(df['date_time']))
        dforig = dforig.set_index(pd.DatetimeIndex(dforig['date_time']))

        # also make sure there are no outliers in the original frame
        dforig.ix[dforig['co2flow'] <= -50, 'co2flow'] = np.nan
        dforig.ix[dforig['co2flow'] >= 50, 'co2flow'] = np.nan
        dforig.ix[dforig['h2oflow'] <= -20, 'h2oflow'] = np.nan
        dforig.ix[dforig['h2oflow'] >= 20, 'h2oflow'] = np.nan

        stop = st + datetime.timedelta(days=1)
        print ("Start date " + st.strftime("%Y-%m-%d"))
        print ("Stop date: " + stop.strftime("%Y-%m-%d"))

        fig, axs = plt.subplots(2, 1, figsize=(3, 5), facecolor='w', edgecolor='k', sharex=True)

        dftmp = df[st:stop].copy()
        dforigtmp = dforig[st:stop].copy()
        corr_coeff = dforigtmp['co2flow'].corr(dforigtmp['h2oflow'])
        dftmp = dftmp.set_index(pd.DatetimeIndex(dftmp['date_time']))
        dforigtmp = dforigtmp.set_index(pd.DatetimeIndex(dforigtmp['date_time']))

        axs[0].plot(dftmp['date_time'], dftmp['co2flow'], color='royalblue', marker='.')
        axs[0].plot(dforigtmp['date_time'], dforigtmp['co2flow'], color='black', marker='.', linewidth=0)
        axs[1].plot(dftmp['date_time'], dftmp['h2oflow'], color='seagreen', marker='.')
        axs[1].plot(dforigtmp['date_time'], dforigtmp['h2oflow'], color='black', marker='.', linewidth=0)

        import matplotlib.dates as mdates
        myFmt = mdates.DateFormatter('%H')
        axs[0].xaxis.set_major_formatter(myFmt)
        axs[1].xaxis.set_major_formatter(myFmt)
        axs[0].xaxis.grid(linestyle='--')
        axs[1].xaxis.grid(linestyle='--')
        axs[1].set_xlabel('hour of day')
        axs[0].set_ylabel('CO$_2$ flux ($\mu$mol m$^{-2}$s$^{-1}$)')
        axs[1].set_ylabel('H$_2$O flux (10 mmol m$^{-2}$s$^{-1}$)')
        axs[0].set_title(st.strftime("%Y-%m-%d"), y=0.85, x=0.3, fontsize=10,
                         bbox=dict(edgecolor='white', facecolor='white', alpha=0.5), fontweight='bold')
        plt.figtext(0.75, 0.6, 'C=' + str(np.round(corr_coeff, 2)), bbox=dict(edgecolor='white', facecolor='white', alpha=0.5), fontsize=8)


        # ax1.set_xticks([])
        # ax2.set_xticks([])
        #
        # ax[1].set_yticks([-10, 0, 10])
        # ax2.set_yticks([-1, 0, 1])
        # ax1.set_ylim([-25, 25])
        # ax2.set_ylim([-1, 1])

        plt.tight_layout()
        outputpath = 'C:\Users\herrmann\Documents\Herrmann Lars\Documentation\\figs\\'
        filename = 'flux_of_day_' + st.strftime("%Y-%m-%d") + '_.png'
        plt.savefig(outputpath + filename, dpi=300)
        plt.show()

if __name__ == "__main__":
    rw_data = DisplayData()

