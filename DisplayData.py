
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib

from scipy import integrate
import numpy as np
import matplotlib.font_manager
import matplotlib.gridspec as gridspec
import matplotlib.dates as mdates
import datetime

class DisplayData:

    def __init__(self):
        """ provides functions to extract and to plot data from eddy pro output files"""


        DIR = 'C:\\Users\\herrmann\\Documents\\Herrmann Lars\\Licor\\corrected\\eddy_out\\complete_run_time_lag_corrected'
        FILE = 'eddypro_LH_full_output_2017-05-10T170241_adv.csv'
        self.integrate_eddy_data(DIR, FILE)
        #self.extract_eddy_data(DIR, FILE)
        #self.compare_eddy_data(DIR)

    def integrate_eddy_data(self, dir, filename):
        """ integrates the data over time to see the net flux"""

        path = dir + '\\' + filename

        # co2 concentration
        # ----------------------------
        df = pd.read_csv(path, sep=',', parse_dates=[['date', 'time']], names=['date', 'time', 'co2flow', 'h2oflow'],
                         usecols=[1, 2, 13, 15], header=2)

        df.replace(to_replace=-9999, value=0, inplace=True, limit=None, regex=False, method='pad')
        df = df.set_index(pd.DatetimeIndex(df['date_time']))
        print df['date_time']

        df_int = integrate.cumtrapz(df['co2flow'], df.index.astype(np.int64) / 10 ** 9, initial=0)
        df['integral'] = df_int/1e6  ## !! convert to mol/m2

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


    def extract_eddy_data(self, dir, filename):
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





if __name__ == "__main__":
    rw_data = DisplayData()

