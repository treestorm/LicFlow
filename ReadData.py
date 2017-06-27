import numpy as np
import os
import datetime
import pandas as pd

"""
    File name: ReadData.py
    Author: treestorm
    Date created: 09/05/2017
    Python Version: 2.7
"""

class ReadData:
    def __init__(self):
        """ Init methdod for read data"""

    @staticmethod
    def read_conc(conc_path):
        """ reads a Licor file with concentrations under conc_path, extracts co2 concentration 
        into an dataframe with two columns[time, conc]
        :param conc_path: the path to the file to be read holding the concentrations
        :return: data format with time stamp and co2 concentration data
        """

        # open file as read only
        f = open(conc_path, "r")

        # usecols       column 0 is timestamp, column 11 the co2 concentration data
        # header        skip first three lines which are default header
        # sep           tab delimited data
        df = 0
        try:
            df = pd.read_table(f, parse_dates=True, infer_datetime_format=True, sep='\t', header=3, usecols=[0, 13, 29])
        except ValueError:
            return 0, df
        # fill all missing values with 0.0
        df.fillna(0.0, inplace=True)
        # rename columns
        df.columns = ['time', 'co2conc', 'h2oconc']
        # check if datatype is datetime object
        if df['time'].dtype == 'object':
            try:
                df['time'] = pd.to_datetime(df['time'])
            except ValueError:
                print("Datetime conversion error")
                return 0, df
        if df['co2conc'].dtype == 'object':
            try:
                df['co2conc'] = pd.to_numeric(df['co2conc'])
            except ValueError:
                print("Float conversion error")
                return 0, df
        if df['h2oconc'].dtype == 'object':
            try:
                df['h2oconc'] = pd.to_numeric(df['h2oconc'])
            except ValueError:
                print("Float conversion error")
                return 0, df
        # convert concentrations to float and take absolute value
        df['co2conc'] = df['co2conc'].apply(lambda x: abs(float(x)))
        df['h2oconc'] = df['h2oconc'].apply(lambda x: abs(float(x)))
        # add thousands secs to time stamp in order to bring into same format as wind data
        df['time'] = df['time'].astype(str) + '.000'
        df['time'] = pd.to_datetime(df['time'])

        # set datetime as index
        df = df.set_index(pd.DatetimeIndex(df['time']))
        # close file
        f.close()
        return 1, df

    def read_wind(self, wind_path):
        """ reads a wind file under wind_path and extracts the three wind speeds (x,y,z), sonic speed and
        sonic temperature.
        :param wind_path: the path to the file to be read
        :return: data format with datetime stamp in 50 ms steps and resampled anemometer data"""

        # general remark: the missing date complicates the procedure since for each file we need to check if there
        # are entries from two separate days. If this is the case, we need to identify the crossover point (midnight)
        # and assign one date to the first entries (before midnight) and another date to the later entries (after
        # midnight)

        # open file as read only
        f = open(wind_path, "r")

        # extract date from path
        # this is required since the date is missing in the file
        dat = wind_path.split('_')[1].split('-')
        year = int('20' + dat[2])
        month = int(dat[1])
        day = int(dat[0])

        # converter to strip characters in front of time stamp string and efficient parsing into datetime
        convertfunc = lambda x: self.dateParser(x.split('\t')[1], year, month, day)

        # load the file
        df = None
        try:
            df = pd.read_table(f, names={'vz', 'vy', 'ftemp', 'wa', 'vx', 'time'},
                               engine='c', sep=',', header=0, usecols=[1,2,3,5,6,8],
                               converters={8:convertfunc})
        except IndexError:
            return 0, df
        df.columns = ['vx', 'vy', 'vz', 'ftemp', 'wa', 'time']

        # fill nan values (back-fill)
        df.fillna(method='backfill', inplace=True)
        # already try to remove duplicates here
        df.drop_duplicates(keep='first', inplace=True)
        # set the time as index
        df = df.set_index(pd.DatetimeIndex(df['time']))

        # reverse column order
        df = df[df.columns[::-1]]

        # now check first and last time stamp
        date1 = df.iloc[0]['time']
        date2 = df.iloc[-1]['time']

        # this helps if some of the first few time entries are faulty
        i = 1
        while type(date1) is not pd.tslib.Timestamp and i<10:
            date1 = df.iloc[i]['time']
            i += 1
        i=2
        while type(date2) is not pd.tslib.Timestamp and i < 10:
            date1 = df.iloc[-i]['time']
            i += 1

        # now check if the file only contains entries from one day or from multiple
        # if the second time is bigger than the first, we assume it is only one day
        # else there was a crossover
        # this method assumes that no wind file is longer than one day which is the case
        # for the present data
        if date1 < date2:       # if datafile does not contain two dates
            rng = pd.date_range(start=date1, end=date2, freq='50 ms')
        else:                   # if datafile contains two dates (i.e. switches over midnight)
            d1 = date1
            d2 = date2
            try:
                d2 = d2 + datetime.timedelta(days=1)
            except TypeError:
                return 0, df
            rng = pd.date_range(start=d1, end=d2, freq='50 ms')

            initial_time = str(date1).split('.')[0].split(' ')[1]
            # now select entries between this date and midnight
            md = datetime.time(hour=23, minute=59, second=59, microsecond=999999)
            df1 = df.between_time(initial_time, md)
            # disable warning for modifying slice
            pd.options.mode.chained_assignment = None  # default='warn'

            # now find second part in file where date is not the same as specified in filename
            end_time = str(date2).split('.')[0].split(' ')[1]
            #
            df2 = df.between_time('00:00:00', end_time)
            # update these values with date extracted from file path + 1 day
            dat = datetime.date(int('20' + dat[2]), int(dat[1]), int(dat[0]))
            dat = dat + datetime.timedelta(days=1)
            df2['time'] = df2['time'] + datetime.timedelta(days=1)

            # merge the two data frames together into one data frame and sort by time
            df = pd.concat([df1, df2])
            df.sort_values('time')

        # drop rows with nan values and duplicates, sets the datetime as index
        df = df.set_index(pd.DatetimeIndex(df['time']))
        df.fillna(method='backfill', inplace=True)
        df.drop_duplicates(keep='first', inplace=True)

        # now reindex onto the newely created timeline with 50 ms steps
        try:
            df_reindexed = df.reindex(rng, method='nearest')
            df_reindexed.interpolate(method='linear')
        except ValueError:  # if there was an error, simply return 0 and dataframe for analysis
            return 0, df

        # close file
        f.close()

        # set datetime as index of the reindexed dataframe
        df_reindexed = df_reindexed.set_index(pd.DatetimeIndex(df_reindexed['time']))

        # return the wind data, now sampled onto a 50 ms timeline
        return 1, df_reindexed

    def read_meteo(self, meteo_path):
        """ reads a meteo file under meteo_path, extracts temperature, pressure, relative hmidity
        into an dataframe with two columns[date, time, Ta, p, RH]
        :param meteo_path: the path to the file to be read
        :return: data frame """

        df = None
        try:
            df = pd.read_table(meteo_path, sep='\t', parse_dates=[['date', 'time']], infer_datetime_format=True, dayfirst=True,
                                 names=['date', 'time', 'Pa', 'RH', 'Ta'],
                                 usecols=[0, 1, 33, 34, 35], header=1)
        except IndexError:
            return 0, df
        df.columns = ['TIMESTAMP_1', 'Pa', 'RH', 'Ta']

        # fill na
        df.fillna(method='backfill', inplace=True)
        # already try to remove duplicates here
        df.drop_duplicates(keep='first', inplace=True)

        df = df.set_index(pd.DatetimeIndex(df['TIMESTAMP_1']))

        # now check first and last date
        date1 = df.iloc[0]['TIMESTAMP_1']
        date1 = date1.replace(second=0, microsecond=0)
        date2 = df.iloc[-1]['TIMESTAMP_1']
        date2 = date2.replace(second=0, microsecond=0)

        # fill na
        df.fillna(method='backfill', inplace=True)
        # already try to remove duplicates here
        df.drop_duplicates(subset='TIMESTAMP_1', keep='first', inplace=True)
        rng = pd.date_range(start=date1, end=date2, freq='1 min')

        try:
            df_reindexed = df.reindex(rng, method='nearest')
            df_reindexed.interpolate(method='linear')
        except ValueError:
            return 0, df
        return 1, df_reindexed

    def dateParser(self, s, y, m, d):
        """ this is a much faster parser tailored for the specific data"""
        return datetime.datetime(year=y, month=m, day=d, hour=int(s[0:2]), minute=int(s[3:5]), second=int(s[6:8]), microsecond=1000 * int(s[9:12]))

    @staticmethod
    def get_wind_paths(basedir_wind, yr, mth):
        """ returns all the wind paths in chronological order as a list """

        # first count how many files there will be
        N = 0
        for f in os.listdir(basedir_wind):
            if f.startswith("WIND_") and f.endswith(".TXT"):
                N += 1
        if N == 0:
            return 0

        # create list to store paths and dates
        wind_paths_list = []
        wind_dates_list = []
        for f in os.listdir(basedir_wind):
            if f.startswith("WIND_") and f.endswith(".TXT"):
                # split file name up into Year Month Day Hour Minute
                spl = f.split("_")
                dat = spl[1]
                tim = spl[2]
                dat = dat.split("-")
                tim = tim.split("-")
                min = tim[1].split(".")
                # assemble datetime object
                base = datetime.datetime(int('20' + dat[2]), int(dat[1]), int(dat[0]), int(tim[0]), int(min[0]))
                # append to list
                wind_dates_list.append(base)
                wind_paths_list.append(str(os.path.join(basedir_wind, f)))

        # sort path list according to date list
        wind_dates = wind_dates_list
        wind_paths = [wind_paths_list for (wind_dates_list, wind_paths_list) in sorted(zip(wind_dates_list, wind_paths_list))]
        wind_dates.sort()

        # now create a list containing sublists with indices which each correspond to one day including one file overlap (before and after)
        index_lists = []
        curr_list = []
        initial_date = datetime.datetime(wind_dates[0].year, wind_dates[0].month, wind_dates[0].day)
        current_date = initial_date
        index = 0
        date_list = [] # list containing the dates in same sequence as list with sublist indices
        for i in wind_dates:
            # compare based on days
            if datetime.datetime(i.year, i.month, i.day) == current_date:
                curr_list.append(index)
            else:  # create a new list with indices
                # also append file from next day since it may still contain some data from current date
                if index < len(wind_dates) - 1:
                    curr_list.append(index)

                # now add list to index list
                if current_date is not initial_date:  # ignore the initial date, because it will not be complete...
                    index_lists.append(curr_list)

                date_list.append(current_date)
                # update date to next day
                current_date = datetime.datetime(wind_dates[index].year, wind_dates[index].month, wind_dates[index].day)

                # reset index list
                curr_list = []
                # in new list add previous index and current index (which is already new date)
                if index is not 0:
                    curr_list.append(index - 1)
                    curr_list.append(index)  #
            # update running index
            index += 1

        # now we only want to return the index of the files which lie in specified year-month

        ar1 =  date_list >= np.datetime64(datetime.datetime(yr, mth, 1)-datetime.timedelta(days=1))
        if(mth+1)>12:
            mth = 0
            yr = yr+1
        ar2 = date_list < np.datetime64(datetime.datetime(yr, mth+1, 1)-datetime.timedelta(days=1))
        sel = np.where(ar1 & ar2)[0]

        ind_list = []
        for i in sel:
            ind_list.append(index_lists[i])

        return [ind_list, wind_paths]

    @staticmethod
    def get_conc_paths(basedir_conc):
        """  this iterates through all files in folder and extracts the timestamp of the first entry """

        # populate list with the paths
        conc_paths = []
        start_date = []
        for f in os.listdir(basedir_conc):
            if f.startswith("Licor"):
                path = str(os.path.join(basedir_conc, f))
                conc_paths.append(path)
                df = pd.read_table(path, names=['date'], parse_dates=True, infer_datetime_format=True, sep='\t', header=2, usecols=[0], nrows=1)
                start_date.append(df.iloc[0]['date'])

        # now order conc_path according to date_list
        st = sorted(zip(start_date, conc_paths))
        conc_paths = [c for (s, c) in st]
        start_date_list = [s for (s, c) in st]

        if len(conc_paths) == 0:
            return 0
        else:
            return conc_paths

    @staticmethod
    def get_meteo_paths(basedir_meteo, yr, mth):
        """ returns all the meteo paths in chronological order as a list as well as a list
        with indiced grouped into days"""

        # first count how many files there will be
        N = 0
        for f in os.listdir(basedir_meteo):
            if f.startswith("DATA_") and f.endswith(".TXT"):
                N += 1
        if N == 0:
            return 0

        # create list to store paths and dates
        meteo_paths_list = []
        meteo_dates_list = []
        for f in os.listdir(basedir_meteo):
            if f.startswith("DATA_") and f.endswith(".TXT"):
                # split file name up into Year Month Day Hour Minute
                spl = f.split("_")
                dat = spl[1]
                tim = spl[2]
                dat = dat.split("-")
                tim = tim.split("-")
                min = tim[1].split(".")
                # assemble datetime object
                base = datetime.datetime(int('20' + dat[2]), int(dat[1]), int(dat[0]), int(tim[0]), int(min[0]))
                # append to list
                meteo_dates_list.append(base)
                meteo_paths_list.append(str(os.path.join(basedir_meteo, f)))

        # sort path list according to date list
        meteo_dates = meteo_dates_list
        meteo_paths = [meteo_paths_list for (meteo_dates_list, meteo_paths_list) in
                      sorted(zip(meteo_dates_list, meteo_paths_list))]
        meteo_dates.sort()

        # now create a list containing sublists with indices which each correspond to one day including one file overlap (before and after)
        index_lists = []
        curr_list = []
        initial_date = datetime.datetime(meteo_dates[0].year, meteo_dates[0].month, meteo_dates[0].day)
        current_date = initial_date
        index = 0
        date_list = []  # list containing the dates in same sequence as list with sublist indices
        for i in meteo_dates:
            # compare based on days
            if datetime.datetime(i.year, i.month, i.day) == current_date:
                curr_list.append(index)
            else:  # create a new list with indices
                # also append file from next day since it may still contain some data from current date
                if index < len(meteo_dates) - 1:
                    curr_list.append(index)

                # now add list to index list
                if current_date is not initial_date:  # ignore the initial date, because it will not be complete...
                    index_lists.append(curr_list)

                date_list.append(current_date)
                # update date to next day
                current_date = datetime.datetime(meteo_dates[index].year, meteo_dates[index].month, meteo_dates[index].day)

                # reset index list
                curr_list = []
                # in new list add previous index and current index (which is already new date)
                if index is not 0:
                    curr_list.append(index - 1)
                    curr_list.append(index)  #
            # update running index
            index += 1

        # now we only want to return the index of the files which lie in specified year-month

        ar1 = date_list >= np.datetime64(datetime.datetime(yr, mth, 1) - datetime.timedelta(days=1))
        if (mth + 1) > 12:
            mth = 0
            yr = yr + 1
        ar2 = date_list < np.datetime64(datetime.datetime(yr, mth + 1, 1) - datetime.timedelta(days=1))
        sel = np.where(ar1 & ar2)[0]

        ind_list = []
        for i in sel:
            ind_list.append(index_lists[i])

        return [ind_list, meteo_paths]

    def get_meteo_paths_tmp(self, basedir_conc):
        """  this iterates through all files in folder and extracts the timestamp of the first entry
        then returns a sorted list of paths in chronological order """

        # populate list with the paths
        meteo_paths = []
        start_date = []
        for f in os.listdir(basedir_conc):
            if f.startswith("DATA_"):
                path = str(os.path.join(basedir_conc, f))
                df = pd.read_csv(path, sep='\t', parse_dates=[['date', 'time']], infer_datetime_format=True,
                                 names=['date', 'time'],
                                 usecols=[0, 1], header=1, nrows=1)
                try:
                    start_date.append(df.iloc[0]['date_time'])
                    meteo_paths.append(path)
                except IndexError:
                    print("Index Error for file " + path)

        # now order conc_path according to date_list
        st = sorted(zip(start_date, meteo_paths))
        meteo_paths = [c for (s, c) in st]
        start_date_list = [s for (s, c) in st]

        if len(meteo_paths) == 0:
            return 0
        else:
            return meteo_paths

if __name__ == "__main__":
    rd_data = ReadData()
    import time









