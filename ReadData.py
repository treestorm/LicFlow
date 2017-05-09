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
    def read_concentration(conc_path):
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
        """ reads a wind file under wind_path, extracts vertical wind speed 
        into an dataframe with two columns[time, zspeed]
        :param wind_path: the path to the file to be read
        :return: data format with time stamp and vertical wind speed """

        # open file as read only
        f = open(wind_path, "r")

        # extract date from path
        dat = wind_path.split('_')[1].split('-')
        dat_string = '20' + dat[2]+ '-' + dat[1] + '-' + dat[0] + ' '
        year = int('20' + dat[2])
        month = int(dat[1])
        day = int(dat[0])

        # converter to strip characters in front of time stamp string and efficient parsing into datetime
        convertfunc = lambda x: self.dateParser(x.split('\t')[1], year, month, day)

        # usecols       column 3 is vertical wind speed, column 8 the time stamp
        # header        skip first three lines which are default header
        # sep           tab delimited data
        df = None
        try:
            df = pd.read_table(f, names={'vz', 'vy', 'ftemp', 'wa', 'vx', 'time'}, engine='c', sep=',', header=0, usecols=[1,2,3,5,6,8], converters={8:convertfunc})
        except IndexError:
            return 0, df
        df.columns = ['vx', 'vy', 'vz', 'ftemp', 'wa', 'time']

        # fill na
        df.fillna(method='backfill', inplace=True)
        # already try to remove duplicates here
        df.drop_duplicates(keep='first', inplace=True)

        df = df.set_index(pd.DatetimeIndex(df['time']))

        # reverse column order
        df = df[df.columns[::-1]]

        # now check first and last date
        date1 = df.iloc[0]['time']
        date2 = df.iloc[-1]['time']

        i = 1
        while type(date1) is not pd.tslib.Timestamp and i<10:
            date1 = df.iloc[i]['time']
            i += 1
        i=2
        while type(date2) is not pd.tslib.Timestamp and i < 10:
            date1 = df.iloc[-i]['time']
            i += 1

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

            df = pd.concat([df1, df2])
            df.sort_values('time')

        # drop rows with nan values
        df = df.set_index(pd.DatetimeIndex(df['time']))
        df.fillna(method='backfill', inplace=True)
        df.drop_duplicates(keep='first', inplace=True)

        try:
            df_reindexed = df.reindex(rng, method='nearest')
            df_reindexed.interpolate(method='linear')
        except ValueError:
            return 0, df

        # close file
        f.close()

        df_reindexed = df_reindexed.set_index(pd.DatetimeIndex(df_reindexed['time']))

        return 1, df_reindexed

    def dateParser(self, s, y, m, d):
        """ this is a much faster parser tailored for the specific data"""
        return datetime.datetime(year=y, month=m, day=d, hour=int(s[0:2]), minute=int(s[3:5]), second=int(s[6:8]), microsecond=1000 * int(s[9:12]))

    @staticmethod
    def get_wind_paths_ordered(basedir_wind, yr, mth):
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
        """ returns all the paths of the LICOR data files (with concentration data) in random order """

        # populate list with the paths
        conc_paths = []
        for f in os.listdir(basedir_conc):
            if f.startswith("Licor"):
                conc_paths.append(str(os.path.join(basedir_conc, f)))
        if len(conc_paths) == 0:
            return 0
        else:
            return conc_paths

    @staticmethod
    def get_conc_paths_ordered2(basedir_conc):
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
    def get_conc_paths_ordered(basedir_conc):
        """ returns all the paths of the LICOR data files (with concentration data) in chronological order as a list 
        based on the file name
        
        However, this method does not appear useful due to inconsistent naming convention"""

        # first count how many files there will be
        N = 0
        for f in os.listdir(basedir_conc):
            if f.startswith("Licor"):
                N += 1
        if N == 0:
            return 0

        # now create empty arrays and list with final paths
        conc_dates = np.empty((N, 1), dtype='datetime64[m]')
        conc_paths = np.empty((N, 1), dtype='object')

        sorted_conc_paths = []
        # and fill array with datetime (extracted from filename) and paths to the files
        k = 0
        for f in os.listdir(basedir_conc):
            if f.startswith("Licor"):
                # split file name up into Year Month Day
                spl = f.split(".")
                dat = spl[0]
                dat = dat.split("-")
                # assemble datetime object
                base = datetime.datetime(int(dat[3]), int(dat[2]), int(dat[1]))
                # populate array
                conc_dates[k] = base
                conc_paths[k] = str(os.path.join(basedir_conc, f))
                k += 1

        # create a combined structured array
        records = np.rec.fromarrays((conc_dates, conc_paths), names=('date', 'path'))

        if records is not None:
            # sort the array according to the datetime objects
            records.sort(axis=0)
            # extract the paths from the structured array
            paths = records['path']
            # convert them into strings and put them into a list
            for x in paths:
                x = str(x)
                sorted_conc_paths.append(x.split("'")[1])

        # return the list containing all the paths in the specified folder
        return sorted_conc_paths


if __name__ == "__main__":
    rd_data = ReadData()
    import time

    err, dd = rd_data.read_wind('C:\Users\herrmann\Documents\Herrmann Lars\Licor\wind\WIND_19-04-17_19-45.txt')








