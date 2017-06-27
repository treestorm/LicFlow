import os
import datetime
import pandas as pd
import time
import numpy as np
import bisect
import ReadData
import ReformatData

"""
    File name: RewriteData.py
    Author: treestorm
    Date created: 09/05/2017
    Python Version: 2.7
"""

class RewriteData:
    """
    The purpose of this class is to combine wind and concentration data in combined file with common timeline.
    """

    def __init__(self):

        # ------------------------------------------------------------------------
        # COMBINE WIND & CONCENTRATION DATA ON UNIFORM TIMELINE IN ONE FILE
        # location of corrected wind and conc files
        DIR_WIND = 'C:\Users\herrmann\Documents\Herrmann Lars\Licor\wind_corrected'
        DIR_CONC = 'C:\Users\herrmann\Documents\Herrmann Lars\Licor\conc_corrected'
        DIR_OUTPUT_COMB = 'C:\Users\herrmann\Documents\Herrmann Lars\Licor\corrected'

        startdate = "2016-04-01 00:00:00.000"
        enddate = "2016-04-30 23:59:59.950"
        # UNCOMMENT HERE TO PROCESS
        #self.create_uniform_timeline(DIR_WIND, DIR_CONC, DIR_OUTPUT_COMB, startdate, enddate)
        # ------------------------------------------------------------------------

        # ------------------------------------------------------------------------
        # REWRITE ALL THE WIND AND CONCENTRATION DATA INTO A HDF5 FILE
        #DIR_IN = 'C:\Users\herrmann\Documents\Herrmann Lars\Licor\corrected'
        #DIR_OUT = 'C:\Users\herrmann\Documents\Herrmann Lars\Licor\corrected'
        # UNCOMMENT HERE TO PROCESS
        #self.create_hdf5(DIR_IN, DIR_OUT)
        # ------------------------------------------------------------------------

        # ------------------------------------------------------------------------
        # REWRITE ALL THE WIND AND CONCENTRATION DATA INTO FORMAT FOR EDDY PRO
        DIR_IN = 'C:\Users\herrmann\Documents\Herrmann Lars\Licor\corrected'
        DIR_OUT = 'C:\Users\herrmann\Documents\Herrmann Lars\Licor\corrected\eddy_resampling\\instant\\'
        # UNCOMMENT HERE TO PROCESS
        #self.resample_for_eddy_pro(DIR_IN, DIR_OUT, startdate, enddate)
        #self.rewrite_for_eddy_pro_resampled(DIR_IN, DIR_OUT, startdate, enddate, 5000)
        # ------------------------------------------------------------------------

        # ------------------------------------------------------------------------
        # LOAD, SORT AND REWRITE ALL METEO DATA IN BIOMET DATA FILE FOR EDDY PRO
        DIR_IN = 'C:\Users\herrmann\Documents\Herrmann Lars\Licor\meteo'
        DIR_OUT = 'C:\Users\herrmann\Documents\Herrmann Lars\Licor\corrected\\biomet\\'
        # UNCOMMENT HERE TO PROCESS
        self.rewrite_meteo_data(DIR_IN, DIR_OUT)
        # ------------------------------------------------------------------------

    def create_uniform_timeline(self, wind_dir, conc_dir, out_dir, startdate, enddate):
        """ 
        This puts the wind and concentration data onto a uniform timeline.
        
        :param wind_dir: input directory of wind files
        :param conc_dir: input directory of concentration files
        :param out_dir: output directory where processed data is stored
        :param startdate: first date of file to process
        :param enddate: final date of file to process
        :return: void (no return)
        """

        # to much data to store in ram, so will follow day by day approach
        d = datetime.datetime.strptime(startdate, "%Y-%m-%d %H:%M:%S.%f")
        de = datetime.datetime.strptime(enddate, "%Y-%m-%d %H:%M:%S.%f")
        total_days = (de-d).days
        processed_days = 0
        ds = d

        while d < de: # while earlier than end date
            # initialise the read data class
            start = time.time()
            print("Processing file: " + d.strftime("%Y-%m-%d %H:%M:%S.%f"))
            df = d + datetime.timedelta(days=1)
            rng = pd.date_range(start=d.strftime("%Y-%m-%d %H:%M:%S.%f"), end=df.strftime("%Y-%m-%d %H:%M:%S.%f"), freq='50 ms')
            rng = rng[:-1]      # drop last entry, i.e. entry of enddate

            # now create an empty dataframe with nan values to be filled latter with potential data
            data_frame_final = pd.DataFrame(index=rng, columns=['time', 'co2conc', 'h2oconc', 'vx', 'vy', 'vz', 'ftemp', 'wa'])
            data_frame_final = data_frame_final.fillna(0)  # with 0s rather than NaNs

            # here we need to load the corresponding concentration and wind data and insert it accordingly
            err_conc, df_conc = self.locate_conc_file(conc_dir, d)
            err_wind, df_wind = self.locate_wind_file(wind_dir, d)

            # since milisecond timestamp is missing in concentration file, we need to add it (to make index unique and enable reindexing)
            if err_conc != 0:
                print("\tProcessing concentration file...")
                df_conc = df_conc.set_index(pd.DatetimeIndex(df_conc['time']))
                df0 = pd.DataFrame(df_conc).set_index(df_conc.groupby(['time']).cumcount(), append=True)
                df0.reset_index(level=1, inplace=True)
                df0.columns = ['ind', 'time', 'co2conc', 'h2oconc']
                try:
                    df0['time'] = map(lambda x, t: self.date_parser(datetime.datetime.strftime(x + datetime.timedelta(microseconds=int(t)*50*1000), '%Y-%m-%d %H:%M:%S.%f'), True), df0['time'], df0['ind'])
                    df0 = df0.set_index(pd.DatetimeIndex(df0['time']))
                    del df0['ind']
                    df0.drop_duplicates(keep='first', inplace=True)
                    df_reindexed = df0.reindex(rng, method='nearest')
                    del df_reindexed['time']
                    data_frame_final.update(df_reindexed)
                except ValueError:
                    print("Error encountered.. skip")

            if err_wind != 0:
                print("\tProcessing wind file...")
                df_wind = df_wind.set_index(pd.DatetimeIndex(df_wind['time']))
                # should already be indexed in correct way, but just to make sure
                df_reindexed = df_wind.reindex(rng, method='nearest')
                df_reindexed = df_wind.set_index(pd.DatetimeIndex(df_reindexed['time']))
                del df_reindexed['time']
                data_frame_final.update(df_reindexed)

            # now it is time to write this datafile to a folder
            # create output path and write file
            path = out_dir + "\DAT_" + d.strftime("%Y-%m-%d") + "_00-00.txt"
            print("Writing file to " + path)
            del data_frame_final['time']
            data_frame_final.to_csv(path, sep=',', index=True, header=['co2conc', 'h2oconc', 'v_x (m/s)', 'v_y (m/s)', 'v_z (m/s)', 'ftemp', 'wa'],
                                   index_label='time')

            stop = time.time()
            time_per_day = stop-start
            rem_time = (total_days-(d-ds).days)*time_per_day
            m, s = divmod(rem_time, 60)
            h, m = divmod(m, 60)
            print("\t\tEstimated time remaining: %d:%02d:%02d" % (h, m, s))

            d = df  # now move one day ahead

    def date_parser(self, s, microsecs):
        """ this is a much faster parser tailored for the specific data 
        It distinguished between dates with microsecond timestamp (wind) and those without (concentration)
        """
        if microsecs:
            return datetime.datetime(year=int(s[0:4]), month=int(s[5:7]), day=int(s[8:10]), hour=int(s[11:13]),
                                     minute=int(s[14:16]), second=int(s[17:19]), microsecond=1000*int(s[20:23]))
        else:
            return datetime.datetime(year=int(s[0:4]), month=int(s[5:7]), day=int(s[8:10]), hour=int(s[11:13]),
                                     minute=int(s[14:16]), second=int(s[17:19]))

    def locate_conc_file(self, conc_dir, date):
        """
        locates the concentration file on given date and loads its into data frame
        :param conc_dir: input directory where concentration data is kept
        :param date: date from which to load concentration data
        :return: dataframe with concentration data
        """
        path = conc_dir + '\CONC_' + date.strftime("%Y-%m-%d") + '.txt'
        df = 0
        if os.path.exists(path):
            # open file as read only
            print("\tLoading file " + str(path))
            f = open(path, "r")
            # converter to strip characters in front of time stamp string and efficient parsing into datetime
            convertfunc = lambda x: self.date_parser(x, False)
            try:
                df = pd.read_table(f, parse_dates=True, names=['time','co2_conc','h2oconc'], infer_datetime_format=True,
                                   sep=',', header=0, usecols=[0, 1, 2], converters={0:convertfunc})
            except ValueError:
                return 0, df
            return 1, df
        else:
            return 0, df

    def locate_wind_file(self, wind_dir, date):
        """
        locates the wind file on given date and loads its into data frame
        :param conc_dir: input directory where wind data is kept
        :param date: date from which to load wind data
        :return: dataframe with wind data
        """
        path = wind_dir + '\WIND_' + date.strftime("%Y-%m-%d") + '.txt'
        df = 0
        if os.path.exists(path):
            print("\tLoading file " + str(path))
            # open file as read only
            f = open(path, "r")
            # converter to strip characters in front of time stamp string and efficient parsing into datetime
            convertfunc = lambda x: self.date_parser(x, True)
            try:
                df = pd.read_table(f, parse_dates=True, names=['time', 'vx', 'vy', 'vz', 'ftemp', 'wa'], infer_datetime_format=True,
                                   sep=',', header=0, usecols=[0, 1, 2, 3, 4, 5], converters={0:convertfunc})
            except ValueError:
                return 0, df
            return 1, df
        else:
            return 0, df

    def create_hdf5(self, input_dir, output_dir):
        """ crawl through data folder and write all the data to an hdf5 file """
        print("Rewrite Data to HDF5 file.")
        # populate list with the paths
        # populate list with the paths
        data_paths = []
        start_date = []
        # read paths and sort according to data
        print("\t ... read file paths and sort them chronologically")
        for f in os.listdir(input_dir):
            if f.startswith("DAT") and f.endswith(".txt"):
                path = str(os.path.join(input_dir, f))
                data_paths.append(path)
                df = pd.read_table(path, names=['date'], parse_dates=True, infer_datetime_format=True, sep='\t',
                                   header=0, usecols=[0], nrows=1)
                start_date.append(df.iloc[0]['date'])

        # now order paths according to date_list
        st = sorted(zip(start_date, data_paths))
        data_paths = [c for (s, c) in st]

        print("\t ... load them one by one and append them to HDF5 file")
        tot = len(data_paths)
        output_path = output_dir + "\\" + "COMBINED_DAT.hdf5"
        for idx, p in enumerate(data_paths):
            print("\t\t processing file "+  str(idx) + "/" + str(tot))
            df = pd.read_table(p, names=['time', 'co2conc', 'h2oconc', 'vx', 'vy',  'vz'], sep=',', header=0)
            df.to_hdf(output_path, 'data', mode='a', complevel=9, complib='zlib')

    def rewrite_for_eddy_pro(self, input_dir, output_dir, st_date, ed_date):
        """ function required to write data in a format which is suitable for eddy pro
        i.e. split data into hourly files and remove time column
        @:param st_date     start date from which it should reprocess
        @:param ed_date     end date to which it should reprocess
        """

        print("Rewrite Data to Eddy Pro Compatible Format.")
        # populate list with the paths
        # populate list with the paths
        data_paths = []
        start_date = []
        # read paths and sort according to data
        print("\t ... read file paths and sort them chronologically")
        for f in os.listdir(input_dir):
            if f.startswith("DAT") and f.endswith(".txt"):
                path = str(os.path.join(input_dir, f))
                data_paths.append(path)
                df = pd.read_table(path, names=['date'], parse_dates=True, infer_datetime_format=True, sep=',',
                                   header=0, usecols=[0], nrows=1)
                start_date.append(df.iloc[0]['date'])

        # now order paths according to date_list
        date_list = start_date
        st = sorted(zip(start_date, data_paths))
        data_paths = [c for (s, c) in st]
        date_list.sort()
        dates_list = [datetime.datetime.strptime(date, '%Y-%m-%d %H:%M:%S.%f') for date in date_list]

        # now need to restrict to data paths as specified by st_date, ed_date
        st = datetime.datetime.strptime(st_date, "%Y-%m-%d %H:%M:%S.%f")
        # not to exclude starting date
        st = st - datetime.timedelta(hours=1)
        ed = datetime.datetime.strptime(ed_date, "%Y-%m-%d %H:%M:%S.%f")
        lower = bisect.bisect_right(dates_list, st)
        upper = bisect.bisect_left(dates_list, ed)
        date_list = date_list[lower:upper]
        data_paths = data_paths[lower:upper]

        # load data and split according to hours, then save as new file without time column
        print("\t ... load them one by one and split into hourly files")
        tot = len(data_paths)
        output_path = output_dir + "\\" + "COMBINED_DAT.hdf5"

        # converter function for date parsing
        convertfunc = lambda x: self.date_parser(x, True)
        for idx, p in enumerate(data_paths):
            st = time.time()
            print("\t\t processing file " + str(idx) + "/" + str(tot))
            print("\t\t\t ... loading file")

            df = pd.read_table(p, names=['time', 'co2conc', 'h2oconc', 'vx', 'vy', 'vz', 'ftemp', 'wa'], sep=',', header=0, converters={0:convertfunc})
            df = df.set_index(pd.DatetimeIndex(df['time']))
            # now remove time column since time is the index
            del df['time']
            # create datetime object from string , then extract year, month, day information
            dt = datetime.datetime.strptime(date_list[idx], '%Y-%m-%d %H:%M:%S.%f')
            path = output_dir + '\\DAT_' + dt.strftime("%Y-%m-%d") + "_"

            print("\t\t\t ... splitting file into hourly intervals")
            start_time = dt.strftime("%Y-%m-%d") + " 00:00:00"
            start_dt = datetime.datetime.strptime(start_time, '%Y-%m-%d %H:%M:%S')
            for h in range(24):
                end_dt = start_dt + datetime.timedelta(hours=1)
                end_time = end_dt.strftime("%H:%M:%S")
                start_time = start_dt.strftime("%H:%M:%S")
                selected = df.between_time(start_time, end_time, include_start=True, include_end=False)

                path_act = path + start_dt.strftime("%H-%M") + ".txt"
                selected.to_csv(path_act, sep=',', index=False, header=['co2conc', 'h2oconc', 'v_x (m/s)', 'v_y (m/s)', 'v_z (m/s)', 'ftemp',
                                                'wa'])
                # now update start_dt to end_dt
                start_dt = end_dt
            sp = time.time()
            secs = (tot - idx) * (sp - st)
            m, s = divmod(secs, 60)
            h, m = divmod(m, 60)
            print("\t Estimated time remaining: " + str(h) + ":" + str(m) + ":" + str(s))

    def resample_for_eddy_pro(self, input_dir, output_dir, st_date, ed_date):
        """ helper method to reample eddy pro data of specific date range with a set of different sampling rates"""
        samplings = [50]

        for s in samplings:
            print ("\t Sampling at rate " + str(s))
            # create new directory for every freshly sampled set
            file_path = output_dir + '\\resampled_' + str(s) + '\\'
            directory = os.path.dirname(file_path)
            if not os.path.exists(directory):
                os.makedirs(directory)
            self.rewrite_for_eddy_pro_resampled(input_dir, file_path, st_date, ed_date, s)

    def rewrite_for_eddy_pro_resampled(self, input_dir, output_dir, st_date, ed_date, sampling):
        """ function required to write data in a format which is suitable for eddy pro

        This methods serves to investigate the influence of different sampling rates on the flux results
        It thus offers the possibility to resample the data

        i.e. split data into hourly files and remove time column
        @:param st_date     start date from which it should reprocess
        @:param ed_date     end date to which it should reprocess
        @:param sampling    sampling time step for the written data
        """

        print("Rewrite Data to Eddy Pro Compatible Format in resampled format.")
        # populate list with the paths
        # populate list with the paths
        data_paths = []
        start_date = []
        # read paths and sort according to data
        print("\t ... read file paths and sort them chronologically")
        for f in os.listdir(input_dir):
            if f.startswith("DAT") and f.endswith(".txt"):
                path = str(os.path.join(input_dir, f))
                data_paths.append(path)
                df = pd.read_table(path, names=['date'], parse_dates=True, infer_datetime_format=True, sep=',',
                                   header=0, usecols=[0], nrows=1)
                start_date.append(df.iloc[0]['date'])

        # now order paths according to date_list
        date_list = start_date
        st = sorted(zip(start_date, data_paths))
        data_paths = [c for (s, c) in st]
        date_list.sort()
        dates_list = [datetime.datetime.strptime(date, '%Y-%m-%d %H:%M:%S.%f') for date in date_list]

        # now need to restrict to data paths as specified by st_date, ed_date
        st = datetime.datetime.strptime(st_date, "%Y-%m-%d %H:%M:%S.%f")
        # not to exclude starting date
        st = st - datetime.timedelta(hours=1)
        ed = datetime.datetime.strptime(ed_date, "%Y-%m-%d %H:%M:%S.%f")
        lower = bisect.bisect_right(dates_list, st)
        upper = bisect.bisect_left(dates_list, ed)
        date_list = date_list[lower:upper]
        data_paths = data_paths[lower:upper]

        # load data and split according to hours, then save as new file without time column
        print("\t ... load them one by one and split into hourly files")
        tot = len(data_paths)

        # convert the sampling into an averaging time
        # initial acquisition freq is 20 Hz, i.e. 50 ms -> thus 10 Hz results in av of
        # 1 Hz is av of 20


        # converter function for date parsing
        convertfunc = lambda x: self.date_parser(x, True)
        for idx, p in enumerate(data_paths):
            st = time.time()
            print("\t\t processing file " + str(idx) + "/" + str(tot))
            print("\t\t\t ... loading file")

            df = pd.read_table(p, names=['time', 'co2conc', 'h2oconc', 'vx', 'vy', 'vz', 'ftemp', 'wa'], sep=',', header=0, converters={0:convertfunc})
            df = df.set_index(pd.DatetimeIndex(df['time']))
            # now remove time column since time is the index
            del df['time']

            # now resample the data onto the specified frequency
            sampling_step = str(sampling) + 'L'
            # take the mean of all values in between
            #resampled = df.resample(sampling_step).mean()
            # take the instant value (drop other values in between)
            resampled = df.resample(sampling_step).asfreq()


            # create datetime object from string , then extract year, month, day information
            dt = datetime.datetime.strptime(date_list[idx], '%Y-%m-%d %H:%M:%S.%f')
            path = output_dir + '\\DAT_' + dt.strftime("%Y-%m-%d") + "_"

            print("\t\t\t ... splitting file into hourly intervals")
            start_time = dt.strftime("%Y-%m-%d") + " 00:00:00"
            start_dt = datetime.datetime.strptime(start_time, '%Y-%m-%d %H:%M:%S')
            for h in range(24):
                end_dt = start_dt + datetime.timedelta(hours=1)
                end_time = end_dt.strftime("%H:%M:%S")
                start_time = start_dt.strftime("%H:%M:%S")
                selected = resampled.between_time(start_time, end_time, include_start=True, include_end=False)

                path_act = path + start_dt.strftime("%H-%M") + ".txt"
                selected.to_csv(path_act, sep=',', index=False, header=['co2conc', 'h2oconc', 'v_x (m/s)', 'v_y (m/s)', 'v_z (m/s)', 'ftemp',
                                                'wa'])
                # now update start_dt to end_dt
                start_dt = end_dt
            sp = time.time()
            secs = (tot - idx) * (sp - st)
            m, s = divmod(secs, 60)
            h, m = divmod(m, 60)
            print("\t Estimated time remaining: " + str(h) + ":" + str(m) + ":" + str(s))


    def rewrite_meteo_data(self, dir_in, dir_out):
        """ loads meteo data and writes it in biomet format which can be read by Eddy Pro"""

        # get paths in chronological order
        #st = time.time()

        rfd = ReformatData.ReformatData()
        lst = [4, 5, 6, 7, 8, 9, 10, 11, 12]
        count = 1
        for i in lst:
            rfd.reformat_meteo(dir_in, dir_out, 2016, i, count)
            count +=1
        lst2 = [1, 2, 3]
        for i in lst2:
            rfd.reformat_meteo(dir_in, dir_out, 2017, i, count)



        # load files in chronological order and sort them into full days, then write in correct format




if __name__ == "__main__":
    rw_data = RewriteData()

