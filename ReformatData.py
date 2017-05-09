import os
import pandas as pd
import ReadData

"""
    File name: ReformatData.py
    Author: treestorm
    Date created: 09/05/2017
    Python Version: 2.7
"""

class ReformatData:
    def __init__(self):
        # initialise the read data class
        print("Initialising read data class")
        self.read_data = ReadData.ReadData()

        DIR_WIND = "C:\Users\herrmann\Documents\Herrmann Lars\Licor\wind"
        ODIR_WIND =  "C:\Users\herrmann\Documents\Herrmann Lars\Licor\wind_corrected"

        DIR_CONC = "C:\Users\herrmann\Documents\Herrmann Lars\Licor\conc"
        ODIR_CONC = "C:\Users\herrmann\Documents\Herrmann Lars\Licor\conc_corrected"

        #self.reformat_conc_data(DIR_CONC, ODIR_CONC)
        for i in range(1,3):
            self.reformat_wind_data(DIR_WIND, ODIR_WIND, 2017, i)

    def reformat_conc_data(self, basedir_conc, outdir_conc):
        """
        
        :return: 
        """

        conc_paths = self.read_data.get_conc_paths_ordered2(basedir_conc)
        tot = len(conc_paths)
        for idx, i in enumerate(conc_paths):
            print("Processing file " + str(idx) + "/" + str(tot))
            err, df = self.read_data.read_concentration(i)
            if err == 0:    # error while loading the file
                fd = open(outdir_conc + '\corrupt_files.txt', 'a')
                fd.write(str(i) + "\r\n")
                fd.close()
            else:           # else write to file
                DFList = [group[1] for group in df.groupby(df.index.day)]
                # append each subgroup to corresponding file
                for k in DFList:
                    #k.columns(['time', 'conc'])
                    k.set_index(pd.DatetimeIndex(k['time']))
                    k.sort_values('time')
                    dat1 = k['time'].iloc[0]
                    # only write conc, time is now the index
                    df_to_write = k[['co2conc', 'h2oconc']]
                    # create output path and write file
                    path = outdir_conc + "\CONC_" + dat1.strftime("%Y-%m-%d") + ".txt"
                    # check if file already exists
                    print("Write to " + dat1.strftime("%Y-%m-%d") + ".txt")
                    if not os.path.isfile(path):
                        df_to_write.to_csv(path, sep=',', index=True, header=['co2_conc', 'h2oconc'], index_label='time')
                    else:  # else it exists so append without writing the header
                        df_to_write.to_csv(path, sep=',', index=True, header=False, index_label='time', mode='a')


    def reformat_wind_data(self, basedir_wind, outdir_wind, year, month):
        """
        This class obtains a list with all wind paths grouped by day (with one file overlap on each side)
        It then iterates over these day by day groups and on each day combines the data into one data frame
        It then filters the dataframe by day (hus cutting away the parts before midnight (previous day) and
        after the next midnight (next day)
        Thereafter it reindexes and nearest neighbour interpolates the file to a given 20 Hz timeline
        Last, it writes the data to a text file
        :return: 
        """

        # make new folder based on year, month
        outdir_wind = outdir_wind
        if not os.path.exists(outdir_wind):
            os.makedirs(outdir_wind)

        print("Read wind paths")
        [index_lists, wind_paths] = self.read_data.get_wind_paths_ordered(basedir_wind, year, month)
        print("Number of wind paths " + str(len(wind_paths)))

        # iterate over the number of day groups in the specified (year, month)
        for sub_list in index_lists:
            # load all files contained in this one day and append to one big data frame
            df = None
            isLoaded = True # loading status - if error is risen,this is set to False
            for i, ind in enumerate(sub_list):
                print("Reading sub file " + str(i+1) + "/" + str(len(sub_list)))
                print(wind_paths[ind])

                err, dfA = self.read_data.read_wind(wind_paths[ind])
                if err == 0:
                    # problem with file: - write to disk
                    fd = open(outdir_wind + '\corrupt_files_' + str(year) + '_' + str(month) + '.txt', 'a')
                    fd.write(str(wind_paths[ind]) + "\r\n")
                    fd.close()
                    isLoaded = False
                    break
                if df is None:
                    df = dfA
                else:
                    df = pd.concat([df, dfA])

            # only proceed if the entire subset constituting one day has been loaded successfully
            if isLoaded:
                # now sort this dataframe and extract subgroups based on date
                df.columns = ['time', 'wa', 'ftemp', 'vz', 'vy', 'vx']
                df = df.set_index(pd.DatetimeIndex(df['time']))
                # just to make sure they are in the right sequence
                df.sort_values('time')
                # filter data by day
                DFList = [group[1] for group in df.groupby(df.index.day)]
                # sort by size (the most dates are from desired day)
                DFList.sort(key=len, reverse=True)

                # should always contain three dates (previous day - desired day - next day)
                if len(DFList) == 3:
                    print("Writing to file... ")
                    # first entry to write, since sorted the list by length of the data frames
                    dfd = DFList[0]
                    dfd.set_index(pd.DatetimeIndex(dfd['time']))
                    dfd.sort_values('time')
                    # extract the first date to determine the output path
                    dat1 = dfd['time'].iloc[0]

                    # regroup and reindex data - i.e. create given timeline (20 Hz)
                    startdate = dat1.strftime("%Y-%m-%d") + " 00:00:00.000"
                    enddate = dat1.strftime("%Y-%m-%d") + " 23:59:59.950"
                    rng = pd.date_range(start=startdate, end=enddate, freq='50 ms')

                    # ensure there are no duplicate entries in the dataframe
                    dfd = dfd.drop_duplicates(keep='last')
                    # reindex the data according to the timeline
                    df_to_write = dfd.reindex(rng, method='nearest')
                    df_to_write.interpolate(method='linear')
                    df_to_write = df_to_write[['vx', 'vy', 'vz', 'ftemp', 'wa']]

                    # create output path and write file
                    path = outdir_wind + "\WIND_" + dat1.strftime("%Y-%m-%d") + ".txt"
                    df_to_write.to_csv(path, sep=',', index=True, header=['v_x (m/s)', 'v_y (m/s)', 'v_z (m/s)', 'ftemp', 'wa'], index_label='time')

                    # append length of dataset to file to check if it matches frequency
                    fd = open(outdir_wind + '\integrity_' + str(year) + '_' + str(month) +'.txt', 'a')
                    fd.write(str(len(df_to_write)) + "\r\n")
                    fd.close()

if __name__ == "__main__":
    cb_data = ReformatData()
