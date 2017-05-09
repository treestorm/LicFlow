
import pandas as pd
import matplotlib.pyplot as plt

class DisplayData:

    def __init__(self):
        """ provides functions to extract and to plot data from eddy pro output files"""

        DIR = 'C:\\Users\\herrmann\\Documents\\Herrmann Lars\\Licor\\corrected\\eddy_out\\processed_data_run1\\2016-04'
        FILE = 'eddypro_LH_essentials_2017-05-09T134926_exp.csv'
        self.extract_eddy_data(DIR, FILE)

    def extract_eddy_data(self, dir, filename):
        """ extractes co2 and h2o flux data from eddy pro files and plots it"""
        path = dir + '\\' + filename
        df = pd.read_csv(path, sep=',', parse_dates=[['date', 'time']], names=['date', 'time', 'co2flow', 'h2oflow'], usecols=[1,2, 12, 14], header=0)
        df.plot(x='date_time', y='co2flow')
        plt.show()

if __name__ == "__main__":
    rw_data = DisplayData()