import numpy as np

import pandas as pd
import os
import csv

import matplotlib.pyplot as plt


class QualityTests:
    """ provides methods to perform custom steady state test (in order to determine optimum averaging length) as well
    as covariance analysis methods to find the optimal time lag """

    def __init__(self):
        basedir = 'C:\Users\herrmann\Documents\Herrmann Lars\Licor\corrected\eddy'
        file = 'DAT_2016-07-08_16-00.txt'

        # ---------- STEADY STATE TEST -----------------------
        #self.steady_state_test_single(basedir + '\\' + file
        #self.steady_state_test_entire_set(basedir)

        # ---------- TIME LAG ANALYSIS -----------------------
        #self.plot_covariance_vs_timelag_single(5)
        #self.plot_heatmap_covariance_complete()
        #self.plot_histogram_covariance_complete()
        #self.compute_covariance_complete(basedir, 'covariance_h2o.txt', True)


    def plot_covariance_vs_timelag_single(self, i):
        """ plots covariance of i-th processd file as a function of time lag"""
        path = './covariance_h2o.txt'
        with open(path) as f:
            ncols = len(f.readline().split(','))

        df = pd.read_csv(path, sep=',', usecols=range(1, ncols), header=0, skiprows=0)
        df.dropna(inplace=True)
        lst = df.iloc[i].tolist()
        time_list = np.linspace(-100, 100, 400)
        plt.plot(time_list, lst)
        plt.show()

    def plot_heatmap_covariance_complete(self):
        """ creates a heatmap of the covariance results  """
        # if True plot co2 flux, else h2o flux covariance data
        co2analysis = True
        if co2analysis:
            path = './covariance_co2.txt'
        else:
            path = './covariance_h2o.txt'
        with open(path) as f:
            ncols = len(f.readline().split(','))

        df = pd.read_csv(path, sep=',', usecols=range(1, ncols), header=0, skiprows=0)
        df[df==0] = np.nan
        df.dropna(inplace=True)
        x0 = np.linspace(-100, 100, 400)
        x0 = x0.tolist()

        y = []
        x = []
        if not co2analysis:
            for i in range(1, 7000):
                if df.iloc[i].abs().max() > 0 and df.iloc[i].abs().max() < 1.0:
                    y = y + df.iloc[i].abs().tolist()
                    x = x + x0
        else:
            for i in range(1, 7700):
                if df.iloc[i].abs().max() > 0.2 and df.iloc[i].abs().max() < 1.0:
                    y = y + df.iloc[i].abs().tolist()
                    x = x + x0

        heatmap, xedges, yedges = np.histogram2d(x, y, bins=50)
        extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]
        plt.figure(figsize=(5, 5))
        plt.clf()
        plt.xlim([-100, 100])
        plt.ylim([0, 0.05])
        plt.xlabel('Time lag (s)')
        plt.ylabel('Covariance')
        plt.imshow(heatmap.T, extent=extent, origin='lower', aspect=4000, cmap='inferno', vmin=100, vmax=750, interpolation='gaussian')
        plt.grid(True, color='white')

        plt.colorbar()
        plt.tight_layout()

        outputpath = 'C:\Users\herrmann\Documents\Herrmann Lars\Documentation\\figs\\'
        if co2analysis:
            filename = 'time_lag_heatmap_zoom_co2.png'
        else:
            filename = 'time_lag_heatmap_zoom_h2o.png'
        plt.savefig(outputpath + filename, dpi=300)
        plt.show()

    def plot_timelag_variation_histogram_co2(self):
        """ creates a histogram of the determined maxima as a function of time lag
        for the co2 data """
        path = './covariance_co2.txt'
        with open(path) as f:
            ncols = len(f.readline().split(','))

        df = pd.read_csv(path, sep=',', usecols=range(1, ncols), header=0, skiprows=0)
        df[df == 0] = np.nan
        df.dropna(inplace=True)
        timeline = np.linspace(-100, 100, 400)

        maxs = []
        mu_lst = []
        sig_lst = []
        x = range(0,12)
        for i in x:
            for j in range(i*600, (i+1)*600):
                if df.iloc[j].abs().max() > 0.2 and df.iloc[j].abs().max() < 1.0:
                    dft = df.iloc[j].abs().tolist()
                    # print dft
                    maxs.append(timeline[dft.index(max(dft))])
            # add a 'best fit' line
            # best fit of data
            from scipy.stats import norm
            rng = []
            mi = 50
            ma = 70
            for i in maxs:
                if ma >= i >= mi:
                    rng.append(i)
            (mu, sigma) = norm.fit(rng)
            mu_lst.append(mu)
            sig_lst.append(sigma)

        fig = plt.figure(figsize=(5, 3))
        x = [i+1 for i in x]
        print mu_lst
        plt.errorbar(x, mu_lst, xerr=0, yerr=sig_lst, color='royalblue')
        plt.xlabel('months')
        plt.ylabel('average time lag (s)')
        plt.ylim([50, 70])
        plt.xlim([0, 13])
        plt.tight_layout()

        outputpath = 'C:\Users\herrmann\Documents\Herrmann Lars\Documentation\\figs\\'
        filename = 'time_lag_histos_time_var.png'
        plt.savefig(outputpath + filename, dpi=300)
        plt.show()

    def plot_timelag_variation_histogram_h2o(self):
        """ creates a histogram of the determined maxima as a function of timelag
         for the h2o data """
        path = './covariance_h2o.txt'
        with open(path) as f:
            ncols = len(f.readline().split(','))

        df = pd.read_csv(path, sep=',', usecols=range(1, ncols), header=0, skiprows=0)
        df[df == 0] = np.nan
        df.dropna(inplace=True)
        timeline = np.linspace(-100, 100, 400)

        maxs = []
        # h2o conc
        for i in range(1, 7000):
            if df.iloc[i].abs().max() > 0.01 and df.iloc[i].abs().max() < 1.0:
                dft = df.iloc[i].abs().tolist()
                # print dft
                maxs.append(timeline[dft.index(max(dft))])

        fig = plt.figure(figsize=(5, 3))
        import matplotlib.mlab as mlab
        # the histogram of the data
        n, bins, patches = plt.hist(maxs, 100, normed=1, facecolor='seagreen', alpha=0.75)

        # add a 'best fit' line
        from scipy.stats import norm
        rng = []
        mi = 50
        ma = 70
        for i in maxs:
            if ma >= i >= mi:
                rng.append(i)

        (mu, sigma) = norm.fit(rng)
        print mu
        print sigma
        x = np.linspace(min(maxs), max(maxs), 300)
        plt.plot(x, mlab.normpdf(x, mu, sigma)/2.3, color='red', linewidth=2.0, alpha=0.5)
        plt.xlabel('time lag (s)')
        plt.ylabel('relative frequency')
        plt.tight_layout()

        outputpath = 'C:\Users\herrmann\Documents\Herrmann Lars\Documentation\\figs\\'
        filename = 'time_lag_histos_h2o.png'
        plt.savefig(outputpath + filename, dpi=300)
        plt.show()

    def compute_covariance_complete(self, basedir, outputname, flag):
        """ calculate the covariance between co2 conc and vertical wind speed for different time lags for the 
        entire dataset and saves the results to a file for later plotting
        
        @:param basedir             the base directory containing all the raw data 
        @:param outputname          outputname of file for covariance analysis
        @:param flag                Analyze CO2 (False) or H2O (True) covariance
        """
        # obtain all paths of the files
        paths, dates = self.get_eddy_paths(basedir)

        # iterate over paths and perform steady state test on each file
        tot = len(paths)
        for idx, p in enumerate(paths):
            print("Processing file " + str(idx) + "/" + str(tot))
            res = self.get_covariance_time_lag(p, flag)

            # now write calculated results as a line to data file
            output = open(outputname, 'a')
            writer = csv.writer(output, delimiter=",")
            writer.writerow([dates[idx]] + res)
            output.close()

    def get_covariance_time_lag(self, path, flag):
        """ calculates the covariance between co2/h2o concentration data and vertical wind speed for different offsets
         
        @:param path    the path of the file to analyze 
        @:param flag    flag to mark whether to analyze co2 (False) or h2o (True) conc
        """
        if flag:
            df = pd.read_csv(path, sep=',', names=['co', 'w'], usecols=[1, 4], header=0)
        elif not flag:
            df = pd.read_csv(path, sep=',', names=['co', 'w'], usecols=[0, 4], header=0)
        else:
            print("Incompatible flag specified.")
            return 1
        df_orig = df.copy()
        df_temp = df.copy()

        # shift in seconds
        shift = 100

        cov_res = []
        steps = np.linspace(0, int(2*shift*20), shift*4)
        for i in steps:
            df_temp.w = df_orig.w.shift(i)
            rs = df_temp.cov()
            cov_res.append(rs['w'][0])
        return cov_res

    def compute_covariance_single(self, basedir, file):
        """ calculates the covariance between co2 concentration data and vertical wind speed for different offsets"""
        path = basedir + '\\' + file
        df = pd.read_csv(path, sep=',', names=['co' , 'w'], usecols=[0, 4], header=0)
        df_orig = df.copy()
        df_temp = df.copy()
        # shift in seconds
        shift = 300

        cov_res = []
        for i in range(-shift*20, shift*20):
            df_temp.w = df_orig.w.shift(i)
            rs = df_temp.cov()
            cov_res.append(rs['w'][0])
        tshift = np.linspace(-shift, shift, 2*shift*20)
        plt.figure(figsize=(7,4))
        plt.plot(tshift, cov_res, color='royalblue')
        plt.xlabel('time lag (s)')
        plt.suptitle(file)
        plt.ylabel('covariance')
        plt.tight_layout()
        outputpath = 'C:\Users\herrmann\Documents\Herrmann Lars\Documentation\\figs\\'
        filename = 'cov_co_w_wide.png'
        plt.savefig(outputpath + filename, dpi=300)
        plt.show()

    def plot_steady_state_test_single(self, path):
        """
        Perform steady state test on single file and plot the result
        :return: 
        """
        ints = [1, 2, 3, 4, 5, 10, 15, 20, 30, 60]
        res = []
        for i in ints:
            t = self.get_steady_state_test_result(path, i, 6)
            res.append(100 * t)

        plt.figure(figsize=(6, 4))
        plt.plot(ints, res, color='royalblue', marker='.')
        plt.ylabel('standard deviation (%)')
        plt.xlabel('interval size (min)')
        plt.suptitle('Steady state test for DAT_2016-07-08_16-00.txt')
        plt.tight_layout()

        outputpath = 'C:\Users\herrmann\Documents\Herrmann Lars\Documentation\\figs\\'
        filename = 'steady_state_test_single02.png'
        plt.savefig(outputpath + filename, dpi=300)

        plt.show()

    def get_eddy_paths(self, basedir):
        """  this iterates through all files in folder and extracts the timestamp of the first entry """
        # populate list with the paths
        paths = []
        start_date = []
        for f in os.listdir(basedir):
            if f.startswith("DAT_"):
                path = str(os.path.join(basedir, f))
                paths.append(path)
                d = f.split("_")
                start_date.append(d[1] + ' ' + d[2].split(".")[0])

        # now order conc_path according to date_list
        st = sorted(zip(start_date, paths))
        paths = [c for (s, c) in st]
        dates = [s for (s, c) in st]
        return paths, dates


    def compute_steady_state_test_complete(self, dir):
        """ performs the steady state tests of all the files in the directory dir and saves the results to a text file"""
        # obtain all paths of the files
        paths, dates = self.get_eddy_paths(dir)

        # save results to file
        output = open('steady_state_results.txt', 'w')
        writer = csv.writer(output, delimiter=",")

        # define the intervals to process
        ints = [1, 2, 3, 4, 5, 10, 15, 20, 30, 60]

        writer.writerow(['time stamp'] + ints)
        output.close()

        # iterate over paths and perform steady state test on each file
        tot = len(paths)
        for idx, p in enumerate(paths):
            print("Processing file " + str(idx) + "/" + str(tot))
            res = []
            for i in ints:
                t = self.get_steady_state_test_result(p, i, 6)
                res.append(100 * t)

            # now write calculated results as a line to data file
            output = open('steady_state_results.txt', 'a')
            writer = csv.writer(output, delimiter=",")
            writer.writerow([dates[idx]] + res)
            output.close()


    def get_steady_state_test_result(self, path, interval, subdivision):
        """ performs the steady state test on the dataset
        @:param interval        the entire interval to perform the test over in minutes
        @:param subdivision     number of parts to divide the interval into (should be a number x with 60%x = 0
        """
        df = pd.read_csv(path, sep=',', names=['u', 'v', 'w'], usecols=[2, 3, 4], header=0)
        M = len(df)

        u = np.array(df['u'])
        v = np.array(df['v'])
        w = np.array(df['w'])

        # need to only maintain the part specified in interval
        num_ints = 60/interval
        s = 60*60*20/num_ints # length of subintervals

        sdev_sum = 0
        for i in range(0, num_ints):
            ut = u[i*s:(i+1)*s]
            vt = v[i*s:(i+1)*s]
            wt = w[i*s:(i+1)*s]
            sdev_sum += self.perform_steady_state(ut, vt, wt, subdivision)
        return sdev_sum/num_ints

    def plot_steady_state_test_results(self):
        """ 
        load processed file from steady state analysis and display data 
        """
        path = './steady_state_results.txt'
        ints = ['i1', 'i2', 'i3', 'i4', 'i5', 'i10', 'i15', 'i20', 'i30', 'i60']
        df = pd.read_csv(path, sep=',', parse_dates=['date_time'], infer_datetime_format=True,
                         names=['date_time', 'i1', 'i2', 'i3', 'i4', 'i5', 'i10', 'i15', 'i20', 'i30', 'i60'], usecols=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10], header=0)
        df.dropna(inplace=True)

        fig, ax = plt.subplots(figsize=(17, 4))
        df.plot(x='date_time', y='i4', marker='.', color='royalblue', linewidth=0, figsize=(15,5), ax=ax)
        df.plot(x='date_time', y='i30', marker='.', color='darkgoldenrod', linewidth=0, figsize=(15, 5), ax=ax)
        ax.set_ylabel('deviation $\delta_\mathrm{cov}$ (%)')
        ax.set_xlabel("time")
        ax.set_ylim([-10,300])
        ax.legend(['4 min', '30 min'], loc='upper right', frameon=True, prop={'size': 10})
        plt.tight_layout()
        outputpath = 'C:\Users\herrmann\Documents\Herrmann Lars\Documentation\\figs\\'
        filename = 'steady_state_plot_year.png'
        plt.savefig(outputpath + filename, dpi=300)
        plt.show()


    def print_steady_state_test_results(self):
        """ 
        load processed file from steady state analysis and calculates statistics
        """
        path = './steady_state_results.txt'
        ints = ['i1', 'i2', 'i3', 'i4', 'i5', 'i10', 'i15', 'i20', 'i30', 'i60']
        df = pd.read_csv(path, sep=',', parse_dates=True, infer_datetime_format=True,
                         names=['date_time', 'i1', 'i2', 'i3', 'i4', 'i5', 'i10', 'i15', 'i20', 'i30', 'i60'], usecols=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10], header=0)
        df.plot(x='date_time', y='i1')

        df.dropna(inplace=True)

        for i in ints:
            cw_bins = [0.0, 30.0, 100.0, 10000000.0]
            out = pd.cut(df[i], bins =cw_bins)
            counts = pd.value_counts(out)
            print(i)
            for a in counts:
                print("\t" + str(100*a/len(df)))


    def perform_steady_state(self, u, v, w, subdivision):
        """ performs the steady state test on the delivered u, v, w wind vectors with subdivisions specifying into how many 
        parts the values are split """

        # the norm of the vectorially added horizontal wind speeds
        # but sorted for quadrants

        h = np.sqrt(np.square(u) + np.square(v))

        hc = self.divide_list(h, subdivision)
        wc = self.divide_list(w, subdivision)

        # now iterate over the subdivisions and compute for each the covariance
        conv_list = []  # contains the covariances of each subinterval
        for idx, hp in enumerate(hc):
            wp = wc[idx]
            n = len(wp)

            sh = np.sum(hp)
            sw = np.sum(wp)

            av = (1.0 / n) * sh * sw

            sum = 0
            for edx, i in enumerate(hp):
                sum += i * wp[edx]
            # covariance value of subinterval (number idx)
            cov_t = (1.0 / (n - 1)) * (sum - av)
            conv_list.append(cov_t)

        # now take the mean of the covariance values in conv_list
        m_int = np.mean(conv_list)

        # now calculate covariance of entire interval
        sum = 0
        for idx, i in enumerate(w):
            sum += i * h[idx]
        sepsum = (1.0 / (subdivision * n)) * np.sum(w) * np.sum(h)
        m_tot = 1.0 / (subdivision * (n - 1)) * (sum - sepsum)

        # obtain the deviation between the two values
        dev = np.abs((m_int - m_tot) / m_tot)
        return dev

    def divide_list(self, seq, num):
        """ divides list into a sequence of lists"""
        avg = len(seq) / float(num)
        out = []
        last = 0.0

        while last < len(seq):
            out.append(seq[int(last):int(last + avg)])
            last += avg
        return out

if __name__ == "__main__":
    quality = QualityTests()
