import numpy as np
import matplotlib.pyplot as plt


class ControlCalculations:
    """ provides methods to perform a number of control calculations such as e.g. footprint estimates
    """

    def __init__(self):
        #self.footprint_distance()
        self.footprint_vs_mean_windspeed()

    def footprint_distance(self):
        """ Estimates the cumulative normalized contribution to the flux based on Schuep et al. (1990)"""

        # INPUT PARAMETERS
        v_f = 0.4   # friction velocity (m/s)
        u = 2       # mean integrated wind speed (m/s)
        h = 2       # measurement height (m)  -  (Beromuenster 212 m)
        delta = 1   # zero plain displacement (m)
        k = 0.4     # von Karman constant

        # distance range
        d = np.linspace(1, 100000, 500000)

        res0 = self.calculate_cnf(d, v_f, u, h, delta, k)
        res0_rel = self.calculate_relative_contribution(d, v_f, u, h, delta, k)

        h=5
        res1 = self.calculate_cnf(d, v_f, u, h, delta, k)
        res1_rel = self.calculate_relative_contribution(d, v_f, u, h, delta, k)

        h = 200
        res2 = self.calculate_cnf(d, v_f, u, h, delta, k)
        res2_rel = self.calculate_relative_contribution(d, v_f, u, h, delta, k)

        # plotting
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10,3.4))

        ax1.set_xscale('log')
        ax1.set_yscale('log')
        ax1.fill_between(d, 0, res0_rel * 100, color='purple', alpha=.4, label='h=2 m', linewidth=2)
        ax1.fill_between(d, 0, res1_rel*100, color='royalblue', alpha=.4, label='h=5 m', linewidth=2)
        ax1.fill_between(d, 0, res2_rel*100, color='darkgoldenrod', alpha=.4, label='h=200 m', linewidth=2)
        ax1.set_xlim([1, 100000])
        ax1.set_ylim([0.0001, 5])
        ax1.legend(loc='upper right', shadow=False, frameon=False)
        ax1.set_ylabel('Relative contribution (%)')
        ax1.set_xlabel('Distance from tower (m)')

        ax2.set_xscale('log')
        ax2.fill_between(d, res1*100, res0 * 100, color='purple', alpha=.4, label='h=2 m', linewidth=2)
        ax2.fill_between(d, res2*100, res1*100, color='darkgoldenrod', alpha=.4, label='h=5 m', linewidth=2)
        ax2.fill_between(d, 0, res2*100, color='royalblue', alpha=.4, label='h=200 m', linewidth=2)
        ax2.set_ylim([0,100])
        ax2.set_xlim([0.1, 100000])
        ax2.legend(loc='upper left', shadow=False, frameon=False)
        ax2.set_ylabel('Cumulative contribution (%)')
        ax2.set_xlabel('Distance from tower (m)')

        plt.tight_layout()
        outputpath = 'C:\Users\herrmann\Documents\Herrmann Lars\Documentation\\figs\\'
        filename = 'footprint_estimate.png'
        plt.savefig(outputpath + filename, dpi=300)
        plt.show()

    def footprint_vs_mean_windspeed(self):
        """ plots the distance where 80% of cumulative normalized contribution to flux is reached vs the mean windspeed"""
        # change the font style and family

        # percentage conribution
        pt = 90

        # distance range
        d = np.linspace(1, 100000, 100000)

        # INPUT PARAMETERS
        v_f = 0.7  # friction velocity (m/s)
        u = 3.6  # mean integrated wind speed (m/s)
        h = 212  # measurement height (m)  -  (Beromuenster 212 m)
        delta = 1  # zero plain displacement (m)
        k = 0.4  # von Karman constant

        cum_dist_1 = []
        for u in range(0, 100):
            res0 = 100*self.calculate_cnf(d, v_f, u/10.0, h, delta, k)
            x_value = np.interp(pt, res0, d)
            ## print x_value
            cum_dist_1.append(x_value)

        cum_dist_2 = []
        x_vals = []
        h = 5
        for u in range(0, 100):
            res0 = 100 * self.calculate_cnf(d, v_f, u / 10.0, h, delta, k)
            x_value = np.interp(pt, res0, d)
            ## print x_value
            cum_dist_2.append(x_value)
            x_vals.append(u / 10.0)

        cum_dist_3 = []
        x_vals = []
        h = 212
        for u in range(0, 100):
            res0 = 100 * self.calculate_cnf(d, v_f, u / 10.0, h, delta, k)
            x_value = np.interp(pt, res0, d)
            ## print x_value
            cum_dist_3.append(x_value)
            x_vals.append(u/10.0)

        plt.figure(figsize=(5, 3.3))

        #plt.loglog(x_vals, cum_dist_1, color='purple', label='h=2 m')
        #plt.loglog(x_vals, cum_dist_2, color='darkgoldenrod', label='h=5 m')
        plt.loglog(x_vals, cum_dist_3, color='royalblue', label='h=200 m')
        plt.ylabel('distance $d_{90}$ (m)')
        plt.xlabel('mean wind speed (m/s)')
        plt.tight_layout()

        plt.legend(frameon=False, loc='lower right', prop={'size':10})

        outputpath = 'C:\Users\herrmann\Documents\Herrmann Lars\Documentation\\figs\\'
        filename = 'footprint_90percent.png'
        plt.savefig(outputpath + filename, dpi=300)
        plt.show()


    def calculate_cnf(self, d, v_f, u, h, delta, k):
        """ calculates the cumulative normalized contribution to flux based on the specified input parameters"""
        return np.exp(- u*(h-delta)/(v_f*k*d))

    def calculate_relative_contribution(self, d, v_f, u, h, delta, k):
        """ calculates the relative contribution to flux based on the specified input parameters"""
        return (u*(h-delta)/(v_f*k*d**2))*np.exp(- u*(h-delta)/(v_f*k*d))


if __name__ == "__main__":
    rw_data = ControlCalculations()
