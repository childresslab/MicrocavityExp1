
from qtpy import QtCore
from collections import OrderedDict
import datetime
import numpy as np
import os
from itertools import product
from time import sleep, time
import matplotlib.pyplot as plt
from io import BytesIO

from scipy import signal
from scipy.optimize import curve_fit
from logic.generic_logic import GenericLogic
from core.util.mutex import Mutex
from core.module import Connector, ConfigOption, StatusVar




class CavityLogic(GenericLogic):
    """
    This is the Logic class for cavity scanning.
    """
    _modclass = 'confocallogic'
    _modtype = 'logic'

    # declare connectors
    nicard = Connector(interface='ConfocalScannerInterface')
    scope = Connector(interface='scopeinterface')
    savelogic = Connector(interface='SaveLogic')


    # signals
    signal_start_scanning = QtCore.Signal(str)
    signal_continue_scanning = QtCore.Signal(str)
    signal_stop_scanning = QtCore.Signal()
    signal_scan_lines_next = QtCore.Signal()
    signal_xy_image_updated = QtCore.Signal()
    signal_depth_image_updated = QtCore.Signal()
    signal_change_position = QtCore.Signal(str)
    signal_xy_data_saved = QtCore.Signal()
    signal_depth_data_saved = QtCore.Signal()
    signal_tilt_correction_active = QtCore.Signal(bool)
    signal_tilt_correction_update = QtCore.Signal()
    signal_draw_figure_completed = QtCore.Signal()
    signal_position_changed = QtCore.Signal()

    sigImageXYInitialized = QtCore.Signal()
    sigImageDepthInitialized = QtCore.Signal()

    signal_history_event = QtCore.Signal()

    def __init__(self, config, **kwargs):
        super().__init__(config=config, **kwargs)

        #locking for thread safety
        self.threadlock = Mutex()

        # counter for scan_image
        self._full_sweep_freq = 2/3
        self._full_sweep_start = 0.0
        self._full_sweep_stop = -3.75
        self._acqusition_time = 2.0
        self.reflection_channel = 0
        self.ramp_channel = 1
        self.position_channel = 3
        self.velocity_channel = 2
        self.SG_scale = 10 # V
        self.lamb = 637e-9 # wavelenght

        self._current_filepath = r'C:\Users\ChildressLab\Desktop\Rasmus notes\Measurements'

    def on_activate(self):
        """ Initialisation performed during activation of the module.
        """
        self._ni = self.get_connector('nicard')
        self._scope = self.get_connector('scope')
        self._save_logic = self.get_connector('savelogic')

        self.cavity_range = self._ni._cavity_position_range[1] - self._ni._cavity_position_range[0]


    def on_deactivate(self):
        """ Reverse steps of activation

        @return int: error code (0:OK, -1:error)
        """

    def _load_full_sweep(self, filepath = None, filename = None):
        '''
        Loads data from full sweep

        :param filepath:
        :param filename:
        :return:
        '''
        delimiter = '\t'

        if filepath is None:
            filepath = self._current_filepath

        if filename is None:
            filename = self._current_filename

        with open(os.path.join(filepath, filename), 'rb') as file:
            data = np.loadtxt(file, delimiter=delimiter)

        self.time = data[0]
        self.volts = data[1:5]


    def _trim_data(self, time, volts):
        '''
        Trims data to the ramp

        :return:
        '''
        total_trace = time[-1]-time[0]  # sec
        ramp_period = 1.0 / self._full_sweep_freq # sec
        period_index = len(time) * ramp_period / total_trace
        ramp_mid = np.argmin(volts[self.ramp_channel])

        low_index = ramp_mid - int(period_index / 2)
        high_index = ramp_mid + int(period_index / 2)

        volts_trim = volts[:, low_index:high_index+1]
        time_trim = time[low_index:high_index+1]

        return time_trim, volts_trim

    def _data_split_up(self):
        [self.RampUp_time, self.RampDown_time] = np.array_split(self.time_trim, 2)
        [self.RampUp_signalR, self.RampDown_signalR] = np.array_split(self.volts_trim[self.reflection_channel], 2)
        [self.RampUp_signalNI, self.RampDown_signalNI] = np.array_split(self.volts_trim[self.ramp_channel], 2)
        [self.RampUp_signalSG, self.RampDown_signalSG] = np.array_split(self.volts_trim[self.position_channel], 2)
        [self.RampUp_signalSG_v, self.RampDown_signalSG_v] = np.array_split(self.volts_trim[self.velocity_channel],2)
        return 0

    def _get_ramp_up_signgals(self):
        self.time_trim, self.volts_trim = self._trim_data(self.time, self.volts)
        self._data_split_up()

        return 0


    def _polyfit_SG(self, order=2, plot=False):
        xdata = self.RampUp_time
        ydata = self.RampUp_signalSG
        p4 = np.poly1d(np.polyfit(xdata, ydata, order))
        self.RampUp_signalSG_polyfit = p4(xdata)

        if plot is True:
            plt.plot(xdata, ydata, '-', xdata, p4(xdata), '--')
            plt.show()

    def _fit_ramp(self):
        # Fitting setup
        parameter_guess = [self._full_sweep_start, self._full_sweep_stop,
                           self._full_sweep_freq, self.time_trim[0]]

        func = self._ni.sweep_function
        xdata = self.time_trim
        ydata = self.volts_trim[self.ramp_channel]

        # Actual fitting
        popt, pcov = curve_fit(func, xdata, ydata, parameter_guess)
        self.popt = popt


    def start_full_sweep(self):
        RepOfSweep = 1

        #set up sweep
        self._ni.set_up_sweep(self._full_sweep_start, self._full_sweep_stop, self._full_sweep_freq, RepOfSweep)
        self._scope.set_acquisition_time(self._acqusition_time)

        # HARD CODED!!!!! ARGHH!!!
        self._scope.set_vertical_scale(2, 500e-3)
        self._scope.set_vertical_position(2, 3500e-3)
        self._scope.set_vertical_scale(3, 5e-3)
        self._scope.set_vertical_position(3, 0)
        self._scope.set_vertical_scale(4, 2)
        self._scope.set_vertical_position(4, -2.5)

        #start sweep
        self._scope.stop_acquisition()
        #self._scope.set_egde_trigger(channel=2, level=-3.0)

        #
        self._scope.run_single()
        sleep(0.5)
        self._ni.start_sweep()


        #stop sweep
        sleep(self._acqusition_time)
        self._ni.stop_sweep()
        self._ni.close_sweep()

    def _get_scope_data(self):
        times, volts = self._scope.aquire_data()
        volts = volts.reshape(4, int(len(volts) / 4))
        times = times.reshape(4, int(len(times) / 4))
        time = times[0]

        self.volts = volts[:,2000:]
        self.time = time[2000:]

    def _linewidth_get_data(self):
        self._scope.run_single()

        nu_times, self.nu_volts = self._scope.aquire_data()
        nu_times = nu_times.reshape(4, int(len(nu_times) / 4))
        #self.nu_volts = nu_volts.reshape(4, int(len(nu_volts) / 4))
        self.nu_time = nu_times[0]

    def _save_raw_data(self):
        date = datetime.datetime.fromtimestamp(time()).strftime('%Y-%m-%d_%H%M%S')
        self._current_filename = date + '_full_sweep_data.dat'

        data = np.vstack([self.time, self.volts])

        fmt = '%.15e'
        header = ''
        delimiter = '\t'
        comments = '#'

        with open(os.path.join(self._current_filepath, self._current_filename), 'wb') as file:
            np.savetxt(file, data, fmt=fmt, delimiter=delimiter, header=header, comments=comments)

    def _linewidth_save_data(self):
        date = datetime.datetime.fromtimestamp(time()).strftime('%Y-%m-%d_%H%M%S')
        self._current_filename = date + 'linewidth_data.dat'

        data = np.hstack([self.nu_time_list, self.nu_volts_list])
        #data = data.reshape(5, int(len(data) / 5))
        fmt = '%.15e'
        header = ''
        delimiter = '\t'
        comments = '#'

        with open(os.path.join(self._current_filepath, self._current_filename), 'wb') as file:
            np.savetxt(file, data, fmt=fmt, delimiter=delimiter, header=header, comments=comments)

    def _detect_peaks(self, x, y=None, mph=None, mpd=1, threshold=0, edge='rising',
                     kpsh=False, valley=False, show=False, ax=None):

        """Detect peaks in data based on their amplitude and other features.

        Parameters
        ----------
        x : 1D array_like
            data.
        mph : {None, number}, optional (default = None)
            detect peaks that are greater than minimum peak height.
        mpd : positive integer, optional (default = 1)
            detect peaks that are at least separated by minimum peak distance (in
            number of data).
        threshold : positive number, optional (default = 0)
            detect peaks (valleys) that are greater (smaller) than `threshold`
            in relation to their immediate neighbors.
        edge : {None, 'rising', 'falling', 'both'}, optional (default = 'rising')
            for a flat peak, keep only the rising edge ('rising'), only the
            falling edge ('falling'), both edges ('both'), or don't detect a
            flat peak (None).
        kpsh : bool, optional (default = False)
            keep peaks with same height even if they are closer than `mpd`.
        valley : bool, optional (default = False)
            if True (1), detect valleys (local minima) instead of peaks.
        show : bool, optional (default = False)
            if True (1), plot data in matplotlib figure.
        ax : a matplotlib.axes.Axes instance, optional (default = None).

        Returns
        -------
        ind : 1D array_like
            indeces of the peaks in `x`.

        Notes
        -----
        The detection of valleys instead of peaks is performed internally by simply
        negating the data: `ind_valleys = detect_peaks(-x)`

        The function can handle NaN's 

        See this IPython Notebook [1]_.

        References
        ----------
        .. [1] http://nbviewer.ipython.org/github/demotu/BMC/blob/master/notebooks/DetectPeaks.ipynb

        """

        x = np.atleast_1d(x).astype('float64')
        if x.size < 3:
            return np.array([], dtype=int)
        if valley:
            x = -x
        # find indices of all peaks
        dx = x[1:] - x[:-1]
        # handle NaN's
        indnan = np.where(np.isnan(x))[0]
        if indnan.size:
            x[indnan] = np.inf
            dx[np.where(np.isnan(dx))[0]] = np.inf
        ine, ire, ife = np.array([[], [], []], dtype=int)
        if not edge:
            ine = np.where((np.hstack((dx, 0)) < 0) & (np.hstack((0, dx)) > 0))[0]
        else:
            if edge.lower() in ['rising', 'both']:
                ire = np.where((np.hstack((dx, 0)) <= 0) & (np.hstack((0, dx)) > 0))[0]
            if edge.lower() in ['falling', 'both']:
                ife = np.where((np.hstack((dx, 0)) < 0) & (np.hstack((0, dx)) >= 0))[0]
        ind = np.unique(np.hstack((ine, ire, ife)))
        # handle NaN's
        if ind.size and indnan.size:
            # NaN's and values close to NaN's cannot be peaks
            ind = ind[np.in1d(ind, np.unique(np.hstack((indnan, indnan - 1, indnan + 1))), invert=True)]
        # first and last values of x cannot be peaks
        if ind.size and ind[0] == 0:
            ind = ind[1:]
        if ind.size and ind[-1] == x.size - 1:
            ind = ind[:-1]
        # remove peaks < minimum peak height
        if ind.size and mph is not None:
            ind = ind[x[ind] >= mph]
        # remove peaks - neighbors < threshold
        if ind.size and threshold > 0:
            dx = np.min(np.vstack([x[ind] - x[ind - 1], x[ind] - x[ind + 1]]), axis=0)
            ind = np.delete(ind, np.where(dx < threshold)[0])
        # detect small peaks closer than minimum peak distance
        if ind.size and mpd > 0:
            ind = ind[np.argsort(x[ind])][::-1]  # sort ind by peak height
            idel = np.zeros(ind.size, dtype=bool)
            for i in range(ind.size):
                if not idel[i]:
                    # keep peaks with the same height if kpsh is True
                    if y is not None:
                        idel = idel | (y[ind] >= y[ind[i]] - mpd) & (y[ind] <= y[ind[i]] + mpd) \
                                      & (x[ind[i]] > x[ind] if kpsh else True)
                    else:
                        idel = idel | (ind >= ind[i] - mpd) & (ind <= ind[i] + mpd) \
                                      & (x[ind[i]] > x[ind] if kpsh else True)
                    idel[i] = 0  # Keep current peak
            # remove the small peaks and sort back the indices by their occurrence
            ind = np.sort(ind[~idel])

        if show:
            if indnan.size:
                x[indnan] = np.nan
            if valley:
                x = -x
            self._plot(x, mph, mpd, threshold, edge, valley, ax, ind)

        return ind

    def _plot(self, x, mph, mpd, threshold, edge, valley, ax, ind):
        """Plot results of the detect_peaks function, see its help."""
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            print('matplotlib is not available.')
        else:
            if ax is None:
                _, ax = plt.subplots(1, 1, figsize=(8, 4))

            ax.plot(x, 'b', lw=1)
            if ind.size:
                label = 'valley' if valley else 'peak'
                label = label + 's' if ind.size > 1 else label
                ax.plot(ind, x[ind], '+', mfc=None, mec='r', mew=2, ms=8,
                        label='%d %s' % (ind.size, label))
                ax.legend(loc='best', framealpha=.5, numpoints=1)
            ax.set_xlim(-.02 * x.size, x.size * 1.02 - 1)
            ymin, ymax = x[np.isfinite(x)].min(), x[np.isfinite(x)].max()
            yrange = ymax - ymin if ymax > ymin else 1
            ax.set_ylim(ymin - 0.1 * yrange, ymax + 0.1 * yrange)
            ax.set_xlabel('Data #', fontsize=14)
            ax.set_ylabel('Amplitude', fontsize=14)
            mode = 'Valley detection' if valley else 'Peak detection'
            ax.set_title("%s (mph=%s, mpd=%d, threshold=%s, edge='%s')"
                         % (mode, str(mph), mpd, str(threshold), edge))
            # plt.grid()
            plt.show()

    def _update_peaks_distance(self, peaks):
        delta_peaks = self.RampUp_signalSG_polyfit[peaks[1:]] - self.RampUp_signalSG_polyfit[peaks[:-1]]
        return delta_peaks

    def _check_for_outliers(self, peaks, outlier_cutoff):
        # Full range of NanoMax

        # Expected fsr in voltage used to
        one_fsr = self.SG_scale / self.cavity_range * (self.lamb / 2.0)  # in Volt

        delta_peaks = self._update_peaks_distance(peaks)

        outliers = np.where(delta_peaks > outlier_cutoff * one_fsr)[0]

        if outliers.size > 0:
            return outliers
        else:
            return np.array([])

    def _peak_search(self, signal, outlier_cutoff=1.5, show=False):
        # minimum peak height
        mph = -(signal.mean() - np.abs(
            signal.max() - signal.mean()))
        # minimum peak distance
        one_fsr = self.SG_scale / self.cavity_range * (self.lamb / 2.0)  # in Volt
        MaxNumPeak = int((self.RampUp_signalSG.max() - self.RampUp_signalSG.min()) / one_fsr) + 10

        contrast = np.abs(signal.min() - signal.mean())

        errors = [0.75, 0.8, 0.85, 0.9, 0.95]
        constants = np.linspace(0.0, 0.1, 10)

        OutlierList = []
        ErrorList = []
        ConstantList = []

        for error, constant in product(errors, constants):
            # Search for the parameters with least outliers
            mpd = error * one_fsr
            threshold = constant * contrast
            resonances = self._detect_peaks(signal, y=self.RampUp_signalSG_polyfit,
                                                   mph=mph, mpd=mpd, threshold=threshold, valley=True, show=False)
            outliers = self._check_for_outliers(resonances, outlier_cutoff=outlier_cutoff)

            # Check to see if there is too many resonances
            if len(resonances) < MaxNumPeak:
                OutlierList.append(len(outliers))
                ErrorList.append(error)
                ConstantList.append(constant)

        OptimalError = ErrorList[np.argmin(OutlierList)]
        OptimalConstant = ConstantList[np.argmin(OutlierList)]

        mpd = OptimalError * one_fsr
        threshold = OptimalConstant * contrast

        resonances = self._detect_peaks(signal, y=self.RampUp_signalSG_polyfit,
                                               mph=mph, mpd=mpd, threshold=threshold, valley=True, show=show)

        return resonances

    def _find_missing_resonances(self, resonances, outlier_cutoff=1.5):
        corrected_resonances = resonances
        i = 0
        while i < int(1 / 4 * len(resonances)):
            outliers = self._check_for_outliers(resonances, outlier_cutoff)
            i += 1
            if len(outliers) > 0:
                outlier = outliers[0]

                delta_peaks = self._update_peaks_distance(resonances)

                value = self.RampUp_signalSG_polyfit[resonances[outlier]] + np.median(delta_peaks)
                # insert new peak in new corrected array
                corrected_resonances = np.insert(corrected_resonances, outlier + 1,
                                       abs(self.RampUp_signalSG_polyfit - value).argmin())

            else:
                # found no new peaks
                break

        return corrected_resonances

