from qtpy import QtCore
from collections import OrderedDict
import datetime
import numpy as np
import os
from itertools import product
from time import sleep, time
import matplotlib.pyplot as plt

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

    sigFullSweepPlotUpdated = QtCore.Signal(np.ndarray, np.ndarray)
    sigLinewidthPlotUpdated = QtCore.Signal(np.ndarray, np.ndarray)
    sigResonancesUpdated = QtCore.Signal(np.ndarray)
    sigSweepNumberUpdated = QtCore.Signal(int)
    sigTargetModeNumberUpdated = QtCore.Signal(int)

    def __init__(self, config, **kwargs):
        super().__init__(config=config, **kwargs)

        #locking for thread safety
        self.threadlock = Mutex()

        self._full_sweep_freq = 2/3
        self.RampUp_time = np.linspace(0,1 ,100)
        self.RampUp_signalR = np.linspace(1,1,100)
        self._full_sweep_start = 0.0
        self._full_sweep_stop = -3.75
        self._acqusition_time = 2.0
        self.reflection_channel = 0
        self.ramp_channel = 1
        self.position_channel = 3
        self.velocity_channel = 2
        self.SG_scale = 10 # V
        self.lamb = 637e-9 # wavelenght
        self.current_mode_number = 10
        self.current_sweep_number = 1

        self.first_sweep = None
        self.first_corrected_resonances = None
        self.last_sweep = None
        self.last_corrected_resonances = None

        self.mode_shift_list = [0]

        self._current_filepath = r'C:\BittorrentSyncDrive\Personal - Rasmus\Rasmus notes\Measurements\test'

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

# ############################################ DATA AQUISITION START #########################################

    def _trim_data(self, times, volts):
        '''
        Trims data to the ramp

        :return:
        '''
        total_trace = times[-1] - times[0]  # sec
        ramp_period = 1.0 / self._full_sweep_freq  # sec
        period_index = len(times) * ramp_period / total_trace
        ramp_mid = np.argmin(volts[self.ramp_channel])

        low_index = ramp_mid - int(period_index / 2)
        high_index = ramp_mid + int(period_index / 2)

        volts_trim = volts[:, low_index:high_index + 1]
        time_trim = times[low_index:high_index + 1]

        return time_trim, volts_trim

    def _data_split_up(self):
        [self.RampUp_time, self.RampDown_time] = np.array_split(self.time_trim, 2)
        [self.RampUp_signalR, self.RampDown_signalR] = np.array_split(self.volts_trim[self.reflection_channel], 2)
        [self.RampUp_signalNI, self.RampDown_signalNI] = np.array_split(self.volts_trim[self.ramp_channel], 2)
        [self.RampUp_signalSG, self.RampDown_signalSG] = np.array_split(self.volts_trim[self.position_channel], 2)
        [self.RampUp_signalSG_v, self.RampDown_signalSG_v] = np.array_split(self.volts_trim[self.velocity_channel],
                                                                            2)
        return 0

    def _get_ramp_up_signgals(self):
        self.time_trim, self.volts_trim = self._trim_data(self.time, self.volts)
        self._data_split_up()
        return 0
# ############################################ DATA AQUSITION END ##########################################


# ############################################ FITTING START ###############################################
    def _polyfit_SG(self, xdata, ydata, order=3, plot=False):
        xdata_trim = xdata[::9]
        ydata_trim = ydata[::9]
        p_fit = np.poly1d(np.polyfit(xdata_trim, ydata_trim, order))

        if plot is True:
            plt.plot(xdata, ydata, '-', xdata, p_fit(xdata), '--')
            plt.show()

        return p_fit(xdata)

    def _fit_ramp(self, xdata, ydata):
        # Fitting setup
        parameter_guess = [self._full_sweep_start, self._full_sweep_stop,
                           self._full_sweep_freq, self.time_trim[0]]

        func = self._ni.sweep_function

        # Actual fitting
        popt, pcov = curve_fit(func, xdata, ydata, parameter_guess)
        return popt

# ############################################  FITTING END  ##############################################


# ################################## SAVE AND LOAD DATA START ##########################################################

    def _load_full_sweep(self, filepath=None, filename=None):
        """
        Loads data from full sweep

        :param filepath:
        :param filename:
        :return:
        """
        delimiter = '\t'

        if filepath is None:
            filepath = self._current_filepath

        if filename is None:
            filename = self._current_filename

        with open(os.path.join(filepath, filename), 'rb') as file:
            data = np.loadtxt(file, delimiter=delimiter)

        self.time = data[:, 0].transpose()
        self.volts = data[:, 1:5].transpose()

    def _save_raw_data(self, label=''):
        date = datetime.datetime.fromtimestamp(time()).strftime('%Y-%m-%d_%H%M%S')
        self._current_filename = date + label + '_full_sweep_data.dat'

        data = np.vstack([self.time, self.volts])

        fmt = ['%.8e', '%.3e', '%.3e', '%.3e', '%.3e']
        header = ''
        delimiter = '\t'
        comments = '#'

        with open(os.path.join(self._current_filepath, self._current_filename), 'wb') as file:
            np.savetxt(file, data.transpose(), fmt=fmt, delimiter=delimiter, header=header, comments=comments)

    def _save_linewidth_data(self, label, data):
        date = datetime.datetime.fromtimestamp(time()).strftime('%Y-%m-%d_%H%M%S')
        self._current_filename = date + label + '_linewidth_data.dat'

        fmt = ['%.8e']
        for i in range(np.shape(data)[0]-1):
            fmt.append('%.3e')
        header = ''
        delimiter = '\t'
        comments = '#'

        with open(os.path.join(self._current_filepath, self._current_filename), 'wb') as file:
            np.savetxt(file, data.transpose(), fmt=fmt, delimiter=delimiter, header=header, comments=comments)

# ############################# Save and load data end #############################################################

# ################################### FULL SWEEPS START ##################################################

    def _get_scope_data(self):
        """
        Get scope data for all four channels

        This is loaded into self.volts and self.time

        :return: 
        """
        times, volts = self._scope.aquire_data()
        volts = volts.reshape(4, int(len(volts) / 4))
        times = times.reshape(4, int(len(times) / 4))
        time = times[0]

        # First point hare not good! Scope outputs 0 or maybe a header
        self.volts = volts[:, 2000:]
        self.time = time[2000:]

    def setup_scope_for_full_sweep(self):
        self._scope.set_record_lenght(linewidth=False)
        self._scope.set_acquisition_time(self._acqusition_time)
        self._scope.set_data_composition_to_env()
        # HARD CODED!!!!!
        self._scope.set_vertical_scale(2, 500E-3)
        self._scope.set_vertical_position(2, 3500E-3)
        self._scope.set_vertical_scale(3.0, 5E-3)
        self._scope.set_vertical_position(3.0, 0)
        self._scope.set_vertical_scale(4.0, 2)
        self._scope.set_vertical_position(4.0, -2.5)
        sleep(1)

    def start_full_sweep(self):
        """
        Starts a single full sweep

        1. set up the scope for a full sweep
        2. sets up a single ramp 
        3. executes the sweep
        4. get_data
        4. closed sweep


        :return: 
        """
        self._ni.cavity_set_voltage(0.0)
        sleep(1.0)

        # Set up scope for full sweep
        self.setup_scope_for_full_sweep()

        # set up ni card for full sweep
        # One full sweep
        RepOfSweep = 1
        self._ni.set_up_sweep(self._full_sweep_start, self._full_sweep_stop, self._full_sweep_freq, RepOfSweep)

        # start sweep
        self._scope.run_single()
        # HARD CODED!!!!!
        sleep(0.5)
        self._ni.start_sweep()

        # stop sweep
        # HARD CODED!!!!!
        sleep(self._acqusition_time)
        self._ni.close_sweep()

        self._get_scope_data()

        return 0

    def get_nth_full_sweep(self, sweep_number=None, save=True):
        """

        :param sweep_number: 
        :return: 
        """
        if sweep_number is None:
            sweep_number = self.current_sweep_number

        if sweep_number > 1:
            self.last_sweep = self.RampUp_signalR
            self.last_corrected_resonances = self.current_resonances

        try_num = 1
        while True:
            try:
                self.start_full_sweep()

                self._get_ramp_up_signgals()

                self.RampUp_signalSG_polyfit = self._polyfit_SG(xdata=self.RampUp_time,ydata=self.RampUp_signalSG,
                                                        order=3, plot=False)

                resonances = self._peak_search(self.RampUp_signalR)
                corrected_resonances = self._find_missing_resonances(resonances)

                if len(resonances) < 3:
                    continue
                else:
                    break
            except:
                # Did not get the full sweep
                if try_num < 3:
                    try_num += 1
                    continue
                else:
                    return -1

        if sweep_number == 1:
            self.first_sweep = self.RampUp_signalR
            self.first_corrected_resonances = corrected_resonances
            self.first_RampUp_signalSG_polyfit = self.RampUp_signalSG_polyfit
            plt.plot(self.first_RampUp_signalSG_polyfit, self.first_sweep)
            plt.plot(self.first_RampUp_signalSG_polyfit[corrected_resonances], self.first_sweep[corrected_resonances],'o', color='r')
            plt.grid()
            plt.show()

        self.current_sweep_number += 1
        self.current_resonances = corrected_resonances

        self.sigFullSweepPlotUpdated.emit(self.RampUp_time, self.RampUp_signalR)
        self.sigResonancesUpdated.emit(self.current_resonances)

        if save is True:
            self._save_raw_data(label='_{}'.format(sweep_number))

        return 0

# #################################### FULL SWEEPS STOP ##################################################

# #################################### TARGET MODE START ###################################################

    def find_phase_difference(self, signal_a, signal_b, show=False):
        # regularize datasets by subtracting mean and dividing by s.d.
        mod_signal_a = signal_a - signal_a.mean()
        mod_signal_a = -mod_signal_a / mod_signal_a.std()
        mod_signal_b = signal_b - signal_b.mean()
        mod_signal_b = -mod_signal_b / mod_signal_b.std()

        # Calculate cross correlation function https://en.wikipedia.org/wiki/Cross-correlation
        xcorr = np.correlate(mod_signal_a, mod_signal_b, 'same')

        nsamples = mod_signal_a.size
        dt = np.arange(-nsamples / 2, nsamples / 2, dtype=int)

        # Add penalty
        penalty = - 0.25 * np.max(xcorr) * np.abs(dt)
        xcorr = xcorr + penalty

        mode_delay = dt[xcorr.argmax()]

        if np.abs(mode_delay) > 3:
            return None

        if show is True:
            plt.plot(dt, xcorr)
            plt.grid()
            plt.savefig(self._current_filepath + r'\correlation{}.png'.format(self.current_sweep_number), dpi=200)
            plt.show()

        return int(mode_delay)

    def get_target_mode(self, resonances, low_mode=None, high_mode=None, plot=False):

        # Find phase difference
        if low_mode is None:
            low_mode = 0
        if high_mode is None:
            high_mode = np.min([len(self.last_corrected_resonances), len(resonances)])


        if self.last_sweep is None:
            self.last_sweep = self.first_sweep
            self.last_corrected_resonances = self.first_corrected_resonances

        #mode_shift = self.find_phase_difference(self.last_sweep[self.last_corrected_resonances[low_mode:high_mode]],
        #                                        self.RampUp_signalR[resonances[low_mode:high_mode]], show=True)

        closet_old_mode = np.argmin(np.abs(self.target_position-self.RampUp_signalSG_polyfit[resonances]))

        #if mode_shift == None:
        #    return None

        # store mode shifts
        #self.mode_shift_list.append(mode_shift+self.mode_shift_list[-1])

        # Find closets mode
        target_mode = closet_old_mode - 1

        if plot is True:
            index = self.first_corrected_resonances[self.current_mode_number]
            new_index = resonances[target_mode]
            plt.plot(self.first_RampUp_signalSG_polyfit, self.first_sweep)
            plt.plot(self.first_RampUp_signalSG_polyfit[index], self.first_sweep[index], 'o', markersize=10, color='r')
            plt.plot(self.RampUp_signalSG_polyfit, self.RampUp_signalR)
            plt.plot(self.RampUp_signalSG_polyfit[new_index], self.RampUp_signalR[new_index], 'x',
                     markersize=20, color='r')
            plt.grid()
            plt.savefig(self._current_filepath+r'\Target_mode_plot_{}.png'.format(self.current_sweep_number),dpi=200)
            plt.show()

        return target_mode
# #################################### TARGET MODE END ###################################################


# ############################# LINEWIDTH MEASUREMENT ####################################################
    def read_position_from_strain_gauge(self):
        """
        This read the strain gauge voltage from the ni card
        :return:
        """
        rawdata = self._ni.read_position()

        position = np.average(rawdata)
        return position

    def _move_closer_to_resonance(self, current_offset, position_error):
        """

        :param current_offset:
        :param position_error:
        :return:
        """
        # Approximate correction
        response = (-3.75 / 20) * 2  # (expansion of pzt) V/um / (position in volt) 20 um/ 10 V  = 2.0

        correction = - response * position_error
        new_offset = current_offset + correction
        self._ni.cavity_set_voltage(new_offset)

        return new_offset


    def _find_resonance_position_from_strain_gauge(self, current_offset, target_position, threshold_pos):
        """

        :return: offset for mode
        """
        self.log.info('Target position = {:0.3f}'.format(2 * target_position))
        i = 0
        while i < 10:
            self._ni.cavity_set_voltage(current_offset)
            sleep(3.0)
            try:
                position_in_volt = self.read_position_from_strain_gauge()
                position_error = position_in_volt - target_position
                self.log.info('Current position = {:0.3f}, Distance from target {:0.3f}'.format(2 * position_in_volt,
                                                                                                position_error * 2))
                if np.abs(position_error) < threshold_pos / 2.0: # Convert from volt to nm
                    break
                else:
                    current_offset = self._move_closer_to_resonance(current_offset, position_error)
                    i += 1
                    continue
            except:
                self.log.error('could not find resonance position')

        if i > 10:
            self.log.warning('Did not find a position')

        return current_offset

    def setup_scope_for_linewidth(self, trigger_level, acquisition_time):
        """
        
        :param trigger_level: 
        :param acquisition_time: 
        :param position: 
        :param scale: 
        :return: 
        """
        # Adjust ramp channel:
        self._scope.set_data_composition_to_yt()
        self._scope.set_acquisition_time(acquisition_time)
        self._scope.set_record_lenght(linewidth=True)
        self._scope.set_egde_trigger(channel=1, level=trigger_level)

        # FIXME: Adjust position and velocity
        # self._scope.set_vertical_scale(channel=4, scale=1.0)

    def _linewidth_get_data(self):
        """
        Get data from scope

        :return: 
        """
        linewidth_times, self.linewidth_volts = self._scope.aquire_data()
        linewidth_times = linewidth_times.reshape(4, int(len(linewidth_times) / 4))
        self.linewidth_time = linewidth_times[0]

    def linewidth_measurement(self, modes, target_mode, repeat, freq=40):
        """
        1. sets up scope for linewidth measurement
        2. start a ramp around the target mode
        3. gets data if triggered
        4. closes ramp and saves data
        
        :param modes: List of NI_card voltages for each resonances
        :param target_mode:
        :param repeat: number of linewidth measurements
        :param freq: 
        :return: 
        """

        # Setup scope for linewidth measurements with trigger on ramp signal
        contrast = np.abs(self.RampUp_signalR.min() - np.median(self.RampUp_signalR))
        trigger_level = np.median(self.RampUp_signalR) - contrast
        self.setup_scope_for_linewidth(trigger_level=trigger_level, acquisition_time=40e-6)

        # start continues ramp
        #Two first cases are for end modes
        # FIXME: End modes still does not work because of bound of 0, -3.75
        if target_mode >= np.size(modes):
            amplitude = 2 * abs(modes[target_mode - 1] - modes[target_mode]) / 2.0
        elif target_mode == 0:
            amplitude = 2 * abs(modes[target_mode] - modes[target_mode + 1]) / 2.0
        else:
            amplitude = abs(modes[target_mode - 1] - modes[target_mode + 1]) / 2.0

        self.linewidth_time_list = np.array([])
        self.linewidth_volts_list = np.array([])
        self.target_position = self.RampUp_signalSG_polyfit[self.current_resonances[target_mode]]
        # Get data from scope
        i = 0
        while i < repeat:
            k = 1.0

            # Refind correct position
            offset = self._find_resonance_position_from_strain_gauge(current_offset=modes[target_mode],
                                                                     target_position=
                                                                     self.RampUp_signalSG_polyfit[
                                                                         self.current_resonances[
                                                                             target_mode]],
                                                                     threshold_pos=0.025)  # 25 nm

            # Start new ramp
            self._ni.set_up_ramp_output(amplitude, offset, freq)
            self._ni.start_ramp()

            trigger_level = np.median(self.RampUp_signalR) - k * contrast
            self._scope.set_egde_trigger(channel=1, level=trigger_level)
            self._scope.run_single()

            while True:
                if k < 0.2:
                    print('did not find resonance {} {}'.format(self.current_mode_number,i))
                    self.linewidth_time = np.zeros_like(self.linewidth_time)
                    self.linewidth_volts = np.zeros_like(self.linewidth_volts)
                    ret_str = 0
                    break
                # if triggered then get data

                self._scope.scope.write('*OPC?')
                sleep(0.1)
                try:
                    ret_str = self._scope.scope.read()
                    break
                except:
                    k -= 0.02
                    trigger_level = np.median(self.RampUp_signalR) - k * contrast
                    self._scope.set_egde_trigger(channel=1, level=trigger_level)
                    continue

            if ret_str == r'1':
                self._linewidth_get_data()
                i += 1
                self.linewidth_time_list = np.concatenate([self.linewidth_time_list, self.linewidth_time])
                self.linewidth_volts_list = np.concatenate([self.linewidth_volts_list, self.linewidth_volts])

                self.sigLinewidthPlotUpdated.emit(self.linewidth_time,
                                              self.linewidth_volts[0:int(len(self.linewidth_volts) / 4)])

            # Make sure we are still at the right position
            self._ni.stop_ramp()
            self._ni.close_ramp()

            #Update plot in gui

        data = self.linewidth_volts_list
        data = data.reshape(4 * repeat, int(len(data) / (4 * repeat)))
        data = np.vstack([self.linewidth_time, data])

        # close ramp
        self._ni.cavity_set_position(20.0e-6)


        self._save_linewidth_data(label='_{}'.format(self.current_mode_number), data=data)

        return 0

# ############################### PEAK DETECTION START #################################################

    def _detect_peaks(self, x, y=None, mph=None, mpd=1, threshold=0, edge='rising',
                      kpsh=False, valley=False, show=False, ax=None):

        """
        Detect peaks in data based on their amplitude and other features.

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

    def _check_for_outliers(self, peaks, outlier_cutoff=1.5):
        """
        Finds the distances between between resonaces and locates where the is a missing resoances.
        The is when the distances is larger than 1.5 fsr.
        
        :param peaks: list with resonances 
        :param outlier_cutoff: the distance in the units of fsr  
        :return: 
        """

        # Expected fsr in voltage
        one_fsr = self.SG_scale / self.cavity_range * (self.lamb / 2.0)  # in Volt

        # Distance between resonances
        delta_peaks = self.RampUp_signalSG_polyfit[peaks[1:]] - self.RampUp_signalSG_polyfit[peaks[:-1]]

        # Find where the distance between resonances is to large compared to the cutoff
        outliers = np.where(delta_peaks > outlier_cutoff * one_fsr)[0]

        if outliers.size > 0:
            return outliers
        else:
            return np.array([])

    def _find_missing_resonances(self, resonances, outlier_cutoff=1.5):
        """
        Inserts a index for a missing resonance if there is more that 1.5 fsr between two resonances
        
        :param resonances: 
        :param outlier_cutoff: 
        :return: 
        """
        corrected_resonances = resonances
        i = 0
        while i < int(1 / 4 * len(resonances)):
            outliers = self._check_for_outliers(resonances, outlier_cutoff)
            i += 1
            if len(outliers) > 0:
                outlier = outliers[0]

                delta_peaks = self.RampUp_signalSG_polyfit[resonances[1:]] - self.RampUp_signalSG_polyfit[resonances[:-1]]

                value = self.RampUp_signalSG_polyfit[resonances[outlier]] + np.median(delta_peaks)
                # insert new peak in new corrected array
                corrected_resonances = np.insert(corrected_resonances, outlier + 1,
                                       np.abs(self.RampUp_signalSG_polyfit - value).argmin())

            else:
                # found no new peaks
                break

        return corrected_resonances

    def _peak_search(self, signal, outlier_cutoff=1.5, show=False):
        """
        This uses the function peak detect with a few different parameters to find the parameter with
        gives the least amount of outliers (in terms of distance between resonances)
        
        :param signal: 
        :param outlier_cutoff: 
        :param show: 
        :return: 
        """
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


        #Optimal parameters for peak search
        OptimalError = ErrorList[np.argmin(OutlierList)]
        OptimalConstant = ConstantList[np.argmin(OutlierList)]

        mpd = OptimalError * one_fsr
        threshold = OptimalConstant * contrast

        resonances = self._detect_peaks(signal, y=self.RampUp_signalSG_polyfit,
                                        mph=mph, mpd=mpd, threshold=threshold, valley=True, show=show)

        return resonances

# ############################################# PEAK DETECTION End ######################################################

    def get_hw_constraints(self):
        """ Return the names of all ocnfigured fit functions.
        @return object: Hardware constraints object
        """
        # FIXME: Should be from hardware
        #constraints = self._ni.get_limits()
        #return constraints

        pass

    def start_ramp(self, amplitude, offset, freq):

        self._ni.set_up_ramp_output(amplitude, offset, freq)
        self._ni.start_ramp()

    def stop_ramp(self):

        self._ni.stop_ramp()
        self._ni.close_ramp()

    def start_finesse_measurement(self):

        ret_val = self.get_nth_full_sweep(sweep_number=1, save=True)
        if ret_val != 0:
            self.log.error('Did not get first sweep!')

        # fit

    def continue_finesse_measurements(self):
        pass

    def stop_finesse_measurement(self):
        pass



