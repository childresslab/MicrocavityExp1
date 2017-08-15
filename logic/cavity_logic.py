
from qtpy import QtCore
from collections import OrderedDict
from copy import copy
import datetime
import numpy as np
import os
from time import sleep, time
import matplotlib as mpl
import matplotlib.pyplot as plt
from io import BytesIO

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
        self.ramp_channel = 1
        self.reflection_channel = 0

    def on_activate(self):
        """ Initialisation performed during activation of the module.
        """
        self._ni = self.get_connector('nicard')
        self._scope = self.get_connector('scope')
        self._save_logic = self.get_connector('savelogic')

        # Reads in the maximal scanning range. The unit of that scan range is micrometer!
        #self.x_range = self._scanning_device.get_position_range()[0]
        #self.y_range = self._scanning_device.get_position_range()[1]
        #self.z_range = self._scanning_device.get_position_range()[2]


        # Sets connections between signals and functions
        #self.signal_scan_lines_next.connect(self._scan_line, QtCore.Qt.QueuedConnection)
        #self.signal_start_scanning.connect(self.start_scanner, QtCore.Qt.QueuedConnection)
        #self.signal_continue_scanning.connect(self.continue_scanner, QtCore.Qt.QueuedConnection)

        #self._change_position('activation')

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


    def _trim_data(self):
        '''
        Trims data to the ramp

        :return:
        '''
        total_trace = self._acqusition_time  # sec
        ramp_period = 1 / self._full_sweep_freq # sec
        period_index = len(self.time) * ramp_period / total_trace
        ramp_mid = np.argmin(self.volts[self.ramp_channel])

        low_index = ramp_mid - int(period_index / 2)
        high_index = ramp_mid + int(period_index / 2)

        self.volts_trim = self.volts[:, low_index:high_index]
        self.time_trim = self.time[low_index:high_index]

        return 0

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

        #start sweep
        self._scope.run_single()
        sleep(0.4)
        self._ni.start_sweep()

        #stop sweep
        sleep(self._acqusition_time)
        self._ni.stop_sweep()
        self._ni.close_sweep()

    def _get_scope_data(self):
        times, volts = self._scope.aquire_data()
        self.volts = volts.reshape(4, int(len(volts) / 4))
        self.times = times.reshape(4, int(len(times) / 4))
        self.time = self.times[0]

    def _save_raw_data(self):
        self._current_filepath = r'C:\Users\ChildressLab\Desktop\Rasmus notes\Measurements'
        date = datetime.datetime.fromtimestamp(time()).strftime('%Y-%m-%d_%H%M%S')
        self._current_filename = date + '_full_sweep_data.dat'

        data = np.vstack([self.time, self.volts])

        fmt = '%.15e'
        header = ''
        delimiter = '\t'
        comments = '#'

        with open(os.path.join(self._current_filepath, self._current_filename), 'wb') as file:
            np.savetxt(file, data, fmt=fmt, delimiter=delimiter, header=header, comments=comments)


