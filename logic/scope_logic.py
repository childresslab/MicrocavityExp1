from qtpy import QtCore
import numpy as np

from logic.generic_logic import GenericLogic
from core.util.mutex import Mutex
from collections import OrderedDict
import time
import matplotlib.pyplot as plt

class ScopeLogic(GenericLogic):
    """
    Control a process via software PID.
    """
    _modclass = 'scopelogic'
    _modtype = 'logic'
    ## declare connectors
    _connectors = {
        'scope': 'ScopeInterface',
        'savelogic': 'SaveLogic'
    }

    # General Signals, used everywhere:
    sigIdleStateChanged = QtCore.Signal(bool)
    sigPosChanged = QtCore.Signal(dict)
    sigRunContinuous = QtCore.Signal()
    sigRunSingle = QtCore.Signal()
    sigStop = QtCore.Signal()
    sigDataUpdated = QtCore.Signal()

    def __init__(self, config, **kwargs):
        super().__init__(config=config, **kwargs)
        self.log.info('The following configuration was found.')
        # checking for the right configuration
        for key in config.keys():
            self.log.info('{0}: {1}'.format(key,config[key]))
        # locking for thread safety
        self.threadlock = Mutex()

    def on_activate(self):
        self._scope = self.get_connector('scope')
        self._save_logic = self.get_connector('savelogic')

        self.sigRunContinuous.connect(self.run_continuous)
        self.sigRunSingle.connect(self._scope.run_single)
        self.sigStop.connect(self.stop_aq)
        self.scopetime = np.arange(0,1,0.1)
        self.scopedata = [np.zeros([10]) for i in range(4)]
        self.active_channels = []

    def on_deactivate(self):
        """ Perform required deactivation. """



    # General functions

    def run_continuous(self):
        self._scope.run_continuous()

    def stop_aq(self):
        self._scope.stop_acquisition()

    def single_aq(self):
        self._scope.run_single()

    def auto_scale(self):
        self._scope.auto_scale()



    # Channel 1 functions

    def set_channel1_DC_couling(self):
        self._scope.set_channel1_DC_couling()

    def set_channel1_AC_couling(self):
        self._scope.set_channel1_AC_couling()

    def set_channel1_vscale(self, value):
        self._scope.set_channel1_vscale(value)

    def set_channel1_impedance_input_50(self):
        self._scope.set_channel1_impedance_input_50()

    def set_channel1_impedance_input_1M(self):
        self._scope.set_channel1_impedance_input_1M()



    # Channel 2 functions

    def set_channel2_DC_couling(self):
        self._scope.set_channel2_DC_couling()

    def set_channel2_AC_couling(self):
        self._scope.set_channel2_AC_couling()

    def set_channel2_vscale(self, value):
        self._scope.set_channel2_vscale(value)

    def set_channel2_impedance_input_50(self):
        self._scope.set_channel2_impedance_input_50()

    def set_channel2_impedance_input_1M(self):
        self._scope.set_channel2_impedance_input_1M()



    # Channel 3 functions

    def set_channel3_DC_couling(self):
        self._scope.set_channel3_DC_couling()

    def set_channel3_AC_couling(self):
        self._scope.set_channel3_AC_couling()

    def set_channel3_vscale(self, value):
        self._scope.set_channel3_vscale(value)

    def set_channel3_impedance_input_50(self):
        self._scope.set_channel3_impedance_input_50()

    def set_channel3_impedance_input_1M(self):
        self._scope.set_channel3_impedance_input_1M()




    # Channel 4 functions

    def set_channel4_DC_couling(self):
        self._scope.set_channel4_DC_couling()

    def set_channel4_AC_couling(self):
        self._scope.set_channel4_AC_couling()

    def set_channel4_vscale(self, value):
        self._scope.set_channel4_vscale(value)

    def set_channel4_impedance_input_50(self):
        self._scope.set_channel4_impedance_input_50()

    def set_channel4_impedance_input_1M(self):
        self._scope.set_channel4_impedance_input_1M()



    # Trigger functions

    def trigger_mode_EDGE(self):
        self.trigger_mode_EDGE()

    def trigger_source(self):
        self._scope.trigger_source()



    # Acquire functions

    def acquire_mode_normal(self):
        self._scope.acquire_mode_normal()

    def aqcuire_mode_highres(self):
        self._scope.aqcuire_mode_highres()

    def aqcuire_mode_peak(self):
        self._scope.aqcuire_mode_peak()

    def aqcuire_mode_average(self):
        self._scope.aqcuire_mode_average()





    def get_data(self):
        t, y = self._scope.aquire_data(self.active_channels)

        self.scopetime = np.array(t)
        self.scopedata = np.array(y)

        self.sigDataUpdated.emit()

    def get_timescale(self):
        return self.scopetime

    def get_channels(self):
        return self._scope.get_channels()

    def get_time_range(self):
        return self._scope.get_time_range()

    def set_time_range(self, time_range):
        self._scope.set_time_range(time_range)

    def change_channel_state(self, channel, state):
        '''
        @param channel:
        @param state:
        @return:
        '''
        if state is 'ON':
            self._scope.turn_on_channel(channel)
            self.active_channels.append(channel)
        else:
            self._scope.turn_off_channel(channel)
            self.active_channels.remove(channel)

    def save_data(self):
        """ Save the counter trace data and writes it to a file.
        @param bool to_file: indicate, whether data have to be saved to file
        @param str postfix: an additional tag, which will be added to the filename upon save
        @return dict parameters: Dictionary which contains the saving parameters
        """
        filelabel = 'scope_trace'
        parameters = OrderedDict()
        parameters['Scope time'] = time.strftime('%d.%m.%Y %Hh:%Mmin:%Ss')
        self._data_to_save = np.vstack((self.scopetime, self.scopedata))
        header = 'Time (s)'
        for i, ch in enumerate(self.get_channels()):
            header = header + ',Channel{0} (V)'.format(i)
        data = {header: self._data_to_save.transpose()}
        filepath = self._save_logic.get_path_for_module(module_name='Scope')
        fig = self.draw_figure(data=np.array(self._data_to_save))
        self._save_logic.save_data(data, filepath=filepath, parameters=parameters,
                                       filelabel=filelabel, plotfig=fig, delimiter='\t')
        self.log.info('Scope Trace saved to:{0}'.format(filepath)) #'Scope Trace saved to:\n{0}
        return 0

    def draw_figure(self, data):
        """ Draw figure to save with data file.
        @param: nparray data: a numpy array containing counts vs time for all detectors
        @return: fig fig: a matplotlib figure object to be saved to file.
        """
        # Use qudi style
        plt.style.use(self._save_logic.mpl_qd_style)
        # Create figure
        fig, ax = plt.subplots()
        for i in range(len(data)-1):
            ax.plot(data[0], data[i+1], linestyle=':', linewidth=0.5)
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Voltage (V)')
        return fig

    def _split_array(self, trigger_channel = 2, cutoff = 15000):
        trigger_data = self.scopedata[trigger_channel]
        time_data = self.scopetime
        treshold = 1.5
        diff_trigger = np.diff(trigger_data ) > treshold
        indices = np.where(diff_trigger == True)
        split_time = np.split(time_data, indices[0])
        split_data = np.split(trigger_data, indices[0])
        split_data2 = np.split(self.scopedata[0], indices[0])
        freq = 1.0 / np.abs(split_time[1][0]-split_time[1][-1])
        print('freq {}'.format(freq))
        plt.figure(2)
        for i in range(len(split_data2)-2):
            plt.plot(split_data2[i+1])
        plt.xlabel('time (arb)')
        plt.ylabel('Voltage (V)')
        plt.show()
        cycles = len(split_data2)-2
        t = np.linspace(0, 1/freq*cycles, cycles)
        pos1 = []
        pos2 = []
        for i in range(cycles):
            # only upward ramp
            firsthalf = split_data2[i+1][0:int(len(split_time[1])/2)]
            firsthalf_time = split_time[i+1][0:int(len(split_time[1])/2)]
            #plt.figure(i+2)
            #plt.plot(firsthalf)
            #plt.show()
            # first resonance
            res1 = firsthalf[0:cutoff]
            #
            res2 = firsthalf[cutoff:]
            min_indices1 = np.argmin(res1)
            min_indices2 = np.argmin(res2)
            pos1.append(min_indices1)
            pos2.append(min_indices2)
        print(pos1, pos2)
        plt.figure(1)
        plt.plot(t*1e3,pos1-pos1[0])
        plt.plot(t*1e3,pos2-pos2[0])
        plt.xlabel('time (ms)')
        plt.ylabel('resonance position (arb)')
        plt.show()




