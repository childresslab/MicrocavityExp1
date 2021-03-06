# -*- coding: utf-8 -*-

"""
This file contains the Qudi Hardware module NICard class.

Qudi is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

Qudi is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with Qudi. If not, see <http://www.gnu.org/licenses/>.

Copyright (c) the Qudi Developers. See the COPYRIGHT.txt file at the
top-level directory of this distribution and at <https://github.com/Ulm-IQO/qudi/>
"""

import numpy as np
import re

import PyDAQmx as daq

from core.module import Base
from interface.slow_counter_interface import SlowCounterInterface
from interface.slow_counter_interface import SlowCounterConstraints
from interface.slow_counter_interface import CountingMode
from interface.confocal_scanner_interface import ConfocalScannerInterface


class NICard(Base, SlowCounterInterface, ConfocalScannerInterface):
    """ stable: Kay Jahnke, Alexander Stark

    A National Instruments device that can count and control microvave generators.

    Basic procedure how the NI card is configurated:
      * At first you have to define a channel, where the APD clicks will be
        received. That can be any PFI input, which is specified to record TTL
        pulses.
      * Then two counter channels have to be configured.
      * One counter channel serves as a timing device, i.e. basically a clock
        which runs at a certain given frequency.
      * The second counter channel will be used as a gated counting device,
        which will, dependent on the clock, count within the clock interval. The faster
        the clock channel is configured, the smaller is the gated counting
        interval and the less counts per clock periode you will count.

    Therefore the whole issue is to establish a time based gated-counting
    channel.

    Text Based NI-DAQmx Data Acquisition Examples:
    http://www.ni.com/example/6999/en/#ANSIC

    Explanation of the termology, which is used in the NI Card and useful to
    know in connection with our implementation:

    Hardware-Timed Counter Tasks:
        Use hardware-timed counter input operations to drive a control loop. A
        really good explanation can be found in:

        http://zone.ni.com/reference/en-XX/help/370466V-01/mxcncpts/controlappcase4/

    Terminals:
        A terminal is a named location where a signal is either generated
        (output or produced) or acquired (input or consumed). A terminal that
        can output only one signal is often named after that signal. A terminal
        with an input that can be used only for one signal is often named after
        the clock or trigger that the signal is used for. Terminals that are
        used for many signals have generic names such as RTSI, PXITrig, or PFI.

        http://zone.ni.com/reference/en-XX/help/370466W-01/mxcncpts/terminal/
        http://zone.ni.com/reference/en-XX/help/370466V-01/mxcncpts/termnames/

    Ctr0Out, Ctr1Out, Ctr2Out, Ctr3Out:
        Terminals at the I/O connector where the output of counter 0,
        counter 1, counter 2, or counter 3 can be emitted. You also can use
        Ctr0Out as a terminal for driving an external signal onto the RTSI bus.

    Ctr0Gate, Ctr1Gate, Ctr2Gate, Ctr3Gate:
        Terminals within a device whose purpose depends on the application.
        Refer to Counter Parts in NI-DAQmx for more information on how the gate
        terminal is used in various applications.

    Ctr0Source, Ctr1Source, Ctr2Source, Ctr3Source:
        Terminals within a device whose purpose depends on the application.
        Refer to Counter Parts in NI-DAQmx for more information on how the
        source terminal is used in various applications.

    Ctr0InternalOutput, Ctr1InternalOutput, Ctr2InternalOutput,
    Ctr3InternalOutput:
        Terminals within a device where you can choose the pulsed or toggled
        output of the counters. Refer to Counter Parts in NI-DAQmx (or MAX)
        for more information on internal output terminals.

    Task State Model:
        NI-DAQmx uses a task state model to improve ease of use and speed up
        driver performance. Have a look at

        http://zone.ni.com/reference/en-XX/help/370466V-01/mxcncpts/taskstatemodel/

        Small explanation: The task state model consists of five states
            1. Unverified,
            2. Verified,
            3. Reserved,
            4. Committed,
            5. Running.
        You call the Start Task function/VI, Stop Task function/VI, and
        Control Task function/VI to transition the task from one state to
        another. The task state model is very flexible. You can choose to
        interact with as little or as much of the task state model as your
        application requires.

    Device limitations:
        Keep in mind that ONLY the X-series of the NI cards is capable of doing
        a Counter Output Pulse Frequency Train with finite numbers of samples
        by using ONE internal device channel clock (that is the function
        DAQmxCreateCOPulseChanFreq or CO Pulse Freq in Labview)! All other card
        series have to use two counters to generate that!
        Check out the description of NI which tells you 'How Many Counters Does
        Each Type of Counter Input or Output Task Take':

        http://digital.ni.com/public.nsf/allkb/9D1780F448D10F4686257590007B15A8

        This code was tested with NI 6323 and NI 6229, where the first one is
        an X-series device and the latter one is a Low-Cost M Series device.
        With the NI 6229 it is not possible at all to perform the scanning
        task unless you have two of that cards. The limitation came from a lack
        of internal counters.
        The NI 6323 was taken as a basis for this hardware module and thus all
        the function are working on that card.
    """

    _modtype = 'NICard'
    _modclass = 'hardware'

    def on_activate(self, e=None):
        """ Starts up the NI Card at activation.

        @param object e: Event class object from Fysom.
                         An object created by the state machine module Fysom,
                         which is connected to a specific event (have a look in
                         the Base Class). This object contains the passed event,
                         the state before the event happened and the destination
                         of the state which should be reached after the event
                         had happened.
        """

        # the tasks used on that hardware device:
        self._counter_daq_tasks = []
        self._clock_daq_task = None
        self._scanner_clock_daq_task = None
        self._scanner_ao_task = None
        self._cavity_ao_task = None
        self._scanner_ai_task = None
        self._scanner_counter_daq_tasks = []
        self._line_length = None
        self._odmr_length = None
        self._gated_counter_daq_task = None
        self._ramp_task = None
        self._sweep_task = None

        # used as a default for expected maximum counts
        self._max_counts = 3e7
        # timeout for the Read or/and write process in s
        self._RWTimeout = 10

        self._clock_frequency_default = 100             # in Hz
        self._scanner_clock_frequency_default = 100     # in Hz
        # number of readout samples, mainly used for gated counter
        self._samples_number_default = 50
        self.SampNum = 10000

        config = self.getConfiguration()

        self._scanner_ao_channels = []
        self._scanner_ai_channels = []
        self._voltage_range = []
        self._cavity_voltage_range = []
        self._position_range = []
        self._cavity_position_range = []
        self._current_position = []
        self._current_cavity_position = []
        self._counter_channels = []
        self._scanner_counter_channels = []
        self._photon_sources = []

        if 'cavity_ao' in config.keys():
            self.cavity_channel = config['cavity_ao']
            self._cavity_voltage_range = [-10,10]
            self._current_cavity_position.append(0)
            self._cavity_position_range.append([0, 20e-6])

        if 'cavity_voltage_range' in config.keys():
            if float(config['cavity_voltage_range'][0]) < float(config['cavity_voltage_range'][1]):
                vlow = float(config['cavity_voltage_range'][0])
                vhigh = float(config['cavity_voltage_range'][1])
                self._cavity_voltage_range = [vlow, vhigh]
            else:
                self.log.warning(
                    'Configuration ({0}) of cavity_voltage range incorrect, taking [-10, 10] instead.'
                    ''.format(config['cavity_voltage range']))

        if 'pd1' in config.keys():
            self._scanner_ai_channels.append(config['pd1'])
        # handle all the parameters given by the config
        if 'scanner_x_ao' in config.keys():
            self._scanner_ao_channels.append(config['scanner_x_ao'])
            self._current_position.append(0)
            self._position_range.append([0, 100e-6])
            self._voltage_range.append([-10, 10])
            if 'scanner_y_ao' in config.keys():
                self._scanner_ao_channels.append(config['scanner_y_ao'])
                self._current_position.append(0)
                self._position_range.append([0, 100e-6])
                self._voltage_range.append([-10, 10])
                if 'scanner_z_ao' in config.keys():
                    self._scanner_ao_channels.append(config['scanner_z_ao'])
                    self._current_position.append(0)
                    self._position_range.append([0, 100e-6])
                    self._voltage_range.append([-10, 10])

        if len(self._scanner_ao_channels) < 1:
            self.log.error(
                'Not enough scanner channels found in the configuration!\n'
                'Be sure to start with scanner_x_ao\n'
                'Assign to that parameter an appropriate channel from your NI Card, '
                'otherwise you cannot control the analog channels!')

        if 'clock_channel' in config.keys():
            self._clock_channel = config['clock_channel']
        else:
            self.log.error(
                'No parameter "clock_channel" configured.'
                'Assign to that parameter an appropriate channel from your NI Card!')

        if 'scanner_counter_channel' in config.keys():
            self._scanner_counter_channels.append(config['scanner_counter_channel'])
            n = 2
            while 'scanner_counter_channel{0}'.format(n) in config.keys():
                self._scanner_counter_channels.append(
                    config['scanner_counter_channel{0}'.format(n)])
                n += 1
        else:
            self.log.error(
                'No parameter "scanner_counter_channel" configured.\n'
                'Assign to that parameter an appropriate channel from your NI Card!')

        if 'photon_source' in config.keys():
            self._photon_sources.append(config['photon_source'])
            n = 2
            while 'photon_source{0}'.format(n) in config.keys():
                self._photon_sources.append(config['photon_source{0}'.format(n)])
                n += 1
        else:
            self.log.error(
                'No parameter "photon_source" configured.\n'
                'Assign to that parameter an appropriated channel from your NI Card!')

        if 'counter_channel' in config.keys():
            self._counter_channels.append(config['counter_channel'])
            n = 2
            while 'counter_channel{0}'.format(n) in config.keys():
                self._counter_channels.append(config['counter_channel{0}'.format(n)])
                n += 1
        else:
            self.log.error(
                'No parameter "counter_channel" configured.\n'
                'Assign to that parameter an appropriate channel from your NI Card!')

        if 'pixel_clock_channel' in config.keys():
            self._pixel_clock_channel = config['pixel_clock_channel']
        else:
            self._pixel_clock_channel = None

        if 'clock_frequency' in config.keys():
            self._clock_frequency = config['clock_frequency']
        else:
            self._clock_frequency = self._clock_frequency_default
            self.log.warning(
                'No clock_frequency configured, taking 100 Hz instead.')

        if 'counting_edge_rising' in config.keys():
            if config['counting_edge_rising']:
                self._counting_edge = daq.DAQmx_Val_Rising
            else:
                self._counting_edge = daq.DAQmx_Val_Falling
        else:
            self.log.warning(
                'No parameter "counting_edge_rising" configured.\n'
                'Set this parameter either to True (rising edge) or to False (falling edge).\n'
                'Taking the default value {0}'.format(self._counting_edge_default))
            self._counting_edge = self._counting_edge_default

        if 'scanner_clock_frequency' in config.keys():
            self._scanner_clock_frequency = config['scanner_clock_frequency']
        else:
            self._scanner_clock_frequency = self._scanner_clock_frequency_default
            self.log.warning(
                'No scanner_clock_frequency configured, taking 100 Hz instead.')

        if 'samples_number' in config.keys():
            self._samples_number = config['samples_number']
        else:
            self._samples_number = self._samples_number_default
            self.log.warning(
                'No parameter "samples_number" configured taking the default value "{0}" instead.'
                ''.format(self._samples_number_default))
            self._samples_number = self._samples_number_default

        if 'x_range' in config.keys() and len(self._position_range) > 0:
            if float(config['x_range'][0]) < float(config['x_range'][1]):
                self._position_range[0] = [float(config['x_range'][0]),
                                           float(config['x_range'][1])]
            else:
                self.log.warning(
                    'Configuration ({}) of x_range incorrect, taking [0,100e-6] instead.'
                    ''.format(config['x_range']))
        else:
            if len(self._position_range) > 0:
                self.log.warning('No x_range configured taking [0,100e-6] instead.')

        if 'y_range' in config.keys() and len(self._position_range) > 1:
            if float(config['y_range'][0]) < float(config['y_range'][1]):
                self._position_range[1] = [float(config['y_range'][0]),
                                           float(config['y_range'][1])]
            else:
                self.log.warning(
                    'Configuration ({}) of y_range incorrect, taking [0,100e-6] instead.'
                    ''.format(config['y_range']))
        else:
            if len(self._position_range) > 1:
                self.log.warning('No y_range configured taking [0,100e-6] instead.')

        if 'z_range' in config.keys() and len(self._position_range) > 2:
            if float(config['z_range'][0]) < float(config['z_range'][1]):
                self._position_range[2] = [float(config['z_range'][0]),
                                           float(config['z_range'][1])]
            else:
                self.log.warning(
                    'Configuration ({}) of z_range incorrect, taking [0,100e-6] instead.'
                    ''.format(config['z_range']))
        else:
            if len(self._position_range) > 2:
                self.log.warning('No z_range configured taking [0,100e-6] instead.')

        if 'cavity_range' in config.keys():
            if float(config['cavity_range'][0]) < float(config['cavity_range'][1]):
                self._cavity_position_range = [float(config['cavity_range'][0]),
                                           float(config['cavity_range'][1])]
            else:
                self.log.warning(
                    'Configuration ({}) of cavity_range incorrect, taking [0,100e-6] instead.'
                    ''.format(config['cavity_range']))
        else:
            if len(self._position_range) > 3:
                self.log.warning('No a_range configured taking [0,100e-6] instead.')

        if 'voltage_range' in config.keys():
            if float(config['voltage_range'][0]) < float(config['voltage_range'][1]):
                vlow = float(config['voltage_range'][0])
                vhigh = float(config['voltage_range'][1])
                self._voltage_range = [
                    [vlow, vhigh], [vlow, vhigh], [vlow, vhigh], [vlow, vhigh]
                    ][0:len(self._voltage_range)]
            else:
                self.log.warning(
                    'Configuration ({}) of voltage_range incorrect, taking [-10,10] instead.'
                    ''.format(config['voltage_range']))
        else:
            self.log.warning('No voltage_range configured, taking [-10,10] instead.')

        if 'x_voltage_range' in config.keys() and len(self._voltage_range) > 0:
            if float(config['x_voltage_range'][0]) < float(config['x_voltage_range'][1]):
                vlow = float(config['x_voltage_range'][0])
                vhigh = float(config['x_voltage_range'][1])
                self._voltage_range[0] = [vlow, vhigh]
            else:
                self.log.warning(
                    'Configuration ({0}) of x_voltage_range incorrect, taking [-10, 10] instead.'
                    ''.format(config['x_voltage_range']))
        else:
            if 'voltage_range' not in config.keys():
                self.log.warning('No x_voltage_range configured, taking [-10, 10] instead.')

        if 'y_voltage_range' in config.keys() and len(self._voltage_range) > 1:
            if float(config['y_voltage_range'][0]) < float(config['y_voltage_range'][1]):
                vlow = float(config['y_voltage_range'][0])
                vhigh = float(config['y_voltage_range'][1])
                self._voltage_range[1] = [vlow, vhigh]
            else:
                self.log.warning(
                    'Configuration ({0}) of y_voltage_range incorrect, taking [-10, 10] instead.'
                    ''.format(config['y_voltage_range']))
        else:
            if 'voltage_range' not in config.keys():
                self.log.warning('No y_voltage_range configured, taking [-10, 10] instead.')

        if 'z_voltage_range' in config.keys() and len(self._voltage_range) > 2:
            if float(config['z_voltage_range'][0]) < float(config['z_voltage_range'][1]):
                vlow = float(config['z_voltage_range'][0])
                vhigh = float(config['z_voltage_range'][1])
                self._voltage_range[2] = [vlow, vhigh]
            else:
                self.log.warning(
                    'Configuration ({0}) of z_voltage_range incorrect, taking [-10, 10] instead.'
                    ''.format(config['z_voltage_range']))
        else:
            if 'voltage_range' not in config.keys():
                self.log.warning('No z_voltage_range configured, taking [-10, 10] instead.')

        # Analog output is always needed and it does not interfere with the
        # rest, so start it always and leave it running
        if self._start_analog_output() < 0:
            self.log.error('Failed to start analog output.')
            raise Exception('Failed to start NI Card module due to analog output failure.')

        if self._start_cavity_analog_output() < 0:
            self.log.error('Failed to start analog output.')
            raise Exception('Failed to start NI Card module due to analog output failure.')

    def on_deactivate(self):

        """ Shut down the NI card.
        """
        self.reset_hardware()

    # =================== SlowCounterInterface Commands ========================

    def get_constraints(self):
        """ Get hardware limits of NI device.

        @return SlowCounterConstraints: constraints class for slow counter

        FIXME: ask hardware for limits when module is loaded
        """
        constraints = SlowCounterConstraints()
        constraints.max_detectors = 4
        constraints.min_count_frequency = 1e-3
        constraints.max_count_frequency = 10e9
        constraints.counting_mode = [CountingMode.CONTINUOUS]
        return constraints

    def set_up_clock(self, clock_frequency=None, clock_channel=None, scanner=False, idle=False):
        """ Configures the hardware clock of the NiDAQ card to give the timing.

        @param float clock_frequency: if defined, this sets the frequency of
                                      the clock in Hz
        @param string clock_channel: if defined, this is the physical channel
                                     of the clock within the NI card.
        @param bool scanner: if set to True method will set up a clock function
                             for the scanner, otherwise a clock function for a
                             counter will be set.
        @param bool idle: set whether idle situation of the counter (where
                          counter is doing nothing) is defined as
                                True  = 'Voltage High/Rising Edge'
                                False = 'Voltage Low/Falling Edge'

        @return int: error code (0:OK, -1:error)
        """

        if not scanner and self._clock_daq_task is not None:
            self.log.error('Another counter clock is already running, close this one first.')
            return -1

        if scanner and self._scanner_clock_daq_task is not None:
            self.log.error('Another scanner clock is already running, close this one first.')
            return -1

        # Create handle for task, this task will generate pulse signal for
        # photon counting
        my_clock_daq_task = daq.TaskHandle()

        # assign the clock frequency, if given
        if clock_frequency is not None:
            if not scanner:
                self._clock_frequency = float(clock_frequency)
            else:
                self._scanner_clock_frequency = float(clock_frequency)

        # use the correct clock in this method
        if scanner:
            my_clock_frequency = self._scanner_clock_frequency * 2
        else:
            my_clock_frequency = self._clock_frequency

        # assign the clock channel, if given
        if clock_channel is not None:
            if not scanner:
                self._clock_channel = clock_channel

        # use the correct clock channel in this method

        my_clock_channel = self._clock_channel

        # check whether only one clock pair is available, since some NI cards
        # only one clock channel pair.

        # Adjust the idle state if necessary
        my_idle = daq.DAQmx_Val_High if idle else daq.DAQmx_Val_Low
        try:
            # create task for clock
            task_name = 'ScannerClock' if scanner else 'CounterClock'
            daq.DAQmxCreateTask(task_name, daq.byref(my_clock_daq_task))

            # create a digital clock channel with specific clock frequency:
            daq.DAQmxCreateCOPulseChanFreq(
                # The task to which to add the channels
                my_clock_daq_task,
                # which channel is used?
                my_clock_channel,
                # Name to assign to task (NIDAQ uses by # default the physical channel name as
                # the virtual channel name. If name is specified, then you must use the name
                # when you refer to that channel in other NIDAQ functions)
                'Clock Producer',
                # units, Hertz in our case
                daq.DAQmx_Val_Hz,
                # idle state
                my_idle,
                # initial delay
                0,
                # pulse frequency, divide by 2 such that length of semi period = count_interval
                my_clock_frequency / 2,
                # duty cycle of pulses, 0.5 such that high and low duration are both
                # equal to count_interval
                0.5)

            # Configure Implicit Timing.
            # Set timing to continuous, i.e. set only the number of samples to
            # acquire or generate without specifying timing:
            daq.DAQmxCfgImplicitTiming(
                # Define task
                my_clock_daq_task,
                # Sample Mode: set the task to generate a continuous amount of running samples
                daq.DAQmx_Val_ContSamps,
                # buffer length which stores temporarily the number of generated samples
                1000)

            if scanner:
                self._scanner_clock_daq_task=my_clock_daq_task
            else:
                # actually start the preconfigured clock task
                daq.DAQmxStartTask(my_clock_daq_task)
                self._clock_daq_task = my_clock_daq_task
        except:
            self.log.exception('Error while setting up clock.')
            return -1
        return 0

    def set_up_counter(self,
                       counter_channels=None,
                       sources=None,
                       clock_channel=None,
                       counter_buffer=None):
        """ Configures the actual counter with a given clock.

        @param list(str) counter_channels: optional, physical channel of the counter
        @param list(str) sources: optional, physical channel where the photons
                                  are to count from
        @param str clock_channel: optional, specifies the clock channel for the
                                  counter
        @param int counter_buffer: optional, a buffer of specified integer
                                   length, where in each bin the count numbers
                                   are saved.

        @return int: error code (0:OK, -1:error)
        """

        if self._clock_daq_task is None and clock_channel is None:
            self.log.error('No clock running, call set_up_clock before starting the counter.')
            return -1
        if len(self._counter_daq_tasks) > 0:
            self.log.error('Another counter is already running, close this one first.')
            return -1

        if counter_channels is not None:
            my_counter_channels = counter_channels
        else:
            my_counter_channels = self.get_counter_channels()

        if sources is not None:
            my_photon_sources = sources
        else:
            my_photon_sources = self._photon_sources

        if clock_channel is not None:
            my_clock_channel = clock_channel
        else:
            my_clock_channel = self._clock_channel

        if len(my_photon_sources) < len(my_counter_channels):
            self.log.error('You have given {0} sources but {1} counting channels.'
                           'Please give an equal or greater number of sources.'
                           ''.format(len(my_photon_sources), len(my_counter_channels)))
            return -1

        try:
            for i, ch in enumerate(my_counter_channels):
                # This task will count photons with binning defined by the clock_channel
                task = daq.TaskHandle()  # Initialize a Task
                # Create task for the counter
                daq.DAQmxCreateTask('Counter{0}'.format(i), daq.byref(task))
                # Create a Counter Input which samples with Semi-Periodes the Channel.
                # set up semi period width measurement in photon ticks, i.e. the width
                # of each pulse (high and low) generated by pulse_out_task is measured
                # in photon ticks.
                #   (this task creates a channel to measure the time between state
                #    transitions of a digital signal and adds the channel to the task
                #    you choose)
                daq.DAQmxCreateCISemiPeriodChan(
                    # define to which task to connect this function
                    task,
                    # use this counter channel
                    ch,
                    # name to assign to it
                    'Counter Channel {0}'.format(i),
                    # expected minimum count value
                    0,
                    # Expected maximum count value
                    self._max_counts / 2 / self._clock_frequency,
                    # units of width measurement, here photon ticks
                    daq.DAQmx_Val_Ticks,
                    # empty extra argument
                    '')

                # Set the Counter Input to a Semi Period input Terminal.
                # Connect the pulses from the counter clock to the counter channel
                daq.DAQmxSetCISemiPeriodTerm(
                        # The task to which to add the counter channel.
                        task,
                        # use this counter channel
                        ch,
                        # assign a named Terminal
                        my_clock_channel + 'InternalOutput')

                # Set a Counter Input Control Timebase Source.
                # Specify the terminal of the timebase which is used for the counter:
                # Define the source of ticks for the counter as self._photon_source for
                # the Scanner Task.
                daq.DAQmxSetCICtrTimebaseSrc(
                    # define to which task to connect this function
                    task,
                    # counter channel
                    ch,
                    # counter channel to output the counting results
                    my_photon_sources[i])

                # Configure Implicit Timing.
                # Set timing to continuous, i.e. set only the number of samples to
                # acquire or generate without specifying timing:
                daq.DAQmxCfgImplicitTiming(
                    # define to which task to connect this function
                    task,
                    # Sample Mode: Acquire or generate samples until you stop the task.
                    daq.DAQmx_Val_ContSamps,
                    # buffer length which stores  temporarily the number of generated samples
                    1000)

                # Set the Read point Relative To an operation.
                # Specifies the point in the buffer at which to begin a read operation.
                # Here we read most recent recorded samples:
                daq.DAQmxSetReadRelativeTo(
                    # define to which task to connect this function
                    task,
                    # Start reading samples relative to the last sample returned by the previously.
                    daq.DAQmx_Val_CurrReadPos)

                # Set the Read Offset.
                # Specifies an offset in samples per channel at which to begin a read
                # operation. This offset is relative to the location you specify with
                # RelativeTo. Here we set the Offset to 0 for multiple samples:
                daq.DAQmxSetReadOffset(task, 0)

                # Set Read OverWrite Mode.
                # Specifies whether to overwrite samples in the buffer that you have
                # not yet read. Unread data in buffer will be overwritten:
                daq.DAQmxSetReadOverWrite(
                    task,
                    daq.DAQmx_Val_DoNotOverwriteUnreadSamps)
                # add task to counter task list
                self._counter_daq_tasks.append(task)
        except:
            self.log.exception('Error while setting up counting task.')
            return -1

        try:
            for i, task in enumerate(self._counter_daq_tasks):
                # Actually start the preconfigured counter task
                daq.DAQmxStartTask(task)
        except:
            self.log.exception('Error while starting Counter')
            try:
                self.close_counter()
            except:
                self.log.exception('Could not close counter after error')
            return -1
        return 0

    def get_counter_channels(self):
        """ Returns the list of counter channel names.

        @return tuple(str): channel names

        Most methods calling this might just care about the number of channels, though.
        """
        return self._counter_channels

    def get_counter(self, samples=None):
        """ Returns the current counts per second of the counter.

        @param int samples: if defined, number of samples to read in one go.
                            How many samples are read per readout cycle. The
                            readout frequency was defined in the counter setup.
                            That sets also the length of the readout array.

        @return float [samples]: array with entries as photon counts per second
        """
        if len(self._counter_daq_tasks) < 1:
            self.log.error(
                'No counter running, call set_up_counter before reading it.')
            # in case of error return a lot of -1
            return np.ones((len(self.get_counter_channels()), samples), dtype=np.uint32) * -1

        if samples is None:
            samples = int(self._samples_number)
        else:
            samples = int(samples)
        try:
            # count data will be written here in the NumPy array of length samples
            count_data = np.empty((len(self._counter_daq_tasks), samples), dtype=np.uint32)

            # number of samples which were actually read, will be stored here
            n_read_samples = daq.int32()
            for i, task in enumerate(self._counter_daq_tasks):
                # read the counter value: This function is blocking and waits for the
                # counts to be all filled:
                daq.DAQmxReadCounterU32(
                    # read from this task
                    task,
                    # number of samples to read
                    samples,
                    # maximal timeout for the read process
                    self._RWTimeout,
                    # write the readout into this array
                    count_data[i],
                    # length of array to write into
                    samples,
                    # number of samples which were read
                    daq.byref(n_read_samples),
                    # Reserved for future use. Pass NULL (here None) to this parameter
                    None)
        except:
            self.log.exception(
                'Getting samples from counter failed.')
            # in case of error return a lot of -1
            return np.ones((len(self.get_counter_channels()), samples), dtype=np.uint32) * -1
        # normalize to counts per second and return data
        return count_data * self._clock_frequency

    def close_counter(self, scanner=False):
        """ Closes the counter or scanner and cleans up afterwards.

        @param bool scanner: specifies if the counter- or scanner- function
                             will be excecuted to close the device.
                                True = scanner
                                False = counter

        @return int: error code (0:OK, -1:error)
        """
        error = 0
        if scanner:
            for i, task in enumerate(self._scanner_counter_daq_tasks):
                try:
                    # stop the counter task
                    daq.DAQmxStopTask(task)
                    # after stopping delete all the configuration of the counter
                    daq.DAQmxClearTask(task)
                except:
                    self.log.exception('Could not close scanner counter.')
                    error = -1
            self._scanner_counter_daq_tasks = []
        else:
            for i, task in enumerate(self._counter_daq_tasks):
                try:
                    # stop the counter task
                    daq.DAQmxStopTask(task)
                    # after stopping delete all the configuration of the counter
                    daq.DAQmxClearTask(task)
                    # set the task handle to None as a safety
                except:
                    self.log.exception('Could not close counter.')
                    error = -1
            self._counter_daq_tasks = []
        return error

    def close_clock(self, scanner=False):
        """ Closes the clock and cleans up afterwards.

        @param bool scanner: specifies if the counter- or scanner- function
                             should be used to close the device.
                                True = scanner
                                False = counter

        @return int: error code (0:OK, -1:error)
        """
        if scanner:
            my_task = self._clock_daq_task
        else:
            my_task = self._clock_daq_task
        try:
            # Stop the clock task:
            daq.DAQmxStopTask(my_task)

            # After stopping delete all the configuration of the clock:
            daq.DAQmxClearTask(my_task)

            # Set the task handle to None as a safety
            if scanner:
                self._scanner_clock_daq_task = None
            else:
                self._clock_daq_task = None
        except:
            self.log.exception('Could not close clock.')
            return -1
        return 0

    # ================ End SlowCounterInterface Commands =======================

    # ================ ConfocalScannerInterface Commands =======================
    def reset_hardware(self):
        """ Resets the NI hardware, so the connection is lost and other
            programs can access it.

        @return int: error code (0:OK, -1:error)
        """
        retval = 0
        chanlist = [
            self._clock_channel,
            self._gate_in_channel
            ]
        chanlist.extend(self._scanner_ao_channels)
        chanlist.extend(self._photon_sources)
        chanlist.extend(self._counter_channels)
        chanlist.extend(self._scanner_counter_channels)

        devicelist = []
        for channel in chanlist:
            if channel is None:
                continue
            match = re.match(
                '^/(?P<dev>[0-9A-Za-z\- ]+[0-9A-Za-z\-_ ]*)/(?P<chan>[0-9A-Za-z]+)',
                channel)
            if match:
                devicelist.append(match.group('dev'))
            else:
                self.log.error('Did not find device name in {0}.'.format(channel))
        for device in set(devicelist):
            self.log.info('Reset device {0}.'.format(device))
            try:
                daq.DAQmxResetDevice(device)
            except:
                self.log.exception('Could not reset NI device {0}'.format(device))
                retval = -1
        return retval

    def get_scanner_axes(self):
        """ Scanner axes depends on how many channels tha analog output task has.
        """
        if self._scanner_ao_task is None:
            self.log.error('Cannot get channel number, analog output task does not exist.')
            return []

        n_channels = daq.uInt32()
        daq.DAQmxGetTaskNumChans(self._scanner_ao_task, n_channels)
        possible_channels = ['x', 'y', 'z', 'a']

        return possible_channels[0:int(n_channels.value)]

    def get_scanner_count_channels(self):
        """ Return list of counter channels """
        return self._scanner_counter_channels

    def get_position_range(self):
        """ Returns the physical range of the scanner.

        @return float [4][2]: array of 4 ranges with an array containing lower
                              and upper limit. The unit of the scan range is
                              meters.
        """
        return self._position_range

    def set_position_range(self, myrange=None):
        """ Sets the physical range of the scanner.

        @param float [4][2] myrange: array of 4 ranges with an array containing
                                     lower and upper limit. The unit of the
                                     scan range is meters.

        @return int: error code (0:OK, -1:error)
        """
        if myrange is None:
            myrange = [[0, 1e-6], [0, 1e-6], [0, 1e-6], [0, 1e-6]]

        if not isinstance( myrange, (frozenset, list, set, tuple, np.ndarray, ) ):
            self.log.error('Given range is no array type.')
            return -1

        if len(myrange) != 4:
            self.log.error(
                'Given range should have dimension 4, but has {0:d} instead.'
                ''.format(len(myrange)))
            return -1

        for pos in myrange:
            if len(pos) != 2:
                self.log.error(
                    'Given range limit {1:d} should have dimension 2, but has {0:d} instead.'
                    ''.format(len(pos), pos))
                return -1
            if pos[0]>pos[1]:
                self.log.error(
                    'Given range limit {0:d} has the wrong order.'.format(pos))
                return -1

        self._position_range = myrange
        return 0

    def set_voltage_range(self, myrange=None):
        """ Sets the voltage range of the NI Card.

        @param float [n][2] myrange: array containing lower and upper limit

        @return int: error code (0:OK, -1:error)
        """
        n_ch = len(self.get_scanner_axes())
        if myrange is None:
            myrange = [[-10., 10.], [-10., 10.], [-10., 10.], [-10., 10.]][0:n_ch]

        if not isinstance(myrange, (frozenset, list, set, tuple, np.ndarray)):
            self.log.error('Given range is no array type.')
            return -1

        if len(myrange) != n_ch:
            self.log.error(
                'Given range should have dimension 2, but has {0:d} instead.'
                ''.format(len(myrange)))
            return -1

        for r in myrange:
            if r[0] > r[1]:
                self.log.error('Given range limit {0:d} has the wrong order.'.format(r))
                return -1

        self._voltage_range = myrange
        return 0

    def _start_analog_output(self):
        """ Starts or restarts the analog output.

        @return int: error code (0:OK, -1:error)
        """
        try:
            # If an analog task is already running, kill that one first
            if self._scanner_ao_task is not None:
                # stop the analog output task
                daq.DAQmxStopTask(self._scanner_ao_task)

                # delete the configuration of the analog output
                daq.DAQmxClearTask(self._scanner_ao_task)

                # set the task handle to None as a safety
                self._scanner_ao_task = None

            # initialize ao channels / task for scanner, should always be active.
            # Define at first the type of the variable as a Task:
            self._scanner_ao_task = daq.TaskHandle()

            # create the actual analog output task on the hardware device. Via
            # byref you pass the pointer of the object to the TaskCreation function:
            daq.DAQmxCreateTask('ScannerAO', daq.byref(self._scanner_ao_task))
            for n, chan in enumerate(self._scanner_ao_channels):
                # Assign and configure the created task to an analog output voltage channel.
                daq.DAQmxCreateAOVoltageChan(
                    # The AO voltage operation function is assigned to this task.
                    self._scanner_ao_task,
                    # use (all) scanner ao_channels for the output
                    chan,
                    # assign a name for that channel
                    'Scanner AO Channel {0}'.format(n),
                    # minimum possible voltage
                    self._voltage_range[n][0],
                    # maximum possible voltage
                    self._voltage_range[n][1],
                    # units is Volt
                    daq.DAQmx_Val_Volts,
                    # empty for future use
                    '')
        except:
            self.log.exception('Error starting analog output task.')
            return -1
        return 0

    def _stop_analog_output(self):
        """ Stops the analog output.

        @return int: error code (0:OK, -1:error)
        """
        if self._scanner_ao_task is None:
            return -1
        retval = 0
        try:
            # stop the analog output task
            daq.DAQmxStopTask(self._scanner_ao_task)
        except:
            self.log.exception('Error stopping analog output.')
            retval = -1
        try:
            daq.DAQmxSetSampTimingType(self._scanner_ao_task, daq.DAQmx_Val_OnDemand)
        except:
            self.log.exception('Error changing analog output mode.')
            retval = -1
        return retval

    def _start_analog_input(self):
        """ Starts or restarts the analog input.

        @return int: error code (0:OK, -1:error)
        """
        try:
            # If an analog task is already running, kill that one first
            if self._scanner_ai_task is not None:
                # stop the analog output task
                daq.DAQmxStopTask(self._scanner_ai_task)

                # delete the configuration of the analog output
                daq.DAQmxClearTask(self._scanner_ai_task)

                # set the task handle to None as a safety
                self._scanner_ai_task = None

            # initialize ao channels / task for scanner, should always be active.
            # Define at first the type of the variable as a Task:
            self._scanner_ai_task = daq.TaskHandle()

            # create the actual analog output task on the hardware device. Via
            # byref you pass the pointer of the object to the TaskCreation function:
            daq.DAQmxCreateTask('ScannerAI', daq.byref(self._scanner_ai_task))
            for n, chan in enumerate(self._scanner_ai_channels):
                # Assign and configure the created task to an analog output voltage channel.
                daq.DAQmxCreateAIVoltageChan(
                    # The AO voltage operation function is assigned to this task.
                    self._scanner_ai_task,
                    # use (all) scanner ao_channels for the output
                    chan,
                    # assign a name for that channel
                    'Scanner AI Channel {0}'.format(n),
                    #
                    daq.DAQmx_Val_Cfg_Default,
                    # minimum possible voltage
                    -10,
                    # maximum possible voltage
                    10,
                    # units is Volt
                    daq.DAQmx_Val_Volts,
                    # empty for future use
                    '')
        except:
            self.log.exception('Error starting analog output task.')
            return -1
        return 0

    def _stop_analog_input(self):
        """ Stops the analog input.

        @return int: error code (0:OK, -1:error)
        """
        if self._scanner_ai_task is None:
            return -1
        retval = 0
        try:
            # stop the analog output task
            daq.DAQmxStopTask(self._scanner_ai_task)
        except:
            self.log.exception('Error stopping analog input.')
            retval = -1
        try:
            daq.DAQmxSetSampTimingType(self._scanner_ai_task, daq.DAQmx_Val_OnDemand)
        except:
            self.log.exception('Error changing analog input mode.')
            retval = -1
        return retval

    def read_position(self):
        """
        Read 100 data points from analog input
        :return:
        """

        self._start_analog_input()
        daq.DAQmxCfgSampClkTiming(self._scanner_ai_task, "", 200, daq.DAQmx_Val_Rising, daq.DAQmx_Val_FiniteSamps,200)
        rawdata = np.zeros(200, dtype=np.float64)
        read = daq.int32()
        daq.DAQmxStartTask(self._scanner_ai_task)
        daq.DAQmxReadAnalogF64(self._scanner_ai_task, 10000, 1.0, daq.DAQmx_Val_GroupByChannel, rawdata, 10000,
                               daq.byref(read), None)

        return rawdata


    def set_up_scanner_clock(self, clock_frequency = None, clock_channel = None):
        """ Configures the hardware clock of the NiDAQ card to give the timing.
        This is a direct pass-through to the scanner HW

        @param float clock_frequency: if defined, this sets the frequency of the clock
        @param string clock_channel: if defined, this is the physical channel of the clock

        @return int: error code (0:OK, -1:error)
        """

        return self.set_up_clock(clock_frequency=clock_frequency, clock_channel=clock_channel)

    def set_up_scanner(self, counter_channel = None, photon_source = None, clock_channel = None, scanner_ao_channels = None):
        """ Configures the actual scanner with a given clock.

        TODO this is not technically required, because the spectrometer scanner does not need clock synchronisation.

        @param string counter_channel: if defined, this is the physical channel of the counter
        @param string photon_source: if defined, this is the physical channel where the photons are to count from
        @param string clock_channel: if defined, this specifies the clock for the counter
        @param string scanner_ao_channels: if defined, this specifies the analoque output channels

        @return int: error code (0:OK, -1:error)
        """
        try:
            self.set_up_counter()
            return 0
        except:
            return -1
        return 0

    def scanner_set_position(self, x=None, y=None, z=None, a=None):
        """Move stage to x, y, z, a (where a is the fourth voltage channel).

        #FIXME: No volts
        @param float x: postion in x-direction (volts)
        @param float y: postion in y-direction (volts)
        @param float z: postion in z-direction (volts)
        @param float a: postion in a-direction (volts)

        @return int: error code (0:OK, -1:error)
        """

        #if self.getState() == 'locked':
        #    self.log.error('Another scan_line is already running, close this one first.')
        #    return -1

        if x is not None:
            if not(self._position_range[0][0] <= x <= self._position_range[0][1]):
                self.log.error('You want to set x out of range: {0:f}.'.format(x))
                return -1
            self._current_position[0] = np.float(x)

        if y is not None:
            if not(self._position_range[1][0] <= y <= self._position_range[1][1]):
                self.log.error('You want to set y out of range: {0:f}.'.format(y))
                return -1
            self._current_position[1] = np.float(y)

        if z is not None:
            if not(self._position_range[2][0] <= z <= self._position_range[2][1]):
                self.log.error('You want to set z out of range: {0:f}.'.format(z))
                return -1
            self._current_position[2] = np.float(z)

        if a is not None:
            if not(self._position_range[3][0] <= a <= self._position_range[3][1]):
                self.log.error('You want to set a out of range: {0:f}.'.format(a))
                return -1
            self._current_position[3] = np.float(a)

        # the position has to be a vstack
        my_position = np.vstack(self._current_position)
        # then directly write the position to the hardware
        try:
            self._write_scanner_ao(
                voltages=self._scanner_position_to_volt(my_position),
                start=True)
        except:
            return -1
        return 0

    def _write_scanner_ao(self, voltages, length=1, start=False):
        """Writes a set of voltages to the analog outputs.

        @param float[][n] voltages: array of n-part tuples defining the voltage
                                    points
        @param int length: number of tuples to write
        @param bool start: write imediately (True)
                           or wait for start of task (False)

        n depends on how many channels are configured for analog output
        """
        # Number of samples which were actually written, will be stored here.
        # The error code of this variable can be asked with .value to check
        # whether all channels have been written successfully.
        self._AONwritten = daq.int32()
        # write the voltage instructions for the analog output to the hardware
        daq.DAQmxWriteAnalogF64(
            # write to this task
            self._scanner_ao_task,
            # length of the command (points)
            length,
            # start task immediately (True), or wait for software start (False)
            start,
            # maximal timeout in seconds for# the write process
            self._RWTimeout,
            # Specify how the samples are arranged: each pixel is grouped by channel number
            daq.DAQmx_Val_GroupByChannel,
            # the voltages to be written
            voltages,
            # The actual number of samples per channel successfully written to the buffer
            daq.byref(self._AONwritten),
            # Reserved for future use. Pass NULL(here None) to this parameter
            None)
        return self._AONwritten.value

    def _scanner_position_to_volt(self, positions=None):
        """ Converts a set of position pixels to acutal voltages.

        @param float[][n] positions: array of n-part tuples defining the pixels

        @return float[][n]: array of n-part tuples of corresponing voltages

        The positions is typically a matrix like
            [[x_values], [y_values], [z_values], [a_values]]
            but x, xy, xyz and xyza are allowed formats.
        """

        if not isinstance(positions, (frozenset, list, set, tuple, np.ndarray, )):
            self.log.error('Given position list is no array type.')
            return np.array([np.NaN])

        vlist = []
        for i, position in enumerate(positions):
            vlist.append(
                (self._voltage_range[i][1] - self._voltage_range[i][0])
                / (self._position_range[i][1] - self._position_range[i][0])
                * (position - self._position_range[i][0])
                + self._voltage_range[i][0]
            )
        volts = np.vstack(vlist)

        for i, v in enumerate(volts):
            if v.min() < self._voltage_range[i][0] or v.max() > self._voltage_range[i][1]:
                self.log.error(
                    'Voltages ({0}, {1}) exceed the limit, the positions have to '
                    'be adjusted to stay in the given range.'.format(v.min(), v.max()))
                return np.array([np.NaN])
        return volts

    def get_scanner_position(self):
        """ Get the current position of the scanner hardware.

        @return float[]: current position in (x, y, z, a).
        """
        return self._current_position

    def _set_up_line(self, length=100):
        """ Sets up the analog output for scanning a line.

        Connect the timing of the Analog scanning task with the timing of the
        counting task.

        @param int length: length of the line in pixel

        @return int: error code (0:OK, -1:error)
        """
        if len(self._scanner_counter_daq_tasks) < 1:
            self.log.error('No counter is running, cannot scan a line without one.')
            return -1

        self._line_length = length

        try:
            # Just a formal check whether length is not a too huge number
            if length < np.inf:

                # Configure the Sample Clock Timing.
                # Set up the timing of the scanner counting while the voltages are
                # being scanned (i.e. that you go through each voltage, which
                # corresponds to a position. How fast the voltages are being
                # changed is combined with obtaining the counts per voltage peak).
                daq.DAQmxCfgSampClkTiming(
                    # add to this task
                    self._scanner_ao_task,
                    # use this channel as clock
                    self._my_scanner_clock_channel + 'InternalOutput',
                    # Maximum expected clock frequency
                    self._scanner_clock_frequency,
                    # Generate sample on falling edge
                    daq.DAQmx_Val_Falling,
                    # generate finite number of samples
                    daq.DAQmx_Val_FiniteSamps,
                    # number of samples to generate
                    self._line_length)

            # Configure Implicit Timing for the clock.
            # Set timing for scanner clock task to the number of pixel.
            daq.DAQmxCfgImplicitTiming(
                # define task
                self._scanner_clock_daq_task,
                # only a limited number of# counts
                daq.DAQmx_Val_FiniteSamps,
                # count twice for each voltage +1 for safety
                self._line_length + 1)

            for i, task in enumerate(self._scanner_counter_daq_tasks):
                # Configure Implicit Timing for the scanner counting task.
                # Set timing for scanner count task to the number of pixel.
                daq.DAQmxCfgImplicitTiming(
                    # define task
                    task,
                    # only a limited number of counts
                    daq.DAQmx_Val_FiniteSamps,
                    # count twice for each voltage +1 for safety
                    2 * self._line_length + 1)

                # Set the Read point Relative To an operation.
                # Specifies the point in the buffer at which to begin a read operation,
                # here we read samples from beginning of acquisition and do not overwrite
                daq.DAQmxSetReadRelativeTo(
                    # define to which task to connect this function
                    task,
                    # Start reading samples relative to the last sample returned
                    # by the previous read
                    daq.DAQmx_Val_CurrReadPos)

                # Set the Read Offset.
                # Specifies an offset in samples per channel at which to begin a read
                # operation. This offset is relative to the location you specify with
                # RelativeTo. Here we do not read the first sample.
                daq.DAQmxSetReadOffset(
                    # connect to this task
                    task,
                    # Offset after which to read
                    1)

                # Set Read OverWrite Mode.
                # Specifies whether to overwrite samples in the buffer that you have
                # not yet read. Unread data in buffer will be overwritten:
                daq.DAQmxSetReadOverWrite(
                    task,
                    daq.DAQmx_Val_DoNotOverwriteUnreadSamps)
        except:
            self.log.exception('Error while setting up scanner to scan a line.')
            return -1
        return 0

    def scan_line(self, line_path=None, pixel_clock=False):
        """ Scans a line and return the counts on that line.

        @param float[c][m] line_path: array of c-tuples defining the voltage points
            (m = samples per line)
        @param bool pixel_clock: whether we need to output a pixel clock for this line

        @return float[m][n]: m (samples per line) n-channel photon counts per second

        The input array looks for a xy scan of 5x5 points at the position z=-2
        like the following:
            [ [1, 2, 3, 4, 5], [1, 1, 1, 1, 1], [-2, -2, -2, -2] ]
        n is the number of scanner axes, which can vary. Typical values are 2 for galvo scanners,
        3 for xyz scanners and 4 for xyz scanners with a special function on the a axis.
        """

        if not isinstance(line_path, (frozenset, list, set, tuple, np.ndarray, ) ):
            self.log.error('Given line_path list is not array type.')
            return np.array([[-1.]])

        self._line_length = np.shape(line_path)[1]
        #self._start_analog_input()
        #daq.DAQmxCfgSampClkTiming(self._scanner_ai_task, "", 10000, daq.DAQmx_Val_Rising, daq.DAQmx_Val_FiniteSamps, 10)
        #rawdata = np.zeros(10, dtype=np.float64)
        #read = daq.int32()
        rawdata = np.zeros(self._samples_number, dtype=np.float64)
        try:
            line_volts = self._scanner_position_to_volt(line_path)

            # count data will be written here
            self.line_counts = np.zeros_like([line_path[0], ])

            for i in range(self._line_length):
                print(line_path[0][i])
                self.scanner_set_position(x=line_path[0][i], y=line_path[1][i], z=line_path[2][i])
                #daq.DAQmxStartTask(self._scanner_ai_task)
                #daq.DAQmxReadAnalogF64(self._scanner_ai_task, 10, 1.0, daq.DAQmx_Val_GroupByChannel, rawdata, 10, daq.byref(read), None)
                #self.line_counts[0, i] = rawdata.sum() / 10
                #self._stop_analog_input()
                rawdata = self.get_counter(samples=self._samples_number)
                self.line_counts[0, i] = rawdata.sum() / self._samples_number


            # stop the analog output task
            #self._stop_analog_input()
            #self._stop_analog_output()

            # update the scanner position instance variable
            self._current_position = list(line_path[:, -1])
        except:
            self.log.exception('Error while scanning line.')
            return np.array([[-1.]])
        # return values is a rate of counts/s
        #return (self.line_counts * self._scanner_clock_frequency).transpose()
        return self.line_counts.transpose()

    def close_scanner(self):
        """ Closes the scanner and cleans up afterwards.

        @return int: error code (0:OK, -1:error)
        """
        a = self._stop_analog_output()
        c = self.close_counter()
        return -1 if a < 0 else 0# or c < 0 else 0

    def close_scanner_clock(self):
        """ Closes the clock and cleans up afterwards.

        @return int: error code (0:OK, -1:error)
        """
        return self.close_clock()

    # ================ End ConfocalScannerInterface Commands ===================
    # ======================= Ramp signal Commands  ============================
    def start_ramp(self):
        """Actually start the preconfigured counter task

        @return int: error code (0:OK, -1:error)
        """
        if self._ramp_task is None:
            self.log.error(
                'Cannot start ramp since it is notconfigured!\n'
                'Run the setup_ramp_output routine.')
            return -1

        try:
            daq.DAQmxStartTask(self._ramp_task)
        except:
            self.log.exception('Error while starting ramp.')
            return -1
        return 0

    def stop_ramp(self):
        """Actually start the preconfigured counter task

        @return int: error code (0:OK, -1:error)
        """
        if self._ramp_task is None:
            self.log.error(
                'Cannot stop ramp since it is not running!\n'
                'Start the ramp singal before you can actually stop it!')
            return -1
        try:
            daq.DAQmxStopTask(self._ramp_task)
        except:
            self.log.exception('Error while stopping ramp.')
            return -1
        return 0

    def close_ramp(self):
        """ Clear tasks, so that counters are not in use any more.

        @return int: error code (0:OK, -1:error)
        """
        retval = 0
        try:
            # stop the task
            daq.DAQmxStopTask(self._ramp_task)
        except:
            self.log.exception('Error while closing ramp.')
            retval = -1
        try:
            # clear the task
            daq.DAQmxClearTask(self._ramp_task)
            self._ramp_task = None
        except:
            self.log.exception('Error while clearing ramp.')
            retval = -1
        return retval

    def ramp_function(self, t, amp, off, freq, t0):
        '''

        :param t:
        :param amp:
        :param off:
        :param freq:
        :param t0:
        :return:
        '''

        f = np.zeros(len(t))
        period = 1 / freq
        for i in range(len(t)):
            x = (t[i]-t0) % period
            if x < period / 2:
                f[i] = amp * x / (period / 2) + off - 0.5 * amp
            else:
                f[i] = 2 * amp - amp * x / (period / 2) + off - 0.5 * amp
        return f

    def set_up_ramp_output(self, amp, off, freq):
        '''
        generates a ramp task and writes a ramp signal to daq

        :param amp: Amplitude of Ramp siganl
        :param off: offset of ramp
        :param freq: Frequency of ramp signal
        :return:
        '''

        # Generate ramp signal
        t = np.linspace(0, 1, self.SampNum)
        data = self.ramp_function(t, amp, off, freq, t0=0)

        # Create task for ramp signal
        self._ramp_task = daq.TaskHandle()
        daq.DAQmxCreateTask('Ramp', daq.byref(self._ramp_task))

        if self.cavity_channel is not None:
            daq.DAQmxCreateAOVoltageChan(self._ramp_task,self.cavity_channel, "", self._cavity_voltage_range[0],
                                         self._cavity_voltage_range[1], daq.DAQmx_Val_Volts,None)
        else:
            self.log.error('No cavity channel to generate ramp, set a cavity channel in config file')

        #Use internal clock for generate the ramp
        daq.DAQmxCfgSampClkTiming(self._ramp_task,"",self.SampNum,daq.DAQmx_Val_Rising,
                                  daq.DAQmx_Val_ContSamps,self.SampNum)

        # Write data to DAQ-card
        daq.DAQmxWriteAnalogF64(self._ramp_task, self.SampNum,0,10.0,daq.DAQmx_Val_GroupByChannel,data, None, None)

        return 0

    def start_sweep(self):
        """Actually start the preconfigured counter task

        @return int: error code (0:OK, -1:error)
        """
        if self._sweep_task is None:
            self.log.error(
                'Cannot start sweep since it is notconfigured!\n'
                'Run the setup_sweep_output routine.')
            return -1

        try:
            daq.DAQmxStartTask(self._sweep_task)
        except:
            self.log.exception('Error while starting sweep.')
            return -1
        return 0

    def stop_sweep(self):
        """Actually start the preconfigured counter task

        @return int: error code (0:OK, -1:error)
        """
        if self._sweep_task is None:
            self.log.error(
                'Cannot stop sweep since it is not running!\n'
                'Start the sweep singal before you can actually stop it!')
            return -1
        try:
            daq.DAQmxStopTask(self._sweep_task)
        except:
            self.log.exception('Error while stopping sweep.')
            return -1
        return 0

    def close_sweep(self):
        """ Clear tasks, so that counters are not in use any more.

        @return int: error code (0:OK, -1:error)
        """
        retval = 0
        try:
            # stop the task
            daq.DAQmxStopTask(self._sweep_task)
        except:
            self.log.exception('Error while closing sweep.')
            retval = -1
        try:
            # clear the task
            daq.DAQmxClearTask(self._sweep_task)
            self._sweep_task = None
        except:
            self.log.exception('Error while clearing sweep.')
            retval = -1
        return retval

    def sweep_function(self, time, start, stop, freq, t0):
        ''' Creating data array for NIcard with sweep singal '''
        # array for output
        f = np.zeros(len(time), dtype=np.float64)
        period = 1 / freq
        tprime = time - t0
        slope = (stop - start) / (period / 2)
        for i in range(len(time)):
            if tprime[i] % period < (1/2) * period:
                f[i] = slope * (tprime[i] % period) + start
            else:
                f[i] = - slope * (tprime[i] % period) + 2 * stop - start
        return f

    def set_up_sweep(self, start_voltage, stop_voltage, freq, RepOfSweep):
        ''' Create the sweep task for the NIcard '''

        t = np.linspace(0, 1/freq, self.SampNum)
        data = self.sweep_function(t, start_voltage, stop_voltage, freq, t0=0)


        if self._sweep_task is not None:
            # stop the analog output task
            daq.DAQmxStopTask(self._sweep_task)

            # delete the configuration of the analog output
            daq.DAQmxClearTask(self._sweep_task)

            # set the task handle to None as a safety
            self._sweep_task = None

        self._sweep_task = daq.TaskHandle()
        daq.DAQmxCreateTask('sweep_task', daq.byref(self._sweep_task))

        if self.cavity_channel is not None:
            daq.DAQmxCreateAOVoltageChan(self._sweep_task, self.cavity_channel, "", self._cavity_voltage_range[0],
                                     self._cavity_voltage_range[1], daq.DAQmx_Val_Volts, None)
        else:
            self.log.error('No cavity channel to generate ramp, set a cavity channel in config file')

        SampRate = freq * self.SampNum
        # Use internal clock for generate the ramp
        daq.DAQmxCfgSampClkTiming(self._sweep_task, "", SampRate, daq.DAQmx_Val_Rising, daq.DAQmx_Val_FiniteSamps,
                                  RepOfSweep * self.SampNum)

        # Write data to DAQ-card
        daq.DAQmxWriteAnalogF64(self._sweep_task, self.SampNum, 0, 100.0, daq.DAQmx_Val_GroupByChannel, data, None, None)

        return 0

    def _start_cavity_analog_output(self):
        """ Starts or restarts the analog output.

        @return int: error code (0:OK, -1:error)
        """
        try:
            # If an analog task is already running, kill that one first
            if self._cavity_ao_task is not None:
                # stop the analog output task
                daq.DAQmxStopTask(self._cavity_ao_task)

                # delete the configuration of the analog output
                daq.DAQmxClearTask(self._cavity_ao_task)

                # set the task handle to None as a safety
                self._cavity_ao_task = None

            # initialize ao channels / task for scanner, should always be active.
            # Define at first the type of the variable as a Task:
            self._cavity_ao_task = daq.TaskHandle()

            # create the actual analog output task on the hardware device. Via
            # byref you pass the pointer of the object to the TaskCreation function:
            daq.DAQmxCreateTask('CavityAO', daq.byref(self._cavity_ao_task))

            # Assign and configure the created task to an analog output voltage channel.
            daq.DAQmxCreateAOVoltageChan(
                # The AO voltage operation function is assigned to this task.
                self._cavity_ao_task,
                # use (all) scanner ao_channels for the output
                self.cavity_channel,
                # assign a name for that channel
                'Scavity AO Channel',
                # minimum possible voltage
                self._cavity_voltage_range[0],
                # maximum possible voltage
                self._cavity_voltage_range[1],
                # units is Volt
                daq.DAQmx_Val_Volts,
                # empty for future use
                '')
        except:
            self.log.exception('Error starting cavity analog output task.')
            return -1
        return 0

    def _stop_cavity_analog_output(self):
        """ Stops the analog output.

        @return int: error code (0:OK, -1:error)
        """
        if self._cavity_ao_task is None:
            return -1
        retval = 0
        try:
            # stop the analog output task
            daq.DAQmxStopTask(self._cavity_ao_task)
        except:
            self.log.exception('Error stopping analog output.')
            retval = -1
        try:
            daq.DAQmxSetSampTimingType(self._cavity_ao_task, daq.DAQmx_Val_OnDemand)
        except:
            self.log.exception('Error changing analog output mode.')
            retval = -1
        return retval

    def _cavity_position_to_volt(self, position=None):
        """ Converts a set of position pixels to acutal voltages.

        @param float[][n] positions: array of n-part tuples defining the pixels

        @return float[][n]: array of n-part tuples of corresponing voltages

        The positions is typically a matrix like
            [[x_values], [y_values], [z_values], [a_values]]
            but x, xy, xyz and xyza are allowed formats.
        """

        if not isinstance(position, (frozenset, list, set, tuple, np.ndarray, )):
            self.log.error('Given position list is no array type.')
            return np.array([np.NaN])
        vlist = []
        vlist.append(float((self._cavity_voltage_range[1] - self._cavity_voltage_range[0])
            / (self._cavity_position_range[1] - self._cavity_position_range[0])
            * (position - self._cavity_position_range[0])
            + self._cavity_voltage_range[0]
            ))
        volts = np.asanyarray(vlist, dtype=float)

        if volts.min() < self._cavity_voltage_range[0] or volts.max() > self._cavity_voltage_range[1]:
            self.log.error(
                'Voltages ({0}, {1}) exceed the limit, the positions have to '
                'be adjusted to stay in the given range.'.format(volts.min(), volts.max()))
            return np.array([np.NaN])
        return volts

    def _write_cavity_ao(self, voltages, length=1, start=False):
        """Writes a set of voltages to the analog outputs.

        @param float[][n] voltages: array of n-part tuples defining the voltage
                                    points
        @param int length: number of tuples to write
        @param bool start: write imediately (True)
                           or wait for start of task (False)

        n depends on how many channels are configured for analog output
        """
        # Number of samples which were actually written, will be stored here.
        # The error code of this variable can be asked with .value to check
        # whether all channels have been written successfully.
        self._AONwrittenC = daq.int32()
        # write the voltage instructions for the analog output to the hardware
        daq.DAQmxWriteAnalogF64(
            # write to this task
            self._cavity_ao_task,
            # length of the command (points)
            length,
            # start task immediately (True), or wait for software start (False)
            start,
            # maximal timeout in seconds for# the write process
            self._RWTimeout,
            # Specify how the samples are arranged: each pixel is grouped by channel number
            daq.DAQmx_Val_GroupByChannel,
            # the voltages to be written
            voltages,
            # The actual number of samples per channel successfully written to the buffer
            daq.byref(self._AONwrittenC),
            # Reserved for future use. Pass NULL(here None) to this parameter
            None)
        return self._AONwrittenC.value

    def cavity_set_position(self, pos=None):
        """Move stage to pos.

        #FIXME: No volts
        @param float pos: postion in x-direction (volts)

        @return int: error code (0:OK, -1:error)
        """

        # if self.getState() == 'locked':
        #    self.log.error('Another scan_line is already running, close this one first.')
        #    return -1

        if pos is not None:
            if not (self._cavity_position_range[0] <= pos <= self._cavity_position_range[1]):
                self.log.error('You want to set x out of range: {0:f}.'.format(pos))
                return -1
            self._current_cavity_position[0] = np.float(pos)

        # the position has to be a vstack
        my_position = np.array(self._current_cavity_position)
        # then directly write the position to the hardware

        try:
            self._write_cavity_ao(
                voltages=self._cavity_position_to_volt(my_position),
                start=True)
        except:
            return -1
        return 0

    def cavity_set_voltage(self, voltage=None):
        """Move stage to pos.

        #FIXME: No volts
        @param float voltage: voltage in x-direction (volts)

        @return int: error code (0:OK, -1:error)
        """

        # if self.getState() == 'locked':
        #    self.log.error('Another scan_line is already running, close this one first.')
        #    return -1

        if voltage is not None:
            if not (self._cavity_voltage_range[0] <= voltage <= self._cavity_voltage_range[1]):
                self.log.error('You want to set x out of range: {0:f}.'.format(voltage))
                return -1

        try:
            self._write_cavity_ao(
                voltages=np.array(voltage),
                start=True)
        except:
            return -1
        return 0

    # ==================== End Ramp signal Commands  ===========================

    # ======================== Gated photon counting ==========================

    def set_up_gated_counter(self, buffer_length, read_available_samples=False):
        """ Initializes and starts task for external gated photon counting.

        @param int buffer_length: Defines how long the buffer to be filled with
                                  samples should be. If buffer is full, program
                                  crashes, so use upper bound. Some reference
                                  calculated with sample_rate (in Samples/second)
                                  divided by Buffer_size:
                                  sample_rate/Buffer_size =
                                      no rate     /  10kS,
                                      (0-100S/s)  /  10kS
                                      (101-10kS/s)/   1kS,
                                      (10k-1MS/s) / 100kS,
                                      (>1MS/s)    / 1Ms
        @param bool read_available_samples: if False, NiDaq waits for the
                                            sample you asked for to be in the
                                            buffer before, if True it returns
                                            what is in buffer until 'samples'
                                            is full
        """
        if self._gated_counter_daq_task is not None:
            self.log.error(
                'Another gated counter is already running, close this one first.')
            return -1

        try:
            # This task will count photons with binning defined by pulse task
            # Initialize a Task
            self._gated_counter_daq_task = daq.TaskHandle()
            daq.DAQmxCreateTask('GatedCounter', daq.byref(self._gated_counter_daq_task))

            # Set up pulse width measurement in photon ticks, i.e. the width of
            # each pulse generated by pulse_out_task is measured in photon ticks:
            daq.DAQmxCreateCIPulseWidthChan(
                # add to this task
                self._gated_counter_daq_task,
                # use this counter
                self._counter_channel,
                # name you assign to it
                'Gated Counting Task',
                # expected minimum value
                0,
                # expected maximum value
                self._max_counts,
                # units of width measurement,  here photon ticks.
                daq.DAQmx_Val_Ticks,
                # start pulse width measurement on rising edge
                self._counting_edge,
                '')

            # Set the pulses to counter self._counter_channel
            daq.DAQmxSetCIPulseWidthTerm(
                self._gated_counter_daq_task,
                self._counter_channel,
                self._gate_in_channel)

            # Set the timebase for width measurement as self._photon_source, i.e.
            # define the source of ticks for the counter as self._photon_source.
            daq.DAQmxSetCICtrTimebaseSrc(
                self._gated_counter_daq_task,
                self._counter_channel,
                self._photon_source)

            # set timing to continuous
            daq.DAQmxCfgImplicitTiming(
                # define to which task to connect this function.
                self._gated_counter_daq_task,
                # Sample Mode: set the task to generate a continuous amount of running samples
                daq.DAQmx_Val_ContSamps,
                # buffer length which stores temporarily the number of generated samples
                buffer_length)

            # Read samples from beginning of acquisition, do not overwrite
            daq.DAQmxSetReadRelativeTo(self._gated_counter_daq_task, daq.DAQmx_Val_CurrReadPos)

            # If this is set to True, then the NiDaq will not wait for the sample
            # you asked for to be in the buffer before read out but immediately
            # hand back all samples until samples is reached.
            if read_available_samples:
                daq.DAQmxSetReadReadAllAvailSamp(self._gated_counter_daq_task, True)

            # Do not read first sample:
            daq.DAQmxSetReadOffset(self._gated_counter_daq_task, 0)

            # Unread data in buffer is not overwritten
            daq.DAQmxSetReadOverWrite(
                self._gated_counter_daq_task,
                daq.DAQmx_Val_DoNotOverwriteUnreadSamps)
        except:
            self.log.exception('Error while setting up gated counting.')
            return -1
        return 0

    def start_gated_counter(self):
        """Actually start the preconfigured counter task

        @return int: error code (0:OK, -1:error)
        """
        if self._gated_counter_daq_task is None:
            self.log.error(
                'Cannot start Gated Counter Task since it is notconfigured!\n'
                'Run the set_up_gated_counter routine.')
            return -1

        try:
            daq.DAQmxStartTask(self._gated_counter_daq_task)
        except:
            self.log.exception('Error while starting up gated counting.')
            return -1
        return 0


    def get_gated_counts(self, samples=None, timeout=None, read_available_samples=False):
        """ Returns latest count samples acquired by gated photon counting.

        @param int samples: if defined, number of samples to read in one go.
                            How many samples are read per readout cycle. The
                            readout frequency was defined in the counter setup.
                            That sets also the length of the readout array.
        @param int timeout: Maximal timeout for the read process. Since nidaq
                            waits for all samples to be acquired, make sure
                            this is long enough.
        @param bool read_available_samples : if False, NiDaq waits for the
                                             sample you asked for to be in the
                                             buffer before, True it returns
                                             what is in buffer until 'samples'
                                             is full.
        """
        if samples is None:
            samples = int(self._samples_number)
        else:
            samples = int(samples)

        if timeout is None:
            timeout = self._RWTimeout

        # Count data will be written here
        _gated_count_data = np.empty([2,samples], dtype=np.uint32)

        # Number of samples which were read will be stored here
        n_read_samples = daq.int32()

        if read_available_samples:
            # If the task acquires a finite number of samples
            # and you set this parameter to -1, the function
            # waits for the task to acquire all requested
            # samples, then reads those samples.
            num_samples = -1
        else:
            num_samples = int(samples)
        try:
            daq.DAQmxReadCounterU32(
                # read from this task
                self._gated_counter_daq_task,
                # read number samples
                num_samples,
                # maximal timeout for the read process
                timeout,
                _gated_count_data[0],
                # write into this array
                # length of array to write into
                samples,
                # number of samples which were actually read.
                daq.byref(n_read_samples),
                # Reserved for future use. Pass NULL (here None) to this parameter
                None)

            # Chops the array or read sample to the length that it exactly returns
            # acquired data and not more
            if read_available_samples:
                return _gated_count_data[0][:n_read_samples.value], n_read_samples.value
            else:
                return _gated_count_data
        except:
            self.log.exception('Error while reading gated count data.')
            return np.array([-1])

    def stop_gated_counter(self):
        """Actually start the preconfigured counter task

        @return int: error code (0:OK, -1:error)
        """
        if self._gated_counter_daq_task is None:
            self.log.error(
                'Cannot stop Gated Counter Task since it is not running!\n'
                'Start the Gated Counter Task before you can actually stop it!')
            return -1
        try:
            daq.DAQmxStopTask(self._gated_counter_daq_task)
        except:
            self.log.exception('Error while stopping gated counting.')
            return -1
        return 0

    def close_gated_counter(self):
        """ Clear tasks, so that counters are not in use any more.

        @return int: error code (0:OK, -1:error)
        """
        retval = 0
        try:
            # stop the task
            daq.DAQmxStopTask(self._gated_counter_daq_task)
        except:
            self.log.exception('Error while closing gated counter.')
            retval = -1
        try:
            # clear the task
            daq.DAQmxClearTask(self._gated_counter_daq_task)
            self._gated_counter_daq_task = None
        except:
            self.log.exception('Error while clearing gated counter.')
            retval = -1
        return retval


class SlowGatedNICard(NICard):
    """ Enable the usage of the gated counter in the slow counter interface.
    Overwrite in this new class therefore the appropriate methods. """

    _modtype = 'SlowGatedNICard'
    _modclass = 'hardware'

    def on_activate(self):
        """ Starts up the NI Card at activation.
        """
        self._gated_counter_daq_task = None
        # used as a default for expected maximum counts
        self._max_counts = 3e7
        # timeout for the Read or/and write process in s
        self._RWTimeout = 5
        # in Hz
        self._clock_frequency_default = 100
        # number of readout samples mainly used for gated counter
        self._samples_number_default = 50
        # count on rising edge mainly used for gated counter
        self._counting_edge_default = True

        self._counter_channels = []

        self._counter_channel = '/NIDAQ/Ctr0'

        config = self.getConfiguration()

        if 'photon_source' in config.keys():
            self._photon_source=config['photon_source']
        else:
            self.log.error(
                'No parameter "photon_source" configured.\n'
                'Assign to that parameter an appropriated channel from your NI Card!')

        if 'gate_in_channel' in config.keys():
            self._gate_in_channel = config['gate_in_channel']
        else:
            self.log.error(
                'No parameter "gate_in_channel" configured. '
                'Choose the proper channel on your NI Card and assign it to that parameter!')

        if 'counting_edge_rising' in config.keys():
            if config['counting_edge_rising']:
                self._counting_edge = daq.DAQmx_Val_Rising
            else:
                self._counting_edge = daq.DAQmx_Val_Falling
        else:
            self.log.warning(
                'No parameter "counting_edge_rising" configured.\n'
                'Set this parameter either to True (rising edge) or to False (falling edge).\n'
                'Taking the default value {0}'.format(self._counting_edge_default))

            self._counting_edge = self._counting_edge_default

        if 'samples_number' in config.keys():
            self._samples_number = config['samples_number']
        else:
            self._samples_number = self._samples_number_default
            self.log.warning(
                'No parameter "samples_number" configured taking the default value "{0}" instead.'
                ''.format(self._samples_number_default))
            self._samples_number = self._samples_number_default

    def get_constraints(self):
        """ Get hardware limits of NI device.

        @return SlowCounterConstraints: constraints class for slow counter

        FIXME: ask hardware for limits when module is loaded
        """
        constraints = SlowCounterConstraints()
        constraints.max_detectors = 4
        constraints.min_count_frequency = 1e-3
        constraints.max_count_frequency = 10e9
        constraints.counting_mode = [CountingMode.FINITE_GATED]
        return constraints

    #overwrite the SlowCounterInterface commands of the class NICard:
    def set_up_clock(self, clock_frequency=None, clock_channel=None):
        """ Configures the hardware clock of the NiDAQ card to give the timing.

        @param float clock_frequency: if defined, this sets the frequency of
                                      the clock
        @param string clock_channel: if defined, this is the physical channel
                                     of the clock

        @return int: error code (0:OK, -1:error)
        """
        # ignore that command. For an gated counter (with external trigger
        # you do not need a clock signal).
        return 0

    def set_up_counter(self,
                       counter_channel=None,
                       photon_source=None,
                       counter_channel2=None,
                       photon_source2=None,
                       clock_channel=None,
                       counter_buffer=None):
        """ Configures the actual counter with a given clock.

        @param str counter_channel: optional, physical channel of the counter
        @param str photon_source: optional, physical channel where the photons
                                  are to count from
        @param str counter_channel2: optional, physical channel of the counter 2
        @param str photon_source2: optional, second physical channel where the
                                   photons are to count from
        @param str clock_channel: optional, specifies the clock channel for the
                                  counter
        @param int counter_buffer: optional, a buffer of specified integer
                                   length, where in each bin the count numbers
                                   are saved.

        @return int: error code (0:OK, -1:error)
        """
        if self.set_up_gated_counter(buffer_length=counter_buffer) < 0:
            return -1
        return self.start_gated_counter()

    def get_counter(self, samples=None):
        """ Returns the current counts per second of the counter.

        @param int samples: if defined, number of samples to read in one go

        @return numpy.array(uint32): the photon counts per second
        """
        return self.get_gated_counts(samples=samples)

    def close_counter(self):
        """ Closes the counter and cleans up afterwards.

        @return int: error code (0:OK, -1:error)
        """
        if self.stop_gated_counter() < 0:
            return -1
        return self.close_gated_counter()

    def close_clock(self):
        """ Closes the clock and cleans up afterwards.

        @return int: error code (0:OK, -1:error)
        """
        return 0



