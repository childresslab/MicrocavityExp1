from core.module import Base
from interface.scope_interface import ScopeInterface
import visa
import string
import sys
import time
import numpy as np
from struct import unpack

class Scope(Base, ScopeInterface):
    """
    This is the Interface class to define the controls for the simple
    microwave hardware.
    """
    _modclass = 'scopeinterface'
    _modtype = 'hardware'

    def __init__(self, config, **kwargs):
        super().__init__(config=config, **kwargs)
        self.log.info('The following configuration was found.')
        # checking for the right configuration
        for key in config.keys():
            self.log.info('{0}: {1}'.format(key, config[key]))
        self.rm = visa.ResourceManager()
        self.res = self.rm.list_resources()

    def on_activate(self):
        """
        Initialisation performed during activation of the module.
        """
        config = self.getConfiguration()
        self.scope = self.rm.open_resource(self.res[0])
        self.scope.timeout = 1000 # ms
        self.scope.read_termination = '\n'
        self.scope.write_termination = '\n'
        k = 0
        while k < 5:
            try:
                print('Connected to ' + self.scope.query('*IDN?'))
                break
            except:
                self.log.error('Did not connect to scope!!!')
                k += 1

        return 0

    def on_deactivate(self):
        """ Deinitialisation performed during deactivation of the module.
        """
        self.scope.close()
        #self.rm.close()
        return

    def set_time_scale(self, division):
        '''
        Set the time on the for one division


        :param division:
        :return:
        '''
        #FIXME: Only certain values are alloed!

        self.scope.write('HORizontal:Scale {}'.format(division))

    def set_acquisition_time(self, acquisition_time):
        '''
        Sets the total time on the horisontal axis on the scope

        :param acquisition_time:
        :return:
        '''
        division = acquisition_time / 10
        self.set_time_scale(division)

    def aquire_data(self, linewidth=False):
        '''
        Aquires data from scope in all four channels
        Full reslution is chosen
        
        :return: times, volts
        
        '''
        channels = ['CH1', 'CH2', 'CH3', 'CH4']
        volts = np.array([])
        times = np.array([])
        for channel in channels:
            self.scope.write('DATA:SOU {}'.format(channel))
            self.scope.write('DATA:WIDTH 1')
            self.scope.write('DATA:ENC RPB')
            if linewidth is False:
                #self.scope.write('HORizontal:RECordlength 1250000')
                self.scope.write('DATA:Start 1')
                self.scope.write('DATA:Stop 1250000')
            else:
                #self.scope.write('HORizontal:RECordlength 100000')
                self.scope.write('DATA:Start 1')
                self.scope.write('DATA:Stop 100000')

            self.scope.write('DATA:Resolution FULL')


            ymult = float(self.scope.ask('WFMPRE:YMULT?'))
            yzero = float(self.scope.ask('WFMPRE:YZERO?'))
            yoff = float(self.scope.ask('WFMPRE:YOFF?'))
            xincr = float(self.scope.ask('WFMPRE:XINCR?'))

            self.scope.write('CURVE?')
            i = 0
            while i < 5:
                try:
                    data = self.scope.read_raw()
                    break
                except:
                    self.log.info('Communication to scope failed')
                    i += 1


            headerlen = 2 + int(data[1])
            header = data[:headerlen]
            ADC_wave = data[headerlen:-1]

            ADC_wave = np.array(unpack('%sB' % len(ADC_wave), ADC_wave))

            volt = (ADC_wave - yoff) * ymult + yzero
            #time = np.arange(0, xincr * len(volt), xincr)
            time = np.linspace(0, xincr * (len(volt)-1),len(volt))
            #2000: to get rid of wired start from scope
            volts = np.concatenate([volts, volt])
            times = np.concatenate([times, time])

        return times, volts

    # General functionn

    def run_continuous(self):
        '''
        Starts continous run on scope
        
        :return: 
        '''
        self.scope.write('ACQuire:STOPAfter Runstop')
        self.scope.write(':ACQuire:STATE on')


    def stop_acquisition(self):
        self.scope.write(':ACQuire:STATE off')

    def run_single(self):
        self.scope.write(':ACQuire:STATE off')
        self.scope.write('ACQuire:STOPAfter SEQ')
        self.scope.write(':ACQuire:STATE on')


    def set_egde_trigger(self, channel, level, slope = 'FALL'):

        while True:
            try:
                self.scope.write('Trigger:A:Type Edge')
                self.scope.write('Trigger:A:Coupling DC')
                self.scope.write('Trigger:A:Edge:Slope '+ slope)
                self.scope.write('Trigger:A:Edge:Source CH{}'.format(channel))
                self.scope.write('Trigger:A:Level:CH{} {}'.format(channel, level))
                self.scope.write('HORizontal:DELay:TIMe 0.0')
                break
            except:
                time.sleep(1)
                self.log.warning('Failed to set up scope')
                self.on_deactivate()
                self.on_activate()
                continue

    def set_vertical_scale(self, channel, scale):
        self.scope.write('CH{}:Scale {}'.format(channel, scale))

    def set_vertical_position(self, channel, position):
        self.scope.write('CH{}:Position {}'.format(channel, position))

    def set_data_composition_to_env(self):
        self.scope.write('DATA:COMPOSITION COMPOSITE_ENV')

    def set_data_composition_to_yt(self):
        self.scope.write('DATA:COMPOSITION COMPOSITE_YT')

    def set_record_lenght(self, linewidth=False):
        if linewidth is False:
            self.scope.write('HORizontal:RECordlength 1250000')
        else:
            self.scope.write('HORizontal:RECordlength 100000')
