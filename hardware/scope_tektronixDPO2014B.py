from core.module import Base
from interface.scope_interface import ScopeInterface
import visa
import string
import sys
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
        self.scope.timeout = 1000
        self.scope.read_termination = None
        self.scope.write_termination = '\n'
        print('Connected to ' + self.scope.query('*IDN?'))
        self.single = False

        return

    def on_deactivate(self):
        """ Deinitialisation performed during deactivation of the module.
        """
        self.scope.close()
        self.rm.close()
        return

    def set_time_scale(self, division):
        self.scope.write('HORizontal:Scale {}'.format(division))

    def aquire_data(self):
        channels = ['CH1', 'CH2', 'CH3']
        volts = np.array([])
        times = np.array([])
        for channel in channels:
            self.scope.write('DATA:SOU {}'.format(channel))
            self.scope.write('DATA:WIDTH 1')
            self.scope.write('DATA:ENC RPB')

            ymult = float(self.scope.ask('WFMPRE:YMULT?'))
            yzero = float(self.scope.ask('WFMPRE:YZERO?'))
            yoff = float(self.scope.ask('WFMPRE:YOFF?'))
            xincr = float(self.scope.ask('WFMPRE:XINCR?'))

            self.scope.write('CURVE?')
            data = self.scope.read_raw()
            headerlen = 2 + int(data[1])
            header = data[:headerlen]
            ADC_wave = data[headerlen:-1]

            ADC_wave = np.array(unpack('%sB' % len(ADC_wave), ADC_wave))

            volt = (ADC_wave - yoff) * ymult + yzero
            time = np.arange(0, xincr * len(volt), xincr)

            volts = np.concatenate([volts, volt])
            times = np.concatenate([times, time])

        return times, volts

    # General functionn

    def run_continuous(self):
        if self.single is True:
            self.scope.write('ACQuire:STOPAfter runstop')
            self.single = False

        self.scope.write(':ACQuire:STATE on')

    def stop_acquisition(self):
        self.scope.write(':ACQuire:STATE off')

    def run_single(self):
        if self.single is True:
            self.scope.write(':ACQuire:STATE on')
        else:
            self.scope.write('ACQuire:STOPAfter SEQ')
            self.single = True
            self.scope.write(':ACQuire:STATE on')


