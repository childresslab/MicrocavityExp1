from core.module import Base
from interface.scope_interface import ScopeInterface
import numpy as np

class ScopeDummy(Base, ScopeInterface):
    """This is the Interface class to define the controls for the simple
    microwave hardware.
    """
    _modclass = 'scopeinterface'
    _modtype = 'hardware'

    def __init__(self, config, **kwargs):
        super().__init__(config=config, **kwargs)

        self.log.info('The following configuration was found.')

        # checking for the right configuration
        for key in config.keys():
            self.log.info('{0}: {1}'.format(key,config[key]))

    def on_activate(self):
        """ Initialisation performed during activation of the module.
        """
        self.statusvar = 0
        self.channel = {1: 'on',
                        2: 'on',
                        3: 'on',
                        4: 'on'}
        return

    def on_deactivate(self):
        """ Deinitialisation performed during deactivation of the module.
        """
        self.statusvar = -1
        return

    def run_continuous(self):
        pass

    def run_single(self):
        pass

    def stop_acquisition(self):
        pass

    def get_channels(self):
        return [1, 2, 3, 4]

    def turn_on_channel(self, channel):
        self.channel[channel] = 'on'

    def turn_off_channel(self, channel):
        self.channel[channel] = 'off'

    def aquire_data(self, channels):
        y_data = []

        for i in range(len(channels)):
            y = 1
            y_data.append(y)
        t_data = np.linspace(0,1,1)
        return t_data, y_data