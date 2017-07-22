from core.base import Base
from interface.scope_interface import ScopeInterface
import visa
import string
import sys
import numpy as np

class Scope3024T(Base, ScopeInterface):
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
        return

    def on_deactivate(self):
        """ Deinitialisation performed during deactivation of the module.
        """
        self.scope.close()
        self.rm.close()
        return

    def aquire_data(self, channels):
        self._do_command(':ACQuire:TYPE NORMal')
        self._do_command(':TIMebase:MODE MAIN')
        self._do_command(':WAVeform:FORMat BYTE')  # 8 bits
        self._do_command(':WAVeform:UNSigned ON')
        self._do_command(':WAVeform:POINTs:MODE MAXimum')
        self._do_command(':WAVeform:POINTs 8000000')
        y_data = []
        t_data = []
        self.stop_acquisition()

        for channel in channels:
            self._do_command(':WAVeform:SOURce CHANnel{}'.format(channel))
            timedata, y = self._get_data()
            y_data.append(y)
        if len(channels)>0:
            t_data = timedata
        self.run_continuous()
        return t_data, y_data



    # General functions

    def auto_scale(self):
        self.do_command(":AUToscale")

    def run_continuous(self):
        self._do_command(':run')

    def run_single(self):
        self._do_command(':SINGle')

    def stop_acquisition(self):
        self._do_command(':stop')

    def set_time_range(self, time_range):
        self._do_command(':Timebase:RANGe ' + str(time_range))

    def get_time_range(self):
        return self._do_query_ascii_values(':Timebase:RANGe?')

    def set_voltage_range(self, channel, voltage_range):
        self._do_command(':Channel{}:RANGe '.format(channel) + str(voltage_range) + 'V')

    def get_voltage_range(self, channel):
        return self._do_query_ascii_values(':Channel{}:Range?'.format(channel))

    def _get_data(self):
        data = self._do_query_binary_values('WAVeform:DATA?')
        preamble = self._do_query_ascii_values('WAVeform:PREamble?')

        self.t_data = self._convert_t_data(data, preamble)
        self.y_data = self._convert_y_data(data, preamble)

        return self.t_data, self.y_data

    def _convert_y_data(self, data, preamble):
        return (data - preamble[9]) * preamble[7] + preamble[8]

    def _convert_t_data(self, data, preamble):
        t = np.linspace(0, len(data) - 1, len(data))
        return (t - preamble[6]) * preamble[4] + preamble[5]



    # Acquisition functions

    def acquire_mode_normal(self):
        self._do_command(':ACQuire:TYPE NORMal')

    def aqcuire_mode_highres(self):
        self._do_command(':ACQuire:TYPE HRESolution')

    def aqcuire_mode_peak(self):
        self._do_command(':ACQuire:TYPE PEAK')

    def aqcuire_mode_average(self):
        self._do_command(':ACQuire:TYPE AVERage')




    # Channel 1 functions

    def set_channel1_vscale(self, value):
        self._do_command('CHANnel1:SCALe {0:.1E}'.format(value))

    def set_channel1_DC_couling(self):
        self._do_command('CHANnel1:COUPling DC')

    def set_channel1_AC_couling(self):
        self._do_command('CHANnel1:COUPling AC')

    def set_channel1_impedance_input_50(self):
        self._do_command('CHANnel1:IMPedance FIFT')

    def set_channel1_impedance_input_1M(self):
        self._do_command('CHANnel1:IMPedance ONEM')




    # Channel 2 functions

    def set_channel2_vscale(self, value):
        self._do_command('CHANnel2:SCALe {0:.1E}'.format(value))

    def set_channel2_DC_couling(self):
        self._do_command('CHANnel2:COUPling DC')

    def set_channel2_AC_couling(self):
        self._do_command('CHANnel2:COUPling AC')

    def set_channel2_impedance_input_50(self):
        self._do_command('CHANnel2:IMPedance FIFT')

    def set_channel2_impedance_input_1M(self):
        self._do_command('CHANnel2:IMPedance ONEM')




    # Channel 3 functions

    def set_channel3_vscale(self, value):
        self._do_command('CHANnel3:SCALe {0:.1E}'.format(value))

    def set_channel3_DC_couling(self):
        self._do_command('CHANnel3:COUPling DC')

    def set_channel3_AC_couling(self):
        self._do_command('CHANnel3:COUPling AC')

    def set_channel3_impedance_input_50(self):
        self._do_command('CHANnel3:IMPedance FIFT')

    def set_channel3_impedance_input_1M(self):
        self._do_command('CHANnel3:IMPedance ONEM')




    # Channel 4 functions

    def set_channel4_vscale(self, value):
        self._do_command('CHANnel4:SCALe {0:.1E}'.format(value))

    def set_channel4_DC_couling(self):
        self._do_command('CHANnel4:COUPling DC')

    def set_channel4_AC_couling(self):
        self._do_command('CHANnel4:COUPling AC')

    def set_channel4_impedance_input_50(self):
        self._do_command('CHANnel4:IMPedance FIFT')

    def set_channel4_impedance_input_1M(self):
        self._do_command('CHANnel4:IMPedance ONEM')



    # All channels functions

    def get_channels(self):
        return [1,2,3,4]

    def turn_on_channel(self, channel):
        self._do_command(":Channel{}:DISPlay ON".format(int(channel)))

    def turn_off_channel(self, channel):
        self._do_command(":Channel{}:DISPlay OFF".format(int(channel)))



    # Trigger functions

    def trigger_mode_EDGE(self):
        self._do_command(':TRIGger:MODE EDGE')




    # =========================================================
    # Send a command and check for errors:
    # =========================================================
    def _do_command(self, command, hide_params=False):
        if hide_params:
            (header, data) = string.split(command, " ", 1)

        self.scope.write("{}".format(command))
        if hide_params:
            self._check_instrument_errors(header)
        else:
            self._check_instrument_errors(command)

    # =========================================================
    # Send a query, check for errors, return string:
    # =========================================================
    def _do_query(self, query):

        result = self.scope.query("{}".format(query))
        self._check_instrument_errors(query)
        return result

    # =========================================================
    # Send a query, check for errors, return values:
    # =========================================================
    def _do_query_values(self, query):

        results = self.scope.ask_for_values("{}".format(query))
        self._check_instrument_errors(query)
        return results

    # =========================================================
    # Send a query, check for errors, return values:
    # =========================================================

    def _do_query_ascii_values(self, query):

        results = self.scope.query_ascii_values("{}".format(query), container=np.array)
        self._check_instrument_errors(query)
        return results

    # =========================================================
    # Send a query, check for errors, return values:
    # =========================================================

    def _do_query_binary_values(self, query):

        results = self.scope.query_binary_values("{}".format(query), container=np.array, is_big_endian=False, datatype='B')
        self._check_instrument_errors(query)
        return results

    # =========================================================
    # Check for instrument errors:
    # =========================================================
    def _check_instrument_errors(self, command):
        while True:
            error_string = self.scope.ask(":SYSTem:ERRor?")

            if error_string:  # If there is an error string value.
                if error_string.find("+0,", 0, 3) == -1:  # Not "No error".
                    print("ERROR: {}, command: '{}'".format(error_string, command))
                    print("Exited because of error.")
                    sys.exit(1)
                else:
                    break

            else:  # :SYSTem:ERRor? should always return string.
                print("ERROR: :SYSTem:ERRor? returned nothing, command: '{}'".format(command))
                print("Exited because of error.")
                sys.exit(1)
