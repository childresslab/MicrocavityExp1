# -*- coding: utf-8 -*-
"""
This file contains the Qudi GUI module for ODMR control.

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
import os
import pyqtgraph as pg

from core.module import Connector
from core.util import units
from gui.guibase import GUIBase
from gui.guiutils import ColorBar
from gui.colordefs import ColorScaleInferno
from gui.colordefs import QudiPalettePale as palette
from gui.fitsettings import FitSettingsDialog, FitSettingsComboBox
from qtpy import QtCore
from qtpy import QtWidgets
from qtpy import uic


class CavityMainWindow(QtWidgets.QMainWindow):
    """ The main window for the Cavity measurement GUI.
    """
    def __init__(self):
        # Get the path to the *.ui file
        this_dir = os.path.dirname(__file__)
        ui_file = os.path.join(this_dir, 'ui_cavitygui.ui')

        # Load it
        super(CavityMainWindow, self).__init__()
        uic.loadUi(ui_file, self)
        self.show()


class CavityGui(GUIBase):
    """
    This is the GUI Class for Cavity measurements
    """

    _modclass = 'CavityGui'
    _modtype = 'gui'

    # declare connectors
    cavitylogic1 = Connector(interface='CavityLogic')
    savelogic = Connector(interface='SaveLogic')

    sigStartCavityScan = QtCore.Signal()
    sigStopCavityScan = QtCore.Signal()
    sigContinueCavityScan = QtCore.Signal()



    def __init__(self, config, **kwargs):
        super().__init__(config=config, **kwargs)

    def on_activate(self):
        """ Definition, configuration and initialisation of the Cavity GUI.

        This init connects all the graphic modules, which were created in the
        *.ui file and configures the event handling between the modules.
        """

        self._cavity_logic = self.get_connector('cavitylogic1')

        # Use the inherited class 'Ui_CavityGuiUI' to create now the GUI element:
        self._mw = CavityMainWindow()

        self.fullsweep_image = pg.PlotDataItem(self._cavity_logic.RampUp_time,
                                          self._cavity_logic.RampUp_signalR,
                                          pen=pg.mkPen(palette.c2, style=QtCore.Qt.DotLine),
                                          symbol='o',
                                          symbolPen=palette.c2,
                                          symbolBrush=palette.c2,
                                          symbolSize=2)

        self._mw.fullsweep_PlotWidget.setLabel(axis='left', text='Signal', units='V')
        self._mw.fullsweep_PlotWidget.setLabel(axis='bottom', text='Time', units='s')
        self._mw.fullsweep_PlotWidget.showGrid(x=True, y=True, alpha=0.8)

        self._mw.fullsweep_PlotWidget.addItem(self.fullsweep_image)

        self.linewidth_image = pg.PlotDataItem(self._cavity_logic.RampUp_time,
                                               self._cavity_logic.RampUp_signalR,
                                               pen=pg.mkPen(palette.c3, style=QtCore.Qt.DotLine),
                                               symbol='o',
                                               symbolPen=palette.c3,
                                               symbolBrush=palette.c3,
                                               symbolSize=2)

        self._mw.linewidth_PlotWidget.setLabel(axis='left', text='Signal', units='V')
        self._mw.linewidth_PlotWidget.setLabel(axis='bottom', text='Time', units='s')
        self._mw.linewidth_PlotWidget.showGrid(x=True, y=True, alpha=0.8)

        self._mw.linewidth_PlotWidget.addItem(self.linewidth_image)
        # Create a QSettings object for the mainwindow and store the actual GUI layout
        self.mwsettings = QtCore.QSettings("QUDI", "Cavity")

        #Connect bottons
        self._mw.fullsweeptest_PushBotton.clicked.connect(self.fullsweeptest)
        self._mw.start_finesse_PushBotton.clicked.connect(self.start_finesse_measurement)
        self._mw.stop_finesse_PushBotton.clicked.connect(self.stop_finesse_measurement)
        self._mw.continue_finesse_PushBotton.clicked.connect(self.contiune_finesse_measurement)
        self._mw.GOpushButton.clicked.connect(self.set_cavity_position)

        # Signals from logic
        self._cavity_logic.sigFullSweepPlotUpdated.connect(self.update_fullsweep_plot, QtCore.Qt.QueuedConnection)
        self._cavity_logic.sigLinewidthPlotUpdated.connect(self.update_linewidth_plot, QtCore.Qt.QueuedConnection)
        self._cavity_logic.sigSweepNumberUpdated.connect(self.update_sweep_number)
        self._cavity_logic.sigTargetModeNumberUpdated.connect(self.update_mode_number)

        # FIXME: Shoud be from Hardware
        # Adjust range of scientific spinboxes above what is possible in Qt Designer
        constraints = self._cavity_logic.get_hw_constraints()
        #self._mw.ramp_frequency_DoubleSpinBox.setMaximum(constraints.max_frequency)
        self._mw.ramp_frequency_DoubleSpinBox.setMaximum(50)
        self._mw.ramp_frequency_DoubleSpinBox.setMinimum(0)
        #self._mw.ramp_frequency_DoubleSpinBox.setOpts(minStep=0.5)  # set the minimal step to 0.5Hz
        self._mw.ramp_offset_DoubleSpinBox.setMaximum(0)
        self._mw.ramp_offset_DoubleSpinBox.setMinimum(-3.75)
        self._mw.ramp_amplitude_DoubleSpinBox.setMaximum(0)
        self._mw.ramp_amplitude_DoubleSpinBox.setMinimum(-3.75/2)

        self._mw.StartRamp_PushButton.clicked.connect(self.start_ramp)
        self._mw.StopRamp_PushButton.clicked.connect(self.stop_ramp)



    def on_deactivate(self):
        """ Reverse steps of activation

        @return int: error code (0:OK, -1:error)
        """
        self._mw.close()
        return 0

    def _show(self):
        """Make window visible and put it above all other windows. """
        self._mw.show()
        self._mw.activateWindow()
        self._mw.raise_()
        return

    def fullsweeptest(self):
        """

        @return:
        """
        self._cavity_logic.get_nth_full_sweep(sweep_number=1, save=False)


    def update_fullsweep_plot(self, time, signal):
        self.fullsweep_image.setData(time, signal)
        # Update raw data matrix plot

    def update_linewidth_plot(self, time, signal):
        self.linewidth_image.setData(time, signal)
        # Update raw data matrix plot

    def start_ramp(self):
        """

        @return:
        """

        amplitude = self._mw.ramp_amplitude_DoubleSpinBox.value()
        offset = self._mw.ramp_offset_DoubleSpinBox.value()
        freq = self._mw.ramp_frequency_DoubleSpinBox.value()

        self._cavity_logic.start_ramp(amplitude, offset, freq)

        # Disable changes to parameters
        self._mw.ramp_amplitude_DoubleSpinBox.setEnabled(False)
        self._mw.ramp_offset_DoubleSpinBox.setEnabled(False)
        self._mw.ramp_frequency_DoubleSpinBox.setEnabled(False)


    def stop_ramp(self):
        self._cavity_logic.stop_ramp()

        # Enable changes to parameters
        self._mw.ramp_amplitude_DoubleSpinBox.setEnabled(True)
        self._mw.ramp_offset_DoubleSpinBox.setEnabled(True)
        self._mw.ramp_frequency_DoubleSpinBox.setEnabled(True)

    def start_finesse_measurement(self):
        #self._cavity_logic.start_finesse_measurement()
        pass

    def stop_finesse_measurement(self):
        pass

    def contiune_finesse_measurement(self):
        pass

    def update_mode_number(self, value):
        self._mw.ModeNumber_DoubleSpinBox.setValue(value=value)

    def update_sweep_number(self, value):
        self._mw.SweepNumber_DoubleSpinBox.setValue(value=value)

    def set_cavity_position(self):
        self._cavity_logic.set_cavity_position(self._mw.doubleSpinBox_2.value()*1e-6)







