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
                                          pen=pg.mkPen(palette.c1, style=QtCore.Qt.DotLine),
                                          symbol='o',
                                          symbolPen=palette.c1,
                                          symbolBrush=palette.c1,
                                          symbolSize=2)

        self._mw.fullsweep_PlotWidget.setLabel(axis='left', text='Signal', units='V')
        self._mw.fullsweep_PlotWidget.setLabel(axis='bottom', text='Time', units='s')
        self._mw.fullsweep_PlotWidget.showGrid(x=True, y=True, alpha=0.8)

        self._mw.fullsweep_PlotWidget.addItem(self.fullsweep_image)
        # Create a QSettings object for the mainwindow and store the actual GUI layout
        self.mwsettings = QtCore.QSettings("QUDI", "Cavity")

        self._mw.fullsweeptest_PushBotton.clicked.connect(self.fullsweeptest)

        self._cavity_logic.sigFullSweepPlotUpdated.connect(self.update_fullsweep_plot, QtCore.Qt.QueuedConnection)


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
        print('test')

    def update_fullsweep_plot(self, time, signal):
        self.fullsweep_image.setData(time, signal)
        # Update raw data matrix plot



