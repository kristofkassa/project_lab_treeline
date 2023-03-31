import sys
import pathlib
import time

from PySide6 import  QtGui, QtWidgets
from PySide6.QtCore import Qt

from simulation.hco_simulation import HomogeneousContactProcessSimulationStrategy
from simulation.simulation_context import SimulationContext
from .main_window.main_window import MainWindow


HERE: pathlib.Path = pathlib.Path(__file__).parent


def __create_application() -> QtWidgets.QApplication:
    app = QtWidgets.QApplication(sys.argv)
    return app

def __show_splash() -> QtWidgets.QSplashScreen:
    splash = QtWidgets.QSplashScreen(QtGui.QPixmap(str(HERE.joinpath("images/splash.png"))))
    splash.show()
    return splash

def __show_main_window(splash: QtWidgets.QSplashScreen) -> MainWindow:
    splash.showMessage("Creating main window...", Qt.AlignBottom, QtGui.QColor("#F0F0F0"))
    icon = QtGui.QIcon(str(HERE.joinpath("images/icon.png")))
    defaultStrategy = HomogeneousContactProcessSimulationStrategy()
    context = SimulationContext(defaultStrategy)
    window = MainWindow(icon, context)
    window.show()
    splash.finish(window)
    return window

def main():
    app = __create_application()
    splash = __show_splash()
    time.sleep(1)
    window = __show_main_window(splash)
    return app.exec_()