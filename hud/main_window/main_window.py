from PySide6.QtWidgets import QMainWindow, QVBoxLayout, QGraphicsView, QGraphicsScene, QGroupBox, QHBoxLayout, QComboBox, QPushButton, QWidget, QLineEdit
from PySide6 import QtGui, QtCore
from simulation.gcp_simulation import GradientContactProcessSimulationStrategy
from simulation.grm_simulation import GradientRandomMapSimulationStrategy
from simulation.hco_simulation import HomogeneousContactProcessSimulationStrategy
from simulation.image_treeline_simulation import ImageTreelineSimulationStrategy
from simulation.simulation_context import SimulationContext

from PySide6.QtGui import QImage, QPixmap, QPainter, QColor
from PySide6.QtCore import QRect, Qt

import pyqtgraph as pg
import numpy as np

grid_size = 5

class CellularAutomataGridView(QGraphicsView):
    """The Qt Graphics Grid where the whole simulation takes place.
    """

    def __init__(self, context: SimulationContext, plotWidget):
        QGraphicsView.__init__(self)

        self.time = 0
        self.context = context
        self.plot_widget = plotWidget
        self.scene = QGraphicsScene()
        self.setScene(self.scene)
        self.setGeometry(0, 0, 1200, 800)

        self.image = QImage(self.context._strategy.grid_size * grid_size, self.context._strategy.grid_size * grid_size, QImage.Format_ARGB32)
        self.image.fill(Qt.white)
        self.pixmap_item = self.scene.addPixmap(QPixmap.fromImage(self.image))

        self.context.initializePopulation()
        self.drawInitialPopulation()
        self.drawGrid()
        self.timer = QtCore.QTimer(self)
        self.timer.timeout.connect(self.simulate)

    def simulate(self):
        self.context.simulatePopularizationWithCallback()
        self.callbackGUIUpdate()

    def callbackGUIUpdate(self):
        self.updateGrid()
        self.updatePlot()

    def updateGrid(self):
        painter = QPainter(self.image)
        for i, j in self.context._strategy.changes:
            color = Qt.black if self.context._strategy.occupied_cells[i, j] else Qt.white
            painter.fillRect(i * grid_size, j * grid_size, grid_size, grid_size, color)
        painter.end()
        self.pixmap_item.setPixmap(QPixmap.fromImage(self.image))

    def updatePlot(self):
        # Update the population plot
        population = self.context._strategy.occupied_cells.sum()
        self.context._strategy.population_data.append((self.time, population))
        self.plot_widget.plot([t for t, _ in self.context._strategy.population_data], [p for _, p in self.context._strategy.population_data], pen='b')
        self.time += 1

    def drawGrid(self):
        pen = QtGui.QPen(QtGui.QColor(200, 200, 200))  # set the pen color to light gray
        # Draw the horizontal grid lines
        for i in range(self.context._strategy.grid_size + 1):
            self.scene.addLine(0, i * 5, self.context._strategy.grid_size * 5, i * 5, pen)
        # Draw the vertical grid lines
        for i in range(self.context._strategy.grid_size + 1):
            self.scene.addLine(i * 5, 0, i * 5, self.context._strategy.grid_size * 5, pen)    

    def drawInitialPopulation(self):
        painter = QPainter(self.image)
        for i in range(self.context._strategy.grid_size):
            for j in range(self.context._strategy.grid_size):
                color = Qt.black if self.context._strategy.occupied_cells[i, j] else Qt.white
                painter.fillRect(i * grid_size, j * grid_size, grid_size, grid_size, color)
        painter.end()
        self.pixmap_item.setPixmap(QPixmap.fromImage(self.image))

    def resetGrid(self):
        self.context._strategy.occupied_cells = np.zeros((self.context._strategy.grid_size, self.context._strategy.grid_size), dtype=bool)
        painter = QPainter(self.image)
        painter.fillRect(QRect(0, 0, self.image.width(), self.image.height()), QColor(Qt.white))
        painter.end()
        self.pixmap_item.setPixmap(QPixmap.fromImage(self.image))
        self.context.initializePopulation()
        self.drawInitialPopulation()

    def startTimer(self):
        self.timer.start(50)

    def stopTimer(self):
        self.timer.stop()  

    def setStrategy(self, index):
        match index:
            case 0:
                self.context.strategy = HomogeneousContactProcessSimulationStrategy()
            case 1:
                self.context.strategy = GradientRandomMapSimulationStrategy()
            case 2:
                self.context.strategy = GradientContactProcessSimulationStrategy()
            case 3:
                self.context.strategy = ImageTreelineSimulationStrategy()
        self.resetGrid()


class MainWindow(QMainWindow):
    def __init__(self, icon: QtGui.QIcon, context: SimulationContext, parent=None):
        super().__init__(parent)

        # Create the plot widget
        population_plot = pg.PlotWidget()
        population_plot.setBackground('w')
        population_plot.setLabel('left', 'Population')
        population_plot.setLabel('bottom', 'Time (s)')
        population_plot.showGrid(x=True, y=True)

        # Create the grid view
        gridView = CellularAutomataGridView(context, population_plot)
        layout = QVBoxLayout()

        # Create the control widgets
        selectionArea = QGroupBox('Simulation parameters')

        # Create the main layout
        selectionLayout = QHBoxLayout()
        selectionArea.setLayout(selectionLayout)

        self.patternBox = QComboBox()
        self.patternBox.addItem('Homogeneous Contact Process')
        self.patternBox.addItem('Gradient Random Map')
        self.patternBox.addItem('Gradient Contract Process')
        self.patternBox.addItem('Real image data')
        self.patternBox.currentIndexChanged.connect(gridView.setStrategy)

        self.startButton = QPushButton('&Start')
        self.stopButton = QPushButton('&Stop')
        self.restetButton = QPushButton('&Reset')

        self.percolationButton = QPushButton('&Identify percolation clusters')
        self.hullButton = QPushButton('&Mark Hull')

        self.startButton.clicked.connect(gridView.startTimer)
        self.stopButton.clicked.connect(gridView.stopTimer)
        self.restetButton.clicked.connect(gridView.resetGrid)

        self.input_colon = QLineEdit()
        self.input_colon.setMaxLength(5)
        self.input_colon.setPlaceholderText("Colonisation rate = 0.2")
        self.input_colon.textChanged.connect(gridView.context.setColonization)

        self.input_extinction = QLineEdit()
        self.input_extinction.setMaxLength(5)
        self.input_extinction.setPlaceholderText("Extinction rate = 0.15")
        self.input_extinction.textChanged.connect(gridView.context.setExtinction)

        selectionLayout.addWidget(self.patternBox)
        selectionLayout.addWidget(self.startButton)
        selectionLayout.addWidget(self.input_colon)
        selectionLayout.addWidget(self.input_extinction)
        selectionLayout.addWidget(self.stopButton)
        selectionLayout.addWidget(self.restetButton)
        selectionLayout.addWidget(self.percolationButton)
        selectionLayout.addWidget(self.hullButton)

        # Create the main widget
        widget = QWidget()
        widget.setLayout(layout)
        self.setCentralWidget(widget)

        layout.addWidget(selectionArea)
        layout.addWidget(gridView)
        layout.addWidget(population_plot, 2)

        self.setGeometry(0, 0, 1200, 800)
        self.setWindowTitle("Treeline Fractals v2.0")
        self.setWindowIcon(icon)

        self.show()

