from PySide6.QtWidgets import QMainWindow, QVBoxLayout, QGraphicsView, QGraphicsScene, QGroupBox, QHBoxLayout, QComboBox, QPushButton, QWidget, QLineEdit, QTextEdit
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

grid_size = 4

class CellularAutomataGridView(QGraphicsView):
    """The Qt Graphics Grid where the whole simulation takes place.
    """

    def __init__(self, context: SimulationContext, plotWidget, densityPlotWidget, box_counting_dimension_textbox, correlation_dimension_textbox, ruler_dimension_textbox, avgdist_dimension_textbox):
        QGraphicsView.__init__(self)

        self.time = 0
        self.context = context
        self.plot_widget = plotWidget
        self.densityPlotWidget = densityPlotWidget
        self.box_counting_dimension_textbox = box_counting_dimension_textbox
        self.correlation_dimension_textbox = correlation_dimension_textbox
        self.ruler_dimension_textbox = ruler_dimension_textbox
        self.avgdist_dimension_textbox = avgdist_dimension_textbox
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

        self.dimensionTimer = QtCore.QTimer(self)
        # self.dimensionTimer.timeout.connect(self.markHull)

    def simulate(self):
        self.context.simulatePopularizationWithCallback()
        self.callbackGUIUpdate()
        # self.markCluster()
        # self.markHull()

    def callbackGUIUpdate(self):
        self.updateGrid()
        self.updatePlot()
        self.updateDensityPlot(self.densityPlotWidget)

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

    def updateDensityPlot(self, plotWidget):
        # Clear the plot before updating
        plotWidget.clear()

        densities = self.getRowDensities()
        x_values = np.arange(len(densities))
        plotWidget.plot(x_values, densities, pen='r')

    def getRowDensities(self):
        return self.context._strategy.occupied_cells.sum(axis=1)

    def drawGrid(self):
        pen = QtGui.QPen(QtGui.QColor(230, 230, 230))  # set the pen color to light gray
        # Draw the horizontal grid lines
        for i in range(self.context._strategy.grid_size + 1):
            self.scene.addLine(0, i * grid_size, self.context._strategy.grid_size * grid_size, i * grid_size, pen)
        # Draw the vertical grid lines
        for i in range(self.context._strategy.grid_size + 1):
            self.scene.addLine(i * grid_size, 0, i * grid_size, self.context._strategy.grid_size * grid_size, pen)    

    def drawInitialPopulation(self):
        painter = QPainter(self.image)
        for i in range(self.context._strategy.grid_size):
            for j in range(self.context._strategy.grid_size):
                color = Qt.black if self.context._strategy.occupied_cells[i, j] else Qt.white
                painter.fillRect(i * grid_size, j * grid_size, grid_size, grid_size, color)
        painter.end()
        self.pixmap_item.setPixmap(QPixmap.fromImage(self.image))

    def markCluster(self): 
        self.context._strategy.identifyPercolationClusters()
        painter = QPainter(self.image)

        for (i, j), value in np.ndenumerate(self.context._strategy.cluster):
            if value:
                color = Qt.green
                painter.fillRect(i * grid_size, j * grid_size, grid_size, grid_size, color)
                
        painter.end()
        self.pixmap_item.setPixmap(QPixmap.fromImage(self.image))

    def markHull(self): 
        self.context._strategy.identifyPercolationClusters()

        box_counting_dimension, correlation_dimension, ruler_dimension, avgdist_dimension = self.context._strategy.calculate_fractal_dimensions()
        self.box_counting_dimension_textbox.append(f"Box counting dimension: {box_counting_dimension}")
        self.correlation_dimension_textbox.append(f"Correlation dimension: {correlation_dimension}")
        self.ruler_dimension_textbox.append(f"Ruler dimension: {ruler_dimension}")
        self.avgdist_dimension_textbox.append(f"AvgDist dimension: {avgdist_dimension}")

        # Scroll to the bottom to display the latest result
        self.box_counting_dimension_textbox.verticalScrollBar().setValue(self.box_counting_dimension_textbox.verticalScrollBar().maximum())
        self.correlation_dimension_textbox.verticalScrollBar().setValue(self.correlation_dimension_textbox.verticalScrollBar().maximum())
        self.ruler_dimension_textbox.verticalScrollBar().setValue(self.ruler_dimension_textbox.verticalScrollBar().maximum())
        self.avgdist_dimension_textbox.verticalScrollBar().setValue(self.avgdist_dimension_textbox.verticalScrollBar().maximum())

        painter = QPainter(self.image)
        for (i, j), value in np.ndenumerate(self.context._strategy.hull):
            if value:
                color = Qt.red
                painter.fillRect(i * grid_size, j * grid_size, grid_size, grid_size, color)

        painter.end()
        self.pixmap_item.setPixmap(QPixmap.fromImage(self.image))

    def resetGrid(self):
        self.plot_widget.clear()
        self.context._strategy.cluster = np.zeros((self.context._strategy.grid_size, self.context._strategy.grid_size), dtype=bool)
        self.context._strategy.occupied_cells = np.zeros((self.context._strategy.grid_size, self.context._strategy.grid_size), dtype=bool)
        painter = QPainter(self.image)
        painter.fillRect(QRect(0, 0, self.image.width(), self.image.height()), QColor(Qt.white))
        painter.end()
        self.pixmap_item.setPixmap(QPixmap.fromImage(self.image))
        self.context.initializePopulation()
        self.drawInitialPopulation()

        if isinstance(self.context._strategy, GradientRandomMapSimulationStrategy) :
            self.context._strategy.running = False

    def autoSimulate(self):
        self.context._strategy.autoSimulate()
        return

    def startTimer(self):
        self.timer.start(100)
        self.dimensionTimer.start(5000)

    def stopTimer(self):
        self.timer.stop()
        self.dimensionTimer.stop()

    def nextImage(self):
        self.resetGrid()
        self.context._strategy.nextImage()
        self.simulate()

    def setStrategy(self, index):
        match index:
            case 0:
                self.context.strategy = GradientRandomMapSimulationStrategy() 
            case 1:
                self.context.strategy = HomogeneousContactProcessSimulationStrategy()
            case 2:
                self.context.strategy = GradientContactProcessSimulationStrategy()
            case 3:
                self.context.strategy = ImageTreelineSimulationStrategy()
        self.resetGrid()


class MainWindow(QMainWindow):
    def __init__(self, icon: QtGui.QIcon, context: SimulationContext, parent=None):
        super().__init__(parent)

        plotLayout = QHBoxLayout()

        # Define the fixed height
        plot_height = 180

        # Create the population plot widget
        population_plot = pg.PlotWidget()
        population_plot.setBackground('w')
        population_plot.setLabel('left', 'Population')
        population_plot.setLabel('bottom', 'Time (s)')
        population_plot.showGrid(x=True, y=True)
        population_plot.setFixedHeight(plot_height)  # Set the fixed height

        # Create another plot for the row densities
        density_plot = pg.PlotWidget()
        density_plot.setBackground('w')
        density_plot.setLabel('left', 'Density')
        density_plot.setLabel('bottom', 'Row')
        density_plot.showGrid(x=True, y=True)
        density_plot.setFixedHeight(plot_height)  # Set the fixed height

        # Add both plots to the plot layout
        plotLayout.addWidget(population_plot)
        plotLayout.addWidget(density_plot)

        textLayout = QHBoxLayout()

        # Create a QTextEdit for the fractal dimension result
        box_counting_dimension_textbox = QTextEdit()
        box_counting_dimension_textbox.setReadOnly(True)
        box_counting_dimension_textbox.setFixedHeight(30)

        # Create a QTextEdit for the fractal dimension result
        correlation_dimension_textbox = QTextEdit()
        correlation_dimension_textbox.setReadOnly(True)
        correlation_dimension_textbox.setFixedHeight(30)

        # Create a QTextEdit for the fractal dimension result
        ruler_dimension_textbox = QTextEdit()
        ruler_dimension_textbox.setReadOnly(True)
        ruler_dimension_textbox.setFixedHeight(30)

        # Create a QTextEdit for the fractal dimension result
        avgdist_dimension_textbox = QTextEdit()
        avgdist_dimension_textbox.setReadOnly(True)
        avgdist_dimension_textbox.setFixedHeight(30)

        textLayout.addWidget(box_counting_dimension_textbox)
        textLayout.addWidget(correlation_dimension_textbox)
        textLayout.addWidget(ruler_dimension_textbox)
        textLayout.addWidget(avgdist_dimension_textbox)

        # Create the grid view
        gridView = CellularAutomataGridView(context, population_plot, density_plot, box_counting_dimension_textbox, 
                                            correlation_dimension_textbox, ruler_dimension_textbox, avgdist_dimension_textbox)
        layout = QVBoxLayout()

        # Create the control widgets
        selectionArea = QGroupBox('Simulation parameters')

        # Create the main layout
        selectionLayout = QHBoxLayout()
        selectionArea.setLayout(selectionLayout)

        self.patternBox = QComboBox()
        self.patternBox.addItem('Gradient Random Map')
        self.patternBox.addItem('Homogeneous Contact Process')
        self.patternBox.addItem('Gradient Contract Process')
        self.patternBox.addItem('Real image data')
        self.patternBox.currentIndexChanged.connect(gridView.setStrategy)

        self.startButton = QPushButton('&Start')
        self.nextImageButton = QPushButton('&>')
        self.nextImageButton.setFixedSize(20, 20)  # Set the width and height of the button
        self.stopButton = QPushButton('&Stop')
        self.restetButton = QPushButton('&Reset')

        self.percolationButton = QPushButton('&Identify percolation clusters')
        self.hullButton = QPushButton('&Mark Hull')
        self.autoSimulate = QPushButton('&Auto Sim')
        self.autoSimulate.setStyleSheet('QPushButton {background-color: green; color: white;}')

        self.startButton.clicked.connect(gridView.startTimer)
        self.nextImageButton.clicked.connect(gridView.nextImage)
        self.stopButton.clicked.connect(gridView.stopTimer)
        self.restetButton.clicked.connect(gridView.resetGrid)
        self.percolationButton.clicked.connect(gridView.markCluster)
        self.hullButton.clicked.connect(gridView.markHull)
        self.autoSimulate.clicked.connect(gridView.autoSimulate)

        self.input_colon = QLineEdit()
        self.input_colon.setMaxLength(5)
        self.input_colon.setPlaceholderText("Colonisation rate = 0.6")
        self.input_colon.textChanged.connect(gridView.context.setColonization)

        self.input_extinction = QLineEdit()
        self.input_extinction.setMaxLength(5)
        self.input_extinction.setPlaceholderText("Extinction rate = 0.2")
        self.input_extinction.textChanged.connect(gridView.context.setExtinction)

        selectionLayout.addWidget(self.patternBox)
        selectionLayout.addWidget(self.nextImageButton)
        selectionLayout.addWidget(self.startButton)
        selectionLayout.addWidget(self.stopButton)
        selectionLayout.addWidget(self.restetButton)
        selectionLayout.addWidget(self.input_colon)
        selectionLayout.addWidget(self.input_extinction)
        selectionLayout.addWidget(self.percolationButton)
        selectionLayout.addWidget(self.hullButton)
        selectionLayout.addWidget(self.autoSimulate)

        # Create the main widget
        widget = QWidget()
        widget.setLayout(layout)
        self.setCentralWidget(widget)

        layout.addWidget(selectionArea)
        layout.addLayout(textLayout)
        layout.addWidget(gridView)
        layout.addLayout(plotLayout)

        self.setGeometry(0, 0, 1200, 900)
        self.setWindowTitle("Treeline Fractals v2.0")
        self.setWindowIcon(icon)

        self.show()

