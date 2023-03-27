from PySide6.QtWidgets import QMainWindow, QVBoxLayout, QGraphicsView, QGraphicsScene, QGroupBox, QHBoxLayout, QComboBox, QPushButton, QWidget, QLineEdit
from PySide6 import QtGui, QtCore
import random
import pyqtgraph as pg
import numpy as np

GRID_SIZE=50


class CellularAutomataGridView(QGraphicsView):
    """The Qt Graphics Grid where the whole simulation takes place.
    """

    def __init__(self, plotWidget):
        QGraphicsView.__init__(self)

        self.plot_widget = plotWidget

        
        self.scene = QGraphicsScene()
        self.setScene(self.scene)
        self.setGeometry(0, 0, 1200, 800)

        self.drawGrid()
        self.occupied_cells = set()
        self.initializePopulation()

        self.timer = QtCore.QTimer(self)
        self.timer.timeout.connect(self.gradientContactProcess)

        self.e = 0.15
        self.c = 0.2
        self.time = 0
        self.population_data = []
        

    def setColonization(self, c_str):
        try:
            c = float(c_str)
            if c > 0:
                self.c = c
        except ValueError:
            pass

    def setExtinction(self, e_str):
        try:
            e = float(e_str)
            if e > 0:
                self.e = e    
        except ValueError:
            pass   

    def drawGrid(self):
        """Draw the universe grid.
        """
        pen = QtGui.QPen(QtGui.QColor(200, 200, 200))  # set the pen color to light gray
        # Draw the horizontal grid lines
        for i in range(GRID_SIZE + 1):
            self.scene.addLine(0, i * 10, GRID_SIZE * 10, i * 10, pen)

        # Draw the vertical grid lines
        for i in range(GRID_SIZE + 1):
            self.scene.addLine(i * 10, 0, i * 10, GRID_SIZE * 10, pen)

    def initializePopulation(self, percentage = 0.003):
        """Randomly occupy a certain percentage of the grid.
        """

        # Initialize the grid with 5% of cells randomly occupied
        self.occupied_cells = set(random.sample([(i, j) for i in range(GRID_SIZE) for j in range(GRID_SIZE)], int(GRID_SIZE * GRID_SIZE * percentage)))
        for i in range(GRID_SIZE):
            for j in range(GRID_SIZE):
                pen = QtGui.QPen()
                if (i, j) in self.occupied_cells:
                    brush = QtGui.QBrush(QtCore.Qt.black)
                else:
                    brush = QtGui.QBrush(QtCore.Qt.white)
                rect = self.scene.addRect(i*10, j*10, 10, 10, pen, brush)
                rect.setZValue(-1)

    def homogeneousContactProcess(self):
        """Populate the grid using a homogeneous contact process.
        c: colonisation probability
        e: extinction probability
        """
        
        changes = set()
        for i in range(GRID_SIZE):
            for j in range(GRID_SIZE):
                # Check if the current cell has a neighboring cell that is occupied
                if any((i+di, j+dj) in self.occupied_cells for di, dj in [(-1,0), (1,0), (0,-1), (0,1)]):
                    # Determine whether the current cell becomes occupied or remains occupied
                    if random.random() < self.c and (i, j) not in self.occupied_cells:
                        changes.add((i, j))
                    elif random.random() < self.e and (i, j) in self.occupied_cells:
                        changes.add((i, j))
        if not changes:
            return
        # Apply the changes to the grid
        for i, j in changes:
            if (i, j) in self.occupied_cells:
                self.occupied_cells.remove((i, j))
            else:
                self.occupied_cells.add((i, j))
            pen = QtGui.QPen()
            brush = QtGui.QBrush(QtCore.Qt.black) if (i, j) in self.occupied_cells else QtGui.QBrush(QtCore.Qt.white)
            rect = self.scene.addRect(i*10, j*10, 10, 10, pen, brush)
            rect.setZValue(-1)

        # Update the population plot
        population = len(self.occupied_cells)
        self.population_data.append((self.time, population))
        self.plot_widget.plot([t for t, _ in self.population_data], [p for _, p in self.population_data], pen='b')
        self.time += 1

    def gradientContactProcess(self):
        """Populate the grid using a gradient contact process.
        c: colonisation probability
        e: extinction probability
        """

        # Pre-calculate the gradient values for each cell
        gradient = np.arange(GRID_SIZE) / GRID_SIZE
        
        changes = set()
        for i in range(GRID_SIZE):
            for j in range(GRID_SIZE):

                # Calculate the gradient value for this cell based on its position
                gradient = 1 - (i / GRID_SIZE)
                # Adjust the colonization and extinction probabilities based on the gradient value
                c_prob = self.c * (1 - gradient)
                e_prob = self.e / (1 - gradient + 0.001)

                # Check if the current cell has a neighboring cell that is occupied
                if any((i+di, j+dj) in self.occupied_cells for di, dj in [(-1,0), (1,0), (0,-1), (0,1)]):
                    # Determine whether the current cell becomes occupied or remains occupied
                    if random.random() < c_prob and (i, j) not in self.occupied_cells:
                        changes.add((i, j))
                    elif random.random() < e_prob and (i, j) in self.occupied_cells:
                        changes.add((i, j))
        if not changes:
            return
        # Apply the changes to the grid
        for i, j in changes:
            if (i, j) in self.occupied_cells:
                self.occupied_cells.remove((i, j))
            else:
                self.occupied_cells.add((i, j))
            pen = QtGui.QPen()
            brush = QtGui.QBrush(QtCore.Qt.black) if (i, j) in self.occupied_cells else QtGui.QBrush(QtCore.Qt.white)
            rect = self.scene.addRect(i*10, j*10, 10, 10, pen, brush)
            rect.setZValue(-1)

        # Update the population plot
        population = len(self.occupied_cells)
        self.population_data.append((self.time, population))
        self.plot_widget.plot([t for t, _ in self.population_data], [p for _, p in self.population_data], pen='b')
        self.time += 1    


    def resetGrid(self):
        self.occupied_cells.clear()
        for i in range(GRID_SIZE):
            for j in range(GRID_SIZE):
                pen = QtGui.QPen()
                brush = QtGui.QBrush(QtCore.Qt.white)
                rect = self.scene.addRect(i*10, j*10, 10, 10, pen, brush)
                rect.setZValue(-1)

        self.initializePopulation()

    def startTimer(self):
        self.timer.start(100)

    def stopTimer(self):
        self.timer.stop()              

class MainWindow(QMainWindow):
    def __init__(self, icon: QtGui.QIcon, parent=None):
        super().__init__(parent)

        # Create the plot widget
        population_plot = pg.PlotWidget()
        population_plot.setBackground('w')
        population_plot.setLabel('left', 'Population')
        population_plot.setLabel('bottom', 'Time (s)')
        population_plot.showGrid(x=True, y=True)

        # Create the grid view
        gridView = CellularAutomataGridView(population_plot)
        layout = QVBoxLayout()

        # Create the control widgets
        selectionArea = QGroupBox('Simulation parameters')

        # Create the main layout
        selectionLayout = QHBoxLayout()
        selectionArea.setLayout(selectionLayout)

        self.patternBox = QComboBox()
        self.patternBox.addItem('Gradient Random Map')
        self.patternBox.addItem('Gradient Contract Process')
        self.patternBox.addItem('Use image')
        self.startButton = QPushButton('&Start')
        self.stopButton = QPushButton('&Stop')
        self.restetButton = QPushButton('&Reset')

        self.startButton.clicked.connect(gridView.startTimer)
        self.stopButton.clicked.connect(gridView.stopTimer)
        self.restetButton.clicked.connect(gridView.resetGrid)

        self.input_colon = QLineEdit()
        self.input_colon.setMaxLength(5)
        self.input_colon.setPlaceholderText("Colonisation Probability = 0.2")
        self.input_colon.textChanged.connect(gridView.setColonization)

        self.input_extinction = QLineEdit()
        self.input_extinction.setMaxLength(5)
        self.input_extinction.setPlaceholderText("Extinction Probability = 0.15")
        self.input_extinction.textChanged.connect(gridView.setExtinction)

        selectionLayout.addWidget(self.patternBox)
        selectionLayout.addWidget(self.startButton)
        selectionLayout.addWidget(self.input_colon)
        selectionLayout.addWidget(self.input_extinction)
        selectionLayout.addWidget(self.stopButton)
        selectionLayout.addWidget(self.restetButton)

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

