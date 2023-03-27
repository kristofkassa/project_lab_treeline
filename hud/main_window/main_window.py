from PySide6.QtWidgets import QMainWindow, QVBoxLayout, QGraphicsView, QGraphicsScene, QGroupBox, QHBoxLayout, QComboBox, QPushButton, QWidget, QLineEdit
from PySide6 import QtGui, QtCore
import random


GRID_SIZE=50


class CellularAutomataGridView(QGraphicsView):
    """The Qt Graphics Grid where the whole simulation takes place.
    """

    def __init__(self):
        QGraphicsView.__init__(self)
        self.scene = QGraphicsScene()
        self.setScene(self.scene)
        self.setGeometry(0, 0, 800, 600)

        self.drawGrid()
        self.occupied_cells = set()
        self.initializePopulation()

        self.timer = QtCore.QTimer(self)
        self.timer.timeout.connect(self.homogeneousContactProcess)

        self.e = 0.15
        self.c = 0.2

    def setColonization(self, c_str):
        try:
            c = float(c_str)
            self.c = c
        except ValueError:
            pass

    def setExtinction(self, e_str):
        try:
            e = float(e_str)
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

        self.gridView = CellularAutomataGridView()
        layout = QVBoxLayout()
        layout.addWidget(self.gridView)

        selectionArea = QGroupBox('Simulation parameters')
        selectionLayout = QHBoxLayout()
        selectionArea.setLayout(selectionLayout)

        self.patternBox = QComboBox()
        self.patternBox.addItem('Gradient Random Map')
        self.patternBox.addItem('Gradient Contract Process')
        self.patternBox.addItem('Use image')
        self.startButton = QPushButton('&Start')
        self.stopButton = QPushButton('&Stop')
        self.restetButton = QPushButton('&Reset')

        self.startButton.clicked.connect(self.gridView.startTimer)
        self.stopButton.clicked.connect(self.gridView.stopTimer)
        self.restetButton.clicked.connect(self.gridView.resetGrid)

        self.input_colon = QLineEdit()
        self.input_colon.setMaxLength(5)
        self.input_colon.setPlaceholderText("colonisation = 0.2")
        self.input_colon.textChanged.connect(self.gridView.setColonization)

        self.input_extinction = QLineEdit()
        self.input_extinction.setMaxLength(5)
        self.input_extinction.setPlaceholderText("extinction = 0.15")
        self.input_extinction.textChanged.connect(self.gridView.setExtinction)


        selectionLayout.addWidget(self.patternBox)
        selectionLayout.addWidget(self.startButton)
        selectionLayout.addWidget(self.input_colon)
        selectionLayout.addWidget(self.input_extinction)
        selectionLayout.addWidget(self.stopButton)
        selectionLayout.addWidget(self.restetButton)

        layout.addWidget(selectionArea)

        widget = QWidget()
        widget.setLayout(layout)
        self.setCentralWidget(widget)

        layout.addWidget(self.gridView)

        self.setGeometry(0, 0, 800, 600)
        version = "v2.0"
        self.setWindowTitle("Treeline Fractals " + version)

        self.setWindowIcon(icon)
        self.show()

