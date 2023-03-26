from PySide6.QtWidgets import QMainWindow, QVBoxLayout, QGraphicsView, QGraphicsScene, QGroupBox, QHBoxLayout, QComboBox, QPushButton, QWidget, QLineEdit
from PySide6 import QtGui, QtCore
import random


GRID_SIZE=40


class CellularAutomataGridView(QGraphicsView):
    """The Qt Graphics Grid where the whole action takes place.
    """

    def __init__(self):
        QGraphicsView.__init__(self)
        self.scene = QGraphicsScene()
        self.setScene(self.scene)
        self.setGeometry(0, 0, 800, 600)
        self.drawGrid()
        self.timer = QtCore.QTimer(self)
        self.timer.timeout.connect(self.occupyGrid)
        self.timer.start(500)

    def drawGrid(self):
        """Draw the universe grid.
        """

        # Draw the horizontal grid lines
        for i in range(GRID_SIZE + 1):
            self.scene.addLine(0, i * 10, GRID_SIZE * 10, i * 10)

        # Draw the vertical grid lines
        for i in range(GRID_SIZE + 1):
            self.scene.addLine(i * 10, 0, i * 10, GRID_SIZE * 10)

    def occupyGrid(self):
        """Randomly occupy a certain percentage of the grid.
        """

        for i in range(GRID_SIZE):
            for j in range(GRID_SIZE):
                pen = QtGui.QPen()
                if random.random() < 0.1:
                    brush = QtGui.QBrush(QtCore.Qt.black)
                else:
                    brush = QtGui.QBrush(QtCore.Qt.white)
                rect = self.scene.addRect(i*10, j*10, 10, 10, pen, brush)
                rect.setZValue(-1)

class MainWindow(QMainWindow):
    def __init__(self, icon: QtGui.QIcon, parent=None):

        super().__init__(parent)

        self.gridView = CellularAutomataGridView()
        layout = QVBoxLayout()
        layout.addWidget(self.gridView)

        selectionArea = QGroupBox('Select a Pattern')
        selectionLayout = QHBoxLayout()
        selectionArea.setLayout(selectionLayout)

        self.patternBox = QComboBox()
        self.patternBox.addItem('Gradient Random Map')
        self.patternBox.addItem('Gradient Contract Process')
        self.patternBox.addItem('Use image')
        self.startButton = QPushButton('&Start')
        self.stopButton = QPushButton('&Stop')

        self.input_colon = QLineEdit()
        self.input_colon.setMaxLength(5)
        self.input_colon.setPlaceholderText("Set colonisation")

        self.input_extinction = QLineEdit()
        self.input_extinction.setMaxLength(5)
        self.input_extinction.setPlaceholderText("Set extinction")

        selectionLayout.addWidget(self.patternBox)
        selectionLayout.addWidget(self.startButton)
        selectionLayout.addWidget(self.input_colon)
        selectionLayout.addWidget(self.input_extinction)
        selectionLayout.addWidget(self.stopButton)

        layout.addWidget(selectionArea)

        widget = QWidget()
        widget.setLayout(layout)
        self.setCentralWidget(widget)

        self.gridView.setRenderHint(QtGui.QPainter.Antialiasing)
        layout.addWidget(self.gridView)

        self.setGeometry(0, 0, 800, 600)
        version = "V2.0"
        self.setWindowTitle("Treeline Fractals " + version)

        self.setWindowIcon(icon)
        self.show()
