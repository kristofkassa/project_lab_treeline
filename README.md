# Project Laboratory I.
## The fractal structure of treelines in high mountains

## Project decription
The occurrence of tree species in high mountains usually ends abruptly at some elevation. This is the treeline. Magnifying this region, we find a characteristic ‘mainland/island’ structure, in which the tree cover is continuous at lower elevations (‘mainland’), and gets fragmented at higher ones (‘islands’). Theoretical considerations from percolation theory suggest that the hull of the ‘mainland’ should be a fractal with dimension 7/4. The task is to test this hypothesis using satellite images of treelines. The main goal is to suggest a feasible method for characterising the fractal structure of treelines, taking into account the limitations of real-life ecological data. The method’s potential applications include the precise delineation of species borders, and thus the detection of population shifts due to climate change.


## Getting started

To run Treeline Fractals Simulator locally, clone the code using git by running the following:

```shell
git clone https://github.com/kristofkassa/project_lab_treeline.git
```
To find out more about git visit [https://git-scm.com](https://git-scm.com/book/en/v2/Getting-Started-Installing-Git).

To start the simulation you'll need Python version 3.11 or above and PySide6.
You can download Python from [www.python.org](https://www.python.org).

Once you have installed Python, you can optionally create a Python virtual
environment, you can find instructions for creating and using virtual
environments at
[docs.python.org/3/library/venv.html](https://docs.python.org/3/library/venv.html).

To use the graphical user interface, you'll need to install
the [PySide6 package](https://pypi.org/project/PySide6/) which you can do by
running

```shell
pip3 install PySide6
```

in your Python virtual environment.

Run the following command to install all required dependencies:
```shell
pip install -r requirements.txt
```

### Running a simulation

To run a simulation simply run from the root folder:

```shell
python3 treeline_fractal.py
```
