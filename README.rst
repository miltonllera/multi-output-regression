# PyMOReg: Multi-Output Regression in Python

This library contains algorithms used to perform regression on multiple (correlated) outputs simultaneously.

## Getting Started

These instructions will get you a copy of the project up and running on your local machine for development and testing purposes. As of now this is only a prototype and only the dev version is available.

### Prerequisites

To run this software you need several libraries:

* numpy >= 1.13
* scipy >= 0.19.1
* networkx >= 1.11
* scikit-learn >= 0.19.0
* matplotlib >= 2.0.2
* pandas >= 0.20.3
* seaborn >= 0.8
* pygraphviz >= 1.3.1

### Installing

These libraries come with the Anaconda bundle (www.anaconda.com) and for Linux users can be obtained by calling:

```
<sudo> apt-get install anaconda
```

In case of Windows users an .exe installer is avalailable at (www.anaconda.com/download/). For a manual install of each
library the user can execute:

```
conda install <library>

```
or for cases when libraries are not available through the conda channel:

```
pip install <library>
```

Once these dependencies are installed the library can be cloned from the GitHub repository:
```
git clone https://github.com/mllera14/multi-output-regression <destination-folder>
```

To install it open a console in <destination-folder> and type:

```
python setup.py install
```

The library can now be imported into any development enviornment as:

```
import pymoreg as mreg
```

## Authors

* **Milton Llera** - *Computational Intelligence Group, Universidad Politecnica de Madrid* - [mllera14](https://github.com/mllera14)

## License

This project is licensed under the MIT License - see the [LICENSE.txt](LICENSE.txt) file for details
