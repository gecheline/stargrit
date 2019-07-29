# STARGRIT

STARGRIT (STellAR Generalized RadiatIve Transfer) is an in-development radiative transfer code for computation of atmosphere tables of differentially rotating and contact binary stars. It supports spherical and cylindrical grids with initial blackbody (gray and monochromatic) atmospheres. The default radiative transfer method of STARGRIT is COBAIN (COntact Binary Atmospheres with INterpolation (arXiv:1804.08781)).

## Getting Started

To install, run
```commandline
python setup.py build
python setup.py install
```
If the installation goes well, try importing to check for potential missing prerequisites.

### Prerequisites

To run STARGRIT, you will need to have numpy, scipy, astropy and quadpy installed on your machine. If you don't, install them via pip (or brew if on a Mac).


## Authors

* **Angela Kochoska** - [gecheline](https://github.com/gecheline)

## License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details
