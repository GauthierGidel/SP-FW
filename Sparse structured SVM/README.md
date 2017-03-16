# SP-FWstruct

This README and its associated code mainly comes from ```https://github.com/ppletscher/BCFWstruct```

The code is organized as follows:
* `solvers` contains the optimization methods, including the saddle point block-coordinate
  Frank-Wolfe solver (SP-BCFW). If you want to use SP-BCFW in your project, you most
  likely only need to run an `addpath(genpath('solvers'))` in your Matlab sources.
* `demos` contains the application-dependent code, such as MAP decoding or the
  feature map computation. The source code includes a sequence prediction demo
  for optical character recognition (OCR).
* `data` is initially empty, it is used to store the data files required for the
  demos.


## Getting Started

1. You need a working installation of Matlab.
2. Obtain the data files required to run the demos. On Unix systems you can
   simply run `./fetch_data.sh`. On Windows, you can use
   [Cygwin](http://www.cygwin.com/) or manually download the listed files and
   put them in the data folder.
3. In order to reproduce the plots `1e` and `1f` of [our paper](https://arxiv.org/abs/1610.07797) change to `demo/chain` and run `ocr1*.m` from within Matlab.


## Usage

If you would like to use the SP-FW solvers for your own structured output
prediction problem, you will need to implement three functions:

* The feature map.
* The maximization oracle.
* The loss function.

You can find an example implementation in the `demo/chain` folder. For an
overview of the exact usage and the supported options, please check the Matlab
documentation of the solvers.
