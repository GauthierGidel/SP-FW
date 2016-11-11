# Graphical games
This experiment correspond to figure (1d) in [our paper](https://arxiv.org/abs/1610.07797):
 - `script.m` plots the figure.
 - `spfw.m` calls the saddle point Frank-Wolfe algorithm.
 - `edmond2.m` is the oracle function. This program uses [BlossomV](http://pub.ist.ac.at/~vnk/software.html), [KOLMOGOROV](http://pub.ist.ac.at/~vnk/)'s implementation of the blossom algorithm.
 In order to use the code, the line 12 of `edmond2.m` has to be changed.
