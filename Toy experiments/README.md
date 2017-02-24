# Toy experiments

This folder correspond to experiments (1a), (1b) and (1c) in [our paper](https://arxiv.org/abs/1610.07797).
You can reproduce each figure running `plot1*.m`
## Functions
The algorithm is decomposed into 3 functions:
 - `SP_FW.m` is the main function.
 - `hashing.m` is the hashing function to memorize to active set.
 - `away_step.m` returns the away corner.

In order to plot the [prettyPlot function](https://www.cs.ubc.ca/~schmidtm/Software/prettyPlot.html) by [Mark Schmidt](http://www.cs.ubc.ca/~schmidtm/) is needed. The authors would like to thank him for his code. We slightly modified it in order to display latex in legend.
