# SPFW

This is the code to reproduce our experiments in our [Axiv Paper](http://arxiv.org/):
```
Frank-Wolfe Algorithm for Saddle Point Problem.
```
The project page of this article is ```http://www.di.ens.fr/sierra/research/SPFW/```. 
This project contains the implementation of SP-FW and SP-AFW on a strongly convex toy example (quadratic objective function with low dimentional constraints.
It also contains the implementation of SP-FW (and SP-BCFW) on the OCR dataset. In that case the objective function is bilinear.

There are two folders :
 - toy example. 
 - structured SVM. This folder is organized the same way as [BCFW](https://github.com/ppletscher/BCFWstruct).

##Aknowledgements

We want to aknowledge [Simon Lacoste-Julien](http://www.di.ens.fr/~slacoste/), [Martin Jaggi](http://www.cmap.polytechnique.fr/~jaggi/), [Mark Schmidt](http://www.di.ens.fr/~mschmidt/) and [Patrick Pletscher](http://pletscher.org) for the open source code on [BCFW](https://github.com/ppletscher/BCFWstruct). Our code on OCR experiments in an extension of theirs.

##Disclaimer

This code is not meant to be the most efficient. Our goal was to simply check if the experimental results matched with our theoretical analysis.

##Citation

Please use the following BibTeX entry to cite this software in your work:
```
@InProceedings{gidel2016saddle,
  author      = {Gidel, Gauthier and Jebara, Tony and Lacoste-Julien, Simon},
  title       = {Frank-{W}olfe {A}lgorithms for {S}addle {P}oint {P}roblems},
  booktitle   = {},
  year        = {2016} 
}
```
##Authors

* [Gauthier Gidel](http://www.di.ens.fr/~gidel/)
* [Tony Jebara](http://www.cs.columbia.edu/~jebara/)
* [Simon Lacoste-Julien](http://www.di.ens.fr/~slacoste/)
