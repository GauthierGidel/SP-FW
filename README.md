# SPFW

This is the code to reproduce our experiments in our [Paper](https://arxiv.org/abs/1610.07797):
```
Frank-Wolfe Algorithms for Saddle Point Problems.
```
The project page of this article is ```http://www.di.ens.fr/sierra/research/SPFW/```.
This project contains the implementation of SP-FW and SP-AFW on a strongly convex toy example (quadratic objective function with low dimentional constraints.
It also contains the implementation of SP-FW (and SP-BCFW) on the OCR dataset. In that case the objective function is bilinear.

There are three folders corresponding to the three paragraphs in our experiments section :
 - Toy experiments.
 - Graphical games.
 - Sparse structured SVM. This folder is organized the same way as [BCFW](https://github.com/ppletscher/BCFWstruct).
More details about these experiments in their respective README.

##Aknowledgements

We want to aknowledge [Simon Lacoste-Julien](http://www.di.ens.fr/~slacoste/), [Martin Jaggi](http://www.cmap.polytechnique.fr/~jaggi/), [Mark Schmidt](http://www.di.ens.fr/~mschmidt/) and [Patrick Pletscher](http://pletscher.org) for the open source code on [BCFW](https://github.com/ppletscher/BCFWstruct). Our code on OCR experiments in an extension of theirs.

##Disclaimer

This code is not meant to be the most efficient. Our goal was to simply check if the experimental results matched with our theoretical analysis.

##Citation

Please use the following BibTeX entry to cite this software in your work:
```
@InProceedings{gidel2017saddle,
  author      = {Gidel, Gauthier and Jebara, Tony and Lacoste-Julien, Simon},
  title       = {Frank-{W}olfe Algorithms for Saddle Point Problems},
  booktitle   = {Proceedings of the 20th International Conference on Artificial Intelligence and Statistics (AISTATS)},
  year        = {2017}
}
```
##Authors

* [Gauthier Gidel](http://www.di.ens.fr/~gidel/)
* [Tony Jebara](http://www.cs.columbia.edu/~jebara/)
* [Simon Lacoste-Julien](http://www.di.ens.fr/~slacoste/)

## Octave
The graphical games and the sparse structured SVM experiments work with [Octave](https://www.gnu.org/software/octave/) with very simple modification (basically just removing the random seed). Otherwise, the toy experiments need a hashmap function not directly available with Octave. We invite users who want to make this code working with Octave to implement similar functions as [containers.Map()](https://www.mathworks.com/help/matlab/ref/containers.map-class.html) MATLAB class or to use the [java package](https://www.gnu.org/software/octave/doc/interpreter/Java-Interface.html) to create java.util.Hashtable.
