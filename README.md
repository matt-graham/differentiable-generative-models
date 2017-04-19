# Differentiable generative models

Python code accompanying the paper [Asymptotically exact inference in differentiable generative models](https://arxiv.org/abs/1605.07826).

## Requirements

The code has only been tested with Python 2.7. As a minimum you will need to have NumPy (1.11.2), SciPy (0.18.1) and Theano (0.8.2) available in the Python environment the code is run from to be able to use the `dgm` module (versions in parentheses are those tested with, others may also work).

To run the experiment notebooks you will need to additionally have Jupyter (1.0.0) and Matplotlib (1.5.3) installed. To analyse the Lotka-Volterra model experiments you will need to have R installled in your environment (with the `coda` package available) and the Python interface `rpy2` (2.7.8) installed.

## Installation

As several of the Python dependencies are included as [submodules](https://github.com/blog/2104-working-with-submodules) you will need to do a recursive clone e.g.

```
git clone --recursive https://github.com/matt-graham/differentiable-generative-models.git
```

There are four sub-modules

  * [`auxiliary-pm-mcmc`](https://github.com/matt-graham/auxiliary-pm-mcmc): Python implementations of MCMC samplers in the auxiliary pseudo-marginal MCMC framework as described in the paper *Pseudo-Marginal Slice Sampling* (Murray and Graham, 2016). These implementations are used to run the ABC MCMC experiments.
  * [`bvh-tools`](https://github.com/matt-graham/bvh-tools): Tools for reading and rendering Biovision Hierarchy data files for human pose model experiments. This includes a copy of the BVH reader class from the [Python Computer Graphics Kit](http://cgkit.sourceforge.net/).
  * [`choldate`](https://github.com/jcrudy/choldate): Python package for low-rank Cholesky updates by [Jason Rudy](https://github.com/jcrudy).
  * [`hamiltonian-monte-carlo`](https://github.com/matt-graham/hamiltonian-monte-carlo): Classes implementing unconstrained and constrained Hamiltonian Monte Carlo samplers for performing inference in the models, including an implementation of the constrained Hamiltonian Monte Carlo method described in *Algorithm 1* in the paper.
  
Each includes a `setup.py` script that should be run using `python setup.py install` (or `python setup.py develop`) to install (in developer mode) the respective Python packages in to the Python environment that will be used to run the experiments.

Additionally there is a further paper specific Python package `dgm` and a corresponding`setup.py` script in the root directory. The `dgm` package includes paper specific code implementing model generators and wrapper class for compiling necessary Theano functions. It can also be used with other Theano based generator functions to perform inference in your own models. It should be installed in to the Python environment by running `python setup.py install`.

## Additional files

There are also two additional directories

  * `notebooks`: Jupyter notebooks for running experiments and producing associated plots.
  * `models`: Data files containing fixed parameters associated with the different models.

## References

> Asymptoptically exact inference in differentiable generative models  
> Matthew M. Graham and Amos J. Storkey  
> *Proceedings of the 20th International Conference on Artificial Intelligence
and Statistics (AISTATS)*, 2017.
