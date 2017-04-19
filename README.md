# Differentiable generative models

Python code accompanying the paper [Asymptotically exact inference in differentiable generative models](https://arxiv.org/abs/1605.07826).

## Requirements

The code has only been tested with Python 2.7. As a minimum you will need to have NumPy (1.11.2), SciPy (0.18.1) and Theano (0.8.2) available in the Python environment the code is run from to be able to use the `dgm` module (versions in parentheses are those tested with, others may also work).

To run the experiment notebooks you will need to additionally have Jupyter (1.0.0) and Matplotlib (1.5.3) installed. To run the MNIST experiments you will also need the [`choldate`](https://github.com/jcrudy/choldate) Python package for low-rank Cholesky updates by Jason Rudy (commit `0d92a52` used in experiments for paper) to be available in your Python environment. To analyse the Lotka-Volterra model experiments you will need to have R installled in your environment (with the `coda` package available) and the Python interface `rpy2` (2.7.8) installed.

## Structure

Three Python packages are included in the repository

  * `dgm`: Paper specific code implementing model generators and wrapper class for compiling necessary Theano functions. Can also be used with other Theano based generator functions to perform inference in your own models.
  * `bvh`: Tools for reading and rendering Biovision Hierarchy data files for human pose model experiments. This includes a copy of the BVH reader class from the [Python Computer Graphics Kit](http://cgkit.sourceforge.net/).
  * `hmc`: Classes implementing unconstrained and constrained Hamiltonian Monte Carlo samplers for performing inference in the models, including an implementation of the constrained Hamiltonian Monte Carlo method described in *Algorithm 1* in the paper.

Running

```
python setup.py install
```

in a terminal will install these three packages to the currently active Python environment.

There are also two additional directories

  * `notebooks`: Jupyter notebooks for running experiments and producing associated plots.
  * `models`: Data files containing fixed parameters associated with the different models.

## References

> Asymptoptically exact inference in differentiable generative models  
> Matthew M. Graham and Amos J. Storkey  
> *Proceedings of the 20th International Conference on Artificial Intelligence
and Statistics (AISTATS)*, 2017.
