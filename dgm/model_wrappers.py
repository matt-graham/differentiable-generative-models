# -*- coding: utf-8 -*-
"""Wrappers around generators to provide functions needed for inference."""

import numpy as np
import time
import scipy.linalg as la
import theano as th
import theano.tensor as tt
import theano.tensor.nlinalg as nla
import logging

logger = logging.getLogger(__name__)


def _timed_func_compilation(inputs, outputs, description):
    """Helper function which times compilation of a Theano function."""
    start_time = time.time()
    logger.info('Compiling {0}...'.format(description))
    func = th.function(inputs, outputs)
    logger.info('... finished in {0:.1f}s'.format(time.time() - start_time))
    return func


def scipy_calc_gram_chol(jac):
    """Calculate Cholesky factor of Jacobian Gram matrix with SciPy cho_factor.

    This is valid for any generator but has a cubic computational cost. In
    some cases it may be possible to exploit structure in the generator to
    calculate the decomposition more efficiently.
    """
    return la.cho_factor(jac.dot(jac.T))


def gaussian_energy(u):
    """Negative log of density for zero-mean, identity covariance Gaussian."""
    return 0.5 * u.dot(u)


class DifferentiableGenerativeModel(object):
    """Wrapper class for differentiable generative models."""

    def __init__(self, generator, constants,
                 base_energy=gaussian_energy,
                 calc_gram_chol=scipy_calc_gram_chol):
        """
        Create a new differentiable generative model wrapper object.

        Args:
            generator (function): Function composed of Theano graph operations
                corresponding to generator for the model of interest. This
                must take two arguments. The first is a Theano tensor variable
                corresponding to the vector of inputs (or batch of input
                vectors, in which case first dimension should correspond to
                the item in the batch and second dimension the output element)
                to the generator (e.g. draws from a Gaussian base density).
                The second is a dictionary containing any constants required
                by the generator function (i.e. any numeric parameters of the
                generator process that are assumed  fixed). The generator
                *must* be able to compute both a one-dimensional vector output
                if passed a one-dimensional vector input as its first
                argument, and a two-dimensional matrix output corresponding to
                a batch of output vectors (first dimension batch dimension),
                if passed a two-dimensional matrix input (batch of input
                vectors) as its first argument. This requirement allows a more
                efficient Jacobian calculation when batched forward
                propagation through the generator can be computed efficiently
                using blocked operations.
            constants (dict): A dictionary of containing any constants
                required by the generator function (i.e. any parameters of the
                generator process that are assumed fixed).
            base_energy (function): Function composed of Theano graph
                operations which specifies energy (negative log density up to
                an additive constant) associated with the inputs to the
                generator. By default this will be set to the energy
                corresponding to inputs with a zero-mean, identity covariance
                Gaussian probability density.
            calc_gram_chol (function): Function which computes the Cholesky
                decomposition of a provided NumPy array corresponding to the
                Jacobian matrix, shape (output_dim, input_dim), of the
                generator for a particular input, i.e. matrix of partial
                derivatives of each output with respect to each input. By
                default this uses the standard SciPy `cho_factor` function to
                do the decomposition. In some cases structure in the Jacobian
                may allow more efficient calculation e.g. using low-rank
                updates. The function should follow the return pattern of
                `scipy.linalg.cho_factor` by returning a tuple the first
                element of which is the array corresponding to the computed
                Cholesky decomposition and the second is a boolean indicating
                whether a lower triangular decomposition was calculated (True)
                or upper triangular (False). For both lower and upper
                decompositions only the corresponding portion of the computed
                array will be read from therefore the entries in the other
                triangle of the array can be arbitrary.
        """
        self.generator = generator
        self.constants = constants
        self.base_energy = base_energy
        self.calc_gram_chol = calc_gram_chol
        self._compile_theano_functions()

    def _compile_theano_functions(self):
        u = tt.vector('u')
        y = self.generator(u, self.constants)
        # Jacobian dy/du calculated by forward propagating batch of repeated
        # input vectors i.e. matrix of shape (output_dim, input_dim) to get
        # batch of repeated output vectors, shape (output_dim, output_dim),
        # and then initialising back propagation of gradients from this
        # repeated output matrix with identity matrix seed. Although convoluted
        # this method of computing Jacobian exploits blocked operations and
        # gives significant improvements in speed over the in-built sequential
        # scan based Jacobian calculation in Theano. See following issue:
        # https://github.com/Theano/Theano/issues/4087
        u_rep = tt.tile(u, (y.shape[0], 1))
        y_rep = self.generator(u_rep, self.constants)
        dy_du = tt.grad(
            cost=None, wrt=u_rep, known_grads={y_rep: tt.identity_like(y_rep)})
        # Direct energy calculation using Jacobian Gram matrix determinant
        energy = (self.base_energy(u) +
                  0.5 * tt.log(nla.det(dy_du.dot(dy_du.T))))
        # Alternative energy gradient calculation uses externally calculated
        # pseudo-inverse of Jacobian dy/du
        dy_du_pinv = tt.matrix('dy_du_pinv')
        base_energy_grad = tt.grad(self.base_energy(u), u)
        # Lop term calculates gradient of log|(dy/du) (dy/du)^T| using
        # externally calculated pseudo-inverse [(dy/du)(dy/du)^T]^(-1) (dy/du)
        energy_grad_alt = (
          base_energy_grad +
          tt.Lop(dy_du, u_rep, dy_du_pinv).sum(0)
        )
        self.generator_func = _timed_func_compilation(
            [u], y, 'generator function')
        self.generator_jacob = _timed_func_compilation(
            [u], dy_du, 'generator Jacobian')
        self.energy_func_direct = _timed_func_compilation(
            [u], energy, 'energy function')
        self.energy_grad_direct = _timed_func_compilation(
            [u], tt.grad(energy, u), 'energy gradient (direct)')
        self.energy_grad_alt = _timed_func_compilation(
            [u, dy_du_pinv], energy_grad_alt, 'energy gradient (alternative)')
        self.base_energy_func = _timed_func_compilation(
            [u], self.base_energy(u), 'base energy function')

    def constr_func(self, u):
        if not hasattr(self, 'y_obs'):
            raise ValueError('y_obs must be set before calling constr_func.')
        return self.generator_func(u) - self.y_obs

    def constr_jacob(self, u, calc_gram_chol=True):
        jac = self.generator_jacob(u)
        cache = {'dc_dpos': jac}
        if calc_gram_chol:
            cache['gram_chol'] = self.calc_gram_chol(jac)
        return cache

    def energy_func(self, u, cache={}):
        if 'gram_chol' not in cache:
            logger.info('Gram matrix Cholesky not available.')
            return self.energy_func_direct(u)
        else:
            gram_chol = cache['gram_chol']
            return (self.base_energy_func(u) +
                    np.log(gram_chol[0].diagonal()).sum())

    def energy_grad(self, u, cache={}):
        if 'gram_chol' not in cache or 'dc_dpos' not in cache:
            logger.info('Gram matrix Cholesky and/or constraint Jacobian not '
                        'available.')
            return self.energy_grad_direct(u)
        else:
            gram_chol = cache['gram_chol']
            dc_dpos = cache['dc_dpos']
            if 'dc_dpos_pinv' not in cache:
                cache['dc_dpos_pinv'] = la.cho_solve(gram_chol, dc_dpos)
            dc_dpos_pinv = cache['dc_dpos_pinv']
            return self.energy_grad_alt(u, dc_dpos_pinv)


class MinimalDifferentiableGenerativeModel(DifferentiableGenerativeModel):
    """Wrapper class for differentiable generative models.

    Assumed cached Gram matrix Cholesky result will always be available
    thus preventing need to compile alternative Theano expressions for
    target density energy function and gradient.
    """

    def _compile_theano_functions(self):
        u = tt.vector('u')
        y = self.generator(u, self.constants)
        u_rep = tt.tile(u, (y.shape[0], 1))
        y_rep = self.generator(u_rep, self.constants)
        dy_du = tt.grad(
            cost=None, wrt=u_rep, known_grads={y_rep: tt.identity_like(y_rep)})
        energy = (self.base_energy(u) +
                  0.5 * tt.log(nla.det(dy_du.dot(dy_du.T))))
        dy_du_pinv = tt.matrix('dy_du_pinv')
        energy_grad = u + tt.Lop(dy_du, u_rep, dy_du_pinv).sum(0)
        self.generator_func = _timed_func_compilation(
            [u], y, 'generator function')
        self.generator_jacob = _timed_func_compilation(
            [u], dy_du, 'generator Jacobian')
        self._energy_grad = _timed_func_compilation(
            [u, dy_du_pinv], energy_grad, 'energy gradient')
        self.base_energy_func = _timed_func_compilation(
            [u], self.base_energy(u), 'base energy function')

    def energy_func(self, u, cache={}):
        gram_chol = cache['gram_chol']
        return (self.base_energy_func(u) +
                np.log(gram_chol[0].diagonal()).sum())

    def energy_grad(self, u, cache={}):
        gram_chol = cache['gram_chol']
        dc_dpos = cache['dc_dpos']
        if 'dc_dpos_pinv' not in cache:
            cache['dc_dpos_pinv'] = la.cho_solve(gram_chol, dc_dpos)
        dc_dpos_pinv = cache['dc_dpos_pinv']
        return self._energy_grad(u, dc_dpos_pinv)
