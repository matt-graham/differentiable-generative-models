# -*- coding: utf-8 -*-
"""Continuous state Lotka-Volterra predator-prey simulation generator."""

import theano as th
import theano.tensor as tt
from dgm.utils import partition, generator_decorator


@generator_decorator
def param_generator(u, constants):
    """Generate model parameters from log-normal model."""
    return tt.exp(constants['log_z_std'] * u + constants['log_z_mean'])


def lotka_volterra_step(n1, n2, y1, y2, z, dt):
    """Perform a single Euler-Maruyama integration step of Lotka-Volterra SDEs.

    Args:
        n1: Standard normal noise term on y1.
        n2: Standard normal noise term on y2.
        y1: Current prey population state.
        y2: Current predator population state.
        z: Model parameters.
        dt: Integrator time step.

    Returns:
        Updated population state pair (y1, y2).
    """
    y1 = (y1 + dt * (z[0] * y1 - z[1] * y1 * y2) + dt**0.5 * n1)
    y2 = (y2 + dt * (-z[2] * y2 + z[3] * y1 * y2) + dt**0.5 * n2)
    return y1, y2


@generator_decorator
def population_seq_generator(u, consts):
    """Generate simulated Lotka-Volterra model population sequences.

    If a vector of random inputs `u` is provided, a single (pair) of sequences
    is returned. If a matrix of random inputs `u` is provided, multiple
    simulations are run in parallel as a batch, with it assumed the first
    dimension of `u` corresponds to the batch dimension.

    Args:
        u: Tensor variable representing random inputs to generator. Either
            a one-dimensional vector or two-dimensional vector for batch
            generation (with first axis batch dimension).
        consts: Dictionary of fixed generator constants. Should define
            dt: integrator time-step,
            y1_init: prey population initial state,
            y2_init: predator population initial state,
            log_z_std: standard dev. of log-normal prior on model parameters,
            log_z_mean: mean of log-normal prior on model parameters.

    Returns:
        Tensor variable representing generated population sequences. If a
        vector of random inputs is provided the output will be a vector of
        interleaved y1 and y2 sequences [y1(1), y2(1) ... y1(T), y2(T)]. If
        a matrix of random inputs this will be matrix with each row
        corresponding to an interleaving of y1 and y2 sequences for the
        corresponding simulation in the batch.
    """
    u_z, u_n = partition(u, [consts['n_param'], consts['n_step'] * 2])
    # Sample model parameters from log-normal prior
    z = param_generator(u_z, consts)
    # Extract initial population states and replicate to batch size
    y1_init = tt.tile(tt.constant(
        consts['y1_init'], 'y1_init', 0, th.config.floatX), u.shape[0])
    y2_init = tt.tile(tt.constant(
        consts['y2_init'], 'y2_init', 0, th.config.floatX), u.shape[0])
    # Extract realisations of SDE white noise process terms as noise sequences
    n1_seq = u_n[:, :consts['n_step']] * consts['noise_std']
    n2_seq = u_n[:, consts['n_step']:] * consts['noise_std']
    # Iterate integrator steps given noise sequences and model parameters
    [y1_seq, y2_seq], updates = th.scan(
        fn=lotka_volterra_step,
        sequences=[n1_seq, n2_seq],
        outputs_info=[y1_init, y2_init],
        non_sequences=[z.T, consts['dt']],
    )
    # Stack y1 and y2 sequences together to produce generator output
    return tt.stack([y1_seq.T, y2_seq.T], axis=2).reshape([u.shape[0], -1])
