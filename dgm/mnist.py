# -*- coding: utf-8 -*-
"""MNIST handwritten digit character generator."""

import theano as th
import theano.tensor as tt
import theano.tensor.slinalg as sla
from dgm.utils import partition, generator_decorator


def mnist_vae_decoder(h, layers):
    for layer in layers:
        h = layer['nonlinearity'](
            h.dot(layer['weights']) + layer['biases'])
    return h


@generator_decorator
def mnist_generator(u, consts):
    """Generate MNIST digit images (as flat vectors)."""
    h, n = partition(u, [consts['n_latent'],
                         consts['n_pixel']])
    m = mnist_vae_decoder(h, consts['mnist_vae_decoder_layers'])
    return m + consts['output_std'] * n


@generator_decorator
def mnist_region_generator(u, consts):
    """Generate region of MNIST digit images (as flat vectors)."""
    h, n = partition(u, [consts['n_latent'],
                         consts['n_observed']])
    m = mnist_vae_decoder(h, consts['mnist_vae_decoder_layers'])
    m_obs = m[:, consts['observed_slice']]
    return m_obs + consts['output_std'] * n
