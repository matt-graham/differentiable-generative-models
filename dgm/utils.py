# -*- coding: utf-8 -*-
"""Utility functions."""

import os
import datetime
import logging
import theano as th
import theano.tensor as tt


def setup_logger(exp_dir):
    """Set up logger for use in experiments."""
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    time_stamp = datetime.datetime.now().strftime('%Y_%m_%d_%H_%M_%S')
    file_handler = logging.FileHandler(
        os.path.join(exp_dir, '{0}_experiment.log'.format(time_stamp)))
    formatter = logging.Formatter(
        '%(asctime)s %(name)s:%(levelname)s %(message)s', '%Y-%m-%d %H:%M:%S')
    file_handler.setFormatter(formatter)
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    logger.handlers = [file_handler, stream_handler]
    return logger


def partition(inputs, lengths):
    """Partition a one/two-dimesional tensor in to parts along last axis."""
    i = 0
    parts = []
    for l in lengths:
        parts.append(inputs.T[i:i+l].T)
        i += l
    return parts


def generator_decorator(generator):
    """Generator decorator adding boilerplate for coping with vector inputs."""
    def wrapped_generator(u, consts):
        if u.ndim == 1:
            u = u[None, :]
        n_batch = u.shape[0]
        return tt.squeeze(generator(u, consts).reshape((n_batch, -1)))
    return wrapped_generator
