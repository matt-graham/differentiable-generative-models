# -*- coding: utf-8 -*-
"""Human pose model experiment matplotlib visualisation utilities."""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


# list of pairs of joint indices defining skeleton bones
bones = [
    (0, 1),
    (1, 2),
    (1, 3),
    (3, 4),
    (4, 5),
    (5, 6),
    (1, 7),
    (7, 8),
    (8, 9),
    (9, 10),
    (0, 11),
    (11, 12),
    (12, 13),
    (13, 14),
    (0, 15),
    (15, 16),
    (16, 17),
    (17, 18),
]


def plot_joint_projections_2d(joints, ax=None, col_joint='r', col_bone='r',
                              ms_joint=1., lw_bone=1.):
    if ax is None:
        fig = plt.figure(figsize=fig_size)
        ax = fig.add_subplot(111)
    ax.plot(joints[:, 0], joints[:, 1], 'o', color=col_joint, ms=ms_joint)
    for (i, j) in bones:
        ax.plot([joints[i, 0], joints[j, 0]],
                [joints[i, 1], joints[j, 1]], '-', color=col_bone, lw=lw_bone)
    return ax


def plot_monocular_projections(joints, fig_size=(6, 6)):
    fig = plt.figure(figsize=fig_size)
    ax = fig.add_subplot(111)
    plot_joint_projections_2d(joints, ax=ax)
    ax.axis(np.array([-1., 1., -1., 1.]) * 2.)
    return fig, ax


def plot_binocular_projections(joints_1, joints_2, fig_size=(12, 6)):
    fig = plt.figure(figsize=fig_size)
    ax_1 = fig.add_subplot(121)
    ax_2 = fig.add_subplot(122)
    for ax, joints in zip([ax_1, ax_2], [joints_1, joints_2]):
        plot_joint_projections_2d(joints, ax=ax)
        ax.axis(np.array([-1., 1., -1., 1.]) * 2.)
    return fig, ax_1, ax_2


def plot_3d_pose(joints_3d, fig_size=(8, 8),
                 col=['b', 'g', 'r', 'c', 'm', 'y', 'k']):
    fig = plt.figure(figsize=fig_size)
    ax = fig.add_subplot(111, projection='3d')
    if joints_3d.ndim == 2:
        joints_3d = joints_3d[None, :, :]
    for k, j3d in enumerate(joints_3d):
        ax.plot(j3d[:, 0], j3d[:, 2], j3d[:, 1], 'o', color=col[k % len(col)])
        for (i, j) in bones:
            ax.plot([j3d[i, 0], j3d[j, 0]],
                    [j3d[i, 2], j3d[j, 2]],
                    [j3d[i, 1], j3d[j, 1]], '-', color=col[k % len(col)])
    ax.set_xlim3d(-1, 1)
    ax.set_ylim3d(-1, 1)
    ax.set_zlim3d(-1, 1)
    fig.tight_layout()
    return fig, ax


def plot_posterior_mean_rmse(true_val, samples, fig_size=(8, 8)):
    mean_ests = np.cumsum(samples, 0) / (
        np.arange(samples.shape[0])[:, None] + 1.)
    rmses = ((mean_ests - true_val)**2).mean(1)**0.5
    fig = plt.figure(figsize=fig_size)
    ax = fig.add_subplot(111)
    ax.plot(rmses)
    ax.set_xlabel('Number of samples')
    ax.set_ylabel('RMSE')
    return fig, ax, rmses
