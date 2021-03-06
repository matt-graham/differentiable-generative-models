# -*- coding: utf-8 -*-
"""Human 3D pose and 2D projection generators."""

import theano as th
import theano.tensor as tt
import theano.tensor.slinalg as sla
from bvh import theano_renderer
from dgm.utils import (
    partition, generator_decorator, multi_output_generator_decorator)


@generator_decorator
def bone_lengths_generator(u, consts):
    """Generate skeleton bone lengths from log-normal model."""
    return tt.exp(consts['log_lengths_mean'] +
                  u.dot(consts['log_lengths_covar_chol']))


def joint_angles_cos_sin_vae_decoder(h, layers, n_joint_angle):
    h = layers[0]['nonlinearity'](
        h.dot(layers[0]['weights']) + layers[0]['biases'])
    # intermediate layers with skip-connections
    for layer in layers[1:-1]:
        h = layer['nonlinearity'](
            h.dot(layer['weights']) + layer['biases']) + h
    h = layers[-1]['nonlinearity'](
        h.dot(layers[-1]['weights']) + layers[-1]['biases'])
    return h[:, :n_joint_angle * 2], tt.exp(0.5 * h[:, n_joint_angle * 2:])


@generator_decorator
def joint_angles_generator(u, consts):
    """Generate joint angles from VAE decoder model."""
    h, n = partition(u, [consts['n_joint_angle_latent'],
                         consts['n_joint_angle'] * 2])
    ang_cos_sin_mean, ang_cos_sin_std = joint_angles_cos_sin_vae_decoder(
        h, consts['joint_angles_vae_decoder_layers'], consts['n_joint_angle'])
    ang_cos_sin = ang_cos_sin_mean + ang_cos_sin_std * n
    return tt.arctan2(ang_cos_sin[:, consts['n_joint_angle']:],
                      ang_cos_sin[:, :consts['n_joint_angle']])


def camera_generator(u, consts):
    """Generate camera parameters from (log-)normal model."""
    cam_foc = tt.ones_like(u[:, 0]) * consts['cam_foc']
    cam_pos = tt.concatenate([
        consts['cam_pos_x_mean'] + consts['cam_pos_x_std'] * u[:, 0:1],
        consts['cam_pos_y_mean'] + consts['cam_pos_y_std'] * u[:, 1:2],
        tt.exp(consts['log_cam_pos_z_mean'] +
               consts['log_cam_pos_z_std'] * u[:, 2:3])
    ], 1)
    cam_ang = tt.ones_like(u[:, :3]) * consts['cam_ang']
    return cam_foc, cam_pos, cam_ang


def joint_3d_pos_generator(u, consts):
    """Generate 3D joint positions.

    Generates bone lengths and joint angles from respective models then uses
    skeleton definition to convert to 3D joint positions in global coordinate
    system.
    """
    input_sizes = [consts['n_bone_length_input'],
                   consts['n_joint_angle_input']]
    u_len, u_ang = partition(u, input_sizes)
    bone_lengths = bone_lengths_generator(u_len, consts)
    joint_angles = joint_angles_generator(u_ang, consts)
    return tt.stack(theano_renderer.joint_positions_batch(
        consts['skeleton'], joint_angles, consts['fixed_joint_angles'],
        lengths=bone_lengths, lengths_map=consts['bone_lengths_map'],
        skip=consts['joints_to_skip']), 2)


def root_position_generator(u, consts):
    return u * consts['root_pos_std'] + consts['root_pos_mean']


def joint_3d_pos_var_root_generator(u, consts):
    """Generate 3D joint positions with variable root position.

    Generates bone lengths and joint angles from respective models then uses
    skeleton definition to convert to 3D joint positions in global coordinate
    system.
    """
    input_sizes = [consts['n_bone_length_input'],
                   consts['n_joint_angle_input'], 3]
    u_len, u_ang, u_pos = partition(u, input_sizes)
    bone_lengths = bone_lengths_generator(u_len, consts)
    joint_angles = joint_angles_generator(u_ang, consts)
    root_position = root_position_generator(u_pos, consts)
    joint_pos_3d = tt.stack(theano_renderer.joint_positions_batch(
        consts['skeleton'], joint_angles, consts['fixed_joint_angles'],
        lengths=bone_lengths, lengths_map=consts['bone_lengths_map'],
        skip=consts['joints_to_skip']), 2)
    joint_pos_3d = tt.inc_subtensor(joint_pos_3d[:, :3, :], root_position[:, :, None])
    return joint_pos_3d


@generator_decorator
def monocular_2d_proj_var_root_generator(u, consts):
    """Generate monocular 2D joint position projections with variable root position.

    Generates bone lengths and joint angles from respective models then uses
    skeleton definition to convert to 3D joint positions in global coordinate
    system, before projecting to 2D image coordinates using a single generated
    camera model.
    """
    input_sizes = [consts['n_bone_length_input'] +
                   consts['n_joint_angle_input'] + 3,
                   consts['n_camera_input']]
    u_ske, u_cam = partition(u, input_sizes)
    joint_pos_3d = joint_3d_pos_var_root_generator(u_ske, consts)
    cam_foc, cam_pos, cam_ang = camera_generator(u_cam, consts)
    camera_matrix = theano_renderer.camera_matrix_batch(
        cam_foc, cam_pos, cam_ang)
    joint_pos_2d_hom = tt.batched_dot(camera_matrix, joint_pos_3d)
    joint_pos_2d = (joint_pos_2d_hom[:, :2] /
                    joint_pos_2d_hom[:, 2][:, None, :])
    return joint_pos_2d


@generator_decorator
def monocular_2d_proj_generator(u, consts):
    """Generate monocular 2D joint position projections.

    Generates bone lengths and joint angles from respective models then uses
    skeleton definition to convert to 3D joint positions in global coordinate
    system, before projecting to 2D image coordinates using a single generated
    camera model.
    """
    input_sizes = [consts['n_bone_length_input'] +
                   consts['n_joint_angle_input'],
                   consts['n_camera_input']]
    u_ske, u_cam = partition(u, input_sizes)
    joint_pos_3d = joint_3d_pos_generator(u_ske, consts)
    cam_foc, cam_pos, cam_ang = camera_generator(u_cam, consts)
    camera_matrix = theano_renderer.camera_matrix_batch(
        cam_foc, cam_pos, cam_ang)
    joint_pos_2d_hom = tt.batched_dot(camera_matrix, joint_pos_3d)
    joint_pos_2d = (joint_pos_2d_hom[:, :2] /
                    joint_pos_2d_hom[:, 2][:, None, :])
    return joint_pos_2d


@generator_decorator
def noisy_monocular_2d_proj_generator(u, consts):
    """Generate noisy monocular 2D joint position projections.

    Generates bone lengths and joint angles from respective models then uses
    skeleton definition to convert to 3D joint positions in global coordinate
    system, before projecting to 2D image coordinates using a single generated
    camera model and adding Gaussian observation noise to projections.
    """
    input_sizes = [consts['n_bone_length_input'] +
                   consts['n_joint_angle_input'] +
                   consts['n_camera_input'],
                   consts['n_joint'] * 2]
    u_pos, u_noi = partition(u, input_sizes)
    return (monocular_2d_proj_generator(u_pos, consts) +
            consts['output_noise_std'] * u_noi)


@generator_decorator
def binocular_2d_proj_generator(u, consts):
    """Generate binocular 2D joint position projections.

    Generates bone lengths and joint angles from respective models then uses
    skeleton definition to convert to 3D joint positions in global coordinate
    system, before projecting to two sets of 2D image coordinates using two
    offset generated camera models.
    """
    n_batch = u.shape[0]
    input_sizes = [consts['n_bone_length_input'] +
                   consts['n_joint_angle_input'],
                   consts['n_camera_input']]
    u_ske, u_cam, = partition(u, input_sizes)
    joint_pos_3d = joint_3d_pos_generator(u_ske, consts)
    cam_foc, cam_pos, cam_ang = camera_generator(u_cam, consts)
    cam_mtx_1 = theano_renderer.camera_matrix_batch(
        cam_foc, cam_pos + consts['cam_pos_offset'],
        cam_ang + consts['cam_ang_offset'])
    cam_mtx_2 = theano_renderer.camera_matrix_batch(
        cam_foc, cam_pos - consts['cam_pos_offset'],
        cam_ang - consts['cam_ang_offset'])
    joint_pos_2d_hom_1 = tt.batched_dot(cam_mtx_1, joint_pos_3d)
    joint_pos_2d_1 = (joint_pos_2d_hom_1[:, :2] /
                      joint_pos_2d_hom_1[:, 2][:, None, :])
    joint_pos_2d_hom_2 = tt.batched_dot(cam_mtx_2, joint_pos_3d)
    joint_pos_2d_2 = (joint_pos_2d_hom_2[:, :2] /
                      joint_pos_2d_hom_2[:, 2][:, None, :])
    return tt.concatenate(
               [joint_pos_2d_1.reshape((n_batch, -1)),
                joint_pos_2d_2.reshape((n_batch, -1))], 1)


@generator_decorator
def noisy_binocular_2d_proj_generator(u, consts):
    """Generate noisy binocular 2D joint position projections.

    Generates bone lengths and joint angles from respective models then uses
    skeleton definition to convert to 3D joint positions in global coordinate
    system, before projecting to two sets of 2D image coordinates using two
    offset generated camera models and adding Gaussian observation noise to
    projections.
    """
    input_sizes = [consts['n_bone_length_input'] +
                   consts['n_joint_angle_input'] +
                   consts['n_camera_input'],
                   consts['n_joint'] * 4]
    u_pos, u_noi = partition(u, input_sizes)
    return (binocular_2d_proj_generator(u_pos, consts) +
            consts['output_noise_std'] * u_noi)


def inputs_to_state(u, consts):
    input_sizes = [consts['n_bone_length_input'],
                   consts['n_joint_angle_latent'],
                   consts['n_joint_angle'] * 2,
                   consts['n_camera_input']]
    u_len, joint_ang_latent, joint_ang_noise, u_cam = partition(u, input_sizes)
    log_bone_lengths = (consts['log_lengths_mean'] +
                        u_len.dot(consts['log_lengths_covar_chol']))
    joint_ang_cos_sin_mean, joint_ang_cos_sin_log_var = (
        consts['joint_angles_cos_sin_vae'].x_gvn_z(joint_ang_latent))
    joint_ang_cos_sin = tt.squeeze(
        joint_ang_cos_sin_mean +
        tt.exp(0.5 * joint_ang_cos_sin_log_var) * joint_ang_noise)
    cam_pos_x = consts['cam_pos_x_mean'] + consts['cam_pos_x_std'] * u_cam[0:1]
    cam_pos_y = consts['cam_pos_y_mean'] + consts['cam_pos_y_std'] * u_cam[1:2]
    log_cam_pos_z = (consts['log_cam_pos_z_mean'] +
                     consts['log_cam_pos_z_std'] * u_cam[2:3])
    return tt.concatenate([
        log_bone_lengths, joint_ang_latent, joint_ang_cos_sin,
        cam_pos_x, cam_pos_y, log_cam_pos_z], 0)


def joint_3d_pos_generator_hier(state, consts):
    state_partition = [
        consts['n_bone_length_input'],
        consts['n_joint_angle_latent'],
        consts['n_joint_angle'] * 2,
    ]
    log_bone_lengths, joint_ang_latent, joint_ang_cos_sin = (
        partition(state, state_partition))
    joint_angles = tt.arctan2(joint_ang_cos_sin.T[consts['n_joint_angle']:].T,
                              joint_ang_cos_sin.T[:consts['n_joint_angle']].T)
    bone_lengths = tt.exp(log_bone_lengths)
    return tt.squeeze(tt.stack(theano_renderer.joint_positions_batch(
        consts['skeleton'], joint_angles, consts['fixed_joint_angles'],
        lengths=bone_lengths, lengths_map=consts['bone_lengths_map'],
        skip=consts['joints_to_skip']), 2))


def energy_func_hier_monocular(state, y_data, consts):
    state_partition = [
        consts['n_bone_length_input'],
        consts['n_joint_angle_latent'],
        consts['n_joint_angle'] * 2,
        1, 1, 1
    ]
    (log_bone_lengths, joint_ang_latent,
     joint_ang_cos_sin, cam_pos_x,
     cam_pos_y, log_cam_pos_z) = partition(state, state_partition)
    ang_cos_sin_mean, ang_cos_sin_std = joint_angles_cos_sin_vae_decoder(
        joint_ang_latent, consts['joint_angles_vae_decoder_layers'],
        consts['n_joint_angle'])
    joint_angles = tt.arctan2(joint_ang_cos_sin[consts['n_joint_angle']:],
                              joint_ang_cos_sin[:consts['n_joint_angle']])
    bone_lengths = tt.exp(log_bone_lengths)
    joint_pos_3d = tt.stack(theano_renderer.joint_positions(
        consts['skeleton'], joint_angles, consts['fixed_joint_angles'],
        lengths=bone_lengths, lengths_map=consts['bone_lengths_map'],
        skip=consts['joints_to_skip']), 1)
    cam_foc = tt.exp(consts['cam_foc'])
    cam_pos = tt.concatenate([cam_pos_x, cam_pos_y, tt.exp(log_cam_pos_z)])
    cam_ang = consts['cam_ang']
    cam_mtx = theano_renderer.camera_matrix(cam_foc, cam_pos, cam_ang)
    joint_pos_2d_hom = cam_mtx.dot(joint_pos_3d)
    joint_pos_2d = joint_pos_2d_hom[:2] / joint_pos_2d_hom[2]
    y_model = joint_pos_2d.flatten()
    log_lengths_minus_mean = log_bone_lengths - consts['log_lengths_mean']
    return 0.5 * (
        (((y_data - y_model) / consts['output_noise_std'])**2).sum() +
        (((joint_ang_cos_sin - ang_cos_sin_mean) / ang_cos_sin_std)**2).sum() +
        joint_ang_latent.dot(joint_ang_latent) +
        log_lengths_minus_mean.dot(sla.solve_upper_triangular(
            consts['log_lengths_covar_chol'],
            sla.solve_lower_triangular(
                consts['log_lengths_covar_chol'].T,
                log_lengths_minus_mean)
        )) +
        ((cam_pos_x - consts['cam_pos_x_mean']) / consts['cam_pos_x_std'])**2 +
        ((cam_pos_y - consts['cam_pos_y_mean']) / consts['cam_pos_y_std'])**2 +
        ((log_cam_pos_z - consts['log_cam_pos_z_mean']) /
            consts['log_cam_pos_z_std'])**2
    )[0]


def energy_func_hier_binocular(state, y_data, consts):
    state_partition = [
        consts['n_bone_length_input'],
        consts['n_joint_angle_latent'],
        consts['n_joint_angle'] * 2,
        1, 1, 1
    ]
    (log_bone_lengths, joint_ang_latent,
     joint_ang_cos_sin,  cam_pos_x,
     cam_pos_y, log_cam_pos_z) = partition(state, state_partition)
    ang_cos_sin_mean, ang_cos_sin_std = joint_angles_cos_sin_vae_decoder(
        joint_ang_latent[None, :], consts['joint_angles_vae_decoder_layers'],
        consts['n_joint_angle'])
    joint_angles = tt.arctan2(joint_ang_cos_sin[consts['n_joint_angle']:],
                              joint_ang_cos_sin[:consts['n_joint_angle']])
    bone_lengths = tt.exp(log_bone_lengths)
    joint_pos_3d = tt.stack(theano_renderer.joint_positions(
        consts['skeleton'], joint_angles, consts['fixed_joint_angles'],
        lengths=bone_lengths, lengths_map=consts['bone_lengths_map'],
        skip=consts['joints_to_skip']), 1)
    cam_foc = consts['cam_foc']
    cam_pos = tt.concatenate([cam_pos_x, cam_pos_y, tt.exp(log_cam_pos_z)])
    cam_ang = consts['cam_ang']
    cam_mtx_1 = theano_renderer.camera_matrix(
        cam_foc, cam_pos + consts['cam_pos_offset'],
        cam_ang + consts['cam_ang_offset'])
    cam_mtx_2 = theano_renderer.camera_matrix(
        cam_foc, cam_pos - consts['cam_pos_offset'],
        cam_ang - consts['cam_ang_offset'])
    joint_pos_2d_hom_1 = tt.dot(cam_mtx_1, joint_pos_3d)
    joint_pos_2d_1 = joint_pos_2d_hom_1[:2] / joint_pos_2d_hom_1[2]
    joint_pos_2d_hom_2 = tt.dot(cam_mtx_2, joint_pos_3d)
    joint_pos_2d_2 = joint_pos_2d_hom_2[:2] / joint_pos_2d_hom_2[2]
    y_model = tt.concatenate([joint_pos_2d_1.flatten(),
                              joint_pos_2d_2.flatten()], 0)
    log_lengths_minus_mean = log_bone_lengths - consts['log_lengths_mean']
    return 0.5 * (
        (y_data - y_model).dot(y_data - y_model) /
        consts['output_noise_std']**2 +
        (((joint_ang_cos_sin - ang_cos_sin_mean) / ang_cos_sin_std)**2).sum() +
        joint_ang_latent.dot(joint_ang_latent) +
        log_lengths_minus_mean.dot(sla.solve_upper_triangular(
            consts['log_lengths_covar_chol'],
            sla.solve_lower_triangular(
                consts['log_lengths_covar_chol'].T,
                log_lengths_minus_mean)
        )) +
        ((cam_pos_x - consts['cam_pos_x_mean']) / consts['cam_pos_x_std'])**2 +
        ((cam_pos_y - consts['cam_pos_y_mean']) / consts['cam_pos_y_std'])**2 +
        ((log_cam_pos_z - consts['log_cam_pos_z_mean']) /
            consts['log_cam_pos_z_std'])**2
    )[0]
