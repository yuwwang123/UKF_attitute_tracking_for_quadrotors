
import numpy as np
from scipy import io
from quaternion import Quaternion
import os
import math
from scipy.spatial.transform import Rotation as R
from numpy.linalg import norm
import matplotlib.pyplot as plt

# # Process & Measurement Noise
# Q = np.zeros((6, 6))
# np.fill_diagonal(Q, [0.01 ** 2, 0.01 ** 2, 0.01 ** 2, 0.001, 0.001, 0.001])
Q = 10.0 * np.identity(6)
R_ACCEL = 1000.0 * np.identity(3)
R_GYRO = 100 * np.identity(3)


def vicon_to_eulers(vicon):
    assert vicon['ts'].shape[1] == vicon['rots'].shape[2]
    vicon_ts = vicon['ts'].reshape(-1)

    vicon_eulers = []
    for i in range(len(vicon_ts)):
        q = Quaternion()
        q.from_rotm(vicon['rots'][:, :, i])
        euler = q.euler_angles()
        vicon_eulers.append(euler)

    return vicon_ts, np.array(vicon_eulers)


def process_imu_raw(accel_raw, gyro_raw):
    accel_x = -accel_raw[0, :]
    accel_y = -accel_raw[1, :]
    accel_z = accel_raw[2, :]

    accel = np.array([accel_x, accel_y, accel_z]).T

    #     accel_sensitivity = 330.0
    #     acc_scaling = 3300.0/(1023.0*accel_sensitivity)
    #     # Take the first 500 timestamps during which the robot is still
    # #     z_bias = np.mean(accel_z[:500]) - 9.81/scaling
    # #     x_bias = np.mean(accel_x[:500])
    # #     y_bias = np.mean(accel_y[:500])
    # #     bias = np.array([x_bias, y_bias, z_bias])

    #     acc_bias = np.mean(accel[:10], axis=0) - np.array([0,0,1])/acc_scaling
    #     accel = (accel - acc_bias)*acc_scaling

    acc_bias = np.array([-510, -501, 500])
    acc_scaling = 0.09328768

    accel = (accel - acc_bias) * acc_scaling

    gyro_x = gyro_raw[1, :]
    gyro_y = gyro_raw[2, :]
    gyro_z = gyro_raw[0, :]
    gyro = np.array([gyro_x, gyro_y, gyro_z]).T

    gyro_sensitivity = 3.33
    gyro_scale_factor = 3300 / 1023 / gyro_sensitivity
    gyro_bias = np.mean(gyro[:10], axis=0)
    gyro = (gyro - gyro_bias) * gyro_scale_factor * (np.pi / 180)

    return accel, gyro


def compute_sigma_points(q, omega, covariance):
    n = covariance.shape[0]
    assert n == 6
    cov_sqrt = np.linalg.cholesky(covariance)
    cov_sqrt *= np.sqrt(2 * n)
    sigma_pts = []
    for i in range(n):
        # quaternion sigma pts
        w_q = cov_sqrt[0:3, i]
        q_w = Quaternion()
        q_w.from_axis_angle(w_q)
        w_omega = cov_sqrt[3:, i]

        sigma_pts.append((q*q_w, omega+w_omega))
        sigma_pts.append((q*q_w.inv(), omega-w_omega))

    return sigma_pts

def compute_quat_mean(qs):
    q_mean = Quaternion()
    error_vectors = []
    count = 0
    found = False
    for t in range(500):
        error_vectors.clear()
        for q in qs:
            e = q * q_mean.inv()
            e_vec = e.axis_angle()
            error_vectors.append(e_vec)

        e_mean = np.mean(np.array(error_vectors), axis=0)
        e_mean_quat = Quaternion()
        e_mean_quat.from_axis_angle(e_mean)
        q_mean = e_mean_quat * q_mean
        # check for convergence

        if norm(e_mean) < 0.002:
            found = True
            break

        count += 1

    # print('found {} ! {} error: {}'.format(found, count, norm(e_mean)))

    return q_mean, np.array(error_vectors)


def process_model(sigma_pts, dq):
    ### Propagate each sigma points with the process model

    sigma_pts_pred = []
    for pt in sigma_pts:
        q_i, omega_i = pt
        sigma_pts_pred.append((q_i*dq, omega_i))
        # nothing to do for omega as it stays the same in the process model
    return sigma_pts_pred

def measurement_model(sigma_qs, sigma_omegas, measurement_type):
    predicted_measurements = []
    if measurement_type == 'gyro':
        predicted_measurements = sigma_omegas
    elif measurement_type == 'accel':
        for q in sigma_qs:
            g_quat = Quaternion(0, [0, 0, 9.81])
            accel_in_body = (q.inv() * g_quat * q).vec()
            predicted_measurements.append(accel_in_body)
    return np.array(predicted_measurements)


def form_W_set(q_error_vectors, omega_mean, sigma_omegas):
    W_set = []
    assert len(q_error_vectors) == len(sigma_omegas)
    for i in range(len(q_error_vectors)):
        w = np.concatenate((q_error_vectors[i], sigma_omegas[i] - omega_mean))
        W_set.append(w)
    return np.array(W_set)


def compute_cov_matrices(W_set, predicted_measurements, measurement_type):
    zz_ls = []
    xz_ls = []
    assert len(W_set) == len(predicted_measurements)
    for i in range(len(W_set)):
        zz_ls.append(np.outer(predicted_measurements[i], predicted_measurements[i]))
        xz_ls.append(np.outer(W_set[i], predicted_measurements[i]))
    zz_ls = np.array(zz_ls)
    xz_ls = np.array(xz_ls)
    cov_zz = np.mean(zz_ls, axis=0)  # 6x6
    cov_xz = np.mean(xz_ls, axis=0)  # 6x3

    if measurement_type == 'gyro':
        cov_zz += R_GYRO
    elif measurement_type == 'accel':
        cov_zz += R_ACCEL
    return cov_xz, cov_zz


def update(q_pred, omega_pred, P_prior, measurement_pred, measurement, cov_xz, cov_zz):
    K = np.matmul(cov_xz, np.linalg.inv(cov_zz))
    P = P_prior - K @ cov_zz @ K.T
    innovation = measurement - measurement_pred
    dq = Quaternion()
    dq.from_axis_angle((K @ innovation)[:3])
    q = dq * q_pred
    omega = omega_pred + (K @ innovation)[3:]
    return q, omega, P



def estimate_rot(data_num=1):
    # load data

    imuRaw = io.loadmat(os.path.join(os.path.dirname(__file__), "imu/imuRaw" + str(data_num) + ".mat"))
    imu = imuRaw['vals']
    # gyros = imug_cal.T
    accels, gyros = process_imu_raw(imu[0:3, :].view(np.int16),
                               imu[3:6, :].view(np.int16))
    # accels = imua_cal.T
    Ts = imuRaw['ts'].reshape(-1)


    # initilize state belief and covariance
    state = Quaternion(), np.zeros(3)
    P = 10.0 * np.identity(6)

    rolls, pitches, yaws = [0.0], [0.0], [0.0]
    ang_vels = [np.zeros(3)]
    cov_traces = [60]
    for i in range(len(Ts) - 1):
        q, omega = state
        dt = Ts[i + 1] - Ts[i]
        dq = Quaternion()
        dq.from_axis_angle(omega * dt)

        sigma_pts = compute_sigma_points(q, omega, P + Q)

        # Process Model: propagate sigma points
        sigma_pts_pred = process_model(sigma_pts, dq)
        sigma_qs_pred = np.array([pt[0] for pt in sigma_pts_pred])
        sigma_omegas_pred = np.array([pt[1] for pt in sigma_pts_pred])

        q_pred_mean, error_vectors = compute_quat_mean(sigma_qs_pred)
        omega_pred_mean = np.mean(sigma_omegas_pred, axis=0)

        W_set = form_W_set(error_vectors, omega_pred_mean, sigma_omegas_pred)
        P_prior = np.mean(np.array([np.outer(w, w) for w in W_set]), axis=0)
        # Measurement Model:
        # Gyroscope
        predicted_measurements = measurement_model(sigma_qs_pred, sigma_omegas_pred, measurement_type='gyro')
        measurement_pred = np.mean(np.array(predicted_measurements), axis=0)
        cov_xz, cov_zz = compute_cov_matrices(W_set, predicted_measurements, measurement_type = 'gyro')
        q_updated, omega_updated, P_updated = update(q_pred_mean, omega_pred_mean, P_prior, measurement_pred, gyros[i], cov_xz, cov_zz)

        #Accelerometer
        predicted_measurements = measurement_model(sigma_qs_pred, sigma_omegas_pred, measurement_type='accel')
        measurement_pred = np.mean(np.array(predicted_measurements), axis=0)
        cov_xz, cov_zz = compute_cov_matrices(W_set, predicted_measurements, measurement_type = 'accel')
        q_updated, omega_updated, P_updated = update(q_updated, omega_updated, P_updated, measurement_pred, accels[i], cov_xz, cov_zz)

        state = q_updated, omega_updated
        P = P_updated

        # logging
        roll, pitch, yaw = q_updated.euler_angles()

        rolls.append(roll)
        pitches.append(pitch)
        yaws.append(yaw)
        ang_vels.append(omega_updated)
        cov_traces.append(np.trace(P_updated))

    # roll, pitch, yaw are numpy arrays of length T
    return np.array(rolls), np.array(pitches), np.array(yaws), np.array(ang_vels), cov_traces


