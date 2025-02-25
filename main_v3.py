# version 2019-03-20
# extension of 2D to 3D
import os
import time
import ecos
import scipy.sparse
import numpy as np
from scipy.optimize import linear_sum_assignment, lsq_linear
import shelve
from scipy import polyfit
import matplotlib.pyplot as plt
import matplotlib as mpl
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt

plt.ion()


def do_association(xyz_target_possible, xyz_target_est, xyz_target_true):
    # d2[i,j] : the squared distance between the estimated location of node i and position j
    d2 = np.sum((xyz_target_est.reshape(-1, 1, 3) - xyz_target_possible.reshape(1, -1, 3)) ** 2, axis=2)
    idx_row, idx_col = linear_sum_assignment(d2)
    xyz_target_assigned = np.zeros_like(xyz_target_possible)
    xyz_target_assigned[idx_row] = xyz_target_possible[idx_col]
    is_correct = np.all(xyz_target_assigned == xyz_target_true, axis=1)
    return is_correct


def for_test_generate_simu_data(use_3d_anchors=False, use_12_anchors=False):
    z_ap = 0
    x_val = [0, 2.36, 4.72]
    y_val = [0.7 * i for i in range(52)]
    _X, _Y = np.meshgrid(x_val, y_val)
    xy_devices = np.c_[_X.flat, _Y.flat]
    is_anchor = (np.isclose(xy_devices[:, 0], min(x_val)) | np.isclose(xy_devices[:, 0], max(x_val))) & \
                (np.isclose(xy_devices[:, 1], min(y_val)) | np.isclose(xy_devices[:, 1], max(y_val)))
    xy_seat = xy_devices[~is_anchor]
    xy_anchor = xy_devices[is_anchor]

    xyz_seat = np.c_[xy_seat, np.ones((len(xy_seat), 1)) * z_ap]
    if use_3d_anchors:
        xyz_anchor = np.r_[
            np.c_[xy_anchor, np.ones((len(xy_anchor), 1)) * z_ap],
            np.c_[xy_anchor, np.ones((len(xy_anchor), 1)) * (z_ap+2.0)]
        ]
    elif use_12_anchors:
        _X, _Y, _Z = np.meshgrid([0, 2.36, 4.72], [0, 0.7*51], [0, 2])
        xyz_anchor = np.c_[_X.flat, _Y.flat, _Z.flat]
    else:
        xyz_anchor = np.c_[xy_anchor, np.ones((len(xy_anchor), 1)) * z_ap]

    x_val = [0, 2.36, 4.72],
    y_val = [7 * i for i in range(6)]
    _X, _Y = np.meshgrid(x_val, y_val)
    xyz_3d_target = np.c_[_X.flat, _Y.flat, np.ones((_X.size, 1)) * 1.5]
    xyz_devices = np.r_[xyz_seat, xyz_3d_target, xyz_anchor]  # 重新排列devices 前面为target 后面为anchor

    gamma = 2
    p0 = -20
    d0 = 1
    d_true = np.sqrt(
        np.sum((xyz_devices.reshape(-1, 1, 3) - xyz_devices.reshape(1, -1, 3)) ** 2, axis=2)
    )  # distance between nodes
    d_true[d_true == 0] = np.nan
    rss_true = p0 - gamma * 10 * np.log10(d_true / d0)  # measured distance
    return xyz_seat, xyz_3d_target, xyz_anchor, z_ap, d_true, gamma, p0, d0, rss_true


"""
# x0, y0, z0, x1, y1, z1, ...., x(N-1),y(N-1),z(N-1) gTx_1,....,gTx_(N+M-1), gRx_1,....,gRx_(N+M-1), gamma, p0
# <---------------       3*N   --------------------> <--    N+M-1       -->  <--     N + M -1 ---->    1   1
# Note: gTx_0 = -[gTx_1+....+gTx_(N-1)]
#       gRx_0 = -[gRx_1+....+gRx_(N-1)]
"""


def get_pos_idx_in_vec(i, N, M):
    """
    :param i: index of target node
    :param N: number of targets
    :param M: number of anchors
    :return: index of the xyz-coordinate of the i-th target in the vector of unknown variable, 3-element list
    """
    if 0 <= i < N:
        return [3 * i, 3 * i + 1, 3 * i + 2]
    else:
        raise ValueError('ERROR.')


def get_gtx_idx_in_vec(i, N, M):
    """
    :param i: index of target node
    :param N: number of targets
    :param M: number of anchors
    :return: index of the Tx Gain of the i-th target in the vector of unknown variable (a list), and the sign that
              should be multiplied, Note: gTx_0 = -[gTx_1+....+gTx_(N+M-1)]
    """
    if 0 == i:
        return [3 * N + i - 1 for i in range(1, N + M)], -1
    elif 0 < i < N + M:
        return [3 * N + i - 1], 1
    else:
        raise ValueError('ERROR.')


def get_grx_idx_in_vec(i, N, M):
    """
    :param i: index of target node
    :param N: number of targets
    :param M: number of anchors
    :return: index of the Rx Gain of the i-th target in the vector of unknown variable (a list), and the sign that
              should be multiplied, Note:    gRx_0 = -[gRx_1+....+gRx_(N+M-1)]
    """
    if 0 == i:
        return [3 * N + N + M - 1 + i - 1 for i in range(1, N + M)], -1
    elif 0 < i < N + M:
        return [3 * N + N + M - 1 + i - 1], 1
    else:
        raise ValueError('ERROR.')


def get_gamma_idx_in_vec(N, M):
    return 3 * N + 2 * (N + M - 1)


def get_p0_idx_in_vec(N, M):
    return 3 * N + 2 * (N + M - 1) + 1


def rss_coop_locn_socp_ecos(rss_meas, xyz_anchor,
                            gamma, p0, d0=1,
                            verbose=True,
                            min_x=None, max_x=None,
                            min_y=None, max_y=None,
                            min_z=None, max_z=None
                            ):
    """
    See rss_coop_locn_socp_scs() for algorithm description.
    This method uses the ecos package, which is more efficient than scs
    #
    Note: use the results of this method as initial estimate only
    #
    :param rss_meas: np.array, rss_{meas_{i,j}} : rss measured by the i-th device for packets from the j-th device
                      the first N devices are targets, the last M are anchors
                      no valid measurement if rss_{meas_{i,j}}== np.nan
                      *Symmetric*
    :param xyz_anchor: np.array with shape = (M, 2), location of the anchors
    :param gamma:
    :param p0:
    :param d0: path loss model  (  :math:`rss = p_0 - 10\gamma\log_{10}(d / d_0) + z`)
    :param verbose: print the progress
    :param min_x:
    :param max_x:
    :param min_y:
    :param max_y:
    :param min_z:
    :param max_z: boundary of the localization area / target locations
    :return: estimated target locations in a Nx3 array
    """
    rss_meas = np.asarray(rss_meas)
    xyz_anchor = np.asarray(xyz_anchor)
    num_anchor = len(xyz_anchor)
    num_target = len(rss_meas) - num_anchor
    N, M = num_target, num_anchor

    assert np.all(np.isclose(rss_meas, rss_meas.T, equal_nan=True))
    assert rss_meas.shape == (num_target + num_anchor, num_target + num_anchor)
    assert xyz_anchor.shape[0] == num_anchor

    # estimated distances
    # path loss model: rss = p0 - gamma * 10 * log10(d / d0) + z
    d_est = d0 * 10 ** ((p0 - rss_meas) / (10 * gamma))  # ignore the effect of shandowing

    # count the total number of valid rss measurements
    # total number of valid rss measurements that involve at least one target node
    num_valid_rss_meas = 0
    for i in range(num_target):
        for j in range(i + 1, num_target + num_anchor):
            if np.isnan(rss_meas[i, j]):
                continue
            num_valid_rss_meas += 1  # total valid rss measurement number (target to anchor)

    # number of inequality constraints ((min_x is not None) is 0 or 1 ) 若有 min_x 意味着有限制， 因此每一个target 都多了一个限制因素
    num_constraints_l = (
                                (min_x is not None)
                                + (max_x is not None)
                                + (min_y is not None)
                                + (max_y is not None)
                                + (min_z is not None)
                                + (max_z is not None)
                        ) * N

    # number of second order cone constraints
    num_constraints_q = num_valid_rss_meas

    # using ecos
    # min  c^T * x
    # s.t. Ax = b
    #      Gx <K= h    i.e.,   h - Gx in K  where K can be positive orthant or Second order cone
    #
    # optimization variable x :
    # x0, y0, z0, x1, y1 ,...., x_N-1, y_N-1, z_N-1, t
    # index of x_i  3 * i
    # index of y_i  3 * i + 1
    # index of z_i  3 * i + 2
    # index of t : -1
    dim_x = N * 3 + 1

    # A and b are not used
    num_row_G = num_constraints_l + 4 * num_constraints_q
    _G = scipy.sparse.lil_matrix((num_row_G, dim_x))
    _h = np.zeros(num_row_G)

    # fill in matrix _G and _h
    idx = 0

    # inequality constraints
    if min_x is not None:
        for i in range(N):
            # x_i >= min_x
            _G[idx, 3 * i] = -1
            _h[idx] = -min_x
            idx += 1
    if max_x is not None:
        for i in range(N):
            _G[idx, 3 * i] = 1
            _h[idx] = max_x
            idx += 1
    if min_y is not None:
        for i in range(N):
            # y_i >= min_y
            _G[idx, 3 * i + 1] = -1
            _h[idx] = -min_y
            idx += 1
    if max_y is not None:
        for i in range(N):
            # y_i <= max_y
            _G[idx, 3 * i + 1] = 1
            _h[idx] = max_y
            idx += 1
    if min_z is not None:
        for i in range(N):
            # z_i >= min_z
            _G[idx, 3 * i + 2] = -1
            _h[idx] = -min_z
            idx += 1
    if max_z is not None:
        for i in range(N):
            # z_i <= max_z
            _G[idx, 3 * i + 2] = 1
            _h[idx] = max_z
            idx += 1

    assert idx == num_constraints_l

    # SOC constraints
    for i in range(num_target):
        for j in range(i + 1, num_target + num_anchor):
            if np.isnan(rss_meas[i, j]):
                continue
            # rss_meas[i, j] valid
            # | (x_jj - x_ii, y_jj - y_ii) | / d2est[i, j] <= t
            if j < num_target:
                # both i and j are targets
                _G[idx + 0, -1] = -1
                _G[idx + 1, 3 * i] = 1 / d_est[i, j]
                _G[idx + 1, 3 * j] = -1 / d_est[i, j]
                _G[idx + 2, 3 * i + 1] = 1 / d_est[i, j]
                _G[idx + 2, 3 * j + 1] = -1 / d_est[i, j]
                _G[idx + 3, 3 * i + 2] = 1 / d_est[i, j]
                _G[idx + 3, 3 * j + 2] = -1 / d_est[i, j]
            else:
                # i is target, j is anchor
                _G[idx + 0, -1] = -1
                _G[idx + 1, 3 * i] = 1 / d_est[i, j]
                _G[idx + 2, 3 * i + 1] = 1 / d_est[i, j]
                _G[idx + 3, 3 * i + 2] = 1 / d_est[i, j]
                _h[idx + 1] = xyz_anchor[j - num_target, 0] / d_est[i, j]
                _h[idx + 2] = xyz_anchor[j - num_target, 1] / d_est[i, j]
                _h[idx + 3] = xyz_anchor[j - num_target, 2] / d_est[i, j]
            idx += 4
    assert idx == _G.shape[0]

    _c = np.r_[np.zeros(dim_x - 1), 1]
    _K = {
        'l': num_constraints_l,
        'q': [4] * num_constraints_q,
    }
    r = ecos.solve(_c, _G.tocsc(), _h, _K, verbose=verbose)

    x = r['x']
    xyz_target_est = x[:-1].reshape(-1, 3)
    # the objective value equals 1 if there is no noise or measurement error
    y_obj = x[-1]  # t
    return xyz_target_est, y_obj


def refine_node_locations_and_path_loss_model(rss_meas, xyz_anchor,
                                              xyz_target_init, gamma_init, p0_init,
                                              d0=1,
                                              coop_weight=1.0,
                                              min_ple=2.0,
                                              max_ple=3.0,
                                              max_gain_err=3.0,
                                              min_x=None, max_x=None,
                                              min_y=None, max_y=None,
                                              min_z=None, max_z=None,
                                              update_locn_only=False,
                                              max_iter=15,
                                              z_ap=0,
                                              w_z_penalty=0):
    """
    node locations and path loss model refined jointly using Gauss-Newton method
    Note: boundaries are critical for preventing divergence
    :param rss_meas: np.array, :math:`rss_{meas_{i,j}}` : rss measured by the i-th device for packets from the j-th device
                      the first num_target devices are targets, the last M are anchors
                      no valid measurement if :math:`rss_{meas_{i,j}}` == np.nan
                      Symmetric
    :param xyz_target_init: initial estimate of the target positions
    :param xyz_anchor: np.array with shape = (M, 2), location of the anchors
    :param gamma_init:
    :param p0_init:
    :param d0: path loss model  (  :math:`rss = p_0 - 10\gamma\log_{10}(d / d_0) + z`)
    :param coop_weight: the weight of peer-to-peer RSS measurement, a float value between 0 and 1
                        peer-to-peer RSS measurements are not used if coop_weight = 0
    :param min_x:
    :param max_x:
    :param min_y:
    :param max_y:
    :param min_z:
    :param max_z: boundary of the localization area / target locations
    :return: estimated target locations in a Nx3 array

    subfunctions:

        1. update_node_locations.obj_func(xyz_target, gamma, p0, p_tx, p_rx, d0=1)

          :param:
            xyz_target: shape  = (num_target, 3), target locations
            gamma : PLE
            p0 : Rx power at reference distance (d0)
            p_tx : Tx Gain
            p_rx : Rx Gain
            d0 : reference distance (1m)
          :return: objective value as a scalar

        2. update_node_locations.calc_first_order_approx(xyz0):

          Formulate a linear least square problem min |Ax - b|, where x stores the target locations,
          by calculating the first-order approximation of each

          p_0 - 10\gamma\log_{10}(d_{i,j}/ d_0) - rss_{meas}_{i,j}

        at xyz0 by:

          p_0 - 10\gamma\log_{10}(d_{i,j}_{0}/d_0)- 10\gamma\log_{10}(d_ij0^2)*(xyz0[i] - xyz0[j])*( xyz_i - xyz0[i] )
         - rss_meas[i,j] - 10\gamma\log_{10}(d_ij0^2) * (xyz0[j] - xyz0[i]).dot( xyz_j - xyz0[j] )

       where
       .. math::
           d_{ij}_{0} = |xyz_0[i] - xyz_0[j]|- 10 * gamma / (log(10) * d_{ij,0} ** 2) * (xyz_0[i] - xyz_0[j])

        d_ij0 is the partial derivative of p0 - gamma * 10 * log10(d_ij / d0) - rss_meas[i,j] w.r.t. xyz_i

       It is assumed that both i and j are targets, the formulation will be simpler if one of them is anchor.

          :param:
            xyz_0: shape  = (num_target, 3), current estimate of target locations
          :return: _A, _b  which forms a linear least square problem min |Ax - b|, where x stores the target locations
            each row of _A encodes  - 10 * gamma / (log(10) * d_ij0 ** 2) * (xyz0[i] - xyz0[j]).dot( xyz_i )
                                       - 10 * gamma / (log(10) * d_ij0 ** 2) * (xyz0[j] - xyz0[i]).dot( xyz_j )
          each element of _b encodes p0 - gamma * 10 * log10(d_ij0 / d0) - rss_meas[i,j]
                                      +10 * gamma / (log(10) * d_ij0 ** 2) * (xyz0[i] - xyz0[j]).dot( xyz0[i] )
                                      +10 * gamma / (log(10) * d_ij0 ** 2) * (xyz0[j] - xyz0[i]).dot( xyz0[j] )
    """
    num_targets = xyz_target_init.shape[0]  # number of targets
    num_nodes = len(rss_meas)  # total number of nodes
    rss_valid = ~np.isnan(rss_meas)

    N = num_targets
    M = num_nodes - num_targets
    # x0, y0, z0, x1, y1, z1, ...., x(N-1),y(N-1),z(N-1) gTx_1,....,gTx_(N+M-1), gRx_1,....,gRx_(N+M-1), gamma, p0
    # <---------------       3*N   --------------------> <--    N+M-1       -->  <--     N + M -1 ---->    1   1
    # Note: gTx_0 = -[gTx_1+....+gTx_(N-1)]
    #       gRx_0 = -[gRx_1+....+gRx_(N-1)]
    dim_x = 3 * N + 2 * (N + M - 1) + 2

    def obj_func(xyz_target, gamma, p0, g_tx, g_rx):
        """
        :param xyz_target: shape  = (num_target, 3), target locations
        :param gamma : PLE
        :param p0 : Rx power at reference distance (d0)
        :param g_tx : Tx Gain
        :param g_rx : Rx Gain
        :return: objective value as a scalar
        """
        # position of all devices
        xy_nodes_all = np.r_[xyz_target, xyz_anchor]
        # num_nodes x num_nodes, pair-wise distance between all nodes
        d = np.sqrt(
            np.sum((xy_nodes_all.reshape(num_nodes, 1, 3) - xy_nodes_all.reshape(1, num_nodes, 3)) ** 2, axis=2)
        )
        rss = p0 - 10 * gamma * np.log10(d / d0) + g_tx.reshape(1, -1) + g_rx.reshape(-1, 1)
        # sum of *weighted* square error
        y = 0
        for i in range(num_nodes):
            for j in range(num_nodes):
                # peer-to-peer
                if not rss_valid[i, j]:
                    continue
                e0 = rss[i, j] - rss_meas[i, j]

                if i < num_targets and j < num_targets:
                    y += coop_weight * e0 ** 2
                else:
                    y += e0 ** 2
        return y + np.sum((xyz_target[:,-1] - z_ap) ** 2) * w_z_penalty

    def calc_first_order_approx(xyz_target_est, gamma):
        """
        Formulate a linear least square problem min |Ax - b|, where x stores the target locations,
        by calculating the first-order approximation of each
           p0 - gamma * 10 * log10(d_{i,j}/ d0) - rss_meas_{i,j}
        at xyz0 by:
           p0 - gamma * 10 * log10(d_ij0 / d0) - rss_meas[i,j]
              - 10 * gamma / (log(10) * d_ij0 ** 2) * (xyz0[i] - xyz0[j]).dot(  xyz_i - xyz0[i] )
              - 10 * gamma / (log(10) * d_ij0 ** 2) * (xyz0[j] - xyz0[i]).dot(  xyz_j - xyz0[j] )
        where d_ij0 = |xy0[i] - xy0[j]|
              - 10 * gamma / (log(10) * d_ij0 ** 2) * (xyz0[i] - xyz0[j]) is the partial derivative of
              p0 - gamma * 10 * log10(d_ij / d0) - rss_meas[i,j] w.r.t. xyz_i
        It is assumed that both i and j are targets, the formulation will be simpler if one of them is anchor.

        :param xyz0: shape  = (num_target, 3), current estimate of target locations
        :return: _A, _b  which forms a linear least square problem min |Ax - b|, where x stores the target locations
        each row of _A encodes  - 10 * gamma / (log(10) * d_ij0 ** 2) * (xyz0[i] - xyz0[j]).dot( xyz_i )
                                       - 10 * gamma / (log(10) * d_ij0 ** 2) * (xyz0[j] - xyz0[i]).dot( xyz_j )
        each element of _b encodes p0 - gamma * 10 * log10(d_ij0 / d0) - rss_meas[i,j]
                                      +10 * gamma / (log(10) * d_ij0 ** 2) * (xyz0[i] - xyz0[j]).dot( xyz0[i] )
                                      +10 * gamma / (log(10) * d_ij0 ** 2) * (xyz0[j] - xyz0[i]).dot( xyz0[j] )
        """
        # position of all devices
        xyz = np.r_[xyz_target_est, xyz_anchor]
        # num_nodes x num_nodes, pair-wise distance between all nodes
        d = np.sqrt(
            np.sum((xyz.reshape(num_nodes, 1, 3) - xyz.reshape(1, num_nodes, 3)) ** 2, axis=2)
        )
        # rss_ij = p0 - 10 * gamma * np.log10(d / d0) + gTx_i + gRx_j
        _A = []
        _b = []
        for i in range(N + M):
            for j in range(N + M):
                if not rss_valid[i, j]:
                    continue
                d_ij = d[i, j]
                if 0 == d_ij:
                    continue

                if i < N and j < N:
                    # both i and j are targets
                    a_this = np.zeros(dim_x)
                    # jac is Jacobian of rss_ij wrt x[i]
                    # Jacobian of rss_ij wrt x[j] is -jac
                    jac_i = -10 * gamma / np.log(10) * (xyz[i] - xyz[j]) / d_ij ** 2
                    jac_j = -10 * gamma / np.log(10) * (xyz[j] - xyz[i]) / d_ij ** 2
                    # rss measured between targets are weighted by coop_weight in the objective function
                    a_this[get_pos_idx_in_vec(i, N, M)] = jac_i * np.sqrt(coop_weight)
                    a_this[get_pos_idx_in_vec(j, N, M)] = jac_j * np.sqrt(coop_weight)
                    a_this[get_gamma_idx_in_vec(N, M)] = -10 * np.log10(d_ij / d0) * np.sqrt(coop_weight)
                    a_this[get_p0_idx_in_vec(N, M)] = 1 * np.sqrt(coop_weight)
                    a_this[get_grx_idx_in_vec(i, N, M)[0]] = get_grx_idx_in_vec(i, N, M)[1] * np.sqrt(coop_weight)
                    a_this[get_gtx_idx_in_vec(j, N, M)[0]] = get_gtx_idx_in_vec(j, N, M)[1] * np.sqrt(coop_weight)
                    b_this = (rss_meas[i, j] + jac_i.dot(xyz[i]) + jac_j.dot(xyz[j])) * np.sqrt(coop_weight)
                    _A.append(a_this)
                    _b.append(b_this)
                elif i < N <= j:
                    # i target, j anchor
                    a_this = np.zeros(dim_x)
                    # jac is Jacobian of rss_ij wrt x[i]
                    # Jacobian of rss_ij wrt x[j] is -jac
                    jac_i = -10 * gamma / np.log(10) * (xyz[i] - xyz[j]) / d_ij ** 2
                    a_this[get_pos_idx_in_vec(i, N, M)] = jac_i
                    a_this[get_gamma_idx_in_vec(N, M)] = -10 * np.log10(d_ij / d0)
                    a_this[get_p0_idx_in_vec(N, M)] = 1
                    a_this[get_grx_idx_in_vec(i, N, M)[0]] = get_grx_idx_in_vec(i, N, M)[1]
                    a_this[get_gtx_idx_in_vec(j, N, M)[0]] = get_gtx_idx_in_vec(j, N, M)[1]
                    b_this = rss_meas[i, j] + jac_i.dot(xyz[i])
                    _A.append(a_this)
                    _b.append(b_this)
                elif j < N <= i:
                    # i anchor, j target
                    a_this = np.zeros(dim_x)
                    # jac is Jacobian of rss_ij wrt x[i]
                    # Jacobian of rss_ij wrt x[j] is -jac
                    jac_j = -10 * gamma / np.log(10) * (xyz[j] - xyz[i]) / d_ij ** 2
                    a_this[get_pos_idx_in_vec(j, N, M)] = jac_j
                    a_this[get_gamma_idx_in_vec(N, M)] = -10 * np.log10(d_ij / d0)
                    a_this[get_p0_idx_in_vec(N, M)] = 1
                    a_this[get_grx_idx_in_vec(i, N, M)[0]] = get_grx_idx_in_vec(i, N, M)[1]
                    a_this[get_gtx_idx_in_vec(j, N, M)[0]] = get_gtx_idx_in_vec(j, N, M)[1]
                    b_this = rss_meas[i, j] + jac_j.dot(xyz[j])
                    _A.append(a_this)
                    _b.append(b_this)
                else:
                    # N <= i and j
                    # both i and j are anchors
                    a_this = np.zeros(dim_x)
                    a_this[get_gamma_idx_in_vec(N, M)] = -10 * np.log10(d_ij / d0)
                    a_this[get_p0_idx_in_vec(N, M)] = 1
                    a_this[get_grx_idx_in_vec(i, N, M)[0]] = get_grx_idx_in_vec(i, N, M)[1]
                    a_this[get_gtx_idx_in_vec(j, N, M)[0]] = get_gtx_idx_in_vec(j, N, M)[1]
                    b_this = rss_meas[i, j]
                    _A.append(a_this)
                    _b.append(b_this)
        if 0 != w_z_penalty:
            for i in range(N):
                a_this = np.zeros(dim_x)
                a_this[get_pos_idx_in_vec(i, N, M)[2]] = 1 * np.sqrt(w_z_penalty)
                b_this = z_ap * np.sqrt(w_z_penalty)
                _A.append(a_this)
                _b.append(b_this)
        return np.array(_A), np.array(_b)

    lb = np.ones(dim_x) * -np.inf
    ub = np.ones(dim_x) * np.inf
    if min_x is not None:
        for i in range(N):
            lb[get_pos_idx_in_vec(i, N, M)[0]] = min_x
    if max_x is not None:
        for i in range(N):
            ub[get_pos_idx_in_vec(i, N, M)[0]] = max_x
    if min_y is not None:
        for i in range(N):
            lb[get_pos_idx_in_vec(i, N, M)[1]] = min_y
    if max_y is not None:
        for i in range(N):
            ub[get_pos_idx_in_vec(i, N, M)[1]] = max_y
    if min_z is not None:
        for i in range(N):
            lb[get_pos_idx_in_vec(i, N, M)[2]] = min_z
    if max_z is not None:
        for i in range(N):
            ub[get_pos_idx_in_vec(i, N, M)[2]] = max_z

    for i in range(1, N + M):
        lb[get_grx_idx_in_vec(i, N, M)[0]] = -max_gain_err
        lb[get_gtx_idx_in_vec(i, N, M)[0]] = -max_gain_err
        ub[get_grx_idx_in_vec(i, N, M)[0]] = max_gain_err
        ub[get_gtx_idx_in_vec(i, N, M)[0]] = max_gain_err

    lb[get_gamma_idx_in_vec(N, M)] = min_ple
    ub[get_gamma_idx_in_vec(N, M)] = max_ple

    last_cost = np.inf
    last_last_cost = np.inf
    xyz_target = xyz_target_init
    gamma = gamma_init

    debug_pos_update = 0
    debug_pl_model = 0

    for i_iter in range(max_iter):
        _A, _b = calc_first_order_approx(xyz_target, gamma)

        # if 0:
        #     x_true = np.r_[xyz_target_init.flatten(), np.zeros(2 * (N + M - 1)), 2.0, -20]
        #     print("cost value: ", np.linalg.norm(_A.dot(x_true) - _b))

        if debug_pos_update or update_locn_only:
            _A1 = _A[:, :3 * N]
            _b1 = _b - _A[:, -2:].dot([gamma_init, p0_init])
            r = lsq_linear(_A1, _b1, (lb[:3 * N], ub[:3 * N]))
            x = r.x
            xyz_target = x.reshape(-1, 3)
            gamma, p0 = gamma_init, p0_init
            gTx = np.zeros(N + M)
            gRx = np.zeros(N + M)
        elif debug_pl_model:
            _A1 = _A[:, 3 * N:]
            _b1 = _b - _A[:, :3 * N].dot(xyz_target_init.flatten())
            r = lsq_linear(_A1, _b1, (lb[3 * N:], ub[3 * N:]))
            x = r.x
            xyz_target = xyz_target_init
            gamma = x[-2]
            p0 = x[-1]
            gTx = np.zeros(N + M)
            gTx[1:] = x[:N + M - 1]
            gRx = np.zeros(N + M)
            gRx[1:] = x[N + M - 1:2 * (N + M - 1)]
            gTx[0] = -sum(gTx[1:])
            gRx[0] = -sum(gRx[1:])
        else:
            r = lsq_linear(_A, _b, (lb, ub))
            x = r.x

            xyz_target = x[:3 * N].reshape(-1, 3)
            gamma = x[get_gamma_idx_in_vec(N, M)]
            p0 = x[get_p0_idx_in_vec(N, M)]
            gTx = np.zeros(N + M)
            gRx = np.zeros(N + M)
            for i in range(1, N + M):
                gTx[i] = x[get_gtx_idx_in_vec(i, N, M)[0]]
                gRx[i] = x[get_grx_idx_in_vec(i, N, M)[0]]
            gTx[0] = -sum(gTx[1:])
            gRx[0] = -sum(gRx[1:])
        c = obj_func(xyz_target, gamma, p0, gTx, gRx)
        # if i_iter % 10 == 0:
        print('iteration {}, cost: {}'.format(i_iter, c))
        if np.abs(c - last_cost) < 5 or np.abs(c - last_last_cost) < 5:
            break
        last_last_cost = last_cost
        last_cost = c

    return xyz_target, gamma, p0, gTx, gRx, last_cost


def do_plot(xyz_targets, xyz_anchor):
    plt.figure()
    plt.plot(xyz_targets[:, 0], xyz_targets[:, 1], 'bo', label='Targets')
    plt.plot(xyz_anchor[:, 0], xyz_anchor[:, 1], 'rd', label='anchor')
    plt.xlabel('X (m)')
    plt.ylabel('Y (m)')
    plt.title('Top view')
    plt.legend()
    plt.axis('equal')
    plt.show()

    plt.figure()
    plt.plot(xyz_targets[:, 0], xyz_targets[:, 2], 'bo', label='Targets')
    plt.plot(xyz_anchor[:, 0], xyz_anchor[:, 2], 'rd', label='anchor')
    plt.xlabel('X (m)')
    plt.ylabel('Z (m)')
    plt.title('Front view')
    plt.legend()
    plt.axis('equal')
    plt.show()

    plt.figure()
    plt.plot(xyz_targets[:, 1], xyz_targets[:, 2], 'bo', label='Targets')
    plt.plot(xyz_anchor[:, 1], xyz_anchor[:, 2], 'rd', label='anchor')
    plt.xlabel('Y (m)')
    plt.ylabel('Z (m)')
    plt.title('Side view')
    plt.legend()
    plt.axis('equal')
    plt.show()


def fit_path_loss_model(rss_anchor, xyz_anchor, gamma=None, p0=None, d0=1):
    """
    Fit the following log-normal path-loss model to RSS measurements

    .. math::
       rss_{i,j}=p_0-10\gamma\log_{10}(d_{i,j}/d_0)+z_{i,j}

    :param rss_anchor: pair-wise RSS measurements between anchors, rss_anchor[i,j] is the RSS measured between anchor i and j. rss_anchor[i,j] = np.nan if it is not available.
    :param xy_anchor:  xy coordinates of anchors
    :param d0: reference distance (=1 m)
    :param gamma: path loss factor :math:`\gamma`, will be fixed if specified
    :param p0: RSS at reference distance (at most one of gamma/p0 can be specified)
    :return: Estimated gamma and p0 as a tuple
    """
    if gamma is not None and p0 is not None:
        return gamma, p0

    xyz_anchor = np.asarray(xyz_anchor)
    m = len(xyz_anchor)  # number of anchors
    d = np.sqrt(np.sum((xyz_anchor.reshape(m, 1, 3) - xyz_anchor.reshape(1, m, 3)) ** 2, axis=2))
    is_valid = ~np.isnan(rss_anchor)
    d_valid = d[is_valid]  # n_dim = 1
    rss_valid = rss_anchor[is_valid]  # n_dim = 1
    a = -10 * np.log10(d_valid / d0)
    # P0 + a * gamma = rss_valid
    if p0 is not None:
        # p0 is given
        gamma = a.dot(rss_valid - p0) / a.dot(a)
    elif gamma is not None:
        # gamma is given
        p0 = np.mean(rss_valid - a * gamma)
    else:
        # neither p0 or gamma is specified
        p = polyfit(a, rss_valid, 1)
        gamma, p0 = p[0], p[1]
        # the algorithm tend to go crazy if the path loss exponent is significantly incorrect
        if gamma < 1.8:  # min gamma = 1.8
            print('fit_path_loss_model(): Path loss exponent less than 1.8, using 1.8...')
            return fit_path_loss_model(rss_anchor, xyz_anchor, gamma=1.8, p0=None, d0=1)
        elif gamma > 3.5:  # max gamma = 3.5
            print('fit_path_loss_model(): Path loss exponent greater than 3.5, using 3.5...')
            return fit_path_loss_model(rss_anchor, xyz_anchor, gamma=3.5, p0=None, d0=1)
    return gamma, p0


def joint_estimation(rss_meas, xyz_anchor,
                     max_iter=15,
                     max_gain_err_db=9,
                     do_plot=False,
                     min_x=None, max_x=None, min_y=None, max_y=None, min_z=None, max_z=None,
                     z_ap=0,
                     w_z_penalty=0
                     ):
    num_anchor = len(xyz_anchor)
    num_target = len(rss_meas) - num_anchor
    # initialize gamma_init and p0_init
    d0 = 1
    gamma, p0 = fit_path_loss_model(rss_meas[num_target:, num_target:], xyz_anchor, d0=d0)
    g_tx = np.zeros(num_target + num_anchor)
    g_rx = np.zeros(num_target + num_anchor)

    # estimate the initial target locations
    rss_meas_c = rss_meas - g_tx.reshape(1, -1) - g_rx.reshape(-1, 1)
    rss_meas_c = (rss_meas_c + rss_meas_c.T) / 2
    # initial gamma set to 1.0 to penalize the credibility of measurements, which are very noisy because Tx/Rx gain
    # are unknown
    # xy_target, *_ = rss_coop_locn_socp_ecos(rss_meas_c, xy_anchor, gamma=1, p0=p0, d0=d0)
    xyz_target, y_obj = rss_coop_locn_socp_ecos(rss_meas_c, xyz_anchor,
                                                gamma=1, p0=p0, d0=d0,
                                                verbose=True,
                                                min_x=min_x, max_x=max_x,
                                                min_y=min_y, max_y=max_y,
                                                min_z=min_z, max_z=max_z
                                                )
    # no need to restrict the boundary, as the algorithm tends to 'shrink/compress' the network.
    # min_x=-0.5, max_x=5, min_y=-0.5, max_y=37)

    if do_plot:
        plt.figure()
        p, *_ = plt.plot(xyz_target[:, 0], xyz_target[:, 1], 'bo', label='SOCP initial estimate')
        plt.legend()
        plt.axis('equal')
        plt.show()
    # do not fully trust peer-to-peer cooperation during initialization
    xyz_target, gamma, p0, g_tx, g_rx, cost = refine_node_locations_and_path_loss_model(
        rss_meas_c,  xyz_anchor, xyz_target, gamma, p0, d0=1,
        coop_weight=0.5,
        min_ple=np.nan, max_ple=np.nan, max_gain_err=np.nan,
        min_x=-1, max_x=5.72, min_y=-2, max_y=39, min_z=0, max_z=2,
        update_locn_only=True,
        z_ap=z_ap,
        w_z_penalty=w_z_penalty
    )
    gamma, p0 = fit_path_loss_model(rss_meas, np.r_[xyz_target, xyz_anchor], d0=d0)
    xyz_target, gamma, p0, g_tx, g_rx, cost = refine_node_locations_and_path_loss_model(
        rss_meas_c,  xyz_anchor, xyz_target, gamma, p0, d0=1,
        coop_weight=1.0,
        min_ple=2.0,  max_ple=3.0, max_gain_err=max_gain_err_db,
        min_x=-1, max_x=5.72, min_y=-2, max_y=39, min_z=0, max_z=2,
        update_locn_only=True, max_iter=max_iter,
        z_ap=z_ap,
        w_z_penalty=w_z_penalty
    )
    if do_plot:
        plt.figure()
        p, *_ = plt.plot(xyz_target[:, 0], xyz_target[:, 1], 'bo', label='After Gauss-Newton refinement')
        plt.legend()
        plt.show()
    return xyz_target, gamma, p0, g_tx, g_rx


def one_test(i, sigma2_rss_meas):
    fname = 'test-{:04d}'.format(i)

    xyz_seat, xyz_3d_target, xyz_anchor, z_ap, d_true, gamma, p0, d0, rss_true = for_test_generate_simu_data()
    n2, n3, m = len(xyz_seat), len(xyz_3d_target), len(xyz_anchor)
    g_tx = np.random.randn(len(rss_true))
    g_rx = np.random.randn(len(rss_true))
    e = np.random.randn(rss_true.size) * np.sqrt(sigma2_rss_meas)

    rss_meas = rss_true + g_tx.reshape(1, -1) + g_rx.reshape(-1, 1) \
               + e.reshape(*rss_true.shape)

    xyz_seat_possible = xyz_seat
    xyz_3d_possible = xyz_3d_target
    try:
        xyz_target_est, gamma_est, p0_est, g_tx_est, g_rx_est = \
            joint_estimation(rss_meas, xyz_anchor,
                             z_ap,
                             n2=n2, n3=n3,
                             max_iter=20,
                             max_gain_err_db=10,
                             min_x=-1, max_x=5.72, min_y=-2, max_y=39, min_z=0, max_z=2,
                             do_plot=False,
                             )
        is_correct_seat = do_association(xyz_target_possible=xyz_seat_possible,
                                         xyz_target_est=xyz_target_est[:n2],
                                         xyz_target_true=xyz_seat)
        is_correct_3d = do_association(xyz_target_possible=xyz_3d_possible,
                                       xyz_target_est=xyz_target_est[n2:n2 + n3],
                                       xyz_target_true=xyz_3d_target)
        # print('accuracy of seat localization: ', sum(is_correct_seat) / len(is_correct_seat))
        # print('accuracy of 3D localization: ', sum(is_correct_3d) / len(is_correct_3d))

        with shelve.open(fname) as d:
            d['g_tx'] = g_tx
            d['g_rx'] = g_rx
            d['rss_true'] = rss_true
            d['rss_meas'] = rss_meas
            d['xyz_target_est'] = xyz_target_est
            d['is_correct_seat'] = is_correct_seat
            d['is_correct_3d'] = is_correct_3d

        return sum(is_correct_seat), len(is_correct_seat), sum(is_correct_3d), len(is_correct_3d)
    except:
        return 0, 0, 0, 0


def test_initialize_methods():
    xyz_seat, xyz_3d_target, xyz_anchor, z_ap, d_true, gamma, p0, d0, rss_true = for_test_generate_simu_data()
    # do_plot(xyz_seat, xyz_3d_target, xyz_anchor)
    rss_meas = rss_true
    xyz_target_est, y_obj = rss_coop_locn_socp_ecos(rss_meas, xyz_anchor,
                                                    gamma, p0, d0,
                                                    verbose=True,
                                                    min_x=-1, max_x=5.72, min_y=-2, max_y=39, min_z=0, max_z=2
                                                    )
    do_plot(xyz_target_est, xyz_anchor)


def test_refinement_method():
    sigma2_rss_meas = 1

    xyz_seat, xyz_3d_target, xyz_anchor, z_ap, d_true, gamma, p0, d0, rss_true = for_test_generate_simu_data()
    xyz_targets = np.r_[xyz_seat, xyz_3d_target]

    g_tx = np.random.randn(len(rss_true))
    g_rx = np.random.randn(len(rss_true))
    e = np.random.randn(rss_true.size) * np.sqrt(sigma2_rss_meas)

    rss_meas = rss_true + g_tx.reshape(1, -1) + g_rx.reshape(-1, 1) + e.reshape(*rss_true.shape)

    # using the true locations as starting point
    xyz_target_est, gamma_est, p0_est, gTx_est, gRx_est, cost = \
        refine_node_locations_and_path_loss_model(rss_meas,
                                                  xyz_anchor,
                                                  xyz_targets,
                                                  gamma,
                                                  d0=1,
                                                  coop_weight=1.0,
                                                  min_ple=2.0,
                                                  max_ple=3.0,
                                                  max_gain_err=10.0,
                                                  min_x=-1,
                                                  max_x=5.72,
                                                  min_y=-2,
                                                  max_y=39,
                                                  min_z=0,
                                                  max_z=2, )
    do_plot(xyz_target_est, xyz_anchor)


def test_overall_algorithm():
    sigma2_rss_meas = 0
    xyz_seat, xyz_3d_target, xyz_anchor, z_ap, d_true, gamma, p0, d0, rss_true = for_test_generate_simu_data()
    g_tx = np.random.randn(len(rss_true)) * 0
    g_rx = np.random.randn(len(rss_true)) * 0
    e = np.random.randn(rss_true.size) * np.sqrt(sigma2_rss_meas)

    rss_meas = rss_true + g_tx.reshape(1, -1) + g_rx.reshape(-1, 1) \
               + e.reshape(*rss_true.shape)

    xyz_target_est, gamma_est, p0_est, g_tx_est, g_rx_est = joint_estimation(
        rss_meas, xyz_anchor,
        max_iter=15,
        max_gain_err_db=9,
        do_plot=False,
        min_x=-1, max_x=5.72, min_y=-2, max_y=39, min_z=0, max_z=2,
    )
    do_plot(xyz_target_est, xyz_anchor)

    xyz_target_true = np.r_[xyz_seat, xyz_3d_target]
    is_correct = do_association(xyz_target_true, xyz_target_est, xyz_target_true)
    print("accuracy {}".format(sum(is_correct)/len(is_correct)))



if "__main__" == __name__:
    # test_refinement_method()
    # test_initialize_methods()
    # test_overall_algorithm()

    sigma_rss_meas = 5
    xyz_seat, xyz_3d_target, xyz_anchor, z_ap, d_true, gamma, p0, d0, rss_true = for_test_generate_simu_data(use_12_anchors=True)
    # raise ValueError(" ")
    g_tx = np.random.randn(len(rss_true)) * 1
    g_rx = np.random.randn(len(rss_true)) * 1
    e = np.random.randn(rss_true.size) * sigma_rss_meas

    if True:
        rss_meas = rss_true + g_tx.reshape(1, -1) + g_rx.reshape(-1, 1) \
                   + e.reshape(*rss_true.shape)
    else:
        rss_meas = np.load('rss_meas.npy')

    xyz_target_est, gamma_est, p0_est, g_tx_est, g_rx_est = joint_estimation(
        rss_meas, xyz_anchor,
        max_iter=15,
        max_gain_err_db=9,
        do_plot=False,
        min_x=-1, max_x=5.72, min_y=-2, max_y=39, min_z=0, max_z=2,
        w_z_penalty=0
    )
    do_plot(xyz_target_est, xyz_anchor)

    xyz_target_true = np.r_[xyz_seat, xyz_3d_target]
    is_correct = do_association(xyz_target_true, xyz_target_est, xyz_target_true)
    print("accuracy {}".format(sum(is_correct) / len(is_correct)))


    #
    # n_good_experiment = 0
    # n_correct_seat = 0
    # n_total_seat = 0
    # n_correct_3d = 0
    # n_total_3d = 0
    # for i in range(100):
    #     fname = 'test-{:04d}'.format(0)
    #     with shelve.open(fname) as d:
    #         is_correct_seat = d['is_correct_seat']
    #         is_correct_3d = d['is_correct_3d']
    #     # if sum(is_correct_seat) / len(is_correct_seat) > 0.7:
    #     n_good_experiment += 1
    #     n_correct_seat += sum(is_correct_seat)
    #     n_correct_3d += sum(is_correct_3d)
    #     n_total_seat += len(is_correct_seat)
    #     n_total_3d += len(is_correct_3d)
    # print('{} successful experiments'.format(n_good_experiment))
    # print('Seat Accuracy: {}/{} = {}%'.format(n_correct_seat, n_total_seat, n_correct_seat * 100 / n_total_seat))
    # print('3D Accuracy: {}/{} = {}%'.format(n_correct_3d, n_total_3d, n_correct_3d * 100 / n_total_3d))

    # from multiprocessing import Pool
    #
    # # number of experiments for each test case
    # num_tests = 100
    # # variance of RSS measurement error
    # sigma2_rss_meas_to_test = [25]
    #
    # num_rows = 52
    # for sigma2_rss_meas in sigma2_rss_meas_to_test:
    #     result_filename = '{}rows_{}dB2'.format(num_rows, sigma2_rss_meas)
    #     if os.path.exists(result_filename + '.dat'):
    #         print('test already done, skipping...')
    #         continue
    #
    #     with Pool(None) as p:
    #         results = p.starmap(one_test, [(i, sigma2_rss_meas) for i in range(num_tests)])
    #
    #     n_successful_experiment = 0
    #     n_correct_seat = 0
    #     n_total_seat = 0
    #     n_correct_3d = 0
    #     n_total_3d = 0
    #     for x1, y1, x2, y2 in results:
    #         n_correct_seat += x1
    #         n_total_seat += y1
    #         n_correct_3d += x2
    #         n_total_3d += y2
    #
    #         if 0 != y1 or 0 != y2:
    #             n_successful_experiment += 1
    #     print('{} experiments successfully finished for test with {} rows and {} dB2 variance'.format(
    #         n_successful_experiment, num_rows, sigma2_rss_meas
    #     ))
    #     with shelve.open(result_filename) as d:
    #         d['n_correct_seat'] = n_correct_seat
    #         d['n_total_seat'] = n_total_seat
    #         d['n_correct_3d'] = n_correct_3d
    #         d['n_total_3d'] = n_total_3d
    #         d['n_successful_experiment'] = n_successful_experiment
    #
    # for sigma2_rss_meas in sigma2_rss_meas_to_test:
    #     with shelve.open('{}rows_{}dB2'.format(num_rows, sigma2_rss_meas)) as d:
    #         n_correct_seat = d['n_correct_seat']
    #         n_total_seat = d['n_total_seat']
    #         n_correct_3d = d['n_correct_3d']
    #         n_total_3d = d['n_total_3d']
    #         n_successful_experiment = d['n_successful_experiment']
    #     print('{} experiments successfully finished for test with {} rows and {} dB2 variance'.format(
    #         n_successful_experiment, num_rows, sigma2_rss_meas
    #     ))
    #     print('Seat Accuracy: {}/{} = {}%'.format(n_correct_seat, n_total_seat, n_correct_seat * 100 / n_total_seat))
    #     print('3D Accuracy: {}/{} = {}%'.format(n_correct_3d, n_total_3d, n_correct_3d * 100 / n_total_3d))
