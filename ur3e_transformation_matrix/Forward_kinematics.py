import numpy as np

ROBOT_PARAMS = {
    'LB': 0.15185,   # d1 (m)
    'a2': 0.24365,   # a2 (m)
    'a3': 0.2132,    # a3 (m)
    'd4': 0.13105,   # d4 (m)
    'd5': 0.08535,   # d5 (m)
    'd6': 0.0921     # d6 (m)
}

# DH table
def get_dh_table(q, params=ROBOT_PARAMS):
    """
    Joint vector q = [q1, q2, q3, q4, q5, q6] (Radian)
      i |  alpha_i (deg) |   a_i  |   d_i  |    theta_i
     ---|----------------|--------|--------|----------------
      1 |       90       |    0   |   LB   |      theta1
      2 |        0       |   a2   |    0   |      theta2
      3 |        0       |   a3   |    0   |      theta3
      4 |       90       |    0   |  -d4   |      theta4
      5 |       90       |    0   |   d5   |      theta5
      6 |        0       |    0   |   d6   |  theta6 + 180 deg
    """
    LB = params['LB']
    a2 = params['a2']
    a3 = params['a3']
    d4 = params['d4']
    d5 = params['d5']
    d6 = params['d6']

    deg90 = np.pi / 2
    deg180 = np.pi

    dh_table = [
        # [alpha_i, a_i, d_i, theta_i]
        [ deg90,  0,    LB, q[0]],
        [     0, a2,     0, q[1]],
        [     0, a3,     0, q[2]],
        [ deg90,  0,   -d4, q[3]],
        [ deg90,  0,    d5, q[4]],
        [     0,  0,    d6, q[5] + deg180]
    ]
    return dh_table

def T_matrix_calc(alpha, a, d, theta):
    ct = np.cos(theta)
    st = np.sin(theta)
    ca = np.cos(alpha)
    sa = np.sin(alpha)

    A = np.array([
        [ct, -st * ca,  st * sa, a * ct],
        [st,  ct * ca, -ct * sa, a * st],
        [ 0,       sa,       ca,      d],
        [ 0,        0,        0,      1]
    ], dtype=np.float64)

    return A

def forward_kinematics(q, params=ROBOT_PARAMS):
    """
    Input: 
        q: Mảng/List 6 góc khớp [q1, q2, q3, q4, q5, q6] (Radian)
    Output: 
        T_0_6: Ma trận biến đổi đồng nhất 4x4 từ Base đến End-Effector
        T_matrices: Danh sách các ma trận T_0_i từng khâu (dùng để tính Jacobian)
    """
    dh_table = get_dh_table(q, params)
    T = np.eye(4)
    T_matrices = [T.copy()]

    for row in dh_table:
        alpha, a, d, theta = row
        A_i = T_matrix_calc(alpha, a, d, theta)
        T = T @ A_i
        T_matrices.append(T.copy())

    return T, T_matrices


def compute_jacobian(q, params=ROBOT_PARAMS):
    """    
    Input:
        q: Góc khớp hiện tại [q1, q2, q3, q4, q5, q6] (Radian)
    Output:
        J: Ma trận Jacobian (6x6)
    """
    _, T_matrices = forward_kinematics(q, params)
    
    # Vị trí End-Effector (p_ee)
    p_ee = T_matrices[-1][0:3, 3]
    
    J = np.zeros((6, 6))

    for i in range(6):
        # Trục Z_(i) và vị trí P_(i) trong hệ tọa độ Base
        z_i = T_matrices[i][0:3, 2]
        p_i = T_matrices[i][0:3, 3]

        # Vận tốc dài: Jv_i = z_(i-1) x (p_ee - p_(i-1))
        J_v = np.cross(z_i, p_ee - p_i)
        
        # Vận tốc góc: Jw_i = z_(i-1)
        J_w = z_i

        J[0:3, i] = J_v
        J[3:6, i] = J_w

    return J