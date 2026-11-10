"""============= BASE FILE: IK SOLVER METHODS =============="""

import numpy as np
from Forward_kinematics import forward_kinematics, compute_jacobian, ROBOT_PARAMS

"""      1. NEWTON RAPHSON METHOD    """
def inverse_kinematics_newton_raphson(target_pos, q_init, max_iter=200, tol=1e-4):
    """
    Giải IK bằng phương pháp Newton-Raphson (Pseudoinverse).
    
    Input:
        target_pos : Vị trí mục tiêu [x, y, z] (3,)
        q_init     : Góc khớp khởi ban đầu (6,)
        max_iter   : Số vòng lặp tối đa
        tol        : Sai số cho phép (mét)
    Output:
        q          : Bộ góc khớp tìm được
        success    : True nếu hội tụ, False nếu thất bại
    """
    q = np.array(q_init, dtype=float)
    
    for i in range(max_iter):
        # 1. Tính FK hiện tại
        T, _ = forward_kinematics(q)
        current_pos = T[0:3, 3]
        
        # 2. Tính sai số vị trí (Error)
        error = target_pos - current_pos
        error_norm = np.linalg.norm(error)
        
        # Kiểm tra điều kiện dừng
        if error_norm < tol:
            return q, True, i, error_norm
            
        # 3. Tính Jacobian (lấy 3 dòng đầu tương ứng vị trí x, y, z)
        J = compute_jacobian(q)
        J_pos = J[0:3, :]  # Kích thước (3x6)
        
        # 4. Tính giả nghịch đảo Moore-Penrose: J^T * (J * J^T)^(-1)
        # Hoặc dùng np.linalg.pinv(J_pos)
        J_pinv = np.linalg.pinv(J_pos)
        
        # 5. Cập nhật góc khớp
        dq = J_pinv @ error
        q += dq
        
    return q, False, max_iter, error_norm


""" 2. DAMPED LEAST SQUARES METHOD """
def inverse_kinematics_dls(target_pos, q_init, damping=0.05, max_iter=200, tol=1e-5):
    """
    Giải IK bằng phương pháp Damped Least Squares (DLS).
    
    Input:
        target_pos : Vị trí mục tiêu [x, y, z] (3,)
        q_init     : Góc khớp ban đầu (6,)
        damping    : Hệ số cản lambda (ví dụ: 0.05 đến 0.1)
        max_iter   : Số vòng lặp tối đa
        tol        : Sai số cho phép (mét)
    Output:
        q          : Bộ góc khớp tìm được
        success    : True nếu hội tụ, False nếu thất bại
    """
    q = np.array(q_init, dtype=float)
    
    for i in range(max_iter):
        # 1. Tính FK hiện tại
        T, _ = forward_kinematics(q)
        current_pos = T[0:3, 3]
        
        # 2. Tính sai số
        error = target_pos - current_pos
        error_norm = np.linalg.norm(error)
        
        if error_norm < tol:
            return q, True, i, error_norm
            
        # 3. Tính Jacobian vị trí (3x6)
        J = compute_jacobian(q)
        J_pos = J[0:3, :]
        
        # 4. Công thức DLS: dq = J^T * (J * J^T + lambda^2 * I)^(-1) * error
        m, n = J_pos.shape
        I = np.eye(m)
        lambda_sq = damping ** 2
        
        # Tính toán ma trận nghịch đảo có cản
        J_dls = J_pos.T @ np.linalg.inv(J_pos @ J_pos.T + lambda_sq * I)
        
        # 5. Cập nhật góc khớp
        dq = J_dls @ error
        q += dq
        
    return q, False, max_iter, error_norm

""" ========== DLS FOR FULL POSITION + ROTATION ============"""
def compute_orientation_error(R_target, R_current):
    """
    Tính sai số hướng giữa 2 ma trận xoay (3x3), trả về véc-tơ sai số góc (3,)
    """
    # Sai số ma trận xoay
    R_err = R_target @ R_current.T
    
    # Chuyển đổi R_err sang dạng trục-góc (Angle-Axis) để lấy véc-tơ sai số góc
    # Công thức tính logarit ma trận xoay cho véc-tơ góc e_o:
    tr = np.trace(R_err)
    cos_theta = np.clip((tr - 1.0) / 2.0, -1.0, 1.0)
    theta = np.arccos(cos_theta)
    
    if abs(theta) < 1e-6:
        return np.zeros(3)
        
    omega = (theta / (2.0 * np.sin(theta))) * np.array([
        R_err[2, 1] - R_err[1, 2],
        R_err[0, 2] - R_err[2, 0],
        R_err[1, 0] - R_err[0, 1]
    ])
    
    # Chuyển đổi vận tốc góc về hệ tọa độ gốc (Base frame)
    e_o = R_current @ omega
    return e_o


def inverse_kinematics_pose_dls(target_pos, target_rot, q_init, damping=0.1, max_iter=200,
                              tol_pos=1e-4, tol_rot=1e-3, max_step=0.5, orientation_weight=0.1):
    """
    Giải IK đầy đủ vị trí và hướng (6-DOF) bằng phương pháp DLS.

    - orientation_weight nhỏ hơn 1 để ưu tiên vị trí hơn hướng, tránh xoay lật khớp
    - max_step giới hạn bước cập nhật, tránh nhảy nhánh IK quá lớn
    """
    q = np.array(q_init, dtype=float)
    prev_q = q.copy()

    for _ in range(max_iter):
        # 1. Tính FK hiện tại
        T, _ = forward_kinematics(q, ROBOT_PARAMS)
        current_pos = T[0:3, 3]
        current_rot = T[0:3, 0:3]

        # 2. Tính sai số vị trí và sai số hướng
        e_pos = target_pos - current_pos
        e_rot = compute_orientation_error(target_rot, current_rot)

        # Giảm trọng số cho hướng để robot không quay lung quá mạnh
        e_rot_weighted = orientation_weight * e_rot

        # Gộp thành véc-tơ sai số 6 chiều [e_pos; e_rot_weighted]
        error = np.concatenate([e_pos, e_rot_weighted])

        pos_err_norm = np.linalg.norm(e_pos)
        rot_err_norm = np.linalg.norm(e_rot_weighted)

        # Kiểm tra điều kiện hội tụ: ưu tiên vị trí, hướng chỉ là mục tiêu mềm
        if pos_err_norm < tol_pos and rot_err_norm < tol_rot:
            return q, True, pos_err_norm

        # 3. Lấy toàn bộ ma trận Geometric Jacobian (6x6)
        J = compute_jacobian(q, ROBOT_PARAMS)

        # 4. Áp dụng công thức DLS cho hệ 6x6
        m, n = J.shape
        I = np.eye(m)
        lambda_sq = damping ** 2

        J_dls = J.T @ np.linalg.inv(J @ J.T + lambda_sq * I)
        dq = J_dls @ error

        # Giới hạn bước cập nhật để tránh nhảy nhánh IK và xoay lật khớp
        step_norm = np.linalg.norm(dq)
        if step_norm > max_step:
            dq = dq / step_norm * max_step

        # 5. Cập nhật góc khớp
        q += dq

        # Nếu cập nhật quá lớn, giữ nhánh cũ để tránh đổi nhánh không kiểm soát
        if np.linalg.norm(q - prev_q) > 1.5:
            q = prev_q.copy()
            break
        prev_q = q.copy()

    return q, False, pos_err_norm


# ==============================================================================
# KIỂM THỬ THỰC TẾ (TEST SCRIPT)
# ==============================================================================
if __name__ == "__main__":

    q_seed = np.array([0.0, -0.5, 0.5, 0.0, 0.0, 0.0])
    target = np.array([0.3, 0.2, 0.4]) 
    print(f"Điểm đích cần tới (Target): {target}")
    
    # Chạy DLS IK
    q_sol_dls, success_dls, iter_dls, err_dls = inverse_kinematics_dls(target, q_seed)
    print("\n--- KẾT QUẢ PHƯƠNG PHÁP DLS ---")
    print(f"Hội tụ: {success_dls} (Số bước: {iter_dls}, Sai số: {err_dls*1000:.4f} mm)")
    print(f"Góc khớp tìm được q (rad): {q_sol_dls.round(4)}")
    print(f"Góc khớp tìm được q (deg): {q_sol_dls.round(4)*180/np.pi}")
    print(f"So sanh voi FK: {forward_kinematics(q_sol_dls, params=ROBOT_PARAMS)}") # HIEN 4 MA TRAN, CHI XET MA TRAN CUOI
    
    # Chạy Newton-Raphson IK
    q_sol_nr, success_nr, iter_nr, err_nr = inverse_kinematics_newton_raphson(target, q_seed)
    print("\n--- KẾT QUẢ PHƯƠNG PHÁP NEWTON-RAPHSON ---")
    print(f"Hội tụ: {success_nr} (Số bước: {iter_nr}, Sai số: {err_nr*1000:.4f} mm)")
    print(f"Góc khớp tìm được q (rad): {q_sol_nr.round(4)}")
    