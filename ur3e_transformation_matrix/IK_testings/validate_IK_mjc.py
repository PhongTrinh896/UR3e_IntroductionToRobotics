import os
import numpy as np
import mujoco

# Import các hàm từ file mã nguồn kinematics của bạn
from ur3e_transformation_matrix.Inverse_kinematics import forward_kinematics, inverse_kinematics_dls, ROBOT_PARAMS

# Đường dẫn tới file XML của bạn
MODEL_PATH = "/home/phongtrinh/mujoco_projects/ur3e_robot_description/ur3e.xml"

def run_validation_tests():
    if not os.path.exists(MODEL_PATH):
        print(f"[LỖI] Không tìm thấy file XML tại: {MODEL_PATH}")
        return

    model = mujoco.MjModel.from_xml_path(MODEL_PATH)
    data = mujoco.MjData(model)

    print("=" * 70)
    print(" BẮT ĐẦU KIỂM THỬ ĐỘ CHÍNH XÁC IK SO VỚI MUJOCO")
    print("=" * 70)

    np.random.seed(42)
    
    success_count = 0
    position_errors = []

    for i in range(20):
        q_random = np.random.uniform(-np.pi/2, np.pi/2, size=6)
        
        data.qpos[:6] = q_random
        mujoco.mj_forward(model, data)
        
        # Lấy vị trí EE thực tế trên MuJoCo làm target
        try:
            site_id = model.site('attachment_site').id
            target_pos = data.site_xpos[site_id].copy()
        except KeyError:
            body_id = model.body('wrist_3_link').id
            target_pos = data.xpos[body_id].copy()

        q_init = q_random + np.random.uniform(-0.1, 0.1, size=6)

        q_sol, success, _, _ = inverse_kinematics_dls(target_pos, q_init, damping=0.1)

        data.qpos[:6] = q_sol
        mujoco.mj_forward(model, data)
        
        try:
            pos_mujoco_after_ik = data.site_xpos[site_id].copy()
        except KeyError:
            pos_mujoco_after_ik = data.xpos[body_id].copy()

        err_m = np.linalg.norm(target_pos - pos_mujoco_after_ik)
        position_errors.append(err_m)

        if success and err_m < 1e-3:
            success_count += 1
            status = "PASSED"
        else:
            status = "WARNING"

        print(f"Test {i+1:02d}/20 | Status: {status} | Sai số vị trí thực tế: {err_m*1000:.4f} mm")

    mean_err = np.mean(position_errors)
    max_err = np.max(position_errors)

    print("=" * 70)
    print(" KẾT QUẢ TỔNG HỢP KIỂM THỬ IK")
    print("=" * 70)
    print(f" Tỷ lệ hội tụ thành công : {success_count}/20 ({success_count*5}%)")
    print(f" Sai số vị trí trung bình  : {mean_err*1000:.4f} mm")
    print(f" Sai số vị trí lớn nhất    : {max_err*1000:.4f} mm")
    print("=" * 70)

if __name__ == "__main__":
    run_validation_tests()