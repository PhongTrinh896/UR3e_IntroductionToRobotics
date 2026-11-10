import os
import time
import numpy as np

try:
    import mujoco
    import mujoco.viewer
except ModuleNotFoundError:
    print("[LỖI] Chưa cài đặt MuJoCo trong môi trường hiện tại.")
    print("Cài đặt bằng lệnh: pip install mujoco")
    raise

from Inverse_kinematics import forward_kinematics, compute_jacobian, inverse_kinematics_dls, inverse_kinematics_pose_dls ,ROBOT_PARAMS

# ==============================================================================
# CẤU HÌNH ĐƯỜNG DẪN MÔ HÌNH MUJOCO
# ==============================================================================
MODEL_PATH = "/home/phongtrinh/mujoco_projects/ur3e_robot_description/ur3e.xml"  

def main():
    if not os.path.exists(MODEL_PATH):
        print(f"[LỖI] Không tìm thấy file XML tại: {MODEL_PATH}")
        return

    # Nạp mô hình MuJoCo
    model = mujoco.MjModel.from_xml_path(MODEL_PATH)
    data = mujoco.MjData(model)

    print("=" * 60)
    print(" ĐANG CHẠY MÔ PHỎNG IK VỚI MUJOCO VIEWER")
    print("=" * 60)

    waypoints = [
        np.array([0.35,  0.15, 0.30]),
        np.array([0.25, -0.20, 0.40]),
        np.array([0.40,  0.00, 0.25]),
        np.array([0.30,  0.20, 0.45])
    ]

    print("[INFO] Đã khóa cố định hướng End-Effector cho toàn bộ quá trình chạy.")

    with mujoco.viewer.launch_passive(model, data) as viewer:

        home_pos = np.array([0, -np.pi/2, np.pi/2, np.pi, -np.pi/2, 0])

    # --- KHÓA CỐ ĐỊNH HƯỚNG MỘT LẦN DUY NHẤT Ở ĐÂY ---
        data.qpos[:6] = home_pos
        mujoco.mj_forward(model, data)

        q_current = data.qpos[:6].copy()
        viewer.sync()
        time.sleep(1)
        fixed_target_rot = np.array([[0, 1, 0],
                                [-1, 0, 0],
                                [0, 0, 1]])

        print("[OK] Đã mở MuJoCo Viewer thành công!\n")
        idx = 0
        while viewer.is_running():
            target_pos = waypoints[idx % len(waypoints)]

            # Sử dụng hướng đã cố định từ đầu, KHÔNG cập nhật lại từ T_current nữa
            target_rot = fixed_target_rot

            print(f"Đang giải IK tới điểm: X={target_pos[0]:.3f}, Y={target_pos[1]:.3f}, Z={target_pos[2]:.3f}")

            q_candidate, success, err = inverse_kinematics_pose_dls(
                target_pos,
                target_rot,
                q_current,
                damping=0.15,
                max_iter=300,
                orientation_weight=0.005
            )

            if success:
                print(f" -> Thành công! Sai số: {err*1000:.4f} mm")
            else:
                print(f" -> Đạt giới hạn lặp. Sai số: {err*1000:.4f} mm")

            q_target = q_candidate.copy()

            # Nội suy chuyển động giữa các góc khớp để robot trôi mượt trên giao diện 3D
            for q_step in np.linspace(q_current, q_target, 30):
                if not viewer.is_running():
                    break
                data.qpos[:6] = q_step
                mujoco.mj_forward(model, data)
                viewer.sync()
                time.sleep(0.02)

            q_current = q_target.copy()
            time.sleep(1.0)
            idx += 1

if __name__ == "__main__":
    main()