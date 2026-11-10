"""========= CHECK THE FK CALCULATED WITH MUJOCO MODEL =========="""
import os
import numpy as np
import mujoco
from Forward_kinematics import forward_kinematics, ROBOT_PARAMS

# ==============================================================================
# 1. DIRECT MUJOCO MODEL
# ==============================================================================
MODEL_PATH = "../ur3e_robot_description/ur3e.xml"

# Tên của Site / End-Effector trong file XML MuJoCo
EE_SITE_NAME = "attachment_site"  


def load_mujoco_model(xml_path):
    """Hàm nạp mô hình MuJoCo với kiểm tra đường dẫn"""
    if not os.path.exists(xml_path):
        # Thử tìm file XML ở thư mục hiện tại nếu đường dẫn trên không thấy
        alt_path = os.path.basename(xml_path)
        if os.path.exists(alt_path):
            xml_path = alt_path
        else:
            raise FileNotFoundError(
                f"Không tìm thấy file XML tại '{xml_path}'. "
                f"Vui lòng kiểm tra lại đường dẫn tới mô hình MuJoCo!"
            )
    
    model = mujoco.MjModel.from_xml_path(xml_path)
    data = mujoco.MjData(model)
    return model, data


def get_mujoco_ee_pose(model, data, q, site_name=EE_SITE_NAME):
    """Cập nhật góc q vào MuJoCo và lấy Vị trí (3x1) & Ma trận xoay (3x3) của EE"""
    data.qpos[:6] = q
    mujoco.mj_kinematics(model, data)  # Cập nhật Động học trong MuJoCo
    
    try:
        site_id = model.site(site_name).id
        pos_mj = data.site_xpos[site_id].copy()
        rot_mj = data.site_xmat[site_id].reshape(3, 3).copy()
    except KeyError:
        # Nếu không tìm thấy site, lấy theo body cuối cùng (wrist_3_link)
        body_id = model.body('wrist_3_link').id
        pos_mj = data.xpos[body_id].copy()
        rot_mj = data.xmat[body_id].reshape(3, 3).copy()
        
    return pos_mj, rot_mj


def validate_fk():
    print("=" * 75)
    print(" STARTING FK CHECKING WITH MUJOCO MODEL")
    print("=" * 75)

    # 1. Nạp mô hình
    try:
        model, data = load_mujoco_model(MODEL_PATH)
        print(f"[OK] SUCCESSFULLY IMPORTED MODEL: '{MODEL_PATH}'\n")
    except Exception as e:
        print(f"[LỖI] {e}")
        return

    # ==========================================================================
    # PHẦN 1: 5 MANUALLY INTERPRETABLE POSES
    # ==========================================================================
    pi = np.pi
    special_poses = [
        ("Zero Position", np.array([0, 0, 0, 0, 0, 0])),
        ("Pose 1", np.array([0, -pi/2, 0, 0, 0, 0])),
        ("Pose 2", np.array([0, -pi/2, pi/2, 0, 0, 0])),
        ("Pose 3", np.array([pi/2, -pi/4, pi/4, 0, pi/2, 0])),
        ("Pose 4", np.array([pi/4, -pi/3, -pi/6, pi/2, 0, pi/2]))
    ]

    print("--- PART 1: 5 CONFIGURATIONS CHECKING ---")
    for name, q in special_poses:
        # 1. Tính toán bằng hàm FK tự viết
        T_fk, _ = forward_kinematics(q, ROBOT_PARAMS)
        pos_fk = T_fk[0:3, 3]
        rot_fk = T_fk[0:3, 0:3]

        # 2. Lấy dữ liệu thực tế từ MuJoCo
        pos_mj, rot_mj = get_mujoco_ee_pose(model, data, q)

        # 3. Tính sai số
        pos_err = np.linalg.norm(pos_fk - pos_mj)
        rot_err = np.linalg.norm(rot_fk - rot_mj)  # Frobenious norm chênh lệch ma trận xoay

        status = "PASSED" if pos_err < 1e-3 else "FAILED"
        print(f"\n[{status}] CONFIG: {name}")
        print(f"  - q (rad): {q}")
        print(f"  - POS FK  : [{pos_fk[0]:.4f}, {pos_fk[1]:.4f}, {pos_fk[2]:.4f}]")
        print(f"  - POS MJ  : [{pos_mj[0]:.4f}, {pos_mj[1]:.4f}, {pos_mj[2]:.4f}]")
        print(f"  - POS ERROR  : {pos_err:.6f} m ({pos_err*1000:.3f} mm)")
        print(f"  - ROT ERROR : {rot_err:.6f}")

    # ==========================================================================
    # PHẦN 2: 20 CẤU HÌNH NGẪU NHIÊN (20 RANDOM JOINT CONFIGURATIONS)
    # ==========================================================================
    print("\n" + "=" * 75)
    print("--- PART 2: 20 RANDOM JOINT CONFIGURATIONS CHECK ---")
    print("=" * 75)

    np.random.seed(42)  # Cố định seed để tái lập kết quả
    pos_errors = []
    rot_errors = []

    for i in range(20):
        # Sinh góc ngẫu nhiên trong khoảng [-pi, pi]
        q_rand = np.random.uniform(-np.pi, np.pi, size=6)

        # Tính FK
        T_fk, _ = forward_kinematics(q_rand, ROBOT_PARAMS)
        pos_fk = T_fk[0:3, 3]
        rot_fk = T_fk[0:3, 0:3]

        # Tính MuJoCo
        pos_mj, rot_mj = get_mujoco_ee_pose(model, data, q_rand)

        # Tính sai số
        p_err = np.linalg.norm(pos_fk - pos_mj)
        r_err = np.linalg.norm(rot_fk - rot_mj)

        pos_errors.append(p_err)
        rot_errors.append(r_err)

        print(f"Test {i+1:02d}/20 | POS ERROR: {p_err*1000:.4f} mm | ROT ERROR: {r_err:.6f}")

    # ==========================================================================
    # TỔNG HỢP KẾT QUẢ KIỂM THỬ
    # ==========================================================================
    max_pos_err = np.max(pos_errors)
    mean_pos_err = np.mean(pos_errors)
    max_rot_err = np.max(rot_errors)

    print("\n" + "=" * 75)
    print(" RESULT OF 20 RANDOM JOINT CONFIGURATIONS")
    print("=" * 75)
    print(f" MEAN POS ERROR  : {mean_pos_err*1000:.4f} mm ({mean_pos_err:.6f} m)")
    print(f" MAX POS ERROR    : {max_pos_err*1000:.4f} mm ({max_pos_err:.6f} m)")
    print(f" MAX ROT ERROR (R)  : {max_rot_err:.6f}")
    
    if max_pos_err < 2e-3:
        print("\n>>> QUALIFIED (ERROR < 2mm)!")
    else:
        print("\n>>> WARNING: REVISE THE DH TABLE")
    print("=" * 75)


if __name__ == "__main__":
    validate_fk()