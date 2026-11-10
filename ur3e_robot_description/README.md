# Universal Robots UR3e Description (MJCF)

Yêu cầu: MuJoCo 2.3.3 hoặc mới hơn.

## Tổng quan

Package này chứa mô hình robot UR3e dưới dạng MJCF (MuJoCo XML) được xây dựng dựa trên mô hình URDF gốc của Universal Robots. Mục tiêu chính là cung cấp một mô hình robot 6-DOF để kiểm tra động học, mô phỏng, và giải bài toán inverse kinematics (IK) trong môi trường MuJoCo.

Project này gồm hai phần chính:
- `ur3e_robot_description/`: mô hình robot và công cụ hiển thị/visualize.
- `ur3e_transformation_matrix/`: các script tính toán động học thuận, động học ngược, kiểm tra độ chính xác với MuJoCo.

## Cấu trúc project

```text
mujoco_projects/
├── ur3e_robot_description/
│   ├── ur3e.xml                  # Mô hình UR3e trong MuJoCo
│   ├── view.py                   # Mở viewer để quan sát robot
│   ├── assets/                   # Mesh và tài nguyên hình học
│   ├── LICENSE.txt               # Giấy phép
│   └── README.md                 # Tài liệu mô tả project
│
├── ur3e_transformation_matrix/
│   ├── Forward_kinematics.py     # Tính động học thuận + Jacobian
│   ├── Inverse_kinematics.py     # Giải IK bằng Newton-Raphson và DLS
│   ├── Inverse_kinematics_mjc.py # Chạy IK trong MuJoCo viewer
│   ├── simulation_check.py       # So sánh FK với MuJoCo
│   ├── IK_testings/
│   │   └── validate_IK_mjc.py    # Kiểm thử độ chính xác IK trên nhiều pose
│   └── __pycache__/             # Cache Python (tự sinh)
│
├── build/                        # Thư mục build ROS/colcon
├── install/                      # Thư mục cài đặt ROS/colcon
├── log/                          # Log build/test
└── ...
```

## Mô tả từng file / script

### 1) `ur3e.xml`
File mô hình robot UR3e dạng XML của MuJoCo, chứa:
- cấu trúc khớp và liên kết vật lý của robot,
- mesh/geometry của từng link,
- actuator điều khiển joint,
- keyframe pose gốc, 
- site `attachment_site` dùng để xác định pose cuối tay cầm.

File này là “nguồn dữ liệu chính” để load robot trong MuJoCo và dùng cho mô phỏng, IK, và kiểm tra động học.

### 2) `view.py`
Script mở MuJoCo viewer để hiển thị robot trong giao diện 3D.

Chức năng chính:
- load mô hình `ur3e.xml`,
- đổi thư mục làm việc về nơi chứa XML để mesh/asset resolve đúng,
- khởi tạo `MjModel` và `MjData`,
- chạy passive viewer, điều chỉnh camera và cập nhật từng bước mô phỏng.

Dùng để quan sát robot ở trạng thái mặc định hoặc theo các phản ứng của mô phỏng, tiện cho debugging hình học và visual check.

### 3) `assets/`
Thư mục chứa mesh, hình ảnh, hoặc tài sản visual của robot.

Nó được tham chiếu bởi file XML khi MuJoCo load mô hình. Nếu asset không đúng đường dẫn hoặc thiếu, robot sẽ không hiển thị đúng hình dạng.

### 4) `Forward_kinematics.py`
File chính để tính động học thuận của robot UR3e.

Chức năng:
- định nghĩa tham số DH (`ROBOT_PARAMS`),
- tạo bảng DH từ góc khớp `q = [q1, q2, q3, q4, q5, q6]`,
- tính ma trận biến đổi đồng nhất `A_i` và `T_0_i`,
- tích lũy ma trận để tìm `T_0_6` (Base -> End-Effector),
- tính Jacobian hình học `J(q)` từ các trục quay và vị trí khớp,
- in từng ma trận biến đổi để debug nếu cần.

Đây là nền tảng để các script khác lấy vị trí và hướng của đầu máy, hoặc giải IK.

### 5) `Inverse_kinematics.py`
File chứa các thuật toán giải động học ngược.

Các hàm chính:
- `inverse_kinematics_newton_raphson(...)`: giải IK bằng Newton-Raphson.
- `inverse_kinematics_dls(...)`: giải IK bằng Damped Least Squares (DLS).
- `compute_orientation_error(...)`: tính sai số hướng giữa 2 ma trận xoay.
- `inverse_kinematics_pose_dls(...)`: giải IK cho cả vị trí và hướng 6-DOF với trọng số hướng mềm.

File này là module IK chính của project.

### 6) `Inverse_kinematics_mjc.py`
Script chạy IK trực tiếp trên mô hình MuJoCo.

Chức năng:
- load mô hình `ur3e.xml`,
- định nghĩa các waypoint đường đi (point-by-point),
- dùng `inverse_kinematics_pose_dls` để tìm góc khớp cho từng điểm đích,
- nội suy chuyển động giữa các pose để robot trượt mượt trong viewer,
- cập nhật `data.qpos` và render qua MuJoCo viewer.

Dùng khi muốn quan sát robot thực hiện chuyển động theo tuyến đường cố định hoặc test IK trong môi trường 3D thực.

### 7) `simulation_check.py`
Script kiểm tra độ chính xác của mô hình động học thuận so với MuJoCo.

Chức năng:
- nạp mô hình MuJoCo,
- xét nhiều pose đặc biệt và 20 pose ngẫu nhiên,
- tính pose đầu cuối bằng phương pháp tự viết (FK),
- lấy pose thực tế từ MuJoCo,
- so sánh sai số vị trí và góc xoay,
- báo kết quả trung bình và lớn nhất của lỗi.

File này giúp xác nhận rằng bảng DH và ma trận biến đổi có đúng với robot thật trong MuJoCo hay chưa.

### 8) `IK_testings/validate_IK_mjc.py`
Script kiểm thử IK trên nhiều pose ngẫu nhiên trong MuJoCo.

Chức năng:
- sinh ngẫu nhiên 20 cấu hình joint,
- lấy target pose từ MuJoCo, 
- chạy giải IK từ pose khởi tạo gần đó,
- so sánh vị trí EE sau khi IK với target mục tiêu,
- thống kê số lần hội tụ và sai số vị trí trung bình/lớn nhất.

Dùng để đánh giá độ tin cậy của giải thuật IK trước khi triển khai cho ứng dụng hoặc mô phỏng thực tế.

## Luồng làm việc của project

Project này đi theo luồng sau:

1. Load mô hình UR3e từ `ur3e.xml`
2. Tính động học thuận bằng `Forward_kinematics.py`
3. Giải động học ngược bằng `Inverse_kinematics.py`
4. Kiểm tra dữ liệu của thuật toán FK/IK trong MuJoCo (`simulation_check.py`, `validate_IK_mjc.py`)
5. Hiển thị/quan sát robot trong viewer (`view.py`, `Inverse_kinematics_mjc.py`)

## Ghi chú

- `build/`, `install/`, `log/` là thư mục sinh ra từ quá trình build/colcon chứ không phải là code nguồn chính của robot.
- Để chạy các script Python, nên sử dụng môi trường có cài MuJoCo và các thư viện NumPy.
- Nếu muốn kiểm tra nhanh: chạy `Forward_kinematics.py` để xem kết quả FK; chạy `Inverse_kinematics.py` để thử nghiệm IK; chạy `simulation_check.py` để đối chiếu với MuJoCo.

## Lịch sử chuyển đổi mô hình

File mô hình này đã được xây dựng theo quy trình chuyển đổi từ URDF sang MJCF:

1. Chuyển mesh DAE sang OBJ.
2. Xử lý geometry bằng `obj2mjcf`.
3. Giữ lại visual geometry trong MuJoCo.
4. Thêm `tracking light`, actuator, damping, keyframe.
5. Tạo collision geometry và scene demo.

## License

Mô hình này được phát hành theo giấy phép [BSD-3-Clause License](LICENSE.txt).