import mujoco
import mujoco.viewer
import time

# Load model
import os
from pathlib import Path

xml_path = Path(__file__).resolve().parent / "ur3e.xml"
# Change working directory to the XML folder so meshdir (e.g. "assets/") resolves correctly
os.chdir(xml_path.parent)

model = mujoco.MjModel.from_xml_path(str(xml_path))
data = mujoco.MjData(model)

print(f"Loaded successfully! Bodies: {model.nbody}, Geoms: {model.ngeom}")

# Mở viewer dạng passive
with mujoco.viewer.launch_passive(model, data) as viewer:
    # Reset camera để nhìn toàn cảnh robot
    viewer.cam.distance = 2.0
    viewer.cam.azimuth = 90
    viewer.cam.elevation = -45
    
    while viewer.is_running():
        step_start = time.time()
        
        # Tiến trình mô phỏng
        mujoco.mj_step(model, data)
        viewer.sync()
        
        # Giữ tốc độ thời gian thực
        time_until_next_step = model.opt.timestep - (time.time() - step_start)
        if time_until_next_step > 0:
            time.sleep(time_until_next_step)