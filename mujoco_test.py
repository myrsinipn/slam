import mujoco
import mujoco.viewer
import numpy as np
import time


# Load the MuJoCo model
model = mujoco.MjModel.from_xml_path("C:\\Users\\myrsi\\slam\\robotis_mujoco_menagerie\\robotis_tb3\\scene_turtlebot3_waffle_pi.xml")
data = mujoco.MjData(model)
for i in range(model.nu):
    name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, i)
    print(i, name)


print("Model loaded successfully")
print("Number of actuators:", model.nu)
print("Number of sensors:", model.nsensordata)


# Simple differential drive controller
def drive_robot(v, w):
    """
    v = linear velocity
    w = angular velocity
    """
    left_wheel = v - w
    right_wheel = v + w

    data.ctrl[0] = left_wheel
    data.ctrl[1] = right_wheel


# Launch the viewer
with mujoco.viewer.launch_passive(model, data) as viewer:

    start_time = time.time()

    while viewer.is_running():

        sim_time = data.time

        # Example control behaviour
        if sim_time < 5:
            # move forward
            drive_robot(2.0, 0.0)

        elif sim_time < 8:
            # turn
            drive_robot(0.0, 1.5)

        else:
            # forward again
            drive_robot(2.0, 0.0)

        # Get robot pose
        x = data.qpos[0]
        y = data.qpos[1]
        theta = data.qpos[2]

        print(f"Robot pose -> x:{x:.2f} y:{y:.2f} theta:{theta:.2f}")

        # Step simulation
        mujoco.mj_step(model, data)

        # Sync viewer
        viewer.sync()