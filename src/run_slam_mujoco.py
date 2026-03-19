import mujoco
import mujoco.viewer
import numpy as np
import time
import matplotlib.pyplot as plt
import os
from ekf_slam_core import EKFSLAM, wrap_angle, spiral_schedule

MODEL_PATH = "path/to/your/model.xml"
def random_landmarks(n=8, x_range=(-5, 5), y_range=(0, 10), seed=None):
    rng = np.random.default_rng(seed)
    xs  = rng.uniform(x_range[0], x_range[1], n)
    ys  = rng.uniform(y_range[0], y_range[1], n)
    return [(float(xs[i]), float(ys[i])) for i in range(n)]
def generate_landmark_xml(landmarks):
    xml = ""
    for i, (x, y) in enumerate(landmarks):
        xml += f"""
        <body name="landmark_{i}" pos="{x} {y} 0.3">
            <geom type="cylinder" size="0.15 0.3" rgba="1 0 0 1"/>
        </body>
        """
    return xml
with open(MODEL_PATH, "r") as f:
    xml_string = f.read()

landmarks = random_landmarks(n=15, x_range=(-4, 4), y_range=(0, 5), seed=42)

landmark_xml = generate_landmark_xml(landmarks)

xml_string = xml_string.replace(
    "</worldbody>",
    landmark_xml + "\n</worldbody>"
)
base_dir = "path\\to\\slam\\robotis_mujoco_menagerie\\robotis_tb3"

assets = {}

for root, _, files in os.walk(base_dir):
    for file in files:
        full_path = os.path.join(root, file)

        rel_path = os.path.relpath(full_path, base_dir)
        rel_path = rel_path.replace("\\", "/")

        with open(full_path, "rb") as f:
            assets[rel_path] = f.read()
model = mujoco.MjModel.from_xml_string(xml_string, assets=assets)
data  = mujoco.MjData(model)


landmark_ids = []
for i in range(model.nbody):
    name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, i)
    if name and "landmark" in name:
        landmark_ids.append(i)

N_LANDMARKS = len(landmark_ids)
print(f"Found {N_LANDMARKS} landmarks")


R_motion = np.diag([0.003**2, 0.003**2, np.deg2rad(0.1)**2])
Q_meas   = np.diag([0.01**2,  np.deg2rad(0.25)**2])


slam = EKFSLAM(N_LANDMARKS)


LIDAR_RANGE = 3.0
LIDAR_FOV   = 2*np.pi

def get_robot_pose():
    x = data.qpos[0]
    y = data.qpos[1]

    qw, qx, qy, qz = data.qpos[3:7]

    theta = np.arctan2(
        2 * (qw*qz + qx*qy),
        1 - 2 * (qy*qy + qz*qz)
    )

    return x, y, theta


def set_robot_pose(x, y, theta):
    data.qpos[0] = x
    data.qpos[1] = y

    qw = np.cos(theta / 2)
    qz = np.sin(theta / 2)

    data.qpos[3] = qw
    data.qpos[4] = 0
    data.qpos[5] = 0
    data.qpos[6] = qz

def move_robot(v, omega, dt):

    x, y, theta = get_robot_pose()

    if abs(omega) < 1e-5:
        omega = 1e-5

    # clean motion
    x += -v/omega*np.sin(theta) + v/omega*np.sin(theta + omega*dt)
    y +=  v/omega*np.cos(theta) - v/omega*np.cos(theta + omega*dt)
    theta = wrap_angle(theta + omega*dt)

    # add noise
    noise = np.random.multivariate_normal(np.zeros(3), R_motion)
    x += noise[0]
    y += noise[1]
    theta = wrap_angle(theta + noise[2])

    set_robot_pose(x, y, theta)

    return x, y, theta


def move_ground_truth(gt, v, omega, dt):

    x, y, theta = gt

    if abs(omega) < 1e-5:
        omega = 1e-5

    x += -v/omega*np.sin(theta) + v/omega*np.sin(theta + omega*dt)
    y +=  v/omega*np.cos(theta) - v/omega*np.cos(theta + omega*dt)
    theta = wrap_angle(theta + omega*dt)

    return np.array([x, y, theta])


def observe_landmarks():

    x, y, theta = get_robot_pose()
    measurements = []

    for i, lm_id in enumerate(landmark_ids):

        pos = data.xpos[lm_id]
        lx, ly = pos[0], pos[1]

        dx = lx - x
        dy = ly - y

        r = np.hypot(dx, dy)
        if r > LIDAR_RANGE:
            continue

        phi = wrap_angle(np.arctan2(dy, dx) - theta)
        if abs(phi) > LIDAR_FOV / 2:
            continue

        # add measurement noise
        noise = np.random.multivariate_normal(np.zeros(2), Q_meas)
        r += noise[0]
        phi = wrap_angle(phi + noise[1])

        measurements.append((i, r, phi))

    return measurements

dt    = model.opt.timestep
omega = 0.2

v_schedule, total_frames = spiral_schedule(
    v_start=0.5,
    omega=omega,
    dt=dt,
    n_laps=6,
    v_decay=0.8,
    v_min=0.2
)


mu_hist    = []
sigma_hist = []
gt_hist    = []
noisy_hist = []
gt_pose = np.array([0.0, 0.0, 0.0])

with mujoco.viewer.launch_passive(model, data) as viewer:

    for t in range(total_frames):

        if not viewer.is_running():
            break

        v = v_schedule[t]

        # 1. noisy robot
        x, y, theta = move_robot(v, omega, dt)

        mujoco.mj_forward(model, data)
        noisy_hist.append([x, y, theta])
        # 2. ground truth
        gt_pose = move_ground_truth(gt_pose, v, omega, dt)

        # 3. EKF predict 
        slam.predict(v, omega, dt)

        # 4. measurements
        measurements = observe_landmarks()

        # 5. EKF update
        for lm_id, r, b in measurements:
            slam.update(lm_id, (r, b))

        # 6. store
        mu_hist.append(slam.mu.copy())
        sigma_hist.append(np.diag(slam.Sigma))
        gt_hist.append(gt_pose.copy())

        if t % 20 == 0:
            est = slam.mu[:2]
            err = np.linalg.norm(est - gt_pose[:2])
            print(f"[{t}] error = {err:.3f}")

        viewer.sync()
        time.sleep(dt)

mu = np.array(mu_hist)
gt = np.array(gt_hist)
noisy = np.array(noisy_hist)
sigma = np.array(sigma_hist)
t_axis = np.arange(len(mu)) * dt


plt.figure(figsize=(8, 8))
plt.plot(gt[:,0], gt[:,1], 'g', label="GT")
plt.plot(mu[:,0], mu[:,1], 'r--', label="EKF")
plt.plot(noisy[:,0], noisy[:,1], 'b', label="Noisy Robot")

# landmarks
for i, lm_id in enumerate(landmark_ids):
    pos = data.xpos[lm_id]
    plt.scatter(pos[0], pos[1], c='black', marker='s')

# estimated landmarks
lx = [slam.mu[3+2*i] for i in range(N_LANDMARKS) if slam.initialized[i]]
ly = [slam.mu[3+2*i+1] for i in range(N_LANDMARKS) if slam.initialized[i]]
plt.scatter(lx, ly, c='red', marker='x')

plt.legend()
plt.title("SLAM Map")
plt.axis('equal')


# states
plt.figure()
plt.subplot(3,1,1)
plt.plot(t_axis, mu[:,0], label="EKF")
plt.plot(t_axis, gt[:,0], '--', label="GT")
plt.legend()

plt.subplot(3,1,2)
plt.plot(t_axis, mu[:,1])
plt.plot(t_axis, gt[:,1], '--')

plt.subplot(3,1,3)
plt.plot(t_axis, mu[:,2])
plt.plot(t_axis, gt[:,2], '--')

plt.show()
