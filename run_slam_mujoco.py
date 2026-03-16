import mujoco
import mujoco.viewer
import numpy as np
import time
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from ekf_slam import EKFSLAM

model = mujoco.MjModel.from_xml_path(
    "C:\\Users\\myrsi\\slam\\robotis_mujoco_menagerie\\robotis_tb3\\scene_turtlebot3_waffle_pi.xml"
)
data = mujoco.MjData(model)

robot_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "base_footprint")

landmark_ids = []
for i in range(model.nbody):
    name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, i)
    if name and "landmark" in name:
        landmark_ids.append(i)

slam = EKFSLAM(len(landmark_ids))

def get_robot_pose():
    x = data.qpos[0]
    y = data.qpos[1]
    qw, qx, qy, qz = data.qpos[3:7]
    theta = np.arctan2(
        2 * (qw * qz + qx * qy),
        1 - 2 * (qy**2 + qz**2)
    )
    return x, y, theta


gt_pose = [0.0, 0.0, 0.0]

def move_robot(v, omega, dt, sigma_v=0.0, sigma_omega=0.0):
    """
    Integrate unicycle motion and write result to data.qpos.
    sigma_v / sigma_omega = 0  -> noiseless (use for ground truth)
    sigma_v / sigma_omega > 0  -> noisy     (use for the actual robot)
    """
    v_cmd     = v     + np.random.normal(0, sigma_v)     if sigma_v     > 0 else v
    omega_cmd = omega + np.random.normal(0, sigma_omega) if sigma_omega > 0 else omega
    if abs(omega_cmd) < 1e-5:
        omega_cmd = 1e-5

    x, y, theta = get_robot_pose()
    dx     = -v_cmd/omega_cmd * np.sin(theta) + v_cmd/omega_cmd * np.sin(theta + omega_cmd*dt)
    dy     =  v_cmd/omega_cmd * np.cos(theta) - v_cmd/omega_cmd * np.cos(theta + omega_cmd*dt)
    dtheta =  omega_cmd * dt
    x     += dx
    y     += dy
    theta += dtheta
    theta  = np.arctan2(np.sin(theta), np.cos(theta))

    data.qpos[0] = x
    data.qpos[1] = y
    qw = np.cos(theta / 2)
    qz = np.sin(theta / 2)
    data.qpos[3] = qw
    data.qpos[4] = 0
    data.qpos[5] = 0
    data.qpos[6] = qz
    return x, y, theta


LIDAR_RANGE = 1
LIDAR_FOV   = np.pi/2

def observe_landmarks():
    """
    Only return a measurement when a landmark is within LiDAR range AND FOV.
    The EKF update is only triggered on actual detections, not every landmark
    every step -- matching how a real LiDAR works.
    """
    x, y, theta = get_robot_pose()
    measurements = []
    for i, lm_id in enumerate(landmark_ids):
        pos = data.xpos[lm_id]
        lx  = pos[0]
        ly  = pos[1]
        dx  = lx - x
        dy  = ly - y
        r   = np.sqrt(dx**2 + dy**2)

        if r > LIDAR_RANGE:
            continue

        phi = np.arctan2(dy, dx) - theta
        phi = np.arctan2(np.sin(phi), np.cos(phi))

        if abs(phi) > LIDAR_FOV / 2:
            continue

        r   += np.random.normal(0, 0.05)
        phi += np.random.normal(0, 0.02)
        phi  = np.arctan2(np.sin(phi), np.cos(phi))
        measurements.append((i, r, phi))

    return measurements


mu_history    = []
sigma_history = []
gt_history    = []
z_history     = []

dt = model.opt.timestep

with mujoco.viewer.launch_passive(model, data) as viewer:
    step_counter = 0
    while viewer.is_running():
        v     = 0.5
        omega = 0.5

        move_robot(v, omega, dt, sigma_v=0.05, sigma_omega=0.02)
        mujoco.mj_forward(model, data)

        if abs(omega) < 1e-5:
            omega_gt = 1e-5
        else:
            omega_gt = omega
        gx, gy, gtheta = gt_pose
        gt_pose[0] += -v/omega_gt * np.sin(gtheta) + v/omega_gt * np.sin(gtheta + omega_gt*dt)
        gt_pose[1] +=  v/omega_gt * np.cos(gtheta) - v/omega_gt * np.cos(gtheta + omega_gt*dt)
        gt_pose[2]  = np.arctan2(np.sin(gt_pose[2] + omega_gt*dt), np.cos(gt_pose[2] + omega_gt*dt))

        slam.predict(v, omega, dt)

        measurements = observe_landmarks()
        z_history.append(list(measurements))
        for lm_id, r, b in measurements:
            slam.update(lm_id, (r, b))

        est_x, est_y, est_theta = slam.mu[0:3]
        gt_x, gt_y, gt_theta    = gt_pose
        error = np.sqrt((gt_x - est_x)**2 + (gt_y - est_y)**2)

        mu_history.append(slam.mu.copy())
        sigma_history.append(np.diag(slam.Sigma).copy())
        gt_history.append(gt_pose.copy())

        step_counter += 1
        if step_counter % 20 == 0:
            print(
                "GT:", round(gt_x, 2), round(gt_y, 2),
                "| EKF:", round(est_x, 2), round(est_y, 2),
                "| error:", round(error, 3)
            )

        viewer.sync()
        time.sleep(dt)


mu     = np.array(mu_history)
sigma  = np.array(sigma_history)
gt     = np.array(gt_history)
t_axis = np.arange(len(mu)) * dt

N = len(landmark_ids)

fig, ax = plt.subplots(figsize=(8, 8))
ax.set_aspect('equal')
ax.set_title("EKF-SLAM | TurtleBot3 Waffle Pi")
ax.plot(gt[:, 0], gt[:, 1], 'g-',  lw=1.5, label="Ground truth")
ax.plot(mu[:, 0], mu[:, 1], 'r--', lw=1.5, label="EKF estimate")

for i, lm_id in enumerate(landmark_ids):
    pos = data.xpos[lm_id]
    ax.scatter(pos[0], pos[1], marker='s', s=120, color='black', zorder=5,
               label="True landmark" if i == 0 else "")

lm_ex = [slam.mu[3+2*i]   for i in range(N) if slam.initialized[i]]
lm_ey = [slam.mu[3+2*i+1] for i in range(N) if slam.initialized[i]]
if lm_ex:
    ax.scatter(lm_ex, lm_ey, marker='x', s=100, color='red', zorder=5,
               label="EKF landmarks")

ax.legend()
ax.grid(True, alpha=0.3)

meas_x, meas_y = [], []
for step_idx, step_meas in enumerate(z_history):
    if not step_meas:
        continue
    rx, ry, rtheta = gt_history[step_idx]
    for (lm_id, r, phi) in step_meas:
        abs_phi = phi + rtheta
        meas_x.append(rx + r * np.cos(abs_phi))
        meas_y.append(ry + r * np.sin(abs_phi))

if meas_x:
    ax.scatter(meas_x, meas_y, s=4, alpha=0.25, color='orange',
               zorder=3, label="Noisy measurements")
    ax.legend()

fig2, axes = plt.subplots(3, 1, figsize=(10, 7), sharex=True)
fig2.suptitle("EKF vs Ground Truth -- Pose States")
for i, lbl in enumerate(['x (m)', 'y (m)', 'theta (rad)']):
    axes[i].plot(t_axis, mu[:, i], label="EKF")
    axes[i].plot(t_axis, gt[:, i], '--', label="Ground Truth")
    axes[i].set_ylabel(lbl)
    axes[i].legend(fontsize=8)
    axes[i].grid(True, alpha=0.3)
axes[-1].set_xlabel("time (s)")

fig3, axes3 = plt.subplots(3, 1, figsize=(10, 7), sharex=True)
fig3.suptitle("EKF Pose Covariance Diagonal")
for i, lbl in enumerate(['sigma2_x', 'sigma2_y', 'sigma2_theta']):
    axes3[i].plot(t_axis, sigma[:, i])
    axes3[i].set_ylabel(lbl)
    axes3[i].grid(True, alpha=0.3)
axes3[-1].set_xlabel("time (s)")

plt.tight_layout()
plt.show()