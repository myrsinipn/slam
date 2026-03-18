import matplotlib
matplotlib.use("TkAgg")

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

from ekf_slam import EKFSLAM, wrap_angle,spiral_schedule

class Robot:


    def __init__(self, x=0.0, y=0.0, theta=0.0,
                 R_motion=None, Q_meas=None,
                 lidar_range=2.0, lidar_fov=np.pi, lidar_beams=60):

        self.x     = x
        self.y     = y
        self.theta = theta

        # R : 3x3 motion noise covariance  
        self.R_motion = R_motion 
        # Q : 2x2 measurement noise covariance
        self.Q_meas = Q_meas

        self.lidar_range = lidar_range
        self.lidar_fov   = lidar_fov
        self.lidar_beams = lidar_beams

        self.history_x = [x]
        self.history_y = [y]

    def move(self, v, omega, dt):
        
        if abs(omega) < 1e-5:
            omega = 1e-5

        self.x     += -v/omega * np.sin(self.theta) + v/omega * np.sin(self.theta + omega*dt)
        self.y     +=  v/omega * np.cos(self.theta) - v/omega * np.cos(self.theta + omega*dt)
        self.theta += omega * dt

        noise = np.random.multivariate_normal(np.zeros(3), self.R_motion)
        self.x     += noise[0]
        self.y     += noise[1]
        self.theta += noise[2]
        self.theta  = wrap_angle(self.theta)

        self.history_x.append(self.x)
        self.history_y.append(self.y)

    def get_pose(self):
        return self.x, self.y, self.theta

    def lidar_scan(self, landmark_map):
       
        scan_lines   = []
        measurements = []

        x, y, theta = self.get_pose()
        half_fov = self.lidar_fov / 2.0
        angles   = np.linspace(-half_fov, half_fov, self.lidar_beams)

        for angle in angles:
            beam_angle       = theta + angle
            min_dist         = self.lidar_range
            detected_id      = None
            detected_bearing = None

            for i, (lx, ly, *_) in enumerate(landmark_map.get_landmarks()):
                dx   = lx - x
                dy   = ly - y
                dist = np.hypot(dx, dy)

                angle_to_lm = np.arctan2(dy, dx)
                angle_diff  = wrap_angle(angle_to_lm - beam_angle)
                beam_width  = self.lidar_fov / self.lidar_beams

                if abs(angle_diff) < beam_width and dist < min_dist:
                    min_dist         = dist
                    detected_id      = i
                    detected_bearing = wrap_angle(angle_to_lm - theta)

            end_x = x + min_dist * np.cos(beam_angle)
            end_y = y + min_dist * np.sin(beam_angle)
            scan_lines.append((end_x, end_y))

            if detected_id is not None:
                meas_noise = np.random.multivariate_normal(np.zeros(2), self.Q_meas)
                r_noisy = min_dist         + meas_noise[0]
                b_noisy = detected_bearing + meas_noise[1]
                b_noisy = wrap_angle(b_noisy)
                measurements.append((detected_id, r_noisy, b_noisy))

        return scan_lines, measurements
class GroundTruthRobot:

    def __init__(self, x=0.0, y=0.0, theta=0.0):
        self.x     = x
        self.y     = y
        self.theta = theta

        self.history_x = [x]
        self.history_y = [y]

    def move(self, v, omega, dt):
        if abs(omega) < 1e-5:
            omega = 1e-5

        self.x     += -v/omega * np.sin(self.theta) + v/omega * np.sin(self.theta + omega*dt)
        self.y     +=  v/omega * np.cos(self.theta) - v/omega * np.cos(self.theta + omega*dt)
        self.theta  = wrap_angle(self.theta + omega*dt)

        self.history_x.append(self.x)
        self.history_y.append(self.y)

    def get_pose(self):
        return self.x, self.y, self.theta

class LandmarkMap:

    def __init__(self, landmarks):
        self.landmarks = landmarks

    def get_landmarks(self):
        return self.landmarks

def random_landmarks(n=8, x_range=(-5, 5), y_range=(0, 10), seed=None):
    
    rng = np.random.default_rng(seed)
    xs  = rng.uniform(x_range[0], x_range[1], n)
    ys  = rng.uniform(y_range[0], y_range[1], n)
    return [(float(xs[i]), float(ys[i]), 0.0) for i in range(n)]


class Simulator:

    def __init__(self, robot, landmark_map, v, omega, dt=0.05, sim_time=20):

        self.robot        = robot                          # noisy robot (R + Q)
        self.gt_robot     = GroundTruthRobot(              # clean ground truth
            x=robot.x, y=robot.y, theta=robot.theta
        )
        self.landmark_map = landmark_map
        self.slam         = EKFSLAM(len(self.landmark_map.get_landmarks()))

        self.omega    = omega
        self.dt       = dt

        if hasattr(v, '__len__'):
            self.v_schedule = np.asarray(v)
        else:
            self.v_schedule = np.full(int(sim_time / dt), float(v))

        self.frames   = len(self.v_schedule)
        self.sim_time = self.frames * dt

        self.est_traj_x    = []
        self.est_traj_y    = []
        self.mu_history    = []
        self.sigma_history = []
        self.gt_history    = []

        self.fig, self.ax = plt.subplots()
        self.ax.set_aspect('equal')
        self.ax.set_xlim(-5, 5)
        self.ax.set_ylim(0, 10)
        self.ax.set_title("2D EKF SLAM Simulation")

        lm = np.array(self.landmark_map.get_landmarks())
        self.ax.scatter(lm[:, 0], lm[:, 1], marker='*', s=150, label="Landmarks")

        self.trajectory_line, = self.ax.plot([], [], label="Trajectory")
        self.robot_point,     = self.ax.plot([], [], 'o', label="Robot")

        self.est_line,  = self.ax.plot([], [], 'r--', label="Estimated Trajectory")
        self.est_robot, = self.ax.plot([], [], 'ro')

        self.est_landmarks = self.ax.scatter([], [], color='green', marker='x')

        self.gt_line, = self.ax.plot([], [], 'g', label="Ground Truth")

        self.lidar_lines = []
        for _ in range(self.robot.lidar_beams):
            line, = self.ax.plot([], [], linewidth=0.5, color="orange")
            self.lidar_lines.append(line)

        self.ax.legend()

    def update(self, frame):

        # current v from spiral schedule
        v = self.v_schedule[frame]

        # noisy robot moves (clean controls + R_motion on state)
        self.robot.move(v, self.omega, self.dt)

        # ground truth moves with clean controls, no noise
        self.gt_robot.move(v, self.omega, self.dt)

        # EKF predict with clean controls
        self.slam.predict(v, self.omega, self.dt)

        gt_x, gt_y, gt_theta = self.gt_robot.get_pose()
        self.gt_history.append([gt_x, gt_y, gt_theta])
        self.gt_line.set_data(self.gt_robot.history_x, self.gt_robot.history_y)

        # lidar scan — Q_meas noise on measurements only
        scan, measurements = self.robot.lidar_scan(self.landmark_map)

        # EKF update
        for lm_id, r, b in measurements:
            self.slam.update(lm_id, (r, b))

        x, y, theta     = self.robot.get_pose()
        est_x, est_y, _ = self.slam.mu[0:3]

        self.est_traj_x.append(est_x)
        self.est_traj_y.append(est_y)

        self.mu_history.append(self.slam.mu.flatten())
        self.sigma_history.append(np.diag(self.slam.Sigma))

        self.trajectory_line.set_data(self.robot.history_x, self.robot.history_y)
        self.robot_point.set_data([x], [y])

        self.est_line.set_data(self.est_traj_x, self.est_traj_y)
        self.est_robot.set_data([est_x], [est_y])

        lx_list, ly_list = [], []
        for i in range(self.slam.N):
            if self.slam.initialized[i]:
                lx_list.append(self.slam.mu[3 + 2*i])
                ly_list.append(self.slam.mu[3 + 2*i + 1])

        if lx_list:
            self.est_landmarks.set_offsets(np.c_[lx_list, ly_list])
        else:
            self.est_landmarks.set_offsets(np.empty((0, 2)))

        x, y, _ = self.robot.get_pose()
        for line, (sx, sy) in zip(self.lidar_lines, scan):
            line.set_data([x, sx], [y, sy])

        true_x, true_y, _ = self.gt_robot.get_pose()
        error = np.sqrt((true_x - est_x)**2 + (true_y - est_y)**2)
        print("Pose error:", error)

        return (
            self.trajectory_line, self.gt_line,
            self.robot_point,
            self.est_line,
            self.est_robot,
            self.est_landmarks,
            *self.lidar_lines
        )

    def plot_state_history(self):

        mu    = np.array(self.mu_history)
        sigma = np.array(self.sigma_history)
        gt    = np.array(self.gt_history)
        time  = np.arange(len(mu)) * self.dt

        plt.figure()
        plt.subplot(3, 1, 1)
        plt.plot(time, mu[:, 0], label="EKF")
        plt.plot(time, gt[:, 0], '--', label="Ground Truth")
        plt.ylabel("x")
        plt.legend()

        plt.subplot(3, 1, 2)
        plt.plot(time, mu[:, 1], label="EKF")
        plt.plot(time, gt[:, 1], '--', label="Ground Truth")
        plt.ylabel("y")
        plt.legend()

        plt.subplot(3, 1, 3)
        plt.plot(time, mu[:, 2], label="EKF")
        plt.plot(time, gt[:, 2], '--', label="Ground Truth")
        plt.ylabel("theta")
        plt.xlabel("time (s)")
        plt.legend()
        plt.suptitle("EKF vs Ground Truth State")

        plt.figure()
        plt.subplot(3, 1, 1)
        plt.plot(time, sigma[:, 0])
        plt.ylabel("σ_x²")

        plt.subplot(3, 1, 2)
        plt.plot(time, sigma[:, 1])
        plt.ylabel("σ_y²")

        plt.subplot(3, 1, 3)
        plt.plot(time, sigma[:, 2])
        plt.ylabel("σ_θ²")
        plt.xlabel("time (s)")
        plt.suptitle("EKF Covariance Σ")

        plt.show()

    def run(self):
        self.ani = FuncAnimation(
            self.fig,
            self.update,
            frames=self.frames,
            interval=self.dt * 1000,
            blit=True
        )
        plt.show()
        self.plot_state_history()


if __name__ == "__main__":

    R_motion = np.diag([0.003**2, 0.003**2, np.deg2rad(0.1)**2])
    Q_meas   = np.diag([0.01**2,  np.deg2rad(0.25)**2])

    robot = Robot(
        x=0.0, y=0.0, theta=0.0,
        R_motion=R_motion,
        Q_meas=Q_meas,
        lidar_range=2.0,
        lidar_fov=2*np.pi,
        lidar_beams=60
    )

    landmarks = random_landmarks(n=19, x_range=(-5, 5), y_range=(0, 10), seed=42)

    landmark_map = LandmarkMap(landmarks)

    dt    = 0.05
    omega = 0.5

    v_schedule, total_frames = spiral_schedule(
        v_start=2.0, omega=omega, dt=dt, n_laps=6, v_decay=0.80, v_min=0.2
    )

    simulator = Simulator(
        robot=robot,
        landmark_map=landmark_map,
        v=v_schedule, 
        omega=omega,
        dt=dt,
        sim_time=total_frames * dt
    )

    simulator.run()