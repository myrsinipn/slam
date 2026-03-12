import matplotlib
matplotlib.use("TkAgg")

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
class EKFSLAM:

    def __init__(self, num_landmarks):

        self.N = num_landmarks
        self.state_size = 3 + 2*self.N

        self.mu = np.zeros(self.state_size)
        self.Sigma = np.eye(self.state_size) * 0.5

        self.R = np.diag([0.1, 0.1, 0.05])
        self.Q = np.diag([0.05, 0.02])

        self.initialized = [False]*self.N


    def predict(self, v, omega, dt):

        x, y, theta = self.mu[0:3]

        if abs(omega) < 1e-5:
            omega = 1e-5

        motion = np.array([
            -v/omega*np.sin(theta) + v/omega*np.sin(theta + omega*dt),
            v/omega*np.cos(theta) - v/omega*np.cos(theta + omega*dt),
            omega*dt
        ])

        # Fx matrix 
        Fx = np.zeros((3, self.state_size))
        Fx[:,0:3] = np.eye(3)

        # Mean prediction
        self.mu = self.mu + Fx.T @ motion

        # Jacobian g_t 
        g = np.array([
            [0, 0, -v/omega*np.cos(theta) + v/omega*np.cos(theta+omega*dt)],
            [0, 0, -v/omega*np.sin(theta) + v/omega*np.sin(theta+omega*dt)],
            [0, 0, 0]
        ])

        # Gt from slide
        G = np.eye(self.state_size) + Fx.T @ g @ Fx

        # Covariance prediction
        self.Sigma = G @ self.Sigma @ G.T + Fx.T @ self.R @ Fx


    def update(self, landmark_id, measurement):

        r, phi = measurement
        x, y, theta = self.mu[0:3]

        lm_index = 3 + 2*landmark_id

        # Landmark initialization
        if not self.initialized[landmark_id]:

            self.mu[lm_index] = x + r*np.cos(phi + theta)
            self.mu[lm_index+1] = y + r*np.sin(phi + theta)

            self.initialized[landmark_id] = True
            return

        lx = self.mu[lm_index]
        ly = self.mu[lm_index+1]

        dx = lx - x
        dy = ly - y

        delta = np.array([dx, dy])
        q = delta.T @ delta
        sqrt_q = np.sqrt(q)

        # Predicted measurement
        z_hat = np.array([
            sqrt_q,
            np.arctan2(dy,dx) - theta
        ])

        z = np.array([r, phi])

        # Fxj matrix (from slide)
        Fxj = np.zeros((5, self.state_size))
        Fxj[0:3,0:3] = np.eye(3)
        Fxj[3, lm_index] = 1
        Fxj[4, lm_index+1] = 1

        # Measurement Jacobian
        H_low = (1/q) * np.array([
            [-sqrt_q*dx, -sqrt_q*dy, 0, sqrt_q*dx, sqrt_q*dy],
            [dy, -dx, -q, -dy, dx]
        ])

        H = H_low @ Fxj

        # Kalman gain
        S = H @ self.Sigma @ H.T + self.Q
        K = self.Sigma @ H.T @ np.linalg.inv(S)

        innovation = z - z_hat
        innovation[1] = np.arctan2(np.sin(innovation[1]), np.cos(innovation[1]))

        # State update
        self.mu = self.mu + K @ innovation

        # Covariance update
        I = np.eye(self.state_size)
        self.Sigma = (I - K @ H) @ self.Sigma
class Robot:
    def __init__(self, x=0.0, y=0.0, theta=0.0,
             sigma_v=0.5, sigma_omega=0.1,
             lidar_range=2.0,
             lidar_fov=np.pi/2,
             lidar_beams=40):

        self.x = x
        self.y = y
        self.theta = theta

        self.sigma_v = sigma_v
        self.sigma_omega = sigma_omega

        # LiDAR parameters
        self.lidar_range = lidar_range
        self.lidar_fov = lidar_fov
        self.lidar_beams = lidar_beams

        self.history_x = [x]
        self.history_y = [y]
    def move(self, v, omega, dt):
        
        # Add Gaussian noise
        v_noisy = v + np.random.normal(0, self.sigma_v)
        omega_noisy = omega + np.random.normal(0, self.sigma_omega)

        self.x += -v_noisy * np.sin(self.theta)/omega_noisy+v_noisy*np.sin(self.theta+omega_noisy*dt)/omega_noisy
        self.y += v_noisy * np.cos(self.theta)/omega_noisy-v_noisy*np.cos(self.theta+omega_noisy*dt)/omega_noisy
        self.theta += omega_noisy * dt
        self.theta = np.arctan2(np.sin(self.theta), np.cos(self.theta))

        self.history_x.append(self.x)
        self.history_y.append(self.y)

    def get_pose(self):
        return self.x, self.y, self.theta
    def lidar_scan(self, landmark_map):

        scan_lines = []
        measurements = []

        x, y, theta = self.get_pose()

        angles = np.linspace(-self.lidar_fov/2, self.lidar_fov/2, self.lidar_beams)

        sigma_range = 0.05
        sigma_bearing = 0.02

        for angle in angles:

            beam_angle = theta + angle

            min_dist = self.lidar_range
            detected_id = None
            detected_bearing = None

            for i,(lx,ly,ltheta) in enumerate(landmark_map.get_landmarks()):

                dx = lx - x
                dy = ly - y

                dist = np.sqrt(dx**2 + dy**2)

                angle_to_lm = np.arctan2(dy,dx)

                angle_diff = np.arctan2(
                    np.sin(angle_to_lm - beam_angle),
                    np.cos(angle_to_lm - beam_angle)
                )

                beam_width = self.lidar_fov / self.lidar_beams

                if abs(angle_diff) < beam_width and dist < min_dist:

                    min_dist = dist
                    detected_id = i
                    detected_bearing = angle_to_lm - theta

            # Add noise
            noisy_dist = max(0, min_dist + np.random.normal(0, sigma_range))

            end_x = x + noisy_dist*np.cos(beam_angle)
            end_y = y + noisy_dist*np.sin(beam_angle)

            scan_lines.append((end_x,end_y))

            # If a landmark was hit, create EKF measurement
            if detected_id is not None:

                r = noisy_dist

                bearing = detected_bearing + np.random.normal(0,sigma_bearing)
                bearing = np.arctan2(np.sin(bearing), np.cos(bearing))

                measurements.append((detected_id,r,bearing))

        return scan_lines, measurements

class LandmarkMap:
    def __init__(self, landmarks):
        
        self.landmarks = landmarks

    def get_landmarks(self):
        return self.landmarks
class Simulator:

    def __init__(self, robot, landmark_map, v, omega, dt=0.05, sim_time=20):

        self.robot = robot
        self.gt_robot = Robot(sigma_v=0.0, sigma_omega=0.0)  # ground truth
        self.landmark_map = landmark_map
        self.slam = EKFSLAM(len(self.landmark_map.get_landmarks()))
        self.v = v
        self.omega = omega
        self.dt = dt
        self.sim_time = sim_time
        self.frames = int(sim_time / dt)

        self.fig, self.ax = plt.subplots()

        self.ax.set_aspect('equal')
        self.ax.set_xlim(-10, 10)
        self.ax.set_ylim(-10, 10)
        self.ax.set_title("2D EKF SLAM Simulation")

        landmarks = np.array(self.landmark_map.get_landmarks())
        self.ax.scatter(landmarks[:,0], landmarks[:,1], marker='*', s=150, label="Landmarks")

        self.trajectory_line, = self.ax.plot([], [], label="Trajectory")

        self.robot_point, = self.ax.plot([], [], 'o', label="Robot")
                
        self.est_traj_x = []
        self.est_traj_y = []

        self.est_line, = self.ax.plot([], [], 'r--', label="Estimated Trajectory")

        self.est_robot, = self.ax.plot([], [], 'ro')

        self.est_landmarks = self.ax.scatter([], [], color='green', marker='x')
        
        self.mu_history = []
        self.sigma_history = []
        self.gt_history = []
        self.lidar_lines = []
        for _ in range(self.robot.lidar_beams):
            line, = self.ax.plot([], [], linewidth=0.5, color="orange")
            self.lidar_lines.append(line)
        self.gt_line, = self.ax.plot([], [], 'g', label="Ground Truth")
        self.ax.legend()


    def update(self, frame):

        self.robot.move(self.v, self.omega, self.dt)

        self.slam.predict(self.v, self.omega, self.dt)
        self.gt_robot.move(self.v, self.omega, self.dt)
        gt_x, gt_y, gt_theta = self.gt_robot.get_pose()
        self.gt_history.append([gt_x, gt_y, gt_theta])
        self.gt_line.set_data(
    self.gt_robot.history_x,
    self.gt_robot.history_y
)
        scan,measurements = self.robot.lidar_scan(self.landmark_map)

        for lm_id, r, b in measurements:
            self.slam.update(lm_id, (r, b))

        x, y, theta = self.robot.get_pose()

        est_x, est_y, est_theta = self.slam.mu[0:3]

        self.est_traj_x.append(est_x)
        self.est_traj_y.append(est_y)

        self.mu_history.append(self.slam.mu.flatten())
        self.sigma_history.append(np.diag(self.slam.Sigma))
        self.trajectory_line.set_data(
            self.robot.history_x,
            self.robot.history_y
        )
        self.robot_point.set_data([x], [y])


        self.est_line.set_data(self.est_traj_x, self.est_traj_y)

        self.est_robot.set_data([est_x], [est_y])

        lx_list = []
        ly_list = []

        for i in range(self.slam.N):

            if self.slam.initialized[i]:

                lx_list.append(self.slam.mu[3 + 2*i])
                ly_list.append(self.slam.mu[3 + 2*i + 1])

        if len(lx_list) > 0:
            self.est_landmarks.set_offsets(np.c_[lx_list, ly_list])
        else:
            self.est_landmarks.set_offsets(np.empty((0,2)))
        x, y, _ = self.robot.get_pose()

        for line,(sx,sy) in zip(self.lidar_lines,scan):
            line.set_data([x,sx],[y,sy])
        true_x, true_y, _ = self.gt_robot.get_pose()

        error = np.sqrt((true_x-est_x)**2 + (true_y-est_y)**2)

        print("Pose error:", error)

        return (
            self.trajectory_line,self.gt_line,
            self.robot_point,
            self.est_line,
            self.est_robot,
            self.est_landmarks,
            *self.lidar_lines
        )
    def plot_state_history(self):

        mu = np.array(self.mu_history)
        sigma = np.array(self.sigma_history)

        time = np.arange(len(mu)) * self.dt

        gt = np.array(self.gt_history)

        time = np.arange(len(mu)) * self.dt

        plt.figure()

        plt.subplot(3,1,1)
        plt.plot(time, mu[:,0], label="EKF")
        plt.plot(time, gt[:,0], '--', label="Ground Truth")
        plt.ylabel("x")
        plt.legend()

        plt.subplot(3,1,2)
        plt.plot(time, mu[:,1], label="EKF")
        plt.plot(time, gt[:,1], '--', label="Ground Truth")
        plt.ylabel("y")
        plt.legend()

        plt.subplot(3,1,3)
        plt.plot(time, mu[:,2], label="EKF")
        plt.plot(time, gt[:,2], '--', label="Ground Truth")
        plt.ylabel("theta")
        plt.xlabel("time (s)")
        plt.legend()

        plt.suptitle("EKF vs Ground Truth State")

        #Σ plot
        plt.figure()

        plt.subplot(3,1,1)
        plt.plot(time, sigma[:,0])
        plt.ylabel("σ_x²")

        plt.subplot(3,1,2)
        plt.plot(time, sigma[:,1])
        plt.ylabel("σ_y²")

        plt.subplot(3,1,3)
        plt.plot(time, sigma[:,2])
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
    robot = Robot(
        x=0.0,
        y=0.0,
        theta=0.0,
        sigma_v=0.5,
        sigma_omega=0.1
    )

    landmarks = [
        (2, 1,np.pi/2),
        (-2, -1,0),
        (0,7.5,np.pi/4),
    ]

    landmark_map = LandmarkMap(landmarks)

    simulator = Simulator(
        robot=robot,
        landmark_map=landmark_map,
        v=2.0,
        omega=0.5,
        dt=0.05,
        sim_time=20
    )

    simulator.run()