import matplotlib
matplotlib.use("TkAgg")

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
class Robot:
    def __init__(self, x=0.0, y=0.0, theta=0.0,
             sigma_v=0.1, sigma_omega=0.05,
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

        self.x += v_noisy * np.cos(self.theta) * dt
        self.y += v_noisy * np.sin(self.theta) * dt
        self.theta += omega_noisy * dt
        self.theta = np.arctan2(np.sin(self.theta), np.cos(self.theta))

        self.history_x.append(self.x)
        self.history_y.append(self.y)

    def get_pose(self):
        return self.x, self.y, self.theta
    def lidar_scan(self, landmark_map):
        scan_points = []
        x, y, theta = self.get_pose()
        angles = np.linspace(-self.lidar_fov / 2, self.lidar_fov / 2, self.lidar_beams)

        sigma_range = 0.05
        sigma_bearing = 0.02 

        for angle in angles:
            beam_angle = theta + angle
            min_dist = self.lidar_range
            detected_landmark = None

            for lx, ly, ltheta in landmark_map.get_landmarks():
                dx, dy = lx - x, ly - y
                dist = np.sqrt(dx**2 + dy**2)
                
                angle_to_lm = np.arctan2(dy, dx)
                angle_diff = np.arctan2(np.sin(angle_to_lm - beam_angle), np.cos(angle_to_lm - beam_angle))

                if abs(angle_diff) < (self.lidar_fov / self.lidar_beams) and dist < min_dist:
                    min_dist = dist
                    detected_landmark = (lx, ly, ltheta)

            noisy_dist = max(0, min_dist + np.random.normal(0, sigma_range))
            
            end_x = x + noisy_dist * np.cos(beam_angle)
            end_y = y + noisy_dist * np.sin(beam_angle)
            scan_points.append((end_x, end_y))

            if detected_landmark:
                noisy_ltheta = detected_landmark[2] + np.random.normal(0, sigma_bearing)
                print(f"Hit LM at ({detected_landmark[0]}, {detected_landmark[1]}, Measured Noisy Pos: ({end_x:.2f}, {end_y:.2f} )"
                      f"True Ori: {detected_landmark[2]:.2f} "
                      f"Sensed Ori: {noisy_ltheta:.2f}")

        return scan_points
class LandmarkMap:
    def __init__(self, landmarks):
        
        self.landmarks = landmarks

    def get_landmarks(self):
        return self.landmarks

class Simulator:
    def __init__(self, robot, landmark_map,v, omega, dt=0.05, sim_time=20):
        self.robot = robot
        self.landmark_map=landmark_map
        self.v = v
        self.omega = omega
        self.dt = dt
        self.sim_time = sim_time
        self.frames = int(sim_time / dt)
        
        self.fig, self.ax = plt.subplots()
        self.ax.set_aspect('equal')
        self.ax.set_xlim(-10, 10)
        self.ax.set_ylim(-10, 10)
        self.ax.set_title("2D Robot Simulation")
        landmarks = np.array(self.landmark_map.get_landmarks())
        self.ax.scatter(landmarks[:,0], landmarks[:,1], marker='*', s=150, label="Landmarks")
        self.trajectory_line, = self.ax.plot([], [], label="Trajectory")
        self.robot_point, = self.ax.plot([], [], 'o', label="Robot")
        self.heading_line, = self.ax.plot([], [], label="Heading")
        self.lidar_lines = []
        for _ in range(robot.lidar_beams):
            line, = self.ax.plot([], [], linewidth=0.5)
            self.lidar_lines.append(line)
        self.ax.legend()

    def update(self, frame):
     
        self.robot.move(self.v, self.omega, self.dt)

        x, y, theta = self.robot.get_pose()

        self.trajectory_line.set_data(
            self.robot.history_x,
            self.robot.history_y
        )

        self.robot_point.set_data([x], [y])

        heading_length = 0.8
        hx = x + heading_length * np.cos(theta)
        hy = y + heading_length * np.sin(theta)
        self.heading_line.set_data([x, hx], [y, hy])
        scan = self.robot.lidar_scan(self.landmark_map)

        x, y, _ = self.robot.get_pose()

        for line, (sx, sy) in zip(self.lidar_lines, scan):
            line.set_data([x, sx], [y, sy])
        return (self.trajectory_line,
        self.robot_point,
        self.heading_line,
        *self.lidar_lines)
    def run(self):
        self.ani = FuncAnimation(
            self.fig,
            self.update,
            frames=self.frames,
            interval=self.dt * 1000,
            blit=True
        )
        
        plt.show()

if __name__ == "__main__":
    robot = Robot(
        x=0.0,
        y=0.0,
        theta=0.0,
        sigma_v=0.1,
        sigma_omega=0.05
    )

    landmarks = [
        (2, 1,np.pi/2),
        (-2, -1,0),
        (0, 0,np.pi/3)
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