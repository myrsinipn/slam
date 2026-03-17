import numpy as np

def wrap_angle(a):
    return (a + np.pi) % (2 * np.pi) - np.pi


class EKFSLAM:

    def __init__(self, num_landmarks):
        self.N          = num_landmarks
        self.state_size = 3 + 2 * self.N

        self.mu    = np.zeros(self.state_size)

        self.Sigma = np.zeros((self.state_size, self.state_size))
        self.Sigma[0, 0] = 0.5**2
        self.Sigma[1, 1] = 0.5**2
        self.Sigma[2, 2] = np.deg2rad(20)**2
        self.Sigma[3:, 3:] = 1e6 * np.eye(self.state_size - 3)

        self.R = np.diag([0.1, 0.1, 0.05])    # motion noise (added to state)
        self.Q = np.diag([0.1, 0.1])          # measurement noise

        self.initialized = [False] * self.N

    def predict(self, v, omega, dt):
        x, y, theta = self.mu[0:3]

        if abs(omega) < 1e-5:
            omega = 1e-5

        motion = np.array([
            -v/omega * np.sin(theta) + v/omega * np.sin(theta + omega*dt),
             v/omega * np.cos(theta) - v/omega * np.cos(theta + omega*dt),
             omega * dt
        ])

        Fx = np.zeros((3, self.state_size))
        Fx[:, 0:3] = np.eye(3)

        self.mu = self.mu + Fx.T @ motion
        self.mu[2] = wrap_angle(self.mu[2])

        g = np.array([
            [0, 0, -v/omega * np.cos(theta) + v/omega * np.cos(theta + omega*dt)],
            [0, 0, -v/omega * np.sin(theta) + v/omega * np.sin(theta + omega*dt)],
            [0, 0,  0]
        ])

        G          = np.eye(self.state_size) + Fx.T @ g @ Fx
        self.Sigma = G @ self.Sigma @ G.T + Fx.T @ self.R @ Fx

    def update(self, landmark_id, measurement):
        r, phi      = measurement
        x, y, theta = self.mu[0:3]
        lm_index    = 3 + 2 * landmark_id

        if not self.initialized[landmark_id]:
            self.mu[lm_index]     = x + r * np.cos(phi + theta)
            self.mu[lm_index + 1] = y + r * np.sin(phi + theta)
            self.initialized[landmark_id] = True
            return

        lx = self.mu[lm_index]
        ly = self.mu[lm_index + 1]

        dx     = lx - x
        dy     = ly - y
        q      = dx**2 + dy**2
        sqrt_q = np.sqrt(q)

        z_hat = np.array([sqrt_q, wrap_angle(np.arctan2(dy, dx) - theta)])
        z     = np.array([r, phi])

        Fxj = np.zeros((5, self.state_size))
        Fxj[0:3, 0:3]        = np.eye(3)
        Fxj[3, lm_index]     = 1
        Fxj[4, lm_index + 1] = 1

        H_low = (1/q) * np.array([
            [-sqrt_q*dx, -sqrt_q*dy,  0,  sqrt_q*dx,  sqrt_q*dy],
            [         dy,        -dx, -q,        -dy,         dx]
        ])
        H = H_low @ Fxj

        S          = H @ self.Sigma @ H.T + self.Q
        K          = self.Sigma @ H.T @ np.linalg.inv(S)

        innovation    = z - z_hat
        innovation[1] = wrap_angle(innovation[1])

        self.mu    = self.mu + K @ innovation
        self.mu[2] = wrap_angle(self.mu[2])
        self.Sigma = (np.eye(self.state_size) - K @ H) @ self.Sigma


def spiral_schedule(v_start, omega, dt, n_laps, v_decay=0.85, v_min=0.2):
   
    
    lap_period     = 2 * np.pi / omega          # seconds per lap
    frames_per_lap = int(round(lap_period / dt))

    v_schedule = []
    v = v_start
    for _ in range(n_laps):
        v_schedule.extend([v] * frames_per_lap)
        v = max(v * v_decay, v_min)

    return np.array(v_schedule), len(v_schedule)
