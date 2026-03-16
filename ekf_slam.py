import numpy as np
class EKFSLAM:

    def __init__(self, num_landmarks):

        self.N = num_landmarks
        self.state_size = 3 + 2*self.N

        self.mu = np.zeros(self.state_size)
        self.Sigma = np.eye(self.state_size) * 0.5

        self.R = np.diag([0.08, 0.08, 0.01])
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