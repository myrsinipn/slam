import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation



class KalmanFilter:

    def __init__(self, A, B, C, R, Q):

        self.A = A      
        self.B = B      
        self.C = C     

        self.R = R     
        self.Q = Q     

        self.m = np.zeros((A.shape[0], 1))    
        self.Sigma = 0.5 * np.eye(A.shape[0])         

    def step(self, u, z):

        m_bar = self.A @ self.m + self.B * u
        Sigma_bar = self.A @ self.Sigma @ self.A.T + self.R

     
        S = self.C @ Sigma_bar @ self.C.T + self.Q
        K = Sigma_bar @ self.C.T @ np.linalg.inv(S)

        self.m = m_bar + K @ (z - self.C @ m_bar)
        self.Sigma = (np.eye(self.A.shape[0]) - K @ self.C) @ Sigma_bar

        return self.m



# class KalmanFilter:

#     def __init__(self, A, B, C, R, Q, dt):

#         self.A = A
#         self.B = B
#         self.C = C
#         self.R = R      # Process noise
#         self.Q = Q      # Measurement noise
#         self.dt = dt

#         self.m = np.zeros((A.shape[0], 1))
#         self.Sigma = np.eye(A.shape[0])

#     def step(self, u, z):

#         m_dot = self.A @ self.m + self.B * u
#         m_bar = self.m + m_dot * self.dt

#         Sigma_dot = self.A @ self.Sigma + self.Sigma @ self.A.T + self.R
#         Sigma_bar = self.Sigma + Sigma_dot * self.dt

#         S = self.C @ Sigma_bar @ self.C.T + self.Q
#         K = Sigma_bar @ self.C.T @ np.linalg.inv(S)

#         self.m = m_bar + K @ (z - self.C @ m_bar)
#         self.Sigma = (np.eye(self.A.shape[0]) - K @ self.C) @ Sigma_bar

#         return self.m
class DoubleIntegrator1D:

    def __init__(self, x0, sigma_pos=0.0, sigma_vel=0.0):

        self.x = np.array(x0, dtype=float)

        self.A = np.array([[0, 1],
                           [0, 0]])

        self.B = np.array([[0],
                           [1]])

        self.Q = np.array([[sigma_pos**2, 0],
                           [0, sigma_vel**2]])

    def step(self, u, dt):

        x_dot = self.A @ self.x + self.B.flatten() * u
        self.x = self.x + x_dot * dt

        if np.any(self.Q):
            noise = np.random.multivariate_normal([0, 0], self.Q)
            self.x = self.x + noise

    def position(self):
        return self.x[0]


dt = 0.02
T = 10
steps = int(T / dt)


clean_system = DoubleIntegrator1D([0, 0])

noisy_system = DoubleIntegrator1D(
    [0, 0],
    sigma_pos=0.5,
    sigma_vel=0
)

A = np.array([[0, 1],
              [0, 0]])

B = np.array([[0],
              [1]])

C = np.array([[1, 0]])

R = np.array([[0.1, 0],
              [0, 0.1]])      # process noise covariance

Q = np.array([[0.5]])         # measurement noise covariance

kf = KalmanFilter(A, B, C, R, Q)

# fig, ax = plt.subplots()
# ax.set_xlim(-1, 60)
# ax.set_ylim(-2, 2)
# ax.set_yticks([])
# ax.set_title("Clean vs Noisy vs Kalman")

# clean_line, = ax.plot([], [], lw=2, label="Clean")
# noisy_line, = ax.plot([], [], lw=2, linestyle="--", label="Noisy")
# kf_line, = ax.plot([], [], lw=2, linestyle=":", label="Kalman")

# clean_point, = ax.plot([], [], "o")
# noisy_point, = ax.plot([], [], "o")
# kf_point, = ax.plot([], [], "o")

# ax.legend()


# def update(frame):

#     u = 1.0

#     clean_system.step(u, dt)
#     noisy_system.step(u, dt)

#     x_clean = clean_system.position()
#     x_noisy = noisy_system.position()

#     # Measurement = noisy position
#     z = np.array([[x_noisy]])

#     mu = kf.step(u, z)
#     x_kf = mu[0, 0]

#     # Lines
#     clean_line.set_data([0, x_clean], [0, 0])
#     noisy_line.set_data([0, x_noisy], [0, 0])
#     kf_line.set_data([0, x_kf], [0, 0])

#     # Points
#     clean_point.set_data([x_clean], [0])
#     noisy_point.set_data([x_noisy], [0])
#     kf_point.set_data([x_kf], [0])

#     return clean_line, noisy_line, kf_line, clean_point, noisy_point, kf_point


# ani = FuncAnimation(fig, update, frames=steps,
#                     interval=dt*1000, blit=True)

# plt.show()
fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(8, 8))

ax1.set_title("Clean System")
ax2.set_title("Noisy System")
ax3.set_title("Kalman Estimate")

for ax in [ax1, ax2, ax3]:
    ax.set_xlim(0, T)
    ax.set_ylim(-5, 60)
    ax.grid()

time_data = []
clean_data = []
noisy_data = []
kf_data = []


def update(frame):

    u = 1.0
    t = frame * dt

    clean_system.step(u, dt)
    noisy_system.step(u, dt)

    x_clean = clean_system.position()
    x_noisy = noisy_system.position()

    z = np.array([[x_noisy]])
    mu = kf.step(u, z)
    x_kf = mu[0, 0]

    time_data.append(t)
    clean_data.append(x_clean)
    noisy_data.append(x_noisy)
    kf_data.append(x_kf)

    ax1.cla()
    ax2.cla()
    ax3.cla()

    for ax in [ax1, ax2, ax3]:
        ax.set_xlim(0, T)
        ax.set_ylim(-5, 60)
        ax.grid()

    ax1.set_title("Clean System")
    ax2.set_title("Noisy System")
    ax3.set_title("Kalman Estimate")

    ax1.plot(time_data, clean_data)
    ax2.plot(time_data, noisy_data)
    ax3.plot(time_data, kf_data)

    return ax1, ax2, ax3


ani = FuncAnimation(fig, update, frames=steps,
                    interval=dt*1000)

plt.tight_layout()
plt.show()