# EKF-SLAM Simulation Project

This repository implements a complete pipeline for Simultaneous Localization and Mapping (SLAM) using the Extended Kalman Filter (EKF).
---
## Overview 
**The project is structured progressively, starting from:**

- **A noisy robot motion model**
- **Linear state estimation (Kalman Filter)**
- **Nonlinear estimation (EKF)**
- **Full EKF-SLAM with LiDAR simulation**
- **Real-time visualization using MuJoCo**

**The goal is to provide both intuitive understanding and practical implementation of probabilistic robotics algorithms.**

---

## Project Structure
 ### simulation.py — Noisy Robot Motion Model

Implements a unicycle (velocity) motion model with Gaussian noise:

- Linear velocity noise:
  
𝑣
+
𝑁
(
0
,
𝜎
𝑣
)


- Angular velocity noise:
  
𝜔
+
𝑁
(
0
,
𝜎
𝜔
)


This forms the foundation for all later estimation algorithms.

---
### kalman.py — Kalman Filter (Double Integrator)

Implements a linear Kalman Filter on a double integrator system:
- State:
  
  -position & velocity
- Demonstrates:
  
	-Prediction vs update

	-Effect of process & measurement noise

	-Convergence behavior
  
  ---

### ekf.py — Extended Kalman Filter

Applies EKF to the nonlinear robot motion model:

- Uses Jacobian linearization

- Handles nonlinear state transitions

- Estimates robot pose under noise
 ---  
 ### ekf_slam.py — EKF-SLAM Core

**Core implementation of EKF-based SLAM:**

- State vector includes:

 - Robot pose 
(
𝑥
,
𝑦
,
𝜃
)

 - Landmark positions
 - Uses known landmark correspondences
 - Sensor model: Simulated LiDAR (range & bearing)


---
### slam.py — Full SLAM Simulation

Runs the full SLAM pipeline:

- Real-time simulation of:

  -Robot motion

  -LiDAR observations

  -EKF updates

- Tracks:

  -Estimated trajectory

  -Ground truth

 - Landmark map

---
### run_slam_mujoco.py — MuJoCo Visualization

Real-time visualization using MuJoCo:

- Displays robot motion and SLAM behavior

- Integrates with simulation output

---
### robotis_mujoco_menagerie/

Assets and models for MuJoCo simulation.

---
**SLAM Setup**

- Sensor: Simulated LiDAR

- Measurements: Range & Bearing

- Landmarks: Known correspondences

- Estimation Method: Extended Kalman Filter (EKF)

- Execution: Real-time simulation

---
**Example Output**

Here’s a typical EKF-SLAM result:

⭐ Landmarks

🔵 Ground truth trajectory

🔴 Estimated trajectory

🟠 Robot position

🟡 LiDAR rays
<img width="916" height="877" alt="image" src="https://github.com/user-attachments/assets/8f25297d-1ab8-4b26-b52a-88debab9a6d4" />

---
### Instalation
Clone the repository and install dependencies:

	git clone <your-repo-url>
	cd slam
	pip install -r requirements.txt
---
**How to Run**
1. **Basic Simulation**
   
	python simulation.py
2. **Kalman Filter Demo**

	python kalman.py
3. **EKF Localization**
  
	python ekf.py
4. **EKF-SLAM**
  
	python slam.py
5. **MuJoCo Visualization**

	python run_slam_mujoco.py
---
 **Outputs**

**The system generates:**

- **Robot trajectory (ground truth vs estimated)**

- **Landmark map**

- **Evolution of:**

  -State mean 
 𝜇

  -Covariance 
 Σ

---
 **Key Concepts Demonstrated**

- Probabilistic robotics

- Gaussian noise modeling

- Kalman Filter vs EKF

- Nonlinear state estimation

- EKF-SLAM formulation

- Sensor modeling (LiDAR)

- Real-time simulation

---
