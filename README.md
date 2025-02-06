# README

## Project Title: Optimal Control of an Actuated Flexible Surface

## Overview

This project focuses on the optimal trajectory generation and control of an actuated flexible surface. Two control strategies—LQR (Linear Quadratic Regulator) and MPC (Model Predictive Control)—were implemented and compared. Based on the analysis, the LQR regulator demonstrated better performance in terms of computational efficiency and trajectory tracking accuracy.

## Files and Descriptions

### animation.py

This script generates an animation of the flexible surface executing the optimized trajectory.
Uses matplotlib.animation to visualize the movement of the surface points over time.
Can be tested by running the test() function inside the script.

### newton_optimal_controller.py

Implements Newton’s Optimal Control method for trajectory optimization.
Uses a regularized Newton-like algorithm with LQR-based affine feedback.
Computes the optimal state and control trajectories based on the given reference.
Can be executed to generate optimized trajectories for the flexible surface.
Dependencies
Ensure the following Python packages are installed before running the scripts:

numpy
matplotlib
control (for LQR calculations)
How to Run the Scripts
Run trajectory optimization

Import noc() from newton_optimal_controller.py and call it with appropriate reference trajectories.
python
Copy
Edit
from newton_optimal_controller import noc  
x_opt, u_opt, cost = noc(x_ref, u_ref, timesteps=100)
Run the animation

Import and call the animate() function from animation.py with an initial state and control input.
python
Copy
Edit
from animation import animate  
animate(x0, u, frames=100)
Results
The optimized trajectory minimizes cost while ensuring smooth transitions.
LQR proved to be computationally efficient and provided better real-time control compared to MPC.
