import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from flexible_dyn import flexible_surface_dynamics as dynamics

def animate(timesteps=100):

    # Initialize plot
    fig, ax = plt.subplots()
    ax.set_xlim(0, 5)
    ax.set_ylim(-1, 1)
    ax.set_aspect('equal')
    line, = ax.plot([], [], 'o-', lw=2)
    time_label = ax.text(0.1, 0.9, '', transform=ax.transAxes)

    # Initial state and input
    x = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    u = np.array([2000, 2000])

    # Simulate the system
    states = [x]
    for _ in range(timesteps):
        x = dynamics(x, u)
        states.append(x)
    states = np.array(states)

    # Extract positions for animation
    z = states[:, :4]

    def init():
        """Initialize the plot."""

        line.set_data([], [])
        time_label.set_text('')
        return line, time_label

    # Update function
    def update(frame):
        """Update the plot for each frame."""

        x_coords = [0, 1, 2, 3, 4, 5]  # Positions of p1, p2, p3, p4 in x-direction
        y_coords = [0]+list(z[frame])+[0] # Positions of p1, p2, p3, p4 in y-direction
        line.set_data(x_coords, y_coords)
        time_label.set_text(f'Timestep: {frame} ')
        return line, time_label

    # Animate
    ani = FuncAnimation(fig, update, init_func=init, frames=timesteps, blit=True)
    plt.grid()
    plt.show()


animate(100)