import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

from flexible_dyn import x_next_lambda as dyn_lambda

def animate(x0, u, frames=100):

    # Initialize plot
    fig, ax = plt.subplots()
    ax.set_xlim(0, 5)
    ax.set_ylim(-0.01, 0.01)
    # ax.set_aspect('equal')
    line, = ax.plot([], [], 'o-', lw=2)
    time_label = ax.text(0.1, 0.9, '', transform=ax.transAxes)

    timesteps = u.shape[1]
    frame_skip = int(timesteps//frames)

    # Simulate the system
    states = [x0]
    x = x0
    x_next = dyn_lambda()

    for t in range(timesteps):
        x = np.array(x_next(x, u[:,t])).flatten()
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
        y_coords = [0]+list(z[frame_skip*frame])+[0] # Positions of p1, p2, p3, p4 in y-direction
        line.set_data(x_coords, y_coords)
        time_label.set_text(f'frame: {frame} ')
        return line, time_label

    # Animate
    ani = FuncAnimation(fig, update, init_func=init, frames=frames, blit=True)
    plt.grid()
    plt.show()

# def state_animation(z, frames):

    # timesteps = z.shape[1]
    # frame_skip = int(timesteps/frames)
    
    # # Initialize plot
    # fig, ax = plt.subplots()
    # ax.set_xlim(0, 5)
    # ax.set_ylim(-0.01, 0.01)
    # # ax.set_aspect('equal')
    # line, = ax.plot([], [], 'o-', lw=2)
    # time_label = ax.text(0.1, 0.9, '', transform=ax.transAxes)

    # def init():
    #     """Initialize the plot."""

    #     line.set_data([], [])
    #     time_label.set_text('')
    #     return line, time_label

    # # Update function
    # def update(frame):
    #     """Update the plot for each frame."""

    #     x_coords = [0, 1, 2, 3, 4, 5]  # Positions of p1, p2, p3, p4 in x-direction
    #     y_coords = [0]+list(z[:,frame_skip*frame])+[0] # Positions of p1, p2, p3, p4 in y-direction
    #     line.set_data(x_coords, y_coords)
    #     time_label.set_text(f'Timestep: {frame} ')
    #     return line, time_label

    # # Animate
    # ani = FuncAnimation(fig, update, init_func=init, frames=timesteps, blit=True)
    # plt.grid()
    # plt.show()


def test():
    x0 = np.array([0.005, 0, 0, 0, 0, 0, 0, 0])
    timesteps = 10000
    frames = 100
    u1, u2 = 0, 0
    u = np.ones((2, timesteps))*np.array([u1, u2]).reshape(-1, 1)
    animate(x0, u, frames)
    # state_animation(x0, frames)

# test()