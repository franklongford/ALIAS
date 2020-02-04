import numpy as np

# matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.animation as anim

from .intrinsic_sampling_method import (
    check_uv, xi, check_pbc)


def view_surface(coeff, pivot, qm, qu, xmol, ymol, zmol, nxy, dim):

    zmol = check_pbc(xmol, ymol, zmol, pivot, dim)

    X = np.linspace(0, dim[0], nxy)
    Y = np.linspace(0, dim[1], nxy)

    vcheck = np.vectorize(check_uv)

    n_waves = 2 * qm + 1
    u_array = np.array(np.arange(n_waves**2) / n_waves, dtype=int) - qm
    v_array = np.array(np.arange(n_waves**2) % n_waves, dtype=int) - qm
    wave_check = (u_array >= -qu) * (u_array <= qu) * (v_array >= -qu) * (v_array <= qu)
    Delta = 1. / 4 * np.sum(coeff**2 * wave_check * vcheck(u_array, v_array))

    surface = np.zeros((2, nxy, nxy))

    for i, x in enumerate(X):
        for j in range(2):
            surface[j][i] += xi(np.ones(nxy) * x, Y, coeff[j], qm, qu, dim)

    surface = np.moveaxis(surface, 1, 2)

    fig = plt.figure(0, figsize=(15,15))
    ax = fig.gca(projection='3d')
    ax.set_xlabel(r'$x$ (\AA)')
    ax.set_ylabel(r'$y$ (\AA)')
    ax.set_zlabel(r'$z$ (\AA)')
    ax.set_xlim3d(0, dim[0])
    ax.set_ylim3d(0, dim[1])
    ax.set_zlim3d(-dim[2]/2, dim[2]/2)
    X_grid, Y_grid = np.meshgrid(X, Y)

    def update(frame):
        ax.clear()
        ax.plot_wireframe(X_grid, Y_grid, surface[0], color='r')
        ax.scatter(xmol[pivot[0]], ymol[pivot[0]], zmol[pivot[0]], color='b')
        ax.plot_wireframe(X_grid, Y_grid, surface[1], color='r')
        ax.scatter(xmol[pivot[1]], ymol[pivot[1]], zmol[pivot[1]], color='g')

    a = anim.FuncAnimation(fig, update, frames=1, repeat=False)
    #plt.savefig('plot_{}_{}.png'.format(len(pivot[0]), len(pivot[1])))
    plt.show()
