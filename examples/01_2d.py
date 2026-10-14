import numpy as np
import matplotlib.pyplot as plt

from utils import *

if __name__ == "__main__":
    device = create_device()
    kernel, = load_kernels(device, "func_2d.slang", ("func_2d",))

    fname = 'f(x,y) = x^2 + y^2 + x*y - 0.5*(x*y)^2 - 0.25'
    logger.info(f"Generating grids for function {fname}...")

    levels = [-1,2,3,4,5,6]
    xmin, xmax = -2, 2
    ymin, ymax = -2, 2
    def f(x,y):
        return x*x + y*y + x*y - (x*y)**2/2 - 0.25

    fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(14,10))
    fig.suptitle(fname)

    x = np.linspace(xmin, xmax, 1024)[None,:]
    y = np.linspace(ymin, ymax, 1024)[:,None]
    z = f(x,y)

    for level, ax in zip(levels, axes.ravel()):

        ax.contour(z, levels=[0], zorder=1, colors=['k'],
                extent=[xmin, xmax, ymin, ymax], origin='lower')
        ax.set_xlim(xmin, xmax)
        ax.set_ylim(ymin, ymax)

        if level >= 0:
            ax.set_title(f'level={level}')
            N = 2**level
            logger.info(f"  computing level={level} with {N} intervals...")
            s = create_buffer(device, N*N, dtype=np.uint32)
            kernel.dispatch(spy.uint3(N, N, 1),
                    vars={"xmin": xmin, "xmax": xmax, "ymin": ymin, "ymax": ymax, "level": level, "sign": s})
            s = read_buffer(s, np.uint32).reshape(N, N)

            # check correctness: the sign of f at the corners of each cell must match the computed sign
            corners = f(np.linspace(xmin, xmax, N+1)[None,:], np.linspace(ymin, ymax, N+1)[:,None])
            for di in range(2):
                for dj in range(2):
                    zc = corners[di:di+N, dj:dj+N]
                    assert (zc[s == 1] > 0).all() and (zc[s == 2] < 0).all()

            ax.imshow(s + 10*(s==0), zorder=0, vmin=0, vmax=10, cmap='Set1',
                    extent=[xmin, xmax, ymin, ymax], origin='lower')
        else:
            ax.set_title(f'f(x,y)')
            ax.imshow(z, zorder=0, vmin=-4, vmax=+4, cmap='seismic',
                    extent=[xmin, xmax, ymin, ymax], origin='lower')

        fig.show()

    logger.info("All done.")
    plt.show()
