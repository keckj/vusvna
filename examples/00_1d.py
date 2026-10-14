import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches

from utils import *

if __name__ == "__main__":
    device = create_device()
    kernel, = load_kernels(device, "func_1d.slang", ("func_1d",))

    xmin, xmax = -2, 2
    Ns = [8,16,32,64]

    for k in range(2):
        fname = 'g(x) = sqrt(x^2 - x + 0.5) / sqrt(x^2 + 0.5)' if k==0 else 'g(g(x))'
        logger.info(f"Generating plots for function {fname}...")

        fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(20,10))
        fig.suptitle(fname)

        x = np.linspace(xmin, xmax, 1024)
        y = np.sqrt(x*x - x + 0.5) / np.sqrt(x*x + 0.5)
        if k == 1:
            y = np.sqrt(y*y - y + 0.5) / np.sqrt(y*y + 0.5)

        for ax, N in zip(axes.ravel(), Ns):
            logger.info(f"  using {N} intervals.")
            a = create_buffer(device, N)
            b = create_buffer(device, N)
            kernel.dispatch(spy.uint3(N, 1, 1),
                    vars={"xmin": xmin, "xmax": xmax, "nintervals": N, "k": k, "a": a, "b": b})
            a, b = read_buffer(a), read_buffer(b)

            dx = (xmax - xmin) / N
            for i in range(N):
                xstart = xmin+i*dx
                xend = xstart + dx

                # check correctness of returned results
                mask = np.where(np.logical_and(x >= xstart, x <= xend))
                assert (y[mask] >= a[i]).all()
                assert (y[mask] <= b[i]).all()

                rectangle = matplotlib.patches.Rectangle((xstart,a[i]), dx, b[i]-a[i],
                        zorder=0, alpha=0.5, facecolor='b', edgecolor='k')
                ax.add_patch(rectangle)

            ax.set_title(f'N={N}')
            ax.plot(x, y, zorder=1, c='r')
            ax.set_xlim(xmin, xmax)
            ax.set_ylim(0, 2)

            fig.show()

    logger.info("All done.")
    plt.show()
