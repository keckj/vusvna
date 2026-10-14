import time

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches

from utils import *

if __name__ == "__main__":
    args = parse_example_arguments("Adaptive quadtree refinement around the zero level set of a 2D function.")
    fname = 'f(x,y) = x^2 + y^2 + x*y - 0.5*(x*y)^2 - 0.25'
    logger.info(f"Generating quadtree for function {fname}.")

    do_plot = False
    nlevels = 6 if do_plot else 25

    start_level = 2
    end_level = start_level + nlevels

    xmin, xmax = -2, 2
    ymin, ymax = -2, 2

    logger.info(f"Plotting has been {'enabled' if do_plot else 'disabled'}.")
    if do_plot:
        def f(x,y):
            return x*x + y*y + x*y - (x*y)**2/2 - 0.25

        fig, axes = plt.subplots(nrows=2, ncols=nlevels//2, figsize=(14,10))
        fig.suptitle(fname)

        x = np.linspace(xmin, xmax, 1024)[None,:]
        y = np.linspace(ymin, ymax, 1024)[:,None]
        z = f(x,y)
    else:
        axes = np.full(nlevels, fill_value=None)

    device = create_device(ignore_rounding=args.fast)
    kernel, = load_kernels(device, "quadtree.slang", ("quadtree",))

    logger.info(f"Computing quad_tree with start_level={start_level} and end_level={end_level-1}:")
    N = 2**start_level

    nallocated = 100_000_000
    active_cell_indices_in  = create_buffer(device, nallocated, np.uint64)  # should be large enough to handle end_level active cells count
    active_cell_indices_out = create_buffer(device, nallocated, np.uint64)
    active_cell_counter = create_buffer(device, 1, np.uint32)
    zero = np.zeros(1, np.uint32)

    # the first dispatch creates the pipeline, run it once without any active cell
    dispatch_1d(kernel, 0, {"current_active_cells": 0, "active_cells_in": active_cell_indices_in,
        "active_cells_out": active_cell_indices_out, "active_cells_counter": active_cell_counter})
    device.wait()

    nactive = (N//2)**2  # correponds to active cells of previous level = start_level - 1.
    for current_level, ax in zip(range(start_level, end_level), axes.ravel()):
        active_cell_counter.copy_from_numpy(zero)   # reset cell counter for next level
        device.wait()

        start = time.perf_counter()
        dispatch_1d(kernel, nactive, {
            "xmin": xmin, "xmax": xmax, "ymin": ymin, "ymax": ymax,
            "start_level": start_level, "current_level": current_level,
            "current_active_cells": nactive, "max_active_cells": nallocated,
            "active_cells_in": active_cell_indices_in, "active_cells_out": active_cell_indices_out,
            "active_cells_counter": active_cell_counter})
        device.wait()
        time_ms = 1000 * (time.perf_counter() - start)

        nactive = int(read_buffer(active_cell_counter, np.uint32)[0])
        Ntot = 2**current_level
        logger.info(f"  {1000*time_ms:8.1f}us -- level {current_level:2d} -- grid {Ntot:8d}^2 = 2^{2*current_level:2d}"
                    f" -- active cells {nactive}/{Ntot**2} ({100.0*nactive/(Ntot**2):2.1e}%)")

        if nactive == 0:
            break

        if nactive > nallocated:
            logger.warning(f"You need to allocate more indices to compute level {current_level}: nactive={nactive:2.2e} > nallocated={nallocated:2.2e}.")
            break

        if do_plot:
            ax.set_title(f'level={current_level}')
            ax.imshow(z, zorder=0, vmin=-4, vmax=+4, cmap='seismic',
                    extent=[xmin, xmax, ymin, ymax], origin='lower')
            ax.contour(z, levels=[0], zorder=1, colors=['k'],
                    extent=[xmin, xmax, ymin, ymax], origin='lower')
            ax.set_xlim(xmin, xmax)
            ax.set_ylim(ymin, ymax)

            idx = read_buffer(active_cell_indices_out, np.uint64, nactive)
            for k in idx:
                i = k // Ntot
                j = k % Ntot
                dy = (ymax - ymin) / Ntot
                dx = (xmax - xmin) / Ntot
                yi = ymin + i*dy
                xi = xmin + j*dx

                rectangle = matplotlib.patches.Rectangle((xi,yi), dx, dy,
                        zorder=2, alpha=1.0, facecolor='b', edgecolor='k')
                ax.add_patch(rectangle)

            fig.show()

        active_cell_indices_in, active_cell_indices_out = active_cell_indices_out, active_cell_indices_in

    logger.info("All done.")

    if do_plot:
        plt.show()
