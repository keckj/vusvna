import time

import numpy as np

from utils import *

if __name__ == "__main__":
    args = parse_example_arguments("Adaptive octree refinement around an implicit surface.")
    fname = r'f(x,y,z) = (x-2)^2(x+2)^2+(y-2)^2(y+2)^2+(z-2)^2(z+2)^2+(x^2y^2+x^2z^2+y^2z^2)+6*x*y*z-10(x^2+y^2+z^2)+22'
    logger.info(f"Generating octree for function {fname}.")

    do_plot = False
    nlevels = 3 if do_plot else 8

    start_level = 4
    end_level = start_level + nlevels

    xmin, xmax = -5, 5
    ymin, ymax = -5, 5
    zmin, zmax = -5, 5

    logger.info(f"Plotting has been {'enabled' if do_plot else 'disabled'}.")
    if do_plot:
        import open3d as o3d
        from skimage import measure

        Nm = 128
        logger.info(f"Running marching cubes on a {Nm}^3 grid...")
        def f(x,y,z):
            return (
                   (x-2)**2 * (x+2)**2
                 + (y-2)**2 * (y+2)**2
                 + (z-2)**2 * (z+2)**2
                 + (x**2 * y**2 + x**2 * z**2 + y**2 * z**2)
                 + 6*x*y*z
                 - 10*(x**2+y**2+z**2)
                 + 22
            )

        x = np.linspace(xmin, xmax, Nm)
        y = np.linspace(ymin, ymax, Nm)
        z = np.linspace(zmin, zmax, Nm)
        x, y, z = np.ix_(x, y, z)
        V, F, _, _ = measure.marching_cubes(f(x,y,z), 0)
        V[:,0] = xmin + (xmax-xmin)/Nm*V[:,0]
        V[:,1] = ymin + (ymax-ymin)/Nm*V[:,1]
        V[:,2] = zmin + (zmax-zmin)/Nm*V[:,2]

        mesh = o3d.geometry.TriangleMesh()
        mesh.vertices = o3d.utility.Vector3dVector(V)
        mesh.triangles = o3d.utility.Vector3iVector(F)

    device = create_device(ignore_rounding=args.fast)
    kernel, = load_kernels(device, "octree.slang", ("octree",))

    logger.info(f"Computing octree with start_level={start_level} and end_level={end_level}:")
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

    nactive = (N//2)**3  # correponds to active cells of previous level = start_level - 1.
    for current_level in range(start_level, end_level+1):
        active_cell_counter.copy_from_numpy(zero)   # reset cell counter for next level
        device.wait()

        start = time.perf_counter()
        dispatch_1d(kernel, nactive, {
            "xmin": xmin, "xmax": xmax, "ymin": ymin, "ymax": ymax, "zmin": zmin, "zmax": zmax,
            "start_level": start_level, "current_level": current_level,
            "current_active_cells": nactive, "max_active_cells": nallocated,
            "active_cells_in": active_cell_indices_in, "active_cells_out": active_cell_indices_out,
            "active_cells_counter": active_cell_counter})
        device.wait()
        time_ms = 1000 * (time.perf_counter() - start)

        nactive = int(read_buffer(active_cell_counter, np.uint32)[0])
        Ntot = 2**current_level
        logger.info(f"  {1000*time_ms:8.1f}us -- level {current_level:2d} -- grid {Ntot:5d}^3 = 2^{3*current_level:2d}"
                    f" -- active cells {nactive}/{Ntot**3} ({100.0*nactive/(Ntot**3):2.2f}%)")

        if nactive == 0:
            break

        if nactive > nallocated:
            logger.warning(f"You need to allocate more indices to compute level {current_level}: nactive={nactive:2.2e} > nallocated={nallocated:2.2e}.")
            break

        if do_plot:
            indices = read_buffer(active_cell_indices_out, np.uint64, nactive)

            dz = (zmax - zmin) / Ntot
            dy = (ymax - ymin) / Ntot
            dx = (xmax - xmin) / Ntot

            i = indices // (Ntot*Ntot)
            j = (indices // Ntot) % Ntot
            k = indices % Ntot

            zi = zmin + (i+0.5)*dz
            yi = ymin + (j+0.5)*dy
            xi = xmin + (k+0.5)*dx

            pts = o3d.geometry.PointCloud()
            pts.points = o3d.utility.Vector3dVector(np.column_stack([xi, yi, zi]))
            pts.colors = o3d.utility.Vector3dVector(np.random.uniform(0, 1, size=(nactive, 3)))

            assert dx==dy==dz, (dx,dy,dz)
            voxel_grid = o3d.geometry.VoxelGrid.create_from_point_cloud(pts, dx)

            o3d.visualization.draw_geometries([voxel_grid, mesh])

        active_cell_indices_in, active_cell_indices_out = active_cell_indices_out, active_cell_indices_in

    logger.info("All done.")
