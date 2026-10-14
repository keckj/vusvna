import os, logging, pathlib

import numpy as np
import slangpy as spy

script_dir = pathlib.Path(__file__).resolve().parent
shader_dir = script_dir / "shaders"
vusvna_dir = script_dir.parent / "vusvna"

formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

stream = logging.StreamHandler()
stream.setFormatter(formatter)
stream.setLevel(os.environ.get("VUSVNA_LOG_LEVEL", "DEBUG"))

logger = logging.getLogger("vusvna")
logger.propagate = False
logger.addHandler(stream)
logger.setLevel(os.environ.get("VUSVNA_LOG_LEVEL", "DEBUG"))

compiler_options = spy.SlangCompilerOptions({"include_paths": [str(vusvna_dir), str(shader_dir)]})


def create_device(**kwargs):
    """Vulkan device that finds the vusvna module and the shaders of the examples."""
    device = spy.Device(type=spy.DeviceType.vulkan, compiler_options=compiler_options, **kwargs)
    info = device.info
    logger.info(f"Using {info.adapter_name} with {info.api_name}.")
    return device


def load_kernels(device, shader, entry_points):
    """Compute kernels of the given entry points of a shader of examples/shaders."""
    logger.info(f"Building kernel(s) {', '.join(entry_points)} of {shader}...")
    kernels = []
    for entry in entry_points:
        program = device.load_program(str(shader_dir / shader), [entry])
        kernels.append(device.create_compute_kernel(program))
    return tuple(kernels)


def create_buffer(device, count=None, dtype=np.float32, data=None):
    """Structured buffer of count elements of a numpy scalar type, optionally initialized with data."""
    usage = spy.BufferUsage.shader_resource | spy.BufferUsage.unordered_access
    if data is not None:
        data = np.ascontiguousarray(data, dtype=dtype)
        return device.create_buffer(element_count=data.size, struct_size=data.itemsize, usage=usage, data=data)
    return device.create_buffer(element_count=count, struct_size=np.dtype(dtype).itemsize, usage=usage)


def read_buffer(buffer, dtype=np.float32, count=None):
    data = buffer.to_numpy().view(dtype)
    return data if count is None else data[:count]


def dispatch_1d(kernel, n, vars, group_size=256):
    """Dispatch at least n threads over a 2D grid of groups (each dimension holds at most 65535 groups),
    the kernel recovers the linear thread index with the uniform row_width."""
    groups = max(1, (n + group_size - 1) // group_size)
    gx = min(groups, 65535)
    gy = (groups + gx - 1) // gx
    kernel.dispatch(spy.uint3(gx * group_size, gy, 1), vars={**vars, "row_width": gx * group_size})
