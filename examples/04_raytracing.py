import sys

import numpy as np
import slangpy as spy

from PIL import Image, ImageOps

from utils import *

display_iterations = False

aspect_ratio = 1024.0 / 768.0
try:
    import screeninfo
    h = screeninfo.get_monitors()[0].height
except Exception:
    h = 768
img_size = (int(h*aspect_ratio), int(h))


def compute_levelsets_bbox(centers, radii):
    out = np.empty((centers.shape[0], 6), dtype=np.float32)
    out[:, :3] = centers - radii.reshape(-1, 1)
    out[:, 3:] = centers + radii.reshape(-1, 1)
    return out


def build_acceleration_structure(device, desc, kind):
    sizes = device.get_acceleration_structure_sizes(desc)
    scratch = device.create_buffer(size=sizes.scratch_size, usage=spy.BufferUsage.unordered_access)
    accel = device.create_acceleration_structure(kind=kind, size=sizes.acceleration_structure_size)
    encoder = device.create_command_encoder()
    encoder.build_acceleration_structure(desc=desc, dst=accel, src=None, scratch_buffer=scratch)
    device.submit_command_buffer(encoder.finish())
    return accel


def create_acceleration_structure(device, bboxes):
    aabbs = device.create_buffer(usage=spy.BufferUsage.shader_resource | spy.BufferUsage.acceleration_structure_build_input,
                                 data=bboxes)

    geometry = spy.AccelerationStructureBuildInputProceduralPrimitives()
    geometry.aabb_buffers = [spy.BufferOffsetPair(aabbs)]
    geometry.aabb_stride = bboxes.itemsize * 6
    geometry.primitive_count = bboxes.shape[0]
    geometry.flags = spy.AccelerationStructureGeometryFlags.none
    blas = build_acceleration_structure(device, spy.AccelerationStructureBuildDesc({"inputs": [geometry]}),
                                        spy.AccelerationStructureKind.bottom_level)

    instances = device.create_acceleration_structure_instance_list(1)
    instances.write(0, {
        "transform": spy.float3x4.identity(),
        "instance_id": 0,
        "instance_mask": 0xFF,
        "instance_contribution_to_hit_group_index": 0,
        "flags": spy.AccelerationStructureInstanceFlags.none,
        "acceleration_structure": blas.handle,
    })
    tlas = build_acceleration_structure(device, spy.AccelerationStructureBuildDesc({"inputs": [instances.build_input_instances()]}),
                                        spy.AccelerationStructureKind.top_level)
    return blas, tlas, aabbs


def create_pipeline(device):
    logger.info("Building ray tracing pipeline of raytracing.slang...")
    program = device.load_program(str(shader_dir / "raytracing.slang"),
                                  ["ray_generation", "miss", "levelset_intersection", "closest_hit"])
    pipeline = device.create_ray_tracing_pipeline(
        program=program,
        hit_groups=[spy.HitGroupDesc(hit_group_name="levelset",
                                     closest_hit_entry_point="closest_hit",
                                     intersection_entry_point="levelset_intersection")],
        max_recursion=1,
        max_ray_payload_size=12,
        max_attribute_size=16)
    shader_table = device.create_shader_table(program=program,
                                              ray_gen_entry_points=["ray_generation"],
                                              miss_entry_points=["miss"],
                                              hit_group_names=["levelset"])
    return pipeline, shader_table


def launch_pipeline(device, pipeline, shader_table, tlas, centers, radii):
    image = create_buffer(device, img_size[0]*img_size[1], np.uint32)
    centers = create_buffer(device, data=centers.ravel(), dtype=np.float32)
    radii = create_buffer(device, data=radii, dtype=np.float32)

    encoder = device.create_command_encoder()
    with encoder.begin_ray_tracing_pass() as pass_encoder:
        cursor = spy.ShaderCursor(pass_encoder.bind_pipeline(pipeline, shader_table))
        cursor.scene = tlas
        cursor.image = image
        cursor.image_width = img_size[0]
        cursor.image_height = img_size[1]
        cursor.ray_marching_steps = 1000
        cursor.ray_marching_threshold = 1e-4
        cursor.cam_eye = spy.float3(0, 0, 2.0)
        cursor.cam_u = spy.float3(1.10457, 0, 0)
        cursor.cam_v = spy.float3(0, 0.828427, 0)
        cursor.cam_w = spy.float3(0, 0, -2.0)
        cursor.rendering_mode = int(display_iterations)
        cursor.background = spy.float3(0.01, 0.01, 0.01)
        cursor.centers = centers
        cursor.radii = radii
        pass_encoder.dispatch_rays(0, [img_size[0], img_size[1], 1])
    device.submit_command_buffer(encoder.finish())
    device.wait()

    return read_buffer(image, np.uint8).reshape(img_size[1], img_size[0], 4)


if __name__ == "__main__":
    args = parse_example_arguments("Ray tracing of implicit surfaces with affine arithmetic.",
                                   output="file name of the rendered image (optional)")
    logger.info("Setting up the Vulkan ray tracing scene...")
    device = create_device(ignore_rounding=args.fast)
    if not (device.has_feature(spy.Feature.acceleration_structure) and device.has_feature(spy.Feature.ray_tracing)):
        logger.error("This example requires a device with hardware ray tracing support.")
        sys.exit(1)

    centers = np.array([[-0.75, -0.75, -0.5], [-0.45, -0.45, -0.5], [0.0, 0.0, 0.0], [0.5, 0.5, 0]], dtype=np.float32)
    radii = np.array([0.1, 0.15, 0.2, 0.25], dtype=np.float32)
    bboxes = compute_levelsets_bbox(centers, radii)
    blas, tlas, aabbs = create_acceleration_structure(device, bboxes)

    pipeline, shader_table = create_pipeline(device)

    logger.info("Rendering image...")
    img = launch_pipeline(device, pipeline, shader_table, tlas, centers, radii)
    img = ImageOps.flip(Image.fromarray(img, 'RGBA'))
    if args.output:
        img.save(args.output)
        logger.info(f"Image saved to {args.output}.")
    else:
        img.show()
