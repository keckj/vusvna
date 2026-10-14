import sys, enum, time, argparse

import numpy as np
import slangpy as spy

from PIL import Image

from utils import *

#------------------------------------------------------------------------------
# Scene
#------------------------------------------------------------------------------

class Shape(enum.IntEnum):
    """Must match the shape constants of shaders/levelset.slang."""
    FLOOR       = 0
    GYROID      = 1
    METABALLS   = 2
    TWISTED_BOX = 3
    KUMMER      = 4
    BARTH       = 5
    HEART       = 6
    TETRAHEDRAL = 7

class RenderingMode(enum.IntEnum):
    SHADED  = 0
    NORMALS = 1
    STEPS   = 2

class MarchingMethod(enum.IntEnum):
    """See march() in shaders/levelset.slang."""
    SIGN_TEST   = 0
    ROOT_WINDOW = 1

# must match the Object struct of shaders/levelset.slang
object_dtype = np.dtype([
    ('center', '3f4'),
    ('scale',  'f4'),
    ('spin',   'f4'),
    ('shape',  'i4'),
    ('phase',  'f4'),
    ('pad',    'f4'),
])

floor_y = -1.5

def make_scene():
    """A gyroid ball surrounded by a ring of animated level sets, standing above a checkered floor."""
    # around the gyroid, starting in front of the camera
    ring = [Shape.HEART, Shape.KUMMER, Shape.BARTH, Shape.TETRAHEDRAL, Shape.TWISTED_BOX, Shape.METABALLS]
    ring_radius, ring_scale = 4.0, 0.95

    objects = np.zeros(2 + len(ring), dtype=object_dtype)
    objects[0] = ((0.0, floor_y, 0.0), 0.0, 0.0, Shape.FLOOR, 0.0, 0.0)
    objects[1] = ((0.0, floor_y + 1.45, 0.0), 1.4, 0.25, Shape.GYROID, 0.0, 0.0)
    for i, shape in enumerate(ring):
        angle = np.pi/2 + 2*np.pi*(i + 0.5)/len(ring)
        center = (ring_radius*np.cos(angle), floor_y + ring_scale + 0.1, -ring_radius*np.sin(angle))
        spin = 0.5 * (-1)**i
        objects[2+i] = (center, ring_scale, spin, shape, 1.7*i, 0.0)
    return objects

def compute_bboxes(objects):
    bboxes = np.empty((objects.size, 6), dtype=np.float32)
    extent = objects['scale'].reshape(-1, 1)
    bboxes[:, :3] = objects['center'] - extent
    bboxes[:, 3:] = objects['center'] + extent
    floor = (objects['shape'] == Shape.FLOOR)
    bboxes[floor] = (-50.0, floor_y - 1e-3, -50.0, 50.0, floor_y + 1e-3, 50.0)
    return bboxes

#------------------------------------------------------------------------------
# Camera
#------------------------------------------------------------------------------

class Camera:
    """Pinhole camera that orbits around its target (left button), looks around (right button) and zooms (wheel)."""
    def __init__(self, eye, look_at, up=(0, 1, 0), fov_y=35.0):
        self.eye = np.asarray(eye, dtype=np.float64)
        self.look_at = np.asarray(look_at, dtype=np.float64)
        self.up = np.asarray(up, dtype=np.float64)
        self.fov_y = fov_y
        offset = self.eye - self.look_at
        self.distance = np.linalg.norm(offset)
        self.yaw = np.arctan2(offset[0], offset[2])
        self.pitch = np.arcsin(offset[1] / self.distance)

    def direction(self):
        """Unit vector from the target to the eye."""
        c = np.cos(self.pitch)
        return np.array([c*np.sin(self.yaw), np.sin(self.pitch), c*np.cos(self.yaw)])

    def rotate(self, dx, dy, eye_fixed=False):
        self.yaw -= dx
        self.pitch = np.clip(self.pitch + dy, -1.5, 1.5)
        if eye_fixed:
            self.look_at = self.eye - self.distance*self.direction()
        else:
            self.eye = self.look_at + self.distance*self.direction()

    def zoom(self, steps):
        self.distance = max(self.distance * 0.9**steps, 1e-2)
        self.eye = self.look_at + self.distance*self.direction()

    def uvw_frame(self, aspect_ratio):
        """Same frame as the camera of the OptiX SDK, W is not normalized: its length is the focal length."""
        W = self.look_at - self.eye
        U = np.cross(W, self.up)
        U /= np.linalg.norm(U)
        V = np.cross(U, W)
        V /= np.linalg.norm(V)
        vlen = np.linalg.norm(W) * np.tan(0.5 * np.deg2rad(self.fov_y))
        return U * (vlen * aspect_ratio), V * vlen, W

#------------------------------------------------------------------------------
# Renderer
#------------------------------------------------------------------------------

class Settings:
    def __init__(self):
        self.ray_marching_steps = 400
        self.ray_marching_threshold = 1e-3
        self.rendering_mode = RenderingMode.SHADED
        self.marching_method = MarchingMethod.ROOT_WINDOW
        self.samples_per_pixel = 2
        self.subframe_index = 0
        self.time = 0.0


class Renderer:
    def __init__(self, device, width, height):
        self.device = device
        self.objects = make_scene()
        self.objects_buffer = device.create_buffer(element_count=self.objects.size, struct_size=self.objects.itemsize,
                usage=spy.BufferUsage.shader_resource, data=self.objects.view(np.uint8))
        self.create_acceleration_structure()
        self.create_pipeline()
        self.resize(width, height)

    def build_acceleration_structure(self, desc, kind):
        device = self.device
        sizes = device.get_acceleration_structure_sizes(desc)
        scratch = device.create_buffer(size=sizes.scratch_size, usage=spy.BufferUsage.unordered_access)
        accel = device.create_acceleration_structure(kind=kind, size=sizes.acceleration_structure_size)
        encoder = device.create_command_encoder()
        encoder.build_acceleration_structure(desc=desc, dst=accel, src=None, scratch_buffer=scratch)
        device.submit_command_buffer(encoder.finish())
        return accel

    def create_acceleration_structure(self):
        bboxes = compute_bboxes(self.objects)
        self.aabbs = self.device.create_buffer(data=bboxes,
                usage=spy.BufferUsage.shader_resource | spy.BufferUsage.acceleration_structure_build_input)

        geometry = spy.AccelerationStructureBuildInputProceduralPrimitives()
        geometry.aabb_buffers = [spy.BufferOffsetPair(self.aabbs)]
        geometry.aabb_stride = bboxes.itemsize * 6
        geometry.primitive_count = bboxes.shape[0]
        geometry.flags = spy.AccelerationStructureGeometryFlags.none
        self.blas = self.build_acceleration_structure(spy.AccelerationStructureBuildDesc({"inputs": [geometry]}),
                                                      spy.AccelerationStructureKind.bottom_level)

        instances = self.device.create_acceleration_structure_instance_list(1)
        instances.write(0, {
            "transform": spy.float3x4.identity(),
            "instance_id": 0,
            "instance_mask": 0xFF,
            "instance_contribution_to_hit_group_index": 0,
            "flags": spy.AccelerationStructureInstanceFlags.none,
            "acceleration_structure": self.blas.handle,
        })
        self.tlas = self.build_acceleration_structure(spy.AccelerationStructureBuildDesc({"inputs": [instances.build_input_instances()]}),
                                                      spy.AccelerationStructureKind.top_level)

    def create_pipeline(self):
        logger.info("Building ray tracing pipeline of levelset.slang...")
        device = self.device
        program = device.load_program(str(shader_dir / "levelset.slang"),
                ["ray_generation", "miss_radiance", "miss_shadow", "levelset_intersection", "closest_hit"])
        self.pipeline = device.create_ray_tracing_pipeline(
            program=program,
            hit_groups=[spy.HitGroupDesc(hit_group_name="levelset",
                                         closest_hit_entry_point="closest_hit",
                                         intersection_entry_point="levelset_intersection")],
            max_recursion=2,  # camera rays and shadow rays
            max_ray_payload_size=12,
            max_attribute_size=4)
        self.shader_table = device.create_shader_table(program=program,
                ray_gen_entry_points=["ray_generation"],
                miss_entry_points=["miss_radiance", "miss_shadow"],
                hit_group_names=["levelset"])

    def resize(self, width, height):
        self.width, self.height = max(width, 1), max(height, 1)
        self.image = self.device.create_texture(format=spy.Format.rgba8_unorm, width=self.width, height=self.height,
                usage=spy.TextureUsage.unordered_access | spy.TextureUsage.shader_resource | spy.TextureUsage.copy_source)
        self.accum_buffer = self.device.create_buffer(element_count=self.width*self.height, struct_size=16,
                usage=spy.BufferUsage.unordered_access)

    def render(self, settings, camera):
        """Renders a subframe in self.image (sRGB encoded) and waits for its completion."""
        U, V, W = camera.uvw_frame(self.width / self.height)
        encoder = self.device.create_command_encoder()
        with encoder.begin_ray_tracing_pass() as pass_encoder:
            cursor = spy.ShaderCursor(pass_encoder.bind_pipeline(self.pipeline, self.shader_table))
            cursor.scene = self.tlas
            cursor.objects = self.objects_buffer
            cursor.image = self.image
            cursor.accum_buffer = self.accum_buffer
            cursor.image_width = self.width
            cursor.image_height = self.height
            cursor.ray_marching_steps = settings.ray_marching_steps
            cursor.ray_marching_threshold = settings.ray_marching_threshold
            cursor.cam_eye = spy.float3(*camera.eye)
            cursor.cam_u = spy.float3(*U)
            cursor.cam_v = spy.float3(*V)
            cursor.cam_w = spy.float3(*W)
            cursor.rendering_mode = int(settings.rendering_mode)
            cursor.subframe_index = settings.subframe_index
            cursor.time = settings.time
            cursor.marching_method = int(settings.marching_method)
            cursor.samples_per_pixel = settings.samples_per_pixel
            pass_encoder.dispatch_rays(0, [self.width, self.height, 1])
        self.device.submit_command_buffer(encoder.finish())
        self.device.wait()

#------------------------------------------------------------------------------
# Interactive application
#------------------------------------------------------------------------------

controls = """ENTER  rendering mode
M      marching method
A      samples per pixel
SPACE  pause
+/-    animation speed (or UP/DOWN)
mouse  rotate, zoom
Q/ESC  quit"""

def make_camera():
    return Camera(eye=(0, 6.0, -12.5), look_at=(0, -0.6, 0), up=(0, 1, 0), fov_y=35)


class LevelsetWindow(spy.AppWindow):
    def __init__(self, app, width, height, settings):
        # the image is already sRGB encoded, the surface must not encode it again
        super().__init__(app, width=width, height=height, title="Vusvna Interval Marching",
                         surface_format=spy.Format.bgra8_unorm, enable_vsync=False)
        self.app = app
        self.settings = settings
        self.camera = make_camera()
        self.renderer = Renderer(self.device, width, height)
        self.paused = False
        self.speed = 1.0
        self.mouse_button = None
        self.mouse_pos = None
        self.image_changed = True
        self.last_time = time.perf_counter()

        self.overlay = spy.ui.Window(self.screen, "vusvna", size=spy.float2(330, 330))
        self.overlay_text = spy.ui.Text(self.overlay, "")
        self.frames = 0
        self.render_time = 0.0
        self.last_stats_time = None
        self.stats_text = ""

    def on_resize(self, width, height):
        if width > 0 and height > 0:
            self.device.wait()
            self.renderer.resize(width, height)
            self.image_changed = True

    def on_keyboard_event(self, event):
        s = self.settings
        if event.type == spy.KeyboardEventType.key_press:
            if event.key == spy.KeyCode.escape:
                self.app.terminate()
            elif event.key in (spy.KeyCode.enter, spy.KeyCode.keypad_enter):
                s.rendering_mode = RenderingMode((s.rendering_mode + 1) % len(RenderingMode))
            elif event.key == spy.KeyCode.space:
                self.paused = not self.paused
            elif event.key == spy.KeyCode.up:
                self.speed *= 1.25
            elif event.key == spy.KeyCode.down:
                self.speed /= 1.25
            else:
                return
            self.image_changed = True
        elif event.type == spy.KeyboardEventType.input:
            # letters and symbols use the codepoint of the keyboard layout (A is KeyCode.q on AZERTY)
            c = chr(event.codepoint).lower()
            if c == 'q':
                self.app.terminate()
            elif c == 'm':
                s.marching_method = MarchingMethod((s.marching_method + 1) % len(MarchingMethod))
            elif c == 'a':
                s.samples_per_pixel = s.samples_per_pixel % 8 * 2 or 1
            elif c in {'+', '='}:
                self.speed *= 1.25
            elif c == '-':
                self.speed /= 1.25
            else:
                return
            self.image_changed = True

    def on_mouse_event(self, event):
        if event.type == spy.MouseEventType.button_down:
            self.mouse_button = event.button
            self.mouse_pos = event.pos
        elif event.type == spy.MouseEventType.button_up:
            self.mouse_button = None
        elif event.type == spy.MouseEventType.move and self.mouse_button in (spy.MouseButton.left, spy.MouseButton.right):
            dx = (event.pos.x - self.mouse_pos.x) / self.renderer.width * np.pi
            dy = (event.pos.y - self.mouse_pos.y) / self.renderer.height * np.pi
            self.camera.rotate(dx, dy, eye_fixed=(self.mouse_button == spy.MouseButton.right))
            self.mouse_pos = event.pos
            self.image_changed = True
        elif event.type == spy.MouseEventType.scroll:
            self.camera.zoom(event.scroll.y)
            self.image_changed = True

    def update_overlay(self, now):
        s = self.settings
        self.frames += 1
        if self.last_stats_time is None:
            self.stats_text = "  --- fps\n\nrender :      --- ms"
            self.frames = 0
            self.render_time = 0.0
            self.last_stats_time = now
        elif now - self.last_stats_time > 0.5:
            self.stats_text = (f"{self.frames / (now - self.last_stats_time):5.1f} fps\n\n"
                               f"render : {1000.0 * self.render_time / self.frames:8.1f} ms")
            self.frames = 0
            self.render_time = 0.0
            self.last_stats_time = now
        settings = (f"mode   : {s.rendering_mode.name.lower()}\n"
                    f"method : {s.marching_method.name.lower()}\n"
                    f"samples: {s.samples_per_pixel} per pixel"
                    + (f", {s.subframe_index + 1} frames" if self.paused else "") + "\n"
                    f"speed  : x{self.speed:.2f}" + (" (paused)" if self.paused else ""))
        self.overlay_text.text = f"{self.stats_text}\n\n{settings}\n\n{controls}"

    def render(self, render_context):
        s = self.settings
        now = time.perf_counter()
        if not self.paused:
            s.time += (now - self.last_time) * self.speed
        self.last_time = now

        # restart the accumulation of samples whenever the image changes
        if not self.paused or self.image_changed:
            s.subframe_index = 0
        self.image_changed = False

        start = time.perf_counter()
        self.renderer.render(s, self.camera)
        self.render_time += time.perf_counter() - start
        s.subframe_index += 1

        render_context.command_encoder.blit(render_context.surface_texture, self.renderer.image)
        self.update_overlay(now)

#------------------------------------------------------------------------------
# Main
#------------------------------------------------------------------------------

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Animated level sets ray traced with affine arithmetic.")
    parser.add_argument("--output", help="render a single image to this file instead of opening a window")
    parser.add_argument("--time", type=float, default=0.0, help="animation time of the image (with --output)")
    parser.add_argument("--mode", choices=[m.name.lower() for m in RenderingMode], default="shaded")
    parser.add_argument("--method", choices=[m.name.lower() for m in MarchingMethod], default="root_window")
    parser.add_argument("--spp", type=int, default=2, help="samples per pixel and per frame")
    parser.add_argument("--frames", type=int, default=1, help="accumulated frames (with --output)")
    parser.add_argument("--size", type=int, nargs=2, default=(1024, 768), metavar=("WIDTH", "HEIGHT"))
    args = parser.parse_args()

    settings = Settings()
    settings.rendering_mode = RenderingMode[args.mode.upper()]
    settings.marching_method = MarchingMethod[args.method.upper()]
    settings.samples_per_pixel = args.spp
    width, height = args.size

    logger.info("Setting up the Vulkan ray tracing scene...")
    device = create_device()
    if not (device.has_feature(spy.Feature.acceleration_structure) and device.has_feature(spy.Feature.ray_tracing)):
        logger.error("This example requires a device with hardware ray tracing support.")
        sys.exit(1)

    if args.output:
        renderer = Renderer(device, width, height)
        camera = make_camera()
        settings.time = args.time
        renderer.render(settings, camera)  # the first launch creates the pipeline
        start = time.perf_counter()
        for k in range(args.frames):
            settings.subframe_index = k
            renderer.render(settings, camera)
        elapsed = (time.perf_counter() - start) / args.frames
        logger.info(f"Rendered {args.frames} frame(s) of {width}x{height} pixels at {settings.samples_per_pixel} spp"
                    f" in {1000*elapsed:.1f} ms per frame.")
        Image.fromarray(renderer.image.to_numpy(), 'RGBA').save(args.output)
        logger.info(f"Image saved to {args.output}.")
    else:
        app = spy.App(device)
        window = LevelsetWindow(app, width, height, settings)
        app.run()
