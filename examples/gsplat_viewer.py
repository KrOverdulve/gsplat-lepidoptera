# Modified from gsplat (https://github.com/nerfstudio-project/gsplat, examples/gsplat_viewer.py, v1.5.3),
# Copyright 2025 Nerfstudio Team, licensed under the Apache License 2.0.
# Changes: training-camera frustums, jump to a training camera, save the camera pose, and toggles for
# the mirror-plane Gaussians and the reflection.

import json
import math
import viser
from pathlib import Path
from typing import Literal
from typing import Tuple, Callable
from nerfview import Viewer, RenderTabState
import numpy as np
import viser.transforms as tf

class GsplatRenderTabState(RenderTabState):
    # non-controlable parameters
    total_gs_count: int = 0
    rendered_gs_count: int = 0

    # controlable parameters
    max_sh_degree: int = 5
    near_plane: float = 1e-2
    far_plane: float = 1e2
    radius_clip: float = 0.0
    eps2d: float = 0.3
    backgrounds: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    render_mode: Literal[
        "rgb", "depth(accumulated)", "depth(expected)", "alpha"
    ] = "rgb"
    normalize_nearfar: bool = False
    inverse: bool = False
    colormap: Literal[
        "turbo", "viridis", "magma", "inferno", "cividis", "gray"
    ] = "turbo"
    rasterize_mode: Literal["classic", "antialiased"] = "classic"
    camera_model: Literal["pinhole", "ortho", "fisheye"] = "pinhole"
    cam_name: str = None

    # Mirror-compositing toggles (only used when a mirror is configured).
    show_plane: bool = True
    show_reflection: bool = True


class GsplatViewer(Viewer):
    """
    Viewer for gsplat.
    """

    def __init__(
        self,
        server: viser.ViserServer,
        render_fn: Callable,
        output_dir: Path,
        trainset,
        scene_scale: float = 1.0,
        mode: Literal["rendering", "training"] = "rendering",
    ):
        self.trainset = trainset
        self.scene_scale = scene_scale
        super().__init__(server, render_fn, output_dir, mode)
        server.gui.set_panel_label("gsplat viewer")

    def _init_rendering_tab(self):
        self.render_tab_state = GsplatRenderTabState()
        self._rendering_tab_handles = {}
        self._rendering_folder = self.server.gui.add_folder("Rendering")

    def _populate_rendering_tab(self):
        server = self.server

        for i, train_cam in enumerate(self.trainset):
            H, W = train_cam['image'].shape[:2]
            fy = train_cam['K'][1, 1].item()
            
            # 1. Update/Add GT Camera (Red)
            server.scene.add_camera_frustum(
                name=f"train/{i}",
                fov=2 * np.arctan2(H / 2, fy),
                aspect=W / H,
                scale=0.1 * self.scene_scale,
                image=train_cam['image'].cpu().numpy() / 255.0,
                position=train_cam['camtoworld'][:3, 3].cpu().numpy(),
                wxyz=tf.SO3.from_matrix(train_cam['camtoworld'][:3, :3].cpu().numpy()).wxyz,
                color=(1.0, 0.0, 0.0),
                line_width=4.0,
            )

        with self._rendering_folder:
            with server.gui.add_folder("Gsplat"):
                total_gs_count_number = server.gui.add_number(
                    "Total",
                    initial_value=self.render_tab_state.total_gs_count,
                    disabled=True,
                    hint="Total number of splats in the scene."
                )

                def move_camera_to_ds_cam(ds, val, client):
                    for f in ds:
                        if f["image_name"] == val:
                            target_se3 = viser.transforms.SE3.from_matrix(f["camtoworld"].detach().cpu().numpy())
                            with client.atomic():
                                client.camera.wxyz = target_se3.rotation().wxyz
                                client.camera.position = target_se3.translation()
                            break
                    self.rerender(_)

                train_cam = server.gui.add_dropdown(
                    "Render Train camera",
                    options=[""] + [f["image_name"] for f in self.trainset],
                    initial_value=self.render_tab_state.cam_name,
                    hint="Render to this train camera",
                )

                @train_cam.on_update
                def _(evt: viser.GuiEvent) -> None:
                    if train_cam.value != "":
                        self.render_tab_state.cam_name = train_cam.value
                        move_camera_to_ds_cam(self.trainset, train_cam.value, evt.client)

                rendered_gs_count_number = server.gui.add_number(
                    "Rendered",
                    initial_value=self.render_tab_state.rendered_gs_count,
                    disabled=True,
                    hint="Number of splats rendered.",
                )

                max_sh_degree_number = server.gui.add_number(
                    "Max SH",
                    initial_value=self.render_tab_state.max_sh_degree,
                    min=0,
                    max=5,
                    step=1,
                    hint="Maximum SH degree used",
                )

                @max_sh_degree_number.on_update
                def _(_) -> None:
                    self.render_tab_state.max_sh_degree = int(
                        max_sh_degree_number.value
                    )
                    self.rerender(_)

                near_far_plane_vec2 = server.gui.add_vector2(
                    "Near/Far",
                    initial_value=(
                        self.render_tab_state.near_plane,
                        self.render_tab_state.far_plane,
                    ),
                    min=(1e-3, 1e1),
                    max=(1e1, 1e3),
                    step=1e-3,
                    hint="Near and far plane for rendering.",
                )

                @near_far_plane_vec2.on_update
                def _(_) -> None:
                    self.render_tab_state.near_plane = near_far_plane_vec2.value[0]
                    self.render_tab_state.far_plane = near_far_plane_vec2.value[1]
                    self.rerender(_)

                radius_clip_slider = server.gui.add_number(
                    "Radius Clip",
                    initial_value=self.render_tab_state.radius_clip,
                    min=0.0,
                    max=100.0,
                    step=1.0,
                    hint="2D radius clip for rendering.",
                )

                @radius_clip_slider.on_update
                def _(_) -> None:
                    self.render_tab_state.radius_clip = radius_clip_slider.value
                    self.rerender(_)

                eps2d_slider = server.gui.add_number(
                    "2D Epsilon",
                    initial_value=self.render_tab_state.eps2d,
                    min=0.0,
                    max=1.0,
                    step=0.01,
                    hint="Epsilon added to the egienvalues of projected 2D covariance matrices.",
                )

                @eps2d_slider.on_update
                def _(_) -> None:
                    self.render_tab_state.eps2d = eps2d_slider.value
                    self.rerender(_)

                backgrounds_slider = server.gui.add_rgb(
                    "Background",
                    initial_value=self.render_tab_state.backgrounds,
                    hint="Background color for rendering.",
                )

                @backgrounds_slider.on_update
                def _(_) -> None:
                    self.render_tab_state.backgrounds = backgrounds_slider.value
                    self.rerender(_)

                render_mode_dropdown = server.gui.add_dropdown(
                    "Render Mode",
                    ("rgb", "depth(accumulated)", "depth(expected)", "alpha"),
                    initial_value=self.render_tab_state.render_mode,
                    hint="Render mode to use.",
                )

                @render_mode_dropdown.on_update
                def _(_) -> None:
                    if "depth" in render_mode_dropdown.value:
                        normalize_nearfar_checkbox.disabled = False
                        inverse_checkbox.disabled = False
                    else:
                        normalize_nearfar_checkbox.disabled = True
                        inverse_checkbox.disabled = True
                    self.render_tab_state.render_mode = render_mode_dropdown.value
                    self.rerender(_)

                normalize_nearfar_checkbox = server.gui.add_checkbox(
                    "Normalize Near/Far",
                    initial_value=self.render_tab_state.normalize_nearfar,
                    disabled=True,
                    hint="Normalize depth with near/far plane.",
                )

                @normalize_nearfar_checkbox.on_update
                def _(_) -> None:
                    self.render_tab_state.normalize_nearfar = (
                        normalize_nearfar_checkbox.value
                    )
                    self.rerender(_)

                inverse_checkbox = server.gui.add_checkbox(
                    "Inverse",
                    initial_value=self.render_tab_state.inverse,
                    disabled=True,
                    hint="Inverse the depth.",
                )

                @inverse_checkbox.on_update
                def _(_) -> None:
                    self.render_tab_state.inverse = inverse_checkbox.value
                    self.rerender(_)

                colormap_dropdown = server.gui.add_dropdown(
                    "Colormap",
                    ("turbo", "viridis", "magma", "inferno", "cividis", "gray"),
                    initial_value=self.render_tab_state.colormap,
                    hint="Colormap used for rendering depth/alpha.",
                )

                @colormap_dropdown.on_update
                def _(_) -> None:
                    self.render_tab_state.colormap = colormap_dropdown.value
                    self.rerender(_)

                rasterize_mode_dropdown = server.gui.add_dropdown(
                    "Anti-Aliasing",
                    ("classic", "antialiased"),
                    initial_value=self.render_tab_state.rasterize_mode,
                    hint="Whether to use classic or antialiased rasterization.",
                )

                @rasterize_mode_dropdown.on_update
                def _(_) -> None:
                    self.render_tab_state.rasterize_mode = rasterize_mode_dropdown.value
                    self.rerender(_)

                camera_model_dropdown = server.gui.add_dropdown(
                    "Camera",
                    ("pinhole", "ortho", "fisheye"),
                    initial_value=self.render_tab_state.camera_model,
                    hint="Camera model used for rendering.",
                )

                @camera_model_dropdown.on_update
                def _(_) -> None:
                    self.render_tab_state.camera_model = camera_model_dropdown.value
                    self.rerender(_)

                with server.gui.add_folder("Mirror"):
                    show_plane_checkbox = server.gui.add_checkbox(
                        "Show plane splats",
                        initial_value=self.render_tab_state.show_plane,
                        hint="Include the mirror-surface (plane) Gaussians in the render.",
                    )

                    @show_plane_checkbox.on_update
                    def _(_) -> None:
                        self.render_tab_state.show_plane = show_plane_checkbox.value
                        self.rerender(_)

                    show_reflection_checkbox = server.gui.add_checkbox(
                        "Show reflection",
                        initial_value=self.render_tab_state.show_reflection,
                        hint="Include the mirrored (reflected) butterfly in the render.",
                    )

                    @show_reflection_checkbox.on_update
                    def _(_) -> None:
                        self.render_tab_state.show_reflection = (
                            show_reflection_checkbox.value
                        )
                        self.rerender(_)

            with server.gui.add_folder("Pose"):
                save_pose_button = server.gui.add_button(
                    "Save pose",
                    hint="Download the current camera pose as JSON (right-click the "
                    "link and 'Save as...' to choose the file).",
                )

                @save_pose_button.on_click
                def _(event: viser.GuiEvent) -> None:
                    client = event.client
                    if client is None:
                        return
                    cam = client.camera
                    pose = {
                        "position": [float(v) for v in cam.position],
                        "wxyz": [float(v) for v in cam.wxyz],
                        "fov": float(cam.fov),
                        "aspect": float(cam.aspect),
                    }
                    client.send_file_download(
                        "pose.json", json.dumps(pose, indent=2).encode("utf-8")
                    )

                load_pose_button = server.gui.add_upload_button(
                    "Load pose",
                    mime_type="application/json",
                    hint="Upload a saved pose JSON; the view jumps to that pose.",
                )

                @load_pose_button.on_upload
                def _(event: viser.GuiEvent) -> None:
                    try:
                        pose = json.loads(event.target.value.content.decode("utf-8"))
                        position = np.asarray(pose["position"], dtype=np.float64)
                        wxyz = np.asarray(pose["wxyz"], dtype=np.float64)
                    except (ValueError, KeyError, UnicodeDecodeError):
                        return
                    clients = (
                        [event.client]
                        if event.client is not None
                        else list(server.get_clients().values())
                    )
                    for client in clients:
                        with client.atomic():
                            client.camera.position = position
                            client.camera.wxyz = wxyz
                            if "fov" in pose:
                                client.camera.fov = float(pose["fov"])
                    self.rerender(event)

        self._rendering_tab_handles.update(
            {
                "total_gs_count_number": total_gs_count_number,
                "rendered_gs_count_number": rendered_gs_count_number,
                "near_far_plane_vec2": near_far_plane_vec2,
                "radius_clip_slider": radius_clip_slider,
                "eps2d_slider": eps2d_slider,
                "backgrounds_slider": backgrounds_slider,
                "render_mode_dropdown": render_mode_dropdown,
                "normalize_nearfar_checkbox": normalize_nearfar_checkbox,
                "inverse_checkbox": inverse_checkbox,
                "colormap_dropdown": colormap_dropdown,
                "rasterize_mode_dropdown": rasterize_mode_dropdown,
                "camera_model_dropdown": camera_model_dropdown,
            }
        )
        super()._populate_rendering_tab()

    def _after_render(self):
        # Update the GUI elements with current values
        self._rendering_tab_handles[
            "total_gs_count_number"
        ].value = self.render_tab_state.total_gs_count
        self._rendering_tab_handles[
            "rendered_gs_count_number"
        ].value = self.render_tab_state.rendered_gs_count
