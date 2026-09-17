# Modified from gsplat (https://github.com/nerfstudio-project/gsplat, examples/simple_trainer.py, v1.5.3),
# Copyright 2025 Nerfstudio Team, licensed under the Apache License 2.0.
# Changes: mirror-aware three-pass compositing with mirror-plane Gaussians, 2DGS support,
# alpha-entropy and distortion regularizers, initialization from a dense cloud, held-out views by name, masked metrics.

import json
import math
import os
import time
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import imageio
import numpy as np
import torch
import torch.nn.functional as F
import tqdm
import tyro
import viser
import yaml
from datasets.colmap import Dataset, Parser
from datasets.traj import (
    generate_ellipse_path_z,
    generate_interpolated_path,
    generate_spiral_path,
)
from torch import Tensor
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.tensorboard import SummaryWriter
from torchmetrics.image import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
from typing_extensions import Literal, assert_never
from utils import AppearanceOptModule, CameraOptModule, knn, rgb_to_sh, set_random_seed

from gsplat import export_splats
from gsplat.compression import PngCompression
from gsplat.distributed import cli
from gsplat.optimizers import SelectiveAdam
from gsplat.rendering import rasterization, rasterization_2dgs
from gsplat.strategy import DefaultStrategy, MCMCStrategy
from gsplat.strategy.ops import _update_param_with_optimizer
from gsplat_viewer import GsplatViewer, GsplatRenderTabState
from nerfview import CameraState, RenderTabState, apply_float_colormap

from plyfile import PlyData

@dataclass
class Config:
    # Detected mirror plane (plane_parameters.json, written by detect_mirror.py).
    # When set, training uses the mirror-aware three-pass composite: the specimen
    # is rendered directly and from the mirrored camera, over a separate set of
    # mirror-surface (plane) Gaussians. When unset, this is plain 2DGS/3DGS.
    mirror_param_file: Optional[str] = None

    # --- Mirror-aware training (the defaults are the paper's full model) ---

    # Model the mirror surface with a separate set of plane Gaussians. False is
    # the "no plane" ablation: the reflection is composited straight over the
    # background.
    model_plane: bool = True
    # Number of plane Gaussians, laid out as a grid on the plane.
    plane_num_pts: int = 400_000
    # SH degree of the plane Gaussians.
    plane_sh_degree: int = 3
    # Plane grid extent as a multiple of the specimen's projected extent.
    plane_extent_scale: float = 1.3
    # Splatting primitive: flat surfels ("2dgs") or volumetric Gaussians ("3dgs").
    model_type: Literal["3dgs", "2dgs"] = "2dgs"
    # Densify/prune the specimen Gaussians with the strategy (ADC). Off keeps the
    # Gaussian count of the dense init cloud; the plane Gaussians densify either way.
    butterfly_densify: bool = False
    # At the end of training, delete specimen Gaussians whose opacity is below
    # this (0 disables).
    final_prune_opacity: float = 0.005
    # Alpha-entropy loss (mirror runs only): binary entropy of the direct pass's
    # accumulated alpha, pushing every pixel toward opaque or empty. 0 disables.
    alpha_entropy_lambda: float = 0.1
    alpha_entropy_start: int = 5000
    # 2DGS distortion loss on the reflected pass (mirror runs with 2dgs only):
    # concentrates the per-ray weight into one surface. 0 disables.
    dist_lambda: float = 1.0
    dist_start_iter: int = 3_000
    # Mirror runs only: every this many steps, prune specimen Gaussians that sit
    # closer to the mirror plane than `plane_prune_dist` or behind it (0 disables).
    farside_prune_every: int = 100
    # World-space distance to the mirror plane below which init points and
    # specimen Gaussians are pruned. <0 uses plane_prune_default_scale * scene
    # scale, 0 disables.
    plane_prune_dist: float = -1.0
    plane_prune_default_scale: float = 0.008

    # Disable viewer
    disable_viewer: bool = False
    show_viewer_cams: bool = False
    clamp_viewer_cams_to: Optional[int] = -1
    # Path to the .pt files. If provide, it will skip training and run evaluation only.
    ckpt: Optional[List[str]] = None
    # Name of compression strategy to use
    compression: Optional[Literal["png"]] = None
    # Render trajectory path
    render_traj_path: str = "interp"

    # Scene directory (images/, sparse/, dense.ply, optionally image_masks/)
    data_dir: str = "data/butterfly1/undistorted"
    # Downsample factor for the dataset
    data_factor: int = 4
    # Directory to save results
    result_dir: str = "results/butterfly1"
    # Filename (no path; extension optional) of the held-out validation image.
    # None (the default) holds out no image: all images are used for training.
    holdout_image_name: Optional[str] = None
    # Random crop size for training  (experimental)
    patch_size: Optional[int] = None
    # A global scaler that applies to the scene size related parameters
    global_scale: float = 1.0
    # Normalize the world space
    normalize_world_space: bool = False
    # Camera model
    camera_model: Literal["pinhole", "ortho", "fisheye"] = "pinhole"

    # Port for the viewer server
    port: int = 8080

    # Batch size for training. Learning rates are scaled automatically
    batch_size: int = 1
    # A global factor to scale the number of training steps
    steps_scaler: float = 1.0

    # Number of training steps
    max_steps: int = 30_000
    # Steps to evaluate the model
    eval_steps: List[int] = field(default_factory=lambda: [7_000, 30_000])
    # Steps to save the model
    save_steps: List[int] = field(default_factory=lambda: [7_000, 30_000])
    # Whether to save ply file (storage size can be large)
    save_ply: bool = False
    # Steps to save the model as ply
    ply_steps: List[int] = field(default_factory=lambda: [7_000, 30_000])
    # Whether to disable video generation during training and evaluation
    disable_video: bool = False

    # Initialization strategy: "sfm" starts from the dense cloud (init_cloud.ply
    # for mirror runs, dense.ply otherwise), "random" from random points.
    init_type: str = "sfm"
    # Initial number of GSs. Ignored if using sfm
    init_num_pts: int = 100_000
    # Initial extent of GSs as a multiple of the camera extent. Ignored if using sfm
    init_extent: float = 3.0
    # Degree of spherical harmonics
    sh_degree: int = 3
    # Turn on another SH degree every this steps
    sh_degree_interval: int = 1000
    # Initial opacity of GS
    init_opa: float = 0.1
    # Initial scale of GS
    init_scale: float = 1.0
    # Weight for SSIM loss
    ssim_lambda: float = 0.2

    # Near plane clipping distance
    near_plane: float = 0.01
    # Far plane clipping distance
    far_plane: float = 1e10

    # Strategy for GS densification
    strategy: Union[DefaultStrategy, MCMCStrategy] = field(
        default_factory=DefaultStrategy
    )
    # Use packed mode for rasterization, this leads to less memory usage but slightly slower.
    packed: bool = False
    # Use sparse gradients for optimization. (experimental)
    sparse_grad: bool = False
    # Use visible adam from Taming 3DGS. (experimental)
    visible_adam: bool = False
    # Anti-aliasing in rasterization. Might slightly hurt quantitative metrics.
    antialiased: bool = False

    # Use random background for training to discourage transparency
    random_bkgd: bool = False

    # LR for 3D point positions
    means_lr: float = 1.6e-4
    # LR for Gaussian scale factors
    scales_lr: float = 5e-3
    # LR for alpha blending weights
    opacities_lr: float = 5e-2

    # LR for orientation (quaternions)
    quats_lr: float = 1e-3
    # LR for SH band 0 (brightness)
    sh0_lr: float = 2.5e-3
    # LR for higher-order SH (detail)
    shN_lr: float = 2.5e-3 / 20

    # Opacity regularization
    opacity_reg: float = 0.0
    # Scale regularization
    scale_reg: float = 0.0

    # Enable camera optimization.
    pose_opt: bool = False
    # Learning rate for camera optimization
    pose_opt_lr: float = 1e-5
    # Regularization for camera optimization as weight decay
    pose_opt_reg: float = 1e-6
    # Add noise to camera extrinsics. This is only to test the camera pose optimization.
    pose_noise: float = 0.0

    # Enable appearance optimization. (experimental)
    app_opt: bool = False
    # Appearance embedding dimension
    app_embed_dim: int = 16
    # Learning rate for appearance optimization
    app_opt_lr: float = 1e-3
    # Regularization for appearance optimization as weight decay
    app_opt_reg: float = 1e-6

    # Enable bilateral grid. (experimental)
    use_bilateral_grid: bool = False
    # Shape of the bilateral grid (X, Y, W)
    bilateral_grid_shape: Tuple[int, int, int] = (16, 16, 8)

    # Enable depth loss. (experimental)
    depth_loss: bool = False
    # Weight for depth loss
    depth_lambda: float = 1e-2

    # Dump information to tensorboard every this steps
    tb_every: int = 100
    # Save training images to tensorboard
    tb_save_image: bool = False

    lpips_net: Literal["vgg", "alex"] = "alex"

    # 3DGUT (uncented transform + eval 3D)
    with_ut: bool = False
    with_eval3d: bool = False

    # Whether use fused-bilateral grid
    use_fused_bilagrid: bool = False

    def adjust_steps(self, factor: float):
        self.eval_steps = [int(i * factor) for i in self.eval_steps]
        self.save_steps = [int(i * factor) for i in self.save_steps]
        self.ply_steps = [int(i * factor) for i in self.ply_steps]
        self.max_steps = int(self.max_steps * factor)
        self.sh_degree_interval = int(self.sh_degree_interval * factor)

        strategy = self.strategy
        if isinstance(strategy, DefaultStrategy):
            strategy.refine_start_iter = int(strategy.refine_start_iter * factor)
            strategy.refine_stop_iter = int(strategy.refine_stop_iter * factor)
            strategy.reset_every = int(strategy.reset_every * factor)
            strategy.refine_every = int(strategy.refine_every * factor)
        elif isinstance(strategy, MCMCStrategy):
            strategy.refine_start_iter = int(strategy.refine_start_iter * factor)
            strategy.refine_stop_iter = int(strategy.refine_stop_iter * factor)
            strategy.refine_every = int(strategy.refine_every * factor)
        else:
            assert_never(strategy)

def import_fused_ply_from_colmap(ply_path):
    plydata = PlyData.read(ply_path)
    
    vertex_data = plydata['vertex']
    
    x = np.asarray(vertex_data['x'])
    y = np.asarray(vertex_data['y'])
    z = np.asarray(vertex_data['z'])
    r = np.asarray(vertex_data['red'])
    g = np.asarray(vertex_data['green'])
    b = np.asarray(vertex_data['blue'])
    
    xyz = np.stack((x, y, z), axis=1)
    rgb = np.stack((r, g, b), axis=1)
    
    return xyz, rgb


def create_splats_with_optimizers(
    parser: Parser,
    init_type: str = "sfm",
    init_num_pts: int = 100_000,
    init_extent: float = 3.0,
    init_opacity: float = 0.1,
    init_scale: float = 1.0,
    means_lr: float = 1.6e-4,
    scales_lr: float = 5e-3,
    opacities_lr: float = 5e-2,
    quats_lr: float = 1e-3,
    sh0_lr: float = 2.5e-3,
    shN_lr: float = 2.5e-3 / 20,
    scene_scale: float = 1.0,
    sh_degree: int = 3,
    sparse_grad: bool = False,
    visible_adam: bool = False,
    batch_size: int = 1,
    feature_dim: Optional[int] = None,
    device: str = "cuda",
    world_rank: int = 0,
    world_size: int = 1,
    mirror_params: Optional[Dict] = None,
    plane_prune_dist: float = -1.0,
) -> Tuple[torch.nn.ParameterDict, Dict[str, torch.optim.Optimizer]]:
    if init_type == "sfm":
        # The init cloud from detect_mirror.py (camera-side specimen with the
        # reflection folded onto it, all behind-plane points removed) is only
        # appropriate WITH the mirror model. A mirror-unaware run instead uses
        # the plain dense patch-match cloud, which keeps the points COLMAP
        # triangulated behind the plane so the baseline can reconstruct the
        # reflection instead of hallucinating it as floaters.
        init_ply = os.path.join(parser.data_dir, "init_cloud.ply")
        if mirror_params is not None and os.path.exists(init_ply):
            points, points_rgb = import_fused_ply_from_colmap(init_ply)
        else:
            points, points_rgb = import_fused_ply_from_colmap(
                os.path.join(parser.data_dir, "dense.ply")
            )
        points = torch.from_numpy(points).float()
        rgbs = torch.from_numpy(points_rgb / 255.0).float()
    elif init_type == "random":
        points = init_extent * scene_scale * (torch.rand((init_num_pts, 3)) * 2 - 1)
        rgbs = torch.rand((init_num_pts, 3))
    else:
        raise ValueError("Please specify a correct init_type: sfm or random")

    # Drop init points hugging the mirror plane. These are background points that
    # would otherwise form an opaque sheet in front of the learned mirror
    # surface, occluding it. The specimen sits well off the plane and survives.
    # `plane_prune_dist` is a world-space distance: <0 uses
    # plane_prune_default_scale * scene scale, 0 disables, >0 is used verbatim.
    if mirror_params is not None and plane_prune_dist != 0.0:
        n = torch.tensor(
            [mirror_params["a"], mirror_params["b"], mirror_params["c"]]
        ).float()
        n = n / torch.norm(n)
        d = float(mirror_params["d"])
        dist = (points @ n + d).abs()
        thr = cfg.plane_prune_default_scale * scene_scale if plane_prune_dist < 0.0 else plane_prune_dist
        keep = dist >= thr
        print(
            f"Plane-prune init: |dist to plane| min={dist.min().item():.4f} "
            f"median={dist.median().item():.4f} max={dist.max().item():.4f}; "
            f"threshold={thr:.4f}; keeping {int(keep.sum())}/{len(keep)} points."
        )
        points = points[keep]
        rgbs = rgbs[keep]

    # Initialize the GS size to be the average dist of the 3 nearest neighbors
    dist2_avg = (knn(points, 4)[:, 1:] ** 2).mean(dim=-1)  # [N,]
    dist_avg = torch.sqrt(dist2_avg)
    scales = torch.log(dist_avg * init_scale).unsqueeze(-1).repeat(1, 3)  # [N, 3]

    # Distribute the GSs to different ranks (also works for single rank)
    points = points[world_rank::world_size]
    rgbs = rgbs[world_rank::world_size]
    scales = scales[world_rank::world_size]

    N = points.shape[0]
    quats = torch.rand((N, 4))  # [N, 4]
    opacities = torch.logit(torch.full((N,), init_opacity))  # [N,]



    params = [
        # name, value, lr
        ("means", torch.nn.Parameter(points), means_lr * scene_scale),
        ("scales", torch.nn.Parameter(scales), scales_lr),
        ("quats", torch.nn.Parameter(quats), quats_lr),
        ("opacities", torch.nn.Parameter(opacities), opacities_lr),
    ]

    if feature_dim is None:
        # color is SH coefficients.
        colors = torch.zeros((N, (sh_degree + 1) ** 2, 3))  # [N, K, 3]
        colors[:, 0, :] = rgb_to_sh(rgbs)
        params.append(("sh0", torch.nn.Parameter(colors[:, :1, :]), sh0_lr))
        params.append(("shN", torch.nn.Parameter(colors[:, 1:, :]), shN_lr))
    else:
        # features will be used for appearance and view-dependent shading
        features = torch.rand(N, feature_dim)  # [N, feature_dim]
        params.append(("features", torch.nn.Parameter(features), sh0_lr))
        colors = torch.logit(rgbs)  # [N, 3]
        params.append(("colors", torch.nn.Parameter(colors), sh0_lr))

    splats = torch.nn.ParameterDict({n: v for n, v, _ in params}).to(device)
    # Scale learning rate based on batch size, reference:
    # https://www.cs.princeton.edu/~smalladi/blog/2024/01/22/SDEs-ScalingRules/
    # Note that this would not make the training exactly equivalent, see
    # https://arxiv.org/pdf/2402.18824v1
    BS = batch_size * world_size
    optimizer_class = None
    if sparse_grad:
        optimizer_class = torch.optim.SparseAdam
    elif visible_adam:
        optimizer_class = SelectiveAdam
    else:
        optimizer_class = torch.optim.Adam
    optimizers = {
        name: optimizer_class(
            [{"params": splats[name], "lr": lr * math.sqrt(BS), "name": name}],
            eps=1e-15 / math.sqrt(BS),
            # TODO: check betas logic when BS is larger than 10 betas[0] will be zero.
            betas=(1 - BS * (1 - 0.9), 1 - BS * (1 - 0.999)),
        )
        for name, _, lr in params
    }
    return splats, optimizers

class Runner:
    """Engine for training and testing."""

    def __init__(
        self, local_rank: int, world_rank, world_size: int, cfg: Config
    ) -> None:
        set_random_seed(42 + local_rank)

        self.cfg = cfg
        self.world_rank = world_rank
        self.local_rank = local_rank
        self.world_size = world_size
        self.device = f"cuda:{local_rank}"

        # Where to dump results.
        os.makedirs(cfg.result_dir, exist_ok=True)

        # Setup output directories.
        self.ckpt_dir = f"{cfg.result_dir}/ckpts"
        os.makedirs(self.ckpt_dir, exist_ok=True)
        self.stats_dir = f"{cfg.result_dir}/stats"
        os.makedirs(self.stats_dir, exist_ok=True)
        self.render_dir = f"{cfg.result_dir}/renders"
        os.makedirs(self.render_dir, exist_ok=True)
        self.ply_dir = f"{cfg.result_dir}/ply"
        os.makedirs(self.ply_dir, exist_ok=True)

        # Tensorboard
        self.writer = SummaryWriter(log_dir=f"{cfg.result_dir}/tb")

        # Load data: Training data should contain initial points and colors.
        self.parser = Parser(
            data_dir=cfg.data_dir,
            factor=cfg.data_factor,
            normalize=cfg.normalize_world_space,
        )
        self.trainset = Dataset(
            self.parser,
            split="train",
            patch_size=cfg.patch_size,
            load_depths=cfg.depth_loss,
            holdout_image_name=cfg.holdout_image_name,
        )
        # Val needs the mask so eval can restrict metrics to the in-mask region.
        self.valset = Dataset(
            self.parser,
            split="val",
            load_masks=True,
            holdout_image_name=cfg.holdout_image_name,
        )
        self.scene_scale = self.parser.scene_scale * 1.1 * cfg.global_scale
        print("Scene scale:", self.scene_scale)

        # Read mirror parameters. Load the plane equation before building the
        # model so init points hugging the plane can be pruned; the rest of the
        # mirror setup (which needs the built model) happens afterwards.
        if cfg.mirror_param_file is None:
            print("Not using a mirror")
            self.mirror_params = None
        else:
            print("Using mirror parameters from:", cfg.mirror_param_file)
            with open(cfg.mirror_param_file, "r") as f:
                self.mirror_params = json.load(f)

        # Model
        feature_dim = 32 if cfg.app_opt else None
        self.real_splats, self.optimizers = create_splats_with_optimizers(
            self.parser,
            init_type=cfg.init_type,
            init_num_pts=cfg.init_num_pts,
            init_extent=cfg.init_extent,
            init_opacity=cfg.init_opa,
            init_scale=cfg.init_scale,
            means_lr=cfg.means_lr,
            scales_lr=cfg.scales_lr,
            opacities_lr=cfg.opacities_lr,
            quats_lr=cfg.quats_lr,
            sh0_lr=cfg.sh0_lr,
            shN_lr=cfg.shN_lr,
            scene_scale=self.scene_scale,
            sh_degree=cfg.sh_degree,
            sparse_grad=cfg.sparse_grad,
            visible_adam=cfg.visible_adam,
            batch_size=cfg.batch_size,
            feature_dim=feature_dim,
            device=self.device,
            world_rank=world_rank,
            world_size=world_size,
            mirror_params=self.mirror_params,
            plane_prune_dist=cfg.plane_prune_dist,
        )
        print("Model initialized. Number of GS:", len(self.real_splats["means"]))

        # Finish mirror setup (needs the built model for the plane extent).
        if self.mirror_params is not None:
            self._setup_mirror()

        # Densification Strategy. 2DGS exposes its densification gradient under a
        # different info key; point DefaultStrategy at it (shared with the plane).
        if cfg.model_type == "2dgs" and isinstance(self.cfg.strategy, DefaultStrategy):
            self.cfg.strategy.key_for_gradient = "gradient_2dgs"
        self.cfg.strategy.check_sanity(self.real_splats, self.optimizers)

        if isinstance(self.cfg.strategy, DefaultStrategy):
            self.strategy_state = self.cfg.strategy.initialize_state(
                scene_scale=self.scene_scale
            )
        elif isinstance(self.cfg.strategy, MCMCStrategy):
            self.strategy_state = self.cfg.strategy.initialize_state()
        else:
            assert_never(self.cfg.strategy)

        # The plane splats densify with the same strategy but a separate state.
        self.plane_strategy_state = None
        if self.mirror_params is not None and self.cfg.model_plane:
            self.cfg.strategy.check_sanity(self.plane_splats, self.plane_optimizers)
            if isinstance(self.cfg.strategy, DefaultStrategy):
                self.plane_strategy_state = self.cfg.strategy.initialize_state(
                    scene_scale=self.scene_scale
                )
            else:
                self.plane_strategy_state = self.cfg.strategy.initialize_state()

        # Compression Strategy
        self.compression_method = None
        if cfg.compression is not None:
            if cfg.compression == "png":
                self.compression_method = PngCompression()
            else:
                raise ValueError(f"Unknown compression strategy: {cfg.compression}")

        self.pose_optimizers = []
        if cfg.pose_opt:
            self.pose_adjust = CameraOptModule(len(self.trainset)).to(self.device)
            self.pose_adjust.zero_init()
            self.pose_optimizers = [
                torch.optim.Adam(
                    self.pose_adjust.parameters(),
                    lr=cfg.pose_opt_lr * math.sqrt(cfg.batch_size),
                    weight_decay=cfg.pose_opt_reg,
                )
            ]
            if world_size > 1:
                self.pose_adjust = DDP(self.pose_adjust)

        if cfg.pose_noise > 0.0:
            self.pose_perturb = CameraOptModule(len(self.trainset)).to(self.device)
            self.pose_perturb.random_init(cfg.pose_noise)
            if world_size > 1:
                self.pose_perturb = DDP(self.pose_perturb)

        self.app_optimizers = []
        if cfg.app_opt:
            assert feature_dim is not None
            self.app_module = AppearanceOptModule(
                len(self.trainset), feature_dim, cfg.app_embed_dim, cfg.sh_degree
            ).to(self.device)
            # initialize the last layer to be zero so that the initial output is zero.
            torch.nn.init.zeros_(self.app_module.color_head[-1].weight)
            torch.nn.init.zeros_(self.app_module.color_head[-1].bias)
            self.app_optimizers = [
                torch.optim.Adam(
                    self.app_module.embeds.parameters(),
                    lr=cfg.app_opt_lr * math.sqrt(cfg.batch_size) * 10.0,
                    weight_decay=cfg.app_opt_reg,
                ),
                torch.optim.Adam(
                    self.app_module.color_head.parameters(),
                    lr=cfg.app_opt_lr * math.sqrt(cfg.batch_size),
                ),
            ]
            if world_size > 1:
                self.app_module = DDP(self.app_module)

        self.bil_grid_optimizers = []
        if cfg.use_bilateral_grid:
            self.bil_grids = BilateralGrid(
                len(self.trainset),
                grid_X=cfg.bilateral_grid_shape[0],
                grid_Y=cfg.bilateral_grid_shape[1],
                grid_W=cfg.bilateral_grid_shape[2],
            ).to(self.device)
            self.bil_grid_optimizers = [
                torch.optim.Adam(
                    self.bil_grids.parameters(),
                    lr=2e-3 * math.sqrt(cfg.batch_size),
                    eps=1e-15,
                ),
            ]

        # Losses & Metrics.
        self.ssim = StructuralSimilarityIndexMeasure(data_range=1.0).to(self.device)
        self.psnr = PeakSignalNoiseRatio(data_range=1.0).to(self.device)

        if cfg.lpips_net == "alex":
            self.lpips = LearnedPerceptualImagePatchSimilarity(
                net_type="alex", normalize=True
            ).to(self.device)
        elif cfg.lpips_net == "vgg":
            # The 3DGS official repo uses lpips vgg, which is equivalent with the following:
            self.lpips = LearnedPerceptualImagePatchSimilarity(
                net_type="vgg", normalize=False
            ).to(self.device)
        else:
            raise ValueError(f"Unknown LPIPS network: {cfg.lpips_net}")

        # Viewer
        if not self.cfg.disable_viewer:
            self.server = viser.ViserServer(port=cfg.port, verbose=False)
            cams = []
            if cfg.show_viewer_cams:
                if cfg.clamp_viewer_cams_to > 0:
                    cams = [ c for i, c in enumerate(self.trainset) if i % (len(self.trainset) // cfg.clamp_viewer_cams_to) == 0 ]
                else:
                    cams = self.trainset
            self.viewer = GsplatViewer(
                server=self.server,
                render_fn=self._viewer_render_fn,
                output_dir=Path(cfg.result_dir),
                mode="training",
                scene_scale=self.scene_scale,
                trainset=cams,
            )

    def _setup_mirror(self):
        """Build the reflection transform, near-side sign and the
        mirror-surface (plane) Gaussians from the detected plane parameters."""
        device = self.device
        mp = self.mirror_params
        n = torch.tensor([mp["a"], mp["b"], mp["c"]], dtype=torch.float32, device=device)
        n = n / torch.norm(n)
        d = float(mp["d"])
        self.mirror_n = n
        self.mirror_d = d

        # Householder reflection S across n·x + d = 0 (improper, det = -1).
        # Mirrored camera-to-world is S @ c2w, i.e. the rasterizer viewmat V·S.
        S = torch.eye(4, device=device)
        S[:3, :3] = torch.eye(3, device=device) - 2.0 * torch.outer(n, n)
        S[:3, 3] = -2.0 * d * n
        self.S = S

        # Which side of the plane the cameras (and thus the real specimen) sit on.
        cam_centers = torch.from_numpy(
            self.parser.camtoworlds[:, :3, 3].copy()
        ).float().to(device)
        self.near_sign = float(torch.sign((cam_centers @ n + d).mean()).item()) or 1.0

        # Resolve the near-plane prune distance (same convention as at init):
        # <0 -> plane_prune_default_scale * scene_scale, 0 -> disabled, >0 -> verbatim.
        if self.cfg.plane_prune_dist == 0.0:
            self.plane_prune_dist = 0.0
        elif self.cfg.plane_prune_dist < 0.0:
            self.plane_prune_dist = self.cfg.plane_prune_default_scale * self.scene_scale
        else:
            self.plane_prune_dist = self.cfg.plane_prune_dist

        # Mirror-surface (plane) Gaussians (skipped for the no-plane variant).
        if self.cfg.model_plane:
            self._init_plane_splats()
            n_plane = len(self.plane_splats["means"])
        else:
            self.plane_splats = None
            self.plane_optimizers = {}
            n_plane = 0
        print(
            f"Mirror plane n={[round(v, 4) for v in n.tolist()]}, d={d:.4f}, "
            f"near_sign={self.near_sign:+.0f}, plane splats={n_plane}"
        )

    def _init_plane_splats(self):
        """Lay out a grid of broad, thin, neutral-gray Gaussians on the plane,
        spanning the specimen's projected extent."""
        cfg = self.cfg
        device = self.device
        n = self.mirror_n
        d = self.mirror_d
        p0 = -d * n  # a point on the plane

        # Orthonormal in-plane basis (u, v).
        a = torch.tensor([1.0, 0.0, 0.0], device=device)
        if torch.abs(torch.dot(a, n)) > 0.9:
            a = torch.tensor([0.0, 1.0, 0.0], device=device)
        u = a - torch.dot(a, n) * n
        u = u / torch.norm(u)
        v = torch.cross(n, u)
        v = v / torch.norm(v)

        # Extent from the butterfly's projection onto the plane.
        pts = self.real_splats["means"].detach()

        def _span(c):
            mid = 0.5 * (c.min() + c.max())
            half = 0.5 * (c.max() - c.min()) * cfg.plane_extent_scale
            return mid, half

        mid_u, half_u = _span(pts @ u)
        mid_v, half_v = _span(pts @ v)

        G = max(int(round(cfg.plane_num_pts ** 0.5)), 2)
        us = torch.linspace(mid_u - half_u, mid_u + half_u, G, device=device)
        vs = torch.linspace(mid_v - half_v, mid_v + half_v, G, device=device)
        gu, gv = torch.meshgrid(us, vs, indexing="ij")
        means = p0[None, :] + gu.reshape(-1, 1) * u[None, :] + gv.reshape(-1, 1) * v[None, :]
        Np = means.shape[0]

        spacing = max((2 * half_u / (G - 1)).item(), (2 * half_v / (G - 1)).item())
        scale_uv = spacing * 1.5  # broad, overlapping in-plane
        scale_n = scale_uv * 0.05  # thin along the normal (a surface, not a slab)
        scales = torch.log(
            torch.tensor([scale_uv, scale_uv, scale_n], device=device)
        )[None, :].repeat(Np, 1)

        # One shared rotation with local axes (u, v, n): the thin axis (local z,
        # scale index 2) is the normal. Re-enforced each step in
        # _project_plane_to_surface so the surface stays flat under optimization.
        quat = self._rotmat_to_quat(torch.stack([u, v, n], dim=1))
        quats = quat[None, :].repeat(Np, 1)
        self._plane_quat = quat            # canonical flat orientation
        self._plane_flat_ratio = scale_n / scale_uv  # normal:in-plane thickness

        opacities = torch.logit(torch.full((Np,), 0.8, device=device))

        K = (cfg.plane_sh_degree + 1) ** 2
        sh0 = torch.zeros((Np, 1, 3), device=device)
        sh0[:, 0, :] = rgb_to_sh(torch.full((Np, 3), 0.5, device=device))
        shN = torch.zeros((Np, K - 1, 3), device=device)

        params = [
            ("means", means, cfg.means_lr * self.scene_scale),
            ("scales", scales, cfg.scales_lr),
            ("quats", quats, cfg.quats_lr),
            ("opacities", opacities, cfg.opacities_lr),
            ("sh0", sh0, cfg.sh0_lr),
            ("shN", shN, cfg.shN_lr),
        ]
        self.plane_splats = torch.nn.ParameterDict(
            {k: torch.nn.Parameter(val) for k, val, _ in params}
        ).to(device)
        # Keyed by param name so the densification strategy's ops can update the
        # plane splats + their optimizer state (the plane densifies like the
        # butterfly, with its own strategy state).
        self.plane_optimizers = {
            k: torch.optim.Adam(
                [{"params": self.plane_splats[k], "lr": lr, "name": k}], eps=1e-15
            )
            for k, _, lr in params
        }

    @staticmethod
    def _rotmat_to_quat(m: Tensor) -> Tensor:
        """3x3 rotation matrix -> [w, x, y, z] quaternion."""
        tr = m[0, 0] + m[1, 1] + m[2, 2]
        if tr > 0:
            s = torch.sqrt(tr + 1.0) * 2
            w, x, y, z = 0.25 * s, (m[2, 1] - m[1, 2]) / s, (m[0, 2] - m[2, 0]) / s, (m[1, 0] - m[0, 1]) / s
        elif (m[0, 0] > m[1, 1]) and (m[0, 0] > m[2, 2]):
            s = torch.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2]) * 2
            w, x, y, z = (m[2, 1] - m[1, 2]) / s, 0.25 * s, (m[0, 1] + m[1, 0]) / s, (m[0, 2] + m[2, 0]) / s
        elif m[1, 1] > m[2, 2]:
            s = torch.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2]) * 2
            w, x, y, z = (m[0, 2] - m[2, 0]) / s, (m[0, 1] + m[1, 0]) / s, 0.25 * s, (m[1, 2] + m[2, 1]) / s
        else:
            s = torch.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1]) * 2
            w, x, y, z = (m[1, 0] - m[0, 1]) / s, (m[0, 2] + m[2, 0]) / s, (m[1, 2] + m[2, 1]) / s, 0.25 * s
        return torch.stack([w, x, y, z])

    def _rasterize(
        self,
        splats: torch.nn.ParameterDict,
        camtoworlds: Tensor,
        Ks: Tensor,
        width: int,
        height: int,
        masks: Optional[Tensor] = None,
        rasterize_mode: Optional[Literal["classic", "antialiased"]] = None,
        camera_model: Optional[Literal["pinhole", "ortho", "fisheye"]] = None,
        backgrounds: Optional[Tensor] = None,
        distloss: bool = False,
        **kwargs,
    ) -> Tuple[Tensor, Tensor, Dict]:
        """Rasterize a single given Gaussian set. For model_type=="2dgs" the
        per-pixel distortion map is stashed in info["render_distort"] (only
        meaningful when `distloss` is True)."""
        means = splats["means"]  # [N, 3]
        quats = splats["quats"]  # [N, 4]
        scales = torch.exp(splats["scales"])  # [N, 3]
        opacities = torch.sigmoid(splats["opacities"])  # [N,]

        image_ids = kwargs.pop("image_ids", None)
        if self.cfg.app_opt and "features" in splats:
            colors = self.app_module(
                features=splats["features"],
                embed_ids=image_ids,
                dirs=means[None, :, :] - camtoworlds[:, None, :3, 3],
                sh_degree=kwargs.pop("sh_degree", self.cfg.sh_degree),
            )
            colors = torch.sigmoid(colors + splats["colors"])
        else:
            colors = torch.cat([splats["sh0"], splats["shN"]], 1)  # [N, K, 3]

        absgrad = (
            self.cfg.strategy.absgrad
            if isinstance(self.cfg.strategy, DefaultStrategy)
            else False
        )
        if self.cfg.model_type == "2dgs":
            if distloss:
                # The distortion loss requires a depth render mode; we drop the
                # appended depth channel from the color below.
                kwargs["render_mode"] = "RGB+ED"
            (
                render_colors, render_alphas, render_normals,
                normals_from_depth, render_distort, render_median, info,
            ) = rasterization_2dgs(
                means=means,
                quats=quats,
                scales=scales,
                opacities=opacities,
                colors=colors,
                viewmats=torch.linalg.inv(camtoworlds),  # [C, 4, 4]
                Ks=Ks,  # [C, 3, 3]
                width=width,
                height=height,
                packed=self.cfg.packed,
                absgrad=absgrad,
                sparse_grad=self.cfg.sparse_grad,
                backgrounds=backgrounds,
                distloss=distloss,
                **kwargs,
            )
            info["render_distort"] = render_distort
            if distloss:
                render_colors = render_colors[..., :3]  # drop the depth channel
        else:
            if rasterize_mode is None:
                rasterize_mode = "antialiased" if self.cfg.antialiased else "classic"
            if camera_model is None:
                camera_model = self.cfg.camera_model
            render_colors, render_alphas, info = rasterization(
                means=means,
                quats=quats,
                scales=scales,
                opacities=opacities,
                colors=colors,
                viewmats=torch.linalg.inv(camtoworlds),  # [C, 4, 4]
                Ks=Ks,  # [C, 3, 3]
                width=width,
                height=height,
                packed=self.cfg.packed,
                absgrad=absgrad,
                sparse_grad=self.cfg.sparse_grad,
                rasterize_mode=rasterize_mode,
                distributed=self.world_size > 1,
                camera_model=camera_model,
                with_ut=self.cfg.with_ut,
                with_eval3d=self.cfg.with_eval3d,
                backgrounds=backgrounds,
                **kwargs,
            )
        if masks is not None:
            render_colors[~masks] = 0
        return render_colors, render_alphas, info

    def rasterize_splats(
        self, camtoworlds: Tensor, Ks: Tensor, width: int, height: int, **kwargs
    ) -> Tuple[Tensor, Tensor, Dict]:
        """Render the butterfly Gaussians (the default single set)."""
        return self._rasterize(self.real_splats, camtoworlds, Ks, width, height, **kwargs)

    def render_composite(
        self,
        camtoworlds: Tensor,
        Ks: Tensor,
        width: int,
        height: int,
        sh_degree: int,
        bg_color: Tensor,
        include_plane: bool = True,
        include_reflection: bool = True,
        distloss: bool = False,
        **kwargs,
    ) -> Dict[str, Tensor]:
        """Three-pass mirror composite.

        Pass A: direct butterfly. Pass B: reflected butterfly via the mirrored
        camera V·S (gsplat applies the reflected linear part to the covariance
        and evaluates SH from the mirrored camera center automatically). Pass C:
        the mirror surface. Each pass is rendered premultiplied (over black);
        `bg_color` is the base layer seen only through the transparent parts of
        the mirror surface — pass a random color during training so the scene is
        forced to explain every pixel via the butterfly or the mirror.
        """
        device = self.device
        C = camtoworlds.shape[0]
        black = torch.zeros((C, 3), device=device)
        kwargs.pop("render_mode", None)
        kwargs.pop("masks", None)
        kwargs.pop("backgrounds", None)

        # Pass A — direct (real) butterfly.
        render_real_color, render_real_alpha, info_real = self._rasterize(
            self.real_splats, camtoworlds, Ks, width, height,
            sh_degree=sh_degree, backgrounds=black, **kwargs,
        )

        # Pass B — reflected butterfly (mirrored camera). The distortion loss is
        # computed here (not Pass A): the reflection-dimmer floaters carry real
        # weight only in the reflected view, so this is where distortion bites.
        render_distort = None
        if include_reflection:
            c2w_mirror = self.S.unsqueeze(0) @ camtoworlds
            render_reflected_color, render_reflected_alpha, info_reflected = self._rasterize(
                self.real_splats, c2w_mirror, Ks, width, height,
                sh_degree=sh_degree, backgrounds=black, distloss=distloss, **kwargs,
            )
            render_distort = info_reflected.get("render_distort")
        else:
            render_reflected_color = torch.zeros_like(render_real_color)
            render_reflected_alpha = torch.zeros_like(render_real_alpha)

        # Pass C — mirror surface (plane).
        info_plane = None
        if include_plane:
            render_plane_color, render_plane_alpha, info_plane = self._rasterize(
                self.plane_splats, camtoworlds, Ks, width, height,
                sh_degree=self.cfg.plane_sh_degree, backgrounds=black, **kwargs,
            )
        else:
            render_plane_color = torch.zeros_like(render_real_color)
            render_plane_alpha = torch.zeros_like(render_real_alpha)

        # Composite: bg -> mirror surface -> reflection (confined) -> real.
        bg = bg_color.to(device).reshape(-1, 1, 1, 3)
        conf = render_plane_alpha if include_plane else torch.ones_like(render_plane_alpha)
        reflected_c = conf * render_reflected_color
        reflected_a = conf * render_reflected_alpha
        over_plane = render_plane_color + (1.0 - render_plane_alpha) * bg
        over_reflection = reflected_c + (1.0 - reflected_a) * over_plane
        rendered_image = render_real_color + (1.0 - render_real_alpha) * over_reflection

        return {
            "rendered_image": rendered_image,
            "info_real": info_real,
            "info_plane": info_plane,
            "render_distort": render_distort,
            "render_real_color": render_real_color, "render_real_alpha": render_real_alpha,
            "render_reflected_color": render_reflected_color, "render_reflected_alpha": render_reflected_alpha,
            "render_plane_color": render_plane_color, "render_plane_alpha": render_plane_alpha,
        }

    @torch.no_grad()
    def _prune_butterfly(self, mask: Tensor):
        """Remove butterfly Gaussians where `mask` is True, keeping params,
        optimizers and per-Gaussian strategy state in sync."""
        if not bool(mask.any()):
            return
        N = mask.shape[0]
        sel = torch.where(~mask)[0]

        def param_fn(name, p):
            return torch.nn.Parameter(p[sel], requires_grad=p.requires_grad)

        def optimizer_fn(key, v):
            return v[sel]

        _update_param_with_optimizer(param_fn, optimizer_fn, self.real_splats, self.optimizers)
        # Update only per-Gaussian strategy state (skip e.g. MCMC's [n_max,n_max] binoms).
        for k, val in self.strategy_state.items():
            if isinstance(val, torch.Tensor) and val.dim() >= 1 and val.shape[0] == N:
                self.strategy_state[k] = val[sel]

    @torch.no_grad()
    def _prune_farside(self):
        """Remove butterfly Gaussians on the far (mirror) side of the plane. The
        real butterfly is entirely on the camera side; anything behind the mirror
        is a stray copy now that the reflection is rendered analytically."""
        signed = (self.real_splats["means"] @ self.mirror_n + self.mirror_d) * self.near_sign
        self._prune_butterfly(signed < 0.0)

    @torch.no_grad()
    def _prune_near_plane(self):
        """Remove butterfly Gaussians hugging the mirror plane — the
        'plane-pretenders' that densification keeps regrowing in front of the
        learned mirror surface, occluding it. Uses the same threshold as init."""
        if self.plane_prune_dist <= 0.0:
            return
        dist = (self.real_splats["means"] @ self.mirror_n + self.mirror_d).abs()
        self._prune_butterfly(dist < self.plane_prune_dist)

    @torch.no_grad()
    def _prune_low_opacity(self, thresh: float):
        """One-shot: delete butterfly Gaussians with opacity below `thresh`.
        Used at end of training when the strategy's own pruning was disabled."""
        mask = torch.sigmoid(self.real_splats["opacities"]) < thresh
        n_before = self.real_splats["means"].shape[0]
        self._prune_butterfly(mask)
        print(f"Final low-opacity prune (<{thresh}): removed "
              f"{n_before - self.real_splats['means'].shape[0]} / {n_before} butterfly splats.")

    @torch.no_grad()
    def _project_plane_to_surface(self):
        """Hard-constrain the plane Gaussians to a flat mirror surface. Projecting means onto the plane only pins the centers;
        the splats stay flat only if their shape is pinned too, else they grow /
        rotate along the normal and render as spikes. Runs after the optimizer +
        plane densification so relocated/split splats are snapped as well."""
        p = self.plane_splats
        # 1) Means onto the plane.
        mu = p["means"]
        signed = mu @ self.mirror_n + self.mirror_d  # [Np]
        mu.sub_(signed[:, None] * self.mirror_n[None, :])
        # 2) Orientation: thin axis (local z) back onto the plane normal.
        p["quats"].data[:] = self._plane_quat
        # 3) Normal-axis scale: thin, tracking the in-plane size so it stays flat
        #    as densification shrinks the splats.
        s = p["scales"].data
        s_uv = torch.exp(s[:, :2]).amin(dim=1)  # [Np], current in-plane extent
        s[:, 2] = torch.log(self._plane_flat_ratio * s_uv)

    def train(self):
        cfg = self.cfg
        device = self.device
        world_rank = self.world_rank
        world_size = self.world_size

        # Dump cfg.
        if world_rank == 0:
            with open(f"{cfg.result_dir}/cfg.yml", "w") as f:
                yaml.dump(vars(cfg), f)

        max_steps = cfg.max_steps
        init_step = 0

        schedulers = [
            # means has a learning rate schedule, that end at 0.01 of the initial value
            torch.optim.lr_scheduler.ExponentialLR(
                self.optimizers["means"], gamma=0.01 ** (1.0 / max_steps)
            ),
        ]
        if cfg.pose_opt:
            # pose optimization has a learning rate schedule
            schedulers.append(
                torch.optim.lr_scheduler.ExponentialLR(
                    self.pose_optimizers[0], gamma=0.01 ** (1.0 / max_steps)
                )
            )
        if cfg.use_bilateral_grid:
            # bilateral grid has a learning rate schedule. Linear warmup for 1000 steps.
            schedulers.append(
                torch.optim.lr_scheduler.ChainedScheduler(
                    [
                        torch.optim.lr_scheduler.LinearLR(
                            self.bil_grid_optimizers[0],
                            start_factor=0.01,
                            total_iters=1000,
                        ),
                        torch.optim.lr_scheduler.ExponentialLR(
                            self.bil_grid_optimizers[0], gamma=0.01 ** (1.0 / max_steps)
                        ),
                    ]
                )
            )

        # Fixed generator so the shuffled training order is identical across
        # runs of the same config (isolated from the global RNG stream, which
        # other per-step randomness advances).
        trainloader = torch.utils.data.DataLoader(
            self.trainset,
            batch_size=cfg.batch_size,
            shuffle=True,
            num_workers=4,
            persistent_workers=True,
            pin_memory=True,
            generator=torch.Generator().manual_seed(42 + self.world_rank),
        )
        trainloader_iter = iter(trainloader)

        # Training loop.
        global_tic = time.time()
        pbar = tqdm.tqdm(range(init_step, max_steps))
        for step in pbar:
            if not cfg.disable_viewer:
                while self.viewer.state == "paused":
                    time.sleep(0.01)
                self.viewer.lock.acquire()
                tic = time.time()

            # End of training: drop dead (near-zero opacity) butterfly splats once,
            # before the final forward + checkpoint/ply save so they reflect it.
            if step == max_steps - 1 and cfg.final_prune_opacity > 0.0:
                self._prune_low_opacity(cfg.final_prune_opacity)

            try:
                data = next(trainloader_iter)
            except StopIteration:
                trainloader_iter = iter(trainloader)
                data = next(trainloader_iter)

            camtoworlds = camtoworlds_gt = data["camtoworld"].to(device)  # [1, 4, 4]
            Ks = data["K"].to(device)  # [1, 3, 3]
            pixels = data["image"].to(device) / 255.0  # [1, H, W, 3]
            num_train_rays_per_step = (
                pixels.shape[0] * pixels.shape[1] * pixels.shape[2]
            )
            image_ids = data["image_id"].to(device)
            masks = data["mask"].to(device) if "mask" in data else None  # [1, H, W]

            if cfg.depth_loss:
                points = data["points"].to(device)  # [1, M, 2]
                depths_gt = data["depths"].to(device)  # [1, M]

            height, width = pixels.shape[1:3]

            if cfg.pose_noise:
                camtoworlds = self.pose_perturb(camtoworlds, image_ids)

            if cfg.pose_opt:
                camtoworlds = self.pose_adjust(camtoworlds, image_ids)

            # sh schedule
            sh_degree_to_use = min(step // cfg.sh_degree_interval, cfg.sh_degree)

            # forward
            passes = None
            if self.mirror_params is not None:
                # Mirror compositing. Random per-iteration background (bg_color)
                # so every pixel must be explained by the butterfly or the mirror.
                bg_color = torch.rand(camtoworlds.shape[0], 3, device=device)
                # Distortion loss is 2dgs-only (the 3dgs rasterizer does not
                # emit render_distort in the stock gsplat build).
                want_distloss = (
                    cfg.model_type == "2dgs"
                    and cfg.dist_lambda > 0.0
                    and step > cfg.dist_start_iter
                )
                passes = self.render_composite(
                    camtoworlds=camtoworlds,
                    Ks=Ks,
                    width=width,
                    height=height,
                    sh_degree=sh_degree_to_use,
                    bg_color=bg_color,
                    include_plane=cfg.model_plane,
                    near_plane=cfg.near_plane,
                    far_plane=cfg.far_plane,
                    image_ids=image_ids,
                    distloss=want_distloss,
                )
                colors, depths = passes["rendered_image"], None
                alphas = passes["render_real_alpha"]
                info = passes["info_real"]
                info_plane = passes["info_plane"]
            else:
                info_plane = None
                renders, alphas, info = self.rasterize_splats(
                    camtoworlds=camtoworlds,
                    Ks=Ks,
                    width=width,
                    height=height,
                    sh_degree=sh_degree_to_use,
                    near_plane=cfg.near_plane,
                    far_plane=cfg.far_plane,
                    image_ids=image_ids,
                    render_mode="RGB+ED" if cfg.depth_loss else "RGB",
                    masks=masks,
                )
                if renders.shape[-1] == 4:
                    colors, depths = renders[..., 0:3], renders[..., 3:4]
                else:
                    colors, depths = renders, None

            if cfg.use_bilateral_grid:
                grid_y, grid_x = torch.meshgrid(
                    (torch.arange(height, device=self.device) + 0.5) / height,
                    (torch.arange(width, device=self.device) + 0.5) / width,
                    indexing="ij",
                )
                grid_xy = torch.stack([grid_x, grid_y], dim=-1).unsqueeze(0)
                colors = slice(
                    self.bil_grids,
                    grid_xy.expand(colors.shape[0], -1, -1, -1),
                    colors,
                    image_ids.unsqueeze(-1),
                )["rgb"]

            if cfg.random_bkgd and self.mirror_params is None:
                bkgd = torch.rand(1, 3, device=device)
                colors = colors + bkgd * (1.0 - alphas)

            if cfg.butterfly_densify:
                self.cfg.strategy.step_pre_backward(
                    params=self.real_splats,
                    optimizers=self.optimizers,
                    state=self.strategy_state,
                    step=step,
                    info=info,
                )
            if info_plane is not None:
                self.cfg.strategy.step_pre_backward(
                    params=self.plane_splats,
                    optimizers=self.plane_optimizers,
                    state=self.plane_strategy_state,
                    step=step,
                    info=info_plane,
                )

            # loss
            l1loss = F.l1_loss(colors, pixels)
            ssimloss = 1.0 - self.ssim(colors.permute(0, 3, 1, 2), pixels.permute(0, 3, 1, 2))
            loss = l1loss * (1.0 - cfg.ssim_lambda) + ssimloss * cfg.ssim_lambda
            if cfg.depth_loss:
                if "depth" in data:
                    disp_gt = data["depth"]
                    disp = torch.where(depths > 0.0, 1.0 / depths, torch.zeros_like(depths))
                else:
                    # query depths from depth map
                    points = torch.stack(
                        [
                            points[:, :, 0] / (width - 1) * 2 - 1,
                            points[:, :, 1] / (height - 1) * 2 - 1,
                        ],
                        dim=-1,
                    )  # normalize to [-1, 1]
                    grid = points.unsqueeze(2)  # [1, M, 1, 2]
                    depths = F.grid_sample(
                        depths.permute(0, 3, 1, 2), grid, align_corners=True
                    )  # [1, 1, M, 1]
                    depths = depths.squeeze(3).squeeze(1)  # [1, M]
                    # calculate loss in disparity space

                    disp = torch.where(depths > 0.0, 1.0 / depths, torch.zeros_like(depths))
                    disp_gt = 1.0 / depths_gt  # [1, M]
                    
                depthloss = F.l1_loss(disp, disp_gt) * self.scene_scale
                loss += depthloss * cfg.depth_lambda

            if cfg.use_bilateral_grid:
                tvloss = 10 * total_variation_loss(self.bil_grids.grids)
                loss += tvloss

            # regularizations
            if cfg.opacity_reg > 0.0:
                loss += cfg.opacity_reg * torch.sigmoid(self.real_splats["opacities"]).mean()
            if cfg.scale_reg > 0.0:
                loss += cfg.scale_reg * torch.exp(self.real_splats["scales"]).mean()

            # mirror-specific regularizers
            if self.mirror_params is not None:
                # Butterfly rendered-alpha binary entropy: the direct (butterfly-
                # only) pass's per-pixel accumulated alpha is pushed toward {0,1}.
                if cfg.alpha_entropy_lambda > 0.0 and step >= cfg.alpha_entropy_start:
                    a = alphas.clamp(1e-6, 1 - 1e-6)
                    alpha_entropy = -(a * torch.log(a) + (1 - a) * torch.log(1 - a)).mean()
                    loss += cfg.alpha_entropy_lambda * alpha_entropy
                # 2DGS distortion loss: concentrate per-ray weight into one
                # surface. The distortion map comes from the REFLECTED pass
                # (see Pass B above), where floaters accumulate the most.
                if want_distloss and passes["render_distort"] is not None:
                    loss += cfg.dist_lambda * passes["render_distort"].mean()

            loss.backward()

            # In mirror mode `info` already comes from the direct butterfly pass
            # (Pass A) over the same N Gaussians as `self.real_splats`, so it is used
            # for densification as-is; the reflected/plane passes drive gradients
            # on the parameters but not the densification statistics.

            desc = f"loss={loss.item():.3f}| " f"sh degree={sh_degree_to_use}| "
            if cfg.depth_loss:
                desc += f"depth loss={depthloss.item():.6f}| "
            if cfg.pose_opt and cfg.pose_noise:
                # monitor the pose error if we inject noise
                pose_err = F.l1_loss(camtoworlds_gt, camtoworlds)
                desc += f"pose err={pose_err.item():.6f}| "
            pbar.set_description(desc)

            if world_rank == 0 and cfg.tb_every > 0 and step % cfg.tb_every == 0:
                mem = torch.cuda.max_memory_allocated() / 1024**3
                self.writer.add_scalar("train/loss", loss.item(), step)
                self.writer.add_scalar("train/l1loss", l1loss.item(), step)
                self.writer.add_scalar("train/ssimloss", ssimloss.item(), step)
                self.writer.add_scalar("train/num_GS", len(self.real_splats["means"]), step)
                self.writer.add_scalar("train/mem", mem, step)
                if cfg.depth_loss:
                    self.writer.add_scalar("train/depthloss", depthloss.item(), step)
                if cfg.use_bilateral_grid:
                    self.writer.add_scalar("train/tvloss", tvloss.item(), step)
                if cfg.tb_save_image:
                    canvas = torch.cat([pixels, colors], dim=2).detach().cpu().numpy()
                    canvas = canvas.reshape(-1, *canvas.shape[2:])
                    self.writer.add_image("train/render", canvas, step)
                self.writer.flush()

            # save checkpoint before updating the model
            if step in [i - 1 for i in cfg.save_steps] or step == max_steps - 1:
                mem = torch.cuda.max_memory_allocated() / 1024**3
                stats = {
                    "mem": mem,
                    "ellipse_time": time.time() - global_tic,
                    "num_GS": len(self.real_splats["means"]),
                }
                print("Step: ", step, stats)
                with open(
                    f"{self.stats_dir}/train_step{step:04d}_rank{self.world_rank}.json",
                    "w",
                ) as f:
                    json.dump(stats, f)
                data = {"step": step, "splats": self.real_splats.state_dict()}
                if self.mirror_params is not None:
                    if self.plane_splats is not None:
                        data["plane_splats"] = self.plane_splats.state_dict()
                    data["mirror_params"] = self.mirror_params
                if cfg.pose_opt:
                    if world_size > 1:
                        data["pose_adjust"] = self.pose_adjust.module.state_dict()
                    else:
                        data["pose_adjust"] = self.pose_adjust.state_dict()
                if cfg.app_opt:
                    if world_size > 1:
                        data["app_module"] = self.app_module.module.state_dict()
                    else:
                        data["app_module"] = self.app_module.state_dict()
                torch.save(
                    data, f"{self.ckpt_dir}/ckpt_{step}_rank{self.world_rank}.pt"
                )
            if (
                step in [i - 1 for i in cfg.ply_steps] or step == max_steps - 1
            ) and cfg.save_ply:

                if self.cfg.app_opt:
                    # eval at origin to bake the appeareance into the colors
                    rgb = self.app_module(
                        features=self.real_splats["features"],
                        embed_ids=None,
                        dirs=torch.zeros_like(self.real_splats["means"][None, :, :]),
                        sh_degree=sh_degree_to_use,
                    )
                    rgb = rgb + self.real_splats["colors"]
                    rgb = torch.sigmoid(rgb).squeeze(0).unsqueeze(1)
                    sh0 = rgb_to_sh(rgb)
                    shN = torch.empty([sh0.shape[0], 0, 3], device=sh0.device)
                else:
                    sh0 = self.real_splats["sh0"]
                    shN = self.real_splats["shN"]

                means = self.real_splats["means"]
                scales = self.real_splats["scales"]
                quats = self.real_splats["quats"]
                opacities = self.real_splats["opacities"]
                export_splats(
                    means=means,
                    scales=scales,
                    quats=quats,
                    opacities=opacities,
                    sh0=sh0,
                    shN=shN,
                    format="ply",
                    save_to=f"{self.ply_dir}/point_cloud_{step}.ply",
                )

            # Turn Gradients into Sparse Tensor before running optimizer
            if cfg.sparse_grad:
                assert cfg.packed, "Sparse gradients only work with packed mode."
                gaussian_ids = info["gaussian_ids"]
                for k in self.real_splats.keys():
                    grad = self.real_splats[k].grad
                    if grad is None or grad.is_sparse:
                        continue
                    self.real_splats[k].grad = torch.sparse_coo_tensor(
                        indices=gaussian_ids[None],  # [1, nnz]
                        values=grad[gaussian_ids],  # [nnz, ...]
                        size=self.real_splats[k].size(),  # [N, ...]
                        is_coalesced=len(Ks) == 1,
                    )

            if cfg.visible_adam:
                gaussian_cnt = self.real_splats.means.shape[0]
                if cfg.packed:
                    visibility_mask = torch.zeros_like(
                        self.real_splats["opacities"], dtype=bool
                    )
                    visibility_mask.scatter_(0, info["gaussian_ids"], 1)
                else:
                    visibility_mask = (info["radii"] > 0).all(-1).any(0)

            # optimize
            for optimizer in self.optimizers.values():
                if cfg.visible_adam:
                    optimizer.step(visibility_mask)
                else:
                    optimizer.step()
                optimizer.zero_grad(set_to_none=True)
            for optimizer in self.pose_optimizers:
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
            for optimizer in self.app_optimizers:
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
            for optimizer in self.bil_grid_optimizers:
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
            if self.mirror_params is not None:
                for optimizer in self.plane_optimizers.values():
                    optimizer.step()
                    optimizer.zero_grad(set_to_none=True)
            for scheduler in schedulers:
                scheduler.step()

            # Run post-backward steps after backward and optimizer
            if not cfg.butterfly_densify:
                pass
            elif isinstance(self.cfg.strategy, DefaultStrategy):
                self.cfg.strategy.step_post_backward(
                    params=self.real_splats,
                    optimizers=self.optimizers,
                    state=self.strategy_state,
                    step=step,
                    info=info,
                    packed=cfg.packed,
                )
            elif isinstance(self.cfg.strategy, MCMCStrategy):
                self.cfg.strategy.step_post_backward(
                    params=self.real_splats,
                    optimizers=self.optimizers,
                    state=self.strategy_state,
                    step=step,
                    info=info,
                    lr=schedulers[0].get_last_lr()[0],
                )
            else:
                assert_never(self.cfg.strategy)

            # Same densification for the plane splats (separate state/optimizers).
            if info_plane is not None:
                if isinstance(self.cfg.strategy, DefaultStrategy):
                    self.cfg.strategy.step_post_backward(
                        params=self.plane_splats,
                        optimizers=self.plane_optimizers,
                        state=self.plane_strategy_state,
                        step=step,
                        info=info_plane,
                        packed=cfg.packed,
                    )
                else:
                    self.cfg.strategy.step_post_backward(
                        params=self.plane_splats,
                        optimizers=self.plane_optimizers,
                        state=self.plane_strategy_state,
                        step=step,
                        info=info_plane,
                        lr=self.plane_optimizers["means"].param_groups[0]["lr"],
                    )

            if self.mirror_params is not None:
                # Snap the plane Gaussians back onto the mirror surface every step.
                if self.plane_splats is not None:
                    self._project_plane_to_surface()
                # Keep the butterfly off the plane: prune Gaussians hugging the
                # plane and those behind it.
                if cfg.farside_prune_every > 0 and step % cfg.farside_prune_every == 0:
                    self._prune_near_plane()
                    self._prune_farside()

            # eval the full set
            if step in [i - 1 for i in cfg.eval_steps]:
                self.eval(step)
                self.render_traj(step)

            # run compression
            if cfg.compression is not None and step in [i - 1 for i in cfg.eval_steps]:
                self.run_compression(step=step)

            if not cfg.disable_viewer:
                self.viewer.lock.release()
                num_train_steps_per_sec = 1.0 / (max(time.time() - tic, 1e-10))
                num_train_rays_per_sec = (
                    num_train_rays_per_step * num_train_steps_per_sec
                )
                # Update the viewer state.
                self.viewer.render_tab_state.num_train_rays_per_sec = (
                    num_train_rays_per_sec
                )
                # Update the scene.
                self.viewer.update(step, num_train_rays_per_step)

    @torch.no_grad()
    def eval(self, step: int, stage: str = "val"):
        """Entry for evaluation."""
        # No held-out validation image: nothing to eval.
        if len(self.valset) == 0:
            print("Skipping evaluation: no held-out validation image.")
            return
        print("Running evaluation...")
        cfg = self.cfg
        device = self.device
        world_rank = self.world_rank
        world_size = self.world_size

        valloader = torch.utils.data.DataLoader(
            self.valset, batch_size=1, shuffle=False, num_workers=1
        )
        ellipse_time = 0
        metrics = defaultdict(list)
        for i, data in enumerate(valloader):
            camtoworlds = data["camtoworld"].to(device)
            Ks = data["K"].to(device)
            pixels = data["image"].to(device) / 255.0
            masks = data["mask"].to(device) if "mask" in data else None
            height, width = pixels.shape[1:3]

            torch.cuda.synchronize()
            tic = time.time()
            if self.mirror_params is not None:
                colors = self.render_composite(
                    camtoworlds=camtoworlds,
                    Ks=Ks,
                    width=width,
                    height=height,
                    sh_degree=cfg.sh_degree,
                    bg_color=torch.zeros(1, 3, device=device),
                    include_plane=cfg.model_plane,
                    near_plane=cfg.near_plane,
                    far_plane=cfg.far_plane,
                )["rendered_image"]
            else:
                colors, _, _ = self.rasterize_splats(
                    camtoworlds=camtoworlds,
                    Ks=Ks,
                    width=width,
                    height=height,
                    sh_degree=cfg.sh_degree,
                    near_plane=cfg.near_plane,
                    far_plane=cfg.far_plane,
                )  # [1, H, W, 3]
            torch.cuda.synchronize()

            ellipse_time += max(time.time() - tic, 1e-10)

            colors = torch.clamp(colors, 0.0, 1.0)  # [1, H, W, 3], full render

            if world_rank == 0:
                image_name = data["image_name"][0]
                stem = os.path.splitext(image_name)[0]

                # Clean, full-frame, render-only PNG for later re-evaluation
                # (so the render can be re-scored later, masked and full-frame).
                render_np = (colors.squeeze(0).cpu().numpy() * 255).astype(np.uint8)
                imageio.imwrite(f"{self.render_dir}/{stem}.png", render_np)

                # GT | render side-by-side canvas for visual inspection.
                canvas = torch.cat([pixels, colors], dim=2).squeeze(0).cpu().numpy()
                imageio.imwrite(
                    f"{self.render_dir}/{stage}_step{step}_{i:04d}.png",
                    (canvas * 255).astype(np.uint8),
                )

                pixels_p = pixels.permute(0, 3, 1, 2)  # [1, 3, H, W]
                colors_p = colors.permute(0, 3, 1, 2)  # [1, 3, H, W]
                # Full-image metrics.
                metrics["psnr"].append(self.psnr(colors_p, pixels_p))
                metrics["ssim"].append(self.ssim(colors_p, pixels_p))
                metrics["lpips"].append(self.lpips(colors_p, pixels_p))
                # Masked metrics: white-composite everything outside the mask in
                # both render and GT, then score only the in-mask specimen.
                if masks is not None:
                    m = masks.unsqueeze(-1).float()  # [1, H, W, 1]
                    colors_m = (colors * m + (1.0 - m)).permute(0, 3, 1, 2)
                    pixels_m = (pixels * m + (1.0 - m)).permute(0, 3, 1, 2)
                    metrics["psnr_masked"].append(self.psnr(colors_m, pixels_m))
                    metrics["ssim_masked"].append(self.ssim(colors_m, pixels_m))
                    metrics["lpips_masked"].append(self.lpips(colors_m, pixels_m))
                if cfg.use_bilateral_grid:
                    cc_colors = color_correct(colors, pixels)
                    cc_colors_p = cc_colors.permute(0, 3, 1, 2)  # [1, 3, H, W]
                    metrics["cc_psnr"].append(self.psnr(cc_colors_p, pixels_p))
                    metrics["cc_ssim"].append(self.ssim(cc_colors_p, pixels_p))
                    metrics["cc_lpips"].append(self.lpips(cc_colors_p, pixels_p))

        if world_rank == 0:
            ellipse_time /= len(valloader)

            stats = {k: torch.stack(v).mean().item() for k, v in metrics.items()}
            stats.update(
                {
                    "ellipse_time": ellipse_time,
                    "num_GS": len(self.real_splats["means"]),
                }
            )
            if cfg.use_bilateral_grid:
                print(
                    f"PSNR: {stats['psnr']:.3f}, SSIM: {stats['ssim']:.4f}, LPIPS: {stats['lpips']:.3f} "
                    f"CC_PSNR: {stats['cc_psnr']:.3f}, CC_SSIM: {stats['cc_ssim']:.4f}, CC_LPIPS: {stats['cc_lpips']:.3f} "
                    f"Time: {stats['ellipse_time']:.3f}s/image "
                    f"Number of GS: {stats['num_GS']}"
                )
            else:
                masked = ""
                if "psnr_masked" in stats:
                    masked = (f" | Masked PSNR: {stats['psnr_masked']:.3f}, "
                              f"SSIM: {stats['ssim_masked']:.4f}, LPIPS: {stats['lpips_masked']:.3f}")
                print(
                    f"Full PSNR: {stats['psnr']:.3f}, SSIM: {stats['ssim']:.4f}, LPIPS: {stats['lpips']:.3f}{masked} "
                    f"Time: {stats['ellipse_time']:.3f}s/image "
                    f"Number of GS: {stats['num_GS']}"
                )
            # save stats as json
            with open(f"{self.stats_dir}/{stage}_step{step:04d}.json", "w") as f:
                json.dump(stats, f)
            # save stats to tensorboard
            for k, v in stats.items():
                self.writer.add_scalar(f"{stage}/{k}", v, step)
            self.writer.flush()

    @torch.no_grad()
    def render_traj(self, step: int):
        """Entry for trajectory rendering."""
        if self.cfg.disable_video:
            return
        print("Running trajectory rendering...")
        cfg = self.cfg
        device = self.device

        camtoworlds_all = self.parser.camtoworlds[5:-5]
        if cfg.render_traj_path == "interp":
            camtoworlds_all = generate_interpolated_path(
                camtoworlds_all, 1
            )  # [N, 3, 4]
        elif cfg.render_traj_path == "ellipse":
            height = camtoworlds_all[:, 2, 3].mean()
            camtoworlds_all = generate_ellipse_path_z(
                camtoworlds_all, height=height
            )  # [N, 3, 4]
        elif cfg.render_traj_path == "spiral":
            camtoworlds_all = generate_spiral_path(
                camtoworlds_all,
                bounds=self.parser.bounds * self.scene_scale,
                spiral_scale_r=self.parser.extconf["spiral_radius_scale"],
            )
        else:
            raise ValueError(
                f"Render trajectory type not supported: {cfg.render_traj_path}"
            )

        camtoworlds_all = np.concatenate(
            [
                camtoworlds_all,
                np.repeat(
                    np.array([[[0.0, 0.0, 0.0, 1.0]]]), len(camtoworlds_all), axis=0
                ),
            ],
            axis=1,
        )  # [N, 4, 4]

        camtoworlds_all = torch.from_numpy(camtoworlds_all).float().to(device)
        K = torch.from_numpy(list(self.parser.Ks_dict.values())[0]).float().to(device)
        width, height = list(self.parser.imsize_dict.values())[0]

        # save to video
        video_dir = f"{cfg.result_dir}/videos"
        os.makedirs(video_dir, exist_ok=True)
        writer = imageio.get_writer(f"{video_dir}/traj_{step}.mp4", fps=30)
        for i in tqdm.trange(len(camtoworlds_all), desc="Rendering trajectory"):
            camtoworlds = camtoworlds_all[i : i + 1]
            Ks = K[None]

            if self.mirror_params is not None:
                colors = self.render_composite(
                    camtoworlds=camtoworlds,
                    Ks=Ks,
                    width=width,
                    height=height,
                    sh_degree=cfg.sh_degree,
                    bg_color=torch.zeros(1, 3, device=device),
                    include_plane=cfg.model_plane,
                    near_plane=cfg.near_plane,
                    far_plane=cfg.far_plane,
                )["rendered_image"]
                colors = torch.clamp(colors, 0.0, 1.0)
                canvas_list = [colors]
            else:
                renders, _, _ = self.rasterize_splats(
                    camtoworlds=camtoworlds,
                    Ks=Ks,
                    width=width,
                    height=height,
                    sh_degree=cfg.sh_degree,
                    near_plane=cfg.near_plane,
                    far_plane=cfg.far_plane,
                    render_mode="RGB+ED",
                )  # [1, H, W, 4]
                colors = torch.clamp(renders[..., 0:3], 0.0, 1.0)  # [1, H, W, 3]
                depths = renders[..., 3:4]  # [1, H, W, 1]
                depths = (depths - depths.min()) / (depths.max() - depths.min())
                canvas_list = [colors, depths.repeat(1, 1, 1, 3)]

            # write images
            canvas = torch.cat(canvas_list, dim=2).squeeze(0).cpu().numpy()
            canvas = (canvas * 255).astype(np.uint8)
            writer.append_data(canvas)
        writer.close()
        print(f"Video saved to {video_dir}/traj_{step}.mp4")

    @torch.no_grad()
    def run_compression(self, step: int):
        """Entry for running compression."""
        print("Running compression...")
        world_rank = self.world_rank

        compress_dir = f"{cfg.result_dir}/compression/rank{world_rank}"
        os.makedirs(compress_dir, exist_ok=True)

        self.compression_method.compress(compress_dir, self.real_splats)

        # evaluate compression
        splats_c = self.compression_method.decompress(compress_dir)
        for k in splats_c.keys():
            self.real_splats[k].data = splats_c[k].to(self.device)
        self.eval(step=step, stage="compress")

    @torch.no_grad()
    def _viewer_render_fn(
        self, camera_state: CameraState, render_tab_state: RenderTabState
    ):
        assert isinstance(render_tab_state, GsplatRenderTabState)
        if render_tab_state.preview_render:
            width = render_tab_state.render_width
            height = render_tab_state.render_height
        else:
            width = render_tab_state.viewer_width
            height = render_tab_state.viewer_height
        c2w = camera_state.c2w
        K = camera_state.get_K((width, height))
        c2w = torch.from_numpy(c2w).float().to(self.device)
        K = torch.from_numpy(K).float().to(self.device)

        RENDER_MODE_MAP = {
            "rgb": "RGB",
            "depth(accumulated)": "D",
            "depth(expected)": "ED",
            "alpha": "RGB",
        }

        backgrounds = (
            torch.tensor([render_tab_state.backgrounds], device=self.device) / 255.0
        )
        if self.mirror_params is not None and render_tab_state.render_mode == "rgb":
            # Full mirror composite, with plane/reflection toggled from the GUI.
            out = self.render_composite(
                camtoworlds=c2w[None],
                Ks=K[None],
                width=width,
                height=height,
                sh_degree=min(render_tab_state.max_sh_degree, self.cfg.sh_degree),
                bg_color=backgrounds,
                include_plane=render_tab_state.show_plane and self.plane_splats is not None,
                include_reflection=render_tab_state.show_reflection,
                near_plane=render_tab_state.near_plane,
                far_plane=render_tab_state.far_plane,
                radius_clip=render_tab_state.radius_clip,
                eps2d=render_tab_state.eps2d,
                rasterize_mode=render_tab_state.rasterize_mode,
                camera_model=render_tab_state.camera_model,
            )
            render_colors, render_alphas, info = (
                out["rendered_image"],
                out["render_real_alpha"],
                out["info_real"],
            )
        else:
            render_colors, render_alphas, info = self.rasterize_splats(
                camtoworlds=c2w[None],
                Ks=K[None],
                width=width,
                height=height,
                sh_degree=min(render_tab_state.max_sh_degree, self.cfg.sh_degree),
                near_plane=render_tab_state.near_plane,
                far_plane=render_tab_state.far_plane,
                radius_clip=render_tab_state.radius_clip,
                eps2d=render_tab_state.eps2d,
                backgrounds=backgrounds,
                render_mode=RENDER_MODE_MAP[render_tab_state.render_mode],
                rasterize_mode=render_tab_state.rasterize_mode,
                camera_model=render_tab_state.camera_model,
            )  # [1, H, W, 3]
        render_tab_state.total_gs_count = len(self.real_splats["means"])
        render_tab_state.rendered_gs_count = (info["radii"] > 0).all(-1).sum().item()

        if render_tab_state.render_mode == "rgb":
            # colors represented with sh are not guranteed to be in [0, 1]
            render_colors = render_colors[0, ..., 0:3].clamp(0, 1)
            renders = render_colors.cpu().numpy()
        elif render_tab_state.render_mode in ["depth(accumulated)", "depth(expected)"]:
            # normalize depth to [0, 1]
            depth = render_colors[0, ..., 0:1]
            if render_tab_state.normalize_nearfar:
                near_plane = render_tab_state.near_plane
                far_plane = render_tab_state.far_plane
            else:
                near_plane = depth.min()
                far_plane = depth.max()
            depth_norm = (depth - near_plane) / (far_plane - near_plane + 1e-10)
            depth_norm = torch.clip(depth_norm, 0, 1)
            if render_tab_state.inverse:
                depth_norm = 1 - depth_norm
            renders = (
                apply_float_colormap(depth_norm, render_tab_state.colormap)
                .cpu()
                .numpy()
            )
        elif render_tab_state.render_mode == "alpha":
            alpha = render_alphas[0, ..., 0:1]
            if render_tab_state.inverse:
                alpha = 1 - alpha
            renders = (
                apply_float_colormap(alpha, render_tab_state.colormap).cpu().numpy()
            )
        return renders


def main(local_rank: int, world_rank, world_size: int, cfg: Config):
    if world_size > 1 and not cfg.disable_viewer:
        cfg.disable_viewer = True
        if world_rank == 0:
            print("Viewer is disabled in distributed training.")

    runner = Runner(local_rank, world_rank, world_size, cfg)

    if cfg.ckpt is not None:
        # run eval only
        ckpts = [
            torch.load(file, map_location=runner.device, weights_only=True)
            for file in cfg.ckpt
        ]
        for k in runner.real_splats.keys():
            runner.real_splats[k].data = torch.cat([ckpt["splats"][k] for ckpt in ckpts])
        if runner.mirror_params is not None and runner.plane_splats is not None and "plane_splats" in ckpts[0]:
            for k in runner.plane_splats.keys():
                runner.plane_splats[k].data = ckpts[0]["plane_splats"][k].to(runner.device)
        step = ckpts[0]["step"]
        runner.eval(step=step)
        runner.render_traj(step=step)
        if cfg.compression is not None:
            runner.run_compression(step=step)
    else:
        runner.train()

    if not cfg.disable_viewer:
        runner.viewer.complete()
        print("Viewer running... Ctrl+C to exit.")
        time.sleep(1000000)


if __name__ == "__main__":
    """
    Usage:

    ```bash
    # Single GPU training
    CUDA_VISIBLE_DEVICES=9 python -m examples.simple_trainer default

    # Distributed training on 4 GPUs: Effectively 4x batch size so run 4x less steps.
    CUDA_VISIBLE_DEVICES=0,1,2,3 python simple_trainer.py default --steps_scaler 0.25

    """

    # Config objects we can choose between.
    # Each is a tuple of (CLI description, config object).
    configs = {
        "default": (
            "Gaussian splatting training using densification heuristics from the original paper.",
            Config(
                strategy=DefaultStrategy(verbose=True),
            ),
        ),
        "mcmc": (
            "Gaussian splatting training using densification from the paper '3D Gaussian Splatting as Markov Chain Monte Carlo'.",
            Config(
                init_opa=0.5,
                init_scale=0.1,
                opacity_reg=0.01,
                scale_reg=0.01,
                strategy=MCMCStrategy(verbose=True),
            ),
        ),
    }
    cfg = tyro.extras.overridable_config_cli(configs)
    cfg.adjust_steps(cfg.steps_scaler)

    # Import BilateralGrid and related functions based on configuration
    if cfg.use_bilateral_grid or cfg.use_fused_bilagrid:
        if cfg.use_fused_bilagrid:
            cfg.use_bilateral_grid = True
            from fused_bilagrid import (
                BilateralGrid,
                color_correct,
                slice,
                total_variation_loss,
            )
        else:
            cfg.use_bilateral_grid = True
            from lib_bilagrid import (
                BilateralGrid,
                color_correct,
                slice,
                total_variation_loss,
            )

    # try import extra dependencies
    if cfg.compression == "png":
        try:
            import plas
            import torchpq
        except:
            raise ImportError(
                "To use PNG compression, you need to install "
                "torchpq (instruction at https://github.com/DeMoriarty/TorchPQ?tab=readme-ov-file#install) "
                "and plas (via 'pip install git+https://github.com/fraunhoferhhi/PLAS.git') "
            )

    if cfg.with_ut:
        assert cfg.with_eval3d, "Training with UT requires setting `with_eval3d` flag."

    cli(main, cfg, verbose=True)
