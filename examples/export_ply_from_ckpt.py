import argparse

import torch
import torch.nn.functional as F

from gsplat import export_splats
from gsplat.distributed import cli
import os

def main(local_rank: int, world_rank, world_size: int, args):
    torch.manual_seed(42)
    device = torch.device("cuda", local_rank)

    means, quats, scales, opacities, sh0, shN = [], [], [], [], [], []
    ckpt_path = args.ckpt
    ply_path = os.path.join(os.path.dirname(ckpt_path), os.pardir, "ply", "point_cloud.ply")
    ckpt = torch.load(ckpt_path, map_location=device)["splats"]
    means.append(ckpt["means"])
    quats.append(ckpt["quats"])
    scales.append(ckpt["scales"])
    opacities.append(ckpt["opacities"])
    sh0.append(ckpt["sh0"])
    shN.append(ckpt["shN"])
    means = torch.cat(means, dim=0)
    quats = torch.cat(quats, dim=0)
    scales = torch.cat(scales, dim=0)
    opacities = torch.cat(opacities, dim=0)
    sh0 = torch.cat(sh0, dim=0)
    shN = torch.cat(shN, dim=0)
    colors = torch.cat([sh0, shN], dim=-2)
    print("Number of Gaussians:", len(means))

    export_splats(
        means=means,
        scales=scales,
        quats=quats,
        opacities=opacities,
        sh0=sh0,
        shN=shN,
        format="ply",
        save_to=ply_path,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--ckpt", type=str, default=None, help="path to the .pt file"
    )
    args = parser.parse_args()
    cli(main, args, verbose=True)
