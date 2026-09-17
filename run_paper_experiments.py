"""Run one of the paper's training experiments on butterfly1, the released
specimen.

Trains one model per evaluation view (the views that have a mask in
image_masks/), holding that view out of training via the trainer's
--holdout-image-name flag. Results land under
<result_root>/<experiment>/butterfly1/<view>/.

Experiments:
    full             our full model: reflect camera, explicit mirror plane,
                     no densification, alpha-entropy + distortion regularizers
    densify-adc      ablation: full model minus regularizers, ADC densification
    densify-none     ablation: full model minus regularizers, no densification
    noplane          ablation: reflect camera, NO mirror-plane Gaussians
    nomirror         baseline: plain 2DGS, no mirror handling, no densification
    primitives-3dgs  baseline: plain 3DGS, ADC densification
    primitives-2dgs  baseline: plain 2DGS, ADC densification

Usage:
    python run_paper_experiments.py full
"""

import argparse
import os
import subprocess
import sys

REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
SCENE = "butterfly1"

# The trainer's defaults are the full model; these switch its parts off.
NO_REG = ["--alpha_entropy_lambda", "0.0", "--dist_lambda", "0.0"]
ADC = ["--butterfly-densify"]

# name -> (needs mirror plane, model type, trainer flags, result subdir)
EXPERIMENTS = {
    "full": (True, "2dgs", [], "full"),
    "densify-adc": (True, "2dgs", ADC + NO_REG, "densify/adc"),
    "densify-none": (True, "2dgs", NO_REG, "densify/none"),
    "noplane": (True, "2dgs", ["--no-model-plane"] + NO_REG, "noplane"),
    "nomirror": (False, "2dgs", ["--random_bkgd"], "nomirror"),
    "primitives-3dgs": (False, "3dgs", ADC + ["--final_prune_opacity", "0.0"], "primitives/3dgs"),
    "primitives-2dgs": (False, "2dgs", ADC + ["--final_prune_opacity", "0.0"], "primitives/2dgs"),
}


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("experiment", choices=sorted(EXPERIMENTS))
    p.add_argument("--data_root", default="data",
                   help="Holds butterfly1/undistorted/ (images, masks, calibration, plane)")
    p.add_argument("--result_root", default="results")
    p.add_argument("--data_factor", type=int, default=4,
                   help="Image downscale factor (the paper evaluates at 4)")
    args = p.parse_args()

    with_mirror, model_type, flags, subdir = EXPERIMENTS[args.experiment]

    data_dir = os.path.join(args.data_root, SCENE, "undistorted") + os.sep
    mask_dir = os.path.join(data_dir, "image_masks")
    if not os.path.isdir(mask_dir):
        sys.exit(f"No image_masks at {mask_dir}")
    stems = sorted(os.path.splitext(f)[0] for f in os.listdir(mask_dir))
    if not stems:
        sys.exit(f"No masks found in {mask_dir}")

    mirror_flags = []
    if with_mirror:
        plane_file = os.path.join(data_dir, "plane_parameters.json")
        if not os.path.isfile(plane_file):
            sys.exit(f"No plane_parameters.json at {plane_file}")
        mirror_flags = ["--mirror-param-file", plane_file]

    print(f"\n# ########## {args.experiment} {SCENE}: "
          f"{len(stems)} held-out views -> {stems} ##########", flush=True)
    for j, stem in enumerate(stems):
        result_dir = os.path.join(args.result_root, subdir, SCENE, stem)
        cmd = [sys.executable, os.path.join(REPO_ROOT, "examples", "simple_trainer.py"),
               "default",
               "--data-dir", data_dir,
               "--result-dir", result_dir,
               "--data-factor", str(args.data_factor),
               "--disable-video", "--disable_viewer", "--save_ply",
               "--holdout-image-name", stem,
               "--model_type", model_type,
               *mirror_flags, *flags]
        print(f"\n# ===== [{j + 1}/{len(stems)}] {args.experiment} {SCENE} {stem} =====")
        print(" ".join(cmd), flush=True)
        proc = subprocess.run(cmd, cwd=REPO_ROOT)
        if proc.returncode != 0:
            sys.exit(f"{args.experiment} failed for {SCENE}/{stem} (exit {proc.returncode})")

    print(f"\n# {args.experiment} finished for {SCENE}")


if __name__ == "__main__":
    main()
