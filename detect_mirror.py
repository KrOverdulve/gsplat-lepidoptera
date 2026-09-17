import open3d as o3d
import numpy as np
import copy
import json
import argparse
import os
from pycolmap import SceneManager

def extract_plane_parameters(T_matrix):
    """
    Extracts the plane equation parameters (a, b, c, d) from the transformation matrix.
    The transformation T maps the artificially X-mirrored cloud back to the real cloud.
    """
    M_x = np.eye(4)
    M_x[0, 0] = -1.0
    
    S = T_matrix @ M_x
    R_s = S[0:3, 0:3]
    t_s = S[0:3, 3]
    
    evals, evecs = np.linalg.eig(R_s)
    idx = np.argmin(np.abs(evals - (-1.0)))
    
    n = np.real(evecs[:, idx])
    n = n / np.linalg.norm(n)
    
    d = -np.dot(n, t_s / 2.0)
    
    return float(n[0]), float(n[1]), float(n[2]), float(d)

def create_init_cloud(pcd, a, b, c, d, reflect_positive=True, include_reflected=False):
    """Return the init cloud for training: the real (camera-side) points. With include_reflected,
    also append the reflected (far-side) points mirrored across the plane onto
    the real side. `reflect_positive` selects which side is the reflected one:
    True -> points with distance > 0 are the reflected side; False -> < 0.
    """
    points = np.asarray(pcd.points)
    n = np.array([a, b, c])
    distances = np.dot(points, n) + d

    reflected_mask = distances > 0 if reflect_positive else distances < 0

    # The original butterfly: the camera-side points, kept in place.
    real_pcd = pcd.select_by_index(np.where(~reflected_mask)[0])
    if not include_reflected:
        return real_pcd

    # Plus the reflected-side points, mirrored across the plane onto the real side.
    refl_pcd = pcd.select_by_index(np.where(reflected_mask)[0])
    sub_pts = np.asarray(refl_pcd.points)
    sub_dist = np.dot(sub_pts, n) + d
    refl_pcd.points = o3d.utility.Vector3dVector(sub_pts - 2.0 * sub_dist[:, np.newaxis] * n)

    if refl_pcd.has_normals():
        sub_normals = np.asarray(refl_pcd.normals)
        dot_products = np.sum(sub_normals * n, axis=1)
        refl_pcd.normals = o3d.utility.Vector3dVector(
            sub_normals - 2.0 * dot_products[:, np.newaxis] * n
        )

    return real_pcd + refl_pcd

def load_camera_centers(data_dir):
    """Load camera centers (world-space positions) from a COLMAP reconstruction."""
    colmap_dir = os.path.join(data_dir, "sparse/0/")
    if not os.path.exists(colmap_dir):
        colmap_dir = os.path.join(data_dir, "sparse")
    assert os.path.exists(colmap_dir), f"COLMAP directory {colmap_dir} does not exist."

    manager = SceneManager(colmap_dir)
    manager.load_cameras()
    manager.load_images()

    centers = []
    for k in manager.images:
        im = manager.images[k]
        R = im.R()
        t = im.tvec.reshape(3)
        # Camera center in world: C = -R^T t
        C = -R.T @ t
        centers.append(C)
    return np.stack(centers, axis=0)


def detect_plane_once(mirrored_down, pcd_down, mirrored_fpfh, pcd_fpfh,
                      ransac_threshold, icp_threshold):
    """One RANSAC global registration + point-to-plane ICP refinement.

    Returns ((a, b, c, d), result_icp). RANSAC is stochastic, so repeated calls
    on a symmetric object land on different symmetry planes.
    """
    result_ransac = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
        mirrored_down, pcd_down, mirrored_fpfh, pcd_fpfh, True,
        ransac_threshold,
        o3d.pipelines.registration.TransformationEstimationPointToPoint(False),
        3, [
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(0.9),
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(ransac_threshold)
        ], o3d.pipelines.registration.RANSACConvergenceCriteria(100000, 0.999))
    result_icp = o3d.pipelines.registration.registration_icp(
        mirrored_down, pcd_down, icp_threshold, result_ransac.transformation,
        o3d.pipelines.registration.TransformationEstimationPointToPlane(),
        o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=100))
    return extract_plane_parameters(result_icp.transformation), result_icp


def _canonical_plane(a, b, c, d):
    """Unit-normal plane with a fixed sign convention so (n, d) == (-n, -d)."""
    n = np.array([a, b, c], dtype=float)
    nrm = np.linalg.norm(n)
    n, dd = n / nrm, d / nrm
    if n[int(np.argmax(np.abs(n)))] < 0:
        n, dd = -n, -dd
    return n, dd


def _same_plane(p1, p2, cos_tol, d_tol):
    """True if two canonical planes are near-parallel and near-coincident."""
    (n1, d1), (n2, d2) = p1, p2
    return float(np.dot(n1, n2)) > cos_tol and abs(d1 - d2) < d_tol


def camera_side_fraction(a, b, c, d, camera_centers):
    """Fraction of cameras on the majority side of the plane. ~1.0 for the true
    mirror (the cameras form a hemisphere entirely on the real side); ~0.5 for a
    plane that bisects the camera hemisphere (a spurious object-symmetry plane).
    """
    s = camera_centers @ np.array([a, b, c], dtype=float) + d
    if len(s) == 0:
        return 0.0
    pos = int(np.sum(s > 0))
    return max(pos, len(s) - pos) / len(s)


def write_plane(out_dir, pcd, pcd_down, plane, result_icp, camera_centers, include_reflected):
    """Write the detected plane to out_dir: plane_parameters.json, the init
    cloud for training (init_cloud.ply) and the downsampled cloud colored by
    whether each point supported the plane (plane_inliers.ply)."""
    os.makedirs(out_dir, exist_ok=True)
    a, b, c, d = plane

    with open(os.path.join(out_dir, "plane_parameters.json"), "w") as f:
        json.dump({"a": a, "b": b, "c": c, "d": d}, f, indent=4)

    # ICP inlier points that supported this plane: green; the rest: red.
    corr = np.asarray(result_icp.correspondence_set)
    colors = np.tile([1.0, 0.0, 0.0], (len(pcd_down.points), 1))
    if corr.size > 0:
        colors[np.unique(corr[:, 1])] = [0.0, 1.0, 0.0]
    inlier_pcd = copy.deepcopy(pcd_down)
    inlier_pcd.colors = o3d.utility.Vector3dVector(colors)
    o3d.io.write_point_cloud(os.path.join(out_dir, "plane_inliers.ply"), inlier_pcd)

    # Init cloud: the real (camera-side) specimen, optionally plus the reflected
    # side mirrored onto it. reflect_positive marks which side is the reflection.
    reflect_positive = float(np.mean(camera_centers @ np.array([a, b, c]) + d)) < 0
    init_pcd = create_init_cloud(pcd, a, b, c, d, reflect_positive=reflect_positive,
                                 include_reflected=include_reflected)
    o3d.io.write_point_cloud(os.path.join(out_dir, "init_cloud.ply"), init_pcd)


def main():
    parser = argparse.ArgumentParser(description="Detect mirror plane in a point cloud with ICP refinement.")
    parser.add_argument("--input", required=True, help="Path to input .ply point cloud")
    parser.add_argument("--output_dir", required=True, help="Directory to save output files")
    parser.add_argument("--data_dir", required=True, help="COLMAP reconstruction directory (containing sparse/) used to determine which side of the plane faces the cameras")
    parser.add_argument("--voxel_size", type=float, default=0.01, help="Voxel size for downsampling")
    parser.add_argument("--include_reflected", action="store_true", help="In the init cloud, also include the reflected (far-side) points folded across the plane onto the real specimen, in addition to the original camera-side points.")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    print(f"Loading point cloud from {args.input}...")
    pcd = o3d.io.read_point_cloud(args.input)
    
    if not pcd.has_normals():
        print("Estimating normals...")
        pcd.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=args.voxel_size * 2, max_nn=30))

    # 1. Create artificially mirrored cloud
    print("Generating artificial mirror clone...")
    mirrored_pcd = copy.deepcopy(pcd)
    points = np.asarray(mirrored_pcd.points)
    points[:, 0] = -points[:, 0]
    mirrored_pcd.points = o3d.utility.Vector3dVector(points)
    
    if mirrored_pcd.has_normals():
        normals = np.asarray(mirrored_pcd.normals)
        normals[:, 0] = -normals[:, 0]
        mirrored_pcd.normals = o3d.utility.Vector3dVector(normals)

    # 2. Downsample and compute FPFH features
    print("Extracting FPFH features...")
    radius_feature = args.voxel_size * 5

    pcd_down = pcd.voxel_down_sample(args.voxel_size)
    pcd_fpfh = o3d.pipelines.registration.compute_fpfh_feature(
        pcd_down, o3d.geometry.KDTreeSearchParamHybrid(radius=radius_feature, max_nn=100))

    mirrored_down = mirrored_pcd.voxel_down_sample(args.voxel_size)
    mirrored_fpfh = o3d.pipelines.registration.compute_fpfh_feature(
        mirrored_down, o3d.geometry.KDTreeSearchParamHybrid(radius=radius_feature, max_nn=100))

    # 3-6. Detect the mirror plane(s). RANSAC is stochastic and a symmetric
    # object yields several competing symmetry planes, so we gather several
    # hypotheses. Hard constraint: the cameras form a hemisphere on the real
    # side, so the mirror can NEVER split them -- any plane with cameras on both
    # sides is rejected. Survivors are ranked by ICP fitness.
    ransac_threshold = args.voxel_size * 1.5
    icp_threshold = args.voxel_size * 0.4

    print(f"Loading COLMAP cameras from {args.data_dir}...")
    camera_centers = load_camera_centers(args.data_dir)

    trials = 10  # T in the paper
    print(f"Running {trials} RANSAC+ICP trials to gather plane hypotheses...")
    extent = np.asarray(pcd_down.get_max_bound()) - np.asarray(pcd_down.get_min_bound())
    d_tol = 0.02 * float(np.linalg.norm(extent))
    cos_tol = float(np.cos(np.deg2rad(10.0)))
    groups = []  # {plane, result_icp, canon, fitness, rmse, support}
    for _ in range(trials):
        plane, result_icp = detect_plane_once(
            mirrored_down, pcd_down, mirrored_fpfh, pcd_fpfh, ransac_threshold, icp_threshold)
        canon = _canonical_plane(*plane)
        fitness, rmse = float(result_icp.fitness), float(result_icp.inlier_rmse)
        match = next((g for g in groups if _same_plane(canon, g["canon"], cos_tol, d_tol)), None)
        if match is None:
            groups.append({"plane": plane, "result_icp": result_icp, "canon": canon,
                           "fitness": fitness, "rmse": rmse, "support": 1})
        else:
            match["support"] += 1
            if fitness > match["fitness"]:  # keep the best-fitness representative
                match.update(plane=plane, result_icp=result_icp, canon=canon,
                             fitness=fitness, rmse=rmse)

    if not groups:
        print("No mirror plane hypotheses found; aborting.")
        return

    for g in groups:
        g["cam_side"] = camera_side_fraction(*g["plane"], camera_centers)

    # Hard constraint: the mirror may never split the camera hemisphere, so drop
    # any plane with cameras on both sides. Among the survivors, rank by fitness.
    valid = [g for g in groups if g["cam_side"] >= 1.0]
    if valid:
        valid.sort(key=lambda g: (-g["fitness"], g["rmse"]))
    else:
        print("WARNING: every hypothesis splits the cameras across the plane -- "
              "detection is unreliable. Falling back to the least-splitting plane(s).")
        valid = sorted(groups, key=lambda g: (-g["cam_side"], -g["fitness"]))

    best = valid[0]
    a, b, c, d = best["plane"]
    print(f"{len(groups)} distinct plane(s), {sum(g['cam_side'] >= 1.0 for g in groups)} "
          f"camera-consistent; writing the best:")
    write_plane(args.output_dir, pcd, pcd_down, best["plane"], best["result_icp"],
                camera_centers, args.include_reflected)
    print(f"  n=({a:.3f}, {b:.3f}, {c:.3f}) d={d:.4f} "
          f"cam_side={best['cam_side']:.2f} fitness={best['fitness']:.4f} rmse={best['rmse']:.4f} "
          f"support={best['support']}/{trials} -> {args.output_dir}")

if __name__ == "__main__":
    main()