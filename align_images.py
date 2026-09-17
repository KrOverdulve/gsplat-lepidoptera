import os
import glob
import numpy as np
import cv2
import argparse


# A focus-stack boundary can only occur every `BOUNDARY_PERIOD` images (the
# bursts are always a multiple of 10). At each candidate boundary we keep the
# stack going while consecutive frames still align well; once the ECC
# correlation drops below `CC_THRESHOLD` (or alignment fails outright) we treat
# it as a new viewpoint and start a new stack. Tune CC_THRESHOLD if stacks are
# split too eagerly (raise) or merged across viewpoints (lower).
BOUNDARY_PERIOD = 10
CC_THRESHOLD = 0.9

IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".tif", ".tiff")


def clamp(image):
    image = np.minimum(255, np.maximum(0, image))
    assert image.max() > 0
    return np.uint8(image)


def compute_pca(image):
    flat_img = image.reshape(-1, 3)
    mean, eigenvectors = cv2.PCACompute(flat_img, np.array(()), cv2.PCA_DATA_AS_ROW, 2)
    T = np.zeros((1, 2), dtype=np.float32)
    T[0, 0] = 1
    weights = cv2.PCABackProject(T, mean, eigenvectors)
    weights -= cv2.PCABackProject(np.zeros((1, 2), dtype=np.float32), mean, eigenvectors)
    weights /= np.sum(weights)
    return weights[0]


def convert_to_grayscale(image, weights):
    assert image.shape[-1] == 3
    w0, w1, w2 = weights[0], weights[1], weights[2]
    result = image[..., 0] * w0 + image[..., 1] * w1 + image[..., 2] * w2
    result = clamp(result)
    return result


def align_images(refcolor, srccolor, transformation, pca_weights):
    """Align srccolor onto refcolor. Returns (aligned_uint8, cc).

    Raises cv2.error when ECC cannot converge (frames too different).
    """
    refcolor, srccolor = np.float32(refcolor), np.float32(srccolor)
    if np.allclose(refcolor, srccolor):
        return clamp(srccolor), 1.0
    refgray = convert_to_grayscale(refcolor, pca_weights)
    srcgray = convert_to_grayscale(srccolor, pca_weights)
    cc, transform = find_transform(srcgray, refgray, transformation)
    image = apply_transform(srccolor, transform)
    image = clamp(image)
    return image, cc


def find_transform(src, ref, transformation, max_res=2048):
    resolution = np.max(ref.shape)
    scale_ratio = 1.
    if resolution > max_res:
        scale_ratio = max_res / resolution
        ref = cv2.resize(ref, None, None, scale_ratio, scale_ratio, cv2.INTER_AREA)
        src = cv2.resize(src, None, None, scale_ratio, scale_ratio, cv2.INTER_AREA)
    transformation[:, 2] *= scale_ratio
    number_of_iterations = 50
    termination_eps = 0.001
    warp_mode = cv2.MOTION_AFFINE
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_COUNT, number_of_iterations, termination_eps)
    cc, transformation = cv2.findTransformECC(src, ref, transformation, warp_mode, criteria)
    transformation[:, 2] /= scale_ratio
    return cc, transformation


def apply_transform(image, transformation):
    h, w = image.shape[:2]
    image = cv2.warpAffine(image, transformation, (w, h), None, cv2.INTER_CUBIC, cv2.BORDER_REFLECT)
    return image


def write_stack(output_dir, start_idx, src_paths, aligned):
    """Write one aligned focus stack to output_dir/<start_idx>/ as max-quality JPEG."""
    if not aligned:
        return
    burst_dir = os.path.join(output_dir, f"{start_idx:05d}")
    if os.path.isdir(burst_dir):
        existing = len(glob.glob(os.path.join(burst_dir, "*.jpg")))
        if existing >= len(aligned):
            print(f"skipping {burst_dir}: already has {existing} images")
            return
    os.makedirs(burst_dir, exist_ok=True)
    jpeg_params = [
        cv2.IMWRITE_JPEG_QUALITY, 100,
        cv2.IMWRITE_JPEG_SAMPLING_FACTOR, cv2.IMWRITE_JPEG_SAMPLING_FACTOR_444,
    ]
    for src_path, image in zip(src_paths, aligned):
        out_name = os.path.splitext(os.path.basename(src_path))[0] + ".jpg"
        bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        cv2.imwrite(os.path.join(burst_dir, out_name), bgr, jpeg_params)
    print(f"Saved focus stack of {len(aligned)} images to {burst_dir}")


def align_bursts(input_dir, output_dir):
    paths = sorted(
        p for p in glob.glob(os.path.join(input_dir, "*"))
        if p.lower().endswith(IMAGE_EXTS)
    )
    if not paths:
        print(f"No images found in {input_dir}")
        return

    os.makedirs(output_dir, exist_ok=True)

    # Resume an interrupted run: skip input frames already covered by complete
    # stack folders. A complete stack holds a multiple of BOUNDARY_PERIOD JPEGs
    # (bursts are always a multiple of 10); a folder with any other count is a
    # partially written stack, so we stop there and redo it.
    resume_at = 0
    while True:
        burst_dir = os.path.join(output_dir, f"{resume_at:05d}")
        if not os.path.isdir(burst_dir):
            break
        n = len(glob.glob(os.path.join(burst_dir, "*.jpg")))
        if n == 0 or n % BOUNDARY_PERIOD != 0:
            break
        resume_at += n
    if resume_at > 0:
        print(f"resuming at frame index {resume_at} ({resume_at} already aligned)")

    current_paths = []      # source paths in the current stack
    current_aligned = []    # aligned uint8 RGB images in the current stack
    pca_weights = None
    reference = None        # last aligned frame, the alignment anchor
    transformation = np.float32(np.eye(2, 3))
    stack_start = 0

    def start_new_stack(idx, path, image):
        nonlocal current_paths, current_aligned, pca_weights
        nonlocal reference, transformation, stack_start
        current_paths = [path]
        current_aligned = [image]
        pca_weights = compute_pca(image)
        reference = image
        transformation = np.float32(np.eye(2, 3))
        stack_start = idx

    for i, path in enumerate(paths):
        if i < resume_at:
            continue
        image = cv2.imread(path, cv2.IMREAD_COLOR)
        if image is None:
            print(f"Warning: could not read {path}")
            continue
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        if reference is None:
            start_new_stack(i, path, image)
            print(f"stack {stack_start:05d}: aligning 1 ({os.path.basename(path)})")
            continue

        boundary = (i % BOUNDARY_PERIOD == 0)
        try:
            aligned, cc = align_images(reference, image, transformation, pca_weights)
            ok = True
        except cv2.error:
            aligned, cc, ok = image, -1.0, False

        if boundary and (not ok or cc < CC_THRESHOLD):
            # Too big a difference at a candidate boundary: new viewpoint.
            print(f"  jump at index {i} (cc={cc:.3f}) -> new viewpoint")
            write_stack(output_dir, stack_start, current_paths, current_aligned)
            start_new_stack(i, path, image)
            print(f"stack {stack_start:05d}: aligning 1 ({os.path.basename(path)})")
        else:
            current_paths.append(path)
            current_aligned.append(aligned)
            reference = aligned
            if not ok:
                transformation = np.float32(np.eye(2, 3))
            print(f"           aligning {len(current_aligned)} (cc={cc:.3f})")

    write_stack(output_dir, stack_start, current_paths, current_aligned)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Segment a flat list of images into focus stacks and align each stack."
    )
    parser.add_argument(
        '--input_dir',
        type=str,
        required=True,
        help="Directory holding a flat list of images in capture order."
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        required=True,
        help="Directory where one subfolder of aligned images is written per focus stack."
    )

    args = parser.parse_args()

    align_bursts(args.input_dir, args.output_dir)
