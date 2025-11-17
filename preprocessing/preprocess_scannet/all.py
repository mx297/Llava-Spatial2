#!/usr/bin/env python3
"""
ScanNet All-in-One Pipeline

Downloads scenes, exports uniformly sampled frames (pose/depth/color/instance + intrinsics),
exports preview videos, builds labeled meshes (ScanNet200 format), then deletes
local scene folders to conserve disk.

Pipeline per scene:
  1) Download via scannet.py
  2) (optional) Export sampled frames
  3) (optional) Export color preview video
  4) (optional) Build labeled mesh (.ply) using ScanNet200 utilities
  5) Delete the local scene folder

Run `python scannet_all_in_one.py -h` for full CLI.

Notes
- The mesh step requires the ScanNet200 helper modules available on PYTHONPATH:
  scannet200_constants, scannet200_splits, scannet200_utils_from_scannet
- The frames step expects SensorData utilities available as `sensor.SensorData`
  (as in official ScanNet repo).
- `scannet.py` must be available and callable. We pass "\n\n" to accept its
  interactive prompts automatically.
"""

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
import os
import shutil
import subprocess
import sys
from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
from tqdm import tqdm

# Import all dependencies once at the top (no lazy imports, no fallbacks, no sys.path tweaks)
import cv2
import png
from .sensor import SensorData
from .scannet200_constants import VALID_CLASS_IDS_200 as CLASS_IDs
from .scannet200_utils_from_scannet import (
    read_plymesh,
    save_plymesh,
    voxelize_pointcloud,
    point_indices_from_group,
)

# ---------------------------
# Utilities
# ---------------------------

def get_scannet_scene_ids_from_jsonl(jsonl_path: str) -> List[str]:
    """Return unique ScanNet scene IDs from a JSONL with 'data_source' and 'scene_name'."""
    try:
        df = pd.read_json(jsonl_path, lines=True)
    except Exception as e:
        print(f"Error reading {jsonl_path}: {e}")
        return []
    if 'data_source' not in df.columns or 'scene_name' not in df.columns:
        print(f"Error: {jsonl_path} must contain 'data_source' and 'scene_name' columns.")
        return []
    return sorted(np.unique(df[df['data_source'] == 'scannet']['scene_name']).tolist())


def download_scannet_scene(scannet_script: str, out_dir: str, scene_id: str) -> bool:
    """Call scannet.py to download a single scene into out_dir. Returns True on success.

    scannet.py writes to {out_dir}/scans/{scene_id}
    """
    cmd = [sys.executable, scannet_script, "-o", out_dir, "--id", scene_id]
    try:
        # Accept license/data terms prompts automatically (two Enters)
        subprocess.run(cmd, input="\n\n", text=True, check=True)
        return True
    except subprocess.CalledProcessError as e:
        print(f"Download failed for {scene_id}: {e}")
        return False


# ---------------------------
# Frames export (pose/depth/color/instance + intrinsics)
# ---------------------------

def _parse_scene_meta_file(meta_txt_path: str) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """Parses {scene_id}.txt to extract axisAlignment (4x4) and depth intrinsics (4x4)."""
    metadata = {}
    try:
        with open(meta_txt_path, 'r') as f:
            for line in f:
                parts = line.strip().split(' = ')
                if len(parts) == 2:
                    key, value = parts
                    metadata[key.strip()] = value.strip()
    except FileNotFoundError:
        print(f"Warning: metadata file not found: {meta_txt_path}")
        return None, None
    except Exception as e:
        print(f"Warning: error reading metadata {meta_txt_path}: {e}")
        return None, None

    axis = None
    if 'axisAlignment' in metadata:
        vals = metadata['axisAlignment'].split()
        if len(vals) == 16:
            axis = np.array(list(map(float, vals))).reshape(4, 4)
        else:
            print(f"Warning: axisAlignment has {len(vals)} values (expected 16) in {meta_txt_path}")

    depth_intr = None
    try:
        fx = float(metadata['fx_depth']); fy = float(metadata['fy_depth'])
        mx = float(metadata['mx_depth']); my = float(metadata['my_depth'])
        depth_intr = np.array([[fx, 0,  mx, 0],
                               [0,  fy, my, 0],
                               [0,  0,  1,  0],
                               [0,  0,  0,  1]])
    except Exception:
        pass

    return axis, depth_intr


def _save_matrix(path: str, mat: np.ndarray) -> None:
    with open(path, 'w') as f:
        for row in mat:
            f.write(' '.join(map(str, row)) + '\n')


def export_scene_sampled_frames(
    sens_file_path: str,
    output_base_dir: str,
    num_frames_to_sample: int,
    split: str,
    image_size: Optional[Tuple[int, int]] = None,  # (H, W)
    min_valid_components_per_frame_initial: int = 0,
    min_valid_frames_per_scene: int = 1,
) -> Optional[str]:
    """Export uniformly-sampled frames (pose/depth/color/instance + intrinsics).

    Returns scene_id on success, else None.
    """
    scene_id = os.path.basename(os.path.dirname(sens_file_path))
    scene_dir = os.path.dirname(sens_file_path)

    print(f"Processing frames for scene: {scene_id}")
    try:
        sd = SensorData(sens_file_path)
    except Exception as e:
        print(f"Error loading SensorData from {sens_file_path}: {e}")
        return None

    if not hasattr(sd, 'frames') or not sd.frames:
        print("Error: SensorData has no frames.")
        return None

    total_raw = len(sd.frames)
    if total_raw == 0:
        print(f"Scene {scene_id} has 0 frames. Skipping.")
        return scene_id

    meta_txt = os.path.join(scene_dir, f"{scene_id}.txt")
    inst_zip_path = os.path.join(scene_dir, f"{scene_id}_2d-instance-filt.zip")
    if not os.path.exists(inst_zip_path):
        alt = os.path.join(scene_dir, f"{scene_id}_2d-instance.zip")
        inst_zip_path = alt if os.path.exists(alt) else None

    inst_zip = None
    if inst_zip_path:
        import zipfile
        try:
            inst_zip = zipfile.ZipFile(inst_zip_path, 'r')
        except Exception as e:
            print(f"Warning: could not open instance zip {inst_zip_path}: {e}")
            inst_zip = None

    # Validate frames to get candidates
    candidates = []
    for i in tqdm(range(total_raw), desc=f"Validating frames {scene_id}", leave=False):
        try:
            fr = sd.frames[i]
            have = 0
            if getattr(fr, 'camera_to_world', None) is not None:
                have += 1
            if hasattr(fr, 'decompress_depth') and getattr(sd, 'depth_compression_type', 'unknown').lower() != 'unknown':
                have += 1
            if hasattr(fr, 'decompress_color') and getattr(sd, 'color_compression_type', '').lower() == 'jpeg':
                have += 1
            if inst_zip is not None:
                try:
                    inst_zip.getinfo(f'instance-filt/{i}.png')
                    have += 1
                except Exception:
                    pass
            if have >= min_valid_components_per_frame_initial:
                candidates.append(i)
        except Exception:
            continue

    if not candidates:
        print(f"No valid frame candidates for {scene_id}.")
        if inst_zip: inst_zip.close()
        return None

    # Choose indices uniformly over candidates
    if len(candidates) <= num_frames_to_sample:
        selected = candidates
    else:
        idxs = np.linspace(0, len(candidates)-1, num_frames_to_sample, dtype=int)
        selected = sorted(set(candidates[j] for j in idxs))

    # make output dirs
    pose_dir = os.path.join(output_base_dir, 'pose', split, scene_id)
    depth_dir = os.path.join(output_base_dir, 'depth', split, scene_id)
    color_dir = os.path.join(output_base_dir, 'color', split, scene_id)
    inst_dir  = os.path.join(output_base_dir, 'instance', split, scene_id)
    intr_dir  = os.path.join(output_base_dir, 'intrinsic', split)
    for d in [pose_dir, depth_dir, color_dir, inst_dir, intr_dir]:
        os.makedirs(d, exist_ok=True)

    axis, depth_intr = _parse_scene_meta_file(meta_txt)
    if depth_intr is not None:
        _save_matrix(os.path.join(intr_dir, f'intrinsic_depth_{scene_id}.txt'), depth_intr)

    valid_frames = 0
    total_attempted = 0

    try:
        for i in tqdm(selected, desc=f"Exporting {scene_id}", leave=False):
            total_attempted += 1
            fr = sd.frames[i]
            components_ok = 0

            # Pose (axis aligned if available)
            pose_path = None
            if getattr(fr, 'camera_to_world', None) is not None:
                p = fr.camera_to_world
                if axis is not None:
                    try:
                        p = np.dot(axis, p)
                    except Exception:
                        pass
                pose_path = os.path.join(pose_dir, f'{i:06d}.txt')
                _save_matrix(pose_path, p)
                components_ok += 1

            # Depth (16-bit PNG)
            depth_path = None
            if hasattr(fr, 'decompress_depth') and getattr(sd, 'depth_compression_type', 'unknown').lower() != 'unknown':
                try:
                    raw = fr.decompress_depth(sd.depth_compression_type)
                    if raw is not None:
                        arr = np.frombuffer(raw, dtype=np.uint16).reshape(sd.depth_height, sd.depth_width)
                        depth_path = os.path.join(depth_dir, f'{i:06d}.png')
                        with open(depth_path, 'wb') as f_png:
                            w = png.Writer(width=arr.shape[1], height=arr.shape[0], bitdepth=16, greyscale=True)
                            w.write(f_png, arr.tolist())
                        components_ok += 1
                except Exception:
                    pass

            # Color (JPEG, resized if requested)
            color_path = None
            critical_color_fail = False
            if hasattr(fr, 'decompress_color') and getattr(sd, 'color_compression_type', '').lower() == 'jpeg':
                try:
                    rgb = fr.decompress_color(sd.color_compression_type)
                    if rgb is not None:
                        if image_size:
                            H, W = image_size
                            rgb = cv2.resize(rgb, (W, H))
                        bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
                        color_path = os.path.join(color_dir, f'{i:06d}.jpg')
                        if not cv2.imwrite(color_path, bgr):
                            raise RuntimeError('cv2.imwrite failed')
                        components_ok += 1
                    else:
                        critical_color_fail = True
                except Exception:
                    critical_color_fail = True
            else:
                critical_color_fail = True

            if critical_color_fail:
                # Clean partial files and skip counting this frame
                for pth in [pose_path, depth_path]:
                    if pth and os.path.exists(pth):
                        try: os.remove(pth)
                        except Exception: pass
                continue

            # Instance mask (optional)
            inst_path = None
            if inst_zip is not None:
                import zipfile
                try:
                    with inst_zip.open(f'instance-filt/{i}.png', 'r') as fz:
                        data = fz.read()
                    mask = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_UNCHANGED)
                    if mask is not None:
                        if image_size:
                            H, W = image_size
                            mask = cv2.resize(mask, (W, H), interpolation=cv2.INTER_NEAREST)
                        inst_path = os.path.join(inst_dir, f'{i:06d}.png')
                        cv2.imwrite(inst_path, mask)
                        components_ok += 1
                except Exception:
                    pass

            if components_ok >= max(1, min_valid_components_per_frame_initial):
                valid_frames += 1
            else:
                # Clean partial outputs for this frame
                for pth in [pose_path, depth_path, color_path, inst_path]:
                    if pth and os.path.exists(pth):
                        try: os.remove(pth)
                        except Exception: pass

    finally:
        if inst_zip:
            try: inst_zip.close()
            except Exception: pass

    if valid_frames >= min_valid_frames_per_scene:
        print(f"Frames export OK for {scene_id}: {valid_frames} valid of {len(selected)} selected.")
        return scene_id
    else:
        print(f"Frames export FAILED for {scene_id}: {valid_frames} < {min_valid_frames_per_scene}.")
        return None


# ---------------------------
# Video export
# ---------------------------

def export_scene_video(
    sens_file_path: str,
    output_video_path: str,
    width: int,
    height: int,
    fps: int,
    frame_skip: int,
    codec: str,
) -> bool:
    print(f"Exporting video for {os.path.basename(os.path.dirname(sens_file_path))}")
    try:
        sd = SensorData(sens_file_path)
    except Exception as e:
        print(f"Error loading SensorData: {e}")
        return False

    size = (width, height)
    fourcc = cv2.VideoWriter_fourcc(*codec)
    writer = cv2.VideoWriter(output_video_path, fourcc, fps, size)
    if not writer.isOpened():
        print(f"Could not open writer: {output_video_path}")
        return False

    ok = True
    try:
        for i in tqdm(range(0, len(sd.frames), frame_skip), desc="Video frames", leave=False):
            try:
                fr = sd.frames[i]
                rgb = fr.decompress_color(sd.color_compression_type)
                if rgb is None:
                    continue
                bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
                if bgr.shape[1] != width or bgr.shape[0] != height:
                    bgr = cv2.resize(bgr, size, interpolation=cv2.INTER_AREA)
                writer.write(bgr)
            except Exception:
                continue
    except Exception:
        ok = False
    finally:
        writer.release()

    print(f"Video saved to: {output_video_path}" if ok else "Video export failed")
    return ok


# ---------------------------
# Mesh processing (ScanNet200)
# ---------------------------

CLOUD_FILE_PFIX = '_vh_clean_2'
SEGMENTS_FILE_PFIX = '.0.010000.segs.json'
AGGREGATIONS_FILE_PFIX = '.aggregation.json'


def _process_mesh_for_scene(
    scene_dir: str,
    output_root: str,
    labels_pd: pd.DataFrame,
    train_scenes: set,
    val_scenes: set,
) -> Tuple[bool, str, str]:
    """Build labeled mesh and write to output_root/{train|val|test}/scene_id.ply
    Returns (success, split, message)
    """
    scene_id = os.path.basename(scene_dir.rstrip('/'))
    mesh_path = os.path.join(scene_dir, f'{scene_id}{CLOUD_FILE_PFIX}.ply')
    segments_file = os.path.join(scene_dir, f'{scene_id}{CLOUD_FILE_PFIX}{SEGMENTS_FILE_PFIX}')
    aggregations_file = os.path.join(scene_dir, f'{scene_id}{AGGREGATIONS_FILE_PFIX}')
    info_file = os.path.join(scene_dir, f'{scene_id}.txt')

    split = 'train' if scene_id in train_scenes else 'val' if scene_id in val_scenes else 'test'
    os.makedirs(os.path.join(output_root, split), exist_ok=True)
    out_ply = os.path.join(output_root, split, f'{scene_id}.ply')

    # Load axis align
    info = {}
    try:
        with open(info_file) as f:
            for line in f:
                if ' = ' in line:
                    k, v = line.split(' = ')
                    info[k] = np.fromstring(v, sep=' ')
    except Exception as e:
        return False, split, f'missing_info_txt:{e}'

    rot = np.identity(4)
    if 'axisAlignment' in info:
        try:
            rot = info['axisAlignment'].reshape(4, 4)
        except Exception:
            pass

    # Load mesh
    try:
        pointcloud, faces = read_plymesh(mesh_path)
    except Exception as e:
        return False, split, f'read_ply_failed:{e}'

    # Rotate
    r_pts = pointcloud[:, :3].T
    r_pts = np.append(r_pts, np.ones((1, r_pts.shape[1])), axis=0)
    r_pts = np.dot(rot, r_pts)
    pointcloud = np.append(r_pts.T[:, :3], pointcloud[:, 3:], axis=1)

    # Load segments/aggregations
    try:
        with open(segments_file) as f:
            seg_indices = np.array(json.load(f)['segIndices'])
        with open(aggregations_file) as f:
            seg_groups = np.array(json.load(f)['segGroups'])
    except Exception as e:
        return False, split, f'seg_or_agg_missing:{e}'

    # Label points
    labelled_pc = np.zeros((pointcloud.shape[0], 1))
    instance_ids = np.zeros((pointcloud.shape[0], 1))
    try:
        for group in seg_groups:
            _, p_inds, label_id = point_indices_from_group(pointcloud, seg_indices, group, labels_pd, CLASS_IDs)
            labelled_pc[p_inds] = label_id
            instance_ids[p_inds] = group['id']
    except Exception as e:
        return False, split, f'labeling_failed:{e}'

    labelled_pc = labelled_pc.astype(int)
    instance_ids = instance_ids.astype(int)
    vertices = np.hstack((pointcloud[:, :6], labelled_pc, instance_ids))

    if (np.any(np.isnan(vertices)) or not np.all(np.isfinite(vertices))):
        return False, split, 'nan_in_vertices'

    try:
        save_plymesh(vertices, faces, out_ply, with_label=True, verbose=False)
    except Exception as e:
        return False, split, f'save_ply_failed:{e}'

    return True, split, 'ok'


# ---------------------------
# Worker: download → (frames/video/mesh) → delete
# ---------------------------

@dataclass
class WorkerConfig:
    base_download_dir: str
    scans_subdir: str  # typically 'scans'
    output_frames_dir: Optional[str]
    output_video_dir: Optional[str]
    output_mesh_root: Optional[str]
    video_width: int
    video_height: int
    video_fps: int
    video_frame_skip: int
    video_codec: str
    num_frames: int
    image_size: Optional[Tuple[int, int]]
    min_valid_components_per_frame: int
    min_valid_frames_per_scene: int
    train_scenes: set
    val_scenes: set
    use_splits_for_frames_video: bool
    scannet_script: str
    labels_pd: Optional[pd.DataFrame]
    enable_frames: bool
    enable_video: bool
    enable_mesh: bool


def worker_download_process_delete(scene_id: str, cfg: WorkerConfig):
    """Returns dict with keys: scene_id, split, frames_ok, video_ok, mesh_ok, message"""
    scans_dir = os.path.join(cfg.base_download_dir, cfg.scans_subdir)
    scene_dir = os.path.join(scans_dir, scene_id)

    # 1) Download
    ok = download_scannet_scene(cfg.scannet_script, cfg.base_download_dir, scene_id)
    if not ok:
        return {"scene_id": scene_id, "split": "unknown", "frames_ok": False, "video_ok": False, "mesh_ok": False, "message": "download_failed"}

    sens_path = os.path.join(scene_dir, f"{scene_id}.sens")
    split = ""
    if cfg.use_splits_for_frames_video:
        split = 'train' if scene_id in cfg.train_scenes else 'val' if scene_id in cfg.val_scenes else ''

    frames_ok = video_ok = mesh_ok = False
    try:
        # 2) Frames
        if cfg.enable_frames:
            if not os.path.exists(sens_path):
                msg = "sens_missing_after_download"
                return {"scene_id": scene_id, "split": split or "nosplit", "frames_ok": False, "video_ok": False, "mesh_ok": False, "message": msg}
            frames_split = split or 'nosplit'
            if cfg.output_frames_dir is None:
                raise ValueError("output_frames_dir must be set when --enable_frames is used")
            os.makedirs(cfg.output_frames_dir, exist_ok=True)
            res = export_scene_sampled_frames(
                sens_file_path=sens_path,
                output_base_dir=cfg.output_frames_dir,
                num_frames_to_sample=cfg.num_frames,
                split=frames_split,
                image_size=cfg.image_size,
                min_valid_components_per_frame_initial=cfg.min_valid_components_per_frame,
                min_valid_frames_per_scene=cfg.min_valid_frames_per_scene,
            )
            frames_ok = res is not None

        # 3) Video
        if cfg.enable_video:
            if cfg.output_video_dir is None:
                raise ValueError("output_video_dir must be set when --enable_video is used")
            os.makedirs(os.path.join(cfg.output_video_dir, split or ''), exist_ok=True)
            ext = '.avi' if cfg.video_codec.lower() in ['mjpg'] else '.mp4'
            out_vid = os.path.join(cfg.output_video_dir, split or '', f"{scene_id}{ext}")
            video_ok = export_scene_video(
                sens_file_path=sens_path,
                output_video_path=out_vid,
                width=cfg.video_width,
                height=cfg.video_height,
                fps=cfg.video_fps,
                frame_skip=cfg.video_frame_skip,
                codec=cfg.video_codec,
            )

        # 4) Mesh
        if cfg.enable_mesh:
            if cfg.output_mesh_root is None or cfg.labels_pd is None:
                raise ValueError("output_mesh_root and label map are required when --enable_mesh is used")
            os.makedirs(cfg.output_mesh_root, exist_ok=True)
            ok_mesh, mesh_split, msg = _process_mesh_for_scene(
                scene_dir=scene_dir,
                output_root=cfg.output_mesh_root,
                labels_pd=cfg.labels_pd,
                train_scenes=cfg.train_scenes,
                val_scenes=cfg.val_scenes,
            )
            mesh_ok = ok_mesh

        return {"scene_id": scene_id, "split": split or "nosplit", "frames_ok": frames_ok, "video_ok": video_ok, "mesh_ok": mesh_ok, "message": "ok"}

    except Exception as e:
        return {"scene_id": scene_id, "split": split or "nosplit", "frames_ok": frames_ok, "video_ok": video_ok, "mesh_ok": mesh_ok, "message": f"processing_exception:{e}"}

    finally:
        # 5) Delete scene folder regardless of success
        try:
            shutil.rmtree(scene_dir, ignore_errors=True)
        except Exception:
            pass


# ---------------------------
# CLI
# ---------------------------

def build_argparser():
    p = argparse.ArgumentParser(
        description="ScanNet AIO: download → frames → video → mesh → delete",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    # Required basics
    p.add_argument('--scannet_script', type=str, default='scannet.py', help='Path to scannet.py downloader')
    p.add_argument('--download_base_dir', required=True, help='Directory passed to scannet.py -o (contains scans/)')

    # Scene selection
    p.add_argument('--jsonl_path', type=str, help='JSONL with data_source/scene_name')
    p.add_argument('--scene_ids', type=str, help='Comma-separated list of scene ids (overrides JSONL if given)')

    # Splits
    p.add_argument('--train_val_splits_path', type=str, help='Directory with scannetv2_train.txt and scannetv2_val.txt (optional)')

    # Frames export
    p.add_argument('--enable_frames', action='store_true', help='Enable sampled frame export')
    p.add_argument('--frames_output_dir', type=str, help='Base dir for frames export (pose/depth/color/instance/intrinsic)')
    p.add_argument('--num_frames', type=int, default=32, help='Frames to sample per scene')
    p.add_argument('--image_size', type=int, nargs=2, metavar=('H', 'W'), default=None, help='Resize H W for color/instance')
    p.add_argument('--min_valid_components_per_frame', type=int, default=0)
    p.add_argument('--min_valid_frames_per_scene', type=int, default=1)

    # Video export
    p.add_argument('--enable_video', action='store_true', help='Enable preview video export')
    p.add_argument('--video_output_dir', type=str, help='Output dir for videos')
    p.add_argument('--video_width', type=int, default=640)
    p.add_argument('--video_height', type=int, default=480)
    p.add_argument('--video_fps', type=int, default=30)
    p.add_argument('--video_frame_skip', type=int, default=1)
    p.add_argument('--video_codec', type=str, default='mp4v', help='mp4v (MP4), avc1 (H.264), or mjpg (AVI)')

    # Mesh export
    p.add_argument('--enable_mesh', action='store_true', help='Enable ScanNet200 labeled mesh export')
    p.add_argument('--mesh_output_root', type=str, help='Root dir for meshes with train/val/test subdirs')
    p.add_argument('--label_map_file', type=str, help='Path to scannetv2-labels.combined.tsv (for mesh step)')

    # Concurrency
    p.add_argument('--max_inflight_scenes', type=int, default=4, help='Max concurrent scenes (disk cap)')

    return p


def main():
    args = build_argparser().parse_args()

    scans_subdir = 'scans'
    base_dir = args.download_base_dir

    # Scene list (kept as in your original code)
    scene_ids: List[str] = [
        "scene0568_00", "scene0568_01", "scene0568_02", "scene0304_00",
        "scene0488_00", "scene0488_01", "scene0412_00", "scene0412_01",
        "scene0217_00", "scene0019_00", "scene0019_01", "scene0414_00",
        "scene0575_00", "scene0575_01", "scene0575_02", "scene0426_00",
        "scene0426_01", "scene0426_02", "scene0426_03", "scene0549_00",
        "scene0549_01", "scene0578_00", "scene0578_01", "scene0578_02",
        "scene0665_00", "scene0665_01", "scene0050_00", "scene0050_01",
        "scene0050_02", "scene0257_00", "scene0025_00", "scene0025_01",
        "scene0025_02", "scene0583_00", "scene0583_01", "scene0583_02",
        "scene0701_00", "scene0701_01", "scene0701_02", "scene0580_00",
        "scene0580_01", "scene0565_00", "scene0169_00", "scene0169_01",
        "scene0655_00", "scene0655_01", "scene0655_02", "scene0063_00",
        "scene0221_00", "scene0221_01", "scene0591_00", "scene0591_01",
        "scene0591_02", "scene0678_00", "scene0678_01", "scene0678_02",
        "scene0462_00", "scene0427_00", "scene0595_00", "scene0193_00",
        "scene0193_01", "scene0164_00", "scene0164_01", "scene0164_02",
        "scene0164_03", "scene0598_00", "scene0598_01", "scene0598_02",
        "scene0599_00", "scene0599_01", "scene0599_02", "scene0328_00",
        "scene0300_00", "scene0300_01", "scene0354_00", "scene0458_00",
        "scene0458_01", "scene0423_00", "scene0423_01", "scene0423_02",
        "scene0307_00", "scene0307_01", "scene0307_02", "scene0606_00",
        "scene0606_01", "scene0606_02", "scene0432_00", "scene0432_01",
        "scene0608_00", "scene0608_01", "scene0608_02", "scene0651_00",
        "scene0651_01", "scene0651_02", "scene0430_00", "scene0430_01",
        "scene0689_00", "scene0357_00", "scene0357_01", "scene0574_00",
        "scene0574_01", "scene0574_02", "scene0329_00", "scene0329_01",
        "scene0329_02", "scene0153_00", "scene0153_01", "scene0616_00",
        "scene0616_01", "scene0671_00", "scene0671_01", "scene0618_00",
        "scene0382_00", "scene0382_01", "scene0490_00", "scene0621_00",
        "scene0607_00", "scene0607_01", "scene0149_00", "scene0695_00",
        "scene0695_01", "scene0695_02", "scene0695_03", "scene0389_00",
        "scene0377_00", "scene0377_01", "scene0377_02", "scene0342_00",
        "scene0139_00", "scene0629_00", "scene0629_01", "scene0629_02",
        "scene0496_00", "scene0633_00", "scene0633_01", "scene0518_00",
        "scene0652_00", "scene0406_00", "scene0406_01", "scene0406_02",
        "scene0144_00", "scene0144_01", "scene0494_00", "scene0278_00",
        "scene0278_01", "scene0316_00", "scene0609_00", "scene0609_01",
        "scene0609_02", "scene0609_03", "scene0084_00", "scene0084_01",
        "scene0084_02", "scene0696_00", "scene0696_01", "scene0696_02",
        "scene0351_00", "scene0351_01", "scene0643_00", "scene0644_00",
        "scene0645_00", "scene0645_01", "scene0645_02", "scene0081_00",
        "scene0081_01", "scene0081_02", "scene0647_00", "scene0647_01",
        "scene0535_00", "scene0353_00", "scene0353_01", "scene0353_02",
        "scene0559_00", "scene0559_01", "scene0559_02", "scene0593_00",
        "scene0593_01", "scene0246_00", "scene0653_00", "scene0653_01",
        "scene0064_00", "scene0064_01", "scene0356_00", "scene0356_01",
        "scene0356_02", "scene0030_00", "scene0030_01", "scene0030_02",
        "scene0222_00", "scene0222_01", "scene0338_00", "scene0338_01",
        "scene0338_02", "scene0378_00", "scene0378_01", "scene0378_02",
        "scene0660_00", "scene0553_00", "scene0553_01", "scene0553_02",
        "scene0527_00", "scene0663_00", "scene0663_01", "scene0663_02",
        "scene0664_00", "scene0664_01", "scene0664_02", "scene0334_00",
        "scene0334_01", "scene0334_02", "scene0046_00", "scene0046_01",
        "scene0046_02", "scene0203_00", "scene0203_01", "scene0203_02",
        "scene0088_00", "scene0088_01", "scene0088_02", "scene0088_03",
        "scene0086_00", "scene0086_01", "scene0086_02", "scene0670_00",
        "scene0670_01", "scene0256_00", "scene0256_01", "scene0256_02",
        "scene0249_00", "scene0441_00", "scene0658_00", "scene0704_00",
        "scene0704_01", "scene0187_00", "scene0187_01", "scene0131_00",
        "scene0131_01", "scene0131_02", "scene0207_00", "scene0207_01",
        "scene0207_02", "scene0461_00", "scene0011_00", "scene0011_01",
        "scene0343_00", "scene0251_00", "scene0077_00", "scene0077_01",
        "scene0684_00", "scene0684_01", "scene0550_00", "scene0686_00",
        "scene0686_01", "scene0686_02", "scene0208_00", "scene0500_00",
        "scene0500_01", "scene0552_00", "scene0552_01", "scene0648_00",
        "scene0648_01", "scene0435_00", "scene0435_01", "scene0435_02",
        "scene0435_03", "scene0690_00", "scene0690_01", "scene0693_00",
        "scene0693_01", "scene0693_02", "scene0700_00", "scene0700_01",
        "scene0700_02", "scene0699_00", "scene0231_00", "scene0231_01",
        "scene0231_02", "scene0697_00", "scene0697_01", "scene0697_02",
        "scene0697_03", "scene0474_00", "scene0474_01", "scene0474_02",
        "scene0474_03", "scene0474_04", "scene0474_05", "scene0355_00",
        "scene0355_01", "scene0146_00", "scene0146_01", "scene0146_02",
        "scene0196_00", "scene0702_00", "scene0702_01", "scene0702_02",
        "scene0314_00", "scene0277_00", "scene0277_01", "scene0277_02",
        "scene0095_00", "scene0095_01", "scene0015_00", "scene0100_00",
        "scene0100_01", "scene0100_02", "scene0558_00", "scene0558_01",
        "scene0558_02", "scene0685_00", "scene0685_01", "scene0685_02",
    ]
    # if args.scene_ids:
    #     scene_ids = [s.strip() for s in args.scene_ids.split(',') if s.strip()]
    # elif args.jsonl_path:
    #     scene_ids = get_scannet_scene_ids_from_jsonl(args.jsonl_path)
    # if not scene_ids:
    #     print("No scenes to process. Provide --scene_ids or --jsonl_path.")
    #     sys.exit(0)

    # Splits
    train_s, val_s = set(), set()
    use_splits_for_frames_video = False
    if args.train_val_splits_path:
        try:
            with open(os.path.join(args.train_val_splits_path, 'scannetv2_train.txt')) as f:
                train_s = set(f.read().splitlines())
            with open(os.path.join(args.train_val_splits_path, 'scannetv2_val.txt')) as f:
                val_s = set(f.read().splitlines())
            use_splits_for_frames_video = True
            print(f"Loaded splits: train={len(train_s)}, val={len(val_s)}")
        except Exception as e:
            print(f"Warning: could not read split files: {e}")

    # Frames/video output dirs
    if args.enable_frames and not args.frames_output_dir:
        print("Error: --frames_output_dir is required when --enable_frames is set")
        sys.exit(1)
    if args.enable_video and not args.video_output_dir:
        print("Error: --video_output_dir is required when --enable_video is set")
        sys.exit(1)

    # Mesh label map
    labels_pd = None
    if args.enable_mesh:
        if not args.mesh_output_root or not args.label_map_file:
            print("Error: --mesh_output_root and --label_map_file are required when --enable_mesh is set")
            sys.exit(1)
        try:
            labels_pd = pd.read_csv(args.label_map_file, sep='\t', header=0)
        except Exception as e:
            print(f"Error reading label map {args.label_map_file}: {e}")
            sys.exit(1)

    # Build worker config
    cfg = WorkerConfig(
        base_download_dir=base_dir,
        scans_subdir=scans_subdir,
        output_frames_dir=args.frames_output_dir,
        output_video_dir=args.video_output_dir,
        output_mesh_root=args.mesh_output_root,
        video_width=args.video_width,
        video_height=args.video_height,
        video_fps=args.video_fps,
        video_frame_skip=args.video_frame_skip,
        video_codec=args.video_codec,
        num_frames=args.num_frames,
        image_size=tuple(args.image_size) if args.image_size else None,
        min_valid_components_per_frame=args.min_valid_components_per_frame,
        min_valid_frames_per_scene=args.min_valid_frames_per_scene,
        train_scenes=train_s,
        val_scenes=val_s,
        use_splits_for_frames_video=use_splits_for_frames_video,
        scannet_script=args.scannet_script,
        labels_pd=labels_pd,
        enable_frames=args.enable_frames,
        enable_video=args.enable_video,
        enable_mesh=args.enable_mesh,
    )

    # Concurrency loop
    print(f"Scenes to process: {len(scene_ids)} | max_inflight={args.max_inflight_scenes}")
    results = []
    with ProcessPoolExecutor(max_workers=args.max_inflight_scenes) as ex:
        futs = [ex.submit(worker_download_process_delete, sid, cfg) for sid in scene_ids]
        for fut in tqdm(as_completed(futs), total=len(futs), desc="Download→Process→Delete"):
            try:
                results.append(fut.result())
            except Exception as e:
                results.append({"scene_id": "?", "split": "?", "frames_ok": False, "video_ok": False, "mesh_ok": False, "message": f"worker_exception:{e}"})

    # Summary
    frames_ok = sum(1 for r in results if r['frames_ok'])
    video_ok  = sum(1 for r in results if r['video_ok'])
    mesh_ok   = sum(1 for r in results if r['mesh_ok'])

    print("\n===== Summary =====")
    print(f"Total scenes: {len(results)}")
    if cfg.enable_frames: print(f"Frames OK: {frames_ok}")
    if cfg.enable_video:  print(f"Video  OK: {video_ok}")
    if cfg.enable_mesh:   print(f"Mesh   OK: {mesh_ok}")
    for r in results:
        print(f"[{r['split']}] {r['scene_id']}: {r['message']} | frames={r['frames_ok']} video={r['video_ok']} mesh={r['mesh_ok']}")
    print("===================")


if __name__ == '__main__':
    main()
