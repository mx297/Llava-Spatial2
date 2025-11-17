import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

import sys
import os
import argparse
import glob
import json
from concurrent.futures import ProcessPoolExecutor,as_completed
from itertools import repeat
import subprocess
import shutil
import pandas as pd
import numpy as np
from tqdm import tqdm
# Load external constants
from .scannet200_constants import *
from .scannet200_splits import *
from .scannet200_utils_from_scannet import *

CLOUD_FILE_PFIX = '_vh_clean_2'
SEGMENTS_FILE_PFIX = '.0.010000.segs.json'
AGGREGATIONS_FILE_PFIX = '.aggregation.json'
CLASS_IDs = VALID_CLASS_IDS_200

def handle_process(scene_path, output_path, labels_pd, train_scenes, val_scenes):

    scene_id = scene_path.split('/')[-1]
    mesh_path = os.path.join(scene_path, f'{scene_id}{CLOUD_FILE_PFIX}.ply')
    segments_file = os.path.join(scene_path, f'{scene_id}{CLOUD_FILE_PFIX}{SEGMENTS_FILE_PFIX}')
    aggregations_file = os.path.join(scene_path, f'{scene_id}{AGGREGATIONS_FILE_PFIX}')
    info_file = os.path.join(scene_path, f'{scene_id}.txt')

    if scene_id in train_scenes:
        output_file = os.path.join(output_path, 'train', f'{scene_id}.ply')
        split_name = 'train'
    elif scene_id in val_scenes:
        output_file = os.path.join(output_path, 'val', f'{scene_id}.ply')
        split_name = 'val'
    else:
        output_file = os.path.join(output_path, 'test', f'{scene_id}.ply')
        split_name = 'test'

    print('Processing: ', scene_id, 'in ', split_name)

    # Rotating the mesh to axis aligned
    info_dict = {}
    with open(info_file) as f:
        for line in f:
            (key, val) = line.split(" = ")
            info_dict[key] = np.fromstring(val, sep=' ')

    if 'axisAlignment' not in info_dict:
        rot_matrix = np.identity(4)
    else:
        rot_matrix = info_dict['axisAlignment'].reshape(4, 4)

    pointcloud, faces_array = read_plymesh(mesh_path)
    points = pointcloud[:, :3]
    colors = pointcloud[:, 3:6]
    alphas = pointcloud[:, -1]

    # Rotate PC to axis aligned
    r_points = pointcloud[:, :3].transpose()
    r_points = np.append(r_points, np.ones((1, r_points.shape[1])), axis=0)
    r_points = np.dot(rot_matrix, r_points)
    pointcloud = np.append(r_points.transpose()[:, :3], pointcloud[:, 3:], axis=1)

    # Load segments file
    with open(segments_file) as f:
        segments = json.load(f)
        seg_indices = np.array(segments['segIndices'])

    # Load Aggregations file
    with open(aggregations_file) as f:
        aggregation = json.load(f)
        seg_groups = np.array(aggregation['segGroups'])

    # Generate new labels
    labelled_pc = np.zeros((pointcloud.shape[0], 1))
    instance_ids = np.zeros((pointcloud.shape[0], 1))
    for group in seg_groups:
        segment_points, p_inds, label_id = point_indices_from_group(pointcloud, seg_indices, group, labels_pd, CLASS_IDs)

        labelled_pc[p_inds] = label_id
        instance_ids[p_inds] = group['id']

    labelled_pc = labelled_pc.astype(int)
    instance_ids = instance_ids.astype(int)

    # Concatenate with original cloud
    processed_vertices = np.hstack((pointcloud[:, :6], labelled_pc, instance_ids))

    if (np.any(np.isnan(processed_vertices)) or not np.all(np.isfinite(processed_vertices))):
        raise ValueError('nan')

    # Save processed mesh
    save_plymesh(processed_vertices, faces_array, output_file, with_label=True, verbose=False)

    # Uncomment the following lines if saving the output in voxelized point cloud
    # quantized_points, quantized_scene_colors, quantized_labels, quantized_instances = voxelize_pointcloud(points, colors, labelled_pc, instance_ids, faces_array)
    # quantized_pc = np.hstack((quantized_points, quantized_scene_colors, quantized_labels, quantized_instances))
    # save_plymesh(quantized_pc, faces=None, filename=output_file, with_label=True, verbose=False)

def get_scannet_scene_ids_from_jsonl(jsonl_path):
    """Return unique ScanNet scene IDs from a JSONL with 'data_source' and 'scene_name' fields."""
    try:
        df = pd.read_json(jsonl_path, lines=True)
    except Exception as e:
        print(f"Error reading {jsonl_path}: {e}")
        return []
    if 'data_source' not in df.columns or 'scene_name' not in df.columns:
        print(f"Error: {jsonl_path} must contain 'data_source' and 'scene_name' columns.")
        return []
    return sorted(np.unique(df[df['data_source'] == 'scannet']['scene_name']).tolist())

def download_scannet_scene(scannet_script, out_dir, scene_id):
    """Call scannet.py to download a single scene into out_dir. Returns True on success."""
    cmd = [sys.executable, scannet_script, "-o", out_dir, "--id", scene_id]
    try:
        subprocess.run(cmd, input="\n\n", text=True, check=True)
        return True
    except subprocess.CalledProcessError as e:
        print(f"Download failed for {scene_id}: {e}")
        return False

def worker_download_process_delete(
    scene_id,
    base_download_dir,
    download_dir,
    output_root,
    labels_pd,
    train_scenes,
    val_scenes,
    scannet_script,
):
    """
    Full lifecycle for ONE scene:
      1) download -> 2) process (handle_process) -> 3) delete local folder
    Returns (scene_id, split, success, message)
    """
    import os, shutil, traceback

    # 1) Download
    ok = download_scannet_scene(scannet_script, base_download_dir, scene_id)
    if not ok:
        return (scene_id, "unknown", False, "download_failed")

    scene_path = os.path.join(download_dir, scene_id)
    # Decide split the same way handle_process does
    if scene_id in train_scenes:
        split = "train"
    elif scene_id in val_scenes:
        split = "val"
    else:
        split = "test"

    # 2) Process
    try:
        
        handle_process(scene_path, output_root, labels_pd, train_scenes, val_scenes)
        msg = "ok"
        success = True
    except Exception as e:
        msg = f"processing_exception:{e}"
        success = False
        traceback.print_exc()

    # 3) Delete local copy regardless of success
    try:
        shutil.rmtree(scene_path, ignore_errors=True)
    except Exception:
        pass

    return (scene_id, split, success, msg)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output_root', required=True, help='Output path where train/val folders will be located')
    parser.add_argument('--label_map_file', required=True, help='path to scannetv2-labels.combined.tsv')
    parser.add_argument('--train_val_splits_path', default='../../Tasks/Benchmark', help='Where the txt files with the train/val splits live')
    
    parser.add_argument('--jsonl_path', required=True,
                        help='Path to JSONL (e.g., vlm3r.jsonl) that lists ScanNet scene IDs.')
    parser.add_argument('--scannet_script', type=str, default='scannet.py',
                        help='Path to scannet.py downloader script.')
    parser.add_argument('--max_inflight_scenes', type=int, default=5,
                        help='Max number of scenes simultaneously present in --scans_dir.')
    config = parser.parse_args()

    BASE_DIR = "/l/users/mohamed.abouelhadid/scannet200_val/"
    DOWNLOAD_DIR = BASE_DIR + 'scans'

    # Load label map
    labels_pd = pd.read_csv(config.label_map_file, sep='\t', header=0)

    # Load train/val splits
    with open(config.train_val_splits_path + '/scannetv2_train.txt') as train_file:
        train_scenes = train_file.read().splitlines()
    with open(config.train_val_splits_path + '/scannetv2_val.txt') as val_file:
        val_scenes = val_file.read().splitlines()

    # Create output directories
    train_output_dir = os.path.join(config.output_root, 'train')
    if not os.path.exists(train_output_dir):
        os.makedirs(train_output_dir)
    val_output_dir = os.path.join(config.output_root, 'val')
    if not os.path.exists(val_output_dir):
        os.makedirs(val_output_dir)

    # Load scene paths
    # scene_paths = sorted(glob.glob(config.dataset_root + '/*'))

    # # Preprocess data.
    # pool = ProcessPoolExecutor(max_workers=config.num_workers)
    # print('Processing scenes...')
    # _ = list(pool.map(handle_process, scene_paths, repeat(config.output_root), repeat(labels_pd), repeat(train_scenes), repeat(val_scenes)))

    #scene_ids = get_scannet_scene_ids_from_jsonl(config.jsonl_path)
    scene_ids = [
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
    print(f"Found {len(scene_ids)} scene IDs in {config.jsonl_path}.")

    processed_counts = {'train': 0, 'val': 0, 'test': 0}
    failed_counts    = {'train': 0, 'val': 0, 'test': 0}

    print(f"Starting parallel Download→Process→Delete with up to {config.max_inflight_scenes} scenes in flight...")

    futures = []
    with ProcessPoolExecutor(max_workers=config.max_inflight_scenes) as ex:
        for sid in scene_ids:
            fut = ex.submit(
                worker_download_process_delete,
                sid,
                BASE_DIR,
                DOWNLOAD_DIR,
                config.output_root,
                labels_pd,
                train_scenes,
                val_scenes,
                config.scannet_script,
            )
            futures.append(fut)

        for fut in tqdm(as_completed(futures), total=len(futures), desc="Parallel Download→Process→Delete"):
            try:
                sid, split, ok, msg = fut.result()
                if split not in processed_counts:
                    split = 'test'  # safety
                if ok:
                    processed_counts[split] += 1
                else:
                    failed_counts[split] += 1
                print(f"[{split}] {sid}: {msg}")
            except Exception as e:
                print(f"Worker retrieval exception: {e}")

    print("\n===== Summary =====")
    print(f"Processed   - train: {processed_counts['train']}, val: {processed_counts['val']}, test: {processed_counts['test']}")
    print(f"Failed      - train: {failed_counts['train']}, val: {failed_counts['val']}, test: {failed_counts['test']}")
    print("===================")