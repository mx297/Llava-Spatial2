import argparse
import os
import pandas as pd
import numpy as np
from .common_utils import calculate_room_area, calculate_room_center
from .scannetv2_utils import SCANNETV2_OBJECT_CATEGORY, get_objects_number_and_bbox, SCANNETV2_VALID_CATEGORY_IDX
# Import specific ScanNet200 constants
from .scannet200_utils_from_vlm3r import SCANNET200_CLASS_REMAPPER_LIST, SCANNET200_CLASS_REMAPPER, remap_categories, SCANNET200_CLASS_NAMES, SCANNET200_VALID_CATEGORY_IDX
from plyfile import PlyData
import logging
from dataclasses import dataclass
from typing import List, Dict, Any

# Import base classes
from .base_processor import BaseProcessorConfig, AbstractSceneProcessor

logger = logging.getLogger(__name__)
# logging.basicConfig setup is handled in base_processor if run standalone, or by runner script

# Constants derived from LanguageGroundedSemseg/lib/constants/scannet_constants.py
# VALID_CLASS_IDS_LONG = ( ... ) # Removed
# CLASS_LABELS_LONG = ( ... ) # Removed

# Helper mapping removed as it's handled within the unified function now
# scannet200_raw_id_to_contiguous_id = {raw_id: i for i, raw_id in enumerate(SCANNET200_CLASS_NAMES)}

def create_unified_label_mapping(raw_id_to_category_name, target_category_list):
    """Creates a mapping from raw label ID (from ply/tsv) to a target category index."""
    category_name_to_target_idx = {name: idx for idx, name in enumerate(target_category_list)}
    raw_id_to_target_idx = {}
    for raw_id, category_name in raw_id_to_category_name.items():
        raw_id_to_target_idx[raw_id] = category_name_to_target_idx.get(category_name, -1) # -1 for invalid/ignored
    return raw_id_to_target_idx

def remap_labels(raw_labels, label_id_to_target_idx):
    """Remaps an array of raw labels using the provided mapping."""
    return np.array([label_id_to_target_idx.get(label, -1) for label in raw_labels])

# --- Configuration ---

@dataclass
class ScanNetProcessorConfig(BaseProcessorConfig):
    """Configuration specific to ScanNet metadata processing."""
    input_dir: str = "data/raw_data/scannet/scans"
    scene_list_file: str = "ScanNet/Tasks/Benchmark/scannetv2_train.txt"
    label_mapping_path: str = "data/raw_data/scannet/scannetv2-labels.combined.tsv"
    video_dir: str = "/mnt/disks/sp-data/shusheng/ScanNet/v2_val_videos" # Or provide a sensible default/None
    # Inherits save_dir, output_filename, num_workers, overwrite, random_seed from BaseProcessorConfig

# --- Processor Implementation ---

class ScanNetProcessor(AbstractSceneProcessor):
    """Processor for ScanNet dataset metadata."""
    def __init__(self, config: ScanNetProcessorConfig):
        super().__init__(config) # Pass the specific config to the base class
        self.config: ScanNetProcessorConfig # Type hint for clarity

        # Pre-load label mappings once
        self._load_label_mappings()

    def _load_label_mappings(self):
        """Loads and prepares the label mappings needed for processing."""
        logger.info(f"Loading label mapping from: {self.config.label_mapping_path}")
        try:
            label_mapping_df = pd.read_csv(self.config.label_mapping_path, sep="\t", engine='python') # Specify engine if needed
            # Ensure 'id' and 'raw_category' columns exist
            if 'id' not in label_mapping_df.columns or 'raw_category' not in label_mapping_df.columns:
                 raise ValueError("Label mapping TSV must contain 'id' and 'raw_category' columns.")
            # Convert IDs safely
            raw_id_to_category_name = {}
            for _, row in label_mapping_df.iterrows():
                 try:
                     raw_id = int(row['id'])
                     category_name = str(row['raw_category'])
                     raw_id_to_category_name[raw_id] = category_name
                 except (ValueError, TypeError):
                     logger.warning(f"Skipping invalid row in label mapping: {row}")
                     continue
            
            self.scannetv2_label_map = create_unified_label_mapping(raw_id_to_category_name, SCANNETV2_OBJECT_CATEGORY)
            self.scannet200_label_map = create_unified_label_mapping(raw_id_to_category_name, SCANNET200_CLASS_NAMES)
            logger.info("Label mappings loaded successfully.")
        except FileNotFoundError:
            logger.error(f"Label mapping file not found: {self.config.label_mapping_path}")
            raise # Re-raise the exception to stop processing if mapping is essential
        except Exception as e:
            logger.error(f"Error loading or processing label mapping file: {e}")
            raise

    def _load_scene_list(self) -> List[str]:
        """Loads the list of scene IDs from the specified file."""
        logger.info(f"Loading scene list from: {self.config.scene_list_file}")
        try:
            with open(self.config.scene_list_file, "r") as f:
                scene_ids = [line.strip() for line in f if line.strip()] # Read and strip non-empty lines
            logger.info(f"Loaded {len(scene_ids)} scene IDs.")
            return scene_ids
        except FileNotFoundError:
            logger.error(f"Scene list file not found: {self.config.scene_list_file}")
            return [] # Return empty list on error

    def _process_single_scene(self, scene_id: str) -> Dict[str, Any] | None:
        """Processes a single ScanNet scene."""
        meta_info = {}
        scene_ply_path = os.path.join(self.config.input_dir, f"{scene_id}.ply") # Standard ScanNet PLY file name
        
        try:
            # 1. Load scene data
            if not os.path.exists(scene_ply_path):
                 logger.error(f"PLY file not found for scene {scene_id} at {scene_ply_path}. Skipping.")
                 return None

            plydata = PlyData.read(scene_ply_path)
            vertices = np.array([list(x) for x in plydata['vertex']])

            # Check vertex data structure (assuming ScanNet standard format: x,y,z,r,g,b,label,instance)
            # Adjust the check if your PLY structure is different
            if vertices.shape[1] != 8:
                 logger.warning(f"Unexpected vertex data shape {vertices.shape} in {scene_ply_path}. Assuming standard format indices might fail. Skipping scene {scene_id}.")
                 return None

            # Indices might need adjustment based on the exact PLY format.
            # Common ScanNet format: x,y,z,r,g,b,label,instance_id
            raw_label = vertices[:, 6].astype(int) # Semantic label ID from ply/tsv
            inst_label = vertices[:, 7].astype(int) # Instance label ID
            aligned_xyz = vertices[:, :3]
            # rgb = vertices[:, 3:6] # Optional: keep if needed

            # --- ScanNetV2 Processing ---
            label_v2 = remap_labels(raw_label, self.scannetv2_label_map) # Mapped to V2 indices (-1 if invalid)

            # 2. Video path
            video_path = os.path.join(self.config.video_dir, f"{scene_id}.mp4")
            # if not os.path.exists(video_path):
                # logger.warning(f"Video file not found for scene {scene_id}: {video_path}. Path will be stored but may be invalid.")
                # Decide if missing video is critical: return None or just log warning.
                # return None # Example: Skip if video is mandatory

            meta_info["video_path"] = video_path # Store relative or absolute path based on config
            meta_info["dataset"] = "scannet"

            # 3. Calculate room area and center
            room_area = calculate_room_area(aligned_xyz)
            room_center = calculate_room_center(aligned_xyz) # Assuming this function returns List[float]
            meta_info["room_size"] = room_area
            meta_info["room_center"] = room_center

            # 4. Count V2 objects and get BBoxes
            object_counts_info_v2, object_bbox_info_v2 = get_objects_number_and_bbox(
                aligned_xyz, label_v2, inst_label, SCANNETV2_VALID_CATEGORY_IDX, SCANNETV2_OBJECT_CATEGORY
            )
            meta_info["object_counts"] = object_counts_info_v2
            meta_info["object_bboxes"] = object_bbox_info_v2

            # --- ScanNet200 Processing ---
            label_200 = remap_labels(raw_label, self.scannet200_label_map) # Mapped to contiguous 200 indices

            object_counts_info_200, object_bbox_info_200 = get_objects_number_and_bbox(
                aligned_xyz, label_200, inst_label, SCANNET200_VALID_CATEGORY_IDX, SCANNET200_CLASS_NAMES
            )

            # --- Merging and Remapping Overlaps for ScanNet200 ---
            existing_categories_in_200 = list(object_counts_info_200.keys())
            overlapped_categories = [name for name in SCANNET200_CLASS_REMAPPER_LIST if name in existing_categories_in_200]

            if overlapped_categories:
                # remap_categories modifies the dicts in place
                object_bbox_info_200_remapped, object_counts_info_200_remapped = remap_categories(
                    object_bbox_info_200, object_counts_info_200, SCANNET200_CLASS_REMAPPER
                )
                meta_info["object_counts"].update(object_counts_info_200_remapped)
                meta_info["object_bboxes"].update(object_bbox_info_200_remapped)
            else:
                meta_info["object_counts"].update(object_counts_info_200)
                meta_info["object_bboxes"].update(object_bbox_info_200)

            return meta_info

        except FileNotFoundError as e:
            logger.error(f"File not found error during processing {scene_id}: {e}")
            return None # Skip scene if essential file is missing
        except ValueError as e:
             logger.error(f"Value error during processing {scene_id}: {e}")
             return None # Skip scene on data format issues
        except Exception as e:
            logger.error(f"Unexpected error processing scene {scene_id}: {e}", exc_info=True) # Log full traceback for unexpected errors
            return None # Skip scene on unexpected errors

# --- Main Execution Logic ---

def main():
    parser = argparse.ArgumentParser(description="Process ScanNet dataset to generate metadata for a specific split (e.g., train or val).")
    # Arguments defining the specific dataset split and paths
    parser.add_argument('--input_dir', type=str, required=True, help='Input directory containing PLY files for the split.')
    parser.add_argument('--scene_list_file', type=str, required=True, help='File containing the list of scene IDs for the split.')
    parser.add_argument('--save_dir', type=str, required=True, help='Directory to save the output JSON metadata.')
    parser.add_argument('--output_filename', type=str, required=True, help='Name of the output JSON file (e.g., scannet_train_metadata.json).')
    parser.add_argument('--label_mapping_path', type=str, default="data/scannet/scannetv2-labels.combined.tsv", help='Path to the label mapping TSV file.')
    parser.add_argument('--video_dir', type=str, default=None, help='Directory containing corresponding video files (optional).')

    # General processing arguments
    parser.add_argument('--num_workers', type=int, default=4, help='Number of workers for multiprocessing.')
    parser.add_argument('--overwrite', action='store_true', help='Allow overwriting existing output file.')
    parser.add_argument('--random_seed', type=int, default=42, help='Random seed for reproducibility.')

    args = parser.parse_args()

    # Create config object directly from arguments
    config = ScanNetProcessorConfig(
        input_dir=args.input_dir,
        scene_list_file=args.scene_list_file,
        save_dir=args.save_dir,
        output_filename=args.output_filename,
        label_mapping_path=args.label_mapping_path,
        video_dir=args.video_dir,
        num_workers=args.num_workers,
        overwrite=args.overwrite,
        random_seed=args.random_seed
    )

    # Initialize and run the processor
    processor = ScanNetProcessor(config)
    processor.process_all_scenes()

if __name__ == "__main__":
    # Configure logging here if not handled by a higher-level script
    # Example basic config (adjust level and format as needed):
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    main()