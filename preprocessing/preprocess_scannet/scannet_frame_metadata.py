# 
import argparse
import os
import numpy as np
import json
import tqdm
import logging
from PIL import Image # Added for mask processing
import warnings # Added to suppress PIL warnings
from dataclasses import dataclass, field
from typing import List, Dict, Any, Tuple # Added Tuple
from pathlib import Path # Added for easier path handling

# Import base classes
from .base_processor import BaseProcessorConfig, AbstractSceneProcessor


# Suppress specific PIL warnings if necessary, e.g., DecompressionBombWarning
# warnings.filterwarnings("ignore", category=Image.DecompressionBombWarning)

# Configure logging in the main execution block (__name__ == "__main__")
logger = logging.getLogger(__name__)

# --- Helper Functions (Remain outside the class as they don't need instance state) ---

def read_matrix_from_txt(file_path: str) -> list[list[float]] | None:
    """Reads a 4x4 matrix from a text file."""
    try:
        matrix = np.loadtxt(file_path)
        if matrix.shape == (4, 4):
            return matrix.tolist()
        else:
            logger.warning(f"Unexpected matrix shape {matrix.shape} in {file_path}")
            return None
    except FileNotFoundError:
        # Log as warning, handled in calling function
        # Logger message updated in calling function for context
        return None
    except Exception as e:
        logger.error(f"Error reading matrix from {file_path}: {e}")
        return None

# def read_axis_align_matrix(file_path: str) -> list[list[float]] | None:
#     """Reads a 4x4 axis alignment matrix from a text file."""
#     matrix_list = read_matrix_from_txt(file_path)
#     if matrix_list:
#         # Basic validation, read_matrix_from_txt already checks shape
#         return matrix_list
#     else:
#         # Log contextually in the calling function if file not found or parsing failed
#         # Logger message updated in calling function for context
#         return None

def read_intrinsic_params(intrinsic_path: str) -> dict | None:
    """Reads camera intrinsics from a matrix file."""
    matrix = read_matrix_from_txt(intrinsic_path)
    if matrix:
        # ScanNet intrinsic matrix is often 3x3 or 4x4, extract top-left 3x3
        try:
            if len(matrix) >= 3 and len(matrix[0]) >= 3 and len(matrix[1]) >= 3:
                return {
                    "fx": matrix[0][0],
                    "fy": matrix[1][1],
                    "cx": matrix[0][2],
                    "cy": matrix[1][2]
                }
            else:
                logger.error(f"Matrix in {intrinsic_path} does not have sufficient dimensions for intrinsics.")
                return None
        except IndexError:
             logger.error(f"Error accessing elements in matrix from {intrinsic_path} for intrinsics.")
             return None
    else:
        # read_matrix_from_txt handles logging FileNotFoundError
        if os.path.exists(intrinsic_path): # Log error only if file exists but couldn't be parsed
            logger.error(f"Could not read matrix from {intrinsic_path} for intrinsics.")
    return None

# --- Helper function get_bboxes_from_mask removed, integrated into _process_single_scene ---
# --- Helper function read_depth_map removed, integrated into _process_single_scene ---

# --- Configuration ---

@dataclass
class ScanNetFrameProcessorConfig(BaseProcessorConfig):
    """Configuration specific to ScanNet frame metadata processing."""
    processed_dir: str = "data/stage2_data/ScanNet" # Default changed
    # scans_dir: str = "data/scannet/scans" # New: For original scans data # Removed
    scene_list_file: str = "ScanNet/Tasks/Benchmark/scannetv2_train.txt" # Default scene list
    img_height: int = 480 # New: Target image height
    img_width: int = 640 # New: Target image width
    # Inherits save_dir, output_filename, num_workers, overwrite, random_seed from BaseProcessorConfig
    # Removed num_frames (now inferred from existing sampled data)
    # Removed axis_align_matrix_dir (consolidated into scans_dir)
    split: str = field(init=False) # Will be inferred from scene_list_file

    def __post_init__(self):
        # Infer split from scene_list_file
        try:
            # Extract split name (e.g., 'train', 'val') from filename like 'scannetv2_train.txt'
            filename = os.path.basename(self.scene_list_file)
            base, _ = os.path.splitext(filename)
            parts = base.split('_')
            if len(parts) > 1 and parts[-1] in ['train', 'val', 'test']: # Basic check
                 self.split = parts[-1]
                 logger.info(f"Inferred split '{self.split}' from scene_list_file: {self.scene_list_file}")
            else:
                 # Fallback or error if split cannot be determined
                 logger.warning(f"Could not infer split from scene_list_file name: {self.scene_list_file}. Defaulting to 'unknown'.")
                 self.split = 'unknown' # Or raise an error?
        except Exception as e:
            logger.error(f"Error inferring split from scene_list_file: {e}. Defaulting to 'unknown'.")
            self.split = 'unknown'

        # Ensure save_dir defaults relative to processed_dir if not explicitly set different
        if self.save_dir == BaseProcessorConfig.save_dir: # Check if using base default
             self.save_dir = self.processed_dir
             logger.info(f"Defaulting save_dir to processed_dir: {self.save_dir}")

        # Update output filename to include split if not already custom
        if self.output_filename == "scannet_frame_metadata.json" and self.split != 'unknown':
             self.output_filename = f"scannet_frame_metadata_{self.split}.json"
             logger.info(f"Updated output filename to include split: {self.output_filename}")


# --- Processor Implementation ---

class ScanNetFrameProcessor(AbstractSceneProcessor[ScanNetFrameProcessorConfig]):
    """
    Processor for ScanNet dataset frame metadata. Reads sampled pose/depth/color
    from processed_dir and original intrinsics/masks/axis_alignment from scans_dir.
    """
    def __init__(self, config: ScanNetFrameProcessorConfig):
        super().__init__(config)
        # Validate essential directories early
        if not os.path.isdir(self.config.processed_dir):
             logger.warning(f"Processed data directory not found: {self.config.processed_dir}. May cause errors.")
        # if not os.path.isdir(self.config.scans_dir):
        #     logger.warning(f"ScanNet scans directory not found: {self.config.scans_dir}. May cause errors.") # Removed scans_dir check


    def _load_scene_list(self) -> List[str]:
        """Loads the list of scene IDs from the specified file."""
        logger.info(f"Loading scene list from: {self.config.scene_list_file}")
        try:
            with open(self.config.scene_list_file, "r") as f:
                scene_ids = [line.strip() for line in f if line.strip()]
            logger.info(f"Loaded {len(scene_ids)} scene IDs.")
            # Optional: Filter scene_ids based on existence in processed_dir?
            # valid_scene_ids = [sid for sid in scene_ids if os.path.isdir(os.path.join(self.config.processed_dir, 'pose', sid))]
            # if len(valid_scene_ids) < len(scene_ids):
            #     logger.warning(f"Found {len(valid_scene_ids)} scenes with pose data in {self.config.processed_dir} out of {len(scene_ids)} listed.")
            # return valid_scene_ids
            return scene_ids
        except FileNotFoundError:
            logger.error(f"Scene list file not found: {self.config.scene_list_file}")
            return [] # Return empty list on error

    def _process_single_scene(self, scene_id: str) -> Dict[str, Any] | None:
        """Processes metadata for sampled frames in a single scene."""
        # Define base paths using config
        processed_scene_base = self.config.processed_dir
        # scans_scene_base is no longer needed for these files
        # scans_scene_base = os.path.join(self.config.scans_dir, scene_id) # Removed

        # Define paths for different data types using processed_dir and the inferred split
        split_dir = self.config.split
        pose_dir = os.path.join(processed_scene_base, "pose", split_dir, scene_id)
        depth_dir = os.path.join(processed_scene_base, "depth", split_dir, scene_id)
        color_dir = os.path.join(processed_scene_base, "color", split_dir, scene_id)
        # Paths for data now also sourced from processed_dir including split
        intrinsic_path = os.path.join(processed_scene_base, "intrinsic", split_dir, f"intrinsic_depth_{scene_id}.txt") # Updated path
        mask_dir = os.path.join(processed_scene_base, "instance", split_dir, scene_id) # Updated path for sampled masks
        # Updated path and filename extension to .txt
        # axis_align_matrix_path = os.path.join(processed_scene_base, "axis_align_matrix", split_dir, scene_id, f"{scene_id}_axis_align_matrix.txt") # Removed axis alignment

        # --- 1. Read Intrinsics (Essential for the scene) ---
        intrinsics = read_intrinsic_params(intrinsic_path)
        if not intrinsics:
            logger.error(f"Could not read valid intrinsics for scene {scene_id} from {intrinsic_path}. Skipping scene.")
            return None

        # --- 2. Read Axis Alignment Matrix (Optional) --- # REMOVED
        # axis_align_matrix_list = read_axis_align_matrix(axis_align_matrix_path)
        # axis_align_matrix_np = np.array(axis_align_matrix_list) if axis_align_matrix_list else None
        # if axis_align_matrix_np is None:
        #      # Added check for file existence before warning about missing matrix
        #      if not os.path.exists(axis_align_matrix_path):
        #          logger.warning(f"Axis alignment matrix file not found for {scene_id} at {axis_align_matrix_path}. Poses will not be aligned.")
        #      else:
        #          logger.warning(f"Axis alignment matrix file found but could not be read/parsed for {scene_id} at {axis_align_matrix_path}. Poses will not be aligned.")

        # --- 3. Identify Sampled Frames ---
        # Use pose directory to find frame files (assume .txt format)
        if not os.path.isdir(pose_dir):
            logger.warning(f"Sampled pose directory not found for scene {scene_id}: {pose_dir}. Returning intrinsics only.")
            return {"camera_intrinsics": intrinsics, "frames": [], "img_width": self.config.img_width, "img_height": self.config.img_height}

        try:
            # Get frame IDs (numbers) from pose filenames (e.g., 000000.txt)
            frame_files = [f for f in os.listdir(pose_dir) if f.endswith(".txt")]
            # Sort numerically based on filename stem
            frame_ids = sorted([int(os.path.splitext(f)[0]) for f in frame_files])
        except (ValueError, OSError) as e:
            logger.error(f"Could not list or parse frame IDs from pose files in {pose_dir}. Error: {e}. Skipping scene {scene_id}.")
            return None # Fatal for the scene if we can't identify frames

        if not frame_ids:
            logger.warning(f"No valid pose files (.txt) found in {pose_dir} for scene {scene_id}. Returning intrinsics only.")
            return {"camera_intrinsics": intrinsics, "frames": [], "img_width": self.config.img_width, "img_height": self.config.img_height}

        logger.info(f"Found {len(frame_ids)} sampled frames for scene {scene_id}.")

        # --- 4. Process Each Sampled Frame ---
        frame_data = []
        processed_a_mask = False # Track if any mask was successfully processed

        for frame_id in frame_ids:
            frame_id_str = f"{frame_id:06d}" # Consistent 6-digit format

            # Define paths for this specific frame
            pose_path = os.path.join(pose_dir, f"{frame_id_str}.txt")
            depth_path = os.path.join(depth_dir, f"{frame_id_str}.png") # Sampled depth
            # Relative path for metadata, includes split
            color_path_rel = os.path.join("color", split_dir, scene_id, f"{frame_id_str}.jpg")
            color_path_abs = os.path.join(color_dir, f"{frame_id_str}.jpg") # Absolute path for existence check
            mask_path = os.path.join(mask_dir, f"{frame_id_str}.png") # Updated path for sampled mask

            # Check essential files exist
            if not os.path.exists(pose_path):
                 logger.warning(f"Pose file missing for frame {frame_id} in {scene_id}: {pose_path}. Skipping frame.")
                 continue # Skip frame if pose is missing/invalid
            if not os.path.exists(depth_path):
                 logger.warning(f"Depth file missing for frame {frame_id} in {scene_id}: {depth_path}. Skipping frame.")
                 continue
            if not os.path.exists(color_path_abs):
                 logger.warning(f"Color file missing for frame {frame_id} in {scene_id}: {color_path_abs}. Skipping frame.")
                 continue

            # --- 4a. Read Pose ---
            pose_matrix_list = read_matrix_from_txt(pose_path)
            if pose_matrix_list is None:
                # Log error specific to this pose file
                logger.error(f"Could not read pose matrix from {pose_path} for frame {frame_id}, scene {scene_id}. Skipping frame.")
                continue # Skip frame if pose is missing/invalid

            # --- 4b. Process Mask and BBoxes (using sampled mask) ---
            bboxes_2d = []
            mask_np_resized = None # Store resized original mask if loaded and valid
            if os.path.exists(mask_path):
                try:
                    # Load sampled mask, resize it to target dims for bbox extraction and depth calculation
                    with Image.open(mask_path) as mask:
                        # Check if mask needs resizing (sampled mask might already be correct size)
                        # For now, assume we still need to resize to ensure consistency with config img_width/height
                        mask_resized = mask.resize((self.config.img_width, self.config.img_height), Image.NEAREST)
                        mask_np_resized = np.array(mask_resized)

                    if mask_np_resized.ndim == 2:
                        processed_a_mask = True # Mark that we successfully processed at least one mask file
                        instance_ids = np.unique(mask_np_resized)
                        valid_instance_ids = [inst_id for inst_id in instance_ids if inst_id > 0] # Filter background (0)

                        for inst_id in valid_instance_ids:
                            coords = np.argwhere(mask_np_resized == inst_id)
                            if coords.size == 0: continue
                            ymin, xmin = coords.min(axis=0)
                            ymax, xmax = coords.max(axis=0)

                            # Ensure valid bbox before adding
                            if xmax >= xmin and ymax >= ymin:
                                bboxes_2d.append({
                                    "instance_id": int(inst_id) - 1, # Convert to 0-based index
                                    "bbox_2d": [int(xmin), int(ymin), int(xmax), int(ymax)],
                                })
                    else:
                        logger.warning(f"Unexpected mask dimensions {mask_np_resized.ndim} after resizing {mask_path}. Cannot extract bboxes for frame {frame_id}.")
                        mask_np_resized = None # Invalidate for depth calculation

                except FileNotFoundError: # Should be caught by os.path.exists, but good practice
                     logger.warning(f"Instance mask file not found (unexpectedly): {mask_path}")
                     mask_np_resized = None
                except Exception as e:
                    # Use logger.exception to include traceback
                    logger.exception(f"Error processing sampled mask file {mask_path} for frame {frame_id}: {e}")
                    bboxes_2d = [] # Ensure empty on error
                    mask_np_resized = None # Invalidate mask

            elif os.path.isdir(mask_dir): # Only warn if mask dir exists but file doesn't
                 logger.warning(f"Sampled instance mask file not found for frame {frame_id}: {mask_path}")
            # else: # Don't warn if the mask directory itself doesn't exist (might be intentional)
            #    pass

            # --- 4c. Read Sampled Depth Map & Calculate Average Depth ---
            depth_map_m = None
            try:
                # Read the already sampled (and possibly resized) depth map
                with Image.open(depth_path) as depth_img:
                    # Assuming depth is saved correctly (e.g., 16-bit PNG)
                    # No need to resize here if export script already did
                    # Verify dimensions if needed:
                    # if depth_img.width != self.config.img_width or depth_img.height != self.config.img_height:
                    #     logger.warning(f"Depth map {depth_path} dimensions ({depth_img.width}x{depth_img.height}) mismatch target ({self.config.img_width}x{self.config.img_height}).")
                    #     # Option: Resize here if necessary, but ideally export is correct
                    #     # depth_img = depth_img.resize((self.config.img_width, self.config.img_height), Image.NEAREST)

                    depth_np = np.array(depth_img, dtype=np.float32) # Use float32 for calculations

                # Convert from millimeters (assuming ScanNet standard 1000 scale) to meters
                # TODO: Make scale_factor configurable if needed
                depth_map_m = depth_np / 1000.0

            except FileNotFoundError: # Should be caught by os.path.exists
                logger.warning(f"Sampled depth map file not found (unexpectedly): {depth_path}")
            except Exception as e:
                logger.exception(f"Error processing depth map file {depth_path} for frame {frame_id}: {e}")
                depth_map_m = None # Ensure depth isn't used

            # --- 4d. Apply Axis Alignment to Pose --- # REMOVED
            final_pose_list = None
            if pose_matrix_list: # Use the pose read earlier
                final_pose_list = pose_matrix_list # Directly use the read pose
            # else: # pose_matrix_list was None
                # Error logged when pose was read, loop continues

            # Ensure final_pose_list is valid before proceeding
            if final_pose_list is None:
                logger.error(f"Pose data is invalid or missing for frame {frame_id}, scene {scene_id} after processing steps. Skipping frame.")
                continue

            # --- 4e. Assemble Frame Information ---
            frame_info = {
                "frame_id": frame_id,
                "file_path_color": color_path_rel, # Relative path to color image
                "file_path_depth": os.path.join("depth", split_dir, scene_id, f"{frame_id_str}.png"), # Relative path to depth
                # Optionally add relative pose path:
                # "file_path_pose": os.path.join("pose", scene_id, f"{frame_id_str}.txt"),
                "camera_pose_camera_to_world": final_pose_list,
                "bboxes_2d": bboxes_2d
            }
            frame_data.append(frame_info)

        # --- 5. Final Scene Summary ---
        if not processed_a_mask and os.path.isdir(mask_dir):
            # Warn only if mask directory exists but no masks were successfully processed for any frame
            logger.warning(f"No sampled instance masks found or processed in {mask_dir} for any sampled frame in scene {scene_id}.")

        return {
            "camera_intrinsics": intrinsics,
            "frames": frame_data,
            "img_width": self.config.img_width,
            "img_height": self.config.img_height
        }

# --- Main Execution Logic ---

def main():
    # Set up basic logging configuration
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    parser = argparse.ArgumentParser(description="Process ScanNet frame metadata using sampled data and original scans.")

    # Arguments for input data locations
    parser.add_argument('--processed_dir', type=str, default=ScanNetFrameProcessorConfig.processed_dir,
                        help='Directory containing sampled pose/depth/color subdirs.')
    # parser.add_argument('--scans_dir', type=str, default=ScanNetFrameProcessorConfig.scans_dir,
    #                     help='Directory containing original ScanNet scans (for intrinsics, masks, axis alignment).') # Removed
    parser.add_argument('--scene_list_file', type=str, default=ScanNetFrameProcessorConfig.scene_list_file,
                        help='Path to the text file listing scene IDs to process.')

    # Arguments for image properties
    parser.add_argument('--img_height', type=int, default=ScanNetFrameProcessorConfig.img_height,
                        help='Target image height for processing (e.g., mask resizing).')
    parser.add_argument('--img_width', type=int, default=ScanNetFrameProcessorConfig.img_width,
                        help='Target image width for processing.')

    # Arguments for output and processing behavior (from BaseProcessorConfig)
    parser.add_argument('--save_dir', type=str, default=BaseProcessorConfig.save_dir, # Use base default initially
                        help=f'Directory to save the output JSON metadata (defaults to processed_dir: {ScanNetFrameProcessorConfig.processed_dir}).')
    parser.add_argument('--output_filename', type=str, default="scannet_frame_metadata.json", # Keep default name
                        help='Name of the output JSON file.')
    parser.add_argument('--num_workers', type=int, default=BaseProcessorConfig.num_workers,
                        help='Number of worker processes for parallel processing.')
    parser.add_argument('--overwrite', action='store_true', default=False,
                        help='Allow overwriting existing output file.')
    parser.add_argument('--random_seed', type=int, default=BaseProcessorConfig.random_seed,
                        help='Random seed for operations.')

    args = parser.parse_args()

    # Create config object using parsed arguments
    config = ScanNetFrameProcessorConfig(
        processed_dir=args.processed_dir,
        # scans_dir=args.scans_dir, # Removed
        scene_list_file=args.scene_list_file,
        img_height=args.img_height,
        img_width=args.img_width,
        save_dir=args.save_dir, # Pass user value or base default
        output_filename=args.output_filename,
        num_workers=args.num_workers,
        overwrite=args.overwrite,
        random_seed=args.random_seed
        # num_frames is removed from config
        # axis_align_matrix_dir is removed from config
        # split is inferred in __post_init__
    )
    # The __post_init__ in the config class will handle setting save_dir default and split inference.

    # Initialize and run the processor
    processor = ScanNetFrameProcessor(config)
    processor.process_all_scenes()

if __name__ == "__main__":
    main() 