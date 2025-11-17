import os
import subprocess
import argparse

def parse_arguments():
    parser = argparse.ArgumentParser(description="Run multiple Python scripts with common arguments.")
    parser.add_argument("--split_path", required=True, help="Path to the split file")
    parser.add_argument("--split_type", required=True, help="Split type")
    parser.add_argument("--processed_data_path", required=True, help="Path to the processed data")
    parser.add_argument("--output_dir", required=True, help="Output directory for QA pairs")
    parser.add_argument("--dataset", required=True, help="Dataset name")
    parser.add_argument("--num_subsample", type=int, default=10000, help="Number of subsamples")
    parser.add_argument('--num_workers', type=int, default=1, help='')
    return parser.parse_args()


def main():
    args = parse_arguments()
    os.makedirs(args.output_dir, exist_ok=True)

    # Configuration for each script: filename -> {subdir, num_subsample}
    script_configs = {
        # === VSIBench Tasks ===
        # "get_obj_abs_distance_qa.py": {"subdir": "vsibench", "num_subsample": "50"},
        # "get_obj_abs_frame_distance.py" : {"subdir": "vsibench", "num_subsample": "50"},
        # "get_obj_count_qa.py": {"subdir": "vsibench", "num_subsample": str(args.num_subsample)},
        # "get_obj_count_frame_qa.py": {"subdir": "vsibench", "num_subsample": str(args.num_subsample)},
        # "get_obj_rel_direction_v1_qa.py": {"subdir": "vsibench", "num_subsample": "20"},
        # "get_obj_rel_direction_v2_qa.py": {"subdir": "vsibench", "num_subsample": "20"},
        # "get_obj_rel_direction_v3_qa.py": {"subdir": "vsibench", "num_subsample": "20"},
        # "get_obj_rel_direction_v1_frame_qa.py": {"subdir": "vsibench", "num_subsample": "20"},
        # "get_obj_rel_direction_v2_frame_qa.py": {"subdir": "vsibench", "num_subsample": "20"},
        # "get_obj_rel_direction_v3_frame_qa.py": {"subdir": "vsibench", "num_subsample": "20"},
        # "get_obj_size_qa.py": {"subdir": "vsibench", "num_subsample": str(args.num_subsample)},
        # "get_obj_size_frame_qa.py": {"subdir": "vsibench", "num_subsample": str(args.num_subsample)},
        # "get_room_size_qa.py": {"subdir": "vsibench", "num_subsample": str(args.num_subsample)},
        # "get_obj_rel_distance_v1_qa.py": {"subdir": "vsibench", "num_subsample": "100"},
        # "get_obj_rel_distance_v2_qa.py": {"subdir": "vsibench", "num_subsample": "100"},
        # "get_obj_rel_distance_v3_qa.py": {"subdir": "vsibench", "num_subsample": "100"},
        # "get_obj_rel_distance_v1_frame_qa.py": {"subdir": "vsibench", "num_subsample": "100"},
        # "get_obj_rel_distance_v2_frame_qa.py": {"subdir": "vsibench", "num_subsample": "100"},
        # "get_obj_rel_distance_v3_frame_qa.py": {"subdir": "vsibench", "num_subsample": "100"},
        # "get_spatial_temporal_appearance_order_qa.py": {"subdir": "vsibench", "num_subsample": "30"},

        # # === VSDRBench Tasks (Added) ===
        #"get_camera_displacement_qa.py": {"subdir": "vsibench", "num_subsample": str(args.num_subsample)},
        #"get_cam_obj_abs_dist_qa.py": {"subdir": "vsibench", "num_subsample": str(args.num_subsample)},
        #"get_cam_obj_dist_change_qa.py": {"subdir": "vsdrbench", "num_subsample": str(args.num_subsample)},
        #"get_cam_obj_rel_dir_qa.py": {"subdir": "vsdrbench", "num_subsample": str(args.num_subsample)},
        "get_cam_obj_rel_dist_qa_v1.py": {"subdir": "vsibench", "num_subsample": str(args.num_subsample)},
        "get_cam_obj_rel_dist_qa_v2.py": {"subdir": "vsibench", "num_subsample": str(args.num_subsample)},
        "get_cam_obj_rel_dist_qa_v3.py": {"subdir": "vsibench", "num_subsample": str(args.num_subsample)},
        "get_obj_obj_rel_pos_lr_qa.py" : {"subdir": "vsibench", "num_subsample": str(args.num_subsample)},
        "get_obj_obj_rel_pos_nf_qa.py" : {"subdir": "vsibench", "num_subsample": str(args.num_subsample)},
        "get_obj_obj_rel_pos_ud_qa.py" : {"subdir": "vsibench", "num_subsample": str(args.num_subsample)},
        "get_camera_movement_direction_qa_v1.py" : {"subdir": "vsibench", "num_subsample": str(args.num_subsample)},
        "get_camera_movement_direction_qa_v2.py" : {"subdir": "vsibench", "num_subsample": str(args.num_subsample)},
        "get_camera_movement_direction_qa_v3.py" : {"subdir": "vsibench", "num_subsample": str(args.num_subsample)},
        # Add other VSDR scripts here if/when implemented (e.g., occlusion, motion)
    }

    # Filter out scripts based on dataset if necessary
    # (Keep existing filtering logic for arkitscenes if still relevant)
    if 'arkitscenes' in args.dataset.lower() or 'arkitscenes' in args.split_path.lower():
        arkit_exclude_script = "get_spatial_temporal_appearance_order_qa.py"
        if arkit_exclude_script in script_configs:
            print(f"Removing '{arkit_exclude_script}' for arkitscenes dataset.")
            del script_configs[arkit_exclude_script]
        # Add any VSDR scripts to exclude for arkitscenes if needed

    # List of Python files to run (derived from the keys of the config dict)
    python_files_to_run = list(script_configs.keys())

    # Filter out scripts based on dataset if necessary
    if 'arkitscenes' in args.dataset.lower() or 'arkitscenes' in args.split_path.lower():
        arkit_exclude_script = "get_spatial_temporal_appearance_order_qa.py"
        if arkit_exclude_script in script_configs:
            print(f"Removing '{arkit_exclude_script}' for arkitscenes dataset.")
            del script_configs[arkit_exclude_script]

    # List of Python files to run (derived from the keys of the config dict)
    python_files_to_run = list(script_configs.keys())

    # Common arguments
    common_args = [
        "--split_path", args.split_path,
        "--split_type", args.split_type,
        "--processed_data_path", args.processed_data_path,
        "--dataset", args.dataset,
        "--num_workers", str(args.num_workers)
    ]

    # Loop through each configured Python file and run it
    for python_file in python_files_to_run:
        config = script_configs[python_file]
        subdir = config.get("subdir")
        subsample_val = config.get("num_subsample", str(args.num_subsample)) # Use default if not specified

        if not subdir:
            # Handle cases where subdir might be missing or explicitly None
            print(f"Warning: Subdirectory not specified for {python_file}. Assuming it's directly in 'src/tasks'.")
            # Adjust module path construction if necessary, or skip/error
            # Example: module_name = f"src.tasks.{python_file[:-3]}"
            # For now, we'll follow the original logic's dependency on subdir being present
            print(f"Error: Subdirectory mapping not found or is None for {python_file}. Cannot construct module path.")
            continue # Skip this script

        module_name = f"qa_generation.{subdir}.{python_file[:-3]}"
        save_dir = os.path.join(args.output_dir, subdir)
        command = ["python", "-m", module_name] + common_args + ["--num_subsample", subsample_val] + ["--output_dir", save_dir]

        print(f"Running: {' '.join(command)}")
        try:
            subprocess.run(command, check=True, text=True)
            print(f"Successfully finished running {python_file}\n")
        except subprocess.CalledProcessError as e:
            print(f"Error running {python_file}:")
            print(f"Return code: {e.returncode}")
            print("Stopping execution due to error.")
            return

    print("All scripts have been executed successfully.")

if __name__ == "__main__":
    main()