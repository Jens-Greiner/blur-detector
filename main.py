import rawpy
import numpy as np
import cv2
import os
import sys
import shutil
from typing import Dict, List, Optional, Tuple
from multiprocessing import Pool, cpu_count
from pprint import pprint
from pathlib import Path

def _process_image_and_calculate_blurriness(image_path: str) -> Tuple[str, Optional[float]]:
    """
    Worker function to process a single image and calculate its blurriness.
    Returns a tuple (image_path, variance) where variance may be None on error.
    This function is designed to be run in a separate process.
    """
    try:
        if not os.path.exists(image_path):
            # Return the path with None to indicate failure
            return (image_path, None)

        file_extension = os.path.splitext(image_path)[1].lower()
        rgb_image = None
        
        if file_extension in ['.cr3', '.dng', '.nef', '.arw']:
            with rawpy.imread(image_path) as raw:
                # Use a fast demosaicing algorithm for better performance
                rgb_image = raw.postprocess(rawpy.Params(
                    demosaic_algorithm=rawpy.DemosaicAlgorithm.LINEAR, 
                    use_camera_wb=True, 
                    no_auto_bright=True,
                    half_size=True
                ))
        else:
            rgb_image = cv2.imread(image_path)
            if rgb_image is None:
                return (image_path, None)

        # Convert the RGB image to grayscale
        gray_image = cv2.cvtColor(rgb_image, cv2.COLOR_BGR2GRAY)

        # Apply the Laplacian filter
        laplacian = cv2.Laplacian(gray_image, cv2.CV_64F)

        # Calculate the variance of the Laplacian
        variance = laplacian.var()

        return (image_path, float(variance))

    except rawpy.FileOpenError:
        return (image_path, None)
    except Exception:
        return (image_path, None)


def get_image_paths(path: str) -> List[str]:
    """
    Finds all image files within a given path (either a single file or a directory)
    and returns a dictionary of their paths and corresponding file types.

    Args:
        path: The full path to a file or a directory to search.

    Returns:
        A dictionary where keys are the absolute paths to image files and
        values are their file extensions (e.g., '.jpg', '.png').
    """
    image_files: List[str] = []
    
    # A set of common image file extensions
    IMAGE_EXTENSIONS: List[str] = ['.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff', '.webp', '.svg', '.cr3', '.dng', '.nef', '.arw']

    # Normalize the path to handle different OS conventions
    normalized_path: str = os.path.abspath(path)

    # Check if the path exists
    if not os.path.exists(normalized_path):
        print(f"Error: Path does not exist - '{normalized_path}'")
        return image_files

    # Case 1: The path is a single file
    if os.path.isfile(normalized_path):
        filename, file_extension = os.path.splitext(normalized_path)
        if file_extension.lower() in IMAGE_EXTENSIONS:
            image_files.append(normalized_path)
        else:
            print(f"The provided file is not a supported image type: '{normalized_path}'")
        return image_files

    # Case 2: The path is a directory
    if os.path.isdir(normalized_path):
        print(f"Searching for images in directory: '{normalized_path}'")
        for dirpath, _, filenames in os.walk(normalized_path):
            for filename in filenames:
                _, file_extension = os.path.splitext(filename)
                if file_extension.lower() in IMAGE_EXTENSIONS:
                    full_path: str = os.path.join(dirpath, filename)
                    image_files.append(full_path)
    
    return image_files
def prompt_for_path(default: str = "raw") -> str:
    """
    Prompt the user for a folder or file path to process.

    Priority order:
    - If a command-line argument is provided, use it (if it exists).
    - Otherwise, prompt interactively with `input()` and validate.

    Returns an existing path string.
    """
    # Command-line override (first positional argument)
    if len(sys.argv) > 1:
        candidate = sys.argv[1]
        if os.path.exists(candidate):
            return candidate
        else:
            print(f"Warning: CLI path provided but does not exist: '{candidate}'")

    # Interactive prompt loop
    while True:
        try:
            user_input = input(f"Enter folder or file to process [default: {default}]: ").strip()
        except EOFError:
            # Non-interactive environment: fall back to default
            user_input = ""

        if user_input == "":
            candidate = default
        else:
            candidate = user_input

        if os.path.exists(candidate):
            return candidate
        else:
            print(f"Path does not exist: '{candidate}'. Please try again.")

def move_files(file_paths, destination_dir):
    """
    Moves a list of files to a given destination directory.

    Args:
        file_paths (list): A list of strings, where each string is the
                          full path to a file to be moved.
        destination_dir (str): The path to the directory where the files
                               should be moved.
    """
    # Ensure the destination directory exists. If not, create it.
    if not os.path.exists(destination_dir):
        print(f"Destination directory '{destination_dir}' does not exist. Creating it.")
        os.makedirs(destination_dir)

    # Loop through each file path in the provided list
    for file_path in file_paths:
        try:
            # Check if the file exists before attempting to move it
            if os.path.exists(file_path):
                # Construct the new path for the file in the destination directory
                file_name = os.path.basename(file_path)
                destination_path = os.path.join(destination_dir, file_name)

                # Move the file
                shutil.move(file_path, destination_path)
                # print(f"Successfully moved '{file_path}' to '{destination_path}'.")
            else:
                print(f"Warning: File '{file_path}' does not exist. Skipping.")

        except shutil.Error as e:
            # Handle specific shutil errors (e.g., file already exists)
            print(f"Error moving '{file_path}': {e}")
        except Exception as e:
            # Handle any other unexpected errors
            print(f"An unexpected error occurred while moving '{file_path}': {e}")

if __name__ == '__main__':

    # Friendly header and usage explanation
    print("\n=== Blur Detector ===")
    print("This script scans a folder (or a single image) and computes a blur score for each image using the Laplacian variance method.")
    print("You can provide a path as the first argument (e.g. `python main.py /path/to/images`), or enter one when prompted.")
    print("(Press Ctrl+C at any time to abort)\n")

    try:
        foldername = prompt_for_path("raw")
    except KeyboardInterrupt:
        print("\nAborted by user.")
        sys.exit(0)

    print(f"\nFinding image files under: '{foldername}'")
    image_paths_to_process = get_image_paths(foldername)

    total = len(image_paths_to_process)
    if total == 0:
        print("No images found. Exiting.")
        sys.exit(0)

    print(f"Found {total} image(s). Showing up to 5 examples:")
    for p in image_paths_to_process[:5]:
        print(f" - {p}")

    try:
        cont = input("Continue and analyze these images? [Y/n]: ").strip().lower()
    except KeyboardInterrupt:
        print("\nAborted by user.")
        sys.exit(0)
    if cont not in ["", "y", "yes"]:
        print("Aborted by user.")
        sys.exit(0)

    # Use multiprocessing to process the images
    num_processes = cpu_count()
    print(f"\nUsing {num_processes} process(es) to analyze {total} image(s). This may take a while...")

    results: Dict[str, Optional[float]] = {}
    # Use imap_unordered so we can show progress as results arrive
    try:
        with Pool(processes=num_processes) as pool:
            it = pool.imap_unordered(_process_image_and_calculate_blurriness, image_paths_to_process)
            seen = 0
            bar_len = 40
            for res in it:
                seen += 1
                if isinstance(res, tuple) and len(res) >= 2:
                    path, score = res
                    results[path] = score
                # progress bar
                pct = seen / total
                filled = int(pct * bar_len)
                bar = "#" * filled + "-" * (bar_len - filled)
                print(f"\rAnalyzing images [{bar}] {seen}/{total}", end="", flush=True)
            print()
    except KeyboardInterrupt:
        print("\nAnalysis aborted by user. Terminating...")
        sys.exit(0)

    print("\nAnalysis complete. Sample results (up to 10), evenly spread across scores:")

    # Prepare a sorted list of (path, score) where score is available
    sorted_results = sorted(
        [(p, s) for p, s in results.items() if s is not None], key=lambda x: x[1]
    )

    if not sorted_results:
        print("No valid blur scores to display.")
    else:
        L = len(sorted_results)
        n_show = min(10, L)

        # If only one item, show it. Otherwise pick indices evenly spaced including 0 and L-1
        if n_show == 1:
            indices = [0]
        else:
            indices = [round(i * (L - 1) / (n_show - 1)) for i in range(n_show)]

        # Ensure unique indices in ascending order
        indices = sorted(set(indices))

        for idx in indices:
            path, score = sorted_results[idx]
            print(f" - {path}: {score}")

    # Before asking for the threshold, display a rough histogram of the blur-score distribution
    scores = [s for s in results.values() if s is not None]
    if scores:
        bins = min(20, len(scores))
        lo = min(scores)
        hi = max(scores)
        print("\nBlur score distribution:")
        if lo == hi:
            print(f" All scores equal: {lo:.2f}")
        else:
            counts = [0] * bins
            for s in scores:
                # compute bin index in [0, bins-1]
                idx = int((s - lo) / (hi - lo) * (bins - 1))
                counts[idx] += 1

            max_count = max(counts)
            max_bar = 40

            # Build fixed-width labels so the bars align nicely
            labels: List[str] = []
            for i in range(bins):
                bin_lo = lo + i * (hi - lo) / bins
                bin_hi = lo + (i + 1) * (hi - lo) / bins
                labels.append(f"{bin_lo:.1f}-{bin_hi:.1f}")

            label_width = max(len(lbl) for lbl in labels)
            count_width = max(len(str(c)) for c in counts)

            for i, lbl in enumerate(labels):
                bar_count = int((counts[i] / max_count) * max_bar) if max_count > 0 else 0
                bar = "#" * bar_count
                print(f" {lbl.rjust(label_width)} | {bar.ljust(max_bar)} ({str(counts[i]).rjust(count_width)})")
    else:
        print("\nNo valid blur scores to build a distribution.")

    # Ask for a numeric threshold, validate input
    threshold = None
    while threshold is None:
        try:
            raw_thresh = input("\nEnter a numeric threshold (images with score < threshold are considered blurry): ")
            threshold = float(raw_thresh)
        except KeyboardInterrupt:
            print("\nAborted by user.")
            sys.exit(0)
        except ValueError:
            print("Invalid number. Please enter a valid numeric threshold (e.g., 100.0).")


    paths_below_threshold = [path for path in results if results[path] is not None and results[path] < threshold]

    print(f"Found {len(paths_below_threshold)} image(s) below the threshold {threshold}.")
    if len(paths_below_threshold) > 0:
        print("Example blurry images:")
        for p in paths_below_threshold[:10]:
            print(f" - {p}")

    # Ask what to do with blurry images
    try:
        while True:
            move_or_delete = input("Should blurry pictures be deleted (d) or moved (m)? [m/d]: ").strip().lower()
            if move_or_delete in ["d", "delete", "deleted"]:
                # Confirm destructive action
                try:
                    confirm = input("You chose to DELETE files. Type 'YES' to confirm deletion: ").strip()
                except KeyboardInterrupt:
                    print("\nDeletion cancelled by user.")
                    sys.exit(0)
                if confirm == "YES":
                    for path in paths_below_threshold:
                        try:
                            os.remove(path)
                            print(f"Deleted: {path}")
                        except Exception as e:
                            print(f"Error deleting {path}: {e}")
                else:
                    print("Deletion cancelled.")
                break
            elif move_or_delete in ["m", "move", "moved"]:
                default_dest = os.path.join(os.getcwd(), "blurry")
                try:
                    dest = input(f"Enter destination directory for blurry images [default: {default_dest}]: ").strip()
                except KeyboardInterrupt:
                    print("\nMove cancelled by user.")
                    sys.exit(0)
                if dest == "":
                    dest = default_dest
                try:
                    move_files(paths_below_threshold, dest)
                except Exception as e:
                    print(f"Error moving files: {e}")
                break
            else:
                print("Invalid option. Enter 'm' to move or 'd' to delete.")
    except KeyboardInterrupt:
        print("\nAborted by user.")
        sys.exit(0)




