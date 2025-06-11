import time
import numpy as np
from PIL import Image
from concurrent.futures import ThreadPoolExecutor
import os
import json
import builtins
from datetime import datetime

import golomb_rice
import predictor


# --- Global list to store log messages ---
log_entries = []

# --- Custom print function to capture output ---
# Keep a reference to the original print function
original_print = builtins.print


def custom_print(*args, **kwargs):
    """
    Custom print function that prints to the console and saves the message
    to a global list for later JSON output.
    """
    # Create the message string from arguments
    message = " ".join(map(str, args))
    # Call the original print function to display output on the console
    original_print(message, **kwargs)
    # Add the message to our log list
    log_entries.append(message)


builtins.print = custom_print


def save_image(numpy_data, file_path):
    """Saves a numpy array as an image file."""
    try:
        img = Image.fromarray(numpy_data)
        img.save(file_path)
        print(f"Saved image to '{file_path}'")
    except Exception as e:
        print(f"Error saving image to '{file_path}': {e}")


def compress_image(image_path, m, num_chunks):
    """
    Loads, processes, and compresses an image into independent chunks.
    Also saves a visual representation of the residuals.
    """
    print(f"Compressing '{image_path}'...")
    # Load image and convert to grayscale numpy array
    try:
        img = Image.open(image_path).convert("L")
        image_data = np.array(img)
    except FileNotFoundError:
        print(f"Error: Image file not found at '{image_path}'")
        return None, None

    # Get prediction errors
    residuals = predictor.get_residuals(image_data)

    # To make residuals viewable, we map them to the 0-255 range.
    # A residual of 0 (perfect prediction) will be mapped to mid-gray (128).
    residuals_visual = residuals.astype(np.int16) + 128
    # Clip values to ensure they are within the valid 0-255 range for an image
    residuals_visual = np.clip(residuals_visual, 0, 255).astype(np.uint8)

    # Get the base name of the input file to create descriptive output names
    base_name = os.path.splitext(os.path.basename(image_path))[0]
    residuals_path = f"results/{base_name}_residuals.png"
    save_image(residuals_visual, residuals_path)

    # Split residuals into chunks for parallel processing
    chunk_data = np.array_split(residuals, num_chunks)

    # Encode each chunk independently
    encoded_chunks = [
        golomb_rice.rice_encode(chunk.flatten(), m) for chunk in chunk_data
    ]

    # The compressed object contains all info needed for decompression
    compressed_object = {
        "original_shape": image_data.shape,
        "m": m,
        "num_residuals_per_chunk": [len(chunk.flatten()) for chunk in chunk_data],
        "chunks": encoded_chunks,
    }

    return compressed_object, image_data


def decompress_serial(compressed_obj):
    """Decompresses chunks one by one in a single thread."""
    m = compressed_obj["m"]
    decoded_residuals = []

    for i, chunk_bits in enumerate(compressed_obj["chunks"]):
        num_residuals = compressed_obj["num_residuals_per_chunk"][i]
        decoded_residuals.extend(golomb_rice.rice_decode(chunk_bits, m, num_residuals))

    residuals_arr = np.array(decoded_residuals).reshape(
        compressed_obj["original_shape"]
    )
    return predictor.reconstruct_from_residuals(residuals_arr)


def decompress_parallel(compressed_obj):
    """Decompresses chunks in parallel using a thread pool."""
    m = compressed_obj["m"]
    chunks = compressed_obj["chunks"]
    num_residuals_list = compressed_obj["num_residuals_per_chunk"]

    # Decode function for a single chunk
    def decode_worker(i):
        return golomb_rice.rice_decode(chunks[i], m, num_residuals_list[i])

    # Use a thread pool to run workers in parallel
    with ThreadPoolExecutor() as executor:
        decoded_chunks = list(executor.map(decode_worker, range(len(chunks))))

    # Combine the results
    decoded_residuals = [item for sublist in decoded_chunks for item in sublist]
    residuals_arr = np.array(decoded_residuals).reshape(
        compressed_obj["original_shape"]
    )
    return predictor.reconstruct_from_residuals(residuals_arr)


if __name__ == "__main__":
    # --- Configuration ---
    # change with custom test images
    IMAGE_PATH = "assets/scdi.jpeg"
    RICE_PARAMETER_M = 16
    NUM_CHUNKS = 8

    # --- Create a sample image if it doesn't exist ---
    if not os.path.exists(IMAGE_PATH):
        print(f"'{IMAGE_PATH}' not found. Creating a sample 1024x1024 grayscale image.")
        sample_img_data = np.zeros((1024, 1024), dtype=np.uint8)
        # Create some gradients and shapes to make it more interesting
        x = np.linspace(0, 255, 1024)
        y = np.linspace(0, 255, 1024)
        xv, yv = np.meshgrid(x, y)
        sample_img_data = ((xv + yv) / 2).astype(np.uint8)
        sample_img_data[256:768, 256:768] = np.random.randint(
            0, 256, (512, 512), dtype=np.uint8
        )
        save_image(sample_img_data, IMAGE_PATH)

    # --- Compression (and saving of residuals image) ---
    compressed_obj, original_image_data = compress_image(
        IMAGE_PATH, RICE_PARAMETER_M, NUM_CHUNKS
    )

    if compressed_obj:
        # --- Evaluation ---
        print("\n--- Starting Decompression & Evaluation ---")

        # 1. Serial Decompression
        start_time_serial = time.perf_counter()
        reconstructed_serial = decompress_serial(compressed_obj)
        end_time_serial = time.perf_counter()
        serial_time = end_time_serial - start_time_serial
        print(f"\nSerial Decompression Time: {serial_time:.6f} seconds")

        # 2. Parallel Decompression
        start_time_parallel = time.perf_counter()
        reconstructed_parallel = decompress_parallel(compressed_obj)
        end_time_parallel = time.perf_counter()
        parallel_time = end_time_parallel - start_time_parallel
        print(f"Parallel Decompression Time: {parallel_time:.6f} seconds")

        # # Save the final reconstructed image ---
        # base_name = os.path.splitext(os.path.basename(IMAGE_PATH))[0]
        # reconstructed_path = f"results/{base_name}_reconstructed.png"
        # save_image(reconstructed_parallel, reconstructed_path)

        # 3. Calculate Speedup & Compression Ratio
        speedup = serial_time / parallel_time if parallel_time > 0 else float("inf")
        print(f"\nSpeedup Factor: {speedup:.2f}x")

        original_size_bits = original_image_data.size * 8
        compressed_size_bits = sum(len(c) for c in compressed_obj["chunks"])
        compression_ratio = original_size_bits / compressed_size_bits
        bits_per_pixel = compressed_size_bits / original_image_data.size

        print(f"Original Size: {original_size_bits / 8192:.2f} KB")
        print(f"Compressed Size: {compressed_size_bits / 8192:.2f} KB")
        print(f"Compression Ratio: {compression_ratio:.2f}:1")
        print(f"Bits Per Pixel (bpp): {bits_per_pixel:.3f}")

        # 4. Verify Correctness
        if np.array_equal(original_image_data, reconstructed_parallel):
            print("\nVerification successful: Decompressed image matches the original.")
        else:
            print(
                "\nVerification failed: Decompressed image does not match the original."
            )

        # --- Function to save logs to a JSON file ---
        def save_logs_to_json(file_path, logs):
            """
            Appends the log messages from the current run to a JSON file.
            Each run is stored as a separate object in a list.
            """
            try:
                # Load existing data from the file
                with open(file_path, "r") as f:
                    all_runs_data = json.load(f)
                # Ensure it's a list
                if not isinstance(all_runs_data, list):
                    all_runs_data = []
            except (FileNotFoundError, json.JSONDecodeError):
                # If file doesn't exist or is invalid, start with an empty list
                all_runs_data = []

            # Create a new entry for the current execution
            current_run_entry = {
                "timestamp": datetime.now().isoformat(),
                "output": logs,
            }
            all_runs_data.append(current_run_entry)

            # Write the updated list back to the file
            with open(file_path, "w") as f:
                json.dump(all_runs_data, f, indent=4)

            # Revert print back to original to not log the final message
            builtins.print = original_print
            print(f"\nLog messages from this run have been appended to '{file_path}'")

        # --- Save all captured print messages to output.json ---
        save_logs_to_json("output.json", log_entries)
