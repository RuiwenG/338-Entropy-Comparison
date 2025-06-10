import time
import numpy as np
from PIL import Image
from concurrent.futures import ThreadPoolExecutor

# Import your custom modules
import golomb_rice
import predictor


def compress_image(image_path, m, num_chunks):
    """
    Loads, processes, and compresses an image into independent chunks.
    """
    print(f"Compressing '{image_path}'...")
    # Load image and convert to grayscale numpy array
    try:
        img = Image.open(image_path).convert("L")
        image_data = np.array(img)
    except FileNotFoundError:
        print(f"Error: Image file not found at '{image_path}'")
        return None

    # Get prediction errors
    residuals = predictor.get_residuals(image_data)

    # Split residuals into chunks for parallel processing
    # Here, each chunk is a set of rows
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

    decoded_chunks = [None] * len(chunks)

    # Decode function for a single chunk
    def decode_worker(i):
        return golomb_rice.rice_decode(chunks[i], m, num_residuals_list[i])

    # Use a thread pool to run workers in parallel
    with ThreadPoolExecutor() as executor:
        # map() runs the worker on each index and returns results in order
        decoded_chunks = list(executor.map(decode_worker, range(len(chunks))))

    # Combine the results
    decoded_residuals = [item for sublist in decoded_chunks for item in sublist]
    residuals_arr = np.array(decoded_residuals).reshape(
        compressed_obj["original_shape"]
    )
    return predictor.reconstruct_from_residuals(residuals_arr)


if __name__ == "__main__":
    # --- Configuration ---
    IMAGE_PATH = "assets/coconut.jpeg"
    RICE_PARAMETER_M = 16  # Power of 2 (e.g., 4, 8, 16, 32). Tune this for your image.
    NUM_CHUNKS = 8  # Number of chunks to split image into (for parallelization)

    # --- Create a sample image if it doesn't exist ---
    try:
        Image.open(IMAGE_PATH)
    except FileNotFoundError:
        print(f"'{IMAGE_PATH}' not found. Creating a sample 1024x1024 grayscale image.")
        sample_img_data = np.zeros((1024, 1024), dtype=np.uint8)
        sample_img_data[256:768, 256:768] = np.random.randint(
            0, 256, (512, 512), dtype=np.uint8
        )
        Image.fromarray(sample_img_data).save(IMAGE_PATH)

    # --- Compression ---
    compressed_obj, original_image_data = compress_image(
        IMAGE_PATH, RICE_PARAMETER_M, NUM_CHUNKS
    )

    if compressed_obj:
        # --- Evaluation ---
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

        # 3. Calculate Speedup & Compression Ratio
        speedup = serial_time / parallel_time if parallel_time > 0 else float("inf")
        print(f"\nSpeedup Factor: {speedup:.2f}x")

        original_size_bits = original_image_data.size * 8  # 8 bits per pixel
        compressed_size_bits = sum(len(c) for c in compressed_obj["chunks"])
        compression_ratio = original_size_bits / compressed_size_bits
        bits_per_pixel = compressed_size_bits / original_image_data.size

        print(f"Original Size: {original_size_bits / 8192:.2f} KB")
        print(f"Compressed Size: {compressed_size_bits / 8192:.2f} KB")
        print(f"Compression Ratio: {compression_ratio:.2f}:1")
        print(f"Bits Per Pixel (bpp): {bits_per_pixel:.3f}")

        # 4. Verify Correctness
        if np.array_equal(original_image_data, reconstructed_serial) and np.array_equal(
            original_image_data, reconstructed_parallel
        ):
            print("\n Verification successful: Decompressed images match the original.")
        else:
            print(
                "\n Verification failed: Decompressed images do not match the original."
            )
