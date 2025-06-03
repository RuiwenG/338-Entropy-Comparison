import math
import time
from PIL import Image
import numpy as np


class GolombCode:
    """
    A class to perform Golomb-Rice encoding and decoding.
    This is used for the binarization of integer values.
    """

    def __init__(self, m):
        """
        Initializes the Golomb-Rice coder with a parameter m.
        If m is a power of 2, it's a more efficient Rice code.
        """
        if m <= 0:
            raise ValueError("M must be a positive integer.")
        self.m = m
        self.is_power_of_two = (m > 0) and ((m & (m - 1)) == 0)
        if self.is_power_of_two:
            self.k = int(math.log2(m))

    def encode(self, n):
        """Encodes a non-negative integer n."""
        if n < 0:
            raise ValueError("Golomb coding is for non-negative integers.")
        q = n // self.m
        r = n % self.m

        # Unary code for the quotient
        quotient_code = "1" * q + "0"

        # Binary code for the remainder
        if self.is_power_of_two:
            remainder_code = format(r, f"0{self.k}b")
        else:
            b = int(math.ceil(math.log2(self.m)))
            if r < (2**b - self.m):
                remainder_code = format(r, f"0{b - 1}b")
            else:
                remainder_code = format(r + (2**b - self.m), f"0{b}b")
        return quotient_code + remainder_code


class CABACEngine:
    """
    A simplified implementation of a Context-Adaptive Binary Arithmetic Coding (CABAC) engine.
    """

    def __init__(self):
        self.low = 0
        self.high = (1 << 16) - 1  # Use 16-bit precision for the interval
        self.bitstream = ""
        self.contexts = {}  # {context_id: {"p_lps": prob, "mps": 0 or 1}}

    def _get_context(self, context_id):
        """Initializes a context if it doesn't exist."""
        if context_id not in self.contexts:
            # Initialize with equal probability and Most Probable Symbol (MPS) as 0
            self.contexts[context_id] = {"p_lps": 0.5, "mps": 0}
        return self.contexts[context_id]

    def _update_context(self, context_id, bit):
        """Updates the probability model for a given context."""
        context = self._get_context(context_id)
        p_lps = context["p_lps"]
        mps = context["mps"]

        if bit == mps:
            # The MPS occurred, decrease the probability of the LPS
            context["p_lps"] *= 0.95
        else:
            # The LPS occurred, increase its probability
            context["p_lps"] *= 1.5
            # If the LPS is now more probable, switch the MPS
            if context["p_lps"] > 0.5:
                context["mps"] = 1 - mps
                context["p_lps"] = 1.0 - context["p_lps"]

    def encode_bin(self, bit, context_id):
        """Encodes a single binary value (bin)."""
        context = self._get_context(context_id)
        p_lps = context["p_lps"]
        mps = context["mps"]

        range_width = self.high - self.low + 1
        lps_range = int(range_width * p_lps)

        if lps_range == 0:
            lps_range = 1

        if bit == mps:
            self.high = self.low + (range_width - lps_range) - 1
        else:
            self.low = self.low + (range_width - lps_range)
            self.high = self.low + lps_range - 1

        # Renormalization
        while True:
            if self.high < (1 << 15):
                self.bitstream += "0"
                self.low <<= 1
                self.high = (self.high << 1) | 1
            elif self.low >= (1 << 15):
                self.bitstream += "1"
                self.low = (self.low - (1 << 15)) << 1
                self.high = ((self.high - (1 << 15)) << 1) | 1
            else:
                break

        self._update_context(context_id, bit)

    def finish_encoding(self):
        """Finalizes the bitstream."""
        # Output the remaining interval
        if self.low < (1 << 14):
            self.bitstream += "01"
        else:
            self.bitstream += "10"
        return self.bitstream


class EntropyEncoder:
    """
    An entropy encoder that uses CABAC with Golomb binarization.
    """

    def __init__(self, golomb_m):
        self.cabac = CABACEngine()
        self.golomb = GolombCode(m=golomb_m)

    def encode_integer(self, value, context_prefix="int_"):
        """
        Encodes an integer by first binarizing it with Golomb coding
        and then encoding the bins with CABAC.
        """
        # Map signed to unsigned
        if value >= 0:
            unsigned_value = 2 * value
        else:
            unsigned_value = -2 * value - 1

        golomb_code = self.golomb.encode(unsigned_value)

        for i, bit_char in enumerate(golomb_code):
            bit = int(bit_char)
            # Use a different context for each position in the Golomb code
            context_id = f"{context_prefix}{i}"
            self.cabac.encode_bin(bit, context_id)

    def get_encoded_bitstream(self):
        """Returns the final encoded bitstream."""
        return self.cabac.finish_encoding()


def get_image_residuals(image_path):
    """
    Loads an image, converts it to grayscale, and calculates prediction residuals.
    The prediction for a pixel is the value of the pixel to its left.
    """
    try:
        with Image.open(image_path) as img:
            # Convert to grayscale (Luminance)
            img_gray = img.convert("L")
            data = np.array(img_gray, dtype=int)
    except FileNotFoundError:
        print(f"Error: Image file not found at '{image_path}'")
        return None

    # Calculate residuals (prediction error)
    # The prediction for pixel p(x, y) is p(x-1, y)
    # The residual is p(x, y) - p(x-1, y)
    # We use numpy.roll for an efficient vectorized implementation
    predicted = np.roll(data, 1, axis=1)
    # The first column has no left neighbor, so its prediction is 0.
    predicted[:, 0] = 0

    residuals = data - predicted

    # Return the flattened stream of residuals
    return residuals.flatten()


if __name__ == "__main__":
    # --- IMPORTANT ---
    # Replace this with the path to your own image file (e.g., a .png or .jpg)
    IMAGE_PATH = "test_image.png"

    # --- Pre-computation ---
    print(f"Loading image and calculating residuals from '{IMAGE_PATH}'...")
    residuals = get_image_residuals(IMAGE_PATH)

    if residuals is not None:
        # We need to choose a parameter 'M' for Golomb coding.
        # A good heuristic is to use an M near the average magnitude of the residuals.
        avg_abs_residual = np.mean(np.abs(residuals))
        # Choose a power of 2 near this average for efficiency (Rice code)
        m_parameter = 2 ** round(math.log2(max(1, avg_abs_residual)))

        print(f"Calculated optimal M for Golomb/Rice coding: {m_parameter}")
        print("-" * 50)

        # --- Experiment 1: Standalone Golomb Coding ---
        print("Running Experiment 1: Standalone Golomb Coding...")
        start_time_golomb = time.time()

        golomb_coder = GolombCode(m=m_parameter)
        total_bits_golomb = 0
        for res in residuals:
            # Map signed residual to unsigned integer
            unsigned_res = 2 * res if res >= 0 else -2 * res - 1
            total_bits_golomb += len(golomb_coder.encode(unsigned_res))

        end_time_golomb = time.time()
        time_golomb = end_time_golomb - start_time_golomb
        size_golomb_bytes = math.ceil(total_bits_golomb / 8)
        print("...Done.")
        print("-" * 50)

        # --- Experiment 2: CABAC with Golomb Binarization ---
        print("Running Experiment 2: CABAC with Golomb Binarization...")
        start_time_cabac = time.time()

        cabac_encoder = EntropyEncoder(golomb_m=m_parameter)
        for res in residuals:
            # The encode_integer method already handles the signed mapping
            cabac_encoder.encode_integer(res)

        total_bits_cabac = cabac_encoder.get_encoded_bitstream_length()

        end_time_cabac = time.time()
        time_cabac = end_time_cabac - start_time_cabac
        size_cabac_bytes = math.ceil(total_bits_cabac / 8)
        print("...Done.")
        print("-" * 50)

        # --- 4. Final Comparison ---
        original_size_bytes = len(residuals)  # Grayscale, so 1 byte per pixel

        print("\n--- COMPARISON RESULTS ---")
        print(f"Original Image Size (Grayscale): {original_size_bytes:,} bytes")
        print("\n")
        print(
            f"{'Method':<25} | {'Execution Time (s)':<20} | {'Compressed Size (bytes)':<25} | {'Compression Ratio':<20}"
        )
        print("-" * 100)
        print(
            f"{'Standalone Golomb':<25} | {time_golomb:<20.4f} | {size_golomb_bytes:<25,} | {original_size_bytes / size_golomb_bytes:<20.2f}x"
        )
        print(
            f"{'CABAC (+Golomb Binarize)':<25} | {time_cabac:<20.4f} | {size_cabac_bytes:<25,} | {original_size_bytes / size_cabac_bytes:<20.2f}x"
        )
