import math
from bitarray import bitarray


# Golomb-Rice works on non-negative integers.
# We map signed prediction errors to unsigned integers.
# zig-zag mapping: 0 -> 0, -1 -> 1, 1 -> 2, -2 -> 3, 2 -> 4, ...
def map_to_unsigned(n):
    """Maps a signed integer to an unsigned integer."""
    if n >= 0:
        return 2 * n
    else:
        return -2 * n - 1


def map_to_signed(n):
    """Maps an unsigned integer back to a signed integer."""
    if n % 2 == 0:
        return n // 2
    else:
        return -(n + 1) // 2


def rice_encode(residuals, m):
    """
    Encodes a list of residuals using Golomb-Rice coding.

    Args:
        residuals (list[int]): The prediction errors to encode.
        m (int): The Rice coding parameter (must be a power of 2).

    Returns:
        bitarray: The encoded bitstream.
    """
    if not (m > 0 and (m & (m - 1)) == 0):
        raise ValueError("m must be a power of 2.")

    bits = bitarray()
    k = int(math.log2(m))

    for r in residuals:
        # 1. Map signed residual to a non-negative integer
        n = map_to_unsigned(r)

        # 2. Calculate quotient and remainder
        quotient = n // m
        remainder = n % m

        # 3. Encode quotient in unary (q '1's followed by a '0')
        bits.extend("1" * quotient + "0")

        # 4. Encode remainder in binary with k bits
        remainder_bits = f"{remainder:0{k}b}"
        bits.extend(remainder_bits)

    return bits


def rice_decode(bits, m, num_residuals):
    """
    Decodes a Golomb-Rice bitstream back into residuals.

    Args:
        bits (bitarray): The bitstream to decode.
        m (int): The Rice coding parameter used for encoding.
        num_residuals (int): The expected number of residuals to decode.

    Returns:
        list[int]: The decoded signed residuals.
    """
    if not (m > 0 and (m & (m - 1)) == 0):
        raise ValueError("m must be a power of 2.")

    residuals = []
    k = int(math.log2(m))
    idx = 0

    while len(residuals) < num_residuals and idx < len(bits):
        # 1. Decode quotient from unary
        quotient = 0
        while bits[idx] == 1:
            quotient += 1
            idx += 1
        idx += 1  # Skip the '0' separator

        # 2. Decode remainder from binary
        if idx + k > len(bits):
            break  # Avoid index error on malformed stream
        remainder_bits = bits[idx : idx + k]
        remainder = int(remainder_bits.to01(), 2)
        idx += k

        # 3. Reconstruct the unsigned number and map back to signed
        n = quotient * m + remainder
        r = map_to_signed(n)
        residuals.append(r)

    return residuals
