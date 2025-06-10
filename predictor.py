import numpy as np


def get_residuals(image_data):
    """
    Calculates prediction errors (residuals) for an image.
    Uses a simple predictor: predict current pixel = left pixel.
    """
    residuals = np.zeros_like(image_data, dtype=np.int16)
    # Predict first column as 0
    residuals[:, 0] = image_data[:, 0]
    # For other columns, residual = actual - predicted (left pixel)
    residuals[:, 1:] = image_data[:, 1:].astype(np.int16) - image_data[:, :-1].astype(
        np.int16
    )
    return residuals


def reconstruct_from_residuals(residuals):
    """
    Reconstructs the image from its residuals.
    This is the inverse of the prediction step.
    """
    reconstructed = np.zeros_like(residuals, dtype=np.uint8)
    # First column is the same as the residual
    reconstructed[:, 0] = residuals[:, 0].astype(np.uint8)
    # For other columns, actual = residual + predicted (left pixel)
    for col in range(1, reconstructed.shape[1]):
        reconstructed[:, col] = (residuals[:, col] + reconstructed[:, col - 1]).astype(
            np.uint8
        )
    return reconstructed
