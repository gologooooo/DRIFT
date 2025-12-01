import numpy as np
import torch


def masked_mae_fft(prediction: torch.Tensor, target: torch.Tensor, null_val: float = np.nan) -> torch.Tensor:
    """
    Calculate the Masked Mean Absolute Error (MAE) between the predicted and target values,
    while ignoring the entries in the target tensor that match the specified null value.

    This function is particularly useful for scenarios where the dataset contains missing or irrelevant
    values (denoted by `null_val`) that should not contribute to the loss calculation. It effectively
    masks these values to ensure they do not skew the error metrics.

    Args:
        prediction (torch.Tensor): The predicted values as a tensor.
        target (torch.Tensor): The ground truth values as a tensor with the same shape as `prediction`.
        null_val (float, optional): The value considered as null or missing in the `target` tensor.
            Default is `np.nan`. The function will mask all `NaN` values in the target.

    Returns:
        torch.Tensor: A scalar tensor representing the masked mean absolute error.

    """

    if np.isnan(null_val):
        mask = ~torch.isnan(target)
    else:
        eps = 5e-5
        mask = ~torch.isclose(target, torch.tensor(null_val).expand_as(target).to(target.device), atol=eps, rtol=0.0)

    mask = mask.float()
    mask /= torch.mean(mask)  # Normalize mask to avoid bias in the loss due to the number of valid entries
    mask = torch.nan_to_num(mask)  # Replace any NaNs in the mask with zero

    temporal_loss = torch.abs(prediction - target)
    temporal_loss = temporal_loss * mask  # Apply the mask to the loss
    temporal_loss = torch.nan_to_num(temporal_loss)  # Replace any NaNs in the loss with zero

    temporal_loss = torch.mean(temporal_loss)

    prediction_clean = torch.nan_to_num(prediction)
    target_clean = torch.nan_to_num(target)
    fft_loss = torch.mean(torch.abs(torch.fft.rfft(prediction_clean, dim=1) - torch.fft.rfft(target_clean, dim=1)))

    return temporal_loss + 0.1 * fft_loss

