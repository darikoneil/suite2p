"""Pure helper utilities for suite2p GUI trace plotting."""

import numpy as np


def gaussian_smooth_trace(trace: np.ndarray, sigma: float) -> np.ndarray:
    """Smooth a 1D trace with a Gaussian kernel when requested.

    Args:
        trace: Trace samples ordered in time.
        sigma: Gaussian width in frames.

    Returns:
        A smoothed copy of the input trace. If ``sigma <= 0``, the original
        samples are returned unchanged.
    """
    trace = np.asarray(trace, dtype=float)
    if sigma <= 0:
        return trace
    radius = max(1, int(np.ceil(4 * sigma)))
    offsets = np.arange(-radius, radius + 1, dtype=float)
    kernel = np.exp(-0.5 * (offsets / sigma) ** 2)
    kernel /= kernel.sum()
    padded = np.pad(trace, pad_width=radius, mode="reflect")
    return np.convolve(padded, kernel, mode="valid")


def build_trace_time_axis(
    n_frames: int,
    use_time_axis: bool,
    frame_rate: float,
) -> tuple[np.ndarray, str]:
    """Build the x-axis used for ROI trace plotting.

    Args:
        n_frames: Number of samples in the trace.
        use_time_axis: Whether to display the x-axis in seconds.
        frame_rate: Frame rate in Hz.

    Returns:
        A tuple containing the x-axis values and the corresponding axis label.

    Raises:
        ValueError: If ``use_time_axis`` is true and ``frame_rate`` is not
            positive.
    """
    if use_time_axis:
        if frame_rate <= 0:
            raise ValueError("Frame rate must be positive when plotting time.")
        return np.arange(n_frames, dtype=float) / frame_rate, "Time (s)"
    return np.arange(n_frames, dtype=float), "Frame"


def convert_behavior_time_axis(
    behavior_time: np.ndarray,
    trace_axis: np.ndarray,
    use_time_axis: bool,
    frame_rate: float,
) -> np.ndarray:
    """Map behavior timestamps onto the active trace axis.

    Args:
        behavior_time: Behavior timestamps or frame indices.
        trace_axis: Active x-axis used for trace plotting.
        use_time_axis: Whether the trace axis is displayed in seconds.
        frame_rate: Frame rate in Hz.

    Returns:
        Behavior x-axis values aligned with the active trace axis.
    """
    behavior_time = np.asarray(behavior_time, dtype=float)
    if not use_time_axis or behavior_time.size == 0:
        return behavior_time

    zero_based = np.arange(behavior_time.size, dtype=float)
    one_based = np.arange(1, behavior_time.size + 1, dtype=float)
    if np.allclose(behavior_time, zero_based):
        return trace_axis
    if np.allclose(behavior_time, one_based):
        return behavior_time / frame_rate
    return behavior_time
