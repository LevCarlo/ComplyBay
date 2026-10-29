"""Teleseismic autocorrelation processing."""

from complybay.acf.errors import (
    StackErrors,
    processing_error_correlation,
    shrink_correlation,
    stack_errors,
)
from complybay.acf.processing import (
    PRESETS,
    ACFProcessing,
    Bandpass,
    Filter,
    GaussianFilter,
    Highpass,
    Lowpass,
    Whitening,
    acf_from_spectrum,
    acf_from_trace,
    preset,
    stack_acfs,
)

__all__ = [
    "PRESETS",
    "ACFProcessing",
    "Bandpass",
    "Filter",
    "GaussianFilter",
    "Highpass",
    "Lowpass",
    "StackErrors",
    "Whitening",
    "acf_from_spectrum",
    "acf_from_trace",
    "preset",
    "processing_error_correlation",
    "shrink_correlation",
    "stack_acfs",
    "stack_errors",
]
