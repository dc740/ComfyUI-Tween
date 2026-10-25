"""Lazy loader for the gfx906 tiled attention kernel used by the SPEED backend.

The kernel object ships prebuilt at ``hip_attn/attn_kernel.gfx906.o``. If it is
missing (e.g. source checkout without the build step), it is rebuilt once with
hipcc and a ``.prebuilt`` marker is written; a failed build writes
``.build_failed`` so later imports do not retry. The torch extension itself is
compiled and cached by ``torch.utils.cpp_extension.load``.
"""

import logging
import shutil
import subprocess
from pathlib import Path

import torch


logger = logging.getLogger("Tween")

_HIP_DIR = Path(__file__).resolve().parent / "hip_attn"
_KERNEL_OBJ = _HIP_DIR / "attn_kernel.gfx906.o"
_PREBUILT = _HIP_DIR / ".prebuilt"
_BUILD_FAILED = _HIP_DIR / ".build_failed"

_module = None
_attempted = False


def is_gfx906() -> bool:
    """True when the current CUDA device is a gfx906 (MI50/MI60) GPU."""
    if not torch.cuda.is_available():
        return False
    try:
        arch = torch.cuda.get_device_properties(
            torch.cuda.current_device()).gcnArchName
    except Exception:  # noqa: BLE001 - non-ROCm builds have no gcnArchName
        return False
    return arch.startswith("gfx906")


def _build_kernel_obj() -> bool:
    if _KERNEL_OBJ.is_file():
        return True
    if _BUILD_FAILED.is_file():
        return False
    hipcc = shutil.which("hipcc") or "/opt/rocm/bin/hipcc"
    result = subprocess.run(
        [hipcc, "--offload-arch=gfx906", "-O3", "-fPIC",
         "-c", "attn_kernel.hip", "-o", _KERNEL_OBJ.name],
        cwd=_HIP_DIR, capture_output=True, text=True)
    if result.returncode == 0 and _KERNEL_OBJ.is_file():
        _PREBUILT.write_text(result.stderr)
        return True
    _BUILD_FAILED.write_text(result.stderr[-4000:])
    logger.warning("gfx906 attention kernel build failed: %s",
                   result.stderr[-500:])
    return False


def load_attention():
    """Return the compiled extension module (has ``attn(q, k, v, scale)``) or None.

    q/k/v are [B, H, S, D] fp16 CUDA tensors; Sq may differ from Skv.
    Returns None off gfx906, on build failure, or after the first attempt.
    """
    global _module, _attempted
    if _attempted:
        return _module
    _attempted = True
    try:
        if not is_gfx906() or not _build_kernel_obj():
            return None
        from torch.utils import cpp_extension
        _module = cpp_extension.load(
            name="tween_gfx906_attn",
            sources=[str(_HIP_DIR / "attn.cpp")],
            extra_ldflags=[str(_KERNEL_OBJ)],
            verbose=False,
        )
        logger.info("gfx906 attention kernel loaded")
    except Exception as exc:  # noqa: BLE001 - fall back to chunked attention
        logger.warning("gfx906 attention kernel unavailable: %s", exc)
        _module = None
    return _module
