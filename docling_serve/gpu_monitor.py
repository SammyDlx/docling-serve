"""GPU memory monitoring for debugging OOM in docling-serve.

Monkey-patches key docling model classes to log GPU memory usage
at model loading and inference time.
"""

import functools
import logging
import os

_log = logging.getLogger("docling_serve.gpu_monitor")

_ENABLED = os.environ.get("GPU_MEMORY_MONITOR", "").lower() in ("1", "true", "yes")


def _fmt_mb(bytes_val: int) -> str:
    return f"{bytes_val / 1024 / 1024:.1f} MB"


def _log_gpu_memory(label: str) -> None:
    try:
        import torch.cuda

        if not torch.cuda.is_available():
            return
        allocated = torch.cuda.memory_allocated()
        reserved = torch.cuda.memory_reserved()
        max_allocated = torch.cuda.max_memory_allocated()
        total = torch.cuda.get_device_properties(0).total_memory
        free, total_nv = torch.cuda.mem_get_info()
        _log.warning(
            f"[GPU] {label} | "
            f"allocated={_fmt_mb(allocated)} "
            f"reserved={_fmt_mb(reserved)} "
            f"max_allocated={_fmt_mb(max_allocated)} "
            f"total_vram={_fmt_mb(total)} "
            f"free_vram={_fmt_mb(free)} "
            f"used_outside_pytorch={_fmt_mb(total_nv - free - reserved)}"
        )
    except Exception as e:
        _log.error(f"[GPU] Failed to read memory stats: {e}")


def _wrap_method(cls, method_name, label):
    """Wrap an existing method to log GPU memory before and after."""
    original = getattr(cls, method_name)

    @functools.wraps(original)
    def wrapper(*args, **kwargs):
        _log_gpu_memory(f"{label} BEFORE")
        try:
            result = original(*args, **kwargs)
            _log_gpu_memory(f"{label} AFTER")
            return result
        except Exception as e:
            _log_gpu_memory(f"{label} ERROR ({type(e).__name__})")
            if "OutOfMemory" in type(e).__name__ or "CUDA" in str(e):
                try:
                    import torch.cuda
                    _log.warning(f"[GPU] Memory summary at OOM:\n{torch.cuda.memory_summary()}")
                except Exception:
                    pass
            raise

    setattr(cls, method_name, wrapper)
    _log.info(f"Patched {cls.__name__}.{method_name} with GPU monitor")


def install_gpu_monitor():
    """Monkey-patch docling model classes with GPU memory logging."""
    if not _ENABLED:
        _log.info("GPU memory monitor disabled (set GPU_MEMORY_MONITOR=1 to enable)")
        return

    _log.warning("GPU memory monitor ENABLED — patching docling models")
    _log_gpu_memory("STARTUP baseline")

    patched = 0

    # 1. Chart extraction model
    try:
        from docling.models.stages.chart_extraction.granite_vision import (
            ChartExtractionModelGraniteVision,
        )

        _wrap_method(
            ChartExtractionModelGraniteVision,
            "__init__",
            "ChartExtraction.__init__ (model load)",
        )
        _wrap_method(
            ChartExtractionModelGraniteVision,
            "__call__",
            "ChartExtraction.__call__ (inference)",
        )
        patched += 1
    except ImportError as e:
        _log.warning(f"Could not patch ChartExtractionModelGraniteVision: {e}")

    # 2. Picture classifier
    try:
        from docling.models.stages.picture_classifier.document_picture_classifier import (
            DocumentPictureClassifier,
        )

        _wrap_method(
            DocumentPictureClassifier,
            "__init__",
            "PictureClassifier.__init__ (model load)",
        )
        _wrap_method(
            DocumentPictureClassifier,
            "__call__",
            "PictureClassifier.__call__ (inference)",
        )
        patched += 1
    except ImportError as e:
        _log.warning(f"Could not patch DocumentPictureClassifier: {e}")

    # 3. Pipeline enrichment
    try:
        from docling.pipeline.base_pipeline import ConvertPipeline

        _wrap_method(
            ConvertPipeline,
            "_enrich_document",
            "Pipeline._enrich_document",
        )
        patched += 1
    except ImportError as e:
        _log.warning(f"Could not patch ConvertPipeline: {e}")

    # 4. Converter manager — log when converters are created/cached
    try:
        from docling_jobkit.convert.manager import DoclingConverterManager

        _wrap_method(
            DoclingConverterManager,
            "convert_documents",
            "ConverterManager.convert_documents",
        )
        patched += 1
    except ImportError as e:
        _log.warning(f"Could not patch DoclingConverterManager: {e}")

    _log.warning(f"GPU monitor installed — {patched} components patched")
