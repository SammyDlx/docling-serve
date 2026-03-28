"""GPU memory monitoring for debugging OOM in docling-serve.

Monkey-patches key docling model classes to log GPU memory usage
at model loading and inference time. Also patches pipeline cache
to log cache hits/misses and option hash changes.
"""

import functools
import hashlib
import logging
import os
import traceback

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
        if "__init__" in method_name:
            stack = "".join(traceback.format_stack(limit=10))
            _log.warning(f"[GPU] {label} call stack:\n{stack}")
        try:
            result = original(*args, **kwargs)
            _log_gpu_memory(f"{label} AFTER")
            return result
        except Exception as e:
            _log_gpu_memory(f"{label} ERROR ({type(e).__name__})")
            _log.warning(f"[GPU] Exception detail: {e}")
            if "OutOfMemory" in type(e).__name__ or "CUDA" in str(e):
                try:
                    import torch.cuda
                    _log.warning(f"[GPU] Memory summary at OOM:\n{torch.cuda.memory_summary()}")
                except Exception:
                    pass
            raise

    setattr(cls, method_name, wrapper)
    _log.info(f"Patched {cls.__name__}.{method_name} with GPU monitor")


def _options_hash(pipeline_options) -> str:
    """Compute the same hash that DocumentConverter._get_pipeline uses."""
    try:
        options_str = str(pipeline_options.model_dump())
        return hashlib.md5(options_str.encode("utf-8"), usedforsecurity=False).hexdigest()[:12]
    except Exception:
        return "?"


def _patch_get_pipeline():
    """Patch DocumentConverter._get_pipeline to log cache hits/misses and hash values."""
    try:
        from docling.document_converter import DocumentConverter

        original = DocumentConverter._get_pipeline

        @functools.wraps(original)
        def patched_get_pipeline(self, doc_format):
            fopt = self.format_to_options.get(doc_format)
            if fopt and fopt.pipeline_options:
                h = _options_hash(fopt.pipeline_options)
                opts = fopt.pipeline_options
                key_fields = {}
                for field in ["do_chart_extraction", "do_picture_classification",
                              "do_table_structure", "do_ocr"]:
                    if hasattr(opts, field):
                        key_fields[field] = getattr(opts, field)
                cache_key = (fopt.pipeline_cls.__name__ if fopt.pipeline_cls else "?", h)
                hit = cache_key in self.initialized_pipelines
                _log.warning(
                    f"[GPU] _get_pipeline(format={doc_format}) | "
                    f"hash={h} cache_hit={hit} "
                    f"cache_keys={[(c[0], c[1]) for c in self.initialized_pipelines.keys()]} "
                    f"opts={key_fields}"
                )
            _log_gpu_memory(f"_get_pipeline({doc_format})")
            result = original(self, doc_format)
            if fopt and fopt.pipeline_options:
                h_after = _options_hash(fopt.pipeline_options)
                if h_after != h:
                    _log.warning(
                        f"[GPU] _get_pipeline HASH CHANGED after pipeline creation! "
                        f"before={h} after={h_after} — this causes cache misses"
                    )
                    opts_after = fopt.pipeline_options
                    after_fields = {}
                    for field in ["do_chart_extraction", "do_picture_classification",
                                  "do_table_structure", "do_ocr"]:
                        if hasattr(opts_after, field):
                            after_fields[field] = getattr(opts_after, field)
                    _log.warning(f"[GPU] Options before: {key_fields} -> after: {after_fields}")
            return result

        DocumentConverter._get_pipeline = patched_get_pipeline
        _log.info("Patched DocumentConverter._get_pipeline with cache monitor")
    except Exception as e:
        _log.warning(f"Could not patch DocumentConverter._get_pipeline: {e}")


def _patch_initialize_pipeline():
    """Patch DocumentConverter.initialize_pipeline to log hash before/after."""
    try:
        from docling.document_converter import DocumentConverter

        original = DocumentConverter.initialize_pipeline

        @functools.wraps(original)
        def patched_init_pipeline(self, format):
            fopt = self.format_to_options.get(format)
            h_before = _options_hash(fopt.pipeline_options) if fopt and fopt.pipeline_options else "?"
            _log.warning(f"[GPU] initialize_pipeline(format={format}) hash_before={h_before}")
            _log_gpu_memory(f"initialize_pipeline({format}) BEFORE")
            result = original(self, format)
            h_after = _options_hash(fopt.pipeline_options) if fopt and fopt.pipeline_options else "?"
            _log.warning(f"[GPU] initialize_pipeline done hash_after={h_after} changed={h_before != h_after}")
            _log_gpu_memory(f"initialize_pipeline({format}) AFTER")
            return result

        DocumentConverter.initialize_pipeline = patched_init_pipeline
        _log.info("Patched DocumentConverter.initialize_pipeline with hash monitor")
    except Exception as e:
        _log.warning(f"Could not patch DocumentConverter.initialize_pipeline: {e}")


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

    # 5. Pipeline cache monitoring — log hash changes and cache hits/misses
    _patch_get_pipeline()
    _patch_initialize_pipeline()

    _log.warning(f"GPU monitor installed — {patched} components patched")
