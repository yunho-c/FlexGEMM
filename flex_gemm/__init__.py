import os
USE_AUTOTUNE_CACHE = os.environ.get('FLEX_GEMM_USE_AUTOTUNE_CACHE', '1') == '1'
AUTOSAVE_AUTOTUNE_CACHE = os.environ.get('FLEX_GEMM_AUTOSAVE_AUTOTUNE_CACHE', '1') == '1'
AUTOTUNE_CACHE_PATH = os.environ.get(
    'FLEX_GEMM_AUTOTUNE_CACHE_PATH',
    os.path.expanduser('~/.flex_gemm/autotune_cache.json')
)
    

def __getattr__(name):
    # The TT backend must be importable on machines without CUDA or Triton.
    # Initialize the legacy backend (including its cache) only when requested.
    if name not in {"kernels", "ops", "utils", "tenstorrent"}:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    import importlib
    if name == "tenstorrent":
        return importlib.import_module(f"{__name__}.{name}")
    if not globals().get("_loading_legacy", False):
        globals()["_loading_legacy"] = True
        try:
            for module in ("kernels", "ops", "utils"):
                importlib.import_module(f"{__name__}.{module}")
            if USE_AUTOTUNE_CACHE:
                globals()["utils"].load_autotune_cache()
        finally:
            globals()["_loading_legacy"] = False
    return importlib.import_module(f"{__name__}.{name}")
