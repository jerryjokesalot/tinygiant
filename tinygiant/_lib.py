import ctypes
from pathlib import Path

VP = ctypes.c_void_p
CI = ctypes.c_int
CF = ctypes.c_float


def find_library():
    """Find the libtinygiant shared library."""
    pkg_dir = Path(__file__).parent
    repo_dir = pkg_dir.parent

    search = [
        pkg_dir / "libtinygiant.dylib",
        pkg_dir / "libtinygiant.so",
        repo_dir / "libtinygiant.dylib",
        repo_dir / "libtinygiant.so",
    ]
    for path in search:
        if path.exists():
            return path
    return None


def load_tinygiant_lib(lib_path=None):
    """Load the fused Q4xQ8 shared library."""
    if lib_path is None:
        lib_path = find_library()
    if lib_path is None:
        return None

    lib = ctypes.CDLL(str(lib_path))

    def reg(fn, args):
        fn.restype = None
        fn.argtypes = args

    reg(lib.tg_expert_forward, [VP, VP, VP, CF, CI, CI])
    reg(lib.tg_expert_forward_mixed,
        [VP, VP, VP, CI, VP, VP, CF, CI, CI])
    reg(lib.tg_matmul_q4k, [VP, VP, VP, CI, CI])
    reg(lib.tg_matmul_q6k, [VP, VP, VP, CI, CI])
    reg(lib.tg_rms_norm, [VP, VP, VP, CI, CF])
    reg(lib.tg_f32_matvec, [VP, VP, VP, CI, CI])
    reg(lib.tg_f16_matvec, [VP, VP, VP, CI, CI])
    reg(lib.tg_attention_decode,
        [VP, VP, VP, VP,
         VP, VP,
         VP, VP, CI, CI,
         VP, VP,
         VP, VP,
         CI, CI, CI, CI, CI])
    reg(lib.tg_attention_decode_q4,
        [VP, VP, VP, VP,
         VP, VP,
         VP, VP, CI, CI,
         VP, VP,
         VP, VP,
         CI, CI, CI, CI, CI])
    reg(lib.tg_prefetch, [VP, ctypes.c_size_t])
    reg(lib.tg_prefetch_batch, [VP, VP, CI])
    lib.tg_mlock.restype = ctypes.c_int
    lib.tg_mlock.argtypes = [VP, ctypes.c_size_t]
    reg(lib.tg_transformer_layer,
        [VP, VP, VP, VP,
         VP, VP, VP, VP,
         VP, VP, CI, CI,
         VP, VP,
         VP, VP, VP, VP,
         VP, CI,
         VP,
         CI, CI,
         CI, CI, CI, CI])


    # Metal attention (Slice A) — present on Darwin builds; stub elsewhere.
    if hasattr(lib, "tg_metal_available"):
        lib.tg_metal_available.restype = ctypes.c_int
        lib.tg_metal_available.argtypes = []
        lib.tg_metal_status.restype = ctypes.c_char_p
        lib.tg_metal_status.argtypes = []
        lib.tg_metal_create.restype = VP
        lib.tg_metal_create.argtypes = [CI, CI, CI, CI, CI, CI]
        lib.tg_metal_destroy.restype = None
        lib.tg_metal_destroy.argtypes = [VP]
        lib.tg_metal_upload_layer_f16.restype = CI
        lib.tg_metal_upload_layer_f16.argtypes = [
            VP, CI, VP, VP, VP, VP, VP, VP]
        lib.tg_metal_upload_layer_q4.restype = CI
        lib.tg_metal_upload_layer_q4.argtypes = [
            VP, CI, VP, VP, VP, VP, VP, VP]
        lib.tg_metal_set_rope.restype = CI
        lib.tg_metal_set_rope.argtypes = [VP, VP, VP, CI]
        lib.tg_metal_reset_kv.restype = None
        lib.tg_metal_reset_kv.argtypes = [VP]
        lib.tg_metal_attention_decode.restype = CI
        lib.tg_metal_attention_decode.argtypes = [VP, CI, VP, VP, CI]
        lib.tg_metal_timing.restype = None
        lib.tg_metal_timing.argtypes = [VP, VP, VP]
        lib.tg_metal_timing_reset.restype = None
        lib.tg_metal_timing_reset.argtypes = [VP]

    return lib
