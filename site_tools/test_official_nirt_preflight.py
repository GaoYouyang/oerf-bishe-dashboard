from site_tools.official_nirt_preflight import (
    detect_static_hazards,
    estimate_memory_floor,
)


def test_static_hazards_detect_blockers_and_default_prediction() -> None:
    sources = {
        "NIRT.py": """
import tensorflow as tf
with tf.device('/GPU'):
    pred_flag           = 1
os.environ['XLA_FLAGS'] = 'xla_gpu_cuda_data_dir=C:\\\\cuda'
os.mkdir(savedir)
pdict = {'model': model}
""",
        "network.py": "x = tf.ones(3,1)\n",
        "fit.py": "value = pdict['auto_w']\n",
        "predict.py": "data_pred[prev_j:j].assign(value)\n",
        "setup.py": "scipy.io.loadmat('../Data/HSOF_'+str(fcase))\n",
    }
    hazards = detect_static_hazards(sources)
    codes = {item["code"] for item in hazards}
    assert "GPU_DEVICE_FORCED" in codes
    assert "WINDOWS_CUDA_PATH_HARDCODED" in codes
    assert "INVALID_TF_ONES_CALL" in codes
    assert "MISSING_AUTO_W_CONTRACT" in codes
    assert "TENSOR_SLICE_ASSIGN" in codes
    assert "CWD_RELATIVE_DATA_PATH" in codes
    assert "DEFAULT_IS_PREDICTION" in codes


def test_auto_w_hazard_disappears_when_contract_is_provided() -> None:
    sources = {
        "NIRT.py": "pdict = {'auto_w': 1}\n",
        "fit.py": "value = pdict['auto_w']\n",
    }
    codes = {item["code"] for item in detect_static_hazards(sources)}
    assert "MISSING_AUTO_W_CONTRACT" not in codes


def test_memory_estimate_is_explicit_lower_bound() -> None:
    result = estimate_memory_floor(10, (2, 3, 4))
    assert result["known_persistent_components_bytes"] == {
        "cam_data_n_by_18_float64": 1440,
        "b_data_n_by_4_float64": 320,
        "xyz_three_grids_float64": 576,
    }
    assert result["known_persistent_floor_bytes"] == 2336
    assert "LOWER_BOUND" in result["scope"]
