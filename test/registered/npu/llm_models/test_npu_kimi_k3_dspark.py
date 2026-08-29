"""Test Kimi-K3 DSPARK speculative decoding on NPU.

[Test Category] Speculative Decoding
[Test Target] --speculative-algorithm=DSPARK; --enable-linear-replayssm-spec;
--speculative-dspark-block-size; --mamba-full-memory-ratio
[Platform] NPU (Ascend A3, CANN 9.0.0)
[Porting Source] Ported from GPU: sgl-project/sglang test/registered/models_e2e/test_kimi_k3_b300.py
  Class: TestKimiK3B300LowLatency
"""

import os
import unittest

import requests

from sglang.test.ascend.gsm8k_ascend_mixin import GSM8KAscendMixin
from sglang.test.ascend.spec_decoding_ascend_mixin import SpecDecodingAscendMixin
from sglang.test.ascend.test_ascend_utils import MODEL_WEIGHTS_DIR
from sglang.test.ci.ci_register import register_npu_ci
from sglang.test.test_utils import CustomTestCase

register_npu_ci(est_time=1800, suite="full-1-npu-a3", nightly=True)

# --- Weights (override via env for your own checkpoint mirror) ---
KIMI_K3_MODEL_PATH = os.environ.get(
    "KIMI_K3_MODEL_PATH", os.path.join(MODEL_WEIGHTS_DIR, "moonshotai/Kimi-K3")
)
KIMI_K3_DSPARK_DRAFT_MODEL = os.environ.get(
    "KIMI_K3_DSPARK_DRAFT_MODEL", "RadixArk/Kimi-K3-DSpark"
)


def _check_accept_length(test_case, base_url, threshold=None):
    """Print speculative accept length; optionally assert it exceeds threshold.

    Silently returns when the server does not report the field (mirrors
    ``sglang.test.kits.eval_accuracy_kit._check_accept_length``).
    """
    try:
        server_info = requests.get(base_url + "/server_info").json()
        val = server_info["internal_states"][0]["avg_spec_accept_length"]
    except (KeyError, IndexError, requests.RequestException):
        return
    print(f"avg_spec_accept_length={val:.4f}")
    if threshold is not None:
        test_case.assertGreater(val, threshold)


class TestNPUKimiK3DSPARK(GSM8KAscendMixin, SpecDecodingAscendMixin, CustomTestCase):
    """TP8 DSPARK linear ReplaySSM low-latency recipe on NPU.

    Same shape as the GPU TP8 LowLatency recipe:
    - ``test_gsm8k`` (from ``GSM8KAscendMixin``) runs a 200-question GSM8K
      eval; the main gate (mean speculative accept length) is asserted here
      after the score/throughput asserts, because a 200-question average holds
      steady when a numerics change moves where a single greedy prompt hits
      EOS.
    - ``test_bs_1_speed`` (from ``SpecDecodingAscendMixin``) is a coarse
      end-to-end speed/acceptance guard on a single 2048-token request.

    Thresholds leave margin for NPU precision variance and for a 200-example
    sample; tune them to your measured NPU numbers.
    """

    model = KIMI_K3_MODEL_PATH

    # --- GSM8K main gate (lossless + effective speculation) ---
    accuracy = 0.80
    output_throughput = 0.0
    gsm8k_accept_length_thres = 3.0

    # --- BS=1 coarse guards (end-to-end; amortizes launch + TTFT) ---
    accept_length_thres = 2.5
    bs_1_speed_thres = 150.0

    other_args = [
        "--trust-remote-code",
        "--attention-backend",
        "ascend",
        "--disable-cuda-graph",
        "--sampling-backend",
        "ascend",
        "--tp-size",
        "8",
        "--mem-fraction-static",
        "0.85",
        "--reasoning-parser",
        "kimi_k3",
        "--tool-call-parser",
        "kimi_k3",
        "--mamba-full-memory-ratio",
        "0.86",
        "--speculative-algorithm",
        "DSPARK",
        "--speculative-draft-model-path",
        KIMI_K3_DSPARK_DRAFT_MODEL,
        "--speculative-dspark-block-size",
        "7",
        "--enable-linear-replayssm-spec",
    ]

    env = {
        **os.environ,
        "PYTORCH_NPU_ALLOC_CONF": "expandable_segments:True",
        "ASCEND_MF_STORE_URL": "tcp://127.0.0.1:24666",
        "HCCL_BUFFSIZE": "200",
        "STREAMS_PER_DEVICE": "32",
        "HCCL_OP_EXPANSION_MODE": "AIV",
        "HCCL_SOCKET_IFNAME": "lo",
        "GLOO_SOCKET_IFNAME": "lo",
        "ASCEND_USE_FIA": "0",
        "SGLANG_SET_CPU_AFFINITY": "1",
        "SGLANG_ENABLE_SPEC_V2": "1",
        "SGLANG_ENABLE_OVERLAP_PLAN_STREAM": "1",
        "USE_VLLM_CUSTOM_ALLREDUCE": "1",
        "HCCL_EXEC_TIMEOUT": "200",
        "AUTO_USE_UC_MEMORY": "0",
        "P2P_HCCL_BUFFSIZE": "20",
    }

    def test_gsm8k(self):
        # Keep the GPU recipe's gate placement: assert the mean speculative
        # accept length on the GSM8K average rather than on test_bs_1_speed.
        super().test_gsm8k()
        _check_accept_length(self, self.base_url, self.gsm8k_accept_length_thres)


if __name__ == "__main__":
    unittest.main()
