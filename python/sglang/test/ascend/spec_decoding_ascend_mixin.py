"""NPU (Ascend) mixin for single-request speculative decoding speed tests.

The BS=1 speed test is platform-agnostic on the HTTP layer, but lives in the
Ascend test tree so NPU ported test classes can pick it up from the same place
as ``GSM8KAscendMixin`` instead of reaching into the generic GPU kits.
"""

import requests

from sglang.test.send_one import BenchArgs, send_one_prompt
from sglang.test.test_utils import is_in_ci, write_github_step_summary


class SpecDecodingAscendMixin:
    """Mixin for a BS=1 speculative decoding speed/acceptance test on NPU.

    Required attributes on the test class:
        base_url: str
        model: str
        accept_length_thres: float
        bs_1_speed_thres: float

    Optional attributes:
        bs_1_speed_attempts: int = 3
        bs_1_speed_max_new_tokens: int = 2048
    """

    accept_length_thres: float
    bs_1_speed_thres: float
    bs_1_speed_attempts: int = 3
    bs_1_speed_max_new_tokens: int = 2048

    def test_bs_1_speed(self):
        args = BenchArgs(
            port=int(self.base_url.split(":")[-1]),
            max_new_tokens=self.bs_1_speed_max_new_tokens,
        )
        acc_length, speed = 0.0, 0.0
        for attempt in range(1, self.bs_1_speed_attempts + 1):
            requests.get(self.base_url + "/flush_cache")
            acc_length, speed = send_one_prompt(
                args, label=f"attempt {attempt}", print_output=False
            )
            if acc_length > self.accept_length_thres and speed > self.bs_1_speed_thres:
                break
        requests.get(self.base_url + "/flush_cache")

        if is_in_ci():
            write_github_step_summary(
                f"### test_bs_1_speed ({self.model})\n"
                f"{acc_length=:.2f}\n"
                f"{speed=:.2f} token/s\n"
            )

        self.assertGreater(acc_length, self.accept_length_thres)
        self.assertGreater(speed, self.bs_1_speed_thres)
