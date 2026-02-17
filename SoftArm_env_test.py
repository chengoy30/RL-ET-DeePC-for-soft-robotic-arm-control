import time
from SoftArm_env import SoftArmEnv


class TimedSoftArmEnv(SoftArmEnv):
    """
    Testing-only subclass of SoftArmEnv that records the wall-clock time
    of every DeePC solve() call.

    - When action=1, the actual solve time is recorded.
    - When action=0, the recorded time is 0.0.
    - Timing data is stored in self.deepc_times (one entry per step).
    """

    def __init__(self, param_deepc, arm_section, y_desired, rho):
        super().__init__(param_deepc, arm_section, y_desired, rho)
        self.deepc_times = []
        self._last_solve_time = 0.0
        self._wrap_deepc_solve()

    def _wrap_deepc_solve(self):
        original_solve = self.deepc.solve

        def timed_solve(*args, **kwargs):
            t0 = time.perf_counter()
            result = original_solve(*args, **kwargs)
            self._last_solve_time = time.perf_counter() - t0
            return result

        self.deepc.solve = timed_solve

    def reset(self, seed=None):
        self.deepc_times = []
        self._last_solve_time = 0.0
        return super().reset(seed=seed)

    def step(self, action):
        self._last_solve_time = 0.0
        result = super().step(action)
        self.deepc_times.append(self._last_solve_time)
        return result

    @property
    def total_deepc_time(self):
        return sum(self.deepc_times)

    @property
    def deepc_call_count(self):
        return sum(1 for t in self.deepc_times if t > 0)
