# Miscellanous utils

from time import perf_counter
import subprocess as sp


class catchtime:
    def __enter__(self):
        self.time = perf_counter()
        return self

    def __exit__(self, type, value, traceback):
        self.time = perf_counter() - self.time
        self.readout = f"Time: {self.time:.3f} seconds"
        print(self.readout)


def gpu_memory_usage():
    try:
        command = "nvidia-smi --query-gpu=memory.used --format=csv"
        output_cmd = sp.check_output(command.split())
        memory_used = output_cmd.decode("ascii").split("\n")[1]
        return memory_used

    except Exception as e:
        print(e)
        return None
