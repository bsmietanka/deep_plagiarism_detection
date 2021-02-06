import time
from collections import defaultdict
from typing import Callable, ContextManager


class MeasureExecutionTimeContext:

    def __init__(self, measurer, context_name: str):
        self.name = context_name
        self.measurer = measurer

    def __enter__(self):
        self.start_time = time.time()

    def __exit__(self, type, value, traceback):
        end_time = time.time()
        self.measurer.record_time(self.name, end_time - self.start_time)


class MeasureExecutionTime:

    def __init__(self):
        self.timing_datalog = defaultdict(lambda: {"times_called": 0, "total_time": 0.})

    def block(self, name: str) -> ContextManager:
        return MeasureExecutionTimeContext(self, name)

    def record_time(self, name: str, time: float):
        self.timing_datalog[name]["times_called"] += 1
        self.timing_datalog[name]["total_time"] += time

    def fun(self, fun: Callable):

        def timed_fun(*args, **kwargs):
            start_time = time.time()
            ret = fun(*args, **kwargs)
            end_time = time.time()
            self.record_time(fun.__name__, end_time - start_time)
            return ret

        return timed_fun

    def summary(self) -> dict:
        for v in self.timing_datalog.values():
            v["avg_call_time"] = v["total_time"] / v["times_called"]
        return dict(self.timing_datalog)


measure = MeasureExecutionTime()
