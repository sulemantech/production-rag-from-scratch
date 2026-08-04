import time
from contextlib import contextmanager


@contextmanager
def timed_stage(name: str, timings: dict):
    """
    Context manager that records how long the enclosed block took to run,
    storing the result (in seconds) into timings[name].
    """
    if timings is None:
        yield
        return
    # Record the start time
    start_time = time.perf_counter()

    try:
        # Execute the code inside the with block
        yield
    finally:
        # Compute elapsed time and store it
        elapsed = time.perf_counter() - start_time
        timings[name] = elapsed


def summarize(all_timings: list[dict]) -> dict:
    """
    Given a list of per-question timings dicts (each like
    {"retrieval": 0.42, "rerank": 0.11, "generation": 0.98}), return a
    summary per stage:
    {
        "retrieval": {"min": ..., "avg": ..., "max": ...},
        ...
    }
    """
    summary = {}

    # Collect all unique stage names
    stage_names = set()
    for timings in all_timings:
        stage_names.update(timings.keys())

    # Compute statistics for each stage
    for stage in stage_names:
        values = [
            timing[stage]
            for timing in all_timings
            if stage in timing
        ]

        summary[stage] = {
            "min": min(values),
            "avg": sum(values) / len(values),
            "max": max(values),
        }

    return summary


if __name__ == "__main__":
    print(summarize([
        {"a": 1, "b": 2},
        {"a": 3, "b": 4}
    ]))