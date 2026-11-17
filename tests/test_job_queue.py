"""The job queue: every rule of ``ipomdp.infra.job_queue`` on tiny real processes.

Each job here is a shell command that sleeps briefly and writes a file, so the
tests exercise real process start, exit status, accounting and files, not a
mock of them. Memory rules are tested with small declared peaks and with the
two measurements (machine memory, process-tree memory) replaced by functions
the test controls; nothing here allocates real memory.
"""

import json
import sys
import time

import pytest

from ipomdp.infra.job_queue import Job, main, read_queue, run_queue

LIMITS = dict(rss_cap_mb=10_000, min_available_mb=0, poll_seconds=0.02, available_mb=lambda: 1e9,
              tree_rss_mb=lambda pid: 0.0)


def writer(out, seconds=0.05, text="done"):
    """A command that waits and then writes ``out`` atomically, as the package's drivers do."""
    return f"sleep {seconds} && printf {text} > {out}.tmp && mv {out}.tmp {out}"


def records(log):
    return [json.loads(line) for line in log.read_text().splitlines()]


def run(jobs, tmp_path, **options):
    return run_queue(jobs, workers=options.pop("workers", 2), log_path=tmp_path / "log.jsonl",
                     stop_file=options.pop("stop_file", tmp_path / "STOP"), **{**LIMITS, **options})


def test_every_job_runs_once_and_is_recorded_by_its_output(tmp_path):
    jobs = [Job(f"cell:{k % 2}", str(tmp_path / f"rows/{k}.json"), writer(tmp_path / f"rows/{k}.json")) for k in range(5)]
    summary = run(jobs, tmp_path)
    assert summary == {"done": 0, "ran": 5, "failed": [], "blocked": [], "stopped": 0}
    rows = records(tmp_path / "log.jsonl")
    assert {row["out"] for row in rows} == {job.out for job in jobs}
    for row in rows:
        assert row["ok"] and row["rc"] == 0 and row["wall_seconds"] >= 0.04
        assert row["cpu_seconds"] >= 0 and row["peak_rss_mb"] > 0 and row["tag"].startswith("cell:")
    # A second invocation finds every output and starts nothing.
    again = run(jobs, tmp_path)
    assert again["done"] == 5 and again["ran"] == 0 and len(records(tmp_path / "log.jsonl")) == 5


def test_a_failed_job_and_a_job_without_its_output_are_failures_and_the_rest_still_run(tmp_path):
    jobs = [
        Job("a", str(tmp_path / "ok.json"), writer(tmp_path / "ok.json")),
        Job("a", str(tmp_path / "crash.json"), "exit 3"),
        Job("a", str(tmp_path / "silent.json"), "true"),  # exit status 0, no output
    ]
    summary = run(jobs, tmp_path)
    assert summary["ran"] == 1
    assert sorted(summary["failed"]) == sorted([(str(tmp_path / "crash.json"), 3), (str(tmp_path / "silent.json"), 0)])
    by_out = {row["out"]: row for row in records(tmp_path / "log.jsonl")}
    assert not by_out[str(tmp_path / "silent.json")]["ok"] and by_out[str(tmp_path / "silent.json")]["rc"] == 0
    # The failures have no output, so the next invocation runs exactly those again.
    assert run(jobs, tmp_path)["done"] == 1


def test_no_more_jobs_run_at_once_than_workers(tmp_path):
    jobs = [Job("a", str(tmp_path / f"{k}.json"), writer(tmp_path / f"{k}.json", 0.15)) for k in range(6)]
    run(jobs, tmp_path, workers=2)
    spans = []
    for row in records(tmp_path / "log.jsonl"):
        end = (tmp_path / row["out"]).stat().st_mtime
        spans.append((end - row["wall_seconds"], end))
    for start, _ in spans:
        assert sum(1 for other_start, other_end in spans if other_start <= start < other_end - 0.02) <= 2
    started = time.monotonic()
    run([Job("a", str(tmp_path / f"p{k}.json"), writer(tmp_path / f"p{k}.json", 0.2)) for k in range(4)], tmp_path, workers=4)
    assert time.monotonic() - started < 0.7  # four at once, not one after another


def test_the_stop_file_stops_new_starts_and_running_jobs_finish(tmp_path):
    stop = tmp_path / "STOP"
    first = tmp_path / "first.json"
    jobs = [Job("a", str(first), f"sleep 0.1 && touch {stop} && printf x > {first}")] + [
        Job("a", str(tmp_path / f"later{k}.json"), writer(tmp_path / f"later{k}.json")) for k in range(3)
    ]
    summary = run(jobs, tmp_path, workers=1, stop_file=stop)
    assert first.exists() and summary["ran"] == 1 and summary["stopped"] == 3
    stop.unlink()
    assert run(jobs, tmp_path, workers=1, stop_file=stop)["ran"] == 3


def test_the_memory_budget_counts_declared_peaks_and_a_job_above_the_cap_runs_alone(tmp_path):
    def overlap(log):
        spans = []
        for row in records(log):
            end = (tmp_path / row["out"]).stat().st_mtime
            spans.append((end - row["wall_seconds"], end, row["out"]))
        return max(sum(1 for s, e, _ in spans if s <= start < e - 0.02) for start, _, _ in spans)

    # Peaks of 600 under a cap of 1,000: never two at once, although four workers are allowed.
    jobs = [Job("a", str(tmp_path / f"{k}.json"), writer(tmp_path / f"{k}.json", 0.1), 600.0) for k in range(3)]
    run_queue(jobs, workers=4, log_path=tmp_path / "budget.jsonl", stop_file=tmp_path / "STOP",
              **{**LIMITS, "rss_cap_mb": 1000})
    assert overlap(tmp_path / "budget.jsonl") == 1
    # Peaks of 400 under the same cap: two at once, not three.
    jobs = [Job("a", str(tmp_path / f"s{k}.json"), writer(tmp_path / f"s{k}.json", 0.15), 400.0) for k in range(4)]
    run_queue(jobs, workers=4, log_path=tmp_path / "pairs.jsonl", stop_file=tmp_path / "STOP",
              **{**LIMITS, "rss_cap_mb": 1000})
    assert overlap(tmp_path / "pairs.jsonl") == 2
    # A job declared above the cap still runs, alone, and in its turn: the small job behind it waits.
    jobs = [
        Job("a", str(tmp_path / "small0.json"), writer(tmp_path / "small0.json", 0.1), 300.0),
        Job("big", str(tmp_path / "big.json"), writer(tmp_path / "big.json", 0.1), 5000.0),
        Job("a", str(tmp_path / "small1.json"), writer(tmp_path / "small1.json", 0.1), 300.0),
    ]
    run_queue(jobs, workers=4, log_path=tmp_path / "alone.jsonl", stop_file=tmp_path / "STOP",
              **{**LIMITS, "rss_cap_mb": 1000})
    assert overlap(tmp_path / "alone.jsonl") == 1
    assert [row["out"].rsplit("/", 1)[1] for row in records(tmp_path / "alone.jsonl")] == [
        "small0.json", "big.json", "small1.json"]


def test_a_job_measured_above_its_declared_peak_reserves_what_it_really_takes(tmp_path):
    # Each job declares 100 but is "measured" at 950: 950 + 100 exceeds the cap of 1,000, so the second waits
    # (by the declared figures alone, 100 + 100, both would have started at once).
    jobs = [Job("a", str(tmp_path / f"{k}.json"), writer(tmp_path / f"{k}.json", 0.15), 100.0) for k in range(2)]
    run_queue(jobs, workers=2, log_path=tmp_path / "log.jsonl", stop_file=tmp_path / "STOP",
              **{**LIMITS, "rss_cap_mb": 1000, "tree_rss_mb": lambda pid: 950.0})
    rows = records(tmp_path / "log.jsonl")
    ends = sorted((tmp_path / row["out"]).stat().st_mtime for row in rows)
    assert ends[1] - ends[0] >= 0.12  # one after the other


def test_nothing_starts_while_the_machine_is_short_of_memory(tmp_path):
    readings = iter([100.0] * 8)  # eight polls below the threshold, then plenty

    def available():
        return next(readings, 50_000.0)

    out = tmp_path / "row.json"
    started = time.monotonic()
    summary = run_queue([Job("a", str(out), writer(out))], workers=1, log_path=tmp_path / "log.jsonl",
                        stop_file=tmp_path / "STOP", **{**LIMITS, "min_available_mb": 8000, "available_mb": available})
    assert summary["ran"] == 1 and time.monotonic() - started >= 8 * 0.02


def test_jobs_wait_for_their_inputs_and_are_reported_blocked_when_those_never_appear(tmp_path):
    reference, arm, orphan = tmp_path / "ref.json", tmp_path / "arm.json", tmp_path / "orphan.json"
    jobs = [
        Job("arms", str(arm), writer(arm), needs=(str(reference),)),      # listed first, runs second
        Job("arms", str(orphan), writer(orphan), needs=(str(tmp_path / "never.json"),)),
        Job("refs", str(reference), writer(reference, 0.1)),
    ]
    summary = run(jobs, tmp_path)
    assert arm.exists() and reference.exists() and not orphan.exists()
    assert summary["ran"] == 2 and summary["blocked"] == [str(orphan)]
    assert reference.stat().st_mtime <= arm.stat().st_mtime


def test_queue_files_are_parsed_strictly_and_the_command_line_runs_a_queue(tmp_path):
    out = tmp_path / "row.json"
    queue = tmp_path / "queue.tsv"
    queue.write_text(f"cell:x\t{out}\t{writer(out)}\t250\t\n\ncell:y\t{tmp_path / 'b.json'}\t{writer(tmp_path / 'b.json')}\n")
    jobs = read_queue(queue)
    assert [(job.tag, job.peak_mb, job.needs) for job in jobs] == [("cell:x", 250.0, ()), ("cell:y", None, ())]
    status = main([str(queue), "--workers", "2", "--log", str(tmp_path / "cli.jsonl"), "--stop-file", str(tmp_path / "STOP"),
                   "--rss-cap-mb", "6000", "--min-available-mb", "0", "--tags", "cell", "--first", "1"])
    assert status == 0 and out.exists() and (tmp_path / "b.json").exists()
    queue.write_text(f"a\t{out}\ttrue\na\t{out}\ttrue\n")
    with pytest.raises(ValueError, match="appears twice"):
        read_queue(queue)
    queue.write_text("a\tonly-two-fields\n")
    with pytest.raises(ValueError, match="3 to 5"):
        read_queue(queue)
    with pytest.raises(ValueError):
        Job("a", str(out), "true", peak_mb=0)
    with pytest.raises(ValueError):
        run_queue([], workers=0, log_path=tmp_path / "x.jsonl", stop_file=tmp_path / "STOP", **LIMITS)


def test_jobs_run_niced_in_their_own_session_with_one_numerical_thread(tmp_path):
    out = tmp_path / "env.json"
    probe = ("import json, os; json.dump({'nice': os.nice(0), 'session': os.getsid(0) == os.getpid() or os.getsid(0) != "
             f"{__import__('os').getsid(0)}, 'threads': os.environ.get('OMP_NUM_THREADS')}}, open('{out}', 'w'))")
    run([Job("a", str(out), f'{sys.executable} -c "{probe}"')], tmp_path)
    seen = json.loads(out.read_text())
    assert seen["nice"] >= 10 and seen["session"] and seen["threads"] == "1"
