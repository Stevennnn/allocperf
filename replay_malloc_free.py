#!/usr/bin/env python3
"""Replay a malloc/free trace and report the resulting footprint.

    python replay_malloc_free.py malloc_free_trace.csv

New switches (independent, may be combined):

    --profile        record allocation history with _record_memory_history("all").
                     Off by default: it costs per-allocation overhead (measured
                     ~3.8x slower on a 149k-row trace), so leave it off when you
                     only want the footprint numbers.
    --snapshot [P]   dump a memory_viz-compatible pickle (pytorch.org/memory_viz).
                     Implies --profile, since device_traces is empty without it.
                     Default path: <trace>.snapshot.pickle, which is what the
                     original script always wrote.
    --last_step N    replay only rows whose `step` column is <= N. The build phase
                     is step -1, so it is always included for N >= -1;
                     --last_step -1 replays the build phase alone. Ignored (with a
                     warning) on a 3-column trace that has no step column.

"""

import argparse
import csv
import os
import sys
import time
from pathlib import Path

_conf = os.environ.get("PYTORCH_CUDA_ALLOC_CONF", "")
if "expandable_segments" not in _conf:
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = (_conf + "," if _conf else "") + "expandable_segments:True"

GiB = 1024**3


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("trace", help="op,size,addr[,step] CSV")
    p.add_argument("--profile", action="store_true",
                   help="record allocation history (needed for --snapshot; adds per-alloc overhead)")
    p.add_argument("--snapshot", nargs="?", const="auto", default=None, metavar="PATH",
                   help="dump a memory_viz pickle at the end (implies --profile); "
                        "default <trace>.snapshot.pickle")
    p.add_argument("--last_step", "--last-step", dest="last_step", type=int, default=None,
                   help="replay only rows with step <= N (build phase is step -1)")
    p.add_argument("--step-report", action=argparse.BooleanOptionalAction, default=True,
                   help="print one line per step with its malloc/free counts and wall time "
                        "(default: on; --no-step-report to silence)")
    return p.parse_args()


def load_trace(path: str, last_step):
    with open(path, newline="") as fh:
        reader = csv.DictReader(fh)
        has_step = "step" in (reader.fieldnames or [])
        raw = [(r["op"], int(r["size"]), int(r["addr"]),
                int(r["step"]) if has_step and r.get("step") not in (None, "") else None)
               for r in reader]
    if last_step is None:
        kept = raw
    elif not has_step:
        print(f"[warn] --last_step {last_step} ignored: {path} has no `step` column")
        kept = raw
    else:
        kept = [r for r in raw if r[3] <= last_step]
    by_step = {}
    for r in kept:
        by_step[r[3]] = by_step.get(r[3], 0) + 1
    return kept, has_step, by_step


def main() -> int:
    args = parse_args()
    src = Path(args.trace)
    if not src.exists():
        raise SystemExit(f"no such file: {src}")

    snapshot_path = args.snapshot
    if snapshot_path == "auto":
        snapshot_path = str(src.with_suffix(".snapshot.pickle"))
    profile = bool(args.profile) or snapshot_path is not None
    if snapshot_path is not None and not args.profile:
        print("[note] --snapshot implies --profile: allocation history recording is ON")

    rows, has_step, by_step = load_trace(str(src), args.last_step)

    import torch

    dev = torch.device("cuda")
    torch.cuda.set_device(0)
    print(f"[env] torch={torch.__version__} cuda={torch.version.cuda} "
          f"device={torch.cuda.get_device_name(0)} conf={os.environ['PYTORCH_CUDA_ALLOC_CONF']}")
    print(f"[opts] profile={profile} snapshot={snapshot_path or '-'} "
          f"last_step={'all' if args.last_step is None else args.last_step}")
    print(f"[trace] {len(rows)} rows from {src}")
    if args.last_step is not None and has_step:
        print("[steps] rows kept per step: " + " ".join(
            f"{('build' if k == -1 else ('post' if k == -2 else k))}:{v}"
            for k, v in sorted(by_step.items(), key=lambda kv: (kv[0] is None, kv[0]))))

    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    if profile:
        torch.cuda.memory._record_memory_history("all")
    torch.cuda.reset_peak_memory_stats(dev)

    def step_label(s):
        return "build" if s == -1 else ("post" if s == -2 else ("all" if s is None else str(s)))

    def report(label, n_malloc, n_free, dt):
        # A trace replay issues no kernels at all (torch.empty + tensor drops are
        # pure allocator calls), so this wall time is host-side allocator time
        # and needs no synchronize to be meaningful.
        print(f"[step {label:>5}] malloc={n_malloc:<8} free={n_free:<8} {dt:8.3f}s", flush=True)

    live = {}
    oom_at = None
    cur_step = object()      # sentinel: differs from any real step value
    n_malloc = n_free = 0
    t_step = time.perf_counter()
    t_all = time.perf_counter()
    for i, (op, size, addr, step) in enumerate(rows):
        if args.step_report and step != cur_step:
            # First row only sets cur_step; the guard keeps it from printing an
            # empty line for the sentinel.
            if n_malloc or n_free:
                report(step_label(cur_step), n_malloc, n_free, time.perf_counter() - t_step)
            cur_step = step
            n_malloc = n_free = 0
            t_step = time.perf_counter()
        try:
            if op == "malloc":
                live[addr] = torch.empty(size, dtype=torch.uint8, device=dev)
                n_malloc += 1
            else:
                live.pop(addr, None)
                n_free += 1
        except torch.cuda.OutOfMemoryError as exc:
            oom_at = i
            print(f"[OOM] row {i} (step {step_label(step)}, {op}, {size / GiB:.3f} GiB): "
                  f"{str(exc)[:120]}")
            break
    if args.step_report and (n_malloc or n_free):
        report(step_label(cur_step), n_malloc, n_free, time.perf_counter() - t_step)
    total_s = time.perf_counter() - t_all

    # Read the peaks from the allocator's own counters instead of calling
    # memory_stats() on every row: same values, and it removes ~150k dict
    # constructions from the loop.
    torch.cuda.synchronize()
    st = torch.cuda.memory_stats(dev)
    peak_alloc = st["allocated_bytes.all.peak"]
    peak_rsv = st["reserved_bytes.all.peak"]

    if snapshot_path:
        Path(snapshot_path).parent.mkdir(parents=True, exist_ok=True)
        torch.cuda.memory._dump_snapshot(snapshot_path)
        print(f"[snapshot] {snapshot_path} ({Path(snapshot_path).stat().st_size / 1e6:.1f} MB) "
              f"live={len(live)} allocated={st['allocated_bytes.all.current'] / GiB:.4f} GiB "
              f"reserved={st['reserved_bytes.all.current'] / GiB:.4f} GiB")
    if profile:
        torch.cuda.memory._record_memory_history(None)

    print(f"[done] rows={len(rows)} live at end={len(live)} oom={oom_at} "
          f"replay={total_s:.3f}s")
    print(f"  peak allocated = {peak_alloc / GiB:.4f} GiB")
    print(f"  peak reserved  = {peak_rsv / GiB:.4f} GiB")
    if not snapshot_path:
        print("  (no snapshot written; add --snapshot for a pytorch.org/memory_viz pickle)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
