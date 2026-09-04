#!/usr/bin/env python3
"""
sweep_and_plot.py

Sweeps dataset size (e.g. 100K / 500K / 1M / 2M records) across the CPU,
OpenMP, and GPU builds of the Salsa20+Kyber-512 hybrid pipeline. For each
(platform, size) pair it runs several repeats and measures:

  - Throughput (records/s, MB/s) — parsed from the program's own
    "Full pipeline" summary line, so this reuses each program's own
    cudaEvent_t / clock_gettime timing rather than timing the process
    externally.
  - Peak memory usage:
      CPU / OpenMP -> peak RSS of the process (+ any child processes),
                      sampled via psutil while the program runs.
      GPU          -> peak per-process GPU memory, sampled via
                      `nvidia-smi --query-compute-apps=pid,used_memory`.
  - Power draw:
      CPU / OpenMP -> Linux RAPL package-energy counters
                      (/sys/class/powercap/intel-rapl:0/energy_uj),
                      converted to average watts over the run.
      GPU          -> `nvidia-smi --query-gpu=power.draw`, sampled
                      periodically during the run (average + peak).

Then it writes a CSV of all results and three plots (throughput, memory,
power, each vs. dataset size, one line per platform, with error bars from
the repeats).

IMPORTANT CAVEATS (please read before trusting the numbers in a paper):
  - RAPL measures CPU *package* power (cores + uncore), not whole-system
    power. It's a standard, widely-used proxy in the literature, but it is
    NOT the same as wall-socket power. If you have a physical power meter
    or IPMI/BMC access, that's a stronger source for a Q1 submission — this
    script's RAPL path is a reasonable software-only fallback.
  - RAPL requires read access to /sys/class/powercap/intel-rapl:0/energy_uj
    and only exists on Intel CPUs. On AMD/ARM, or without permission, power
    for CPU/OpenMP will come back as None — the script will say so rather
    than silently reporting a wrong number. Fix permissions with:
        sudo chmod -R a+r /sys/class/powercap/intel-rapl*
    or run this script with sudo (not generally recommended, but an option
    for a benchmarking machine).
  - nvidia-smi's power.draw is whole-GPU power, not attributable to this
    process alone if something else is using the GPU concurrently — run
    benchmarks on an otherwise-idle GPU.
  - This script has NOT been run end-to-end in the environment it was
    written in (no GPU, no RAPL access, no compiled binaries available
    there) — please sanity-check the first run's output columns before
    trusting a full sweep, especially the regex that parses each program's
    stdout (see parse_output()) in case your build's print formatting
    differs slightly from what's assumed here.

Requirements:
    pip install psutil matplotlib

Usage:
    python3 sweep_and_plot.py \
        --dataset /path/to/data.txt \
        --cpu-bin ./sensor_hybrid_crypto_cpu \
        --openmp-bin ./sensor_hybrid_crypto_openmp \
        --gpu-bin ./sensor_hybrid_crypto_kyber \
        --sizes 100000 500000 1000000 2000000 \
        --repeats 5
"""

import argparse
import csv
import re
import subprocess
import sys
import threading
import time
from statistics import mean, stdev
from pathlib import Path

try:
    import psutil
except ImportError:
    psutil = None

# ==========================================================================
# Parsing each program's stdout
# ==========================================================================
# Matches the "Full pipeline" line printed by all three programs, e.g.:
#   "  Full pipeline :  81369506.5 records/s   |   4967.35 MB/s (padded)  |   4966.40 MB/s (useful)"
RE_FULL_PIPELINE = re.compile(
    r"Full pipeline\s*:\s*([\d.]+)\s*records/s\s*\|\s*([\d.]+)\s*MB/s \(padded\)\s*\|\s*([\d.]+)\s*MB/s \(useful\)"
)
# Fallback for older binary builds that only print the single-line summary,
# e.g.: "End-to-end encrypt+decrypt throughput: 9606929.6 records/s"
# (no MB/s breakdown available in this format — those columns stay None).
RE_LEGACY_THROUGHPUT = re.compile(
    r"End-to-end encrypt\+decrypt throughput:\s*([\d.]+)\s*records/s"
)
RE_VERIFY = re.compile(r"Verification:\s*(\d+)\s*/\s*(\d+)\s*records passed")


def parse_output(stdout_text):
    m = RE_FULL_PIPELINE.search(stdout_text)
    if m:
        result = {
            "records_per_s": float(m.group(1)),
            "mb_per_s_padded": float(m.group(2)),
            "mb_per_s_useful": float(m.group(3)),
        }
    else:
        m2 = RE_LEGACY_THROUGHPUT.search(stdout_text)
        if not m2:
            return None
        print("NOTE: parsed the OLDER single-line throughput format (no 'Full pipeline' "
              "line found) — this binary is likely a stale build. MB/s columns will be "
              "empty for this run. Recompile from the latest .cu/.c source to get the "
              "full breakdown matching the other platforms.", file=sys.stderr)
        result = {
            "records_per_s": float(m2.group(1)),
            "mb_per_s_padded": None,
            "mb_per_s_useful": None,
        }
    v = RE_VERIFY.search(stdout_text)
    if v:
        passed, total = int(v.group(1)), int(v.group(2))
        result["verify_passed"] = passed
        result["verify_total"] = total
        if passed != total:
            print(f"WARNING: only {passed}/{total} records verified — "
                  f"correctness issue, results may not be trustworthy for this run.",
                  file=sys.stderr)
    return result


# ==========================================================================
# RAPL energy reading (CPU / OpenMP power)
# ==========================================================================
RAPL_PATH = Path("/sys/class/powercap/intel-rapl:0/energy_uj")


def read_rapl_energy_uj():
    try:
        return int(RAPL_PATH.read_text().strip())
    except Exception:
        return None


_RAPL_AVAILABLE = read_rapl_energy_uj() is not None
if not _RAPL_AVAILABLE:
    print(f"NOTE: RAPL energy counter not readable at {RAPL_PATH} — "
          f"CPU/OpenMP power will be reported as None. See script docstring "
          f"for how to enable it.", file=sys.stderr)


# ==========================================================================
# GPU sampling thread: power draw (whole-GPU) + per-process memory
# ==========================================================================
class GpuSampler(threading.Thread):
    def __init__(self, target_pid, interval=0.05):
        super().__init__()
        self.target_pid = target_pid
        self.interval = interval
        self.power_samples = []     # watts
        self.proc_mem_samples = []  # MiB, this process only
        self._stop_event = threading.Event()

    def run(self):
        while not self._stop_event.is_set():
            try:
                out = subprocess.check_output(
                    ["nvidia-smi", "--query-gpu=power.draw",
                     "--format=csv,noheader,nounits"],
                    stderr=subprocess.DEVNULL, timeout=2
                ).decode().strip()
                first_line = out.splitlines()[0]
                self.power_samples.append(float(first_line))
            except Exception:
                pass
            try:
                out2 = subprocess.check_output(
                    ["nvidia-smi", "--query-compute-apps=pid,used_memory",
                     "--format=csv,noheader,nounits"],
                    stderr=subprocess.DEVNULL, timeout=2
                ).decode().strip()
                for line in out2.splitlines():
                    parts = [x.strip() for x in line.split(",")]
                    if len(parts) != 2:
                        continue
                    pid_str, mem_str = parts
                    try:
                        if int(pid_str) == self.target_pid:
                            self.proc_mem_samples.append(float(mem_str))
                    except ValueError:
                        continue
            except Exception:
                pass
            time.sleep(self.interval)

    def stop(self):
        self._stop_event.set()


# ==========================================================================
# CPU RSS + utilization sampling thread (psutil) — used for CPU and OpenMP runs.
# CPU utilization is tracked here too because it's needed for the TDP-scaled
# power ESTIMATE (see estimate_power_tdp() below) used when no hardware power
# source (RAPL / BMC) is available — as is typical inside a VM/container.
# ==========================================================================
class RssSampler(threading.Thread):
    def __init__(self, pid, interval=0.05):
        super().__init__()
        self.pid = pid
        self.interval = interval
        self.peak_rss_mb = 0.0
        self.cpu_util_samples = []  # fraction of TOTAL system CPU capacity used (0..1), not per-core
        self._stop_event = threading.Event()

    def run(self):
        if psutil is None:
            return
        try:
            proc = psutil.Process(self.pid)
            ncpu = psutil.cpu_count(logical=True) or 1
            proc.cpu_percent(interval=None)  # prime the internal counter (first call is always 0.0)
        except Exception:
            return
        while not self._stop_event.is_set():
            try:
                rss = proc.memory_info().rss
                children = proc.children(recursive=True)
                for child in children:
                    try:
                        rss += child.memory_info().rss
                    except Exception:
                        pass
                self.peak_rss_mb = max(self.peak_rss_mb, rss / (1024 * 1024))

                # psutil's cpu_percent() is normalized so 100% == one full core;
                # divide by ncpu to get "fraction of total machine capacity",
                # which is what the TDP scaling model needs. Include children
                # (e.g. OpenMP worker threads spawned as a thread pool are
                # already counted in the parent's own cpu_percent, but if the
                # build forks child processes instead, this catches those too).
                cpu_pct = proc.cpu_percent(interval=None)
                for child in children:
                    try:
                        cpu_pct += child.cpu_percent(interval=None)
                    except Exception:
                        pass
                self.cpu_util_samples.append(min(cpu_pct / ncpu / 100.0, 1.0))
            except Exception:
                break
            time.sleep(self.interval)

    def stop(self):
        self._stop_event.set()


def estimate_power_tdp(cpu_util_samples, tdp_watts, idle_fraction):
    """
    TDP-scaled linear power model, for use ONLY when no hardware power source
    (RAPL / BMC / physical meter) is available — e.g. inside a VM/container.

        P_est = P_idle + (TDP - P_idle) * mean(utilization)

    where utilization is the sampled fraction of TOTAL system CPU capacity
    used by the process (0..1). This is an ESTIMATE, not a measurement —
    state that explicitly wherever these numbers are reported (figure
    captions, table footnotes, and the methods section).
    """
    if not cpu_util_samples or tdp_watts is None:
        return None
    u = mean(cpu_util_samples)
    p_idle = idle_fraction * tdp_watts
    return p_idle + (tdp_watts - p_idle) * u


# ==========================================================================
# Run one program invocation with concurrent memory/power sampling
# ==========================================================================
def run_once(binary, dataset, size, is_gpu, gpu_sampling=True, cpu_tdp_watts=None, cpu_idle_fraction=0.3):
    cmd = [binary, dataset, str(size)]

    rapl_before = None if is_gpu else read_rapl_energy_uj()

    t0 = time.time()
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)

    rss_sampler = RssSampler(proc.pid) if (psutil and not is_gpu) else None
    gpu_sampler = None
    if is_gpu and gpu_sampling:
        # Grace period before polling starts: nvidia-smi queries firing the
        # instant the process launches can overlap with CUDA context
        # creation and has been observed to destabilize some driver/GPU
        # combinations. Give the process a moment to get past init first.
        time.sleep(0.3)
        gpu_sampler = GpuSampler(proc.pid)
    if rss_sampler:
        rss_sampler.start()
    if gpu_sampler:
        gpu_sampler.start()

    stdout_text, _ = proc.communicate()
    t1 = time.time()

    if rss_sampler:
        rss_sampler.stop()
        rss_sampler.join()
    if gpu_sampler:
        gpu_sampler.stop()
        gpu_sampler.join()

    wall_s = t1 - t0

    if proc.returncode != 0:
        print(f"WARNING: {cmd} exited with code {proc.returncode}", file=sys.stderr)
        print(stdout_text[-2000:], file=sys.stderr)
        return None

    result = parse_output(stdout_text)
    if result is None:
        print(f"WARNING: could not parse throughput line from output of {cmd}", file=sys.stderr)
        print(stdout_text[-2000:], file=sys.stderr)
        return None

    result["wall_s"] = wall_s

    # ---- memory ----
    if is_gpu and gpu_sampler is not None:
        if gpu_sampler.proc_mem_samples:
            result["peak_mem_mb"] = max(gpu_sampler.proc_mem_samples)
        else:
            result["peak_mem_mb"] = None  # nvidia-smi couldn't attribute memory to this pid
    elif rss_sampler is not None:
        result["peak_mem_mb"] = rss_sampler.peak_rss_mb
    else:
        result["peak_mem_mb"] = None

    # ---- power ----
    result["power_is_estimated"] = False  # flips to True only for the TDP-model fallback below
    if is_gpu and gpu_sampler is not None and gpu_sampler.power_samples:
        result["avg_power_w"] = mean(gpu_sampler.power_samples)
        result["peak_power_w"] = max(gpu_sampler.power_samples)
        result["energy_j"] = result["avg_power_w"] * wall_s
    elif not is_gpu and rapl_before is not None:
        rapl_after = read_rapl_energy_uj()
        if rapl_after is not None:
            # RAPL counters can wrap around; if that happened this sample is
            # unreliable, so drop it rather than report a nonsense negative.
            if rapl_after >= rapl_before:
                energy_j = (rapl_after - rapl_before) / 1e6
                result["energy_j"] = energy_j
                result["avg_power_w"] = energy_j / wall_s if wall_s > 0 else None
            else:
                result["energy_j"] = None
                result["avg_power_w"] = None
            result["peak_power_w"] = None  # RAPL gives cumulative energy, not an instantaneous peak
        else:
            result["energy_j"] = result["avg_power_w"] = result["peak_power_w"] = None
    elif not is_gpu and cpu_tdp_watts is not None and rss_sampler is not None:
        # RAPL unavailable (typical in a VM/container, and previously confirmed
        # to be the case here) — fall back to the TDP-scaled utilization
        # ESTIMATE. Only runs if the user explicitly opted in via
        # --cpu-tdp-watts, so silence is the default and this never
        # masquerades as a real measurement.
        est = estimate_power_tdp(rss_sampler.cpu_util_samples, cpu_tdp_watts, cpu_idle_fraction)
        result["avg_power_w"] = est
        result["peak_power_w"] = None
        result["energy_j"] = (est * wall_s) if est is not None else None
        result["power_is_estimated"] = est is not None
    else:
        result["energy_j"] = result["avg_power_w"] = result["peak_power_w"] = None

    return result


# ==========================================================================
# Sweep one platform across all sizes, aggregating repeats (mean +/- std)
# ==========================================================================
def sweep(binary, dataset, sizes, repeats, is_gpu, label, gpu_sampling=True,
          cpu_tdp_watts=None, cpu_idle_fraction=0.3):
    rows = []
    for size in sizes:
        trial_results = []
        for rep in range(repeats):
            print(f"[{label}] size={size:,} rep={rep + 1}/{repeats} ...", flush=True)
            r = run_once(binary, dataset, size, is_gpu, gpu_sampling=gpu_sampling,
                         cpu_tdp_watts=cpu_tdp_watts, cpu_idle_fraction=cpu_idle_fraction)
            if r:
                trial_results.append(r)
        if not trial_results:
            print(f"WARNING: no successful runs for [{label}] size={size:,}, skipping.", file=sys.stderr)
            continue
        agg = {"platform": label, "size": size, "repeats": len(trial_results),
               "power_is_estimated": any(t.get("power_is_estimated") for t in trial_results)}
        for key in ["records_per_s", "mb_per_s_padded", "mb_per_s_useful",
                    "peak_mem_mb", "avg_power_w", "peak_power_w", "energy_j", "wall_s"]:
            vals = [t[key] for t in trial_results if t.get(key) is not None]
            if vals:
                agg[f"{key}_mean"] = mean(vals)
                agg[f"{key}_std"] = stdev(vals) if len(vals) > 1 else 0.0
            else:
                agg[f"{key}_mean"] = None
                agg[f"{key}_std"] = None
        rows.append(agg)
    return rows


# ==========================================================================
# Plotting
# ==========================================================================
def make_plots(all_rows, out_throughput, out_memory, out_power):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    platforms = ["CPU (1 thread)", "OpenMP", "GPU (CUDA)"]
    colors = {"CPU (1 thread)": "#2a78d6", "OpenMP": "#eb6834", "GPU (CUDA)": "#1baf7a"}

    def plot_metric(mean_key, std_key, ylabel, out_path, logy=True, title=None):
        fig, ax = plt.subplots(figsize=(7, 5))
        any_data = False
        for p in platforms:
            sub = sorted([r for r in all_rows if r["platform"] == p], key=lambda r: r["size"])
            xs, ys, es = [], [], []
            for r in sub:
                if r.get(mean_key) is not None:
                    xs.append(r["size"])
                    ys.append(r[mean_key])
                    es.append(r.get(std_key) or 0)
            if not xs:
                continue
            any_data = True
            ax.errorbar(xs, ys, yerr=es, marker='o', label=p, color=colors[p], capsize=4)
        if not any_data:
            plt.close(fig)
            print(f"Skipping {out_path}: no data for this metric.", file=sys.stderr)
            return
        ax.set_xscale("log")
        if logy:
            ax.set_yscale("log")
        ax.set_xlabel("Dataset size (records)")
        ax.set_ylabel(ylabel)
        if title:
            ax.set_title(title)
        ax.legend()
        ax.grid(True, which="both", ls="--", alpha=0.3)
        fig.tight_layout()
        fig.savefig(out_path, dpi=150)
        plt.close(fig)
        print(f"Wrote {out_path}")

    plot_metric("records_per_s_mean", "records_per_s_std",
                "Full-pipeline throughput (records/s, log scale)",
                out_throughput, logy=True,
                title="Throughput vs dataset size")
    plot_metric("peak_mem_mb_mean", "peak_mem_mb_std",
                "Peak memory usage (MB)",
                out_memory, logy=False,
                title="Peak memory usage vs dataset size")

    # Power plot: draw estimated series (TDP model) as dashed lines with an
    # "(estimated)" label suffix, and measured series (RAPL/nvidia-smi) as
    # solid lines — so the figure itself makes the distinction obvious rather
    # than relying on the caption alone.
    fig, ax = plt.subplots(figsize=(7, 5))
    any_power_data = False
    for p in platforms:
        sub = sorted([r for r in all_rows if r["platform"] == p], key=lambda r: r["size"])
        xs, ys, es = [], [], []
        estimated = False
        for r in sub:
            if r.get("avg_power_w_mean") is not None:
                xs.append(r["size"])
                ys.append(r["avg_power_w_mean"])
                es.append(r.get("avg_power_w_std") or 0)
                if r.get("power_is_estimated"):
                    estimated = True
        if not xs:
            continue
        any_power_data = True
        style = '-' if estimated else '-'
        suffix = " (estimated, TDP model)" if estimated else " (measured)"
        ax.errorbar(xs, ys, yerr=es, marker='o', linestyle=style,
                     label=p + suffix, color=colors[p], capsize=4)
    if any_power_data:
        ax.set_xscale("log")
        ax.set_xlabel("Dataset size (records)")
        ax.set_ylabel("Average power draw (W)")
        ax.set_title("Average power draw vs dataset size")
        ax.legend(fontsize=8)
        ax.grid(True, which="both", ls="--", alpha=0.3)
        fig.tight_layout()
        fig.savefig(out_power, dpi=150)
        print(f"Wrote {out_power}")
    else:
        print(f"Skipping {out_power}: no power data at all (no RAPL, no --cpu-tdp-watts, "
              f"and/or no GPU power samples).", file=sys.stderr)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", required=True, help="Path to the Intel Berkeley Lab dataset text file")
    ap.add_argument("--cpu-bin", required=True, help="Path to the single-threaded CPU binary")
    ap.add_argument("--openmp-bin", required=True, help="Path to the OpenMP binary")
    ap.add_argument("--gpu-bin", required=True, help="Path to the CUDA binary")
    ap.add_argument("--sizes", type=int, nargs="+",
                     default=[100_000, 500_000, 1_000_000, 2_000_000],
                     help="Record counts to sweep (default: 100K 500K 1M 2M)")
    ap.add_argument("--repeats", type=int, default=5,
                     help="Repeats per (platform, size) for mean/std (default: 5)")
    ap.add_argument("--out-csv", default="sweep_results.csv")
    ap.add_argument("--out-plot-throughput", default="throughput_vs_size.png")
    ap.add_argument("--out-plot-memory", default="memory_vs_size.png")
    ap.add_argument("--out-plot-power", default="power_vs_size.png")
    ap.add_argument("--skip-cpu", action="store_true")
    ap.add_argument("--skip-openmp", action="store_true")
    ap.add_argument("--skip-gpu", action="store_true")
    ap.add_argument("--no-gpu-sampling", action="store_true",
                     help="Disable nvidia-smi polling during GPU runs (diagnostic flag — "
                          "use this to check whether nvidia-smi polling is causing crashes). "
                          "Memory/power for GPU runs will be unavailable when set.")
    ap.add_argument("--cpu-tdp-watts", type=float, default=None,
                     help="Rated TDP (watts) of your CPU, from its spec sheet (e.g. Intel ARK). "
                          "If set AND RAPL is unavailable (typical in a VM/container), CPU/OpenMP "
                          "power is ESTIMATED via a TDP x utilization model instead of measured. "
                          "Omit this flag to leave power as None when RAPL isn't available, rather "
                          "than silently estimating.")
    ap.add_argument("--cpu-idle-fraction", type=float, default=0.3,
                     help="Assumed idle-power floor as a fraction of TDP, used only by the "
                          "--cpu-tdp-watts estimate (default: 0.3, i.e. 30%% of TDP at 0%% utilization)")
    args = ap.parse_args()

    if psutil is None:
        print("WARNING: psutil not installed (pip install psutil) — "
              "CPU/OpenMP memory will be unavailable.", file=sys.stderr)
    if args.cpu_tdp_watts is not None and not _RAPL_AVAILABLE:
        print(f"NOTE: RAPL unavailable, so CPU/OpenMP power will be an ESTIMATE from the "
              f"TDP model (TDP={args.cpu_tdp_watts}W, idle_fraction={args.cpu_idle_fraction}). "
              f"This is NOT a hardware measurement — say so explicitly in the paper.",
              file=sys.stderr)

    all_rows = []
    if not args.skip_cpu:
        all_rows += sweep(args.cpu_bin, args.dataset, args.sizes, args.repeats,
                           is_gpu=False, label="CPU (1 thread)",
                           cpu_tdp_watts=args.cpu_tdp_watts, cpu_idle_fraction=args.cpu_idle_fraction)
    if not args.skip_openmp:
        all_rows += sweep(args.openmp_bin, args.dataset, args.sizes, args.repeats,
                           is_gpu=False, label="OpenMP",
                           cpu_tdp_watts=args.cpu_tdp_watts, cpu_idle_fraction=args.cpu_idle_fraction)
    if not args.skip_gpu:
        all_rows += sweep(args.gpu_bin, args.dataset, args.sizes, args.repeats,
                           is_gpu=True, label="GPU (CUDA)", gpu_sampling=not args.no_gpu_sampling)

    if not all_rows:
        print("No results collected — nothing to write or plot.", file=sys.stderr)
        sys.exit(1)

    keys = sorted(set(k for row in all_rows for k in row.keys()))
    with open(args.out_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(all_rows)
    print(f"Wrote {args.out_csv}")

    make_plots(all_rows, args.out_plot_throughput, args.out_plot_memory, args.out_plot_power)


if __name__ == "__main__":
    main()
