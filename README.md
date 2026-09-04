# Fully GPU-Resident Hybrid Salsa20–Kyber-512 Encryption Pipeline for IoT Sensor Data

A quantum-resistant encryption pipeline for high-throughput IoT sensor telemetry, combining **CRYSTALS-Kyber-512** (a NIST-standardized, lattice-based post-quantum Key Encapsulation Mechanism) with **Salsa20** (a fast symmetric stream cipher) — with every stage, including SHA3-256 hashing, running **entirely on-device** on the GPU. No key material, plaintext, or intermediate hash ever makes a round trip to the host once a session starts.

CPU (single-threaded), OpenMP (multi-threaded), and CUDA (GPU) implementations of the same pipeline are included side by side, plus a benchmarking harness that sweeps dataset size across all three and produces throughput/memory/power comparison plots.

## Why this exists

Most published GPU-accelerated Kyber work only offloads the arithmetic-heavy lattice operations to the GPU and leaves SHA3 hashing on the CPU — which turns out to dominate runtime (>85% in some baselines), quietly capping the real-world speedup. This project instead keeps the **entire** KEM + symmetric-cipher pipeline GPU-resident, so the throughput numbers reflect what an actual GPU-accelerated sensor gateway would see, not just an isolated microbenchmark of the lattice math.

## Repository contents

| File | Description |
|---|---|
| `sensor_hybrid_crypto_kyber.cu` | GPU (CUDA) implementation — Kyber-512 keygen/encaps/decaps and batched Salsa20 encrypt/decrypt/verify, all as device kernels |
| `sensor_hybrid_crypto_kyber_cpu.c` | Single-threaded CPU baseline (identical algorithmic logic, sequential) |
| `sensor_hybrid_crypto_kyber_openmp.c` | Multi-threaded CPU baseline using OpenMP (`#pragma omp parallel for`) |
| `sweep_and_plot.py` | Benchmark harness: sweeps dataset size across all three binaries, measures throughput/memory/power, writes a CSV and comparison plots |

## Pipeline overview

1. **Dataset preparation** — parses raw sensor readings into fixed 64-byte `PackedRecord` structs (one Salsa20 block per record), each carrying an embedded FNV-1a checksum for later per-record tamper detection.
2. **Kyber-512 session-key establishment** — each parallel session independently runs IND-CPA-secure key generation (NTT-domain matrix sampling over ℤ₃₃₂₉, centered binomial noise), encapsulation, and decapsulation, deriving matching 256-bit shared secrets via SHA3-256 without ever exchanging a symmetric key in the clear.
3. **Batched Salsa20 encryption** — one GPU thread per sensor record, keyed by its session's Kyber-derived secret.
4. **Batched Salsa20 decryption + verification** — decrypts and independently recomputes each record's checksum, giving per-record integrity checking rather than one payload-wide check.
5. **End-to-end orchestration** — a single driver stitches the above together with only two unavoidable host↔device transfers (initial plaintext + CSPRNG seed upload, final ciphertext + verification-flag download), and reports throughput and correctness.

## Requirements

- **GPU build:** CUDA Toolkit (`nvcc`), an NVIDIA GPU (developed/benchmarked on an RTX A4000)
- **CPU / OpenMP builds:** a C compiler with C11 support (`gcc` or `clang`); OpenMP support for the multi-threaded build
- **Benchmark harness:** Python 3, `pip install psutil matplotlib`
- **Dataset:** a sensor log in the [Intel Berkeley Research Lab](https://db.csail.mit.edu/labdata/labdata.html) text format (`date time epoch moteid temperature humidity light voltage`)

## Building

```bash
# GPU (CUDA)
nvcc -O3 -arch=sm_86 sensor_hybrid_crypto_kyber.cu -o sensor_hybrid_crypto_kyber
# adjust -arch to your GPU's compute capability (e.g. sm_75 for Turing, sm_89 for Ada)

# CPU (single-threaded baseline)
gcc -O3 sensor_hybrid_crypto_kyber_cpu.c -o sensor_hybrid_crypto_cpu -lm

# CPU (OpenMP, multi-threaded)
gcc -O3 -fopenmp sensor_hybrid_crypto_kyber_openmp.c -o sensor_hybrid_crypto_openmp -lm
```

## Running a single binary

All three binaries share the same CLI:

```bash
./sensor_hybrid_crypto_kyber   <path_to_intel_lab_data.txt> [max_records]
./sensor_hybrid_crypto_cpu     <path_to_intel_lab_data.txt> [max_records]
./sensor_hybrid_crypto_openmp  <path_to_intel_lab_data.txt> [max_records]
```

`max_records` is an optional cap on how many rows to process (defaults to 2,000,000). Each run prints per-stage timing, full-pipeline throughput (records/s and MB/s), and a verification summary (`N / N records passed`).

## Running the full benchmark sweep

```bash
pip install psutil matplotlib

python3 sweep_and_plot.py \
    --dataset /path/to/data.txt \
    --cpu-bin ./sensor_hybrid_crypto_cpu \
    --openmp-bin ./sensor_hybrid_crypto_openmp \
    --gpu-bin ./sensor_hybrid_crypto_kyber \
    --sizes 100000 500000 1000000 2000000 \
    --repeats 5
```

This produces:
- `sweep_results.csv` — raw and aggregated (mean ± std) throughput, memory, and power for every (platform, size, repeat)
- `throughput_vs_size.png`, `memory_vs_size.png`, `power_vs_size.png` — comparison plots across all three platforms

**Power measurement caveats** (see the script's own docstring for full detail): CPU/OpenMP power uses Intel RAPL package-energy counters (`/sys/class/powercap/intel-rapl:0/energy_uj`) — Intel-only, and reports `None` without read permission (fixable with `sudo chmod -R a+r /sys/class/powercap/intel-rapl*`, or pass `--cpu-tdp-watts` for a clearly-labeled TDP-based estimate instead). GPU power uses `nvidia-smi --query-gpu=power.draw`, which is whole-GPU, not process-attributed — benchmark on an otherwise-idle GPU for meaningful numbers.

## Security scope

This implementation demonstrates architectural throughput, not a hardened production KEM. In particular:

- The KEM as implemented derives the shared secret as `SHA3-256(m || ciphertext)` **without** the implicit-rejection re-encryption check from the full Fujisaki–Okamoto transform. This is what makes decapsulation faster than encapsulation here — the inverse of the pattern in comparable baselines — and it means the current implementation is **not CCA2-secure** as-is. A CCA2-hardened build would add a PKE re-encryption step to decapsulation, at a corresponding throughput cost.
- Kyber keygen/encapsulation seed material is generated once on the host via a CSPRNG (`std::random_device` over `/dev/urandom`) and uploaded to the device — the only deliberate exception to the pipeline's otherwise strict no-host-transfer design, since this seed material isn't secret and is discarded after use, unlike the session keys and Kyber keypairs, which never leave the device.

## Citing

If you use this code, please cite the accompanying paper (details to be added on publication).
