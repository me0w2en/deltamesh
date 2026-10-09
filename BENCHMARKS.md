# Benchmarks

Measurements taken with the `deltamesh` CLI in this repository. The input data sets are not public.

## Setup

**Data sets** (aerial photogrammetry point clouds, full resolution)

| Name | Content | Steps |
|---|---|---|
| A | 7 segments. Each segment arrives as a preview point cloud and is then replaced by its refined point cloud. 14.8 M refined points in total. | 14 (7 previews + 7 refined) |
| B | 20 chunks, 8.3 M points in total. A synthetic preview (every 4th point, no normals) is ingested before each chunk. | 40; medians below count the 20 chunk steps |

A **step** is one call to `ingest` followed by `extract`. Step time excludes file reading and writing.
Each configuration was run twice in alternating order and the median per step was taken.

**Machines**

| Name | CPU | GPU |
|---|---|---|
| Laptop | Apple M5, 10 cores, 16 GB | Apple M5 integrated (Metal) |
| Server | Intel Xeon Gold 6140, 16 vCPU (virtual machine) | NVIDIA Tesla V100-PCIE-16GB (Vulkan, driver 550) |

**Detail levels** (`--voxel` / `--bin` / `--splat-radius` / `--simplify-error`, metres)

| Level | Voxel | Bin | Splat radius | Simplify error | Final triangles (A / B) |
|---|---|---|---|---|---|
| L1 | 0.20 | 0.10 | 0.40 | 0.20 | 79 k / 102 k |
| L2 | 0.20 | 0.10 | 0.40 | 0.10 | 112 k / 148 k |
| L3 | 0.20 | 0.10 | 0.40 | 0.05 | 172 k / 235 k |
| L4 (default) | 0.20 | 0.10 | 0.40 | 0 | 661 k / 801 k |
| L5 | 0.10 | 0.05 | 0.15 | 0.02 | 687 k / 1.04 M |
| L6 | 0.10 | 0.05 | 0.15 | 0 | 2.13 M / 2.55 M |
| L7 | 0.07 | 0.035 | 0.12 | 0 | 5.22 M / 6.21 M |
| L8 | 0.05 | 0.025 | 0.10 | 0 | 11.8 M / 14.3 M |

**Execution modes**

| Mode | Options |
|---|---|
| CPU, 1 thread | `--gpu off --threads 1` |
| CPU, all cores | `--gpu off` |
| GPU, resident (default) | `--gpu on --gpu-path resident` |
| GPU, splat | `--gpu on --gpu-path splat` (accumulation only on the GPU) |

## Median step time (ms)

### Laptop (Apple M5)

| Level | A: CPU 1 | A: CPU 10 | A: GPU resident | A: GPU splat | B: CPU 1 | B: CPU 10 | B: GPU resident | B: GPU splat |
|---|---|---|---|---|---|---|---|---|
| L1 | 212 | 44 | 30 | 56 | 146 | 30 | 23 | 41 |
| L2 | 209 | 46 | 34 | 53 | 145 | 30 | 23 | 41 |
| L3 | 208 | 46 | 31 | 57 | 143 | 30 | 23 | 45 |
| L4 | 165 | 37 | 22 | 47 | 99 | 22 | 13 | 30 |
| L5 | 410 | 82 | 61 | 115 | 301 | 57 | 52 | 82 |
| L6 | 305 | 64 | 38 | 95 | 155 | 32 | 23 | 58 |
| L7 | 397 | 80 | 56 | 111 | 254 | 47 | 36 | 83 |
| L8 | 352 | 73 | 56 | 108 | 451 | 83 | 56 | 136 |

### Server (Xeon Gold 6140 + Tesla V100)

| Level | A: CPU 1 | A: CPU 16 | A: GPU resident | A: GPU splat | B: CPU 1 | B: CPU 16 | B: GPU resident | B: GPU splat |
|---|---|---|---|---|---|---|---|---|
| L1 | 740 | 97 | 58 | 83 | 512 | 67 | 36 | 56 |
| L2 | 687 | 100 | 62 | 82 | 507 | 68 | 37 | 56 |
| L3 | 687 | 97 | 58 | 82 | 516 | 68 | 37 | 56 |
| L4 | 559 | 83 | 41 | 70 | 356 | 54 | 21 | 45 |
| L5 | 1,192 | 170 | 106 | 163 | 969 | 112 | 73 | 119 |
| L6 | 965 | 143 | 67 | 136 | 509 | 72 | 31 | 82 |
| L7 | 1,131 | 166 | 83 | 211 | 838 | 96 | 51 | 146 |
| L8 | 1,083 | 171 | 101 | 215 | 1,486 | 158 | 90 | 260 |

- The resident GPU path is faster than all CPU cores at every level: 0.57 to 0.92 of the CPU time on the laptop and
  0.39 to 0.65 on the server.
- The slowest single step on the server at L8 was 263 ms (A) and 128 ms (B) with the resident GPU path.
- On the server the GPU kernels take about as long as on the laptop; the difference comes from the work that stays on
  the CPU (normal orientation, block lists, simplification), which runs 1.8 to 4.2 times slower on the server CPU.

## Compared with full re-meshing

The baseline rebuilds the whole mesh from all points received so far after every step: voxel down-sampling, outlier
removal, screened Poisson reconstruction (Open3D, single thread), trimming of faces far from the points, and quadric
decimation to the same triangle count as deltamesh. Poisson depth and down-sampling follow the level's voxel size
(depth 10 for L1 to L4, 11 for L5 to L7, 12 for L8). Open3D's Poisson crashed on this data when run multi-threaded,
so both sides are compared on one thread. Laptop only.

Step time after the last refined segment of data set A:

| Level | Full re-mesh (1 thread) | deltamesh (1 thread) | Speed-up | deltamesh (10 cores) | deltamesh (GPU) |
|---|---|---|---|---|---|
| L1 | 23.0 s | 83 ms | 279× | 18 ms | 16 ms |
| L4 | 19.6 s | 53 ms | 371× | 13 ms | 9 ms |
| L6 | 94.8 s | 101 ms | 937× | 24 ms | 22 ms |
| L7 | 78.3 s | 168 ms | 467× | 36 ms | 31 ms |
| L8 | > 276 s (reconstruction only) | 277 ms | > 999× | 55 ms | 42 ms |

Across both data sets, levels L1 to L7 and two points in time (half-way and final), the speed-up on one thread is
83× to 2,464×. At L8 the baseline's decimation did not fit in 16 GB of memory, so only its reconstruction time is
given and the speed-up is a lower bound.

### Quality

F-score against a reference point cloud (precision and recall of mesh surface samples within 10 cm, F10, and 25 cm,
F25), data set A, final mesh:

| Level | F10 full re-mesh | F10 deltamesh | F25 full re-mesh | F25 deltamesh |
|---|---|---|---|---|
| L1 | 0.773 | 0.699 | 0.973 | 0.979 |
| L2 | 0.773 | 0.759 | 0.971 | 0.982 |
| L3 | 0.773 | 0.780 | 0.969 | 0.983 |
| L4 | 0.772 | 0.786 | 0.966 | 0.983 |
| L5 | 0.803 | 0.891 | 0.972 | 0.999 |
| L6 | 0.803 | 0.891 | 0.972 | 0.999 |
| L7 | 0.799 | 0.901 | 0.968 | 1.000 |
| L8 | n/a | 0.912 | n/a | 1.000 |

With heavy simplification (L1, L2) the smooth Poisson surface scores higher at 10 cm. From L3 on deltamesh matches or
exceeds it, and at finer levels its accuracy keeps improving while the baseline levels off.
The GPU path's F-scores are within 0.003 of the CPU path's.
