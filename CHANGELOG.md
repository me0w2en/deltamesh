# Changelog

All notable changes to this project are documented in this file.
The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the project uses
[Semantic Versioning](https://semver.org/).

## [0.1.0] - Unreleased

### Added

- Incremental block meshing with a signed-distance field and Surface Nets (`SdfMesher`).
- Exact preview-to-refined segment replacement through per-segment layers.
- Block versions that change only when a block mesh changes.
- Normal estimation and sign propagation for points without reliable normals.
- Optional per-block simplification with locked block boundaries.
- Height-field mesher (`HeightMesher`) as a lightweight alternative.
- `gpu` feature: GPU-resident mesher (`GpuMesher`) and a GPU accumulation path for `SdfMesher`.
- Benchmark CLI (`deltamesh-cli`, not published).
