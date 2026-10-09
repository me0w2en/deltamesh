//! Benchmark CLI for the deltamesh incremental mesher.
//!
//! The CLI reads a folder of point-cloud segments, feeds each segment first as a preview and then as its
//! refined replacement, and extracts the changed blocks after every step. For each step it appends one row
//! to `bench.csv` with timings, block counts, memory use and an optional distance check, and it can write
//! the merged mesh as PLY and the updated blocks as SLMB files.

mod far;
mod ply;
#[cfg(not(target_arch = "wasm32"))]
#[global_allocator]
static GLOBAL: mimalloc::MiMalloc = mimalloc::MiMalloc;

use anyhow::{Context, Result, bail};
use clap::{Parser, ValueEnum};
use deltamesh::{
    BlockMesh, Config, Level, Mesher, Point, SegmentId, height::HeightMesher, orient::Orient,
    sdf::SdfMesher,
};
use rayon::prelude::*;
use rustc_hash::FxHashMap;
use std::io::Write;
use std::path::PathBuf;
use std::time::Instant;

/// Input folder layout.
#[derive(Clone, Copy, Debug, ValueEnum, PartialEq)]
enum Layout {
    /// A folder of per-segment pairs `r{k}_preview_new.ply` and `r{k}_refined.ply`.
    Segments,
    /// A folder of `tc_*.ply` chunk files. The preview of each chunk is synthesized from the same file by
    /// keeping every 4th point and dropping the normals.
    Chunks,
}

/// Order in which preview (P) and refined (R) segments are ingested.
#[derive(Clone, Copy, Debug, ValueEnum, PartialEq)]
enum Order {
    /// P0, R0, P1, R1, ...: each segment's preview is ingested, then replaced by its refined version.
    Pr,
    /// Arrival order of a live capture: P0, (R0+P1), (R1+P2), ..., R_last.
    Realistic,
}

/// Mesher implementation.
#[derive(Clone, Copy, Debug, ValueEnum, PartialEq)]
enum Kind {
    /// Signed-distance field with Surface Nets extraction.
    Sdf,
    /// 2.5D height field.
    Height,
}

/// Sign rule for estimated preview normals.
#[derive(Clone, Copy, Debug, ValueEnum, PartialEq)]
enum OrientKind {
    /// Propagate orientation between neighbouring bins.
    Prop,
    /// Flip every normal towards +z.
    Up,
}

/// Command-line arguments.
#[derive(Parser, Debug)]
#[command(about = "Incremental meshing benchmark for segmented point clouds")]
struct Args {
    /// Layout of the input folder.
    #[arg(long, value_enum)]
    layout: Layout,
    /// Input folder.
    #[arg(long)]
    data: PathBuf,
    /// Output folder for bench.csv, run.txt and the merged PLY files.
    #[arg(long)]
    out: PathBuf,
    /// Mesher implementation.
    #[arg(long, value_enum, default_value = "sdf")]
    mesher: Kind,
    /// Ingest order of preview and refined segments.
    #[arg(long, value_enum, default_value = "pr")]
    order: Order,
    /// Voxel size in metres.
    #[arg(long, default_value_t = 0.2)]
    voxel: f32,
    /// Block edge length in voxels.
    #[arg(long, default_value_t = 32)]
    block_dim: i32,
    /// Point binning cell size in metres.
    #[arg(long, default_value_t = 0.1)]
    bin: f32,
    /// Splat radius in metres.
    #[arg(long, default_value_t = 0.4)]
    splat_radius: f32,
    /// Minimum accumulated weight for a voxel to count as observed.
    #[arg(long, default_value_t = 0.5)]
    min_weight: f32,
    /// Block mesh simplification tolerance in metres; 0 disables simplification (SDF mesher only).
    #[arg(long, default_value_t = 0.0)]
    simplify_error: f32,
    /// Worker thread count; 0 uses the rayon default (number of cores).
    #[arg(long, default_value_t = 0)]
    threads: usize,
    /// Use only the first N segments (0 = all).
    #[arg(long, default_value_t = 0)]
    limit: usize,
    /// Do not write merged PLY files.
    #[arg(long)]
    no_ply: bool,
    /// Write the merged PLY for the final step only (useful for parameter sweeps).
    #[arg(long)]
    ply_final_only: bool,
    /// Report the fraction of mesh vertices farther than this distance (m) from any input point.
    /// Not included in the step timings; 0 disables the check.
    #[arg(long, default_value_t = 0.4)]
    check_dist: f32,
    /// Write the blocks updated in each step as SLMB binary files.
    #[arg(long)]
    emit_blocks: bool,
    /// Sign rule for estimated preview normals: prop (neighbour propagation) or up (align with +z).
    #[arg(long, value_enum, default_value = "prop")]
    orient: OrientKind,
    /// Do not use the existing refined distance field to orient normals first.
    #[arg(long)]
    no_oracle: bool,
    /// Value written to the `data` column of the CSV.
    #[arg(long, default_value = "")]
    tag: String,
    /// Accumulation device: auto = a suitable GPU if present, else CPU; on = require a GPU; off = CPU only.
    /// Software rasterizers such as llvmpipe are never chosen. Set DELTAMESH_GPU_ADAPTER to pick an
    /// adapter by name.
    #[arg(long, value_enum, default_value = "auto")]
    gpu: GpuMode,
    /// GPU path: resident = keep the field on the GPU and run binning, normals, accumulation and extraction
    /// there; splat = accumulate on the GPU and merge on the CPU.
    #[arg(long, value_enum, default_value = "resident")]
    gpu_path: GpuPath,
    /// Print the detected GPU adapters with their selection scores and exit.
    #[arg(long)]
    list_gpus: bool,
}

/// GPU execution path used by the SDF mesher.
#[derive(Clone, Copy, Debug, ValueEnum, PartialEq)]
enum GpuPath {
    Resident,
    Splat,
}

/// GPU selection policy.
#[derive(Clone, Copy, Debug, ValueEnum, PartialEq)]
enum GpuMode {
    Auto,
    On,
    Off,
}

/// Selects the accumulation device according to `mode`.
///
/// # Returns
///
/// The GPU handle (if one is used) and a human-readable backend description for the logs.
///
/// # Errors
///
/// Fails when `mode` is [`GpuMode::On`] and no GPU is usable, or the binary was built without the `gpu`
/// feature.
fn pick_gpu(mode: GpuMode) -> Result<(Option<std::sync::Arc<deltamesh::sdf::Gpu>>, String)> {
    if mode == GpuMode::Off {
        return Ok((None, "cpu (--gpu off)".into()));
    }
    #[cfg(feature = "gpu")]
    {
        match deltamesh::gpu::GpuSplat::new() {
            Ok(g) => {
                let d = format!("gpu: {}", g.info);
                Ok((Some(std::sync::Arc::new(g)), d))
            }
            Err(e) if mode == GpuMode::Auto => Ok((None, format!("cpu (no GPU: {e})"))),
            Err(e) => bail!("--gpu on was given but no GPU is usable: {e}"),
        }
    }
    #[cfg(not(feature = "gpu"))]
    {
        if mode == GpuMode::On {
            bail!("--gpu on was given but this binary was built without the gpu feature");
        }
        Ok((None, "cpu (built without gpu feature)".into()))
    }
}

/// Where the points of one ingest come from.
#[derive(Clone)]
enum Source {
    /// All points of a PLY file.
    File(PathBuf),
    /// Synthetic preview: every `stride`-th point of a PLY file, with normals removed.
    Synth(PathBuf, usize),
}

/// One segment ingest within a step.
struct Ingest {
    seg: SegmentId,
    level: Level,
    src: Source,
}

/// A group of ingests followed by one extraction; produces one CSV row.
struct Step {
    label: String,
    ingests: Vec<Ingest>,
}

/// Builds the list of steps from the input layout and ingest order.
///
/// # Errors
///
/// Fails when the folder cannot be read, contains no segments, or a refined segment has no preview.
fn plan(a: &Args) -> Result<Vec<Step>> {
    let mut segs: Vec<(Source, Source)> = Vec::new();
    match a.layout {
        Layout::Segments => {
            for k in 0.. {
                let r = a.data.join(format!("r{k}_refined.ply"));
                let p = a.data.join(format!("r{k}_preview_new.ply"));
                if !r.exists() {
                    break;
                }
                if !p.exists() {
                    bail!("missing {}", p.display());
                }
                segs.push((Source::File(p), Source::File(r)));
            }
        }
        Layout::Chunks => {
            let mut files: Vec<PathBuf> = std::fs::read_dir(&a.data)?
                .filter_map(|e| e.ok().map(|e| e.path()))
                .filter(|p| {
                    let n = p.file_name().unwrap().to_string_lossy();
                    n.starts_with("tc_") && n.ends_with(".ply")
                })
                .collect();
            files.sort();
            for f in files {
                segs.push((Source::Synth(f.clone(), 4), Source::File(f)));
            }
        }
    }
    if segs.is_empty() {
        bail!("no input segments in {}", a.data.display());
    }
    if a.limit > 0 {
        segs.truncate(a.limit);
    }
    let mut steps = Vec::new();
    let n = segs.len();
    match a.order {
        Order::Pr => {
            for (k, (p, r)) in segs.into_iter().enumerate() {
                let k = k as u32;
                steps.push(Step {
                    label: format!("r{k}_after_preview"),
                    ingests: vec![Ingest {
                        seg: k,
                        level: Level::Preview,
                        src: p,
                    }],
                });
                steps.push(Step {
                    label: format!("r{k}_after_refined"),
                    ingests: vec![Ingest {
                        seg: k,
                        level: Level::Refined,
                        src: r,
                    }],
                });
            }
        }
        Order::Realistic => {
            let segs: Vec<_> = segs.into_iter().collect();
            steps.push(Step {
                label: "step_01".into(),
                ingests: vec![Ingest {
                    seg: 0,
                    level: Level::Preview,
                    src: segs[0].0.clone(),
                }],
            });
            for k in 1..=n {
                let mut ing = vec![Ingest {
                    seg: k as u32 - 1,
                    level: Level::Refined,
                    src: segs[k - 1].1.clone(),
                }];
                if k < n {
                    ing.push(Ingest {
                        seg: k as u32,
                        level: Level::Preview,
                        src: segs[k].0.clone(),
                    });
                }
                steps.push(Step {
                    label: if k < n {
                        format!("step_{:02}", k + 1)
                    } else {
                        "step_final".into()
                    },
                    ingests: ing,
                });
            }
        }
    }
    Ok(steps)
}

/// Loads the points of one source.
///
/// For [`Source::Synth`] the skipped points are never decoded; the result equals
/// `read_points(p)?.into_iter().step_by(s)` with all normals set to NaN.
fn load(src: &Source) -> Result<Vec<Point>> {
    match src {
        Source::File(p) => ply::read_points(p),
        Source::Synth(p, s) => {
            let mut v = ply::read_points_stride(p, *s)?;
            v.par_iter_mut()
                .with_min_len(4096)
                .for_each(|q| q.normal = [f32::NAN; 3]);
            Ok(v)
        }
    }
}

/// Current resident set size in bytes; 0 on platforms other than macOS.
fn rss_now() -> u64 {
    #[cfg(target_os = "macos")]
    // SAFETY: `info` is a zeroed plain C struct and `sz` is its exact size, as `proc_pidinfo` requires.
    unsafe {
        let mut info: libc::proc_taskinfo = std::mem::zeroed();
        let sz = std::mem::size_of::<libc::proc_taskinfo>() as i32;
        if libc::proc_pidinfo(
            libc::getpid(),
            libc::PROC_PIDTASKINFO,
            0,
            &mut info as *mut _ as *mut libc::c_void,
            sz,
        ) == sz
        {
            return info.pti_resident_size;
        }
    }
    0
}

/// Process CPU time (user + sys) in milliseconds.
///
/// Serves as a secondary metric next to wall-clock time on machines with other load.
fn cpu_ms() -> f64 {
    // SAFETY: `getrusage` fills the zeroed plain C struct passed by pointer.
    unsafe {
        let mut ru: libc::rusage = std::mem::zeroed();
        libc::getrusage(libc::RUSAGE_SELF, &mut ru);
        let t = |v: libc::timeval| v.tv_sec as f64 * 1e3 + v.tv_usec as f64 / 1e3;
        t(ru.ru_utime) + t(ru.ru_stime)
    }
}

/// Peak resident set size in bytes.
///
/// `ru_maxrss` is reported in bytes on macOS and in kilobytes on Linux.
fn rss_peak() -> u64 {
    // SAFETY: `getrusage` fills the zeroed plain C struct passed by pointer.
    unsafe {
        let mut ru: libc::rusage = std::mem::zeroed();
        libc::getrusage(libc::RUSAGE_SELF, &mut ru);
        if cfg!(target_os = "macos") {
            ru.ru_maxrss as u64
        } else {
            ru.ru_maxrss as u64 * 1024
        }
    }
}

/// CPU model string reported by `sysctl` at runtime, with commas removed for CSV.
///
/// Empty when `sysctl` is unavailable.
fn machine() -> String {
    let brand = std::process::Command::new("sysctl")
        .args(["-n", "machdep.cpu.brand_string"])
        .output()
        .map(|o| String::from_utf8_lossy(&o.stdout).trim().to_string())
        .unwrap_or_default();
    brand.replace(',', " ")
}

/// Runs the benchmark.
///
/// Distance check: for every segment the current input points are kept in a [`far::SegGrid`], built once
/// when the segment arrives and replaced when its refined version arrives. Points with non-finite
/// coordinates cannot lie within any distance of a vertex, so they are left out of the grid.
///
/// Block output: deleted blocks are still written, as empty meshes carrying the current version, so a
/// client that applies only newer versions also removes them.
fn main() -> Result<()> {
    let a = Args::parse();
    if a.list_gpus {
        #[cfg(feature = "gpu")]
        for (i, s) in deltamesh::gpu::list_adapters() {
            println!(
                "{} {i}",
                s.map(|v| format!("[score {v:>3}]"))
                    .unwrap_or_else(|| "[excluded ]".into())
            );
        }
        #[cfg(not(feature = "gpu"))]
        println!("built without the gpu feature");
        return Ok(());
    }
    if a.threads > 0 {
        rayon::ThreadPoolBuilder::new()
            .num_threads(a.threads)
            .build_global()?;
    }
    let threads = rayon::current_num_threads();
    let cfg = Config {
        voxel: a.voxel,
        block_dim: a.block_dim,
        bin: a.bin,
        splat_radius: a.splat_radius,
        min_weight: a.min_weight,
        simplify_error: a.simplify_error,
        orient: match a.orient {
            OrientKind::Prop => Orient::default(),
            OrientKind::Up => Orient::Up,
        },
        orient_oracle: !a.no_oracle,
        ..Config::default()
    };
    let steps = plan(&a)?;
    std::fs::create_dir_all(&a.out)?;
    let ply_dir = a.out.join("ply");
    if !a.no_ply {
        std::fs::create_dir_all(&ply_dir)?;
    }
    let (gpu, mut backend) = if a.mesher == Kind::Sdf {
        pick_gpu(a.gpu)?
    } else {
        (None, "cpu (height field)".into())
    };
    #[cfg(feature = "gpu")]
    let resident = gpu.is_some() && a.gpu_path == GpuPath::Resident;
    #[cfg(not(feature = "gpu"))]
    let resident = false;
    if gpu.is_some() {
        backend = format!(
            "{backend} [{}]",
            if resident { "resident" } else { "splat" }
        );
    }
    eprintln!("accumulation backend: {backend}");
    let mut mesher: Box<dyn Mesher> = match a.mesher {
        #[cfg(feature = "gpu")]
        Kind::Sdf if resident => {
            let ctx = gpu.as_ref().unwrap().ctx.clone();
            Box::new(
                deltamesh::gpu_mesher::GpuMesher::new(ctx, cfg).map_err(|e| {
                    anyhow::anyhow!("failed to initialize the GPU-resident mesher: {e}")
                })?,
            )
        }
        Kind::Sdf => {
            let mut m = SdfMesher::new(cfg);
            m.set_gpu(gpu);
            Box::new(m)
        }
        Kind::Height => Box::new(HeightMesher::new(cfg)),
    };
    let mach = machine();
    {
        let mut f = std::fs::File::create(a.out.join("run.txt"))?;
        writeln!(f, "args: {:?}", a)?;
        writeln!(f, "config: {:?}", cfg)?;
        writeln!(f, "machine: {mach}, rayon threads {threads}")?;
        writeln!(f, "backend: {backend}")?;
    }
    let mut csv = std::fs::File::create(a.out.join("bench.csv"))?;
    writeln!(
        csv,
        "data,mesher,voxel_m,threads,machine,step,label,segments,levels,input_points,bins,normals_estimated,bins_dropped,\
         replaced_preview,updated_blocks,ingest_ms,extract_ms,total_ms,load_ms,nonempty_blocks,total_tris,welded_verts,\
         field_mb,mesh_mb,rss_mb,peak_rss_mb,far_frac,weld_ms,far_ms,ply_ms,\
         reextracted_blocks,reextracted_direct,unchanged_direct,unchanged_neighbor,unchanged_empty,\
         ph_binning_ms,ph_normals_ms,ph_orient_ms,ph_drop_ms,ph_lists_ms,ph_splat_ms,ph_dirty_ms,cpu_ms,backend,\
         ph_gpu_prep_ms,ph_gpu_upload_ms,ph_gpu_wait_ms,ph_gpu_merge_ms,ph_gather_ms,ph_extract_gpu_ms,ph_simplify_ms"
    )?;
    let mut effective: FxHashMap<SegmentId, far::SegGrid> = FxHashMap::default();

    for (si, st) in steps.iter().enumerate() {
        let t_load = Instant::now();
        let batches: Vec<Vec<Point>> = st
            .ingests
            .par_iter()
            .map(|i| load(&i.src))
            .collect::<Result<_>>()?;
        let load_ms = t_load.elapsed().as_secs_f64() * 1e3;

        let _ = deltamesh::sdf::take_phase_ms();
        let c0 = cpu_ms();
        let t0 = Instant::now();
        let mut stats = Vec::new();
        for (ing, pts) in st.ingests.iter().zip(&batches) {
            stats.push(
                mesher
                    .ingest(ing.seg, ing.level, pts)
                    .with_context(|| st.label.clone())?,
            );
        }
        let t1 = Instant::now();
        let updated = mesher.extract();
        let ph = deltamesh::sdf::take_phase_ms();
        let t2 = Instant::now();
        let step_cpu_ms = cpu_ms() - c0;
        let xs = mesher.extract_stats();

        let ingest_ms = (t1 - t0).as_secs_f64() * 1e3;
        let extract_ms = (t2 - t1).as_secs_f64() * 1e3;

        let t_far0 = Instant::now();
        if a.check_dist > 0.0 {
            for (ing, pts) in st.ingests.iter().zip(&batches) {
                let finite = |p: &Point| p.pos.iter().all(|v| v.is_finite());
                let grid = if pts.iter().all(finite) {
                    far::SegGrid::from_points(pts, |p| p.pos, a.check_dist)
                } else {
                    let ok: Vec<Point> = pts.iter().filter(|p| finite(p)).copied().collect();
                    far::SegGrid::from_points(&ok, |p| p.pos, a.check_dist)
                };
                effective.insert(ing.seg, grid);
            }
        }
        let t_far0 = t_far0.elapsed();
        let mut ms: Vec<&BlockMesh> = mesher.meshes().collect();
        ms.sort_by_key(|m| m.id);
        let t_weld = Instant::now();
        let welded = ply::weld(&ms);
        let verts = &welded.0;
        let weld_ms = t_weld.elapsed().as_secs_f64() * 1e3;
        let t_far1 = Instant::now();
        let far = if a.check_dist > 0.0 {
            let grids: Vec<&far::SegGrid> = effective.values().filter(|g| g.len() > 0).collect();
            format!("{:.6}", far::far_fraction(verts, &grids, a.check_dist))
        } else {
            String::new()
        };
        let far_ms = (t_far0 + t_far1.elapsed()).as_secs_f64() * 1e3;
        let t_ply = Instant::now();
        if !a.no_ply && (!a.ply_final_only || si + 1 == steps.len()) {
            ply::write_welded(&ply_dir.join(format!("{}.ply", st.label)), &welded)?;
        }
        let ply_ms = t_ply.elapsed().as_secs_f64() * 1e3;
        if a.emit_blocks {
            let d = a.out.join("blocks").join(&st.label);
            std::fs::create_dir_all(&d)?;
            for id in &updated {
                let empty = BlockMesh {
                    id: *id,
                    version: mesher.version(id),
                    ..Default::default()
                };
                let m = mesher.mesh(id).unwrap_or(&empty);
                ply::write_block(&d.join(format!("{}_{}_{}.slmb", id[0], id[1], id[2])), m)?;
            }
        }
        let sum = |f: fn(&deltamesh::IngestStats) -> usize| stats.iter().map(f).sum::<usize>();
        let segs: Vec<String> = st.ingests.iter().map(|i| i.seg.to_string()).collect();
        let levels: Vec<&str> = st
            .ingests
            .iter()
            .map(|i| if i.level == Level::Preview { "P" } else { "R" })
            .collect();
        let mb = |b: usize| b as f64 / 1048576.0;
        let line = format!(
            "{},{:?},{},{},{},{},{},{},{},{},{},{},{},{},{},{:.1},{:.1},{:.1},{:.1},{},{},{},{:.1},{:.1},{:.1},{:.1},{},{:.1},{:.1},{:.1},{},{},{},{},{},{:.2},{:.2},{:.2},{:.2},{:.2},{:.2},{:.2},{:.1},{},{:.2},{:.2},{:.2},{:.2},{:.2},{:.2},{:.2}",
            a.tag,
            a.mesher,
            a.voxel,
            threads,
            mach,
            si,
            st.label,
            segs.join("+"),
            levels.join("+"),
            sum(|s| s.input_points),
            sum(|s| s.bins),
            sum(|s| s.normals_estimated),
            sum(|s| s.bins_dropped),
            stats.iter().any(|s| s.removed_preview) as u8,
            updated.len(),
            ingest_ms,
            extract_ms,
            ingest_ms + extract_ms,
            load_ms,
            mesher.block_count(),
            mesher.total_tris(),
            verts.len(),
            mb(mesher.field_bytes()),
            mb(mesher.mesh_bytes()),
            mb(rss_now() as usize),
            mb(rss_peak() as usize),
            far,
            weld_ms,
            far_ms,
            ply_ms,
            xs.reextracted,
            xs.reextracted_direct,
            xs.unchanged_direct,
            xs.unchanged_neighbor,
            xs.unchanged_empty,
            ph.binning,
            ph.normals,
            ph.orient,
            ph.drop_layer,
            ph.lists,
            ph.splat,
            ph.dirty,
            step_cpu_ms,
            backend.replace(',', ";"),
            ph.gpu_prep,
            ph.gpu_upload,
            ph.gpu_wait,
            ph.gpu_merge,
            ph.gather,
            ph.extract_gpu,
            ph.simplify
        );
        writeln!(csv, "{line}")?;
        eprintln!(
            "{:<20} pts {:>8} blocks {:>4}/{:<4} {:>8.1} ms  tris {:>9}  field {:>7.1}MB far {}",
            st.label,
            sum(|s| s.input_points),
            updated.len(),
            xs.reextracted,
            ingest_ms + extract_ms,
            mesher.total_tris(),
            mb(mesher.field_bytes()),
            far
        );
    }
    Ok(())
}
