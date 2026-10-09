//! Reading point PLY files and writing mesh PLY and SLMB block files.
//!
//! Input files must be binary little-endian with a single `vertex` element whose properties are exactly
//! `x y z` (float), `red green blue` (uchar), `nx ny nz` (float), i.e. 27 bytes per point. Both `float` and
//! `float32`, and both `uchar` and `uint8`, are accepted as type names. Any other layout is rejected.
//! Files are memory-mapped and decoded in parallel.

use anyhow::{Context, Result, bail};
use deltamesh::{BlockMesh, Point};
use rayon::prelude::*;
use std::io::{BufWriter, Write};
use std::path::Path;

/// Required vertex properties in order, with the accepted type names for each.
const PROPS: [(&str, &[&str]); 9] = [
    ("x", &["float", "float32"]),
    ("y", &["float", "float32"]),
    ("z", &["float", "float32"]),
    ("red", &["uchar", "uint8"]),
    ("green", &["uchar", "uint8"]),
    ("blue", &["uchar", "uint8"]),
    ("nx", &["float", "float32"]),
    ("ny", &["float", "float32"]),
    ("nz", &["float", "float32"]),
];
/// Bytes per input point record.
const REC: usize = 27;

/// Validates the PLY header.
///
/// # Returns
///
/// The byte offset where the body starts and the number of points.
///
/// # Errors
///
/// Fails when the header is malformed, does not match the expected layout, or the body is shorter than
/// the declared point count requires.
fn parse_header(map: &[u8]) -> Result<(usize, usize)> {
    let end = map
        .windows(11)
        .position(|w| w == b"end_header\n")
        .context("missing end_header")?
        + 11;
    let header = std::str::from_utf8(&map[..end])?;
    let mut lines = header.lines();
    if lines.next() != Some("ply") {
        bail!("not a PLY file");
    }
    let mut n = None;
    let mut props = Vec::new();
    for l in lines {
        let t: Vec<&str> = l.split_whitespace().collect();
        match t.as_slice() {
            ["format", "binary_little_endian", _] => {}
            ["format", ..] => bail!("only binary_little_endian is supported: {l}"),
            ["element", "vertex", c] => n = Some(c.parse::<usize>()?),
            ["element", e, _] => bail!("unsupported element (only vertex is allowed): {e}"),
            ["property", ty, name] => props.push((ty.to_string(), name.to_string())),
            ["comment", ..] | ["obj_info", ..] | ["end_header"] => {}
            _ => bail!("unknown header line: {l}"),
        }
    }
    let n = n.context("missing vertex count")?;
    if props.len() != PROPS.len()
        || props
            .iter()
            .zip(PROPS)
            .any(|((ty, name), (pn, tys))| name != pn || !tys.contains(&ty.as_str()))
    {
        bail!(
            "properties must be x y z float, red green blue uchar, nx ny nz float; got {props:?}"
        );
    }
    let need = n.checked_mul(REC).context("vertex count too large")?;
    if map.len() - end < need {
        bail!("body too short: {} < {}", map.len() - end, need);
    }
    Ok((end, n))
}

/// Decodes one 27-byte record.
///
/// Taking a fixed-size array lets the compiler drop all bounds checks.
#[inline(always)]
fn rec_to_point(r: &[u8; REC]) -> Point {
    let f = |o: usize| f32::from_le_bytes([r[o], r[o + 1], r[o + 2], r[o + 3]]);
    Point {
        pos: [f(0), f(4), f(8)],
        rgb: [r[12], r[13], r[14]],
        normal: [f(15), f(19), f(23)],
    }
}

/// Number of points decoded per parallel task.
const PAR_CHUNK: usize = 1 << 16;

/// Decodes points 0, `stride`, 2·`stride`, ... of the body in parallel.
fn parse_body(body: &[u8], n: usize, stride: usize) -> Vec<Point> {
    let m = n.div_ceil(stride);
    let mut out: Vec<Point> = Vec::with_capacity(m);
    out.spare_capacity_mut()[..m]
        .par_chunks_mut(PAR_CHUNK)
        .enumerate()
        .for_each(|(ci, dst)| {
            let base = ci * PAR_CHUNK;
            for (j, d) in dst.iter_mut().enumerate() {
                let o = (base + j) * stride * REC;
                let r: &[u8; REC] = body[o..o + REC].try_into().unwrap();
                d.write(rec_to_point(r));
            }
        });
    // SAFETY: every slot in 0..m was initialized by the loop above.
    unsafe { out.set_len(m) };
    out
}

/// Memory-maps `path` read-only.
fn map_file(path: &Path) -> Result<memmap2::Mmap> {
    let f = std::fs::File::open(path).with_context(|| format!("cannot open {}", path.display()))?;
    // SAFETY: the mapping is read-only; the file must not be modified by another process while mapped.
    Ok(unsafe { memmap2::Mmap::map(&f)? })
}

/// Reads all points of a PLY file.
///
/// # Errors
///
/// Fails when the file cannot be opened or does not match the expected layout.
pub fn read_points(path: &Path) -> Result<Vec<Point>> {
    read_points_stride(path, 1)
}

/// Reads every `stride`-th point of a PLY file without decoding the skipped points.
///
/// The result equals `read_points(path)?.into_iter().step_by(stride)`.
///
/// # Errors
///
/// Fails when the file cannot be opened or does not match the expected layout.
///
/// # Panics
///
/// Panics if `stride` is 0.
pub fn read_points_stride(path: &Path, stride: usize) -> Result<Vec<Point>> {
    assert!(stride >= 1);
    let map = map_file(path)?;
    let (end, n) = parse_header(&map)?;
    Ok(parse_body(&map[end..end + n * REC], n, stride))
}

/// Welds the block meshes and writes them as one PLY file.
///
/// Vertices are `x y z` float plus `red green blue` uchar; faces are `vertex_indices` as a uchar-counted
/// int list.
///
/// # Returns
///
/// The number of welded vertices and triangles.
#[allow(dead_code)]
pub fn write_merged(path: &Path, meshes: &[&BlockMesh]) -> Result<(usize, usize)> {
    let w = weld(meshes);
    write_welded(path, &w)?;
    Ok((w.0.len(), w.2.len()))
}

/// Welded mesh: vertex positions, vertex colours and triangles.
pub type Welded = (Vec<[f32; 3]>, Vec<[u8; 3]>, Vec<[u32; 3]>);

/// Writes a welded mesh as a binary PLY file.
///
/// The whole file is assembled in memory in parallel and written with a single call.
pub fn write_welded(path: &Path, (verts, cols, tris): &Welded) -> Result<()> {
    let header = format!(
        "ply\nformat binary_little_endian 1.0\ncomment deltamesh merged, ENU m\nelement vertex {}\n\
         property float x\nproperty float y\nproperty float z\nproperty uchar red\nproperty uchar green\nproperty uchar blue\n\
         element face {}\nproperty list uchar int vertex_indices\nend_header\n",
        verts.len(),
        tris.len()
    );
    const VB: usize = 15;
    const FB: usize = 13;
    let h = header.len();
    let nv = verts.len() * VB;
    let mut buf = vec![0u8; h + nv + tris.len() * FB];
    buf[..h].copy_from_slice(header.as_bytes());
    let (vbuf, fbuf) = buf[h..].split_at_mut(nv);
    vbuf.par_chunks_exact_mut(VB)
        .zip(verts.par_iter().zip(cols.par_iter()))
        .with_min_len(4096)
        .for_each(|(d, (p, c))| {
            let d: &mut [u8; VB] = d.try_into().unwrap();
            d[0..4].copy_from_slice(&p[0].to_le_bytes());
            d[4..8].copy_from_slice(&p[1].to_le_bytes());
            d[8..12].copy_from_slice(&p[2].to_le_bytes());
            d[12..15].copy_from_slice(c);
        });
    fbuf.par_chunks_exact_mut(FB)
        .zip(tris.par_iter())
        .with_min_len(4096)
        .for_each(|(d, t)| {
            let d: &mut [u8; FB] = d.try_into().unwrap();
            d[0] = 3;
            d[1..5].copy_from_slice(&(t[0] as i32).to_le_bytes());
            d[5..9].copy_from_slice(&(t[1] as i32).to_le_bytes());
            d[9..13].copy_from_slice(&(t[2] as i32).to_le_bytes());
        });
    std::fs::write(path, &buf)?;
    Ok(())
}

/// Merges vertices whose positions are bitwise equal, keeping first-occurrence order.
///
/// The hash map and output vectors are sized up front so no rehashing or reallocation happens. Because
/// keys are raw bits, `-0.0` and `0.0` stay distinct.
pub fn weld(meshes: &[&BlockMesh]) -> Welded {
    let nv: usize = meshes.iter().map(|m| m.vertex_count()).sum();
    let nt: usize = meshes.iter().map(|m| m.tri_count()).sum();
    let mut key: rustc_hash::FxHashMap<[u32; 3], u32> =
        rustc_hash::FxHashMap::with_capacity_and_hasher(nv, Default::default());
    let (mut verts, mut cols) = (Vec::with_capacity(nv), Vec::with_capacity(nv));
    let mut tris = Vec::with_capacity(nt);
    let mut map: Vec<u32> = Vec::new();
    for m in meshes {
        map.clear();
        map.extend(
            m.positions
                .chunks_exact(3)
                .zip(m.colors.chunks_exact(3))
                .map(|(p, c)| {
                    *key.entry([p[0].to_bits(), p[1].to_bits(), p[2].to_bits()])
                        .or_insert_with(|| {
                            verts.push([p[0], p[1], p[2]]);
                            cols.push([c[0], c[1], c[2]]);
                            verts.len() as u32 - 1
                        })
                }),
        );
        tris.extend(
            m.indices
                .chunks_exact(3)
                .map(|t| [map[t[0] as usize], map[t[1] as usize], map[t[2] as usize]]),
        );
    }
    (verts, cols, tris)
}

/// Writes one block as an SLMB binary file for clients.
///
/// All values are little-endian: magic `SLMB`, format version `1u32`, block id `3 × i32`, block version
/// `u32`, vertex count `u32`, triangle count `u32`, positions `3 × f32` per vertex, colours `3 × u8` per
/// vertex padded with zeros to a multiple of 4 bytes, then indices `3 × u32` per triangle.
pub fn write_block(path: &Path, m: &BlockMesh) -> Result<()> {
    let mut w = BufWriter::new(std::fs::File::create(path)?);
    w.write_all(b"SLMB")?;
    w.write_all(&1u32.to_le_bytes())?;
    for x in m.id {
        w.write_all(&x.to_le_bytes())?;
    }
    for x in [m.version, m.vertex_count() as u32, m.tri_count() as u32] {
        w.write_all(&x.to_le_bytes())?;
    }
    for x in &m.positions {
        w.write_all(&x.to_le_bytes())?;
    }
    w.write_all(&m.colors)?;
    let pad = (4 - m.colors.len() % 4) % 4;
    w.write_all(&[0u8; 3][..pad])?;
    for x in &m.indices {
        w.write_all(&x.to_le_bytes())?;
    }
    w.flush()?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tmp(name: &str) -> std::path::PathBuf {
        let d = std::env::temp_dir().join(format!("deltamesh_ply_test_{}", std::process::id()));
        std::fs::create_dir_all(&d).unwrap();
        d.join(name)
    }

    fn header(props: &str, n: usize) -> String {
        format!(
            "ply\nformat binary_little_endian 1.0\ncomment t\nelement vertex {n}\n{props}end_header\n"
        )
    }
    const OK_PROPS: &str = "property float x\nproperty float y\nproperty float z\nproperty uchar red\nproperty uchar green\nproperty uchar blue\nproperty float32 nx\nproperty float32 ny\nproperty float32 nz\n";

    fn body(n: usize) -> (Vec<u8>, Vec<Point>) {
        let mut b = Vec::new();
        let mut want = Vec::new();
        for i in 0..n {
            let f = i as f32;
            let p = Point {
                pos: [f, -f * 0.5, f + 0.25],
                rgb: [i as u8, (i * 3) as u8, (i * 7) as u8],
                normal: [0.1 * f, f32::NAN, 1e15],
            };
            for x in p.pos {
                b.extend_from_slice(&x.to_le_bytes());
            }
            b.extend_from_slice(&p.rgb);
            for x in p.normal {
                b.extend_from_slice(&x.to_le_bytes());
            }
            want.push(p);
        }
        (b, want)
    }

    fn same(a: &Point, b: &Point) -> bool {
        a.pos.map(f32::to_bits) == b.pos.map(f32::to_bits)
            && a.rgb == b.rgb
            && a.normal.map(f32::to_bits) == b.normal.map(f32::to_bits)
    }

    /// Full and strided reads match the written points bit for bit, across several parallel chunks.
    #[test]
    fn read_and_stride() {
        let n = 200_003;
        let (b, want) = body(n);
        let p = tmp("ok.ply");
        let mut f = header(OK_PROPS, n).into_bytes();
        f.extend_from_slice(&b);
        std::fs::write(&p, &f).unwrap();
        let got = read_points(&p).unwrap();
        assert_eq!(got.len(), n);
        assert!(got.iter().zip(&want).all(|(a, b)| same(a, b)));
        for s in [2, 4, 7] {
            let g = read_points_stride(&p, s).unwrap();
            let w: Vec<&Point> = want.iter().step_by(s).collect();
            assert_eq!(g.len(), w.len());
            assert!(g.iter().zip(w).all(|(a, b)| same(a, b)));
        }
    }

    /// Wrong types, wrong order, missing properties, extra elements, ASCII format and a short body are all
    /// rejected.
    #[test]
    fn rejects_non_contract() {
        let (b, _) = body(3);
        let bad = [
            header(&OK_PROPS.replace("uchar red", "float red"), 3),
            header(
                &OK_PROPS.replace(
                    "property float x\nproperty float y\n",
                    "property float y\nproperty float x\n",
                ),
                3,
            ),
            header(&OK_PROPS.replace("property float32 nz\n", ""), 3),
            header(
                &format!("{OK_PROPS}element face 1\nproperty list uchar int vertex_indices\n"),
                3,
            ),
            header(OK_PROPS, 3).replace("binary_little_endian", "ascii"),
            header(OK_PROPS, 4),
        ];
        for (i, h) in bad.iter().enumerate() {
            let p = tmp(&format!("bad{i}.ply"));
            let mut f = h.clone().into_bytes();
            f.extend_from_slice(&b);
            std::fs::write(&p, &f).unwrap();
            assert!(read_points(&p).is_err(), "case {i}");
        }
    }

    /// Straightforward reference: hash-map welding followed by many small `write_all` calls.
    fn write_ref(path: &Path, meshes: &[&BlockMesh]) {
        let mut key: rustc_hash::FxHashMap<[u32; 3], u32> = Default::default();
        let (mut verts, mut cols, mut tris) = (
            Vec::<[f32; 3]>::new(),
            Vec::<[u8; 3]>::new(),
            Vec::<[u32; 3]>::new(),
        );
        for m in meshes {
            let map: Vec<u32> = m
                .positions
                .chunks_exact(3)
                .zip(m.colors.chunks_exact(3))
                .map(|(p, c)| {
                    *key.entry([p[0].to_bits(), p[1].to_bits(), p[2].to_bits()])
                        .or_insert_with(|| {
                            verts.push([p[0], p[1], p[2]]);
                            cols.push([c[0], c[1], c[2]]);
                            verts.len() as u32 - 1
                        })
                })
                .collect();
            for t in m.indices.chunks_exact(3) {
                tris.push([map[t[0] as usize], map[t[1] as usize], map[t[2] as usize]]);
            }
        }
        let mut w = BufWriter::new(std::fs::File::create(path).unwrap());
        write!(
            w,
            "ply\nformat binary_little_endian 1.0\ncomment deltamesh merged, ENU m\nelement vertex {}\n\
             property float x\nproperty float y\nproperty float z\nproperty uchar red\nproperty uchar green\nproperty uchar blue\n\
             element face {}\nproperty list uchar int vertex_indices\nend_header\n",
            verts.len(),
            tris.len()
        )
        .unwrap();
        for (p, c) in verts.iter().zip(&cols) {
            for x in p {
                w.write_all(&x.to_le_bytes()).unwrap();
            }
            w.write_all(c).unwrap();
        }
        for t in &tris {
            w.write_all(&[3u8]).unwrap();
            for i in t {
                w.write_all(&(*i as i32).to_le_bytes()).unwrap();
            }
        }
        w.flush().unwrap();
    }

    /// The parallel writer produces the same bytes as the reference, for empty input too.
    ///
    /// Grid vertices are shared between blocks so they coincide at block borders; the data also contains
    /// duplicates within a block and both `-0.0` and `0.0`.
    #[test]
    fn weld_and_write_same_as_reference() {
        let mut meshes = Vec::new();
        for b in 0..40i32 {
            let mut m = BlockMesh {
                id: [b, 0, 0],
                version: 1,
                ..Default::default()
            };
            let nv = 50 + (b as usize * 7) % 30;
            for j in 0..nv {
                let x = ((b * 13 + j as i32 * 5) % 97) as f32 * 0.2;
                let y = if j % 11 == 0 {
                    -0.0
                } else {
                    ((j * 3) % 17) as f32 * 0.2
                };
                m.positions.extend_from_slice(&[x, y, -31.0]);
                m.colors.extend_from_slice(&[b as u8, j as u8, 7]);
            }
            for j in 0..nv - 2 {
                m.indices
                    .extend_from_slice(&[j as u32, j as u32 + 1, j as u32 + 2]);
            }
            meshes.push(m);
        }
        meshes.push(BlockMesh {
            id: [99, 0, 0],
            ..Default::default()
        });
        let refs: Vec<&BlockMesh> = meshes.iter().collect();
        let (a, b) = (tmp("w_new.ply"), tmp("w_ref.ply"));
        write_merged(&a, &refs).unwrap();
        write_ref(&b, &refs);
        assert_eq!(std::fs::read(&a).unwrap(), std::fs::read(&b).unwrap());
        write_merged(&a, &[]).unwrap();
        write_ref(&b, &[]);
        assert_eq!(std::fs::read(&a).unwrap(), std::fs::read(&b).unwrap());
    }
}
