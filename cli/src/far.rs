//! Per-segment point grids for the `far_frac` distance check.
//!
//! A [`SegGrid`] is built once when a segment's points arrive and reused in every later step.
//! [`far_fraction`] counts mesh vertices that have no input point within distance `d`, searching the 27
//! grid cells around each vertex. The cell key, the 27-cell neighbourhood and the squared-distance
//! comparison match a single hash grid over all points, so the result is bitwise identical to that
//! reference (see the tests).

use rayon::prelude::*;
use rustc_hash::FxHashMap;
use std::sync::atomic::{AtomicU32, Ordering::Relaxed};

/// Grid cell key of `p` for cell size `1 / inv`.
///
/// Must stay identical to the reference implementation in the tests.
#[inline(always)]
fn key(p: &[f32; 3], inv: f32) -> [i32; 3] {
    [
        (p[0] * inv).floor() as i32,
        (p[1] * inv).floor() as i32,
        (p[2] * inv).floor() as i32,
    ]
}

/// Cell lookup structure of a [`SegGrid`].
enum Index {
    /// Dense CSR used when the key range is small: the points of cell `c` are `pts[start[c]..start[c + 1]]`.
    Dense { dims: [i64; 3], start: Vec<u32> },
    /// Hash map used when the key range is too large (for example because of outliers): cell to
    /// (start, count).
    Sparse(FxHashMap<[i32; 3], (u32, u32)>),
}

/// Points of one segment bucketed by grid cell.
pub struct SegGrid {
    inv: f32,
    lo: [i32; 3],
    hi: [i32; 3],
    /// Points sorted by cell.
    pts: Vec<[f32; 3]>,
    index: Index,
}

impl SegGrid {
    /// Builds a grid from bare positions.
    #[cfg(test)]
    pub fn new(src: &[[f32; 3]], d: f32) -> Self {
        Self::from_points(src, |p| *p, d)
    }

    /// Builds a grid with cell size `d`, reading positions through `pos` without copying them first.
    ///
    /// The key range is computed in parallel without storing keys. If the dense cell count fits in
    /// `max(n, 2^20)` cells (and does not overflow), a parallel counting sort fills a CSR index; the
    /// order of points inside a cell may vary between runs, which does not affect the result because a
    /// query only asks whether any point lies within `d`. Otherwise the points are sorted by key and
    /// indexed with a hash map.
    ///
    /// The counting sort uses an inclusive prefix sum so that `start[c]` is the end of cell `c`; filling
    /// each cell from its end with `fetch_sub` leaves `start[c]` at the cell start once all points are
    /// placed.
    ///
    /// # Panics
    ///
    /// Panics if `src` has `u32::MAX` or more points.
    pub fn from_points<T: Sync>(src: &[T], pos: impl Fn(&T) -> [f32; 3] + Sync, d: f32) -> Self {
        let inv = 1.0 / d;
        let n = src.len();
        assert!(n < u32::MAX as usize);
        const MIN: usize = 1 << 14;
        let (lo, hi) = src
            .par_iter()
            .with_min_len(MIN)
            .fold(
                || ([i32::MAX; 3], [i32::MIN; 3]),
                |(mut lo, mut hi), p| {
                    let k = key(&pos(p), inv);
                    for a in 0..3 {
                        lo[a] = lo[a].min(k[a]);
                        hi[a] = hi[a].max(k[a]);
                    }
                    (lo, hi)
                },
            )
            .reduce(
                || ([i32::MAX; 3], [i32::MIN; 3]),
                |a, b| {
                    let mut r = a;
                    for i in 0..3 {
                        r.0[i] = a.0[i].min(b.0[i]);
                        r.1[i] = a.1[i].max(b.1[i]);
                    }
                    r
                },
            );
        if n == 0 {
            return Self {
                inv,
                lo,
                hi,
                pts: Vec::new(),
                index: Index::Sparse(Default::default()),
            };
        }
        let dims = [0, 1, 2].map(|a| hi[a] as i64 - lo[a] as i64 + 1);
        let cells = dims[0]
            .checked_mul(dims[1])
            .and_then(|c| c.checked_mul(dims[2]));
        if let Some(cells) = cells.filter(|&c| c <= (n as i64).max(1 << 20) && c < u32::MAX as i64)
        {
            let cells = cells as usize;
            let lin = |p: &T| {
                let k = key(&pos(p), inv);
                (((k[2] as i64 - lo[2] as i64) * dims[1] + (k[1] as i64 - lo[1] as i64)) * dims[0]
                    + (k[0] as i64 - lo[0] as i64)) as usize
            };
            let mut start = vec![0u32; cells + 1];
            {
                // SAFETY: AtomicU32 has the same size and alignment as u32, and `start` is exclusively
                // borrowed for the lifetime of `cnt`.
                let cnt: &[AtomicU32] =
                    unsafe { &*(start.as_mut_slice() as *mut [u32] as *const [AtomicU32]) };
                src.par_iter().with_min_len(MIN).for_each(|p| {
                    cnt[lin(p)].fetch_add(1, Relaxed);
                });
            }
            for c in 1..cells {
                start[c] += start[c - 1];
            }
            start[cells] = n as u32;
            let mut pts: Vec<[f32; 3]> = Vec::with_capacity(n);
            {
                let dst = pts.spare_capacity_mut().as_mut_ptr() as usize;
                // SAFETY: as above.
                let cur: &[AtomicU32] =
                    unsafe { &*(start.as_mut_slice() as *mut [u32] as *const [AtomicU32]) };
                src.par_iter().with_min_len(MIN).for_each(|p| {
                    let i = cur[lin(p)].fetch_sub(1, Relaxed) - 1;
                    // SAFETY: each point receives a distinct index `i` in 0..n, within the capacity of `pts`.
                    unsafe { (dst as *mut [f32; 3]).add(i as usize).write(pos(p)) };
                });
            }
            // SAFETY: all n slots were written above.
            unsafe { pts.set_len(n) };
            Self {
                inv,
                lo,
                hi,
                pts,
                index: Index::Dense { dims, start },
            }
        } else {
            let mut order: Vec<([i32; 3], u32)> = src
                .par_iter()
                .enumerate()
                .with_min_len(MIN)
                .map(|(i, p)| (key(&pos(p), inv), i as u32))
                .collect();
            order.par_sort_unstable();
            let pts: Vec<[f32; 3]> = order
                .par_iter()
                .with_min_len(MIN)
                .map(|&(_, i)| pos(&src[i as usize]))
                .collect();
            let mut map: FxHashMap<[i32; 3], (u32, u32)> = FxHashMap::default();
            let mut i = 0;
            while i < n {
                let k = order[i].0;
                let mut j = i;
                while j < n && order[j].0 == k {
                    j += 1;
                }
                map.insert(k, (i as u32, (j - i) as u32));
                i = j;
            }
            Self {
                inv,
                lo,
                hi,
                pts,
                index: Index::Sparse(map),
            }
        }
    }

    /// Number of points in the grid.
    pub fn len(&self) -> usize {
        self.pts.len()
    }

    /// Whether key `k` is within one cell of this grid's key range.
    ///
    /// If not, all 27 cells around `k` are empty.
    #[inline(always)]
    fn near(&self, k: &[i32; 3]) -> bool {
        (0..3).all(|a| k[a] as i64 >= self.lo[a] as i64 - 1 && k[a] as i64 <= self.hi[a] as i64 + 1)
    }

    /// Points in cell `k`; empty if the cell is outside the grid.
    #[inline(always)]
    fn cell(&self, k: [i32; 3]) -> &[[f32; 3]] {
        match &self.index {
            Index::Dense { dims, start } => {
                let r = [0, 1, 2].map(|a| k[a] as i64 - self.lo[a] as i64);
                if (0..3).any(|a| r[a] < 0 || r[a] >= dims[a]) {
                    return &[];
                }
                let c = ((r[2] * dims[1] + r[1]) * dims[0] + r[0]) as usize;
                &self.pts[start[c] as usize..start[c + 1] as usize]
            }
            Index::Sparse(m) => match m.get(&k) {
                Some(&(s, l)) => &self.pts[s as usize..(s + l) as usize],
                None => &[],
            },
        }
    }
}

/// The 27 neighbour offsets, centre cell first.
///
/// The visiting order does not affect the result.
const OFFS: [[i32; 3]; 27] = {
    let mut o = [[0i32; 3]; 27];
    let mut i = 1;
    let mut dz = -1;
    while dz <= 1 {
        let mut dy = -1;
        while dy <= 1 {
            let mut dx = -1;
            while dx <= 1 {
                if !(dx == 0 && dy == 0 && dz == 0) {
                    o[i] = [dx, dy, dz];
                    i += 1;
                }
                dx += 1;
            }
            dy += 1;
        }
        dz += 1;
    }
    o
};

/// Fraction of `verts` whose nearest input point is farther than `d`.
///
/// All grids must have been built with the same `d`. For each vertex, the grids whose range touches the
/// vertex's neighbourhood are collected first (up to 64); with more candidates every grid is tested
/// directly. Returns 0 for an empty vertex list.
pub fn far_fraction(verts: &[[f32; 3]], grids: &[&SegGrid], d: f32) -> f64 {
    if verts.is_empty() {
        return 0.0;
    }
    let inv = 1.0 / d;
    debug_assert!(grids.iter().all(|g| g.inv == inv));
    let d2 = d * d;
    let far = verts
        .par_iter()
        .with_min_len(1024)
        .filter(|v| {
            let k = key(v, inv);
            let mut cand = [0usize; 64];
            let mut nc = 0;
            for (gi, g) in grids.iter().enumerate() {
                if g.near(&k) {
                    if nc == cand.len() {
                        break;
                    }
                    cand[nc] = gi;
                    nc += 1;
                }
            }
            let hit = |g: &SegGrid, o: &[i32; 3]| {
                let c = [
                    k[0].wrapping_add(o[0]),
                    k[1].wrapping_add(o[1]),
                    k[2].wrapping_add(o[2]),
                ];
                g.cell(c).iter().any(|p| {
                    let e = [p[0] - v[0], p[1] - v[1], p[2] - v[2]];
                    e[0] * e[0] + e[1] * e[1] + e[2] * e[2] <= d2
                })
            };
            if nc < cand.len() {
                for o in &OFFS {
                    for &gi in &cand[..nc] {
                        if hit(grids[gi], o) {
                            return false;
                        }
                    }
                }
            } else {
                for o in &OFFS {
                    for g in grids {
                        if g.near(&k) && hit(g, o) {
                            return false;
                        }
                    }
                }
            }
            true
        })
        .count();
    far as f64 / verts.len() as f64
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Reference implementation: one hash grid over all points.
    fn far_ref(verts: &[[f32; 3]], pts: &[&[[f32; 3]]], d: f32) -> f64 {
        if verts.is_empty() {
            return 0.0;
        }
        let inv = 1.0 / d;
        let mut grid: FxHashMap<[i32; 3], Vec<[f32; 3]>> = FxHashMap::default();
        for set in pts {
            for p in set.iter() {
                grid.entry(key(p, inv)).or_default().push(*p);
            }
        }
        let d2 = d * d;
        let far = verts
            .iter()
            .filter(|v| {
                let k = key(v, inv);
                for dz in -1..=1 {
                    for dy in -1..=1 {
                        for dx in -1..=1 {
                            if let Some(c) = grid.get(&[k[0] + dx, k[1] + dy, k[2] + dz]) {
                                if c.iter().any(|p| {
                                    let e = [p[0] - v[0], p[1] - v[1], p[2] - v[2]];
                                    e[0] * e[0] + e[1] * e[1] + e[2] * e[2] <= d2
                                }) {
                                    return false;
                                }
                            }
                        }
                    }
                }
                true
            })
            .count();
        far as f64 / verts.len() as f64
    }

    fn rng(s: &mut u64) -> f32 {
        *s ^= *s << 13;
        *s ^= *s >> 7;
        *s ^= *s << 17;
        (*s >> 40) as f32 / (1u64 << 24) as f32
    }

    /// Matches the reference bit for bit for several distances.
    ///
    /// One outlier in the last set exceeds the dense-array limit so the sparse path is exercised too.
    #[test]
    fn same_as_reference() {
        let mut s = 0x9e3779b97f4a7c15u64;
        let mut sets: Vec<Vec<[f32; 3]>> = (0..5)
            .map(|i| {
                (0..3000)
                    .map(|_| {
                        [
                            rng(&mut s) * 20.0 + i as f32 * 3.0 - 10.0,
                            rng(&mut s) * 15.0 - 7.0,
                            rng(&mut s) * 2.0 - 31.0,
                        ]
                    })
                    .collect()
            })
            .collect();
        sets[4].push([1.0e6, -1.0e6, 5.0e5]);
        let verts: Vec<[f32; 3]> = (0..20000)
            .map(|_| {
                [
                    rng(&mut s) * 40.0 - 15.0,
                    rng(&mut s) * 20.0 - 10.0,
                    rng(&mut s) * 4.0 - 33.0,
                ]
            })
            .collect();
        for d in [0.1f32, 0.4, 0.7] {
            let grids: Vec<SegGrid> = sets.iter().map(|p| SegGrid::new(p, d)).collect();
            assert!(matches!(grids[4].index, Index::Sparse(_)));
            assert!(matches!(grids[0].index, Index::Dense { .. }));
            let gr: Vec<&SegGrid> = grids.iter().collect();
            let pr: Vec<&[[f32; 3]]> = sets.iter().map(|v| v.as_slice()).collect();
            let a = far_fraction(&verts, &gr, d);
            let b = far_ref(&verts, &pr, d);
            assert_eq!(a.to_bits(), b.to_bits(), "d={d}");
            assert!(a > 0.0 && a < 1.0);
        }
    }
}
