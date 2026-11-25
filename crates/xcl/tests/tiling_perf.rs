//! Timing of texture detiling / tiling on realistic sizes (`gpu::tiling`). Ignored by default:
//! `cargo test --release -p xcl --test tiling_perf -- --ignored --nocapture`

use std::time::Instant;

use xcl::gpu::tiling::{Chain, Layout, PipeConfig};

/// median time of `f` over `n` runs, in milliseconds
fn median_ms(n: usize, mut f: impl FnMut()) -> f64 {
    f();
    let mut t: Vec<f64> = (0..n)
        .map(|_| {
            let s = Instant::now();
            f();
            s.elapsed().as_secs_f64() * 1e3
        })
        .collect();
    t.sort_by(|a, b| a.partial_cmp(b).unwrap());
    t[n / 2]
}

fn bytes(n: usize) -> Vec<u8> {
    let mut s = 0x1234_5678u32;
    (0..n)
        .map(|_| {
            s = s.wrapping_mul(1_103_515_245).wrapping_add(12345);
            (s >> 16) as u8
        })
        .collect()
}

#[test]
#[ignore]
fn bench_detile_and_tile() {
    // (what, mode, element log2, pixels per element side, width, height, mips, pipes, packers)
    let cases = [
        ("RGBA8 3840x2160 64KB_D", 10, 2, 1, 3840, 2160, 1, 0, 0),
        ("RGBA8 3840x2160 4KB_S", 5, 2, 1, 3840, 2160, 1, 0, 0),
        ("RGBA8 3840x2160 64KB_R_X SeriesX", 27, 2, 1, 3840, 2160, 1, 3, 2),
        ("RGBA8 3840x2160 64KB_Z_X SeriesX", 24, 2, 1, 3840, 2160, 1, 3, 2),
        ("BC7 3840x2160 64KB_D", 10, 4, 4, 3840, 2160, 1, 0, 0),
        ("BC7 3840x2160 64KB_S_X SeriesX", 25, 4, 4, 3840, 2160, 1, 3, 2),
        ("BC7 4096x4096 64KB_D 13 mips", 10, 4, 4, 4096, 4096, 13, 0, 0),
    ];
    for (what, mode, e, side, w, h, mips, p, k) in cases {
        let c = Chain::new(Layout::new(mode, e, PipeConfig { pipes_log2: p, pkrs_log2: k }).unwrap(), w, h, side, mips);
        let mem = bytes(c.slice_bytes);
        let n = c.levels.len();
        let detile = median_ms(9, || {
            for i in 0..n {
                std::hint::black_box(c.detile_level(std::hint::black_box(&mem), i, 0));
            }
        });
        let lin: Vec<Vec<u8>> = (0..n).map(|i| c.detile_level(&mem, i, 0)).collect();
        let mut dst = vec![0u8; c.slice_bytes];
        let tile = median_ms(9, || {
            for (i, l) in lin.iter().enumerate() {
                c.tile_level(std::hint::black_box(&mut dst), i, 0, l);
            }
        });
        // the floor: the same bytes copied into a new Vec
        let floor = median_ms(9, || {
            std::hint::black_box(std::hint::black_box(&mem[..]).to_vec());
        });
        println!("{what}: detile_level {detile:.2} ms, tile_level {tile:.2} ms (memcpy to a new Vec {floor:.2} ms)");
    }
    // a 3D texture: 256x256x64 RGBA8, 4KB_S thick blocks, 1 mip
    let c = Chain::new_3d(Layout::new_3d(5, 2, PipeConfig { pipes_log2: 0, pkrs_log2: 0 }).unwrap(), 256, 256, 64, 1, 1);
    let mut mem = bytes(c.slice_bytes * c.layers());
    let detile = median_ms(9, || {
        std::hint::black_box(c.detile_volume(std::hint::black_box(&mem), 0));
    });
    let lin = c.detile_volume(&mem, 0);
    let tile = median_ms(9, || c.tile_volume(std::hint::black_box(&mut mem), 0, &lin));
    println!("RGBA8 256x256x64 4KB_S thick: detile_volume {detile:.2} ms, tile_volume {tile:.2} ms");
}
