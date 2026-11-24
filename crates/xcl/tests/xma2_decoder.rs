//! The XMA1/XMA2 decoder (`hle::xma2`) through its public API only: packet framing and stream
//! routing, frames written here bit by bit, and malformed input. Every XAudio2 voice a title
//! submits as WAVE_FORMAT_XMA2 goes through it. Extradata parsing: `xma2_extradata.rs`.
//!
//! The syntax written below is FFmpeg's `libavcodec/wmaprodec.c`, which the module ports: packet
//! header in `decode_packet`, frame in `decode_frame`, tile header in `decode_tilehdr`, subframe in
//! `decode_subframe`, with the code tables of `wmaprodata.h` (here `xma2::tables`).

use std::panic::catch_unwind;
use xcl::hle::xma2::tables::{COEF0_LENS, COEF0_SYMS, CRITICAL_FREQ, SCALE_TABLE, VEC4_LENS, VEC4_SYMS};
use xcl::hle::xma2::{Decoder, StreamConfig, Version, PACKET_SIZE, SAMPLES_PER_FRAME};

#[derive(Clone, Default)]
struct Bits(Vec<bool>);

impl Bits {
    fn put(&mut self, n: usize, v: u32) {
        self.0.extend((0..n).rev().map(|i| (v >> i) & 1 != 0));
    }
    fn code(&mut self, (len, code): (usize, u32)) {
        self.put(len, code);
    }
}

fn bytes(bits: &[bool], size: usize) -> Vec<u8> {
    let mut out = vec![0u8; size.max(bits.len().div_ceil(8))];
    for (i, &b) in bits.iter().enumerate() {
        out[i / 8] |= (b as u8) << (7 - i % 8);
    }
    out
}

/// Canonical code of `sym`: `ff_vlc_init_from_lengths` gives the table's entries codes in order,
/// each the previous one plus 2^(32 - len) as a left-aligned 32-bit value.
fn code_of(table: impl IntoIterator<Item = (i32, u8)>, sym: i32) -> (usize, u32) {
    let mut code = 0u64;
    for (s, len) in table.into_iter().filter(|e| e.1 > 0) {
        if s == sym {
            return (len as usize, (code >> (32 - len)) as u32);
        }
        code += 1 << (32 - len);
    }
    panic!("symbol {sym} not in table");
}

/// Scale-factor bands of a 512-sample subframe: `decode_init`'s sfb_offsets loop over
/// `ff_wmapro_critical_freq`, at XMA's rate classes (`get_rate`: 24k / 32k / 44.1k / 48k).
fn num_bands(sample_rate: u32) -> usize {
    let rate = match sample_rate {
        r if r > 44_100 => 48_000,
        r if r > 32_000 => 44_100,
        r if r > 24_000 => 32_000,
        _ => 24_000,
    };
    let mut offs = vec![0usize];
    for &cf in &CRITICAL_FREQ {
        if *offs.last().unwrap() >= SAMPLES_PER_FRAME || offs.len() >= 29 {
            break;
        }
        let o = ((SAMPLES_PER_FRAME * 2 * cf as usize) / rate + 2) & !3;
        if o > *offs.last().unwrap() {
            offs.push(o);
        }
        if o >= SAMPLES_PER_FRAME {
            break;
        }
    }
    offs.len() - 1
}

#[derive(Clone, Copy, PartialEq)]
enum Coupling {
    Independent,
    MidSide,
    MidSideNoBands,
}

/// Spectral coefficient ±1 at bin `pos` (0..4, the first vector-coded group).
#[derive(Clone, Copy)]
struct Tone {
    pos: usize,
    negative: bool,
}

#[derive(Clone)]
struct Frame {
    tones: Vec<Option<Tone>>,
    coupling: Coupling,
    gain: i32,
    fill: usize,
    trim: Option<(u32, u32)>,
}

fn silent(channels: usize) -> Frame {
    Frame { tones: vec![None; channels], coupling: Coupling::Independent, gain: 0, fill: 0, trim: None }
}

fn tone(tones: Vec<Option<Tone>>, gain: i32) -> Frame {
    Frame { tones, gain, ..silent(0) }
}

const T0: Option<Tone> = Some(Tone { pos: 0, negative: false });

/// One XMA frame (len_prefix, DRC, one 512-sample subframe): `decode_frame` reads a 15-bit total
/// length, tile header, [stereo: postproc bit], 8-bit DRC gain, trim flags, the subframe, then
/// requires length == bits read + 2: one skipped bit and the "more frames" bit.
fn encode(f: &Frame, bands: usize, more: bool) -> Bits {
    let nch = f.tones.len();
    let mut b = Bits::default();
    b.put(2, 0b10); // tile header: fixed channel layout, subframe length code 0 (512)
    if nch == 2 {
        b.put(1, 0); // no postproc transform
    }
    b.put(8, 0); // DRC gain
    match f.trim {
        Some((start, end)) => {
            b.put(2, 0b11);
            b.put(10, start);
            b.put(1, 1);
            b.put(10, end);
        }
        None => b.put(1, 0),
    }
    if f.fill > 0 {
        b.put(3, 0b100); // extended header, fill count follows as 4-bit width + value
        b.put(4, 15);
        b.put(15, f.fill as u32 - 1);
        b.0.resize(b.0.len() + f.fill, false);
    } else {
        b.put(1, 0);
    }
    b.put(1, 0); // reserved
    if nch == 2 {
        b.put(1, 0); // no extended channel transform
        match f.coupling {
            Coupling::Independent => b.put(2, 0b10),
            Coupling::MidSide => b.put(2, 0b01),
            Coupling::MidSideNoBands => b.put(2 + bands, 0),
        }
    }
    for t in &f.tones {
        b.put(1, t.is_some() as u32);
    }
    if f.tones.iter().any(Option::is_some) {
        b.put(1, 1); // num_vec_coeffs transmitted: 8 bits each, value 1 = 4 coefficients
        for _ in 0..nch {
            b.put(8, 1);
        }
        // quant step 90 + gain dB: 6-bit signed, -32 / 31 escape into 5-bit chunks of 31.
        let (head, mut rest) = match f.gain {
            g if (-31..=30).contains(&g) => (g, -1),
            g if g > 30 => (31, g - 31),
            g => (-32, -32 - g),
        };
        b.put(6, head as u32 & 63);
        while rest >= 0 {
            b.put(5, rest.min(31) as u32);
            rest = if rest >= 31 { rest - 31 } else { -1 };
        }
        if nch == 2 {
            b.put(5, 0); // modifier width 0, no modifier on either channel
        }
        let sf0 = code_of(SCALE_TABLE.iter().map(|&(s, l)| (s as i32 - 60, l)), 0);
        for _ in 0..nch {
            b.put(2, 0); // scale factor step 1, every band delta 0 (all 45)
            (0..bands).for_each(|_| b.code(sf0));
        }
        let vec4 = VEC4_SYMS.iter().zip(VEC4_LENS.iter()).map(|(&s, &l)| (s as i32 - 1, l));
        let vec4: Vec<_> = vec4.collect();
        for t in f.tones.iter().flatten() {
            b.put(1, 0); // coefficient table 0
            b.code(code_of(vec4.iter().copied(), 1 << (4 * (3 - t.pos))));
            b.put(1, !t.negative as u32);
            b.code(code_of(COEF0_SYMS.iter().zip(COEF0_LENS.iter()).map(|(&s, &l)| (s as i32, l)), 1)); // end of block
        }
    }
    let mut out = Bits::default();
    out.put(15, (15 + b.0.len() + 2) as u32);
    out.0.extend(b.0);
    out.put(2, more as u32);
    out
}

/// Packets per xma2defs.h / `decode_packet`: a 32-bit header of frame count (6; XMA1: 4-bit
/// sequence + 2 unused), bit offset of the first frame starting in the packet (15), 3 unused bits
/// and packet skip (8), then the frames back to back; a frame running past the end continues
/// right after the next packet's header.
fn packetize(frames: &[Frame], bands: usize, version: Version, skip: u8) -> Vec<Vec<u8>> {
    const PAYLOAD: usize = PACKET_SIZE * 8 - 32;
    let (mut stream, mut starts) = (Vec::new(), Vec::new());
    for (i, f) in frames.iter().enumerate() {
        starts.push(stream.len());
        stream.extend(encode(f, bands, i + 1 < frames.len()).0);
    }
    (0..stream.len().div_ceil(PAYLOAD))
        .map(|k| {
            let (lo, hi) = (k * PAYLOAD, ((k + 1) * PAYLOAD).min(stream.len()));
            let inside: Vec<usize> = starts.iter().copied().filter(|&s| s >= lo && s < lo + PAYLOAD).collect();
            let mut h = Bits::default();
            match version {
                Version::Xma2 => h.put(6, inside.len() as u32),
                Version::Xma1 => h.put(6, (k as u32 & 15) << 2),
            }
            h.put(15, inside.first().map_or(0x7FFF, |&s| s - lo) as u32);
            h.put(11, skip as u32);
            h.0.extend(&stream[lo..hi]);
            bytes(&h.0, PACKET_SIZE)
        })
        .collect()
}

/// Streams take packets in turn when each packet's skip is (streams - 1) (`xma_decode_packet`).
fn interleave(streams: &[Vec<Vec<u8>>]) -> Vec<u8> {
    let n = streams[0].len();
    assert!(streams.iter().all(|s| s.len() == n));
    (0..n).flat_map(|k| streams.iter().flat_map(move |s| s[k].clone())).collect()
}

/// Same frames for every stream of a default layout (2+2+...+1), as XMA2 packets.
fn layout(channels: u16, rate: u32, per_stream: &dyn Fn(usize, usize) -> Vec<Frame>) -> (Decoder, Vec<u8>) {
    let cfg = StreamConfig::from_xma2_wfx(channels, rate, &[]).unwrap();
    let skip = cfg.stream_channels.len() as u8 - 1;
    let pkts: Vec<_> = (cfg.stream_channels.iter().enumerate())
        .map(|(i, &ch)| packetize(&per_stream(i, ch as usize), num_bands(rate), Version::Xma2, skip))
        .collect();
    (Decoder::new(cfg).unwrap(), interleave(&pkts))
}

fn mono(frames: &[Frame], rate: u32) -> Vec<f32> {
    let (mut d, data) = layout(1, rate, &|_, _| frames.to_vec());
    d.decode_packets_f32(&data, true).unwrap()
}

fn ch(out: &[f32], c: usize, n: usize) -> Vec<f32> {
    out.iter().skip(c).step_by(n).copied().collect()
}

struct Lcg(u64);
impl Lcg {
    fn next(&mut self) -> u32 {
        self.0 = self.0.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        (self.0 >> 33) as u32
    }
}

/// `Decoder::new` takes a hand-built config only when its streams add up to its channel count.
#[test]
fn decoder_new_checks_hand_built_configs() {
    let cfg = |channels, stream_channels: Vec<u8>| {
        StreamConfig { version: Version::Xma2, sample_rate: 48_000, channels, stream_channels }
    };
    assert!(Decoder::new(cfg(0, vec![])).is_err());
    assert!(Decoder::new(cfg(9, vec![1; 9])).is_err());
    assert!(Decoder::new(cfg(3, vec![2, 2])).is_err());
    assert!(Decoder::new(cfg(2, vec![2])).is_ok());
}

/// Silent frames (no coefficients) decode to exact zeros for every layout and both versions: the
/// first frame only primes the overlap, the stream's end adds the last half frame, and 64
/// encoder-delay samples go: N frames give N * 512 - 64 samples per channel.
#[test]
fn silent_frames_decode_to_zeros_of_the_right_length() {
    let layouts = [(1, Version::Xma2), (2, Version::Xma2), (2, Version::Xma1), (6, Version::Xma2), (16, Version::Xma2)];
    for (channels, version) in layouts {
        let cfg = StreamConfig { version, ..StreamConfig::from_xma2_wfx(channels, 44_100, &[]).unwrap() };
        let skip = cfg.stream_channels.len() as u8 - 1;
        let pkts: Vec<_> = (cfg.stream_channels.iter())
            .map(|&c| packetize(&vec![silent(c as usize); 5], num_bands(44_100), version, skip))
            .collect();
        let mut d = Decoder::new(cfg).unwrap();
        let out = d.decode_packets_f32(&interleave(&pkts), true).unwrap();
        assert_eq!(out.len(), (5 * SAMPLES_PER_FRAME - 64) * channels as usize, "{channels}ch {version:?}");
        assert!(out.iter().all(|&s| s == 0.0), "{channels}ch {version:?}");
    }
}

/// trim_end from stream 0's frames cuts trim_end + trim_start - 192 samples off every channel's
/// tail at the end of the stream (`xma_decode_packet` at EOF); trim_start alone cuts nothing.
#[test]
fn bitstream_trim_cuts_the_tail_of_every_channel() {
    for (trim, cut) in [((0, 400), 208), ((100, 300), 208), ((64, 0), 0), ((0, 192), 0)] {
        let (mut d, data) = layout(6, 48_000, &|s, c| {
            let mut f = vec![silent(c); 4];
            f[1].trim = (s == 0).then_some(trim);
            f
        });
        assert_eq!(d.decode_packets_f32(&data, true).unwrap().len(), (4 * 512 - 64 - cut) * 6, "{trim:?}");
    }
}

/// Output level follows the quant step: +20 dB is exactly 10x in amplitude, also through the
/// -32 / 31 escape codes; a negative coefficient negates the output.
#[test]
fn quant_step_scales_and_sign_negates() {
    let frames = |gain, negative| {
        (0..6).map(|i| tone(vec![(i != 3).then_some(Tone { pos: i % 4, negative })], gain)).collect::<Vec<_>>()
    };
    let base = mono(&frames(-10, false), 48_000);
    assert_eq!(base.len(), 6 * 512 - 64);
    let peak = base.iter().fold(0f32, |m, s| m.max(s.abs()));
    assert!(peak > 1e-4, "tone decoded silent");
    for gain in [10, 30, 50, -30, -50, -70] {
        let scale = 10f32.powf((gain + 10) as f32 / 20.0);
        let out = mono(&frames(gain, false), 48_000);
        let err = out.iter().zip(&base).fold(0f32, |m, (a, b)| m.max((a - b * scale).abs()));
        assert!(err <= peak * scale * 1e-4, "gain {gain}: max error {err}");
    }
    let neg = mono(&frames(-10, true), 48_000);
    assert!(neg.iter().zip(&base).all(|(a, b)| *a == -*b));
}

/// Stereo coupling (`decode_channel_transform`, `inverse_channel_transform`): independent
/// channels stay apart; mid/side with only M gives L == R == the mono signal, only S gives
/// L == -R; mid/side with no band coded scales both channels by 181/128.
#[test]
fn stereo_coupling_modes() {
    let mut f = tone(vec![T0], -10);
    let reference = mono(&vec![f.clone(); 4], 48_000);
    let mut stereo = |tones: Vec<Option<Tone>>, coupling| {
        f.tones = tones;
        f.coupling = coupling;
        let (mut d, data) = layout(2, 48_000, &|_, _| vec![f.clone(); 4]);
        let out = d.decode_packets_f32(&data, true).unwrap();
        (ch(&out, 0, 2), ch(&out, 1, 2))
    };
    let (l, r) = stereo(vec![T0, None], Coupling::Independent);
    assert!(l == reference && r.iter().all(|&s| s == 0.0));
    let (l, r) = stereo(vec![None, T0], Coupling::Independent);
    assert!(r == reference && l.iter().all(|&s| s == 0.0));
    let (l, r) = stereo(vec![T0, None], Coupling::MidSide);
    assert!(l == reference && r == reference);
    let (l, r) = stereo(vec![None, T0], Coupling::MidSide);
    assert!(r == reference && l.iter().zip(&r).all(|(a, b)| *a == -*b));
    let (l, r) = stereo(vec![T0, None], Coupling::MidSideNoBands);
    assert!(r.iter().all(|&s| s == 0.0));
    assert!(l.iter().zip(&reference).all(|(a, b)| (a - b * 181.0 / 128.0).abs() <= 1e-6));
}

/// A tone in one stream's channel of a 5.1 or 16-channel layout lands in that output channel
/// only, sample for sample equal to the mono decode, at every rate class.
#[test]
fn multistream_channels_stay_isolated() {
    let cases = [(6u16, 2usize, 48_000), (6, 5, 32_000), (16, 15, 24_000), (5, 4, 44_100), (16, 0, 22_050)];
    for (channels, target, rate) in cases {
        let reference = mono(&vec![tone(vec![T0], 0); 3], rate);
        let (mut d, data) = layout(channels, rate, &|s, c| {
            let mut tones = vec![None; c];
            if s == target / 2 {
                tones[target % 2] = T0;
            }
            vec![tone(tones, 0); 3]
        });
        let out = d.decode_packets_f32(&data, true).unwrap();
        for c in 0..channels as usize {
            let got = ch(&out, c, channels as usize);
            let ok = if c == target { got == reference } else { got.iter().all(|&s| s == 0.0) };
            assert!(ok, "{channels}ch {rate}: channel {c}");
        }
    }
}

fn varied(fill: usize) -> Vec<Frame> {
    let t = |i: usize| tone(vec![Some(Tone { pos: i % 4, negative: i % 3 == 0 })], i as i32 - 6);
    (0..12).map(|i| Frame { fill: fill + 37 * i, ..t(i) }).collect()
}

/// Fill bits that push frames across packet boundaries change nothing: the frames decode the
/// same as when they sit in one packet, whole or fed one packet per call, with 0xFF padding
/// packets and a leading all-zero packet in between.
#[test]
fn frames_spanning_packets_decode_like_unsplit_frames() {
    let compact = mono(&varied(0), 48_000);
    assert_eq!(compact.len(), 12 * 512 - 64);
    let pkts = packetize(&varied(4000), num_bands(48_000), Version::Xma2, 0);
    assert!(pkts.len() >= 3);
    let mut d = Decoder::new(StreamConfig::from_xma2_wfx(1, 48_000, &[]).unwrap()).unwrap();
    assert_eq!(d.decode_packets_f32(&pkts.concat(), true).unwrap(), compact);
    let mut streamed = d.decode_packets_f32(&[0u8; PACKET_SIZE], false).unwrap();
    for p in &pkts {
        streamed.extend(d.decode_packets_f32(p, false).unwrap());
        streamed.extend(d.decode_packets_f32(&[0xFF; PACKET_SIZE], false).unwrap());
    }
    streamed.extend(d.decode_packets_f32(&[], true).unwrap());
    assert_eq!(streamed, compact);
}

/// Every split point around a packet boundary: frame 0's end swept over the 40 bits either side,
/// so the continuation is 1..40 bits (the trailer bits alone at the low end), ends exactly at the
/// boundary, or frame 1's 15-bit length straddles it. All decode like the unsplit stream.
#[test]
fn every_split_point_at_a_packet_boundary() {
    const PAYLOAD: usize = PACKET_SIZE * 8 - 32;
    let want = mono(&varied(0), 48_000);
    let mut frames = varied(0);
    frames[0].fill = 1;
    let base = encode(&frames[0], num_bands(48_000), true).0.len() - 1;
    for end in PAYLOAD - 40..=PAYLOAD + 40 {
        frames[0].fill = end - base;
        let pkts = packetize(&frames, num_bands(48_000), Version::Xma2, 0);
        let mut d = Decoder::new(StreamConfig::from_xma2_wfx(1, 48_000, &[]).unwrap()).unwrap();
        let got = d.decode_packets_f32(&pkts.concat(), true).unwrap();
        assert!(got == want, "frame 0 ending at payload bit {end}: {} samples, want {}", got.len(), want.len());
    }
}

/// A trailing packet shorter than 2048 bytes decodes the frames it holds whole, with or without
/// end_of_stream; one cut mid-frame loses only that frame; under 4 bytes is ignored.
#[test]
fn partial_trailing_packet() {
    let frames = varied(0);
    let full = packetize(&frames, num_bands(48_000), Version::Xma2, 0).concat();
    let used = 32 + frames.iter().map(|f| encode(f, num_bands(48_000), true).0.len()).sum::<usize>();
    let expect = mono(&frames, 48_000);
    let mut d = Decoder::new(StreamConfig::from_xma2_wfx(1, 48_000, &[]).unwrap()).unwrap();
    assert_eq!(d.decode_packets_f32(&full[..used.div_ceil(8)], true).unwrap(), expect);
    let mut out = d.decode_packets_f32(&full[..used.div_ceil(8) + 5], false).unwrap();
    out.extend(d.decode_packets_f32(&full[..3], false).unwrap());
    out.extend(d.decode_packets_f32(&[], true).unwrap());
    assert_eq!(out, expect);
    let cut = d.decode_packets_f32(&full[..used / 8 - 2], true).unwrap();
    assert_eq!(cut.len(), 11 * 512 - 64);
    assert_eq!(cut[..10 * 512 - 64], expect[..10 * 512 - 64]);
}

/// `decode_buffer_i16` is `decode_packets_f32` scaled by 32768, rounded and saturated, including
/// for a tone loud enough to clip.
#[test]
fn i16_output_is_saturated_f32() {
    for gain in [-20, 61, 75] {
        let frames: Vec<_> = (0..5).map(|i| tone(vec![Some(Tone { pos: i % 4, negative: i == 2 })], gain)).collect();
        let pkts = packetize(&frames, num_bands(48_000), Version::Xma2, 0).concat();
        let mut d = Decoder::new(StreamConfig::from_xma2_wfx(1, 48_000, &[]).unwrap()).unwrap();
        let f = d.decode_packets_f32(&pkts, true).unwrap();
        let pcm = d.decode_buffer_i16(&pkts, true).unwrap();
        let want: Vec<i16> = (f.iter()).map(|&s| (s as f64 * 32768.0).round().clamp(-32768.0, 32767.0) as i16).collect();
        assert_eq!(pcm, want, "gain {gain}");
        let clipped = pcm.iter().filter(|&&s| s == i16::MAX || s == i16::MIN).count();
        let peak = f.iter().fold(0f32, |m, s| m.max(s.abs()));
        assert_eq!(clipped > 0, gain > 60, "gain {gain}: {clipped} clipped, peak {peak}");
    }
}

/// `flush` drops all state mid-stream: after it the same packets decode as on a fresh decoder,
/// twice over.
#[test]
fn flush_resets_mid_stream() {
    let (mut d, data) = layout(6, 48_000, &|s, c| {
        varied(3000).into_iter().map(|mut f| {
            f.tones = vec![f.tones[0]; c];
            f.gain -= s as i32;
            f
        }).collect()
    });
    let (mut fresh, _) = layout(6, 48_000, &|_, _| vec![]);
    let want = fresh.decode_packets_f32(&data, true).unwrap();
    assert_eq!(want.len(), (12 * 512 - 64) * 6);
    for cut in [PACKET_SIZE, 5 * PACKET_SIZE, data.len() - PACKET_SIZE] {
        d.decode_packets_f32(&data[..cut], false).unwrap();
        d.flush();
        assert_eq!(d.decode_packets_f32(&data, true).unwrap(), want, "flush after {cut}");
    }
}

/// Garbage never panics: random packets, packets with a random header over a valid stream
/// (frame offset or skip count past the packet), and bit flips in valid packets, for mono,
/// stereo XMA1, 5.1 and 16 channels; output is always whole sample frames.
#[test]
fn random_packets_never_panic() {
    let valid = packetize(&varied(500), num_bands(48_000), Version::Xma2, 0).concat();
    for seed in 0..400u64 {
        let mut r = Lcg(seed);
        let layouts = [(1, Version::Xma2), (2, Version::Xma1), (6, Version::Xma2), (16, Version::Xma2)];
        let (channels, version) = layouts[seed as usize % 4];
        let mut data = match seed / 4 % 3 {
            0 => (0..PACKET_SIZE * (1 + r.next() as usize % 3)).map(|_| r.next() as u8).collect(),
            1 => {
                let mut v = valid.clone();
                for p in v.chunks_mut(PACKET_SIZE) {
                    p[..4].copy_from_slice(&r.next().to_be_bytes());
                }
                v
            }
            _ => {
                let mut v = valid.clone();
                for _ in 0..1 + r.next() % 40 {
                    let bit = r.next() as usize % (v.len() * 8);
                    v[bit / 8] ^= 0x80 >> (bit % 8);
                }
                v
            }
        };
        data.truncate(data.len() - r.next() as usize % 64);
        let eos = r.next() % 2 == 0;
        let res = catch_unwind(|| {
            let cfg = StreamConfig { version, ..StreamConfig::from_xma2_wfx(channels, 48_000, &[]).unwrap() };
            let mut d = Decoder::new(cfg).unwrap();
            let mut lens = vec![];
            for chunk in [&data[..], &data[..data.len() / 2], &[]] {
                lens.push(d.decode_packets_f32(chunk, eos).map_or(0, |v| v.len()));
                lens.push(d.decode_buffer_i16(chunk, !eos).map_or(0, |v| v.len()));
            }
            lens
        });
        let n = data.len();
        let lens = res.unwrap_or_else(|_| panic!("seed {seed}: panicked ({channels}ch {version:?}, {n} bytes)"));
        assert!(lens.iter().all(|n| n % channels as usize == 0), "seed {seed}: {lens:?}");
    }
}
