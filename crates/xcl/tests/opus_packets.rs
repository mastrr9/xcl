//! Opus framing of the packets a title submits to XAudio2 (`hle::opus`): the RFC 6716 TOC byte
//! and frame packing, RFC 6716 Appendix B self-delimited multistream packets, the RFC 7845
//! OpusHead and the buffer extraction / length estimate, checked against reference parsers
//! written here from the RFCs.

use std::borrow::Cow;
use std::panic::{catch_unwind, AssertUnwindSafe};

use xcl::hle::opus::{
    build_opus_head, estimate_buffer_frames, extract_packets, parse_opus_head, parse_packet,
    split_multistream_packet, Bandwidth, Mode,
};

// ---- reference model (RFC 6716) ----

/// RFC 6716 section 3.1 Table 2, from its ranges: config -> (mode, bandwidth, samples at 48 kHz).
fn ref_config(c: u8) -> (Mode, Bandwidth, u32) {
    use Bandwidth::*;
    let c = usize::from(c);
    match c {
        0..=11 => (Mode::Silk, [Narrow, Medium, Wide][c / 4], [480, 960, 1920, 2880][c % 4]),
        12..=15 => (Mode::Hybrid, if c < 14 { SuperWide } else { Full }, [480, 960][c % 2]),
        _ => (Mode::Celt, [Narrow, Wide, SuperWide, Full][(c - 16) / 4], [120, 240, 480, 960][c % 4]),
    }
}

/// RFC 6716 section 3.2.1: one byte below 252, else first + 4 * second.
fn ref_len(d: &[u8], p: &mut usize) -> Option<usize> {
    let b0 = usize::from(*d.get(*p)?);
    *p += 1;
    if b0 < 252 {
        return Some(b0);
    }
    let b1 = usize::from(*d.get(*p)?);
    *p += 1;
    Some(b0 + 4 * b1)
}

/// The (unique) section 3.2.1 coding of a frame length up to 1275.
fn enc_len(n: usize) -> Vec<u8> {
    assert!(n <= 1275);
    if n < 252 { vec![n as u8] } else { vec![(252 + n % 4) as u8, ((n - 252) / 4) as u8] }
}

/// Section 3.2.5 padding length: 255 means 254 bytes and another length byte.
fn enc_pad(mut n: usize) -> Vec<u8> {
    let mut v = Vec::new();
    while n >= 255 {
        v.push(255);
        n -= 254;
    }
    v.push(n as u8);
    v
}

/// Frame lengths of a standard packet, refusing what section 3.4 (R1 to R7) refuses.
fn ref_frames(p: &[u8]) -> Option<Vec<usize>> {
    let (&toc, rest) = p.split_first()?; // R1
    let fs = ref_config(toc >> 3).2;
    let lens = match toc & 3 {
        0 => vec![rest.len()],
        1 if rest.len() % 2 == 0 => vec![rest.len() / 2; 2], // R3
        1 => return None,
        2 => {
            let mut q = 0;
            let a = ref_len(rest, &mut q)?;
            vec![a, rest.len().checked_sub(q + a)?] // R4
        }
        _ => {
            let (&fcb, data) = rest.split_first()?; // R6: N >= 2
            let m = usize::from(fcb & 0x3F);
            if m == 0 || m as u32 * fs > 5760 {
                return None; // R5
            }
            let (mut q, mut pad) = (0, 0);
            if fcb & 0x40 != 0 {
                loop {
                    let b = *data.get(q)?;
                    q += 1;
                    pad += if b == 255 { 254 } else { usize::from(b) };
                    if b != 255 {
                        break;
                    }
                }
            }
            let region = data.get(q..data.len().checked_sub(pad)?)?; // R6 / R7: P <= N - 2
            if fcb & 0x80 != 0 {
                let mut r = 0;
                let mut v = Vec::new();
                for _ in 1..m {
                    v.push(ref_len(region, &mut r)?);
                }
                let last = region.len().checked_sub(r + v.iter().sum::<usize>())?; // R7
                v.push(last);
                v
            } else if region.len() % m == 0 {
                vec![region.len() / m; m] // R6
            } else {
                return None;
            }
        }
    };
    lens.iter().all(|&l| l <= 1275).then_some(lens) // R2
}

/// (code, frames) the parser should report; multistream only checks the leading TOC / count.
fn ref_info(p: &[u8], multistream: bool) -> Option<(u8, u8)> {
    let toc = *p.first()?;
    let n = if !multistream {
        ref_frames(p)?.len()
    } else {
        match toc & 3 {
            0 => 1,
            1 => 2,
            2 if p.len() >= 2 => 2,
            3 if p.len() >= 2 => usize::from(p[1] & 0x3F),
            _ => return None,
        }
    };
    let fs = ref_config(toc >> 3).2;
    (n >= 1 && n as u32 * fs <= 5760).then_some((toc & 3, n as u8))
}

/// parse_packet agrees with the reference on validity and every reported field.
fn check_parse(p: &[u8], multistream: bool) {
    let got = parse_packet(p, multistream);
    let want = ref_info(p, multistream);
    assert_eq!(got.is_some(), want.is_some(), "validity of {:02x?} (multistream {multistream})", p);
    if let (Some(i), Some((code, frames))) = (got, want) {
        let (mode, bw, fs) = ref_config(p[0] >> 3);
        assert_eq!(
            (i.config, i.mode, i.bandwidth, i.stereo, i.code, i.frames, i.frame_samples_48k, i.total_samples_48k),
            (p[0] >> 3, mode, bw, p[0] & 4 != 0, code, frames, fs, fs * u32::from(frames)),
            "{:02x?}",
            p
        );
    }
}

/// Appendix B self-delimited packet at d[*p..] as a standard packet.
fn ref_self_delimited(d: &[u8], p: &mut usize) -> Option<Vec<u8>> {
    let toc = *d.get(*p)?;
    *p += 1;
    let mut out = vec![toc];
    let (lens, pad) = match toc & 3 {
        0 => (vec![ref_len(d, p)?], 0),
        1 => (vec![ref_len(d, p)?; 2], 0),
        2 => {
            let a = ref_len(d, p)?;
            out.extend(enc_len(a));
            (vec![a, ref_len(d, p)?], 0)
        }
        _ => {
            let fcb = *d.get(*p)?;
            *p += 1;
            out.push(fcb);
            let m = usize::from(fcb & 0x3F);
            let mut pad = 0;
            if fcb & 0x40 != 0 {
                loop {
                    let b = *d.get(*p)?;
                    *p += 1;
                    out.push(b);
                    pad += if b == 255 { 254 } else { usize::from(b) };
                    if b != 255 {
                        break;
                    }
                }
            }
            if fcb & 0x80 != 0 {
                let v = (0..m).map(|_| ref_len(d, p)).collect::<Option<Vec<_>>>()?;
                v.iter().take(m.saturating_sub(1)).for_each(|&l| out.extend(enc_len(l)));
                (v, pad)
            } else {
                (vec![ref_len(d, p)?; m], pad)
            }
        }
    };
    let n = lens.iter().sum::<usize>() + pad;
    out.extend_from_slice(d.get(*p..*p + n)?);
    *p += n;
    ref_frames(&out)?;
    Some(out)
}

/// RFC 7845 section 5.1.1: all streams but the last self-delimited, the last standard.
fn ref_split(d: &[u8], streams: usize) -> Option<Vec<Vec<u8>>> {
    if streams == 0 {
        return None;
    }
    let mut p = 0;
    let mut v = (1..streams).map(|_| ref_self_delimited(d, &mut p)).collect::<Option<Vec<_>>>()?;
    ref_frames(&d[p..])?;
    v.push(d[p..].to_vec());
    Some(v)
}

struct Lcg(u64);
impl Lcg {
    fn next(&mut self) -> u32 {
        self.0 = self.0.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        (self.0 >> 33) as u32
    }
    fn below(&mut self, n: usize) -> usize {
        self.next() as usize % n
    }
    fn bytes(&mut self, n: usize) -> Vec<u8> {
        (0..n).map(|_| self.next() as u8).collect()
    }
}

/// A random valid packet in both framings: (standard, self-delimited).
fn random_packet(r: &mut Lcg) -> (Vec<u8>, Vec<u8>) {
    let toc = (r.below(256) as u8) & !3;
    let fs = ref_config(toc >> 3).2;
    let len = |r: &mut Lcg| if r.below(8) == 0 { 252 + r.below(150) } else { r.below(30) };
    let code = r.below(4) as u8;
    let (lens, vbr, pad) = match code {
        0 => (vec![len(r)], false, None),
        1 => (vec![len(r); 2], false, None),
        2 => (vec![len(r), len(r)], false, None),
        _ => {
            let m = 1 + r.below((5760 / fs as usize).min(48));
            let vbr = r.below(2) == 0;
            let lens: Vec<usize> = if vbr { (0..m).map(|_| r.below(20)).collect() } else { vec![r.below(20); m] };
            let pad = [None, Some(0), Some(r.below(600))][r.below(3)];
            (lens, vbr, pad)
        }
    };
    let frames: Vec<u8> = r.bytes(lens.iter().sum());
    let pad_bytes = r.bytes(pad.unwrap_or(0));
    let (mut std, mut sd) = (vec![toc | code], vec![toc | code]);
    match code {
        0 | 1 => sd.extend(enc_len(lens[0])),
        2 => {
            std.extend(enc_len(lens[0]));
            sd.extend(enc_len(lens[0]));
            sd.extend(enc_len(lens[1]));
        }
        _ => {
            let fcb = lens.len() as u8 | if vbr { 0x80 } else { 0 } | if pad.is_some() { 0x40 } else { 0 };
            let hdr = pad.map(enc_pad).unwrap_or_default();
            for v in [&mut std, &mut sd] {
                v.push(fcb);
                v.extend(&hdr);
            }
            if vbr {
                lens[..lens.len() - 1].iter().for_each(|&l| std.extend(enc_len(l)));
                lens.iter().for_each(|&l| sd.extend(enc_len(l)));
            } else {
                sd.extend(enc_len(lens[0]));
            }
        }
    }
    for v in [&mut std, &mut sd] {
        v.extend(&frames);
        v.extend(&pad_bytes);
    }
    (std, sd)
}

/// A code 0 packet of `len` bytes.
fn code0(toc: u8, len: usize) -> Vec<u8> {
    let mut v = vec![toc & !3];
    v.resize(len, 0x5A);
    v
}

fn frames_at(samples_48k: u64, rate: u32) -> u64 {
    samples_48k * u64::from(rate) / 48_000
}

// ---- tests ----

/// Every TOC byte (32 configs x stereo x codes 0 to 3) with the shortest and longest packets of
/// its code, and code 3 with every frame count byte: mode, bandwidth, frame size and count match
/// Table 2 and the 120 ms limit (RFC 6716 sections 3.1, 3.2.5), in both parser modes.
#[test]
fn every_toc_byte_and_frame_count() {
    for toc in 0..=255u8 {
        let mut packets = match toc & 3 {
            0 => vec![vec![toc], code0(toc, 1276), code0(toc, 1277)],
            1 => vec![vec![toc], code0(toc, 2551), code0(toc, 2553), vec![toc, 0]],
            2 => vec![vec![toc], vec![toc, 0], vec![toc, 1], vec![toc, 1, 7]],
            _ => (0..=255u8).map(|m| vec![toc, m]).collect(),
        };
        packets.iter_mut().filter(|p| p[0] & 3 != 3).for_each(|p| p[0] = toc);
        for p in &packets {
            check_parse(p, false);
            check_parse(p, true);
        }
    }
    // 48 x 2.5 ms and 2 x 60 ms are exactly 120 ms
    assert_eq!(parse_packet(&[16 << 3 | 3, 48], false).unwrap().total_samples_48k, 5760);
    assert!(parse_packet(&[16 << 3 | 3, 49], false).is_none());
    assert_eq!(parse_packet(&[3 << 3 | 1], false).unwrap().total_samples_48k, 5760);
    assert!(parse_packet(&[3 << 3 | 3, 3], false).is_none());
}

/// Frame lengths at the one / two byte boundary (251 / 252) and the 1275 byte maximum in code 2
/// and code 3 VBR, with the implicit last frame up to and past 1275 (RFC 6716 3.2.1, 3.2.4, 3.2.5).
#[test]
fn frame_length_coding_boundaries() {
    let bounds = [0, 1, 250, 251, 252, 253, 254, 255, 256, 1023, 1274, 1275];
    for &a in &bounds {
        for b in [0, 1, 251, 252, 1275, 1276] {
            let mut p = vec![0xFA];
            p.extend(enc_len(a));
            p.resize(p.len() + a + b, 0x33);
            check_parse(&p, false);
            assert_eq!(parse_packet(&p, false).is_some(), b <= 1275, "code 2 {a} + {b}");

            let mut p = vec![0x7B, 0x83];
            p.extend(enc_len(a));
            p.extend(enc_len(a));
            p.resize(p.len() + 2 * a + b, 0x44);
            check_parse(&p, false);
            assert_eq!(parse_packet(&p, false).map(|i| i.frames), (b <= 1275).then_some(3), "VBR {a} {a} {b}");
        }
    }
    // the second length byte of a two byte length is missing
    for b0 in 252..=255u8 {
        assert!(parse_packet(&[0xFA, b0], false).is_none());
    }
    assert!(parse_packet(&code0(0xF8, 1276), false).is_some());
    assert!(parse_packet(&code0(0xF8, 1277), false).is_none());
}

/// Code 3 CBR and VBR with padding of 0 to 2000 bytes, including lengths that take several 255
/// bytes (254 each, RFC 6716 3.2.5), and every RFC 6716 3.4 malformed case R1 to R7 refused.
#[test]
fn code3_padding_and_malformed_packets() {
    for pad in [0, 1, 253, 254, 255, 256, 507, 508, 509, 762, 1000, 2000] {
        for (vbr, frames) in [(false, vec![7, 7, 7]), (true, vec![300, 0, 5]), (true, vec![1275])] {
            let mut p = vec![0x13, 0x40 | frames.len() as u8 | if vbr { 0x80 } else { 0 }];
            p.extend(enc_pad(pad));
            if vbr {
                frames[..frames.len() - 1].iter().for_each(|&l| p.extend(enc_len(l)));
            }
            p.resize(p.len() + frames.iter().sum::<usize>() + pad, 0);
            assert_eq!(ref_frames(&p), Some(frames.clone()), "reference, pad {pad}");
            check_parse(&p, false);
            assert!(parse_packet(&p, false).is_some(), "pad {pad} {frames:?}");
            // one byte fewer: the padding no longer fits or the frames do not divide
            check_parse(&p[..p.len() - 1], false);
        }
    }
    let r2_cbr = [&[0x7B, 0x01][..], &[0x55; 1276]].concat();
    let r2_vbr = [&[0x7B, 0x82, 1][..], &[0; 1277]].concat();
    let malformed: [(&str, &[u8]); 15] = [
        ("R1 empty", &[]),
        ("R2 code 3 CBR frame of 1276", &r2_cbr),
        ("R2 code 3 VBR last frame of 1276", &r2_vbr),
        ("R3 code 1 even length", &[0x79, 1]),
        ("R4 code 2 no length", &[0x7A]),
        ("R4 code 2 length past the end", &[0x7A, 5, 1, 2, 3]),
        ("R4 code 2 two byte length cut", &[0x7A, 253]),
        ("R5 code 3 no frames", &[0x7B, 0x00]),
        ("R5 code 3 180 ms", &[0x1B, 0x03]),
        ("R6 code 3 no count byte", &[0x7B]),
        ("R6 CBR not a multiple", &[0x7B, 0x02, 1, 2, 3]),
        ("R6 padding longer than the packet", &[0x7B, 0x41, 5, 1, 2]),
        ("R6 padding length bytes cut", &[0x7B, 0x41, 255, 255]),
        ("R7 VBR length past the end", &[0x7B, 0x83, 10, 1, 2]),
        ("R7 VBR lengths cut", &[0x7B, 0x83, 1]),
    ];
    for (what, p) in malformed {
        assert!(ref_frames(p).is_none(), "reference accepts {what}");
        assert!(parse_packet(p, false).is_none(), "{what}: {:02x?}", p);
    }
}

/// split_multistream_packet for 1 to 255 streams: streams 0 to N-2 self-delimited (RFC 6716
/// Appendix B), the last undelimited (RFC 7845 5.1.1), each returned as the standard packet;
/// every truncation either refused or, inside the last stream, judged like a standard packet.
#[test]
fn multistream_split_1_to_255_streams() {
    let mut r = Lcg(0x0705_1716);
    for n in 1..=255usize {
        let parts: Vec<_> = (0..n).map(|_| random_packet(&mut r)).collect();
        let mut packet: Vec<u8> = parts[..n - 1].iter().flat_map(|(_, sd)| sd.clone()).collect();
        let last_start = packet.len();
        packet.extend(&parts[n - 1].0);
        let want: Vec<&[u8]> = parts.iter().map(|(s, _)| s.as_slice()).collect();
        let got = split_multistream_packet(&packet, n).unwrap_or_else(|| panic!("{n} streams refused"));
        assert_eq!(got.iter().map(|c| c.as_ref()).collect::<Vec<_>>(), want, "{n} streams");
        let Cow::Borrowed(last) = &got[n - 1] else { panic!("the last stream is copied") };
        assert_eq!(last.as_ptr(), packet[last_start..].as_ptr());
        let more = split_multistream_packet(&packet, n + 1).map(|v| v.iter().map(|c| c.to_vec()).collect());
        assert_eq!(more, ref_split(&packet, n + 1), "{n} streams split into {}", n + 1);
        if n <= 8 {
            for cut in 0..packet.len() {
                let got = split_multistream_packet(&packet[..cut], n);
                let ok = cut > last_start && ref_frames(&packet[last_start..cut]).is_some();
                assert_eq!(got.is_some(), ok, "{n} streams cut at {cut} of {}", packet.len());
            }
        }
    }
    assert!(split_multistream_packet(&[0xF8, 1, 2], 0).is_none());
    // the longest explicit lengths, coded again in the standard packet (RFC 6716 3.2.1)
    for a in [251, 252, 255, 256, 1274, 1275] {
        let frames = vec![0x11; 2 * a];
        let sd = [&[0xFA][..], &enc_len(a), &enc_len(a), &frames].concat();
        let std = [&[0xFA][..], &enc_len(a), &frames].concat();
        let packet = [&sd[..], &[0xF8]].concat();
        let got = split_multistream_packet(&packet, 2).map(|v| v.iter().map(|c| c.to_vec()).collect());
        assert_eq!(got, Some(vec![std, vec![0xF8]]), "code 2 frames of {a}");
    }
}

/// build_opus_head lays out RFC 7845 5.1 (magic, version 1, channels, little-endian pre-skip,
/// rate, gain, family 0 for 1 / 2 channels and 1 for 3 to 8) and parse_opus_head reads it back;
/// channel counts outside 1 to 8 are clamped.
#[test]
fn opus_head_build_parse_round_trip() {
    for ch in 1..=8u8 {
        let fields = [(48_000, 312, 0), (0, 0, i16::MIN), (u32::MAX, u16::MAX, i16::MAX), (44_100, 3840, -1)];
        for (rate, skip, gain) in fields {
            let h = build_opus_head(ch, rate, skip, gain);
            assert_eq!(&h[..10], &[b'O', b'p', b'u', b's', b'H', b'e', b'a', b'd', 1, ch]);
            assert_eq!(&h[10..18], &[&skip.to_le_bytes()[..], &rate.to_le_bytes(), &gain.to_le_bytes()].concat());
            assert_eq!(h.len(), if ch <= 2 { 19 } else { 21 + usize::from(ch) });
            let p = parse_opus_head(&h).unwrap();
            assert_eq!((p.channels, p.pre_skip, p.input_rate, p.output_gain), (ch, skip, rate, gain));
            assert_eq!(p.raw, h);
            assert_eq!(p.mapping.len(), usize::from(ch));
            if ch <= 2 {
                assert_eq!((p.mapping_family, p.stream_count, p.coupled_count), (0, 1, ch - 1));
            } else {
                // each decoded channel is used once (the stream split itself is left to PR #622)
                let mut m = p.mapping.clone();
                m.sort();
                assert_eq!(m, (0..ch).collect::<Vec<_>>());
                assert_eq!((p.mapping_family, p.stream_count + p.coupled_count), (1, ch));
                assert!(p.coupled_count <= p.stream_count && h[19] == p.stream_count && h[20] == p.coupled_count);
            }
        }
    }
    assert_eq!(build_opus_head(0, 48_000, 0, 0), build_opus_head(1, 48_000, 0, 0));
    assert_eq!(build_opus_head(9, 48_000, 0, 0), build_opus_head(8, 48_000, 0, 0));
    assert_eq!(build_opus_head(255, 48_000, 0, 0), build_opus_head(8, 48_000, 0, 0));
}

/// A head written byte by byte: [OpusHead, version, channels, 0 pre-skip, 48 kHz, 0 gain, family]
/// then, for a family other than 0, stream count, coupled count and the mapping.
fn head(version: u8, channels: u8, family: u8, table: &[u8]) -> Vec<u8> {
    let mut h = b"OpusHead".to_vec();
    h.extend([version, channels, 0, 0, 0x80, 0xBB, 0, 0, 0, 0, family]);
    h.extend(table);
    h
}

/// Hand-written heads: families 0, 1 and 255 with silent (255) channels, reserved families 2 to
/// 254 read like 255 (RFC 7845 5.1.1.4), trailing bytes left out of `raw`, and short or invalid
/// heads refused (5.1: version 16 and above, zero channels; 5.1.1: N = 0, M > N, index >= M + N).
#[test]
fn opus_head_families_and_invalid_heads() {
    let h = parse_opus_head(&head(1, 3, 255, &[2, 1, 255, 0, 2])).unwrap();
    assert_eq!((h.mapping_family, h.stream_count, h.coupled_count, h.mapping), (255, 2, 1, vec![255, 0, 2]));
    let h = parse_opus_head(&head(1, 2, 1, &[1, 1, 0, 1])).unwrap();
    assert_eq!((h.mapping_family, h.stream_count, h.coupled_count), (1, 1, 1));
    for family in [2, 3, 128, 254] {
        let h = parse_opus_head(&head(1, 4, family, &[4, 0, 0, 1, 2, 3])).unwrap();
        assert_eq!((h.mapping_family, h.stream_count, h.mapping), (family, 4, vec![0, 1, 2, 3]));
    }
    let long = [head(1, 1, 0, &[]), vec![0xEE; 7]].concat();
    assert_eq!(parse_opus_head(&long).unwrap().raw, head(1, 1, 0, &[]));
    let long = [head(1, 3, 1, &[2, 1, 0, 2, 1]), vec![0xEE; 3]].concat();
    assert_eq!(parse_opus_head(&long).unwrap().raw.len(), 24);

    let bad: [(&str, Vec<u8>); 12] = [
        ("18 bytes", head(1, 1, 0, &[])[..18].to_vec()),
        ("magic", [b"OpusTags", &head(1, 1, 0, &[])[8..]].concat()),
        ("version 16", head(16, 1, 0, &[])),
        ("version 255", head(255, 2, 0, &[])),
        ("zero channels", head(1, 0, 0, &[])),
        ("family 0 with 3 channels", head(1, 3, 0, &[])),
        ("family 1 without a table", head(1, 1, 1, &[])),
        ("mapping cut", head(1, 3, 1, &[2, 1, 0, 2])),
        ("zero streams", head(1, 1, 255, &[0, 0, 255])),
        ("more coupled than streams", head(1, 2, 255, &[1, 2, 0, 1])),
        ("index == M + N", head(1, 2, 1, &[1, 0, 0, 1])),
        ("index 254", head(1, 1, 255, &[1, 1, 254])),
    ];
    for (what, h) in bad {
        assert!(parse_opus_head(&h).is_none(), "{what}");
    }
}

/// Length-prefixed blocks ([u16 LE len][packet]..., zero tail): one block, several blocks of
/// block_align with an all-zero block skipped, the 2048 B default; packets are slices of the
/// input and the estimate is their 48 kHz length scaled down at each output rate.
#[test]
fn length_prefixed_blocks_and_rates() {
    let pkts = [code0(0xF8, 1), code0(0x18, 2), code0(0x78, 252), code0(0x80, 253), vec![0xE1, 1, 2], code0(0x10, 600)];
    let samples: u64 = 960 + 2880 + 960 + 120 + 2 * 120 + 1920;
    let mut block = Vec::new();
    for p in &pkts {
        block.extend((p.len() as u16).to_le_bytes());
        block.extend(p);
    }
    let offsets: Vec<usize> = pkts.iter().scan(2, |o, p| Some(std::mem::replace(o, *o + p.len() + 2))).collect();
    block.resize(4096, 0);
    let three = [&block[..], &[0; 4096], &block].concat();
    for (data, ba, copies) in [(block[..2048].to_vec(), 0u16, 1), (block.clone(), 4096, 1), (three, 4096, 2)] {
        let got = extract_packets(&data, ba, false);
        assert_eq!(got.len(), pkts.len() * copies);
        for (i, g) in got.iter().enumerate() {
            let at = offsets[i % pkts.len()] + (i / pkts.len()) * 8192;
            assert!(matches!(g, Cow::Borrowed(s) if s.as_ptr() == data[at..].as_ptr()), "packet {i}");
            assert_eq!(g.as_ref(), pkts[i % pkts.len()].as_slice());
        }
        for rate in [8_000, 12_000, 16_000, 24_000, 44_100, 48_000, 96_000] {
            let want = frames_at(samples * copies as u64, rate);
            assert_eq!(estimate_buffer_frames(&data, ba, rate, false), Some(want), "{rate} Hz");
        }
    }
    // one [len][packet] that is also a valid raw packet is taken raw; an invalid raw one is not
    assert_eq!(extract_packets(&[1, 0, 0xF8], 0, false), vec![Cow::Borrowed(&[1u8, 0, 0xF8][..])]);
    assert_eq!(extract_packets(&[3, 0, 0xF8, 1, 2], 0, false), vec![Cow::Borrowed(&[0xF8u8, 1, 2][..])]);
    assert!(extract_packets(&[], 0, false).is_empty());
    assert_eq!(estimate_buffer_frames(&[0x7B, 0], 0, 48_000, false), None);
}

/// One RFC 3533 page (CRC included): version 0, header type, granule, serial, sequence, laces.
fn page(out: &mut Vec<u8>, flags: u8, granule: u64, seq: u32, laces: &[u8], body: &[u8]) {
    let start = out.len();
    out.extend(b"OggS\0");
    out.push(flags);
    out.extend(granule.to_le_bytes());
    out.extend(9u32.to_le_bytes());
    out.extend(seq.to_le_bytes());
    out.extend([0; 4]);
    out.push(laces.len() as u8);
    out.extend(laces);
    out.extend(body);
    let mut crc = 0u32;
    for &b in &out[start..] {
        crc ^= u32::from(b) << 24;
        (0..8).for_each(|_| crc = (crc << 1) ^ if crc & 0x8000_0000 != 0 { 0x04C1_1DB7 } else { 0 });
    }
    out[start + 22..start + 26].copy_from_slice(&crc.to_le_bytes());
}

/// Ogg Opus (RFC 7845 3, 4): packets laced with 255s (a 255 byte packet ends in a 0 lace, a 254
/// byte one is one lace), one continued over three pages; packets come back whole, and the EOS
/// granule minus pre-skip gives the length at 48 and 16 kHz. Without an EOS page the packets are
/// summed; a cut page is refused.
#[test]
fn ogg_pages_lacing_and_continued_packets() {
    let mut ogg = Vec::new();
    page(&mut ogg, 0x02, 0, 0, &[19], &build_opus_head(1, 48_000, 312, 0));
    page(&mut ogg, 0, 0, 1, &[16], b"OpusTags\0\0\0\0\0\0\0\0");
    let audio = ogg.len();
    let (p0, p1, p2, p3) = (code0(0xF8, 300), code0(0x18, 255), code0(0x78, 1100), code0(0x80, 254));
    let body = [&p0[..], &p1, &p2[..510]].concat();
    page(&mut ogg, 0, 960 + 2880, 2, &[255, 45, 255, 0, 255, 255], &body);
    page(&mut ogg, 0x01, u64::MAX, 3, &[255, 255], &p2[510..1020]);
    let eos = ogg.len();
    page(&mut ogg, 0x05, 4920, 4, &[80, 254], &[&p2[1020..], &p3[..]].concat());

    let got = extract_packets(&ogg, 0, false);
    assert_eq!(got.iter().map(|c| c.as_ref()).collect::<Vec<_>>(), [&p0[..], &p1, &p2, &p3]);
    assert_eq!(estimate_buffer_frames(&ogg, 0, 48_000, false), Some(4920 - 312));
    assert_eq!(estimate_buffer_frames(&ogg, 0, 16_000, false), Some(1640 - 104));

    // no EOS page: the decoded length of every packet
    let mut open = ogg[..eos].to_vec();
    page(&mut open, 0x01, 4920, 4, &[80, 254], &[&p2[1020..], &p3[..]].concat());
    assert_eq!(estimate_buffer_frames(&open, 0, 48_000, false), Some(4920));
    assert_eq!(estimate_buffer_frames(&open[audio..], 0, 24_000, false), Some(2460));

    // a page that promises more body than there is
    assert!(extract_packets(&ogg[..ogg.len() - 1], 0, false).is_empty());
    assert_eq!(estimate_buffer_frames(&ogg[..ogg.len() - 1], 0, 48_000, false), None);
}

/// Seeded random packets, multistream packets, heads and buffers (raw, length-prefixed, Ogg),
/// valid and mutated: nothing panics, parse_packet and split_multistream_packet agree with the
/// references, and extraction returns only valid packets lying in the input with a consistent
/// length estimate.
#[test]
fn random_sweep_never_panics() {
    let mut r = Lcg(6716_7845);
    for i in 0..6000 {
        let mut d = match i % 4 {
            0 => {
                let n = r.below(48);
                r.bytes(n)
            }
            1 => random_packet(&mut r).0,
            2 => (0..1 + r.below(6)).flat_map(|_| random_packet(&mut r).1).collect(),
            _ => {
                let mut ogg = Vec::new();
                page(&mut ogg, 0x02, 0, 0, &[19], &build_opus_head(2, 48_000, r.below(2000) as u16, 0));
                page(&mut ogg, 0, 0, 1, &[16], b"OpusTags\0\0\0\0\0\0\0\0");
                let p = random_packet(&mut r).0;
                let mut laces = vec![255; p.len() / 255];
                laces.push((p.len() % 255) as u8);
                page(&mut ogg, 0x04, r.below(3000) as u64, 2, &laces, &p);
                ogg
            }
        };
        for _ in 0..r.below(3) {
            if !d.is_empty() {
                let at = r.below(d.len());
                d[at] = r.next() as u8;
            }
        }
        let streams = [1, 2, 3, 6, 8, 255][r.below(6)];
        let (ba, rate, ms) = ([0u16, 64, 2048][r.below(3)], [8_000, 24_000, 48_000][r.below(3)], r.below(2) == 1);
        let ok = catch_unwind(AssertUnwindSafe(|| {
            check_parse(&d, false);
            check_parse(&d, true);
            let split = split_multistream_packet(&d, streams).map(|v| v.iter().map(|c| c.to_vec()).collect());
            assert_eq!(split, ref_split(&d, streams), "split into {streams}");
            if let Some(h) = parse_opus_head(&d) {
                assert!(d.starts_with(&h.raw) && h.channels == d[9] && h.mapping.len() == usize::from(h.channels));
            }
            let pk = extract_packets(&d, ba, ms);
            let mut total = 0u64;
            for p in &pk {
                total += u64::from(parse_packet(p, ms).expect("an extracted packet parses").total_samples_48k);
                if let Cow::Borrowed(s) = p {
                    assert!(d.as_ptr_range().contains(&s.as_ptr()) && s.as_ptr_range().end <= d.as_ptr_range().end);
                }
            }
            let est = estimate_buffer_frames(&d, ba, rate, ms);
            assert_eq!(est.is_some(), !pk.is_empty());
            if d.starts_with(b"OggS") {
                assert!(est.unwrap_or(0) <= frames_at(total, rate));
            } else {
                assert_eq!(est.unwrap_or(0), frames_at(total, rate));
            }
        }));
        assert!(ok.is_ok(), "input {i}: {:02x?} (streams {streams}, block_align {ba}, multistream {ms})", d);
    }
}
