//! The pure format and buffer helpers of `hle::xaudio2` that every title's audio goes through:
//! the SDK struct layouts, Opus / XMA format recognition, the PCM16 stand-in format, and the
//! play / loop region rules a decoded or silent buffer must keep so Windows XAudio2 accepts it.
//! Layouts per mmeapi.h / mmreg.h, xma2defs.h and xaudio2.h (all `#pragma pack(1)`); region
//! rules per the XAUDIO2_BUFFER reference (learn.microsoft.com, xaudio2/ns-xaudio2-xaudio2_buffer).

use std::ffi::c_void;
use std::mem::{offset_of, size_of};
use std::panic::{catch_unwind, AssertUnwindSafe};

use xcl::hle::xaudio2::{
    clamp_regions, estimate_frames, is_plain_pcm, is_xma, opus_format, pcm16_for, pcm16_for_rate, silent_buffer,
    xma_info, xma_stream_config, WaveFormatEx, WaveFormatExtensible, XAudio2Buffer, XAudio2VoiceState, XmaInfo,
    Xma2WaveFormatEx, ENGINE_FLAGS, MASTERING_FLAGS, SUBFORMAT_OPUS, VOICE_FLAGS, WAVE_FORMAT_EXTENSIBLE,
    WAVE_FORMAT_IEEE_FLOAT, WAVE_FORMAT_OPUS, WAVE_FORMAT_PCM, WAVE_FORMAT_XMA, WAVE_FORMAT_XMA2,
};
use xcl::hle::xma2::Version;

/// xaudio2.h: XAUDIO2_MIN_SAMPLE_RATE, XAUDIO2_MAX_SAMPLE_RATE, XAUDIO2_LOOP_INFINITE.
const MIN_RATE: u32 = 1_000;
const MAX_RATE: u32 = 200_000;
const LOOP_INFINITE: u32 = 255;

struct Lcg(u64);
impl Lcg {
    fn next(&mut self) -> u32 {
        self.0 = self.0.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        (self.0 >> 33) as u32
    }
    fn pick(&mut self, v: &[u32]) -> u32 {
        v[self.next() as usize % v.len()]
    }
    /// One of `v`, or now and then a random value.
    fn pick_or(&mut self, v: &[u32]) -> u32 {
        let k = self.next() as usize % (v.len() + 1);
        if k == v.len() { self.next() } else { v[k] }
    }
}

fn bytes_of<T: Copy>(v: &T) -> Vec<u8> {
    unsafe { std::slice::from_raw_parts(v as *const T as *const u8, size_of::<T>()) }.to_vec()
}

/// A WAVEFORMATEX header followed by exactly `cbSize` extra bytes, nothing more.
fn format(tag: u16, channels: u16, rate: u32, extra: &[u8]) -> Vec<u8> {
    let w = WaveFormatEx { tag, channels, rate, avg_bytes: 4000, block_align: 2048, bits: 0, cb_size: extra.len() as u16 };
    let mut v = bytes_of(&w);
    v.extend_from_slice(extra);
    v
}

fn wfx(v: &[u8]) -> *const WaveFormatEx {
    v.as_ptr() as *const WaveFormatEx
}

fn buffer(pb: u32, pl: u32, lb: u32, ll: u32, lc: u32) -> XAudio2Buffer {
    XAudio2Buffer {
        flags: 0x40,
        audio_bytes: 0x1000,
        audio_data: std::ptr::null(),
        play_begin: pb,
        play_length: pl,
        loop_begin: lb,
        loop_length: ll,
        loop_count: lc,
        context: 0xC0DE as *mut c_void,
    }
}

fn regions(b: &XAudio2Buffer) -> (u32, u32, u32, u32, u32) {
    (b.play_begin, b.play_length, b.loop_begin, b.loop_length, b.loop_count)
}

/// What SubmitSourceBuffer accepts for a PCM buffer of `frames` frames (XAUDIO2_BUFFER docs):
/// PlayLength 0 plays the whole buffer and needs PlayBegin 0; the play region fits the buffer;
/// LoopCount 0 needs LoopBegin = LoopLength = 0; LoopBegin < play end; LoopBegin + LoopLength
/// lies in (PlayBegin, play end]; LoopLength 0 loops the whole buffer (LoopBegin 0, full play end).
fn accepts(b: &XAudio2Buffer, frames: u32) -> Result<(), String> {
    let (pb, pl, lb, ll, lc) = regions(b);
    let (pb, pl, lb, ll, frames) = (pb as u64, pl as u64, lb as u64, ll as u64, frames as u64);
    if frames == 0 {
        return Err("empty buffer".into());
    }
    if pl == 0 && pb != 0 {
        return Err("PlayLength 0 with PlayBegin != 0".into());
    }
    if pb + pl > frames {
        return Err("play region past the end".into());
    }
    let play_end = if pl == 0 { frames } else { pb + pl };
    if lc == 0 {
        return if lb == 0 && ll == 0 { Ok(()) } else { Err("loop fields set with LoopCount 0".into()) };
    }
    if lb >= play_end {
        return Err("LoopBegin at or past the play end".into());
    }
    if ll == 0 {
        return if lb == 0 && play_end == frames { Ok(()) } else { Err("LoopLength 0 on a partial region".into()) };
    }
    if lb + ll <= pb || lb + ll > play_end {
        return Err("loop end outside (PlayBegin, play end]".into());
    }
    Ok(())
}

/// Offsets of the named fields of a struct.
macro_rules! offsets {
    ($t:ty: $($f:ident),*) => { [$(offset_of!($t, $f)),*] };
}

/// Sizes and field offsets of the packed SDK structs, written out from the headers.
#[test]
fn struct_layouts_match_the_sdk_headers() {
    // mmeapi.h WAVEFORMATEX
    assert_eq!(size_of::<WaveFormatEx>(), 18);
    let o = offsets!(WaveFormatEx: tag, channels, rate, avg_bytes, block_align, bits, cb_size);
    assert_eq!(o, [0, 2, 4, 8, 12, 14, 16]);
    // mmreg.h WAVEFORMATEXTENSIBLE: cbSize 22 covers Samples + dwChannelMask + SubFormat
    assert_eq!(size_of::<WaveFormatExtensible>(), 40);
    assert_eq!(offsets!(WaveFormatExtensible: samples, channel_mask, sub_format), [18, 20, 24]);
    // xma2defs.h XMA2WAVEFORMATEX: cbSize 34
    assert_eq!(size_of::<Xma2WaveFormatEx>(), 52);
    let o = offsets!(Xma2WaveFormatEx: num_streams, channel_mask, samples_encoded, bytes_per_block, play_begin,
        play_length, loop_begin, loop_length, loop_count, encoder_version, block_count);
    assert_eq!(o, [18, 20, 24, 28, 32, 36, 40, 44, 48, 49, 50]);
    // xaudio2.h XAUDIO2_BUFFER and XAUDIO2_VOICE_STATE (64-bit pointers)
    assert_eq!(size_of::<XAudio2Buffer>(), 44);
    let o = offsets!(XAudio2Buffer: flags, audio_bytes, audio_data, play_begin, play_length, loop_begin, loop_length,
        loop_count, context);
    assert_eq!(o, [0, 4, 8, 16, 20, 24, 28, 32, 36]);
    assert_eq!(size_of::<XAudio2VoiceState>(), 20);
    assert_eq!(offsets!(XAudio2VoiceState: current_buffer_context, buffers_queued, samples_played), [0, 8, 12]);
    // and the bytes are little-endian WAVEFORMATEX: PCM, 2 ch., 48 kHz, 192000 B/s, align 4, 16 bit
    let pcm = pcm16_for_rate(2, 48_000);
    assert_eq!(bytes_of(&pcm), [1, 0, 2, 0, 0x80, 0xBB, 0, 0, 0x00, 0xEE, 0x02, 0, 4, 0, 16, 0, 0, 0]);
}

/// Format tags, the Opus subformat GUID and the flag masks are the header values.
#[test]
fn constants_are_the_header_values() {
    // mmreg.h / xma2defs.h / the Opus GDK headers
    assert_eq!(
        [WAVE_FORMAT_PCM, WAVE_FORMAT_IEEE_FLOAT, WAVE_FORMAT_XMA, WAVE_FORMAT_XMA2, WAVE_FORMAT_OPUS, WAVE_FORMAT_EXTENSIBLE],
        [0x0001, 0x0003, 0x0165, 0x0166, 0x704F, 0xFFFE]
    );
    // DEFINE_WAVEFORMATEX_GUID(tag) = {tag-0000-0010-8000-00AA00389B71}, Data1..3 little-endian
    let mut guid = [0u8; 16];
    guid[..4].copy_from_slice(&(WAVE_FORMAT_OPUS as u32).to_le_bytes());
    guid[4..6].copy_from_slice(&0u16.to_le_bytes());
    guid[6..8].copy_from_slice(&0x0010u16.to_le_bytes());
    guid[8..].copy_from_slice(&[0x80, 0x00, 0x00, 0xAA, 0x00, 0x38, 0x9B, 0x71]);
    assert_eq!(SUBFORMAT_OPUS, guid);
    // xaudio2.h: STOP_ENGINE_WHEN_IDLE | 1024_QUANTUM | NO_VIRTUAL_AUDIO_CLIENT; DEBUG_ENGINE (0x1) is not passed on
    assert_eq!(ENGINE_FLAGS, 0x2000 | 0x8000 | 0x1_0000);
    assert_eq!(ENGINE_FLAGS & 0x1, 0);
    // VOICE_NOPITCH | VOICE_NOSRC | VOICE_USEFILTER; MASTERING: NO_VIRTUAL_AUDIO_CLIENT
    assert_eq!((VOICE_FLAGS, MASTERING_FLAGS), (0x2 | 0x4 | 0x8, 0x1_0000));
    for tag in 0..=u16::MAX {
        assert_eq!(is_xma(tag), tag == 0x165 || tag == 0x166, "{tag:#x}");
        assert_eq!(is_plain_pcm(tag), matches!(tag, 1 | 3 | 0xFFFE), "{tag:#x}");
    }
}

/// Opus as WAVE_FORMAT_OPUS (extra bytes right after the 18-byte header) or as
/// WAVE_FORMAT_EXTENSIBLE with the Opus subformat (extra bytes after the 40-byte struct).
#[test]
fn opus_format_reads_both_spellings_and_nothing_else() {
    let head = [b'O', b'p', b'u', b's', b'H', 1, 2];
    let f = format(WAVE_FORMAT_OPUS, 2, 24_000, &head);
    let of = unsafe { opus_format(wfx(&f)) }.unwrap();
    assert_eq!((of.channels, of.rate, of.block_align, of.avg_bytes, of.extra), (2, 24_000, 2048, 4000, &head[..]));
    let f = format(WAVE_FORMAT_OPUS, 0, 0, &[]);
    let of = unsafe { opus_format(wfx(&f)) }.unwrap();
    assert_eq!((of.channels, of.rate, of.extra.len()), (1, 48_000, 0));
    let f = format(WAVE_FORMAT_OPUS, 255, 48_000, &[]);
    assert_eq!(unsafe { opus_format(wfx(&f)) }.unwrap().channels, 8);

    let ext = |cb: u16, sub: [u8; 16], tail: &[u8]| {
        let e = WaveFormatExtensible {
            wfx: WaveFormatEx { tag: WAVE_FORMAT_EXTENSIBLE, channels: 6, rate: 48_000, avg_bytes: 0, block_align: 0, bits: 0, cb_size: cb },
            samples: 0,
            channel_mask: 0x3F,
            sub_format: sub,
        };
        let mut v = bytes_of(&e);
        v.extend_from_slice(tail);
        v.truncate(18 + cb as usize);
        v
    };
    let f = ext(26, SUBFORMAT_OPUS, &head[..4]);
    let of = unsafe { opus_format(wfx(&f)) }.unwrap();
    assert_eq!((of.channels, of.extra), (6, &head[..4]));
    // cbSize too small to hold SubFormat: not read, not Opus
    let f = ext(21, SUBFORMAT_OPUS, &[]);
    assert!(unsafe { opus_format(wfx(&f)) }.is_none());
    // another subformat (PCM's GUID, or Opus' with one byte changed)
    let mut pcm_guid = SUBFORMAT_OPUS;
    pcm_guid[..2].copy_from_slice(&1u16.to_le_bytes());
    let mut near = SUBFORMAT_OPUS;
    near[15] ^= 1;
    for sub in [pcm_guid, near, [0; 16]] {
        assert!(unsafe { opus_format(wfx(&ext(22, sub, &[]))) }.is_none());
    }
    for tag in [WAVE_FORMAT_PCM, WAVE_FORMAT_XMA2, WAVE_FORMAT_XMA, 0x704E, 0x4F70, 0] {
        assert!(unsafe { opus_format(wfx(&format(tag, 2, 48_000, &head))) }.is_none(), "{tag:#x}");
    }
}

/// xma_info: channels clamped to 1..=8, rate 0 → 48 kHz, XMA2WAVEFORMATEX fields only for
/// WAVE_FORMAT_XMA2 with cbSize >= 34, is_opus exactly when opus_format recognizes the format.
#[test]
fn xma_info_reads_xma2_fields_only_when_present() {
    let x = |tag: u16, cb: u16, channels: u16, rate: u32| Xma2WaveFormatEx {
        wfx: WaveFormatEx { tag, channels, rate, avg_bytes: 777, block_align: 2048, bits: 16, cb_size: cb },
        num_streams: 1,
        samples_encoded: 96_000,
        bytes_per_block: 0x8000,
        block_count: 3,
        ..Default::default()
    };
    // the struct plus whatever cbSize declares beyond it
    let info = |f: &Xma2WaveFormatEx| {
        let mut v = bytes_of(f);
        v.resize(v.len().max(18 + { f.wfx.cb_size } as usize), 0);
        unsafe { xma_info(wfx(&v)) }
    };
    let full = XmaInfo { channels: 2, rate: 44_100, avg_bytes: 777, block_align: 2048, samples_encoded: 96_000, bytes_per_block: 0x8000, block_count: 3, is_opus: false };
    assert_eq!(info(&x(WAVE_FORMAT_XMA2, 34, 2, 44_100)), full);
    assert_eq!(info(&x(WAVE_FORMAT_XMA2, 40, 2, 44_100)), full);
    let bare = XmaInfo { samples_encoded: 0, bytes_per_block: 0, block_count: 0, ..full };
    assert_eq!(info(&x(WAVE_FORMAT_XMA2, 33, 2, 44_100)), bare);
    assert_eq!(info(&x(WAVE_FORMAT_XMA, 34, 2, 44_100)), bare);
    assert_eq!(info(&x(WAVE_FORMAT_OPUS, 34, 2, 44_100)), XmaInfo { is_opus: true, ..bare });
    let i = info(&x(WAVE_FORMAT_XMA2, 34, 0, 0));
    assert_eq!((i.channels, i.rate), (1, 48_000));
    assert_eq!(info(&x(WAVE_FORMAT_XMA2, 34, 64, 1)).channels, 8);
}

/// xma_stream_config: None for every non-XMA tag; XMA2 extradata starts at byte 18 (NumStreams),
/// XMA1 (0x165) goes to the XMAWAVEFORMAT parser; bad channel counts are errors, not panics.
#[test]
fn xma_stream_config_dispatches_by_tag() {
    let cfg = |v: &[u8]| unsafe { xma_stream_config(wfx(v)) };
    for tag in [WAVE_FORMAT_PCM, WAVE_FORMAT_IEEE_FLOAT, WAVE_FORMAT_EXTENSIBLE, WAVE_FORMAT_OPUS, 0x164, 0x167, 0] {
        assert!(cfg(&format(tag, 2, 48_000, &[])).is_none(), "{tag:#x}");
    }
    let xma2wfx = |channels: u16, streams: u16| Xma2WaveFormatEx {
        wfx: WaveFormatEx { tag: WAVE_FORMAT_XMA2, channels, rate: 0, avg_bytes: 0, block_align: 2048, bits: 16, cb_size: 34 },
        num_streams: streams,
        ..Default::default()
    };
    let ok = |f: Xma2WaveFormatEx| cfg(&bytes_of(&f)).unwrap().unwrap();
    let c = ok(xma2wfx(2, 2));
    assert_eq!((c.version, c.sample_rate, c.channels, c.stream_channels), (Version::Xma2, 48_000, 2, vec![1, 1]));
    assert_eq!(ok(xma2wfx(2, 1)).stream_channels, vec![2]);
    assert_eq!(ok(xma2wfx(16, 8)).stream_channels, vec![2; 8]);
    assert!(cfg(&bytes_of(&xma2wfx(2, 9))).unwrap().is_err());
    assert!(cfg(&bytes_of(&xma2wfx(0, 1))).unwrap().is_err());
    assert!(cfg(&bytes_of(&xma2wfx(17, 0))).unwrap().is_err());
    // no extradata: default 2+2+...+1 layout
    assert_eq!(cfg(&format(WAVE_FORMAT_XMA2, 5, 32_000, &[])).unwrap().unwrap().stream_channels, vec![2, 2, 1]);
    let c = cfg(&format(WAVE_FORMAT_XMA, 2, 44_100, &[])).unwrap().unwrap();
    assert_eq!((c.version, c.sample_rate, c.stream_channels), (Version::Xma1, 44_100, vec![2]));
    // XMAWAVEFORMAT: 8-byte header, NumStreams at 4, 20-byte XMASTREAMFORMAT with Channels at 17
    let mut x1 = vec![0u8; 8 + 20 * 2];
    x1[4] = 2;
    x1[8 + 17] = 2;
    x1[28 + 17] = 1;
    let c = cfg(&format(WAVE_FORMAT_XMA, 3, 48_000, &x1)).unwrap().unwrap();
    assert_eq!((c.version, c.stream_channels), (Version::Xma1, vec![2, 1]));
    assert!(cfg(&format(WAVE_FORMAT_XMA, 4, 48_000, &x1)).unwrap().is_err());
}

/// PCM16 stand-in: nBlockAlign = nChannels * 16 / 8, nAvgBytesPerSec = rate * nBlockAlign
/// (mmeapi.h), channels clamped to 1..=8, over XAudio2's whole sample-rate range.
#[test]
fn pcm16_format_follows_the_waveformatex_rules() {
    for ch in 0..=20u16 {
        for rate in [MIN_RATE, 8_000, 11_025, 22_050, 24_000, 32_000, 44_100, 48_000, 96_000, 192_000, MAX_RATE] {
            let p = pcm16_for_rate(ch, rate);
            let c = ch.clamp(1, 8);
            assert_eq!((p.tag, p.channels, p.rate, p.bits, p.cb_size), (WAVE_FORMAT_PCM, c, rate, 16, 0));
            assert_eq!((p.block_align, p.avg_bytes), (c * 2, rate * c as u32 * 2));
            let i = XmaInfo { channels: ch, rate, avg_bytes: 1, block_align: 3, ..Default::default() };
            assert_eq!(bytes_of(&pcm16_for(&i)), bytes_of(&p));
        }
    }
}

/// Documented XAUDIO2_BUFFER cases: LOOP_INFINITE kept, regions past the end, a loop starting
/// before PlayBegin (allowed), total_frames 0, and u32::MAX positions.
#[test]
fn clamp_regions_documented_cases() {
    let cases: [((u32, u32, u32, u32, u32), u32, (u32, u32, u32, u32, u32)); 9] = [
        // already valid: untouched, including XAUDIO2_LOOP_INFINITE and a whole-buffer loop
        ((0, 0, 0, 0, LOOP_INFINITE), 960, (0, 0, 0, 0, LOOP_INFINITE)),
        ((10, 900, 5, 100, LOOP_INFINITE), 960, (10, 900, 5, 100, LOOP_INFINITE)),
        // play region entirely past the end: the whole buffer plays, a loop inside it stays
        ((2000, 10, 100, 50, 3), 960, (0, 0, 100, 50, 3)),
        ((2000, 10, 2000, 5, 3), 960, (0, 0, 0, 0, 0)),
        // PlayBegin at the last frame
        ((959, 0, 0, 0, 0), 960, (959, 1, 0, 0, 0)),
        // u32::MAX play / loop starts
        ((u32::MAX, u32::MAX, u32::MAX, u32::MAX, 1), 960, (0, 0, 0, 0, 0)),
        ((0, 0, u32::MAX, 0, LOOP_INFINITE), 960, (0, 0, 0, 0, 0)),
        // empty decode: nothing to play or loop
        ((5, 10, 5, 10, LOOP_INFINITE), 0, (0, 0, 0, 0, 0)),
        ((u32::MAX - 1, 0, 0, 0, 1), u32::MAX, (u32::MAX - 1, 1, 0, 0, 1)),
    ];
    for ((pb, pl, lb, ll, lc), total, want) in cases {
        let mut b = buffer(pb, pl, lb, ll, lc);
        clamp_regions(&mut b, total);
        assert_eq!(regions(&b), want, "({pb}, {pl}, {lb}, {ll}, {lc}) in {total} frames");
        assert_eq!(({ b.flags }, { b.audio_bytes }, { b.context } as usize), (0x40, 0x1000, 0xC0DE));
    }
}

/// Over a seeded sweep of edge and random regions: never panics, always yields a region the
/// reference model accepts, leaves valid regions and every non-region field alone, is idempotent.
#[test]
fn clamp_regions_always_yields_a_region_xaudio2_accepts() {
    let mut r = Lcg(0x5EED_A0D1);
    for n in 0..60_000 {
        let total = match n % 4 {
            0 => r.next() % 64,
            1 => r.next() % 100_000,
            2 => r.pick(&[1, 2, 960, u32::MAX - 1, u32::MAX]),
            _ => r.next(),
        };
        let v = |r: &mut Lcg| match r.next() % 5 {
            0 => r.pick(&[0, 1, 2]),
            1 => total.wrapping_add(r.pick(&[0, 1, 2, u32::MAX, u32::MAX - 1])),
            2 => r.pick(&[u32::MAX, u32::MAX - 1, 0x8000_0000]),
            3 => r.next() % total.max(1).saturating_add(2),
            _ => r.next(),
        };
        let (pb, pl, lb, ll) = (v(&mut r), v(&mut r), v(&mut r), v(&mut r));
        let lc = r.pick(&[0, 0, 1, 2, 254, LOOP_INFINITE]);
        let input = buffer(pb, pl, lb, ll, lc);
        let mut b = input;
        catch_unwind(AssertUnwindSafe(|| clamp_regions(&mut b, total)))
            .unwrap_or_else(|_| panic!("clamp_regions panicked on {:?} in {total} frames", regions(&input)));
        let ctx = format!("{:?} in {total} frames -> {:?}", regions(&input), regions(&b));
        assert_eq!(({ b.flags }, { b.audio_bytes }, { b.context } as usize), (0x40, 0x1000, 0xC0DE), "{ctx}");
        assert!({ b.loop_count } == lc || { b.loop_count } == 0, "{ctx}");
        if total == 0 {
            assert_eq!(regions(&b), (0, 0, 0, 0, 0), "{ctx}");
            continue;
        }
        accepts(&b, total).unwrap_or_else(|e| panic!("{e}: {ctx}"));
        if accepts(&input, total).is_ok() {
            assert_eq!(regions(&b), regions(&input), "valid region changed: {ctx}");
        }
        let mut again = b;
        clamp_regions(&mut again, total);
        assert_eq!(regions(&again), regions(&b), "not idempotent: {ctx}");
    }
}

/// estimate_frames: never under 128 frames, each of its three ratios at their u32 limits
/// (computed here in u128), and partial / zero-byte buffers.
#[test]
fn estimate_frames_edges() {
    let xma = XmaInfo { channels: 2, rate: 48_000, samples_encoded: u32::MAX, bytes_per_block: u32::MAX, block_count: u16::MAX, ..Default::default() };
    let whole = u32::MAX as u128 * u16::MAX as u128;
    for bytes in [0, 1, 2048, 0x10_0000, u32::MAX] {
        let want = (u32::MAX as u128 * bytes as u128 / whole).max(128) as u64;
        assert_eq!(estimate_frames(&xma, bytes), want, "{bytes}");
    }
    // the whole XMA2 stream decodes to SamplesEncoded
    let x = XmaInfo { samples_encoded: 123_456, bytes_per_block: 0x8000, block_count: 5, avg_bytes: 1, ..xma };
    assert_eq!(estimate_frames(&x, 5 * 0x8000), 123_456);
    assert_eq!(estimate_frames(&x, 2048), 123_456 * 2048 / (5 * 0x8000));
    // half the XMA2 metadata (no BlockCount, or no BytesPerBlock): the byte rate is used
    for i in [XmaInfo { block_count: 0, ..x }, XmaInfo { bytes_per_block: 0, ..x }] {
        assert_eq!(estimate_frames(&i, 1000), 1000 * 48_000);
    }
    let avg = XmaInfo { channels: 1, rate: u32::MAX, avg_bytes: 1, ..Default::default() };
    assert_eq!(estimate_frames(&avg, u32::MAX), u32::MAX as u64 * u32::MAX as u64);
    // no rate information: 4 frames per byte over the channels; 0 channels counts as 1
    for (ch, want) in [(0, 4 * u32::MAX as u64), (1, 4 * u32::MAX as u64), (8, u32::MAX as u64 / 2)] {
        assert_eq!(estimate_frames(&XmaInfo { channels: ch, ..Default::default() }, u32::MAX), want);
    }
    assert_eq!(estimate_frames(&XmaInfo { channels: 8, ..Default::default() }, 255), 128);
    // more bytes never means fewer frames
    let mut r = Lcg(7);
    for _ in 0..5_000 {
        let i = XmaInfo { channels: r.next() as u16 % 9, rate: r.next(), avg_bytes: r.pick_or(&[0, 1]), samples_encoded: r.pick_or(&[0]), bytes_per_block: r.pick_or(&[0]), block_count: r.next() as u16, ..Default::default() };
        let (a, b) = (r.next(), r.next());
        assert!(estimate_frames(&i, a.min(b)) <= estimate_frames(&i, a.max(b)), "{i:?} {a} {b}");
    }
}

/// silent_buffer: the length follows the play region, else the Opus packet durations, else the
/// estimate; it is a whole number of PCM16 frames, never over max_bytes nor
/// XAUDIO2_MAX_BUFFER_BYTES (0x80000000); flags, context and fitting regions are kept.
#[test]
fn silent_buffer_lengths_and_kept_fields() {
    let z = 0x5000 as *const u8;
    let six = XmaInfo { channels: 6, rate: 48_000, avg_bytes: 8000, ..Default::default() };
    let tiny = XmaInfo { avg_bytes: 4_800_000, ..six };
    // 1000 bytes hold 83 six-channel frames (996 bytes); the region is reset to the whole buffer
    let b = silent_buffer(&six, &buffer(0, 5000, 0, 0, 0), z, 1000);
    assert_eq!(({ b.audio_bytes }, regions(&b), b.audio_data), (996, (0, 0, 0, 0, 0), z));
    // plenty of room: exactly the play region, everything else as the title submitted it
    let mut t = buffer(100, 400, 150, 200, LOOP_INFINITE);
    t.flags = 0x40 | 0x8000_0000;
    let b = silent_buffer(&six, &t, z, 1 << 20);
    assert_eq!(({ b.audio_bytes }, regions(&b)), (500 * 12, (100, 400, 150, 200, LOOP_INFINITE)));
    assert_eq!(({ b.flags }, { b.context } as usize), (0x40 | 0x8000_0000, 0xC0DE));
    // smaller than one frame: empty
    let b = silent_buffer(&six, &buffer(0, 10, 0, 0, 1), z, 11);
    assert_eq!(({ b.audio_bytes }, regions(&b)), (0, (0, 0, 0, 0, 0)));
    // an unbounded silence source still stops at XAUDIO2_MAX_BUFFER_BYTES
    let eight = XmaInfo { channels: 8, ..six };
    let b = silent_buffer(&eight, &buffer(0, u32::MAX, 0, 0, 1), z, usize::MAX);
    assert_eq!(({ b.audio_bytes }, regions(&b)), (0x8000_0000, (0, 0, 0, 0, 1)));
    // PlayBegin past the estimate (PlayLength 0): the buffer reaches one frame past it
    let b = silent_buffer(&tiny, &buffer(10_000, 0, 0, 0, 0), z, 1 << 20);
    assert_eq!(({ b.audio_bytes }, regions(&b)), (10_001 * 12, (10_000, 1, 0, 0, 0)));

    // Opus: RFC 6716 code-0 packet, config 1 (SILK NB 20 ms = 960 samples at 48 kHz)
    let mut one = vec![0x08u8];
    one.extend([0xFF; 10]);
    let opus = |rate: u32| XmaInfo { channels: 1, rate, avg_bytes: 8000, block_align: 0, is_opus: true, ..Default::default() };
    let on = |data: &[u8], pl: u32| XAudio2Buffer { audio_data: data.as_ptr(), audio_bytes: data.len() as u32, ..buffer(0, pl, 0, 0, 0) };
    assert_eq!({ silent_buffer(&opus(48_000), &on(&one, 0), z, 1 << 20).audio_bytes }, 960 * 2);
    assert_eq!({ silent_buffer(&opus(24_000), &on(&one, 0), z, 1 << 20).audio_bytes }, 480 * 2);
    // [u16 len][packet] twice: 40 ms
    let two = [3, 0, 0x08, 1, 2, 3, 0, 0x08, 3, 4];
    assert_eq!({ silent_buffer(&opus(48_000), &on(&two, 0), z, 1 << 20).audio_bytes }, 1920 * 2);
    // an explicit play region wins over the packet durations
    assert_eq!({ silent_buffer(&opus(48_000), &on(&one, 300), z, 1 << 20).audio_bytes }, 300 * 2);
    // not Opus (code 3 with 0 frames), or no data pointer: the byte-rate estimate (>= 128 frames)
    let bad = [0x0B, 0x00, 0x00];
    assert_eq!({ silent_buffer(&opus(48_000), &on(&bad, 0), z, 1 << 20).audio_bytes }, 128 * 2);
    let null = XAudio2Buffer { audio_bytes: 8000, ..buffer(0, 0, 0, 0, 0) };
    assert_eq!({ silent_buffer(&opus(48_000), &null, z, 1 << 20).audio_bytes }, 48_000 * 2);
}

/// Seeded random formats through opus_format / xma_stream_config / xma_info / pcm16_for and
/// random buffers through silent_buffer: no panics, and every silent buffer is whole PCM16
/// frames within max_bytes with a region the reference model accepts.
#[test]
fn random_formats_and_buffers_never_panic() {
    let mut r = Lcg(0xA0D10_F0E3);
    let tags = [1, 3, 0x165, 0x166, 0x704F, 0xFFFE, 0x0002, 0x0161, 0];
    for n in 0..4_000 {
        let tag = if n % 8 == 7 { r.next() as u16 } else { r.pick(&tags) as u16 };
        let cb = (r.pick_or(&[0, 1, 2, 21, 22, 26, 33, 34, 40, 48]) % 300) as usize;
        let mut f: Vec<u8> = (0..18 + cb).map(|_| r.next() as u8).collect();
        f[..2].copy_from_slice(&tag.to_le_bytes());
        f[2..4].copy_from_slice(&(r.pick(&[0, 1, 2, 6, 8, 9, 16, 17, 255, 0xFFFF]) as u16).to_le_bytes());
        f[4..8].copy_from_slice(&r.pick(&[0, 1, MIN_RATE, 48_000, MAX_RATE, u32::MAX]).to_le_bytes());
        f[16..18].copy_from_slice(&(cb as u16).to_le_bytes());
        if tag == WAVE_FORMAT_EXTENSIBLE && cb >= 22 && r.next() % 2 == 0 {
            f[24..40].copy_from_slice(&SUBFORMAT_OPUS);
        }
        let res = catch_unwind(AssertUnwindSafe(|| unsafe {
            let p = wfx(&f);
            let of = opus_format(p);
            let _ = xma_stream_config(p);
            let i = xma_info(p);
            assert_eq!(i.is_opus, of.is_some());
            assert!((1..=8).contains(&i.channels) && i.rate != 0);
            // rates past XAudio2's range: see pcm16_for_rate_survives_out_of_range_rates
            if i.rate <= MAX_RATE {
                let p = pcm16_for(&i);
                assert_eq!({ p.block_align }, i.channels * 2);
            }
            i
        }));
        let Ok(i) = res else { panic!("format helpers panicked on {f:02x?}") };

        let data: Vec<u8> = (0..r.next() % 4096).map(|_| r.next() as u8).collect();
        let mut v = || r.pick_or(&[0, 1, 100, 960, 48_000, 0x7FFF_FFFF, u32::MAX]);
        let (pb, pl, lb, ll) = (v(), v(), v(), v());
        let mut b = buffer(pb, pl, lb, ll, r.pick(&[0, 1, 3, 254, LOOP_INFINITE]));
        b.audio_data = data.as_ptr();
        b.audio_bytes = data.len() as u32;
        let max = r.pick(&[0, 3, 1000, 1 << 20, 32 << 20, u32::MAX]) as usize;
        let ctx = format!("{i:?} {:?} audio_bytes {} max_bytes {max}", regions(&b), data.len());
        let s = catch_unwind(AssertUnwindSafe(|| silent_buffer(&i, &b, 0x5000 as *const u8, max)))
            .unwrap_or_else(|_| panic!("silent_buffer panicked on {ctx}"));
        let fb = i.channels as u32 * 2;
        assert!({ s.audio_bytes } % fb == 0 && { s.audio_bytes } as usize <= max && { s.audio_bytes } <= 0x8000_0000, "{ctx}");
        assert_eq!(({ s.flags }, { s.context } as usize), (0x40, 0xC0DE), "{ctx}");
        if s.audio_bytes > 0 {
            accepts(&s, s.audio_bytes / fb).unwrap_or_else(|e| panic!("{e}: {ctx} -> {:?}", regions(&s)));
        }
        // a valid explicit play region that fits is kept as is
        let fits = (pb as u64 + pl as u64) * fb as u64 <= max.min(0x8000_0000) as u64;
        if pl > 0 && fits && pb.checked_add(pl).is_some_and(|end| accepts(&b, end).is_ok()) {
            assert_eq!(regions(&s), regions(&b), "{ctx}");
        }
    }
}
