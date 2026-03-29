//! Ternsig Audio Experiment
//!
//! Tests whether audio can survive a round-trip through ternary signal encoding:
//!   PCM → PackedSignal → PCM
//!
//! The question: is 1-byte-per-sample ternary encoding viable for audio,
//! and is representational decomposition (p × m × k) more efficient than
//! raw sample depth?

mod codec;

use std::env;
use std::fs::File;
use std::path::Path;
use std::time::Instant;

use symphonia::core::audio::SampleBuffer;
use symphonia::core::codecs::DecoderOptions;
use symphonia::core::formats::FormatOptions;
use symphonia::core::io::MediaSourceStream;
use symphonia::core::meta::MetadataOptions;
use symphonia::core::probe::Hint;

use ternary_signal::PackedSignal;

// ---------------------------------------------------------------------------
// Encoding: PCM i16 → PackedSignal
// ---------------------------------------------------------------------------

/// Build a lookup table mapping each of the 256 possible PackedSignal bytes
/// to their decoded current value. Used for finding nearest encodings.
fn build_decode_lut() -> [i32; 256] {
    let mut lut = [0i32; 256];
    for i in 0..256u16 {
        lut[i as usize] = PackedSignal::from_raw(i as u8).current();
    }
    lut
}

/// Build sorted list of (current_value, packed_byte) for binary search encoding.
/// Only includes distinct current values, deduped to the first byte that produces each.
fn build_encode_table(decode_lut: &[i32; 256]) -> Vec<(i32, u8)> {
    let mut pairs: Vec<(i32, u8)> = (0..=255u8)
        .map(|b| (decode_lut[b as usize], b))
        .collect();
    // Sort by current value
    pairs.sort_by_key(|&(cur, _)| cur);
    // Deduplicate by current value (keep first byte for each)
    pairs.dedup_by_key(|p| p.0);
    pairs
}

/// Encode a PCM i16 sample to the nearest PackedSignal byte.
///
/// Maps the i16 range [-32768, 32767] into the PackedSignal current range [-65025, 65025],
/// then finds the nearest representable value via binary search.
fn encode_sample(sample: i16, encode_table: &[(i32, u8)]) -> u8 {
    // Scale i16 to the full current range: sample/32768 * 65025
    let target: i32 = (sample as i64 * 65025 / 32768) as i32;

    // Binary search for nearest
    match encode_table.binary_search_by_key(&target, |&(cur, _)| cur) {
        Ok(idx) => encode_table[idx].1,
        Err(idx) => {
            if idx == 0 {
                encode_table[0].1
            } else if idx >= encode_table.len() {
                encode_table[encode_table.len() - 1].1
            } else {
                let lo = encode_table[idx - 1];
                let hi = encode_table[idx];
                if (target - lo.0).abs() <= (target - hi.0).abs() {
                    lo.1
                } else {
                    hi.1
                }
            }
        }
    }
}

/// Decode a PackedSignal byte back to PCM i16.
fn decode_sample(byte: u8) -> i16 {
    let current = PackedSignal::from_raw(byte).current();
    // Scale current range [-65025, 65025] back to i16 [-32768, 32767]
    ((current as i64 * 32767) / 65025) as i16
}

// ---------------------------------------------------------------------------
// Smoothing: post-decode interpolation to soften quantization steps
// ---------------------------------------------------------------------------

/// Linear interpolation between consecutive decoded samples.
/// For each original sample, inserts intermediate values so transitions
/// between quantization levels are gradual, not stepped.
fn smooth_linear(samples: &[i16], passes: u32) -> Vec<i16> {
    let mut buf = samples.to_vec();
    for _ in 0..passes {
        let prev = buf.clone();
        for i in 1..buf.len() - 1 {
            // 3-tap weighted average: 25% prev, 50% self, 25% next
            let avg = (prev[i - 1] as i32 + 2 * prev[i] as i32 + prev[i + 1] as i32) / 4;
            buf[i] = avg as i16;
        }
    }
    buf
}

/// Cubic Hermite interpolation — same algorithm ternsig-oscillator uses.
/// Smooths using 4-point neighborhoods for natural-feeling curves.
fn smooth_cubic(samples: &[i16], passes: u32) -> Vec<i16> {
    let mut buf = samples.to_vec();
    for _ in 0..passes {
        let prev = buf.clone();
        for i in 2..buf.len() - 2 {
            let p0 = prev[i - 1] as f64;
            let p1 = prev[i] as f64;
            let p2 = prev[i + 1] as f64;
            let p3 = prev[i + 2] as f64;

            // Catmull-Rom at t=0.5 (midpoint tendency)
            // This pulls the sample toward a smooth curve through its neighbors
            let t = 0.5;
            let t2 = t * t;
            let t3 = t2 * t;

            let v = 0.5
                * ((2.0 * p1)
                    + (-p0 + p2) * t
                    + (2.0 * p0 - 5.0 * p1 + 4.0 * p2 - p3) * t2
                    + (-p0 + 3.0 * p1 - 3.0 * p2 + p3) * t3);

            // Blend: 70% original, 30% interpolated to preserve transients
            let blended = 0.7 * prev[i] as f64 + 0.3 * v;
            buf[i] = blended.clamp(-32768.0, 32767.0) as i16;
        }
    }
    buf
}

// ---------------------------------------------------------------------------
// MP3 decoding
// ---------------------------------------------------------------------------

struct AudioData {
    samples: Vec<i16>,
    sample_rate: u32,
    channels: u16,
}

fn decode_mp3(path: &Path) -> AudioData {
    let file = File::open(path).expect("Failed to open input file");
    let mss = MediaSourceStream::new(Box::new(file), Default::default());

    let mut hint = Hint::new();
    hint.with_extension("mp3");

    let probed = symphonia::default::get_probe()
        .format(
            &hint,
            mss,
            &FormatOptions::default(),
            &MetadataOptions::default(),
        )
        .expect("Failed to probe MP3");

    let mut format = probed.format;
    let track = format.default_track().expect("No audio track found");
    let track_id = track.id;
    let sample_rate = track
        .codec_params
        .sample_rate
        .expect("No sample rate in track");
    let channels = track
        .codec_params
        .channels
        .map(|c| c.count() as u16)
        .unwrap_or(2);

    let mut decoder = symphonia::default::get_codecs()
        .make(&track.codec_params, &DecoderOptions::default())
        .expect("Failed to create decoder");

    let mut all_samples: Vec<i16> = Vec::new();

    loop {
        let packet = match format.next_packet() {
            Ok(p) => p,
            Err(_) => break,
        };
        if packet.track_id() != track_id {
            continue;
        }
        let decoded = match decoder.decode(&packet) {
            Ok(d) => d,
            Err(_) => continue,
        };

        let spec = *decoded.spec();
        let duration = decoded.capacity();
        let mut sample_buf = SampleBuffer::<i16>::new(duration as u64, spec);
        sample_buf.copy_interleaved_ref(decoded);
        all_samples.extend_from_slice(sample_buf.samples());
    }

    AudioData {
        samples: all_samples,
        sample_rate,
        channels,
    }
}

// ---------------------------------------------------------------------------
// WAV writing
// ---------------------------------------------------------------------------

fn write_wav(path: &Path, samples: &[i16], sample_rate: u32, channels: u16) {
    let spec = hound::WavSpec {
        channels,
        sample_rate,
        bits_per_sample: 16,
        sample_format: hound::SampleFormat::Int,
    };
    let mut writer = hound::WavWriter::create(path, spec).expect("Failed to create WAV");
    for &s in samples {
        writer.write_sample(s).expect("Failed to write sample");
    }
    writer.finalize().expect("Failed to finalize WAV");
}

// ---------------------------------------------------------------------------
// Analysis
// ---------------------------------------------------------------------------

fn compute_snr(original: &[i16], reconstructed: &[i16]) -> f64 {
    let mut signal_power: f64 = 0.0;
    let mut noise_power: f64 = 0.0;

    for (&orig, &recon) in original.iter().zip(reconstructed.iter()) {
        let s = orig as f64;
        let n = (orig as f64) - (recon as f64);
        signal_power += s * s;
        noise_power += n * n;
    }

    if noise_power == 0.0 {
        return f64::INFINITY;
    }
    10.0 * (signal_power / noise_power).log10()
}

fn count_distinct_values(encoded: &[u8]) -> usize {
    let mut seen = [false; 256];
    for &b in encoded {
        seen[b as usize] = true;
    }
    seen.iter().filter(|&&v| v).count()
}

/// Histogram of how many of the 256 possible PackedSignal values are actually used.
fn print_value_distribution(encoded: &[u8]) {
    let mut counts = [0u64; 256];
    for &b in encoded {
        counts[b as usize] += 1;
    }

    // Find top 20 most-used values
    let mut indexed: Vec<(u8, u64)> = counts
        .iter()
        .enumerate()
        .filter(|(_, &c)| c > 0)
        .map(|(i, &c)| (i as u8, c))
        .collect();
    indexed.sort_by(|a, b| b.1.cmp(&a.1));

    println!("\n  Top 20 PackedSignal values used:");
    println!("  {:>6}  {:>8}  {:>10}  {:>8}", "Byte", "Current", "Count", "Pct");
    println!("  {}  {}  {}  {}", "-".repeat(6), "-".repeat(8), "-".repeat(10), "-".repeat(8));
    let total = encoded.len() as f64;
    for &(byte, count) in indexed.iter().take(20) {
        let current = PackedSignal::from_raw(byte).current();
        let pct = (count as f64 / total) * 100.0;
        println!("  0x{:02X}    {:>8}  {:>10}  {:>7.2}%", byte, current, count, pct);
    }
}

// ---------------------------------------------------------------------------
// Compression analysis
// ---------------------------------------------------------------------------

fn print_compression_analysis(encoded: &[u8]) {
    let total = encoded.len();

    // --- 6-bit packing (41 values fit in 6 bits) ---
    let six_bit_size = (total * 6 + 7) / 8;

    // --- Delta encoding analysis ---
    let mut delta_counts: std::collections::HashMap<i32, u64> = std::collections::HashMap::new();
    let mut zero_deltas: u64 = 0;
    let mut small_deltas: u64 = 0; // fits in 4 bits signed (-8..+7)

    for i in 1..encoded.len() {
        let prev = PackedSignal::from_raw(encoded[i - 1]).current();
        let curr = PackedSignal::from_raw(encoded[i]).current();
        let delta = curr - prev;
        *delta_counts.entry(delta).or_insert(0) += 1;
        if delta == 0 {
            zero_deltas += 1;
        }
    }

    // Compute delta distribution by magnitude
    let total_deltas = (total - 1) as f64;
    let distinct_deltas = delta_counts.len();

    // Sort deltas by frequency
    let mut sorted_deltas: Vec<(i32, u64)> = delta_counts.into_iter().collect();
    sorted_deltas.sort_by(|a, b| b.1.cmp(&a.1));

    // Count how many deltas fall within various bit widths
    // Map deltas to LUT-step space for tighter encoding
    let mut step_counts: std::collections::HashMap<i16, u64> = std::collections::HashMap::new();
    for i in 1..encoded.len() {
        // Delta in raw byte space
        let prev = encoded[i - 1];
        let curr = encoded[i];
        let byte_delta = curr as i16 - prev as i16;
        *step_counts.entry(byte_delta).or_insert(0) += 1;
        if byte_delta.abs() <= 7 {
            small_deltas += 1;
        }
    }

    // Shannon entropy of raw bytes (theoretical minimum bits/sample)
    let mut byte_counts = [0u64; 256];
    for &b in encoded {
        byte_counts[b as usize] += 1;
    }
    let entropy_raw = {
        let t = total as f64;
        let mut h = 0.0f64;
        for &c in &byte_counts {
            if c > 0 {
                let p = c as f64 / t;
                h -= p * p.log2();
            }
        }
        h
    };

    // Shannon entropy of byte deltas
    let mut byte_delta_counts: std::collections::HashMap<i16, u64> = std::collections::HashMap::new();
    for i in 1..encoded.len() {
        let d = encoded[i] as i16 - encoded[i - 1] as i16;
        *byte_delta_counts.entry(d).or_insert(0) += 1;
    }
    let entropy_delta = {
        let t = total_deltas;
        let mut h = 0.0f64;
        for &c in byte_delta_counts.values() {
            if c > 0 {
                let p = c as f64 / t;
                h -= p * p.log2();
            }
        }
        h
    };

    // Huffman estimate for raw bytes (entropy + ~5% overhead)
    let huffman_raw_bps = entropy_raw * 1.05;
    let huffman_delta_bps = entropy_delta * 1.05;

    // Projected sizes
    let huffman_raw_bytes = ((total as f64 * huffman_raw_bps) / 8.0) as u64;
    let huffman_delta_bytes = ((total as f64 * huffman_delta_bps) / 8.0) as u64;
    let pcm_bytes = (total * 2) as u64;
    let mp3_approx = pcm_bytes * 44 / 120; // rough ratio from our test file

    println!("\n  === COMPRESSION ANALYSIS ===\n");

    println!("  Shannon entropy:");
    println!("    Raw bytes:    {:.2} bits/sample", entropy_raw);
    println!("    Byte deltas:  {:.2} bits/sample", entropy_delta);

    println!("\n  Delta encoding stats:");
    println!("    Zero deltas (identical adjacent): {:>10} ({:.1}%)", zero_deltas, zero_deltas as f64 / total_deltas * 100.0);
    println!("    Small deltas (±7 byte-steps):     {:>10} ({:.1}%)", small_deltas, small_deltas as f64 / total_deltas * 100.0);
    println!("    Distinct delta values:            {:>10}", distinct_deltas);

    println!("\n  Top 15 byte-deltas:");
    println!("    {:>8}  {:>10}  {:>8}", "Delta", "Count", "Pct");
    println!("    {}  {}  {}", "-".repeat(8), "-".repeat(10), "-".repeat(8));
    let mut sorted_byte_deltas: Vec<(i16, u64)> = byte_delta_counts.into_iter().collect();
    sorted_byte_deltas.sort_by(|a, b| b.1.cmp(&a.1));
    for &(delta, count) in sorted_byte_deltas.iter().take(15) {
        let pct = count as f64 / total_deltas * 100.0;
        println!("    {:>+8}  {:>10}  {:>7.2}%", delta, count, pct);
    }

    println!("\n  Projected stream sizes ({}s audio):", total as f64 / 44100.0);
    println!("    {:.<40} {:>10} bytes  ({:.1} bits/sample)", "PCM i16 (baseline)", pcm_bytes, 16.0);
    println!("    {:.<40} {:>10} bytes  ({:.1} bits/sample)", "MP3 (approx from source)", mp3_approx, mp3_approx as f64 * 8.0 / total as f64);
    println!("    {:.<40} {:>10} bytes  ({:.1} bits/sample)", "Ternsig raw (current)", total as u64, 8.0);
    println!("    {:.<40} {:>10} bytes  ({:.1} bits/sample)", "Ternsig 6-bit packed", six_bit_size as u64, 6.0);
    println!("    {:.<40} {:>10} bytes  ({:.1} bits/sample)", "Ternsig + Huffman (raw)", huffman_raw_bytes, huffman_raw_bps);
    println!("    {:.<40} {:>10} bytes  ({:.1} bits/sample)", "Ternsig + Huffman (delta)", huffman_delta_bytes, huffman_delta_bps);

    println!("\n  Livestream bandwidth at 44.1 kHz mono:");
    println!("    {:.<40} {:>8.1} kbps", "PCM i16", 44100.0 * 16.0 / 1000.0);
    println!("    {:.<40} {:>8.1} kbps", "MP3 (typical)", mp3_approx as f64 * 8.0 / (total as f64 / 44100.0) / 1000.0);
    println!("    {:.<40} {:>8.1} kbps", "Ternsig raw", 44100.0 * 8.0 / 1000.0);
    println!("    {:.<40} {:>8.1} kbps", "Ternsig 6-bit", 44100.0 * 6.0 / 1000.0);
    println!("    {:.<40} {:>8.1} kbps", "Ternsig Huffman (raw)", 44100.0 * huffman_raw_bps / 1000.0);
    println!("    {:.<40} {:>8.1} kbps", "Ternsig Huffman (delta)", 44100.0 * huffman_delta_bps / 1000.0);
}

// ---------------------------------------------------------------------------
// Frame envelope split analysis
// ---------------------------------------------------------------------------

fn print_envelope_analysis(encoded: &[u8], original_pcm: &[i16]) {
    // Analyze the m×k decomposition approach:
    // - Magnitude (envelope) transmitted once per frame
    // - Polarity per sample (1 bit)
    // - Multiplier delta per sample (corrections)

    println!("\n  === FRAME ENVELOPE SPLIT ANALYSIS ===\n");

    for &frame_size in &[32u32, 64, 128, 256] {
        let num_frames = (encoded.len() as u32 + frame_size - 1) / frame_size;
        let mut total_bits: u64 = 0;

        // Per frame:
        //   - 3 bits: magnitude code (envelope)
        //   - Per sample: 2 bits polarity + 3 bits multiplier code = 5 bits
        //   But if we delta-encode the mul codes within a frame...
        let mut mul_deltas_zero: u64 = 0;
        let mut mul_deltas_total: u64 = 0;
        let mut pol_flips: u64 = 0;
        let mut pol_total: u64 = 0;

        for frame_start in (0..encoded.len()).step_by(frame_size as usize) {
            let frame_end = (frame_start + frame_size as usize).min(encoded.len());
            let frame = &encoded[frame_start..frame_end];

            // Envelope: max magnitude code in the frame
            let _envelope_mag = frame.iter()
                .map(|&b| PackedSignal::from_raw(b).mag_code())
                .max()
                .unwrap_or(0);

            // Per-sample costs
            // Polarity: 1 bit if it rarely changes within a frame, 2 bits always
            let mut prev_pol = PackedSignal::from_raw(frame[0]).polarity();
            let mut prev_mul = PackedSignal::from_raw(frame[0]).mul_code();
            for &b in &frame[1..] {
                let p = PackedSignal::from_raw(b);
                let pol = p.polarity();
                let mul = p.mul_code();

                pol_total += 1;
                if pol != prev_pol {
                    pol_flips += 1;
                }

                mul_deltas_total += 1;
                if mul == prev_mul {
                    mul_deltas_zero += 1;
                }

                prev_pol = pol;
                prev_mul = mul;
            }

            // Cost model for this frame:
            // Header: 3 bits (envelope mag code)
            // Per sample: 2 bits (polarity) + variable mul
            let frame_len = frame.len() as u64;
            let header_bits: u64 = 3;
            let pol_bits: u64 = frame_len * 2; // always 2 bits for safety
            // Mul: if same as prev, 1 bit (flag=0). If different, 1+3 bits (flag=1 + code)
            // Estimate from actual data in this frame
            let mut mul_bits: u64 = 0;
            let mut pm = PackedSignal::from_raw(frame[0]).mul_code();
            mul_bits += 3; // first sample: full mul code
            for &b in &frame[1..] {
                let mc = PackedSignal::from_raw(b).mul_code();
                if mc == pm {
                    mul_bits += 1; // just the "same" flag
                } else {
                    mul_bits += 4; // flag + 3-bit code
                }
                pm = mc;
            }

            total_bits += header_bits + pol_bits + mul_bits;
        }

        let total_bytes = (total_bits + 7) / 8;
        let bps = total_bits as f64 / encoded.len() as f64;
        let kbps = 44100.0 * bps / 1000.0;
        let pcm_bytes = (encoded.len() * 2) as u64;

        println!("  Frame size: {} samples ({:.1}ms at 44.1kHz)", frame_size, frame_size as f64 / 44.1);
        println!("    Polarity flips:     {:.1}% of samples", pol_flips as f64 / pol_total.max(1) as f64 * 100.0);
        println!("    Mul unchanged:      {:.1}% of samples", mul_deltas_zero as f64 / mul_deltas_total.max(1) as f64 * 100.0);
        println!("    Projected size:     {} bytes ({:.2} bits/sample)", total_bytes, bps);
        println!("    Compression vs PCM: {:.1}%", total_bytes as f64 / pcm_bytes as f64 * 100.0);
        println!("    Stream bandwidth:   {:.1} kbps", kbps);
        println!();
    }
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------

fn main() {
    let args: Vec<String> = env::args().collect();
    let input_path = if args.len() > 1 {
        args[1].clone()
    } else {
        eprintln!("Usage: ternsig-audio-experiment <input.mp3>");
        eprintln!("  Encodes audio to ternary signals, decodes back, writes WAV for comparison.");
        std::process::exit(1);
    };

    let input = Path::new(&input_path);
    let stem = input.file_stem().unwrap().to_string_lossy();

    // Output paths next to the input
    let parent = input.parent().unwrap_or(Path::new("."));
    let original_wav = parent.join(format!("{}_original.wav", stem));
    let encoded_path = parent.join(format!("{}_encoded.ternsig", stem));
    let decoded_wav = parent.join(format!("{}_decoded.wav", stem));

    // --- Decode MP3 ---
    println!("Decoding MP3...");
    let t = Instant::now();
    let audio = decode_mp3(input);
    println!(
        "  {} samples, {}ch, {} Hz ({:.2}s audio) — decoded in {:.1}ms",
        audio.samples.len(),
        audio.channels,
        audio.sample_rate,
        audio.samples.len() as f64 / audio.sample_rate as f64 / audio.channels as f64,
        t.elapsed().as_secs_f64() * 1000.0,
    );

    // Write original as WAV for clean A/B comparison
    println!("Writing original WAV...");
    write_wav(&original_wav, &audio.samples, audio.sample_rate, audio.channels);

    // --- Encode: PCM → PackedSignal ---
    println!("Encoding to ternary signals...");
    let t = Instant::now();
    let decode_lut = build_decode_lut();
    let encode_table = build_encode_table(&decode_lut);
    let encoded: Vec<u8> = audio
        .samples
        .iter()
        .map(|&s| encode_sample(s, &encode_table))
        .collect();
    let encode_ms = t.elapsed().as_secs_f64() * 1000.0;

    // Write raw encoded bytes
    std::fs::write(&encoded_path, &encoded).expect("Failed to write encoded file");

    println!(
        "  Encoded {} samples in {:.1}ms ({:.0} samples/ms)",
        encoded.len(),
        encode_ms,
        encoded.len() as f64 / encode_ms,
    );

    // --- Compress: delta + Huffman ---
    let compressed_path = parent.join(format!("{}_compressed.ternsig-c", stem));
    println!("Compressing (delta + Huffman)...");
    let t = Instant::now();
    let compressed = codec::encode(&encoded, audio.sample_rate, audio.channels);
    let compress_ms = t.elapsed().as_secs_f64() * 1000.0;
    std::fs::write(&compressed_path, &compressed.data).expect("Failed to write compressed file");
    println!(
        "  {} bytes → {} bytes ({:.2} bits/sample) in {:.1}ms",
        compressed.stats.raw_bytes,
        compressed.stats.compressed_bytes,
        compressed.stats.bits_per_sample,
        compress_ms,
    );
    println!(
        "  Header: {} bytes ({} table entries), bitstream: {} bytes",
        compressed.stats.header_bytes,
        compressed.stats.table_entries,
        compressed.stats.bitstream_bytes,
    );

    // --- Decompress: verify round-trip ---
    println!("Decompressing (verify round-trip)...");
    let t = Instant::now();
    let decompressed = codec::decode(&compressed.data);
    let decompress_ms = t.elapsed().as_secs_f64() * 1000.0;
    assert_eq!(
        decompressed.samples, encoded,
        "Round-trip mismatch! Compressed codec is lossy or broken."
    );
    println!(
        "  Verified {} samples in {:.1}ms — lossless round-trip confirmed",
        decompressed.samples.len(),
        decompress_ms,
    );

    // Decode compressed back to PCM for WAV output
    let compressed_decoded_wav = parent.join(format!("{}_compressed_decoded.wav", stem));
    let compressed_pcm: Vec<i16> = decompressed.samples.iter().map(|&b| decode_sample(b)).collect();
    write_wav(&compressed_decoded_wav, &compressed_pcm, audio.sample_rate, audio.channels);

    // --- Decode: PackedSignal → PCM ---
    println!("Decoding from ternary signals...");
    let t = Instant::now();
    let decoded: Vec<i16> = encoded.iter().map(|&b| decode_sample(b)).collect();
    let decode_ms = t.elapsed().as_secs_f64() * 1000.0;

    println!(
        "  Decoded {} samples in {:.1}ms ({:.0} samples/ms)",
        decoded.len(),
        decode_ms,
        decoded.len() as f64 / decode_ms,
    );

    // Write decoded WAV (raw — no smoothing)
    println!("Writing decoded WAV (raw)...");
    write_wav(&decoded_wav, &decoded, audio.sample_rate, audio.channels);

    // --- Smoothed variants ---
    let smoothed_linear_wav = parent.join(format!("{}_smoothed_linear.wav", stem));
    let smoothed_cubic_wav = parent.join(format!("{}_smoothed_cubic.wav", stem));

    println!("Smoothing (linear, 2 passes)...");
    let t = Instant::now();
    let smoothed_lin = smooth_linear(&decoded, 2);
    let lin_ms = t.elapsed().as_secs_f64() * 1000.0;
    println!("  {:.1}ms", lin_ms);
    write_wav(&smoothed_linear_wav, &smoothed_lin, audio.sample_rate, audio.channels);

    println!("Smoothing (cubic Hermite, 2 passes)...");
    let t = Instant::now();
    let smoothed_cub = smooth_cubic(&decoded, 2);
    let cub_ms = t.elapsed().as_secs_f64() * 1000.0;
    println!("  {:.1}ms", cub_ms);
    write_wav(&smoothed_cubic_wav, &smoothed_cub, audio.sample_rate, audio.channels);

    // --- Analysis ---
    println!("\n=== RESULTS ===\n");

    let input_size = std::fs::metadata(input).map(|m| m.len()).unwrap_or(0);
    let original_wav_size = std::fs::metadata(&original_wav).map(|m| m.len()).unwrap_or(0);
    let encoded_size = encoded.len() as u64;
    let decoded_wav_size = std::fs::metadata(&decoded_wav).map(|m| m.len()).unwrap_or(0);
    let pcm_size = (audio.samples.len() * 2) as u64; // i16 = 2 bytes/sample

    let compressed_size = compressed.stats.compressed_bytes;

    println!("  File sizes:");
    println!("    Source MP3:          {:>10} bytes", input_size);
    println!("    Original PCM (i16):  {:>10} bytes ({} bytes/sample)", pcm_size, 2);
    println!("    Ternsig raw:         {:>10} bytes ({} byte/sample)", encoded_size, 1);
    println!("    Ternsig compressed:  {:>10} bytes ({:.2} bits/sample)", compressed_size, compressed.stats.bits_per_sample);
    println!();
    println!("    Compressed vs PCM:   {:>9.1}%", (compressed_size as f64 / pcm_size as f64) * 100.0);
    println!("    Compressed vs MP3:   {:>9.1}%", (compressed_size as f64 / input_size as f64) * 100.0);
    println!("    Compressed vs raw:   {:>9.1}%", (compressed_size as f64 / encoded_size as f64) * 100.0);

    let snr_raw = compute_snr(&audio.samples, &decoded);
    let snr_lin = compute_snr(&audio.samples, &smoothed_lin);
    let snr_cub = compute_snr(&audio.samples, &smoothed_cub);
    let distinct = count_distinct_values(&encoded);

    println!("\n  Quality:");
    println!("    SNR (raw):           {:>10.2} dB", snr_raw);
    println!("    SNR (linear smooth): {:>10.2} dB", snr_lin);
    println!("    SNR (cubic smooth):  {:>10.2} dB", snr_cub);
    println!("    Distinct values:     {:>10} / 256", distinct);
    println!("    Effective bit depth: {:>10.1} bits", (distinct as f64).log2());

    // Bandwidth at various sample rates
    println!("\n  Streaming bandwidth (ternsig @ 1 byte/sample, mono):");
    for &rate in &[8000u32, 22050, 44100] {
        let kbps = rate as f64 * 8.0 / 1000.0;
        let kbs = rate as f64 / 1000.0;
        println!("    {} Hz:  {:.1} kbps ({:.1} KB/s)", rate, kbps, kbs);
    }

    println!("\n  Streaming bandwidth (PCM i16 @ 2 bytes/sample, mono):");
    for &rate in &[8000u32, 22050, 44100] {
        let kbps = rate as f64 * 16.0 / 1000.0;
        let kbs = rate as f64 * 2.0 / 1000.0;
        println!("    {} Hz:  {:.1} kbps ({:.1} KB/s)", rate, kbps, kbs);
    }

    print_value_distribution(&encoded);
    print_compression_analysis(&encoded);

    // --- Frame envelope split analysis ---
    print_envelope_analysis(&encoded, &audio.samples);

    // Compressed stream bandwidth
    println!("\n  Livestream bandwidth (compressed, 44.1 kHz mono):");
    let comp_kbps = 44100.0 * compressed.stats.bits_per_sample / 1000.0;
    println!("    Ternsig compressed: {:.1} kbps ({:.1} KB/s)", comp_kbps, comp_kbps / 8.0);
    println!("    MP3 (this file):    {:.1} kbps", input_size as f64 * 8.0 / (audio.samples.len() as f64 / audio.sample_rate as f64 / audio.channels as f64) / 1000.0);

    println!("\n  Output files:");
    println!("    Original WAV:        {}", original_wav.display());
    println!("    Ternsig raw:         {}", encoded_path.display());
    println!("    Ternsig compressed:  {}", compressed_path.display());
    println!("    Decoded (raw):       {}", decoded_wav.display());
    println!("    Decoded (compressed):{}", compressed_decoded_wav.display());
    println!("    Decoded (linear):    {}", smoothed_linear_wav.display());
    println!("    Decoded (cubic):     {}", smoothed_cubic_wav.display());
    println!("\n  The compressed file is lossless vs raw ternsig — identical audio output.");
    println!("  A/B compare decoded WAVs against the original.");
}
