//! Ternsig Audio Experiment
//!
//! Tests whether audio can survive a round-trip through ternary signal encoding:
//!   PCM → PackedSignal → PCM
//!
//! The question: is 1-byte-per-sample ternary encoding viable for audio,
//! and is representational decomposition (p × m × k) more efficient than
//! raw sample depth?

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

    // Write decoded WAV
    println!("Writing decoded WAV...");
    write_wav(&decoded_wav, &decoded, audio.sample_rate, audio.channels);

    // --- Analysis ---
    println!("\n=== RESULTS ===\n");

    let input_size = std::fs::metadata(input).map(|m| m.len()).unwrap_or(0);
    let original_wav_size = std::fs::metadata(&original_wav).map(|m| m.len()).unwrap_or(0);
    let encoded_size = encoded.len() as u64;
    let decoded_wav_size = std::fs::metadata(&decoded_wav).map(|m| m.len()).unwrap_or(0);
    let pcm_size = (audio.samples.len() * 2) as u64; // i16 = 2 bytes/sample

    println!("  File sizes:");
    println!("    Source MP3:          {:>10} bytes", input_size);
    println!("    Original PCM (i16):  {:>10} bytes ({} bytes/sample)", pcm_size, 2);
    println!("    Encoded ternsig:     {:>10} bytes ({} byte/sample)", encoded_size, 1);
    println!("    Compression vs PCM:  {:>9.1}%", (encoded_size as f64 / pcm_size as f64) * 100.0);
    println!("    Compression vs MP3:  {:>9.1}%", (encoded_size as f64 / input_size as f64) * 100.0);

    let snr = compute_snr(&audio.samples, &decoded);
    let distinct = count_distinct_values(&encoded);

    println!("\n  Quality:");
    println!("    SNR:                 {:>10.2} dB", snr);
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

    println!("\n  Output files:");
    println!("    Original WAV:  {}", original_wav.display());
    println!("    Encoded raw:   {}", encoded_path.display());
    println!("    Decoded WAV:   {}", decoded_wav.display());
    println!("\n  Listen to both WAVs and compare. The decoded file is what");
    println!("  ternary-encoded audio sounds like after a full round-trip.");
}
