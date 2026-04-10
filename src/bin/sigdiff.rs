//! Signal Diff — compare two audio files in ternary signal space.
//!
//! Encodes both WAVs to PackedSignal streams and produces a structural
//! comparison: magnitude envelope, polarity alignment, delta distribution,
//! transition smoothness, and per-region error analysis.

use std::env;
use std::path::Path;
use std::time::Instant;

use hound::WavReader;
use ternary_signal::PackedSignal;

// ---------------------------------------------------------------------------
// WAV loading
// ---------------------------------------------------------------------------

struct AudioData {
    samples: Vec<i16>,
    sample_rate: u32,
}

fn load_wav(path: &Path) -> AudioData {
    let reader = WavReader::open(path).expect("Failed to open WAV");
    let spec = reader.spec();

    // Convert to mono i16 regardless of source format
    let samples: Vec<i16> = if spec.sample_format == hound::SampleFormat::Float {
        reader
            .into_samples::<f32>()
            .map(|s| (s.unwrap() * 32767.0) as i16)
            .collect()
    } else if spec.bits_per_sample == 16 {
        reader.into_samples::<i16>().map(|s| s.unwrap()).collect()
    } else {
        reader
            .into_samples::<i32>()
            .map(|s| (s.unwrap() >> (spec.bits_per_sample - 16)) as i16)
            .collect()
    };

    // Mix to mono if stereo
    let mono = if spec.channels > 1 {
        let ch = spec.channels as usize;
        samples
            .chunks(ch)
            .map(|frame| {
                let sum: i32 = frame.iter().map(|&s| s as i32).sum();
                (sum / ch as i32) as i16
            })
            .collect()
    } else {
        samples
    };

    AudioData {
        samples: mono,
        sample_rate: spec.sample_rate,
    }
}

// ---------------------------------------------------------------------------
// Encoding (same as main.rs)
// ---------------------------------------------------------------------------

fn build_encode_table() -> Vec<(i32, u8)> {
    let mut pairs: Vec<(i32, u8)> = (0..=255u8)
        .map(|b| (PackedSignal::from_raw(b).current(), b))
        .collect();
    pairs.sort_by_key(|&(cur, _)| cur);
    pairs.dedup_by_key(|p| p.0);
    pairs
}

fn encode_sample(sample: i16, table: &[(i32, u8)]) -> u8 {
    let target = (sample as i64 * 65025 / 32768) as i32;
    match table.binary_search_by_key(&target, |&(cur, _)| cur) {
        Ok(idx) => table[idx].1,
        Err(idx) => {
            if idx == 0 {
                table[0].1
            } else if idx >= table.len() {
                table[table.len() - 1].1
            } else {
                let lo = table[idx - 1];
                let hi = table[idx];
                if (target - lo.0).abs() <= (target - hi.0).abs() {
                    lo.1
                } else {
                    hi.1
                }
            }
        }
    }
}

fn encode_audio(samples: &[i16], table: &[(i32, u8)]) -> Vec<u8> {
    samples.iter().map(|&s| encode_sample(s, table)).collect()
}

// ---------------------------------------------------------------------------
// Analysis
// ---------------------------------------------------------------------------

/// Magnitude envelope: extract mag_code (0-7) per sample
fn magnitude_envelope(encoded: &[u8]) -> Vec<u8> {
    encoded
        .iter()
        .map(|&b| PackedSignal::from_raw(b).mag_code())
        .collect()
}

/// Multiplier envelope: extract mul_code (0-7) per sample
fn multiplier_envelope(encoded: &[u8]) -> Vec<u8> {
    encoded
        .iter()
        .map(|&b| PackedSignal::from_raw(b).mul_code())
        .collect()
}

/// Polarity sequence: -1, 0, or +1 per sample
fn polarity_sequence(encoded: &[u8]) -> Vec<i8> {
    encoded
        .iter()
        .map(|&b| PackedSignal::from_raw(b).polarity())
        .collect()
}

/// Count polarity transitions (sign changes)
fn polarity_transitions(pol: &[i8]) -> (u64, u64, f64) {
    let mut transitions = 0u64;
    let mut zero_crossings = 0u64;
    for i in 1..pol.len() {
        if pol[i] != pol[i - 1] {
            transitions += 1;
            if (pol[i] == 1 && pol[i - 1] == -1) || (pol[i] == -1 && pol[i - 1] == 1) {
                zero_crossings += 1; // skipped zero — polarity violation
            }
        }
    }
    let rate = transitions as f64 / (pol.len() - 1) as f64;
    (transitions, zero_crossings, rate)
}

/// Delta distribution of encoded byte stream
fn delta_stats(encoded: &[u8]) -> (f64, f64, u64, u64) {
    let mut zeros = 0u64;
    let mut small = 0u64;
    for i in 1..encoded.len() {
        let d = encoded[i] as i16 - encoded[i - 1] as i16;
        if d == 0 {
            zeros += 1;
        }
        if d.abs() <= 7 {
            small += 1;
        }
    }
    let total = (encoded.len() - 1) as f64;
    (zeros as f64 / total * 100.0, small as f64 / total * 100.0, zeros, small)
}

/// RMS error in current-value space
fn current_rms_error(a: &[u8], b: &[u8]) -> f64 {
    let len = a.len().min(b.len());
    let sum_sq: f64 = a[..len]
        .iter()
        .zip(b[..len].iter())
        .map(|(&x, &y)| {
            let cx = PackedSignal::from_raw(x).current() as f64;
            let cy = PackedSignal::from_raw(y).current() as f64;
            (cx - cy) * (cx - cy)
        })
        .sum();
    (sum_sq / len as f64).sqrt()
}

/// Windowed magnitude envelope comparison
fn envelope_comparison(a: &[u8], b: &[u8], window_ms: f64, sample_rate: u32) {
    let window = (window_ms * sample_rate as f64 / 1000.0) as usize;
    let len = a.len().min(b.len());
    let num_windows = len / window;

    if num_windows == 0 {
        println!("    (audio too short for windowed analysis)");
        return;
    }

    let mut max_error: f64 = 0.0;
    let mut total_error: f64 = 0.0;
    let mut worst_window = 0;

    for w in 0..num_windows {
        let start = w * window;
        let end = start + window;

        // Average magnitude in this window
        let avg_a: f64 = a[start..end]
            .iter()
            .map(|&b| PackedSignal::from_raw(b).current().abs() as f64)
            .sum::<f64>()
            / window as f64;
        let avg_b: f64 = b[start..end]
            .iter()
            .map(|&b| PackedSignal::from_raw(b).current().abs() as f64)
            .sum::<f64>()
            / window as f64;

        let error = (avg_a - avg_b).abs();
        total_error += error;
        if error > max_error {
            max_error = error;
            worst_window = w;
        }
    }

    let avg_error = total_error / num_windows as f64;
    let worst_time = worst_window as f64 * window_ms / 1000.0;

    println!("    Window size:     {:.0}ms ({} samples)", window_ms, window);
    println!("    Windows:         {}", num_windows);
    println!("    Avg envelope Δ:  {:.1}", avg_error);
    println!("    Max envelope Δ:  {:.1} (at {:.2}s)", max_error, worst_time);
}

/// Transition smoothness: how often does the signal jump more than N levels?
fn transition_smoothness(encoded: &[u8]) -> (u64, u64, u64) {
    let mut smooth = 0u64; // delta 0-1 levels
    let mut medium = 0u64; // delta 2-4 levels
    let mut harsh = 0u64;  // delta 5+ levels

    for i in 1..encoded.len() {
        let prev = PackedSignal::from_raw(encoded[i - 1]);
        let curr = PackedSignal::from_raw(encoded[i]);
        let mag_delta = (curr.mag_code() as i16 - prev.mag_code() as i16).unsigned_abs();
        let mul_delta = (curr.mul_code() as i16 - prev.mul_code() as i16).unsigned_abs();
        let max_delta = mag_delta.max(mul_delta);

        if max_delta <= 1 {
            smooth += 1;
        } else if max_delta <= 4 {
            medium += 1;
        } else {
            harsh += 1;
        }
    }
    (smooth, medium, harsh)
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------

fn main() {
    let args: Vec<String> = env::args().collect();

    if args.len() < 2 {
        eprintln!("Usage: sigdiff <file_a.wav> [file_b.wav]");
        eprintln!("  One file:  analyze signal structure");
        eprintln!("  Two files: compare signal structures (A = engine, B = reference)");
        std::process::exit(1);
    }

    let table = build_encode_table();

    let path_a = Path::new(&args[1]);
    println!("Loading A: {}", path_a.display());
    let audio_a = load_wav(path_a);
    let encoded_a = encode_audio(&audio_a.samples, &table);
    println!(
        "  {} samples, {} Hz, {:.2}s",
        audio_a.samples.len(),
        audio_a.sample_rate,
        audio_a.samples.len() as f64 / audio_a.sample_rate as f64,
    );

    let analyze_single = |label: &str, encoded: &[u8], sample_rate: u32| {
        println!("\n=== {} SIGNAL ANALYSIS ===\n", label);

        // Distinct values
        let mut seen = [false; 256];
        for &b in encoded {
            seen[b as usize] = true;
        }
        let distinct = seen.iter().filter(|&&v| v).count();
        println!("  Distinct PackedSignal values: {}", distinct);

        // Polarity
        let pol = polarity_sequence(encoded);
        let (transitions, violations, rate) = polarity_transitions(&pol);
        let pos = pol.iter().filter(|&&p| p > 0).count();
        let neg = pol.iter().filter(|&&p| p < 0).count();
        let zero = pol.iter().filter(|&&p| p == 0).count();
        println!("\n  Polarity:");
        println!("    Positive: {:>8} ({:.1}%)", pos, pos as f64 / pol.len() as f64 * 100.0);
        println!("    Negative: {:>8} ({:.1}%)", neg, neg as f64 / pol.len() as f64 * 100.0);
        println!("    Zero:     {:>8} ({:.1}%)", zero, zero as f64 / pol.len() as f64 * 100.0);
        println!("    Transitions:      {}", transitions);
        println!("    Transition rate:  {:.4} per sample", rate);
        println!("    Zero-skip violations: {} {}", violations,
            if violations > 0 { "← polarity jumped ±1→∓1 without passing 0" } else { "(clean)" });

        // Delta distribution
        let (zero_pct, small_pct, _, _) = delta_stats(encoded);
        println!("\n  Delta distribution:");
        println!("    Zero deltas: {:.1}%", zero_pct);
        println!("    Small (±7):  {:.1}%", small_pct);

        // Transition smoothness
        let (smooth, medium, harsh) = transition_smoothness(encoded);
        let total_t = (encoded.len() - 1) as f64;
        println!("\n  Transition smoothness:");
        println!("    Smooth (0-1 LUT steps): {:>8} ({:.1}%)", smooth, smooth as f64 / total_t * 100.0);
        println!("    Medium (2-4 LUT steps): {:>8} ({:.1}%)", medium, medium as f64 / total_t * 100.0);
        println!("    Harsh  (5+  LUT steps): {:>8} ({:.1}%)", harsh, harsh as f64 / total_t * 100.0);

        // Magnitude distribution
        let mut mag_counts = [0u64; 8];
        for &b in encoded {
            mag_counts[PackedSignal::from_raw(b).mag_code() as usize] += 1;
        }
        println!("\n  Magnitude code distribution:");
        let lut = [0u8, 1, 4, 16, 32, 64, 128, 255];
        for (code, &count) in mag_counts.iter().enumerate() {
            let pct = count as f64 / encoded.len() as f64 * 100.0;
            let bar_len = (pct * 0.5) as usize;
            println!("    mag={} (val={:>3}): {:>7} ({:>5.1}%) {}",
                code, lut[code], count, pct, "#".repeat(bar_len));
        }

        // Envelope analysis
        println!("\n  Envelope (20ms windows):");
        envelope_comparison(encoded, encoded, 20.0, sample_rate);
    };

    analyze_single("A", &encoded_a, audio_a.sample_rate);

    // If two files, load B and compare
    if args.len() >= 3 {
        let path_b = Path::new(&args[2]);
        println!("\n\nLoading B: {}", path_b.display());
        let audio_b = load_wav(path_b);
        let encoded_b = encode_audio(&audio_b.samples, &table);
        println!(
            "  {} samples, {} Hz, {:.2}s",
            audio_b.samples.len(),
            audio_b.sample_rate,
            audio_b.samples.len() as f64 / audio_b.sample_rate as f64,
        );

        analyze_single("B", &encoded_b, audio_b.sample_rate);

        // === COMPARISON ===
        println!("\n\n=== SIGNAL COMPARISON (A vs B) ===\n");

        let len = encoded_a.len().min(encoded_b.len());
        println!("  Comparing first {} samples ({:.2}s)",
            len, len as f64 / audio_a.sample_rate as f64);

        // RMS error
        let rms = current_rms_error(&encoded_a, &encoded_b);
        println!("\n  RMS error (current space): {:.1}", rms);
        println!("  RMS error (% of range):    {:.2}%", rms / 65025.0 * 100.0);

        // Polarity alignment
        let pol_a = polarity_sequence(&encoded_a);
        let pol_b = polarity_sequence(&encoded_b);
        let mut pol_match = 0u64;
        let mut pol_sign_match = 0u64;
        for i in 0..len {
            if pol_a[i] == pol_b[i] {
                pol_match += 1;
            }
            if pol_a[i].signum() == pol_b[i].signum() {
                pol_sign_match += 1;
            }
        }
        println!("\n  Polarity alignment:");
        println!("    Exact match:  {:.1}%", pol_match as f64 / len as f64 * 100.0);
        println!("    Sign match:   {:.1}%", pol_sign_match as f64 / len as f64 * 100.0);

        // Magnitude code agreement
        let mut mag_match = 0u64;
        let mut mag_off_by_1 = 0u64;
        for i in 0..len {
            let ma = PackedSignal::from_raw(encoded_a[i]).mag_code();
            let mb = PackedSignal::from_raw(encoded_b[i]).mag_code();
            if ma == mb {
                mag_match += 1;
            }
            if (ma as i16 - mb as i16).abs() <= 1 {
                mag_off_by_1 += 1;
            }
        }
        println!("\n  Magnitude code alignment:");
        println!("    Exact match:    {:.1}%", mag_match as f64 / len as f64 * 100.0);
        println!("    Within 1 step:  {:.1}%", mag_off_by_1 as f64 / len as f64 * 100.0);

        // Transition smoothness comparison
        let (smooth_a, medium_a, harsh_a) = transition_smoothness(&encoded_a[..len]);
        let (smooth_b, medium_b, harsh_b) = transition_smoothness(&encoded_b[..len]);
        let t = (len - 1) as f64;
        println!("\n  Transition smoothness comparison:");
        println!("    {:>20}  {:>8}  {:>8}", "", "A", "B");
        println!("    {:>20}  {:>7.1}%  {:>7.1}%", "Smooth (0-1 steps)",
            smooth_a as f64 / t * 100.0, smooth_b as f64 / t * 100.0);
        println!("    {:>20}  {:>7.1}%  {:>7.1}%", "Medium (2-4 steps)",
            medium_a as f64 / t * 100.0, medium_b as f64 / t * 100.0);
        println!("    {:>20}  {:>7.1}%  {:>7.1}%", "Harsh (5+ steps)",
            harsh_a as f64 / t * 100.0, harsh_b as f64 / t * 100.0);

        // Envelope comparison
        println!("\n  Envelope difference (20ms windows):");
        envelope_comparison(&encoded_a[..len], &encoded_b[..len], 20.0, audio_a.sample_rate);
    }

    println!("\n  Done.");
}
