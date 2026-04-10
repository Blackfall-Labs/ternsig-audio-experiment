//! Learn — analyze real audio to extract motor signal timing.
//!
//! Takes a reference audio file (e.g. a Protectron saying "protect"),
//! encodes it to ternary signals, detects energy events (onsets, peaks,
//! decays), and outputs a pronunciation memory: when to fire each
//! articulator, at what intensity, for how long.
//!
//! The output is a sequence of MotorEvents that can drive the relay
//! runner system with real timing instead of guessed physics constants.

use std::env;
use std::fs::File;
use std::path::Path;

use symphonia::core::audio::SampleBuffer;
use symphonia::core::codecs::DecoderOptions;
use symphonia::core::formats::FormatOptions;
use symphonia::core::io::MediaSourceStream;
use symphonia::core::meta::MetadataOptions;
use symphonia::core::probe::Hint;

use ternary_signal::PackedSignal;

// ---------------------------------------------------------------------------
// Audio loading (supports MP3, OGG, WAV via symphonia)
// ---------------------------------------------------------------------------

struct AudioData {
    samples: Vec<i16>,
    sample_rate: u32,
}

fn load_audio(path: &Path) -> AudioData {
    let file = File::open(path).expect("Failed to open audio file");
    let mss = MediaSourceStream::new(Box::new(file), Default::default());

    let ext = path.extension().and_then(|e| e.to_str()).unwrap_or("");
    let mut hint = Hint::new();
    hint.with_extension(ext);

    let probed = symphonia::default::get_probe()
        .format(&hint, mss, &FormatOptions::default(), &MetadataOptions::default())
        .expect("Failed to probe audio");

    let mut format = probed.format;
    let track = format.default_track().expect("No audio track");
    let track_id = track.id;
    let sample_rate = track.codec_params.sample_rate.unwrap_or(44100);
    let channels = track.codec_params.channels.map(|c| c.count()).unwrap_or(1);

    let mut decoder = symphonia::default::get_codecs()
        .make(&track.codec_params, &DecoderOptions::default())
        .expect("Failed to create decoder");

    let mut all_samples: Vec<i16> = Vec::new();

    loop {
        let packet = match format.next_packet() {
            Ok(p) => p,
            Err(_) => break,
        };
        if packet.track_id() != track_id { continue; }
        let decoded = match decoder.decode(&packet) {
            Ok(d) => d,
            Err(_) => continue,
        };
        let spec = *decoded.spec();
        let duration = decoded.capacity();
        let mut buf = SampleBuffer::<i16>::new(duration as u64, spec);
        buf.copy_interleaved_ref(decoded);
        all_samples.extend_from_slice(buf.samples());
    }

    // Mix to mono
    let mono = if channels > 1 {
        all_samples.chunks(channels)
            .map(|frame| {
                let sum: i32 = frame.iter().map(|&s| s as i32).sum();
                (sum / channels as i32) as i16
            })
            .collect()
    } else {
        all_samples
    };

    AudioData { samples: mono, sample_rate }
}

// ---------------------------------------------------------------------------
// Ternary encoding
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
            if idx == 0 { table[0].1 }
            else if idx >= table.len() { table[table.len() - 1].1 }
            else {
                let lo = table[idx - 1];
                let hi = table[idx];
                if (target - lo.0).abs() <= (target - hi.0).abs() { lo.1 } else { hi.1 }
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Motor event detection
// ---------------------------------------------------------------------------

/// A detected motor event — a moment where articulation changes.
#[derive(Debug, Clone)]
struct MotorEvent {
    /// Frame index in the source audio.
    frame: usize,
    /// Time in milliseconds.
    time_ms: f64,
    /// Event type.
    kind: MotorKind,
    /// Energy level at this point (0.0 to 1.0).
    energy: f32,
    /// Rate of energy change (positive = onset, negative = decay).
    energy_slope: f32,
}

#[derive(Debug, Clone, Copy)]
enum MotorKind {
    /// Energy crossing upward past threshold — articulation begins.
    Onset,
    /// Energy reaches local maximum — peak articulation.
    Peak,
    /// Energy crossing downward past threshold — articulation ending.
    Offset,
    /// Rapid energy spike — plosive burst or impact.
    Impact,
}

/// Compute windowed RMS energy envelope from encoded signal.
fn energy_envelope(encoded: &[u8], window_frames: usize) -> Vec<f32> {
    let half = window_frames / 2;
    let len = encoded.len();
    let mut envelope = Vec::with_capacity(len);

    for i in 0..len {
        let start = i.saturating_sub(half);
        let end = (i + half).min(len);
        let window = &encoded[start..end];

        let rms: f32 = {
            let sum_sq: f64 = window.iter()
                .map(|&b| {
                    let c = PackedSignal::from_raw(b).current() as f64;
                    c * c
                })
                .sum();
            (sum_sq / window.len() as f64).sqrt() as f32
        };

        // Normalize to 0-1 (max current is 65025)
        envelope.push(rms / 65025.0);
    }

    envelope
}

/// Detect motor events from the energy envelope.
fn detect_events(
    envelope: &[f32],
    sample_rate: u32,
    onset_threshold: f32,
    impact_threshold: f32,
) -> Vec<MotorEvent> {
    let mut events = Vec::new();
    let len = envelope.len();
    if len < 3 { return events; }

    // Compute energy slope (derivative)
    let mut slopes = vec![0.0f32; len];
    for i in 1..len - 1 {
        slopes[i] = (envelope[i + 1] - envelope[i - 1]) / 2.0;
    }

    let mut above_threshold = envelope[0] > onset_threshold;

    for i in 1..len - 1 {
        let e = envelope[i];
        let s = slopes[i];
        let time_ms = i as f64 / sample_rate as f64 * 1000.0;

        // Onset: crossing upward past threshold
        if !above_threshold && e > onset_threshold {
            above_threshold = true;
            events.push(MotorEvent {
                frame: i,
                time_ms,
                kind: MotorKind::Onset,
                energy: e,
                energy_slope: s,
            });
        }

        // Offset: crossing downward past threshold
        if above_threshold && e < onset_threshold * 0.7 {
            above_threshold = false;
            events.push(MotorEvent {
                frame: i,
                time_ms,
                kind: MotorKind::Offset,
                energy: e,
                energy_slope: s,
            });
        }

        // Peak: local maximum above threshold
        if e > onset_threshold && slopes[i - 1] > 0.0 && s <= 0.0 && e > onset_threshold * 1.5 {
            events.push(MotorEvent {
                frame: i,
                time_ms,
                kind: MotorKind::Peak,
                energy: e,
                energy_slope: s,
            });
        }

        // Impact: very rapid energy spike
        if s > impact_threshold {
            // Only if we haven't already logged an onset within 10ms
            let recent_onset = events.iter().rev()
                .take(3)
                .any(|ev| matches!(ev.kind, MotorKind::Impact) && (time_ms - ev.time_ms) < 10.0);
            if !recent_onset {
                events.push(MotorEvent {
                    frame: i,
                    time_ms,
                    kind: MotorKind::Impact,
                    energy: e,
                    energy_slope: s,
                });
            }
        }
    }

    events
}

// ---------------------------------------------------------------------------
// Pronunciation memory
// ---------------------------------------------------------------------------

/// A segment of pronunciation: one articulatory gesture.
#[derive(Debug, Clone)]
struct PronunciationSegment {
    /// Start time in ms.
    start_ms: f64,
    /// End time in ms.
    end_ms: f64,
    /// Duration in ms.
    duration_ms: f64,
    /// Peak energy (0.0-1.0).
    peak_energy: f32,
    /// Attack time: onset to peak (ms).
    attack_ms: f64,
    /// Decay time: peak to offset (ms).
    decay_ms: f64,
    /// Whether this segment has an impact (plosive-like onset).
    has_impact: bool,
    /// Signal data for this segment (PackedSignal bytes).
    signal: Vec<u8>,
}

/// Extract pronunciation segments from motor events.
fn extract_segments(
    events: &[MotorEvent],
    encoded: &[u8],
    sample_rate: u32,
) -> Vec<PronunciationSegment> {
    let mut segments = Vec::new();

    // Pair onsets with offsets
    let mut i = 0;
    while i < events.len() {
        let onset = match events[i].kind {
            MotorKind::Onset | MotorKind::Impact => &events[i],
            _ => { i += 1; continue; }
        };

        let has_impact = matches!(onset.kind, MotorKind::Impact);

        // Find the peak within this segment
        let peak = events[i..].iter()
            .take_while(|e| !matches!(e.kind, MotorKind::Offset))
            .filter(|e| matches!(e.kind, MotorKind::Peak))
            .max_by(|a, b| a.energy.partial_cmp(&b.energy).unwrap());

        // Find the offset
        let offset = events[i..].iter()
            .find(|e| matches!(e.kind, MotorKind::Offset));

        let peak_time = peak.map(|p| p.time_ms).unwrap_or(onset.time_ms + 20.0);
        let peak_energy = peak.map(|p| p.energy).unwrap_or(onset.energy);
        let end_time = offset.map(|o| o.time_ms).unwrap_or(onset.time_ms + 100.0);

        let start_frame = onset.frame;
        let end_frame = offset.map(|o| o.frame).unwrap_or((start_frame + 800).min(encoded.len()));

        let signal = encoded[start_frame..end_frame.min(encoded.len())].to_vec();

        segments.push(PronunciationSegment {
            start_ms: onset.time_ms,
            end_ms: end_time,
            duration_ms: end_time - onset.time_ms,
            peak_energy,
            attack_ms: peak_time - onset.time_ms,
            decay_ms: end_time - peak_time,
            has_impact,
            signal,
        });

        // Skip past offset
        i = events[i..].iter()
            .position(|e| matches!(e.kind, MotorKind::Offset))
            .map(|pos| i + pos + 1)
            .unwrap_or(events.len());
    }

    segments
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------

fn main() {
    let args: Vec<String> = env::args().collect();
    let input_path = if args.len() > 1 {
        args[1].clone()
    } else {
        eprintln!("Usage: learn <audio_file>");
        eprintln!("  Analyzes real audio to extract motor signal timing for pronunciation memory.");
        std::process::exit(1);
    };

    let input = Path::new(&input_path);
    let stem = input.file_stem().unwrap().to_string_lossy();

    println!("=== PRONUNCIATION LEARNING ===\n");
    println!("  Source: {}", input.display());

    // Load audio
    let audio = load_audio(input);
    let duration_ms = audio.samples.len() as f64 / audio.sample_rate as f64 * 1000.0;
    println!("  {} samples, {} Hz, {:.1}ms\n", audio.samples.len(), audio.sample_rate, duration_ms);

    // Encode to ternary
    let table = build_encode_table();
    let encoded: Vec<u8> = audio.samples.iter().map(|&s| encode_sample(s, &table)).collect();

    // Compute energy envelope (5ms window)
    let window = (audio.sample_rate as f32 * 0.005) as usize;
    let envelope = energy_envelope(&encoded, window.max(1));

    // Find envelope stats for threshold calibration
    let max_energy = envelope.iter().cloned().fold(0.0f32, f32::max);
    let mean_energy: f32 = envelope.iter().sum::<f32>() / envelope.len() as f32;
    println!("  Energy envelope:");
    println!("    Max:  {:.4}", max_energy);
    println!("    Mean: {:.4}", mean_energy);

    // Detect motor events
    let onset_threshold = mean_energy * 1.5; // above average = onset
    let impact_threshold = max_energy * 0.15; // rapid spike = impact
    println!("    Onset threshold:  {:.4}", onset_threshold);
    println!("    Impact threshold: {:.4}", impact_threshold);

    let events = detect_events(&envelope, audio.sample_rate, onset_threshold, impact_threshold);

    println!("\n  Motor events detected: {}\n", events.len());
    println!("  {:>8}  {:>8}  {:>8}  {:>8}", "Time(ms)", "Type", "Energy", "Slope");
    println!("  {}  {}  {}  {}", "-".repeat(8), "-".repeat(8), "-".repeat(8), "-".repeat(8));
    for event in &events {
        let kind_str = match event.kind {
            MotorKind::Onset => "ONSET",
            MotorKind::Peak => "PEAK",
            MotorKind::Offset => "OFFSET",
            MotorKind::Impact => "IMPACT",
        };
        println!("  {:>7.1}  {:>8}  {:>7.4}  {:>+7.4}", event.time_ms, kind_str, event.energy, event.energy_slope);
    }

    // Extract pronunciation segments
    let segments = extract_segments(&events, &encoded, audio.sample_rate);

    println!("\n  Pronunciation segments: {}\n", segments.len());
    println!("  {:>4}  {:>8}  {:>8}  {:>8}  {:>8}  {:>8}  {:>7}  {:>8}",
        "#", "Start", "End", "Dur(ms)", "Attack", "Decay", "Peak E", "Impact?");
    println!("  {}  {}  {}  {}  {}  {}  {}  {}",
        "-".repeat(4), "-".repeat(8), "-".repeat(8), "-".repeat(8),
        "-".repeat(8), "-".repeat(8), "-".repeat(7), "-".repeat(8));

    for (i, seg) in segments.iter().enumerate() {
        println!("  {:>4}  {:>7.1}  {:>7.1}  {:>7.1}  {:>7.1}  {:>7.1}  {:>6.4}  {:>8}",
            i, seg.start_ms, seg.end_ms, seg.duration_ms,
            seg.attack_ms, seg.decay_ms, seg.peak_energy,
            if seg.has_impact { "YES" } else { "" });
    }

    // Summary: this is what the runner system needs
    println!("\n  === FLYWHEEL PHYSICS FROM REAL AUDIO ===\n");
    for (i, seg) in segments.iter().enumerate() {
        let inertia = (seg.attack_ms / seg.duration_ms).clamp(0.0, 1.0) as f32;
        let ring = (seg.decay_ms / seg.duration_ms).clamp(0.0, 1.0) as f32;
        let friction = if seg.decay_ms > 0.0 {
            (1.0 / (seg.decay_ms / 20.0)).clamp(0.05, 1.0) as f32
        } else {
            1.0
        };

        println!("  Segment {} ({:.0}-{:.0}ms):", i, seg.start_ms, seg.end_ms);
        println!("    inertia:   {:.3}  (attack {:.1}ms / {:.1}ms total)", inertia, seg.attack_ms, seg.duration_ms);
        println!("    friction:  {:.3}  (decay {:.1}ms)", friction, seg.decay_ms);
        println!("    ring:      {:.3}  ({:.1}ms tail)", ring, seg.decay_ms);
        println!("    peak:      {:.4}", seg.peak_energy);
        println!("    signal:    {} bytes ({:.1}ms)", seg.signal.len(),
            seg.signal.len() as f64 / audio.sample_rate as f64 * 1000.0);
        println!();
    }

    // Total word timing
    if let (Some(first), Some(last)) = (segments.first(), segments.last()) {
        println!("  Word timing: {:.1}ms onset → {:.1}ms total", first.start_ms, last.end_ms);
        println!("  Segment count: {} articulatory gestures", segments.len());
    }

    println!("\n  Done.");
}
