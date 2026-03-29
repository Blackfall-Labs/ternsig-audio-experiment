//! Ternsig Video Experiment
//!
//! Tests ternary signal encoding for images and video frames:
//!   Pixel (u8 grayscale) → PackedSignal → Delta + Huffman → Wire
//!
//! Two compression strategies:
//!   1. Scanline delta — spatial correlation within a single frame
//!   2. Frame delta — temporal correlation between consecutive frames
//!
//! Dynamic resolution: given a bandwidth budget, find the highest
//! resolution that fits. Latency controls quality in realtime.

use std::env;
use std::io::{Read as _, Seek as _, SeekFrom};
use std::path::Path;
use std::time::Instant;

use image::imageops::FilterType;
use image::{GrayImage, ImageReader};
use ternary_signal::PackedSignal;

use ternsig_audio_experiment::codec;

// ---------------------------------------------------------------------------
// Pixel ↔ PackedSignal mapping
// ---------------------------------------------------------------------------

/// Build encode table: for each possible pixel value (0-255), find the nearest
/// PackedSignal byte. Pixels map to the positive half of the current range.
fn build_pixel_encode_table() -> [u8; 256] {
    // Build all possible positive current values from PackedSignal
    let mut candidates: Vec<(i32, u8)> = Vec::new();
    for byte in 0..=255u8 {
        let ps = PackedSignal::from_raw(byte);
        let cur = ps.current();
        if cur >= 0 {
            candidates.push((cur, byte));
        }
    }
    candidates.sort_by_key(|&(cur, _)| cur);
    candidates.dedup_by_key(|p| p.0);

    let max_current = candidates.last().map(|&(c, _)| c).unwrap_or(1) as f64;

    let mut table = [0u8; 256];
    for pixel in 0..=255u8 {
        // Map pixel 0-255 to current 0-max_current
        let target = (pixel as f64 / 255.0 * max_current) as i32;

        // Binary search for nearest
        let idx = match candidates.binary_search_by_key(&target, |&(c, _)| c) {
            Ok(i) => i,
            Err(i) => {
                if i == 0 {
                    0
                } else if i >= candidates.len() {
                    candidates.len() - 1
                } else {
                    let lo = (target - candidates[i - 1].0).abs();
                    let hi = (target - candidates[i].0).abs();
                    if lo <= hi { i - 1 } else { i }
                }
            }
        };
        table[pixel as usize] = candidates[idx].1;
    }
    table
}

/// Build decode table: for each PackedSignal byte, what pixel value does it produce?
fn build_pixel_decode_table() -> [u8; 256] {
    // Find max positive current for scaling
    let mut max_current: i32 = 0;
    for byte in 0..=255u8 {
        let cur = PackedSignal::from_raw(byte).current();
        if cur > max_current {
            max_current = cur;
        }
    }

    let mut table = [0u8; 256];
    for byte in 0..=255u8 {
        let cur = PackedSignal::from_raw(byte).current();
        // Map current back to pixel 0-255 (clamp negatives to 0)
        let pixel = if cur <= 0 || max_current == 0 {
            0
        } else {
            ((cur as f64 / max_current as f64) * 255.0).round() as u8
        };
        table[byte as usize] = pixel;
    }
    table
}

// ---------------------------------------------------------------------------
// Image encoding
// ---------------------------------------------------------------------------

struct FrameResult {
    width: u32,
    height: u32,
    raw_pixels: usize,
    encoded_bytes: Vec<u8>,
    scanline_compressed: codec::CompressedStream,
    decoded_pixels: Vec<u8>,
    psnr: f64,
    distinct_values: usize,
}

fn encode_frame(
    gray: &GrayImage,
    encode_table: &[u8; 256],
    decode_table: &[u8; 256],
) -> FrameResult {
    let (w, h) = gray.dimensions();
    let pixels: Vec<u8> = gray.as_raw().to_vec();

    // Encode each pixel → PackedSignal byte
    let encoded: Vec<u8> = pixels.iter().map(|&p| encode_table[p as usize]).collect();

    // Decode back to pixels
    let decoded: Vec<u8> = encoded.iter().map(|&b| decode_table[b as usize]).collect();

    // PSNR
    let psnr = compute_psnr(&pixels, &decoded);

    // Distinct PackedSignal values used
    let mut seen = [false; 256];
    for &b in &encoded {
        seen[b as usize] = true;
    }
    let distinct = seen.iter().filter(|&&v| v).count();

    // Scanline delta + Huffman compress
    let compressed = codec::encode(&encoded, w, 1);

    FrameResult {
        width: w,
        height: h,
        raw_pixels: pixels.len(),
        encoded_bytes: encoded,
        scanline_compressed: compressed,
        decoded_pixels: decoded,
        psnr,
        distinct_values: distinct,
    }
}

fn compute_psnr(original: &[u8], reconstructed: &[u8]) -> f64 {
    let mse: f64 = original
        .iter()
        .zip(reconstructed.iter())
        .map(|(&o, &r)| {
            let diff = o as f64 - r as f64;
            diff * diff
        })
        .sum::<f64>()
        / original.len() as f64;

    if mse == 0.0 {
        return f64::INFINITY;
    }
    10.0 * (255.0 * 255.0 / mse).log10()
}

// ---------------------------------------------------------------------------
// Frame delta (temporal compression)
// ---------------------------------------------------------------------------

struct FrameDeltaResult {
    label: String,
    delta_bytes: Vec<u8>,
    compressed: codec::CompressedStream,
    zero_pct: f64,
    small_pct: f64,
}

/// Compute frame delta between two encoded frames and compress it.
fn compute_frame_delta(
    prev: &[u8],
    curr: &[u8],
    label: &str,
) -> FrameDeltaResult {
    assert_eq!(prev.len(), curr.len());

    // Delta in byte space (wrapping subtraction, stored as u8)
    let delta_bytes: Vec<u8> = prev
        .iter()
        .zip(curr.iter())
        .map(|(&p, &c)| c.wrapping_sub(p))
        .collect();

    // Stats
    let total = delta_bytes.len() as f64;
    let zeros = delta_bytes.iter().filter(|&&d| d == 0).count();
    let small = delta_bytes.iter().filter(|&&d| d <= 7 || d >= 249).count(); // ±7

    // Compress the delta frame
    let compressed = codec::encode(&delta_bytes, 0, 0);

    FrameDeltaResult {
        label: label.to_string(),
        delta_bytes,
        compressed,
        zero_pct: zeros as f64 / total * 100.0,
        small_pct: small as f64 / total * 100.0,
    }
}

/// Create a shifted version of the frame (simulates camera pan / motion)
fn shift_frame(encoded: &[u8], width: u32, height: u32, dx: i32, dy: i32) -> Vec<u8> {
    let w = width as i32;
    let h = height as i32;
    let mut shifted = vec![0u8; encoded.len()];

    for y in 0..h {
        for x in 0..w {
            let sx = x - dx;
            let sy = y - dy;
            if sx >= 0 && sx < w && sy >= 0 && sy < h {
                shifted[(y * w + x) as usize] = encoded[(sy * w + sx) as usize];
            }
        }
    }
    shifted
}

/// Create a brightness-adjusted version (simulates lighting change)
fn adjust_brightness(encoded: &[u8], encode_table: &[u8; 256], decode_table: &[u8; 256], delta: i16) -> Vec<u8> {
    encoded
        .iter()
        .map(|&b| {
            let pixel = decode_table[b as usize] as i16;
            let adjusted = (pixel + delta).clamp(0, 255) as u8;
            encode_table[adjusted as usize]
        })
        .collect()
}

// ---------------------------------------------------------------------------
// Dynamic resolution engine
// ---------------------------------------------------------------------------

struct ResolutionTier {
    width: u32,
    height: u32,
    scale_label: String,
    frame_result: FrameResult,
    fps_at_budget: Vec<(f64, f64)>, // (budget_kbps, achievable_fps)
}

fn test_resolution(
    original: &GrayImage,
    target_width: u32,
    target_height: u32,
    scale_label: &str,
    encode_table: &[u8; 256],
    decode_table: &[u8; 256],
) -> ResolutionTier {
    let resized = image::imageops::resize(original, target_width, target_height, FilterType::Triangle);
    let result = encode_frame(&resized, encode_table, decode_table);

    let compressed_bytes = result.scanline_compressed.stats.compressed_bytes as f64;

    // For various bandwidth budgets, how many FPS can we achieve?
    let budgets = [50.0, 100.0, 200.0, 500.0, 1000.0]; // kbps
    let fps_at_budget: Vec<(f64, f64)> = budgets
        .iter()
        .map(|&budget_kbps| {
            let budget_bytes_per_sec = budget_kbps * 1000.0 / 8.0;
            let fps = (budget_bytes_per_sec / compressed_bytes).min(60.0);
            (budget_kbps, fps)
        })
        .collect();

    ResolutionTier {
        width: target_width,
        height: target_height,
        scale_label: scale_label.to_string(),
        frame_result: result,
        fps_at_budget,
    }
}

// ---------------------------------------------------------------------------
// MP4 frame extraction via openh264
// ---------------------------------------------------------------------------

fn extract_mp4_frames(path: &Path, max_frames: usize) -> (Vec<GrayImage>, f64) {
    use mp4::TrackType;
    use openh264::formats::YUVSource;

    let file = std::fs::File::open(path).expect("Failed to open MP4");
    let file_size = file.metadata().unwrap().len();
    let reader = std::io::BufReader::new(file);

    let mp4_reader = mp4::Mp4Reader::read_header(reader, file_size)
        .expect("Failed to read MP4 header");

    // Find the video track
    let video_track_id = mp4_reader
        .tracks()
        .values()
        .find(|t| t.track_type().unwrap_or(TrackType::Audio) == TrackType::Video)
        .expect("No video track found")
        .track_id();

    let track = &mp4_reader.tracks()[&video_track_id];
    let width = track.width() as u32;
    let height = track.height() as u32;
    let sample_count = track.sample_count();
    let duration_s = track.duration().as_secs_f64();
    let fps = if duration_s > 0.0 { sample_count as f64 / duration_s } else { 30.0 };

    // Extract SPS/PPS from the container
    let sps = track.sequence_parameter_set().ok().map(|s| s.to_vec());
    let pps = track.picture_parameter_set().ok().map(|s| s.to_vec());

    println!("  Video track: {}x{}, {} samples, {:.1}fps, {:.1}s",
        width, height, sample_count, fps, duration_s);

    // Re-open for sample reading
    let file2 = std::fs::File::open(path).unwrap();
    let reader2 = std::io::BufReader::new(file2);
    let mut mp4_reader2 = mp4::Mp4Reader::read_header(reader2, file_size).unwrap();

    let mut decoder = openh264::decoder::Decoder::new()
        .expect("Failed to create H.264 decoder");

    // Feed SPS and PPS first so decoder can handle P/B frames
    if let Some(ref sps_data) = sps {
        let mut nalu = vec![0x00, 0x00, 0x00, 0x01];
        nalu.extend_from_slice(sps_data);
        let _ = decoder.decode(&nalu);
    }
    if let Some(ref pps_data) = pps {
        let mut nalu = vec![0x00, 0x00, 0x00, 0x01];
        nalu.extend_from_slice(pps_data);
        let _ = decoder.decode(&nalu);
    }

    let mut frames: Vec<GrayImage> = Vec::new();
    let limit = max_frames.min(sample_count as usize);

    for sample_idx in 1..=limit as u32 {
        let sample = match mp4_reader2.read_sample(video_track_id, sample_idx) {
            Ok(Some(s)) => s,
            Ok(None) => break,
            Err(_) => continue,
        };

        let avcc_data = &sample.bytes;
        let annexb = avcc_to_annexb(avcc_data);

        match decoder.decode(&annexb) {
            Ok(Some(yuv)) => {
                let (yw, yh) = yuv.dimensions();
                let (y_stride, _, _) = yuv.strides();
                let y_data = yuv.y();

                let mut gray_pixels = Vec::with_capacity(yw * yh);
                for row in 0..yh {
                    let start = row * y_stride;
                    let end = start + yw;
                    if end <= y_data.len() {
                        gray_pixels.extend_from_slice(&y_data[start..end]);
                    }
                }

                if let Some(img) = GrayImage::from_raw(yw as u32, yh as u32, gray_pixels) {
                    frames.push(img);
                }
            }
            _ => {}
        }
    }

    println!("  Decoded {} / {} frames", frames.len(), limit);
    (frames, fps)
}

/// Convert AVCC (length-prefixed) NALUs to Annex B (start-code prefixed)
fn avcc_to_annexb(data: &[u8]) -> Vec<u8> {
    let mut out = Vec::with_capacity(data.len() + 64);
    let mut pos = 0;

    while pos + 4 <= data.len() {
        let nalu_len = u32::from_be_bytes([
            data[pos], data[pos + 1], data[pos + 2], data[pos + 3],
        ]) as usize;
        pos += 4;

        if pos + nalu_len > data.len() {
            break;
        }

        // Annex B start code
        out.extend_from_slice(&[0x00, 0x00, 0x00, 0x01]);
        out.extend_from_slice(&data[pos..pos + nalu_len]);
        pos += nalu_len;
    }

    out
}

// ---------------------------------------------------------------------------
// GIF writer (grayscale, universally playable)
// ---------------------------------------------------------------------------

fn write_gif(path: &Path, frames: &[Vec<u8>], width: u32, height: u32, fps: f64) {
    use gif::{Encoder, Frame, Repeat};
    use std::fs::File;

    let mut file = File::create(path).expect("Failed to create GIF");

    // Build a 256-entry grayscale palette
    let mut palette = Vec::with_capacity(256 * 3);
    for i in 0..=255u8 {
        palette.push(i);
        palette.push(i);
        palette.push(i);
    }

    let mut encoder = Encoder::new(&mut file, width as u16, height as u16, &palette)
        .expect("Failed to create GIF encoder");
    encoder.set_repeat(Repeat::Infinite).expect("Failed to set repeat");

    // GIF delay is in centiseconds (1/100th of a second)
    let delay_cs = (100.0 / fps).round() as u16;

    for frame_data in frames {
        let mut frame = Frame::default();
        frame.width = width as u16;
        frame.height = height as u16;
        frame.delay = delay_cs;
        frame.buffer = std::borrow::Cow::Borrowed(frame_data);
        encoder.write_frame(&frame).expect("Failed to write GIF frame");
    }
}

// ---------------------------------------------------------------------------
// Real video analysis
// ---------------------------------------------------------------------------

fn analyze_video_frames(
    frames: &[GrayImage],
    encode_table: &[u8; 256],
    decode_table: &[u8; 256],
    parent: &Path,
    stem: &str,
    fps: f64,
) {
    if frames.is_empty() {
        println!("  No frames to analyze.");
        return;
    }

    let (w, h) = frames[0].dimensions();
    println!("\n=== REAL VIDEO ANALYSIS ({} frames, {}x{}) ===\n", frames.len(), w, h);

    // Encode all frames
    let mut encoded_frames: Vec<Vec<u8>> = Vec::new();
    let t = Instant::now();
    for frame in frames {
        let pixels: Vec<u8> = frame.as_raw().to_vec();
        let encoded: Vec<u8> = pixels.iter().map(|&p| encode_table[p as usize]).collect();
        encoded_frames.push(encoded);
    }
    let encode_ms = t.elapsed().as_secs_f64() * 1000.0;
    let total_pixels: usize = encoded_frames.iter().map(|f| f.len()).sum();
    println!("  Encoded {} frames ({} pixels) in {:.1}ms ({:.0} pixels/ms)",
        encoded_frames.len(), total_pixels, encode_ms, total_pixels as f64 / encode_ms);

    // Keyframe compression (first frame, scanline delta)
    let keyframe_compressed = codec::encode(&encoded_frames[0], w, 1);
    println!("\n  Keyframe (frame 0):");
    println!("    Raw:        {:>10} bytes", encoded_frames[0].len());
    println!("    Compressed: {:>10} bytes ({:.2} bits/pixel)",
        keyframe_compressed.stats.compressed_bytes,
        keyframe_compressed.stats.bits_per_sample,
    );

    // Frame-to-frame delta analysis
    if encoded_frames.len() >= 2 {
        println!("\n  Frame-to-frame deltas:");
        println!("    {:>6}  {:>10}  {:>8}  {:>7}  {:>7}",
            "Frame", "Compressed", "Bits/px", "Zero%", "Small%");
        println!("    {}  {}  {}  {}  {}",
            "-".repeat(6), "-".repeat(10), "-".repeat(8), "-".repeat(7), "-".repeat(7));

        let mut total_delta_bytes: u64 = 0;
        let mut frame_delta_sizes: Vec<u64> = Vec::new();

        for i in 1..encoded_frames.len() {
            let prev = &encoded_frames[i - 1];
            let curr = &encoded_frames[i];

            let delta_bytes: Vec<u8> = prev.iter()
                .zip(curr.iter())
                .map(|(&p, &c)| c.wrapping_sub(p))
                .collect();

            let total = delta_bytes.len() as f64;
            let zeros = delta_bytes.iter().filter(|&&d| d == 0).count() as f64;
            let small = delta_bytes.iter().filter(|&&d| d <= 7 || d >= 249).count() as f64;

            let compressed = codec::encode(&delta_bytes, w, 1);
            let comp_bytes = compressed.stats.compressed_bytes;
            total_delta_bytes += comp_bytes;
            frame_delta_sizes.push(comp_bytes);

            // Print first 10, last 3, and summary
            if i <= 10 || i > encoded_frames.len() - 3 {
                println!("    {:>6}  {:>10}  {:>7.2}  {:>6.1}  {:>6.1}",
                    i, comp_bytes, compressed.stats.bits_per_sample,
                    zeros / total * 100.0, small / total * 100.0);
            } else if i == 11 {
                println!("    {:>6}  ...", "...");
            }
        }

        let num_deltas = (encoded_frames.len() - 1) as f64;
        let avg_delta_bytes = total_delta_bytes as f64 / num_deltas;
        let min_delta = frame_delta_sizes.iter().min().unwrap_or(&0);
        let max_delta = frame_delta_sizes.iter().max().unwrap_or(&0);

        println!("\n  Delta frame stats:");
        println!("    Average: {:.0} bytes/frame", avg_delta_bytes);
        println!("    Min:     {} bytes", min_delta);
        println!("    Max:     {} bytes", max_delta);

        // Bandwidth estimate for real video
        println!("\n  Real video bandwidth (keyframe every 30 frames):");

        for &fps in &[10.0f64, 15.0, 24.0] {
            let gop_frames = 30.0f64.min(encoded_frames.len() as f64);
            let keyframe_bytes = keyframe_compressed.stats.compressed_bytes as f64;
            let delta_per_frame = avg_delta_bytes;
            let gop_bytes = keyframe_bytes + (gop_frames - 1.0) * delta_per_frame;
            let gop_seconds = gop_frames / fps;
            let kbps = gop_bytes * 8.0 / gop_seconds / 1000.0;
            let kbs = gop_bytes / gop_seconds / 1000.0;
            println!("    {}x{} @ {:.0}fps: {:.1} kbps ({:.1} KB/s)",
                w, h, fps, kbps, kbs);
        }

        // Resolution tiers with real delta data
        println!("\n  Dynamic resolution with real frame deltas:");
        println!("    {:>10}  {:>10}  {:>10}  {:>10}  {:>10}",
            "Resolution", "Keyframe", "Avg Delta", "@10fps", "@15fps");
        println!("    {}  {}  {}  {}  {}",
            "-".repeat(10), "-".repeat(10), "-".repeat(10), "-".repeat(10), "-".repeat(10));

        for &(tw, th) in &[(w, h), (320, 320*h/w), (160, 160*h/w), (80, 80*h/w)] {
            let tw = tw.max(1);
            let th = th.max(1);
            // Resize and encode first two frames to get real delta ratio
            let f0 = image::imageops::resize(&frames[0], tw, th, FilterType::Triangle);
            let f1 = if frames.len() > 1 {
                image::imageops::resize(&frames[1], tw, th, FilterType::Triangle)
            } else {
                f0.clone()
            };

            let enc0: Vec<u8> = f0.as_raw().iter().map(|&p| encode_table[p as usize]).collect();
            let enc1: Vec<u8> = f1.as_raw().iter().map(|&p| encode_table[p as usize]).collect();

            let key_c = codec::encode(&enc0, tw, 1);
            let delta: Vec<u8> = enc0.iter().zip(enc1.iter()).map(|(&p, &c)| c.wrapping_sub(p)).collect();
            let delta_c = codec::encode(&delta, tw, 1);

            let key_bytes = key_c.stats.compressed_bytes as f64;
            let delta_bytes = delta_c.stats.compressed_bytes as f64;

            for &fps in &[10.0, 15.0] {
                let gop_bytes = key_bytes + 29.0 * delta_bytes;
                let kbps = gop_bytes * 8.0 / (30.0 / fps) / 1000.0;

                if fps == 10.0 {
                    println!("    {:>4}x{:<5}  {:>8.0} B  {:>8.0} B  {:>7.1}kbps  {:>7.1}kbps",
                        tw, th, key_bytes, delta_bytes, kbps,
                        gop_bytes * 8.0 / (30.0 / 15.0) / 1000.0);
                    break;
                }
            }
        }
    }

    // === OUTPUT VIDEO FILES ===
    println!("\n  Writing output videos...");

    // Output at multiple resolution tiers
    let output_tiers: Vec<(u32, u32, &str)> = {
        let mut t = Vec::new();
        if w >= 1280 && h >= 720 {
            t.push((1280, 720, "720p"));
        }
        if w >= 854 && h >= 480 {
            t.push((854, 480, "480p"));
        }
        // Always output a 320-wide version
        let h320 = (320 * h / w).max(1);
        t.push((320, h320, "320w"));
        t
    };

    for &(tw, th, label) in &output_tiers {
        let t = Instant::now();

        // Resize, encode through ternsig, decode back — for every frame
        let mut output_frames: Vec<Vec<u8>> = Vec::with_capacity(frames.len());
        for frame in frames {
            let resized = image::imageops::resize(frame, tw, th, FilterType::Triangle);
            let enc: Vec<u8> = resized.as_raw().iter().map(|&p| encode_table[p as usize]).collect();
            let dec: Vec<u8> = enc.iter().map(|&b| decode_table[b as usize]).collect();
            output_frames.push(dec);
        }

        let process_ms = t.elapsed().as_secs_f64() * 1000.0;

        // Write as animated GIF (universally playable)
        let gif_path = parent.join(format!("{}_ternsig_{}.gif", stem, label));
        write_gif(&gif_path, &output_frames, tw, th, fps);

        let gif_size = std::fs::metadata(&gif_path).map(|m| m.len()).unwrap_or(0);
        println!("    {} — {}x{}, {} frames, {:.0} KB ({:.1}ms encode)",
            gif_path.display(), tw, th, output_frames.len(),
            gif_size as f64 / 1024.0, process_ms);
    }

    // Save first frame as PNG for quick preview
    let decoded_pixels: Vec<u8> = encoded_frames[0].iter()
        .map(|&b| decode_table[b as usize])
        .collect();
    let quantized_path = parent.join(format!("{}_frame0_quantized.png", stem));
    if let Some(img) = GrayImage::from_raw(w, h, decoded_pixels) {
        img.save(&quantized_path).ok();
        println!("    {}", quantized_path.display());
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
        eprintln!("Usage: ternsig-video <input.png|jpg|mp4>");
        eprintln!("  Images: encodes to ternary signals, tests spatial compression.");
        eprintln!("  Video:  extracts frames, tests real temporal compression.");
        std::process::exit(1);
    };

    let input = Path::new(&input_path);
    let stem = input.file_stem().unwrap().to_string_lossy();
    let parent = Path::new("C:/Users/silen/Downloads");
    let ext = input.extension().and_then(|e| e.to_str()).unwrap_or("").to_lowercase();

    // Build encode/decode tables
    let encode_table = build_pixel_encode_table();
    let decode_table = build_pixel_decode_table();

    // Print the effective gray levels
    let mut gray_levels: Vec<u8> = (0..=255u8)
        .map(|p| decode_table[encode_table[p as usize] as usize])
        .collect();
    gray_levels.sort();
    gray_levels.dedup();

    if ext == "mp4" || ext == "m4v" || ext == "mov" {
        // --- VIDEO MODE ---
        println!("Loading video...");
        let (frames, fps) = extract_mp4_frames(input, 300); // up to 300 frames (~10s)
        if frames.is_empty() {
            eprintln!("No frames decoded. Is this H.264?");
            std::process::exit(1);
        }

        let (w, h) = frames[0].dimensions();
        println!("  {}x{}, {} distinct gray levels", w, h, gray_levels.len());

        // Run image analysis on first frame
        println!("\n=== FIRST FRAME (SPATIAL) ===\n");
        let frame = encode_frame(&frames[0], &encode_table, &decode_table);
        println!("  PSNR:           {:.2} dB", frame.psnr);
        println!("  Distinct vals:  {} / 256", frame.distinct_values);
        println!("  Raw pixels:           {:>10} bytes", frame.raw_pixels);
        println!("  Scanline compressed:  {:>10} bytes ({:.2} bits/pixel)",
            frame.scanline_compressed.stats.compressed_bytes,
            frame.scanline_compressed.stats.bits_per_sample,
        );
        println!("  Compression ratio:    {:>9.1}%",
            frame.scanline_compressed.stats.compressed_bytes as f64 / frame.raw_pixels as f64 * 100.0,
        );

        // Run real temporal analysis
        analyze_video_frames(&frames, &encode_table, &decode_table, parent, &stem, fps);

        println!("\n  Done.");
    } else {
        // --- IMAGE MODE ---
        println!("Loading image...");
        let img = ImageReader::open(input)
            .expect("Failed to open image")
            .decode()
            .expect("Failed to decode image");
        let gray = img.to_luma8();
        let (w, h) = gray.dimensions();
        println!("  {}x{} grayscale ({} pixels)", w, h, w * h);
        println!("  {} distinct gray levels after round-trip", gray_levels.len());

    // Build encode/decode tables
    let encode_table = build_pixel_encode_table();
    let decode_table = build_pixel_decode_table();

    // Print the effective gray levels
    let mut gray_levels: Vec<u8> = (0..=255u8)
        .map(|p| decode_table[encode_table[p as usize] as usize])
        .collect();
    gray_levels.sort();
    gray_levels.dedup();
    println!("  {} distinct gray levels after round-trip", gray_levels.len());

    // =========================================================================
    // SCENARIO 1: Single frame — scanline delta compression
    // =========================================================================
    println!("\n=== SCENARIO 1: SCANLINE DELTA (SPATIAL) ===\n");

    let t = Instant::now();
    let frame = encode_frame(&gray, &encode_table, &decode_table);
    let encode_ms = t.elapsed().as_secs_f64() * 1000.0;

    // Save outputs
    let quantized_path = parent.join(format!("{}_quantized.png", stem));
    let quantized_img = GrayImage::from_raw(w, h, frame.decoded_pixels.clone())
        .expect("Failed to create quantized image");
    quantized_img.save(&quantized_path).expect("Failed to save quantized image");

    println!("  Encoding:       {:.1}ms ({:.0} pixels/ms)", encode_ms, frame.raw_pixels as f64 / encode_ms);
    println!("  PSNR:           {:.2} dB", frame.psnr);
    println!("  Distinct vals:  {} / 256", frame.distinct_values);
    println!();
    println!("  Raw pixels:           {:>10} bytes", frame.raw_pixels);
    println!("  Ternsig bytes:        {:>10} bytes (1 byte/pixel)", frame.encoded_bytes.len());
    println!("  Scanline compressed:  {:>10} bytes ({:.2} bits/pixel)",
        frame.scanline_compressed.stats.compressed_bytes,
        frame.scanline_compressed.stats.bits_per_sample,
    );
    println!("  Compression ratio:    {:>9.1}%",
        frame.scanline_compressed.stats.compressed_bytes as f64 / frame.raw_pixels as f64 * 100.0,
    );

    // Delta distribution within the frame
    let mut zero_deltas = 0u64;
    let mut small_deltas = 0u64;
    for i in 1..frame.encoded_bytes.len() {
        let d = frame.encoded_bytes[i] as i16 - frame.encoded_bytes[i - 1] as i16;
        if d == 0 { zero_deltas += 1; }
        if d.abs() <= 7 { small_deltas += 1; }
    }
    let total_d = (frame.encoded_bytes.len() - 1) as f64;
    println!();
    println!("  Scanline delta stats:");
    println!("    Zero deltas:  {:.1}%", zero_deltas as f64 / total_d * 100.0);
    println!("    Small (±7):   {:.1}%", small_deltas as f64 / total_d * 100.0);

    println!("\n  Quantized image: {}", quantized_path.display());

    // =========================================================================
    // SCENARIO 2: Frame delta — temporal compression
    // =========================================================================
    println!("\n=== SCENARIO 2: FRAME DELTA (TEMPORAL) ===\n");

    // Test various realistic frame-to-frame changes
    let scenarios: Vec<(&str, Vec<u8>)> = vec![
        ("Static (identical frame)", frame.encoded_bytes.clone()),
        ("1px pan right", shift_frame(&frame.encoded_bytes, w, h, 1, 0)),
        ("2px pan right", shift_frame(&frame.encoded_bytes, w, h, 2, 0)),
        ("5px pan right", shift_frame(&frame.encoded_bytes, w, h, 5, 0)),
        ("1px diagonal", shift_frame(&frame.encoded_bytes, w, h, 1, 1)),
        ("Brightness +5", adjust_brightness(&frame.encoded_bytes, &encode_table, &decode_table, 5)),
        ("Brightness +20", adjust_brightness(&frame.encoded_bytes, &encode_table, &decode_table, 20)),
    ];

    println!("  {:.<35} {:>10}  {:>8}  {:>6}  {:>6}",
        "Scenario", "Compressed", "Bits/px", "Zero%", "Small%");
    println!("  {}  {}  {}  {}  {}",
        "-".repeat(35), "-".repeat(10), "-".repeat(8), "-".repeat(6), "-".repeat(6));

    for (label, next_frame) in &scenarios {
        let delta = compute_frame_delta(&frame.encoded_bytes, next_frame, label);
        println!("  {:.<35} {:>10}  {:>7.2}  {:>5.1}  {:>5.1}",
            label,
            delta.compressed.stats.compressed_bytes,
            delta.compressed.stats.bits_per_sample,
            delta.zero_pct,
            delta.small_pct,
        );
    }

    // Keyframe + delta cost analysis
    let static_delta = compute_frame_delta(&frame.encoded_bytes, &frame.encoded_bytes, "static");
    let pan1_delta = compute_frame_delta(
        &frame.encoded_bytes,
        &shift_frame(&frame.encoded_bytes, w, h, 1, 0),
        "1px pan",
    );

    println!("\n  Video stream cost estimate (keyframe every 30 frames):");
    println!("  At 10 FPS = keyframe every 3 seconds\n");

    let keyframe_bytes = frame.scanline_compressed.stats.compressed_bytes as f64;

    for (label, delta_bytes) in [
        ("Static scene", static_delta.compressed.stats.compressed_bytes as f64),
        ("Slow pan (1px/frame)", pan1_delta.compressed.stats.compressed_bytes as f64),
    ] {
        // 1 keyframe + 29 delta frames per GOP
        let gop_bytes = keyframe_bytes + 29.0 * delta_bytes;
        let gop_seconds = 30.0 / 10.0; // 30 frames at 10fps = 3 seconds
        let kbps = gop_bytes * 8.0 / gop_seconds / 1000.0;
        let kbs = gop_bytes / gop_seconds / 1000.0;
        println!("    {}", label);
        println!("      Keyframe: {:.0} bytes, Delta: {:.0} bytes", keyframe_bytes, delta_bytes);
        println!("      GOP (30 frames): {:.0} bytes", gop_bytes);
        println!("      Bandwidth: {:.1} kbps ({:.1} KB/s)", kbps, kbs);
        println!();
    }

    // =========================================================================
    // DYNAMIC RESOLUTION ENGINE
    // =========================================================================
    println!("=== DYNAMIC RESOLUTION ENGINE ===\n");
    println!("  Given a bandwidth budget, what resolution and FPS can we achieve?\n");

    // Test at multiple resolution tiers
    let tiers: Vec<(u32, u32, &str)> = {
        let mut t = vec![
            (w, h, "Native"),
        ];
        if w >= 1280 {
            t.push((1280, (1280 * h / w), "1280w"));
        }
        if w >= 640 {
            t.push((640, (640 * h / w), "640w"));
        }
        t.push((320, (320 * h / w).max(1), "320w"));
        t.push((160, (160 * h / w).max(1), "160w"));
        t.push((80, (80 * h / w).max(1), "80w"));
        t
    };

    let mut resolution_results: Vec<ResolutionTier> = Vec::new();
    for (tw, th, label) in &tiers {
        let tier = test_resolution(&gray, *tw, *th, label, &encode_table, &decode_table);
        resolution_results.push(tier);
    }

    // Print resolution tier table
    println!("  {:>10}  {:>10}  {:>8}  {:>10}  {:>8}",
        "Resolution", "Pixels", "PSNR", "Compressed", "Bits/px");
    println!("  {}  {}  {}  {}  {}",
        "-".repeat(10), "-".repeat(10), "-".repeat(8), "-".repeat(10), "-".repeat(8));

    for tier in &resolution_results {
        println!("  {:>4}x{:<5}  {:>10}  {:>7.1}  {:>10}  {:>7.2}",
            tier.width, tier.height,
            tier.width * tier.height,
            tier.frame_result.psnr,
            tier.frame_result.scanline_compressed.stats.compressed_bytes,
            tier.frame_result.scanline_compressed.stats.bits_per_sample,
        );
    }

    // Dynamic resolution selection for various budgets
    println!("\n  Achievable FPS at each resolution tier:\n");

    let budgets_kbps = [50.0, 100.0, 200.0, 500.0, 1000.0];

    // Header
    print!("  {:>10}", "Budget →");
    for b in &budgets_kbps {
        print!("  {:>8}", format!("{:.0}kbps", b));
    }
    println!();
    print!("  {:>10}", "----------");
    for _ in &budgets_kbps {
        print!("  {:>8}", "--------");
    }
    println!();

    for tier in &resolution_results {
        print!("  {:>4}x{:<5}", tier.width, tier.height);
        for &(_, fps) in &tier.fps_at_budget {
            if fps >= 1.0 {
                print!("  {:>7.1}f", fps);
            } else {
                print!("  {:>8}", "<1");
            }
        }
        println!();
    }

    // Recommend best tier for each budget at 10fps target
    println!("\n  Recommended resolution at 10 FPS target:\n");
    for &budget in &budgets_kbps {
        let budget_bytes_per_frame = budget * 1000.0 / 8.0 / 10.0;
        let best = resolution_results
            .iter()
            .filter(|t| (t.frame_result.scanline_compressed.stats.compressed_bytes as f64) <= budget_bytes_per_frame)
            .max_by_key(|t| t.width * t.height);

        match best {
            Some(tier) => println!("    {:>6.0} kbps → {}x{} ({:.0} bytes/frame, {:.1} KB/s)",
                budget, tier.width, tier.height,
                tier.frame_result.scanline_compressed.stats.compressed_bytes,
                budget / 8.0,
            ),
            None => println!("    {:>6.0} kbps → Below minimum resolution", budget),
        }
    }

    // Save all resolution tier images
    println!("\n  Output images:");
    println!("    {}", quantized_path.display());
    for tier in &resolution_results {
        if tier.width != w || tier.height != h {
            let tier_path = parent.join(format!("{}_{}x{}.png", stem, tier.width, tier.height));
            let tier_img = GrayImage::from_raw(
                tier.width,
                tier.height,
                tier.frame_result.decoded_pixels.clone(),
            )
            .expect("Failed to create tier image");
            tier_img.save(&tier_path).expect("Failed to save tier image");
            println!("    {}", tier_path.display());
        }
    }

    println!("\n  Done.");
    }
}
