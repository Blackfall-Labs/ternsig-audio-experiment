# Ternsig Audio/Video Experiment

Can audio and video survive a round-trip through ternary signal encoding?

```
PCM (i16) → PackedSignal (1 byte) → Delta + Huffman → Wire → Decompress → PCM (i16)
```

Yes. And it beats MP3 on bandwidth while sounding like a vinyl record.

---

## The Blackfall Signal Equation

Every sample is encoded as:

```
s = polarity × magnitude × multiplier
```

Packed into a single byte: `[pol:2 | mag:3 | mul:3]`

- **Polarity** (2 bits): +1, -1, or 0
- **Magnitude** (3 bits): index into log-scale LUT `[0, 1, 4, 16, 32, 64, 128, 255]`
- **Multiplier** (3 bits): same LUT

The decoded value `polarity × LUT[mag] × LUT[mul]` ranges from -65,025 to +65,025. Only 41 of the 256 possible byte values produce distinct current levels — an effective bit depth of 5.4 bits.

---

## Results (Mozart — Bastien und Bastienne K.50)

Source: 23 minutes of classical opera. MP3 at 44,100 Hz mono.

### File Sizes

| Format | Size | Bits/Sample |
|--------|------|-------------|
| PCM i16 (WAV) | 120 MB | 16.0 |
| Source MP3 | 45.7 MB | 5.9 |
| Ternsig raw | 60 MB | 8.0 |
| **Ternsig compressed (TSC1)** | **21.1 MB** | **2.68** |

Ternsig compressed is **46% the size of the MP3**. Less than half.

### Signal Quality

| Decode Method | SNR | Character |
|---------------|-----|-----------|
| Raw (no smoothing) | 14.26 dB | Vinyl crackle — stepped quantization, structured noise |
| Linear smooth (2 pass) | 16.76 dB | Softened edges, less crackle, slightly mushier |
| Cubic Hermite smooth (2 pass) | 14.86 dB | Preserves transients, rounds off harshest steps |

14 dB SNR is poor on paper. In practice, the log-spaced LUT means the quantization noise is signal-correlated and harmonically structured — the ear perceives warmth and texture, not digital destruction. The same psychoacoustic principle that makes vinyl sound warm despite its physical limitations: stepped etches in a groove, not a smooth continuous signal.

---

## Realtime Performance

All numbers from 63 million samples (23 min of audio), release build:

| Operation | Time | Throughput | Realtime Factor |
|-----------|------|-----------|-----------------|
| Encode (PCM → ternsig) | 421 ms | 149,537 samples/ms | **3,381x** |
| Compress (delta + Huffman) | 2,033 ms | 30,992 samples/ms | **702x** |
| Decompress | 832 ms | 75,730 samples/ms | **1,717x** |
| Decode (ternsig → PCM) | 80 ms | 792,481 samples/ms | **17,970x** |
| Smooth (linear, 2 pass) | 137 ms | 459,970 samples/ms | 10,430x |
| Smooth (cubic, 2 pass) | 411 ms | 153,303 samples/ms | 3,475x |

**Realtime is not a concern.** At 44.1 kHz you need 44.1 samples/ms. The slowest operation (compression) runs at 702x realtime. A single core could encode 700 simultaneous radio stations.

For voice calls at 8 kHz, all numbers scale proportionally — compression would run at ~3,900x realtime. You could run a voice call over a dial-up modem at **~2.1 KB/s**.

---

## Livestream Bandwidth

The numbers that matter for Lighthouse Radio:

### Compressed (TSC1 — Delta + Huffman)

| Sample Rate | Bandwidth | KB/s | Use Case |
|-------------|-----------|------|----------|
| **8,000 Hz** | **~21 kbps** | **~2.7** | Voice calls |
| **22,050 Hz** | **~59 kbps** | **~7.4** | AM radio quality |
| **44,100 Hz** | **~118 kbps** | **~14.8** | Full quality music |

### Comparison at 44.1 kHz Mono

| Format | Bandwidth | vs Ternsig TSC1 |
|--------|-----------|-----------------|
| PCM i16 | 706 kbps | 6.0x more |
| Ternsig raw | 353 kbps | 3.0x more |
| **MP3 (this file)** | **256 kbps** | **2.2x more** |
| Opus 64kbps | 64 kbps | 0.5x (but needs complex codec) |
| **Ternsig TSC1** | **118 kbps** | **baseline** |

Ternsig TSC1 beats MP3 with zero psychoacoustic modeling, no FFT, no MDCT, no bit reservoir. Delta encoding + Huffman coding on integer signal bytes. The encoder and decoder are each under 200 lines of Rust.

---

## Why Delta + Huffman Works So Well

The PackedSignal byte stream is incredibly predictable:

- **57.6%** of adjacent samples are identical (delta = 0)
- **90.7%** of deltas fit within ±7 byte-steps
- Only **432** distinct delta values across 63 million samples
- Shannon entropy: **2.65 bits/sample** (codec achieves 2.68 — near theoretical minimum)

Audio waveforms change smoothly. The log-quantized LUT means small amplitude changes often map to the same byte. Huffman assigns a ~1-bit code to delta=0 (the majority case) and progressively longer codes to rarer deltas. The result: most of the bitstream is just "same as last sample" flags.

---

## Broadcast Architecture with Temporal Fields

For Lighthouse Radio, the receiver side can leverage `temporal-field-rs` as a shared subscriber buffer:

```
Station broadcasts TSC1 stream over LRIN
    ↓
Lighthouse node receives, decompresses once
    ↓
Decoded PackedSignals written into a TemporalField (decay disabled, retention=1.0)
    ↓
All local subscribers read from the same field
    ↓
Each subscriber decodes PackedSignal → PCM independently (smoothing preference per listener)
```

The temporal field acts as a **shared, time-bound ring buffer**. With decay disabled (`retention = 1.0`), it holds the stream window without degradation. The node decompresses the TSC1 bitstream once regardless of subscriber count, writes the samples into the field, and every local listener reads from the same buffer. Decode to PCM is per-listener (17,970x realtime — negligible) and each can apply their own smoothing preference.

This means a Lighthouse node serving 100 listeners to the same station does **one** decompress operation, not 100. The per-listener cost is just the LUT lookup and scaling — the operation that runs at 792,481 samples/ms.

---

## TSC1 File Format

```
[magic: 4 bytes "TSC1"]
[sample_rate: u32 LE]
[channels: u16 LE]
[sample_count: u64 LE]
[first_sample: u8]
[table_size: u16 LE]
[table entries: (symbol: i16 LE, code_len: u8, code_bits: u32 LE) × table_size]
[bitstream: Huffman-coded deltas, MSB first]
```

Header is ~1.6 KB (232 table entries for this test file). The rest is pure bitstream. For streaming, the Huffman table is transmitted once at subscription time, then raw bitstream frames flow continuously.

---

## How It Works

### Encoding (PCM → PackedSignal)

1. Scale the i16 sample [-32768, 32767] to the current range [-65025, 65025]
2. Binary search a sorted table of all 256 PackedSignal current values for the nearest match
3. Store the 1-byte result

### Compression (PackedSignal → TSC1)

1. Delta-encode: store `byte[n] - byte[n-1]` instead of absolute values
2. Count delta frequencies across the stream
3. Build Huffman tree — common deltas get short codes (delta=0 → ~1 bit)
4. Encode all deltas as variable-length Huffman codes into a packed bitstream

### Decompression (TSC1 → PackedSignal)

1. Read Huffman table from header
2. Walk the bitstream, decode each Huffman symbol to a delta
3. Accumulate: `byte[n] = byte[n-1] + delta`
4. Lossless — output is identical to pre-compression PackedSignal bytes

### Decoding (PackedSignal → PCM)

1. Look up the byte in a precomputed 256-entry current LUT (1 KB, fits in L1 cache)
2. Scale the current value back to i16 range

### Smoothing (post-decode, optional)

Three modes, receiver's choice:
- **Raw**: direct decode. Maximum vinyl character.
- **Linear**: 3-tap weighted average (25/50/25), multiple passes. Kills crackle.
- **Cubic Hermite**: Catmull-Rom interpolation blended 70/30 with original. Preserves transients.

---

## Video Encoding

The same PackedSignal encoding works for images and video. A grayscale pixel (0-255) maps directly to signal magnitude — no scaling needed.

### Pipeline

```
Pixel (u8 grayscale) → PackedSignal (1 byte) → Scanline Delta + Huffman → Wire
Frame N delta vs Frame N-1 → Delta + Huffman → Wire
```

### Results (Real MP4, 1440x1080, 30fps)

| Metric | Value |
|--------|-------|
| Gray levels after quantization | 10 |
| PSNR | 18.62 dB |
| Scanline compression | 1.03 bits/pixel (12.9% of raw) |
| Frame-to-frame zeros | 88-100% |
| Encoding speed | 1.58M pixels/ms |

### Video Streaming Bandwidth

| Resolution | @10fps | @15fps |
|-----------|--------|--------|
| 320x240 | 990 kbps | 1.5 Mbps |
| 160x120 | 280 kbps | 420 kbps |
| 80x60 | 77 kbps | 116 kbps |

80x60 at 10fps costs less bandwidth than the audio stream. A video call at potato quality for 77 kbps.

### Dynamic Resolution Engine

Given a bandwidth budget, the encoder picks the highest resolution that fits the target FPS. Latency goes up, resolution drops instantly. Latency recovers, resolution climbs back.

| Budget | Resolution @10fps |
|--------|-------------------|
| 500 kbps | 160x120 |
| 200 kbps | 80x60 |
| 100 kbps | 80x60 at 6.4fps |

### Current Limitations

- Only 10 gray levels — the log-quantized LUT was designed for neural signals, not images. Produces a high-contrast monochrome look. Recognizable but rough.
- openh264 decoder only gets keyframes + some P-frames without full container parsing. More frames would improve temporal compression stats.
- Audio encoding sounds like vinyl (good). Video encoding looks like a surveillance feed from 1997 (functional but ugly). Audio is the stronger use case today.

---

## Usage

### Audio
```bash
cargo build --release
./target/release/ternsig-audio <input.mp3>
```

Outputs:
- `*_original.wav` — clean PCM decode of the MP3 (baseline)
- `*_encoded.ternsig` — raw ternary bytes (1 byte/sample)
- `*_compressed.ternsig-c` — TSC1 compressed (delta + Huffman)
- `*_decoded.wav` — round-tripped, no smoothing
- `*_compressed_decoded.wav` — round-tripped through TSC1 (identical to raw decoded)
- `*_smoothed_linear.wav` — linear interpolation (2 passes)
- `*_smoothed_cubic.wav` — cubic Hermite interpolation (2 passes)

### Video
```bash
./target/release/ternsig-video <input.mp4|png|jpg>
```

For images: spatial compression analysis, quantized output, resolution tiers.
For video: frame extraction, real temporal delta analysis, GIF outputs at 720p/480p/320w.

---

## Why This Matters

This is the first proof that Blackfall's ternary signal encoding — designed for neuromorphic spike propagation — can carry audio and video.

The same `PackedSignal` type used in neuropool membrane integration, databank pattern storage, and ternsig-oscillator synthesis now encodes, compresses, streams, and decodes real-world media. One signal type across the entire stack.

- **Lighthouse Radio**: live audio streaming at 2.7-14.8 KB/s per station over LRIN mesh
- **Voice calls**: ternsig-native voice at ~2.1 KB/s — viable over any connection
- **Video calls**: 77 kbps at 80x60, 280 kbps at 160x120 — functional over mesh
- **Phonetic voice engine**: speech synthesis and recognition share the same signal substrate as the neural processing
- **Temporal field broadcast**: one decompress per station, shared buffer for all subscribers
- **One signal type everywhere**: storage, compute, transport, audio, and video all speak PackedSignal

---

*Blackfall Labs — Every system starts with a signal.*
