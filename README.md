# Ternsig Audio Experiment

Can audio survive a round-trip through ternary signal encoding?

```
PCM (i16) → PackedSignal (1 byte) → PCM (i16)
```

This experiment encodes raw audio samples into Blackfall's 1-byte ternary signal format (`s = p × m × k`), then decodes them back. The result sounds like a vinyl record — structured quantization noise produces crackles, not hiss. 41 discrete signal levels replace 65,536 PCM levels, and it still sounds like music.

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

| Format | Size | Bytes/Sample |
|--------|------|-------------|
| Source MP3 | 44 MB | ~0.7 (compressed) |
| PCM i16 (WAV) | 120 MB | 2 |
| **Ternsig encoded** | **60 MB** | **1** |

Ternsig is always exactly 50% of PCM — 1 byte per sample vs 2.

### Signal Quality

| Decode Method | SNR | Character |
|---------------|-----|-----------|
| Raw (no smoothing) | 14.26 dB | Vinyl crackle — stepped quantization, structured noise |
| Linear smooth (2 pass) | 16.76 dB | Softened edges, less crackle, slightly mushier |
| Cubic Hermite smooth (2 pass) | 14.86 dB | Preserves transients, rounds off harshest steps |

14 dB SNR is poor on paper. In practice, the log-spaced LUT means the quantization noise is signal-correlated and harmonically structured — the ear perceives warmth and texture, not digital destruction. The same psychoacoustic principle that makes vinyl sound warm despite its physical limitations.

### Realtime Performance

| Operation | Time | Throughput |
|-----------|------|-----------|
| MP3 decode | 822 ms | 76,627 samples/ms |
| **Encode (PCM → ternsig)** | **398 ms** | **158,374 samples/ms** |
| **Decode (ternsig → PCM)** | **80 ms** | **792,481 samples/ms** |
| Smooth (linear, 2 pass) | 137 ms | 459,970 samples/ms |
| Smooth (cubic, 2 pass) | 411 ms | 153,303 samples/ms |

For 63 million samples (23 min of audio):
- **Encoding runs at 3.6x realtime** at 44.1 kHz — fast enough for live broadcast
- **Decoding runs at 18x realtime** — effectively free on the receiver

At 8 kHz (speech/call quality), encoding would run at ~20x realtime, decoding at ~99x realtime.

### Livestream Bandwidth

The number that matters for Lighthouse Radio:

| Format | Sample Rate | Bandwidth | KB/s |
|--------|-------------|-----------|------|
| **Ternsig** | **8,000 Hz** | **64 kbps** | **8.0** |
| **Ternsig** | **22,050 Hz** | **176 kbps** | **22.1** |
| **Ternsig** | **44,100 Hz** | **353 kbps** | **44.1** |
| PCM i16 | 8,000 Hz | 128 kbps | 16.0 |
| PCM i16 | 22,050 Hz | 353 kbps | 44.1 |
| PCM i16 | 44,100 Hz | 706 kbps | 88.2 |
| Opus (reference) | 48,000 Hz | 64 kbps | 8.0 |

Ternsig at 8 kHz matches Opus bandwidth (64 kbps) while being trivially simple — no codec state, no frame assembly, no bitstream packing. One byte in, one byte out. The tradeoff is fidelity: Opus at 64 kbps sounds like a phone call. Ternsig at 64 kbps sounds like a phone call on a record player.

For music broadcast, 22 kHz (176 kbps) is the sweet spot — AM radio frequency range, reasonable bandwidth, covers the harmonic content that matters.

---

## How It Works

### Encoding (PCM → PackedSignal)

1. Scale the i16 sample [-32768, 32767] to the current range [-65025, 65025]
2. Binary search a sorted table of all 256 PackedSignal current values for the nearest match
3. Store the 1-byte result

### Decoding (PackedSignal → PCM)

1. Look up the byte in a precomputed 256-entry current LUT (1 KB, fits in L1 cache)
2. Scale the current value back to i16 range

### Smoothing (post-decode)

Three output modes:
- **Raw**: direct decode, no processing. Maximum crackle, maximum character.
- **Linear**: 3-tap weighted average (25/50/25) applied in multiple passes. Kills high-frequency quantization noise.
- **Cubic Hermite**: Catmull-Rom interpolation blended 70/30 with the original. Follows natural curves while preserving transient attacks.

Smoothing is receiver-side only — the encoded stream is identical regardless of decode mode. A receiver can choose its own smoothing preference.

---

## Usage

```bash
cargo build --release
./target/release/ternsig-audio-experiment <input.mp3>
```

Outputs (next to the input file):
- `*_original.wav` — clean PCM decode of the MP3 (baseline)
- `*_encoded.ternsig` — raw ternary bytes (1 byte/sample)
- `*_decoded.wav` — round-tripped, no smoothing
- `*_smoothed_linear.wav` — linear interpolation (2 passes)
- `*_smoothed_cubic.wav` — cubic Hermite interpolation (2 passes)

---

## Why This Matters

This is the first proof that Blackfall's ternary signal encoding — designed for neuromorphic spike propagation — can carry audio. The same `PackedSignal` type used in neuropool membrane integration, databank pattern storage, and ternsig-oscillator synthesis can now encode and decode real-world audio.

Implications:
- **Lighthouse Radio**: live audio streaming at 8-44 KB/s per listener over LRIN mesh
- **Voice calls**: ternsig-native voice at 8 kHz / 64 kbps with zero codec complexity
- **Phonetic voice engine**: speech synthesis and recognition share the same signal substrate as the neural processing
- **One signal type everywhere**: storage, compute, transport, and audio all speak PackedSignal

---

*Blackfall Labs — Every system starts with a signal.*
