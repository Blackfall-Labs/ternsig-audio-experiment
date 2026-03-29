//! Delta + Huffman codec for ternary signal audio streams.
//!
//! Encoding pipeline:
//!   PackedSignal bytes → delta encoding → Huffman coding → bitstream
//!
//! File format (.ternsig-c):
//!   [magic: 4 bytes "TSC1"]
//!   [sample_rate: u32 LE]
//!   [channels: u16 LE]
//!   [sample_count: u64 LE]
//!   [first_sample: u8]
//!   [table_size: u16 LE]          — number of Huffman table entries
//!   [table entries: (i16 LE, u8 code_len, u32 LE code_bits) × table_size]
//!   [bitstream: packed bits, MSB first]

use std::collections::BinaryHeap;
use std::collections::HashMap;
use std::cmp::Reverse;

const MAGIC: &[u8; 4] = b"TSC1";

// ---------------------------------------------------------------------------
// Huffman tree
// ---------------------------------------------------------------------------

#[derive(Debug)]
enum HuffNode {
    Leaf { symbol: i16, freq: u64 },
    Internal { freq: u64, left: Box<HuffNode>, right: Box<HuffNode> },
}

impl HuffNode {
    fn freq(&self) -> u64 {
        match self {
            HuffNode::Leaf { freq, .. } => *freq,
            HuffNode::Internal { freq, .. } => *freq,
        }
    }
}

impl PartialEq for HuffNode {
    fn eq(&self, other: &Self) -> bool {
        self.freq() == other.freq()
    }
}
impl Eq for HuffNode {}

impl PartialOrd for HuffNode {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}
impl Ord for HuffNode {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.freq().cmp(&other.freq())
    }
}

/// Build Huffman code table from frequency counts.
fn build_huffman_table(freq: &HashMap<i16, u64>) -> HashMap<i16, (u32, u8)> {
    if freq.is_empty() {
        return HashMap::new();
    }

    // Single symbol edge case
    if freq.len() == 1 {
        let (&symbol, _) = freq.iter().next().unwrap();
        let mut table = HashMap::new();
        table.insert(symbol, (0u32, 1u8)); // code = 0, len = 1
        return table;
    }

    // Build min-heap
    let mut heap: BinaryHeap<Reverse<HuffNode>> = BinaryHeap::new();
    for (&symbol, &f) in freq {
        heap.push(Reverse(HuffNode::Leaf { symbol, freq: f }));
    }

    // Build tree
    while heap.len() > 1 {
        let Reverse(left) = heap.pop().unwrap();
        let Reverse(right) = heap.pop().unwrap();
        let combined = HuffNode::Internal {
            freq: left.freq() + right.freq(),
            left: Box::new(left),
            right: Box::new(right),
        };
        heap.push(Reverse(combined));
    }

    let root = heap.pop().unwrap().0;

    // Walk tree to assign codes
    let mut table = HashMap::new();
    fn walk(node: &HuffNode, code: u32, len: u8, table: &mut HashMap<i16, (u32, u8)>) {
        match node {
            HuffNode::Leaf { symbol, .. } => {
                table.insert(*symbol, (code, len));
            }
            HuffNode::Internal { left, right, .. } => {
                walk(left, code << 1, len + 1, table);
                walk(right, (code << 1) | 1, len + 1, table);
            }
        }
    }
    walk(&root, 0, 0, &mut table);
    table
}

// ---------------------------------------------------------------------------
// Bitstream writer
// ---------------------------------------------------------------------------

struct BitWriter {
    bytes: Vec<u8>,
    current: u8,
    bit_pos: u8, // bits written into current byte (0-7)
}

impl BitWriter {
    fn new() -> Self {
        Self {
            bytes: Vec::new(),
            current: 0,
            bit_pos: 0,
        }
    }

    fn write_bits(&mut self, code: u32, len: u8) {
        for i in (0..len).rev() {
            let bit = (code >> i) & 1;
            self.current = (self.current << 1) | (bit as u8);
            self.bit_pos += 1;
            if self.bit_pos == 8 {
                self.bytes.push(self.current);
                self.current = 0;
                self.bit_pos = 0;
            }
        }
    }

    fn finish(mut self) -> Vec<u8> {
        if self.bit_pos > 0 {
            // Pad remaining bits with zeros
            self.current <<= 8 - self.bit_pos;
            self.bytes.push(self.current);
        }
        self.bytes
    }
}

// ---------------------------------------------------------------------------
// Bitstream reader
// ---------------------------------------------------------------------------

struct BitReader<'a> {
    data: &'a [u8],
    byte_pos: usize,
    bit_pos: u8, // next bit to read within current byte (7 = MSB, 0 = LSB)
}

impl<'a> BitReader<'a> {
    fn new(data: &'a [u8]) -> Self {
        Self {
            data,
            byte_pos: 0,
            bit_pos: 7,
        }
    }

    fn read_bit(&mut self) -> Option<u8> {
        if self.byte_pos >= self.data.len() {
            return None;
        }
        let bit = (self.data[self.byte_pos] >> self.bit_pos) & 1;
        if self.bit_pos == 0 {
            self.byte_pos += 1;
            self.bit_pos = 7;
        } else {
            self.bit_pos -= 1;
        }
        Some(bit)
    }
}

// ---------------------------------------------------------------------------
// Encode
// ---------------------------------------------------------------------------

pub struct CompressedStream {
    pub data: Vec<u8>,
    pub stats: CompressionStats,
}

pub struct CompressionStats {
    pub raw_bytes: u64,
    pub compressed_bytes: u64,
    pub bits_per_sample: f64,
    pub table_entries: usize,
    pub bitstream_bytes: u64,
    pub header_bytes: u64,
}

pub fn encode(
    samples: &[u8],
    sample_rate: u32,
    channels: u16,
) -> CompressedStream {
    assert!(!samples.is_empty());

    // Delta encode
    let mut deltas: Vec<i16> = Vec::with_capacity(samples.len() - 1);
    for i in 1..samples.len() {
        deltas.push(samples[i] as i16 - samples[i - 1] as i16);
    }

    // Count frequencies
    let mut freq: HashMap<i16, u64> = HashMap::new();
    for &d in &deltas {
        *freq.entry(d).or_insert(0) += 1;
    }

    // Build Huffman table
    let table = build_huffman_table(&freq);

    // Write header
    let mut out: Vec<u8> = Vec::new();

    // Magic
    out.extend_from_slice(MAGIC);

    // Sample rate (u32 LE)
    out.extend_from_slice(&sample_rate.to_le_bytes());

    // Channels (u16 LE)
    out.extend_from_slice(&channels.to_le_bytes());

    // Sample count (u64 LE)
    out.extend_from_slice(&(samples.len() as u64).to_le_bytes());

    // First sample (u8)
    out.push(samples[0]);

    // Table size (u16 LE)
    let table_size = table.len() as u16;
    out.extend_from_slice(&table_size.to_le_bytes());

    // Table entries: (symbol: i16 LE, code_len: u8, code_bits: u32 LE)
    // Sort by symbol for determinism
    let mut table_vec: Vec<(i16, u32, u8)> = table
        .iter()
        .map(|(&sym, &(code, len))| (sym, code, len))
        .collect();
    table_vec.sort_by_key(|&(sym, _, _)| sym);

    for (sym, code, len) in &table_vec {
        out.extend_from_slice(&sym.to_le_bytes());
        out.push(*len);
        out.extend_from_slice(&code.to_le_bytes());
    }

    let header_bytes = out.len() as u64;

    // Encode bitstream
    let mut writer = BitWriter::new();
    for &d in &deltas {
        let &(code, len) = table.get(&d).unwrap();
        writer.write_bits(code, len);
    }
    let bitstream = writer.finish();
    let bitstream_bytes = bitstream.len() as u64;
    out.extend_from_slice(&bitstream);

    let compressed_bytes = out.len() as u64;
    let raw_bytes = samples.len() as u64;

    CompressedStream {
        data: out,
        stats: CompressionStats {
            raw_bytes,
            compressed_bytes,
            bits_per_sample: compressed_bytes as f64 * 8.0 / samples.len() as f64,
            table_entries: table.len(),
            bitstream_bytes,
            header_bytes,
        },
    }
}

// ---------------------------------------------------------------------------
// Decode
// ---------------------------------------------------------------------------

pub struct DecodedStream {
    pub samples: Vec<u8>,
    pub sample_rate: u32,
    pub channels: u16,
}

pub fn decode(data: &[u8]) -> DecodedStream {
    let mut pos = 0;

    // Magic
    assert_eq!(&data[pos..pos + 4], MAGIC, "Invalid magic bytes");
    pos += 4;

    // Sample rate
    let sample_rate = u32::from_le_bytes(data[pos..pos + 4].try_into().unwrap());
    pos += 4;

    // Channels
    let channels = u16::from_le_bytes(data[pos..pos + 2].try_into().unwrap());
    pos += 2;

    // Sample count
    let sample_count = u64::from_le_bytes(data[pos..pos + 8].try_into().unwrap()) as usize;
    pos += 8;

    // First sample
    let first_sample = data[pos];
    pos += 1;

    // Table size
    let table_size = u16::from_le_bytes(data[pos..pos + 2].try_into().unwrap()) as usize;
    pos += 2;

    // Read table entries and build decode lookup
    // Map (code, len) → symbol
    let mut decode_table: Vec<(u32, u8, i16)> = Vec::with_capacity(table_size);
    for _ in 0..table_size {
        let sym = i16::from_le_bytes(data[pos..pos + 2].try_into().unwrap());
        pos += 2;
        let len = data[pos];
        pos += 1;
        let code = u32::from_le_bytes(data[pos..pos + 4].try_into().unwrap());
        pos += 4;
        decode_table.push((code, len, sym));
    }

    // Build prefix decode: walk bits, match against table
    // For efficiency, group by code length
    let max_len = decode_table.iter().map(|&(_, l, _)| l).max().unwrap_or(0);
    let mut by_len: Vec<Vec<(u32, i16)>> = vec![Vec::new(); max_len as usize + 1];
    for &(code, len, sym) in &decode_table {
        by_len[len as usize].push((code, sym));
    }

    // Decode bitstream
    let bitstream = &data[pos..];
    let mut reader = BitReader::new(bitstream);

    let mut samples = Vec::with_capacity(sample_count);
    samples.push(first_sample);

    let mut current = first_sample;

    for _ in 0..sample_count - 1 {
        // Read bits until we match a code
        let mut code: u32 = 0;
        let mut len: u8 = 0;
        let mut found = false;

        for target_len in 1..=max_len {
            let bit = reader.read_bit().expect("Unexpected end of bitstream");
            code = (code << 1) | bit as u32;
            len = target_len;

            // Check if this matches any code at this length
            for &(c, sym) in &by_len[len as usize] {
                if c == code {
                    // Found it — apply delta
                    let next = (current as i16 + sym) as u8;
                    samples.push(next);
                    current = next;
                    found = true;
                    break;
                }
            }
            if found {
                break;
            }
        }

        if !found {
            panic!("Failed to decode Huffman symbol at sample {}", samples.len());
        }
    }

    DecodedStream {
        samples,
        sample_rate,
        channels,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn round_trip_simple() {
        let samples: Vec<u8> = vec![100, 101, 100, 99, 100, 107, 107, 107, 100];
        let compressed = encode(&samples, 44100, 1);
        let decoded = decode(&compressed.data);
        assert_eq!(decoded.samples, samples);
        assert_eq!(decoded.sample_rate, 44100);
        assert_eq!(decoded.channels, 1);
    }

    #[test]
    fn round_trip_constant() {
        let samples: Vec<u8> = vec![42; 1000];
        let compressed = encode(&samples, 8000, 2);
        assert!(compressed.stats.compressed_bytes < compressed.stats.raw_bytes);
        let decoded = decode(&compressed.data);
        assert_eq!(decoded.samples, samples);
    }

    #[test]
    fn round_trip_all_values() {
        let mut samples: Vec<u8> = (0..=255).collect();
        samples.extend(samples.clone().into_iter().rev());
        let compressed = encode(&samples, 44100, 1);
        let decoded = decode(&compressed.data);
        assert_eq!(decoded.samples, samples);
    }
}
