//! `XSalsa20` keystream with a scalar Salsa20 core and a portable SIMD path.
//!
//! The SIMD path computes N blocks at once in column layout: vector `i`
//! holds state word `i` of N consecutive blocks. N is the `u32` lane count
//! of the detected level (4 on SSE4.2 and NEON, 8 on AVX2, 16 on AVX-512).

#![expect(
    clippy::inline_always,
    reason = "SIMD kernels must inline into the dispatched target-feature function"
)]

use fearless_simd::prelude::*;
use fearless_simd::{Level, Simd, dispatch, u32x4};
use zeroize::Zeroize;

const STATE_WORDS: usize = 16;
const BLOCK_LEN: usize = 64;
const FIRST_BLOCK_DATA_OFFSET: usize = 32;
const CONSTANTS: [u32; 4] = [0x6170_7865, 0x3320_646e, 0x7962_2d32, 0x6b20_6574];
/// Largest SIMD batch: 16 blocks of AVX-512.
const MAX_BATCH_LEN: usize = 16 * BLOCK_LEN;
/// Blocks of the 4-lane path that covers what full batches leave.
const TAIL_LANES: usize = 4;

#[cfg(feature = "std")]
static LEVEL: std::sync::LazyLock<Level> = std::sync::LazyLock::new(Level::new);

fn level() -> Level {
    #[cfg(feature = "std")]
    {
        *LEVEL
    }
    #[cfg(not(feature = "std"))]
    {
        Level::baseline()
    }
}

/// `XSalsa20` keystream of one key and nonce. Block 0 supplies the
/// Poly1305 key and the keystream of the first 32 data bytes; later data
/// uses blocks 1 and up.
pub(crate) struct Keystream {
    state: [u32; STATE_WORDS],
    first_block: [u8; BLOCK_LEN],
}

impl Keystream {
    pub(crate) fn new(key: &[u8; 32], nonce: &[u8; 24]) -> Self {
        let mut subkey = hsalsa20(key, nonce[..16].try_into().unwrap());
        let mut state = salsa20_state(&subkey, nonce[16..].try_into().unwrap(), 0);
        subkey.zeroize();
        let first_block = salsa20_block(&state);
        set_counter(&mut state, 1);
        Self { state, first_block }
    }

    pub(crate) fn poly1305_key(&self) -> &[u8; 32] {
        self.first_block[..FIRST_BLOCK_DATA_OFFSET]
            .try_into()
            .unwrap()
    }

    /// XOR the keystream into `data`, starting at data byte 0.
    pub(crate) fn xor(self, data: &mut [u8]) {
        self.xor_at_level(data, level());
    }

    fn xor_at_level(self, data: &mut [u8], level: Level) {
        let first_len = data.len().min(BLOCK_LEN - FIRST_BLOCK_DATA_OFFSET);
        let (head, rest) = data.split_at_mut(first_len);
        xor_bytes(head, &self.first_block[FIRST_BLOCK_DATA_OFFSET..]);
        if rest.len() > BLOCK_LEN {
            let state = &self.state;
            dispatch!(level, simd => xor_stream(simd, state, rest));
        } else if !rest.is_empty() {
            xor_block(&self.state, rest);
        }
    }
}

impl Drop for Keystream {
    fn drop(&mut self) {
        self.state.zeroize();
        self.first_block.zeroize();
    }
}

pub(crate) fn hsalsa20(key: &[u8; 32], input: &[u8; 16]) -> [u8; 32] {
    let mut state = [0u32; STATE_WORDS];
    state[0] = CONSTANTS[0];
    read_words(&key[..16], &mut state[1..5]);
    state[5] = CONSTANTS[1];
    read_words(input, &mut state[6..10]);
    state[10] = CONSTANTS[2];
    read_words(&key[16..], &mut state[11..15]);
    state[15] = CONSTANTS[3];

    for _ in 0..10 {
        double_round(&mut state);
    }

    let key_idx = [0, 5, 10, 15, 6, 7, 8, 9];
    let mut out = [0u8; 32];
    for (chunk, idx) in out.as_chunks_mut::<4>().0.iter_mut().zip(key_idx) {
        chunk.copy_from_slice(&state[idx].to_le_bytes());
    }
    state.zeroize();
    out
}

fn salsa20_state(key: &[u8; 32], nonce: [u8; 8], counter: u64) -> [u32; STATE_WORDS] {
    let mut state = [0u32; STATE_WORDS];
    state[0] = CONSTANTS[0];
    read_words(&key[..16], &mut state[1..5]);
    state[5] = CONSTANTS[1];
    read_words(&nonce, &mut state[6..8]);
    set_counter(&mut state, counter);
    state[10] = CONSTANTS[2];
    read_words(&key[16..], &mut state[11..15]);
    state[15] = CONSTANTS[3];
    state
}

fn salsa20_block(state: &[u32; STATE_WORDS]) -> [u8; BLOCK_LEN] {
    let mut words = *state;
    for _ in 0..10 {
        double_round(&mut words);
    }
    let mut out = [0u8; BLOCK_LEN];
    for ((chunk, word), initial) in out.as_chunks_mut::<4>().0.iter_mut().zip(words).zip(state) {
        chunk.copy_from_slice(&word.wrapping_add(*initial).to_le_bytes());
    }
    words.zeroize();
    out
}

/// XOR the keystream of `state` and the following blocks into `data`.
/// Full batches of N blocks run at the level's width. A 4-lane pass
/// covers the rest when it spans more than one block, and a scalar block
/// covers a last block on its own.
#[inline(always)]
fn xor_stream<S: Simd>(simd: S, state: &[u32; STATE_WORDS], data: &mut [u8])
where
    S::u32s: SimdOps,
{
    let lanes = S::u32s::LEN;
    let mut counter = get_counter(state);
    let mut scratch = BatchScratch([0; MAX_BATCH_LEN / 4]);

    let mut batches = data.chunks_exact_mut(lanes * BLOCK_LEN);
    for chunk in &mut batches {
        let words = batch_rounds::<S, S::u32s>(simd, state, counter);
        xor_keystream_column(&words, lanes, &mut scratch, chunk);
        counter = counter.wrapping_add(lanes as u64);
    }

    for chunk in batches.into_remainder().chunks_mut(TAIL_LANES * BLOCK_LEN) {
        if chunk.len() <= BLOCK_LEN {
            let mut block_state = *state;
            set_counter(&mut block_state, counter);
            xor_block(&block_state, chunk);
            block_state.zeroize();
            break;
        }
        let words = batch_rounds::<S, u32x4<S>>(simd, state, counter);
        xor_keystream_column(&words, TAIL_LANES, &mut scratch, chunk);
        counter = counter.wrapping_add(TAIL_LANES as u64);
    }
}

/// Run 20 rounds plus feed-forward on consecutive blocks starting at
/// `counter`, one block per lane.
#[inline(always)]
fn batch_rounds<S, V>(simd: S, state: &[u32; STATE_WORDS], counter: u64) -> [V; STATE_WORDS]
where
    S: Simd,
    V: SimdBase<S, Element = u32> + SimdOps,
{
    let mut words: [V; STATE_WORDS] = core::array::from_fn(|i| V::simd_from(simd, state[i]));
    words[8] = V::from_fn(simd, |lane| counter.wrapping_add(lane as u64) as u32);
    words[9] = V::from_fn(simd, |lane| {
        (counter.wrapping_add(lane as u64) >> 32) as u32
    });
    let initial = words;

    for _ in 0..10 {
        double_round_simd(&mut words);
    }
    for (word, initial) in words.iter_mut().zip(initial) {
        *word += initial;
    }
    words
}

/// XOR one scalar keystream block of `state` into at most 64 bytes.
fn xor_block(state: &[u32; STATE_WORDS], data: &mut [u8]) {
    let mut block = salsa20_block(state);
    xor_bytes(data, &block);
    block.zeroize();
}

/// Keystream words of one SIMD batch in block order. Word storage keeps
/// the wipe on drop at one volatile write per 4 bytes.
struct BatchScratch([u32; MAX_BATCH_LEN / 4]);

impl Drop for BatchScratch {
    fn drop(&mut self) {
        self.0.zeroize();
    }
}

/// Store the column-layout keystream of `lanes` blocks in block order,
/// then XOR it into `chunk` 4 bytes at a time. The separate XOR pass
/// vectorizes.
#[inline(always)]
fn xor_keystream_column<V>(
    words: &[V; STATE_WORDS],
    lanes: usize,
    scratch: &mut BatchScratch,
    chunk: &mut [u8],
) where
    V: SimdOps + core::ops::Index<usize, Output = u32>,
{
    for (index, word) in words.iter().enumerate() {
        for lane in 0..lanes {
            scratch.0[lane * STATE_WORDS + index] = word[lane];
        }
    }

    let (whole, tail) = chunk.as_chunks_mut::<4>();
    for (bytes, key) in whole.iter_mut().zip(&scratch.0) {
        *bytes = (u32::from_le_bytes(*bytes) ^ key).to_le_bytes();
    }
    if !tail.is_empty() {
        xor_bytes(tail, &scratch.0[whole.len()].to_le_bytes());
    }
}

pub(crate) trait SimdOps:
    Copy
    + core::ops::Add<Output = Self>
    + core::ops::BitXor<Output = Self>
    + core::ops::BitOr<Output = Self>
    + core::ops::Shl<u32, Output = Self>
    + core::ops::Shr<u32, Output = Self>
{
}

impl<T> SimdOps for T where
    T: Copy
        + core::ops::Add<Output = T>
        + core::ops::BitXor<Output = T>
        + core::ops::BitOr<Output = T>
        + core::ops::Shl<u32, Output = T>
        + core::ops::Shr<u32, Output = T>
{
}

#[inline(always)]
fn rotl<V: SimdOps>(x: V, n: u32) -> V {
    (x << n) | (x >> (32 - n))
}

#[inline(always)]
#[expect(clippy::many_single_char_names)]
fn quarter_round_simd<V: SimdOps>(
    s: &mut [V; STATE_WORDS],
    a: usize,
    b: usize,
    c: usize,
    d: usize,
) {
    s[b] = s[b] ^ rotl(s[a] + s[d], 7);
    s[c] = s[c] ^ rotl(s[b] + s[a], 9);
    s[d] = s[d] ^ rotl(s[c] + s[b], 13);
    s[a] = s[a] ^ rotl(s[d] + s[c], 18);
}

#[inline(always)]
fn double_round_simd<V: SimdOps>(s: &mut [V; STATE_WORDS]) {
    quarter_round_simd(s, 0, 4, 8, 12);
    quarter_round_simd(s, 5, 9, 13, 1);
    quarter_round_simd(s, 10, 14, 2, 6);
    quarter_round_simd(s, 15, 3, 7, 11);
    quarter_round_simd(s, 0, 1, 2, 3);
    quarter_round_simd(s, 5, 6, 7, 4);
    quarter_round_simd(s, 10, 11, 8, 9);
    quarter_round_simd(s, 15, 12, 13, 14);
}

#[inline(always)]
fn double_round(state: &mut [u32; STATE_WORDS]) {
    // Column round
    quarter_round(0, 4, 8, 12, state);
    quarter_round(5, 9, 13, 1, state);
    quarter_round(10, 14, 2, 6, state);
    quarter_round(15, 3, 7, 11, state);
    // Row round
    quarter_round(0, 1, 2, 3, state);
    quarter_round(5, 6, 7, 4, state);
    quarter_round(10, 11, 8, 9, state);
    quarter_round(15, 12, 13, 14, state);
}

#[inline(always)]
#[allow(clippy::many_single_char_names)]
fn quarter_round(a: usize, b: usize, c: usize, d: usize, state: &mut [u32; STATE_WORDS]) {
    state[b] ^= state[a].wrapping_add(state[d]).rotate_left(7);
    state[c] ^= state[b].wrapping_add(state[a]).rotate_left(9);
    state[d] ^= state[c].wrapping_add(state[b]).rotate_left(13);
    state[a] ^= state[d].wrapping_add(state[c]).rotate_left(18);
}

fn read_words(input: &[u8], out: &mut [u32]) {
    for (word, chunk) in out.iter_mut().zip(input.as_chunks::<4>().0) {
        *word = u32::from_le_bytes(*chunk);
    }
}

fn xor_bytes(out: &mut [u8], key_stream: &[u8]) {
    for (byte, key_byte) in out.iter_mut().zip(key_stream.iter()) {
        *byte ^= *key_byte;
    }
}

#[inline]
fn get_counter(state: &[u32; STATE_WORDS]) -> u64 {
    (u64::from(state[9]) << 32) | u64::from(state[8])
}

#[inline]
fn set_counter(state: &mut [u32; STATE_WORDS], counter: u64) {
    state[8] = counter as u32;
    state[9] = (counter >> 32) as u32;
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::testing::{FIRSTKEY, NONCE, Rng, hex};
    use sha2::{Digest, Sha256};
    use std::vec;
    use std::vec::Vec;

    /// libsodium `test/default/stream.exp`: `XSalsa20` keystreams under the
    /// `NaCl` `firstkey` and nonce.
    const SODIUM_STREAM_EXP: &str = include_str!("../testdata/libsodium-stream.exp");

    /// XOR with one scalar block per 64 bytes, the reference for SIMD paths.
    fn xor_scalar(key: &[u8; 32], nonce: &[u8; 24], data: &mut [u8]) -> [u8; 32] {
        let keystream = Keystream::new(key, nonce);
        let poly_key = *keystream.poly1305_key();
        let first_len = data.len().min(BLOCK_LEN - FIRST_BLOCK_DATA_OFFSET);
        let (head, rest) = data.split_at_mut(first_len);
        xor_bytes(head, &keystream.first_block[FIRST_BLOCK_DATA_OFFSET..]);
        let mut state = keystream.state;
        for chunk in rest.chunks_mut(BLOCK_LEN) {
            xor_bytes(chunk, &salsa20_block(&state));
            let next = get_counter(&state) + 1;
            set_counter(&mut state, next);
        }
        poly_key
    }

    /// Every level this CPU supports, from the fallback up. Levels named in
    /// `SALSABOX_REQUIRE_LEVEL` (comma separated) must be among them, so a
    /// CI job cannot silently skip the level it exists for.
    fn levels() -> Vec<(&'static str, Level)> {
        let detected = Level::new();
        let mut out = vec![
            ("fallback", Level::fallback()),
            ("baseline", Level::baseline()),
        ];
        #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
        {
            if let Some(level) = detected.as_sse4_2() {
                out.push(("sse4.2", Level::Sse4_2(level)));
            }
            if let Some(level) = detected.as_avx2() {
                out.push(("avx2", Level::Avx2(level)));
            }
            if let Some(level) = detected.as_avx512() {
                out.push(("avx512", Level::Avx512(level)));
            }
        }
        #[cfg(target_arch = "aarch64")]
        if let Some(level) = detected.as_neon() {
            out.push(("neon", Level::Neon(level)));
        }
        if let Ok(required) = std::env::var("SALSABOX_REQUIRE_LEVEL") {
            for name in required
                .split(',')
                .map(str::trim)
                .filter(|name| !name.is_empty())
            {
                assert!(
                    out.iter().any(|(level, _)| *level == name),
                    "required level {name} is not available"
                );
            }
        }
        out
    }

    /// Raw `XSalsa20` keystream under the `NaCl` key: block 0, then blocks 1
    /// and up.
    fn raw_keystream(level: Level, len: usize) -> Vec<u8> {
        let keystream = Keystream::new(&FIRSTKEY, &NONCE);
        let mut out = vec![0u8; len];
        let head = len.min(FIRST_BLOCK_DATA_OFFSET);
        out[..head].copy_from_slice(&keystream.poly1305_key()[..head]);
        keystream.xor_at_level(&mut out[head..], level);
        out
    }

    /// Keystream under the `NaCl` key from block `counter` on, through the
    /// batch path.
    fn keystream_from(level: Level, counter: u64, len: usize) -> Vec<u8> {
        let mut state = Keystream::new(&FIRSTKEY, &NONCE).state;
        set_counter(&mut state, counter);
        let mut out = vec![0u8; len];
        dispatch!(level, simd => xor_stream(simd, &state, &mut out));
        out
    }

    #[test]
    fn simd_levels_match_scalar() {
        let key = [0xC3u8; 32];
        let nonce = [0x9Du8; 24];
        let mut sizes: Vec<usize> = (0..=2 * MAX_BATCH_LEN + 2 * BLOCK_LEN).collect();
        sizes.extend([4096, 16_384, 70_001]);
        for (name, level) in levels() {
            for &len in &sizes {
                let plaintext: Vec<u8> = (0..len).map(|i| (i as u8).wrapping_mul(31)).collect();
                let mut expected = plaintext.clone();
                let expected_poly = xor_scalar(&key, &nonce, &mut expected);

                let keystream = Keystream::new(&key, &nonce);
                assert_eq!(*keystream.poly1305_key(), expected_poly, "{name} poly key");
                let mut actual = plaintext;
                keystream.xor_at_level(&mut actual, level);
                assert_eq!(actual, expected, "{name} len {len}");
            }
        }
    }

    #[test]
    fn counter_carries_into_high_word() {
        let key = [0x5Au8; 32];
        let nonce = [0xA5u8; 24];
        for (name, level) in levels() {
            let mut keystream = Keystream::new(&key, &nonce);
            set_counter(&mut keystream.state, u64::from(u32::MAX) - 5);
            let start = keystream.state;
            let mut actual = vec![0u8; 32 + 40 * BLOCK_LEN];
            keystream.xor_at_level(&mut actual, level);

            let mut state = start;
            for (index, chunk) in actual[32..].chunks(BLOCK_LEN).enumerate() {
                assert_eq!(chunk, salsa20_block(&state), "{name} block {index}");
                let next = get_counter(&state) + 1;
                set_counter(&mut state, next);
            }
        }
    }
    #[test]
    fn random_inputs_match_scalar() {
        let levels = levels();
        let mut rng = Rng::new(0x5a15_ab0c);
        for _ in 0..200 {
            let mut key = [0u8; 32];
            let mut nonce = [0u8; 24];
            rng.fill(&mut key);
            rng.fill(&mut nonce);
            let mut plaintext = vec![0u8; rng.below(3 * MAX_BATCH_LEN)];
            rng.fill(&mut plaintext);
            let mut expected = plaintext.clone();
            xor_scalar(&key, &nonce, &mut expected);

            for (name, level) in &levels {
                let mut actual = plaintext.clone();
                Keystream::new(&key, &nonce).xor_at_level(&mut actual, *level);
                assert_eq!(actual, expected, "{name} len {}", plaintext.len());
            }
        }
    }

    #[test]
    fn libsodium_stream_vectors() {
        let lines: Vec<&str> = SODIUM_STREAM_EXP.lines().collect();
        assert_eq!(lines.len(), 83);
        for (name, level) in levels() {
            let mut stream = raw_keystream(level, 4_194_304);
            assert_eq!(hex(&Sha256::digest(&stream)), lines[0], "{name} 4 MiB");

            stream[..4000].copy_from_slice(&keystream_from(level, 1, 4000));
            assert_eq!(
                hex(&Sha256::digest(&stream)),
                lines[1],
                "{name} from block 1"
            );

            for (len, line) in lines[2..66].iter().enumerate() {
                let mut out = vec![len as u8; BLOCK_LEN];
                out[..len].copy_from_slice(&raw_keystream(level, len));
                assert_eq!(hex(&out), *line, "{name} prefix {len}");
            }

            let carry = keystream_from(level, (1 << 32) - 1, 192);
            assert_eq!(hex(&carry), lines[66], "{name} carry");
            for (back, line) in (1..=16u64).rev().zip(&lines[67..]) {
                let out = keystream_from(level, (1 << 32) - back, 17 * BLOCK_LEN);
                assert_eq!(hex(&out), *line, "{name} carry from 2^32 - {back}");
            }
        }
    }
}
