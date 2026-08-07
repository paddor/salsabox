#![allow(
    unsafe_op_in_unsafe_fn,
    clippy::cast_possible_wrap,
    clippy::inline_always,
    clippy::wildcard_imports
)]

#[cfg(target_arch = "x86")]
use core::arch::x86::*;
#[cfg(target_arch = "x86_64")]
use core::arch::x86_64::*;

use zeroize::Zeroize;

use super::{STATE_WORDS, increment_counter, xor_bytes};

const BLOCK_LEN: usize = 64;
const PAR_BLOCKS: usize = 4;
const PAR_LEN: usize = PAR_BLOCKS * BLOCK_LEN;
const N: usize = PAR_BLOCKS / 2;

const IDX: [[usize; 4]; 4] = [[0, 5, 10, 15], [4, 9, 14, 3], [8, 13, 2, 7], [12, 1, 6, 11]];

macro_rules! rotl_epi32 {
    ($v:expr, $n:literal) => {{
        let v = $v;
        _mm256_or_si256(_mm256_slli_epi32(v, $n), _mm256_srli_epi32(v, 32 - $n))
    }};
}

#[target_feature(enable = "avx2")]
pub(crate) unsafe fn xor_stream(state: &mut [u32; STATE_WORDS], data: &mut [u8]) {
    let par_len = data.len() / PAR_LEN * PAR_LEN;
    let (head, tail) = data.split_at_mut(par_len);

    for chunk in head.chunks_exact_mut(PAR_LEN) {
        let res = rounds(state);
        let mut blocks = [[0u8; BLOCK_LEN]; PAR_BLOCKS];
        for i in 0..N {
            let lo = extract_lo(&res[i]);
            let hi = extract_hi(&res[i]);
            store_block(&lo, &mut blocks[2 * i]);
            store_block(&hi, &mut blocks[2 * i + 1]);
        }
        for (block_index, block) in blocks.iter().enumerate() {
            let offset = block_index * BLOCK_LEN;
            xor_bytes(&mut chunk[offset..offset + BLOCK_LEN], block);
        }
        blocks.zeroize();
        increment_counter(state, PAR_BLOCKS as u64);
    }

    super::xor_stream_scalar(state, tail);
}

#[target_feature(enable = "avx2")]
unsafe fn rounds(state: &[u32; STATE_WORDS]) -> [[__m256i; 4]; N] {
    let counter = get_counter(state);
    let mut res = [[_mm256_setzero_si256(); 4]; N];
    for (i, block) in res.iter_mut().enumerate() {
        let base = counter + (i * 2) as u64;
        *block = load_diag_pair(state, base, base + 1);
    }
    let initial = res;

    for _ in 0..20 {
        for block in &mut res {
            quarter_round(block);
            shuffle(block);
        }
    }

    for i in 0..N {
        for j in 0..4 {
            res[i][j] = _mm256_add_epi32(res[i][j], initial[i][j]);
        }
    }

    res
}

#[inline(always)]
unsafe fn load_diag_pair(state: &[u32; STATE_WORDS], ctr0: u64, ctr1: u64) -> [__m256i; 4] {
    let s = state;
    [
        _mm256_set_epi32(
            s[15] as i32,
            s[10] as i32,
            s[5] as i32,
            s[0] as i32,
            s[15] as i32,
            s[10] as i32,
            s[5] as i32,
            s[0] as i32,
        ),
        _mm256_set_epi32(
            s[3] as i32,
            s[14] as i32,
            (ctr1 >> 32) as i32,
            s[4] as i32,
            s[3] as i32,
            s[14] as i32,
            (ctr0 >> 32) as i32,
            s[4] as i32,
        ),
        _mm256_set_epi32(
            s[7] as i32,
            s[2] as i32,
            s[13] as i32,
            ctr1 as i32,
            s[7] as i32,
            s[2] as i32,
            s[13] as i32,
            ctr0 as i32,
        ),
        _mm256_set_epi32(
            s[11] as i32,
            s[6] as i32,
            s[1] as i32,
            s[12] as i32,
            s[11] as i32,
            s[6] as i32,
            s[1] as i32,
            s[12] as i32,
        ),
    ]
}

#[inline(always)]
unsafe fn quarter_round(block: &mut [__m256i; 4]) {
    let [a, b, c, d] = block;
    *b = _mm256_xor_si256(*b, rotl_epi32!(_mm256_add_epi32(*a, *d), 7));
    *c = _mm256_xor_si256(*c, rotl_epi32!(_mm256_add_epi32(*b, *a), 9));
    *d = _mm256_xor_si256(*d, rotl_epi32!(_mm256_add_epi32(*c, *b), 13));
    *a = _mm256_xor_si256(*a, rotl_epi32!(_mm256_add_epi32(*d, *c), 18));
}

#[inline(always)]
unsafe fn shuffle(block: &mut [__m256i; 4]) {
    let [_, b, c, d] = block;
    let tmp = *b;
    *b = _mm256_shuffle_epi32(*d, 0b_00_11_10_01);
    *c = _mm256_shuffle_epi32(*c, 0b_01_00_11_10);
    *d = _mm256_shuffle_epi32(tmp, 0b_10_01_00_11);
}

#[inline(always)]
unsafe fn extract_lo(set: &[__m256i; 4]) -> [__m128i; 4] {
    [
        _mm256_castsi256_si128(set[0]),
        _mm256_castsi256_si128(set[1]),
        _mm256_castsi256_si128(set[2]),
        _mm256_castsi256_si128(set[3]),
    ]
}

#[inline(always)]
unsafe fn extract_hi(set: &[__m256i; 4]) -> [__m128i; 4] {
    [
        _mm256_extracti128_si256(set[0], 1),
        _mm256_extracti128_si256(set[1], 1),
        _mm256_extracti128_si256(set[2], 1),
        _mm256_extracti128_si256(set[3], 1),
    ]
}

#[inline(always)]
unsafe fn store_block(diag: &[__m128i; 4], block: &mut [u8]) {
    let mut vals = [[0u32; 4]; 4];
    for (i, v) in diag.iter().enumerate() {
        _mm_storeu_si128(vals[i].as_mut_ptr().cast(), *v);
    }
    for (reg, idxs) in vals.iter().zip(IDX.iter()) {
        for (&val, &idx) in reg.iter().zip(idxs.iter()) {
            block[idx * 4..idx * 4 + 4].copy_from_slice(&val.to_le_bytes());
        }
    }
    vals.zeroize();
}

#[inline(always)]
fn get_counter(state: &[u32; STATE_WORDS]) -> u64 {
    (u64::from(state[9]) << 32) | u64::from(state[8])
}
