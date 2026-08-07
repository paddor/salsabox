use zeroize::Zeroize;

const STATE_WORDS: usize = 16;
const BLOCK_LEN: usize = 64;
const FIRST_BLOCK_DATA_OFFSET: usize = 32;
const CONSTANTS: [u32; 4] = [0x6170_7865, 0x3320_646e, 0x7962_2d32, 0x6b20_6574];

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
mod avx2;

#[derive(Clone, Copy)]
enum StreamBackend {
    Runtime,
    #[cfg(test)]
    Scalar,
    #[cfg(all(test, any(target_arch = "x86", target_arch = "x86_64")))]
    Avx2,
}

pub(crate) fn poly1305_key(key: &[u8; 32], nonce: &[u8; 24], poly_key: &mut [u8; 32]) {
    let mut subkey = hsalsa20(key, nonce[..16].try_into().unwrap());
    let mut state = salsa20_state(&subkey, nonce[16..].try_into().unwrap(), 0);
    let mut block = salsa20_block(&state);
    poly_key.copy_from_slice(&block[..32]);
    block.zeroize();
    state.zeroize();
    subkey.zeroize();
}

pub(crate) fn xor_xsalsa20(
    key: &[u8; 32],
    nonce: &[u8; 24],
    data: &mut [u8],
    poly_key: &mut [u8; 32],
) {
    xor_xsalsa20_with_backend(key, nonce, data, poly_key, StreamBackend::Runtime);
}

#[cfg(test)]
pub(crate) fn xor_xsalsa20_scalar_for_test(
    key: &[u8; 32],
    nonce: &[u8; 24],
    data: &mut [u8],
    poly_key: &mut [u8; 32],
) {
    xor_xsalsa20_with_backend(key, nonce, data, poly_key, StreamBackend::Scalar);
}

#[cfg(all(test, any(target_arch = "x86", target_arch = "x86_64")))]
pub(crate) unsafe fn xor_xsalsa20_avx2_for_test(
    key: &[u8; 32],
    nonce: &[u8; 24],
    data: &mut [u8],
    poly_key: &mut [u8; 32],
) {
    xor_xsalsa20_with_backend(key, nonce, data, poly_key, StreamBackend::Avx2);
}

fn xor_xsalsa20_with_backend(
    key: &[u8; 32],
    nonce: &[u8; 24],
    data: &mut [u8],
    poly_key: &mut [u8; 32],
    backend: StreamBackend,
) {
    let mut subkey = hsalsa20(key, nonce[..16].try_into().unwrap());
    let mut state = salsa20_state(&subkey, nonce[16..].try_into().unwrap(), 0);
    let mut first_block = salsa20_block(&state);

    poly_key.copy_from_slice(&first_block[..FIRST_BLOCK_DATA_OFFSET]);
    let first_len = data.len().min(BLOCK_LEN - FIRST_BLOCK_DATA_OFFSET);
    xor_bytes(
        &mut data[..first_len],
        &first_block[FIRST_BLOCK_DATA_OFFSET..FIRST_BLOCK_DATA_OFFSET + first_len],
    );

    if data.len() > first_len {
        set_counter(&mut state, 1);
        xor_stream(&mut state, &mut data[first_len..], backend);
    }

    first_block.zeroize();
    state.zeroize();
    subkey.zeroize();
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
        column_rounds(&mut state);
        diagonal_rounds(&mut state);
    }

    let key_idx = [0, 5, 10, 15, 6, 7, 8, 9];
    let mut out = [0u8; 32];
    for (chunk, idx) in out.chunks_exact_mut(4).zip(key_idx) {
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
    state[8] = counter as u32;
    state[9] = (counter >> 32) as u32;
    state[10] = CONSTANTS[2];
    read_words(&key[16..], &mut state[11..15]);
    state[15] = CONSTANTS[3];
    state
}

fn salsa20_block(state: &[u32; STATE_WORDS]) -> [u8; BLOCK_LEN] {
    let words = run_rounds(state);
    let mut out = [0u8; BLOCK_LEN];
    for (chunk, word) in out.chunks_exact_mut(4).zip(words) {
        chunk.copy_from_slice(&word.to_le_bytes());
    }
    out
}

fn xor_stream(state: &mut [u32; STATE_WORDS], data: &mut [u8], backend: StreamBackend) {
    match backend {
        StreamBackend::Runtime => {
            #[cfg(all(feature = "std", any(target_arch = "x86", target_arch = "x86_64")))]
            {
                if std::arch::is_x86_feature_detected!("avx2") {
                    unsafe {
                        avx2::xor_stream(state, data);
                    }
                    return;
                }
            }
            xor_stream_scalar(state, data);
        }
        #[cfg(test)]
        StreamBackend::Scalar => xor_stream_scalar(state, data),
        #[cfg(all(test, any(target_arch = "x86", target_arch = "x86_64")))]
        StreamBackend::Avx2 => unsafe {
            avx2::xor_stream(state, data);
        },
    }
}

fn xor_stream_scalar(state: &mut [u32; STATE_WORDS], data: &mut [u8]) {
    for chunk in data.chunks_mut(BLOCK_LEN) {
        let mut block = salsa20_block(state);
        xor_bytes(chunk, &block[..chunk.len()]);
        increment_counter(state, 1);
        block.zeroize();
    }
}

fn run_rounds(state: &[u32; STATE_WORDS]) -> [u32; STATE_WORDS] {
    let mut res = *state;

    for _ in 0..10 {
        column_rounds(&mut res);
        diagonal_rounds(&mut res);
    }

    for (word, initial) in res.iter_mut().zip(state.iter()) {
        *word = word.wrapping_add(*initial);
    }
    res
}

fn column_rounds(state: &mut [u32; STATE_WORDS]) {
    quarter_round(0, 4, 8, 12, state);
    quarter_round(5, 9, 13, 1, state);
    quarter_round(10, 14, 2, 6, state);
    quarter_round(15, 3, 7, 11, state);
}

fn diagonal_rounds(state: &mut [u32; STATE_WORDS]) {
    quarter_round(0, 1, 2, 3, state);
    quarter_round(5, 6, 7, 4, state);
    quarter_round(10, 11, 8, 9, state);
    quarter_round(15, 12, 13, 14, state);
}

#[allow(clippy::many_single_char_names)]
fn quarter_round(a: usize, b: usize, c: usize, d: usize, state: &mut [u32; STATE_WORDS]) {
    state[b] ^= state[a].wrapping_add(state[d]).rotate_left(7);
    state[c] ^= state[b].wrapping_add(state[a]).rotate_left(9);
    state[d] ^= state[c].wrapping_add(state[b]).rotate_left(13);
    state[a] ^= state[d].wrapping_add(state[c]).rotate_left(18);
}

fn read_words(input: &[u8], out: &mut [u32]) {
    for (word, chunk) in out.iter_mut().zip(input.chunks_exact(4)) {
        *word = u32::from_le_bytes(chunk.try_into().unwrap());
    }
}

fn xor_bytes(out: &mut [u8], key_stream: &[u8]) {
    for (byte, key_byte) in out.iter_mut().zip(key_stream.iter()) {
        *byte ^= *key_byte;
    }
}

#[inline]
fn increment_counter(state: &mut [u32; STATE_WORDS], blocks: u64) {
    set_counter(state, get_counter(state).wrapping_add(blocks));
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
