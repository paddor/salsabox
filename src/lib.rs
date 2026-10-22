//! NaCl-compatible XSalsa20-Poly1305 boxes with a SIMD Salsa20 core.
//!
//! [`SecretBox`] is `crypto_secretbox` under a 32-byte key. [`BoxKey`] is
//! the symmetric half of `crypto_box`, built from a precomputed X25519
//! shared secret. Both encrypt in place and return a detached 16-byte
//! tag. `NaCl`'s combined format is `tag || ciphertext`.
//!
//! ```
//! use salsabox::SecretBox;
//!
//! let key = SecretBox::new([7; 32]);
//! let nonce = [1; 24];
//! let mut buf = *b"attack at dawn";
//! let tag = key.encrypt_detached(&nonce, &mut buf);
//! key.decrypt_detached(&nonce, &mut buf, &tag)?;
//! assert_eq!(&buf, b"attack at dawn");
//! # Ok::<(), salsabox::Error>(())
//! ```
//!
//! # Nonces
//!
//! Never seal two messages under the same key and nonce. That reveals the
//! XOR of their plaintexts and lets an attacker forge tags. Random 24-byte
//! nonces and counters are both fine.
//!
//! # Implementation
//!
//! Salsa20 and `HSalsa20` live in this crate. They use only 32-bit add,
//! rotate, and XOR, with no secret-dependent branches or memory access.
//! With `std`, `fearless_simd` picks SSE4.2, AVX2, AVX-512, or NEON at
//! runtime; without it, the level follows compile-time target features.
//! Poly1305 is `RustCrypto` `poly1305`, and tags are compared with `subtle`.
//! Keys, subkeys, and keystream buffers are wiped on drop. Poly1305's
//! internal state is not, nor are copies the compiler keeps in registers or
//! on the stack.

#![no_std]

#[cfg(any(feature = "std", test))]
extern crate std;

use core::fmt;

use poly1305::Poly1305;
use poly1305::universal_hash::KeyInit;
use subtle::ConstantTimeEq;
use zeroize::{Zeroize, Zeroizing};

mod salsa20;
#[cfg(test)]
mod testing;

/// Authentication failure: the tag does not match the box.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct Error;

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str("authentication failed")
    }
}

#[cfg(feature = "std")]
impl std::error::Error for Error {}

/// XSalsa20-Poly1305 secret box keyed by a 32-byte symmetric key.
pub struct SecretBox {
    key: Zeroizing<[u8; 32]>,
}

impl SecretBox {
    /// Create a secret box from an owned key.
    pub fn new(key: [u8; 32]) -> Self {
        Self {
            key: Zeroizing::new(key),
        }
    }

    /// Create a secret box by copying a key.
    pub fn from_key(key: &[u8; 32]) -> Self {
        Self::new(*key)
    }

    /// Encrypt `buf` in place and return the detached Poly1305 tag.
    #[must_use = "the tag is needed to open the box"]
    pub fn encrypt_detached(&self, nonce: &[u8; 24], buf: &mut [u8]) -> [u8; 16] {
        let keystream = salsa20::Keystream::new(self.key(), nonce);
        let poly_key = Zeroizing::new(*keystream.poly1305_key());
        keystream.xor(buf);
        poly1305_tag(&poly_key, buf)
    }

    /// Verify `tag`, then decrypt `buf` in place. On failure `buf` is left
    /// untouched.
    pub fn decrypt_detached(
        &self,
        nonce: &[u8; 24],
        buf: &mut [u8],
        tag: &[u8; 16],
    ) -> Result<(), Error> {
        let keystream = salsa20::Keystream::new(self.key(), nonce);
        let mut expected = poly1305_tag(keystream.poly1305_key(), buf);
        let valid = expected.ct_eq(tag).unwrap_u8();
        expected.zeroize();

        if valid == 0 {
            return Err(Error);
        }

        keystream.xor(buf);
        Ok(())
    }

    #[inline]
    fn key(&self) -> &[u8; 32] {
        &self.key
    }
}

impl fmt::Debug for SecretBox {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str("SecretBox { .. }")
    }
}

/// XSalsa20-Poly1305 box key derived from a precomputed X25519 shared secret.
pub struct BoxKey {
    inner: SecretBox,
}

impl BoxKey {
    /// Derive a NaCl-compatible box key from the X25519 shared secret of
    /// our secret key and the peer's public key.
    ///
    /// Returns `None` for an all-zero shared secret, which a low-order peer
    /// key produces. libsodium's `crypto_box_beforenm` rejects it too. The
    /// check runs in constant time.
    pub fn from_shared_secret(shared_secret: &[u8; 32]) -> Option<Self> {
        if bool::from(shared_secret.ct_eq(&[0u8; 32])) {
            return None;
        }
        Some(Self {
            inner: SecretBox::new(hsalsa20(shared_secret, &[0u8; 16])),
        })
    }

    /// Encrypt `buf` in place and return the detached Poly1305 tag.
    #[must_use = "the tag is needed to open the box"]
    pub fn encrypt_detached(&self, nonce: &[u8; 24], buf: &mut [u8]) -> [u8; 16] {
        self.inner.encrypt_detached(nonce, buf)
    }

    /// Verify `tag`, then decrypt `buf` in place. On failure `buf` is left
    /// untouched.
    pub fn decrypt_detached(
        &self,
        nonce: &[u8; 24],
        buf: &mut [u8],
        tag: &[u8; 16],
    ) -> Result<(), Error> {
        self.inner.decrypt_detached(nonce, buf, tag)
    }
}

impl fmt::Debug for BoxKey {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str("BoxKey { .. }")
    }
}

/// `HSalsa20` key derivation used by `XSalsa20` and `NaCl` boxes.
pub fn hsalsa20(key: &[u8; 32], input: &[u8; 16]) -> [u8; 32] {
    salsa20::hsalsa20(key, input)
}

fn poly1305_tag(poly_key: &[u8; 32], data: &[u8]) -> [u8; 16] {
    Poly1305::new(poly_key.into()).compute_unpadded(data).into()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::testing::{
        ALICE_SECRET, BOB_PUBLIC, BOXED, FIRSTKEY, MESSAGE, NONCE, Rng, SECONDKEY, SHARED,
    };
    use std::vec;
    use std::vec::Vec;

    use crypto_box::aead::AeadInPlace as _;
    use crypto_box::aead::KeyInit as _;
    use crypto_box::aead::generic_array::GenericArray;
    use crypto_secretbox::XSalsa20Poly1305;
    use x25519_dalek::{PublicKey, StaticSecret};

    fn bytes(len: usize) -> Vec<u8> {
        (0..len)
            .map(|i| (i as u8).wrapping_mul(73).wrapping_add(19))
            .collect()
    }

    fn sizes() -> Vec<usize> {
        let mut out = vec![
            0, 1, 2, 15, 16, 17, 31, 32, 33, 63, 64, 65, 127, 128, 129, 255, 256, 257, 511, 512,
            513, 1024, 2048, 4096, 16_384,
        ];
        out.extend(300..330);
        out
    }

    fn rustcrypto_seal(key: &[u8; 32], nonce: &[u8; 24], buf: &mut [u8]) -> [u8; 16] {
        XSalsa20Poly1305::new(GenericArray::from_slice(key))
            .encrypt_in_place_detached(GenericArray::from_slice(nonce), b"", buf)
            .unwrap()
            .into()
    }

    #[test]
    fn nacl_hsalsa20_vectors() {
        assert_eq!(hsalsa20(&SHARED, &[0; 16]), FIRSTKEY);
        assert_eq!(
            hsalsa20(&FIRSTKEY, NONCE[..16].try_into().unwrap()),
            SECONDKEY
        );
    }

    #[test]
    fn nacl_secretbox_vector() {
        let key = SecretBox::new(FIRSTKEY);
        let mut buf = MESSAGE;
        let tag = key.encrypt_detached(&NONCE, &mut buf);
        assert_eq!(tag, BOXED[..16]);
        assert_eq!(buf, BOXED[16..]);

        key.decrypt_detached(&NONCE, &mut buf, &tag).unwrap();
        assert_eq!(buf, MESSAGE);
    }

    #[test]
    fn nacl_box_vector() {
        let shared = StaticSecret::from(ALICE_SECRET).diffie_hellman(&PublicKey::from(BOB_PUBLIC));
        assert_eq!(shared.as_bytes(), &SHARED);

        let key = BoxKey::from_shared_secret(shared.as_bytes()).unwrap();
        let mut buf = MESSAGE;
        let tag = key.encrypt_detached(&NONCE, &mut buf);
        assert_eq!(tag, BOXED[..16]);
        assert_eq!(buf, BOXED[16..]);
    }

    #[test]
    fn secretbox_matches_rustcrypto() {
        let key = [0x42u8; 32];
        let nonce = [0x24u8; 24];
        let ours = SecretBox::from_key(&key);

        for len in sizes() {
            let plaintext = bytes(len);

            let mut ours_buf = plaintext.clone();
            let ours_tag = ours.encrypt_detached(&nonce, &mut ours_buf);

            let mut expected_buf = plaintext.clone();
            let expected_tag = rustcrypto_seal(&key, &nonce, &mut expected_buf);

            assert_eq!(ours_tag, expected_tag, "tag len {len}");
            assert_eq!(ours_buf, expected_buf, "ciphertext len {len}");

            let mut opened = ours_buf.clone();
            ours.decrypt_detached(&nonce, &mut opened, &ours_tag)
                .unwrap();
            assert_eq!(opened, plaintext, "open ours len {len}");
        }
    }

    #[test]
    fn random_boxes_match_rustcrypto() {
        let mut rng = Rng::new(0xb0c5_e1ec);
        for _ in 0..300 {
            let mut key = [0u8; 32];
            let mut nonce = [0u8; 24];
            rng.fill(&mut key);
            rng.fill(&mut nonce);
            let mut plaintext = vec![0u8; rng.below(5000)];
            rng.fill(&mut plaintext);
            let len = plaintext.len();

            let mut ours_buf = plaintext.clone();
            let ours_tag = SecretBox::new(key).encrypt_detached(&nonce, &mut ours_buf);
            let mut expected_buf = plaintext.clone();
            let expected_tag = rustcrypto_seal(&key, &nonce, &mut expected_buf);
            assert_eq!(ours_tag, expected_tag, "tag len {len}");
            assert_eq!(ours_buf, expected_buf, "ciphertext len {len}");

            SecretBox::new(key)
                .decrypt_detached(&nonce, &mut expected_buf, &expected_tag)
                .unwrap();
            assert_eq!(expected_buf, plaintext, "open len {len}");
        }
    }

    #[test]
    fn boxkey_matches_crypto_box() {
        let alice_secret_bytes = [0x11u8; 32];
        let bob_secret_bytes = [0x72u8; 32];
        let alice_static = StaticSecret::from(alice_secret_bytes);
        let bob_static = StaticSecret::from(bob_secret_bytes);
        let bob_public = PublicKey::from(&bob_static);

        let crypto_secret = crypto_box::SecretKey::from(alice_secret_bytes);
        let crypto_peer = crypto_box::PublicKey::from(*bob_public.as_bytes());
        let crypto = crypto_box::SalsaBox::new(&crypto_peer, &crypto_secret);

        let shared = alice_static.diffie_hellman(&bob_public);
        let ours = BoxKey::from_shared_secret(shared.as_bytes()).unwrap();
        let nonce = [0x31u8; 24];

        for len in sizes() {
            let plaintext = bytes(len);

            let mut ours_buf = plaintext.clone();
            let ours_tag = ours.encrypt_detached(&nonce, &mut ours_buf);

            let mut expected_buf = plaintext.clone();
            let expected_tag = crypto
                .encrypt_in_place_detached(GenericArray::from_slice(&nonce), b"", &mut expected_buf)
                .unwrap();

            assert_eq!(
                ours_tag.as_slice(),
                expected_tag.as_slice(),
                "tag len {len}"
            );
            assert_eq!(ours_buf, expected_buf, "ciphertext len {len}");
        }
    }

    #[test]
    fn boxkey_shared_secret_is_symmetric() {
        let alice_secret = StaticSecret::from([0xA1u8; 32]);
        let alice_public = PublicKey::from(&alice_secret);
        let bob_secret = StaticSecret::from([0xB2u8; 32]);
        let bob_public = PublicKey::from(&bob_secret);
        let alice_box =
            BoxKey::from_shared_secret(alice_secret.diffie_hellman(&bob_public).as_bytes())
                .unwrap();
        let bob_box =
            BoxKey::from_shared_secret(bob_secret.diffie_hellman(&alice_public).as_bytes())
                .unwrap();
        let nonce = [0x55u8; 24];
        let mut msg = bytes(1024);

        let tag = alice_box.encrypt_detached(&nonce, &mut msg);
        bob_box.decrypt_detached(&nonce, &mut msg, &tag).unwrap();
        assert_eq!(msg, bytes(1024));
    }

    #[test]
    fn boxkey_rejects_zero_shared_secret() {
        let low_order = PublicKey::from([0u8; 32]);
        let shared = StaticSecret::from([0x11u8; 32]).diffie_hellman(&low_order);
        assert_eq!(shared.as_bytes(), &[0u8; 32]);
        assert!(BoxKey::from_shared_secret(shared.as_bytes()).is_none());

        let mut almost_zero = [0u8; 32];
        almost_zero[31] = 0x80;
        assert!(BoxKey::from_shared_secret(&almost_zero).is_some());
    }

    #[test]
    fn decrypt_rejects_any_flipped_bit() {
        let key = [0x7Bu8; 32];
        let nonce = [0xE1u8; 24];
        let box_ = SecretBox::from_key(&key);
        let mut sealed = bytes(129);
        let tag = box_.encrypt_detached(&nonce, &mut sealed);

        for bit in 0..sealed.len() * 8 {
            let mut buf = sealed.clone();
            buf[bit / 8] ^= 1 << (bit % 8);
            let before = buf.clone();
            assert_eq!(box_.decrypt_detached(&nonce, &mut buf, &tag), Err(Error));
            assert_eq!(buf, before, "ciphertext bit {bit}");
        }

        for bit in 0..128 {
            let mut bad_tag = tag;
            bad_tag[bit / 8] ^= 1 << (bit % 8);
            let mut buf = sealed.clone();
            assert_eq!(
                box_.decrypt_detached(&nonce, &mut buf, &bad_tag),
                Err(Error)
            );
            assert_eq!(buf, sealed, "tag bit {bit}");
        }

        let mut wrong_nonce = nonce;
        wrong_nonce[23] ^= 1;
        let mut buf = sealed.clone();
        assert_eq!(
            box_.decrypt_detached(&wrong_nonce, &mut buf, &tag),
            Err(Error)
        );

        let mut wrong_key = key;
        wrong_key[0] ^= 1;
        assert_eq!(
            SecretBox::new(wrong_key).decrypt_detached(&nonce, &mut buf, &tag),
            Err(Error)
        );
        assert_eq!(buf, sealed);
    }
}
