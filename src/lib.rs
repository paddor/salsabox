#![no_std]

#[cfg(any(feature = "std", test))]
extern crate std;

use core::fmt;

use poly1305::universal_hash::KeyInit;
use poly1305::{Key, Poly1305};
use subtle::ConstantTimeEq;
use zeroize::{Zeroize, Zeroizing};

mod salsa20;

/// Authentication failure.
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
    pub fn encrypt_detached(&self, nonce: &[u8; 24], buf: &mut [u8]) -> [u8; 16] {
        let keystream = salsa20::Keystream::new(self.key(), nonce);
        let poly_key = Zeroizing::new(*keystream.poly1305_key());
        keystream.xor(buf);
        poly1305_tag(&poly_key, buf)
    }

    /// Verify `tag`, then decrypt `buf` in place.
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
    /// Derive a NaCl-compatible box key from a 32-byte X25519 shared secret.
    pub fn from_shared_secret(shared_secret: &[u8; 32]) -> Self {
        let subkey = hsalsa20(shared_secret, &[0u8; 16]);
        Self {
            inner: SecretBox::new(subkey),
        }
    }

    /// Encrypt `buf` in place and return the detached Poly1305 tag.
    pub fn encrypt_detached(&self, nonce: &[u8; 24], buf: &mut [u8]) -> [u8; 16] {
        self.inner.encrypt_detached(nonce, buf)
    }

    /// Verify `tag`, then decrypt `buf` in place.
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
    let tag = Poly1305::new(Key::from_slice(poly_key)).compute_unpadded(data);
    let mut out = [0u8; 16];
    out.copy_from_slice(&tag);
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::vec;
    use std::vec::Vec;

    use crypto_box::aead::AeadInPlace as _;
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

    #[test]
    fn secretbox_matches_rustcrypto() {
        let key = [0x42u8; 32];
        let nonce = [0x24u8; 24];
        let ours = SecretBox::from_key(&key);
        let rustcrypto = XSalsa20Poly1305::new(GenericArray::from_slice(&key));

        for len in sizes() {
            let plaintext = bytes(len);

            let mut ours_buf = plaintext.clone();
            let ours_tag = ours.encrypt_detached(&nonce, &mut ours_buf);

            let mut expected_buf = plaintext.clone();
            let expected_tag = rustcrypto
                .encrypt_in_place_detached(GenericArray::from_slice(&nonce), b"", &mut expected_buf)
                .unwrap();

            assert_eq!(
                ours_tag.as_slice(),
                expected_tag.as_slice(),
                "tag len {len}"
            );
            assert_eq!(ours_buf, expected_buf, "ciphertext len {len}");

            let mut opened = ours_buf.clone();
            ours.decrypt_detached(&nonce, &mut opened, &ours_tag)
                .unwrap();
            assert_eq!(opened, plaintext, "open ours len {len}");

            let mut expected_tag_bytes = [0u8; 16];
            expected_tag_bytes.copy_from_slice(&expected_tag);
            let mut cross = expected_buf;
            ours.decrypt_detached(&nonce, &mut cross, &expected_tag_bytes)
                .unwrap();
            assert_eq!(cross, plaintext, "open rustcrypto len {len}");
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
        let ours = BoxKey::from_shared_secret(shared.as_bytes());
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
            BoxKey::from_shared_secret(alice_secret.diffie_hellman(&bob_public).as_bytes());
        let bob_box =
            BoxKey::from_shared_secret(bob_secret.diffie_hellman(&alice_public).as_bytes());
        let nonce = [0x55u8; 24];
        let mut msg = bytes(1024);

        let tag = alice_box.encrypt_detached(&nonce, &mut msg);
        bob_box.decrypt_detached(&nonce, &mut msg, &tag).unwrap();
        assert_eq!(msg, bytes(1024));
    }

    #[test]
    fn decrypt_rejects_bad_tag_and_preserves_ciphertext() {
        let key = [0x7Bu8; 32];
        let nonce = [0xE1u8; 24];
        let box_ = SecretBox::from_key(&key);
        let mut buf = bytes(513);
        let mut tag = box_.encrypt_detached(&nonce, &mut buf);
        tag[3] ^= 0x80;
        let before = buf.clone();

        assert_eq!(box_.decrypt_detached(&nonce, &mut buf, &tag), Err(Error));
        assert_eq!(buf, before);
    }
}
