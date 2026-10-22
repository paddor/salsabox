# salsabox

NaCl-compatible XSalsa20-Poly1305 boxes with a SIMD Salsa20 core.

- `SecretBox` is `crypto_secretbox` under a 32-byte key. `BoxKey` is the
  symmetric half of `crypto_box`, built from an X25519 shared secret.
- Boxes encrypt in place with a detached 16-byte tag. Tags are checked in
  constant time before anything is decrypted.
- SIMD via `fearless_simd`: SSE4.2, AVX2, AVX-512, NEON. `no_std`, no
  `unsafe` code.

```rust
use salsabox::SecretBox;

let key = SecretBox::new([7; 32]);
let nonce = [1; 24];
let mut buf = *b"attack at dawn";
let tag = key.encrypt_detached(&nonce, &mut buf);
key.decrypt_detached(&nonce, &mut buf, &tag).unwrap();
assert_eq!(&buf, b"attack at dawn");
```

Never seal two messages with the same key and nonce.

## What is implemented here

Salsa20, `HSalsa20`, and the box construction. Poly1305 is RustCrypto
`poly1305`, and constant-time comparison is `subtle`. Tests check every SIMD
level against the scalar core and against libsodium's NaCl test vectors.
They also cross-check RustCrypto `crypto_secretbox` and `crypto_box`. CI
runs the AVX-512 path under Intel SDE.

## License

MIT or Apache-2.0, at your option.
