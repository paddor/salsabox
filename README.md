# salsabox

NaCl-compatible XSalsa20-Poly1305 secret boxes.

This crate keeps the high-level construction small:

- XSalsa20 from local Salsa20 code with runtime SIMD dispatch via
  `fearless_simd` (SSE4.2, AVX2, AVX-512, NEON).
- Poly1305 from RustCrypto.
- Constant-time tag verification through `subtle`.
- No X25519 key exchange dependency in the crate API. Callers pass a
  precomputed X25519 shared secret to `BoxKey`.
