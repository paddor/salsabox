# salsabox

NaCl-compatible XSalsa20-Poly1305 secret boxes.

This crate keeps the high-level construction small:

- XSalsa20 from local Salsa20 code with runtime AVX2 on x86/x86_64.
- Poly1305 from RustCrypto.
- Constant-time tag verification through `subtle`.
- No X25519 key exchange dependency in the crate API. Callers pass a
  precomputed X25519 shared secret to `BoxKey`.
