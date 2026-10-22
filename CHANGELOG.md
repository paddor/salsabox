# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- `SecretBox` and `BoxKey`: NaCl-compatible XSalsa20-Poly1305 with in-place
  encryption and detached tags. `BoxKey::from_shared_secret` rejects an
  all-zero shared secret.
- Salsa20 and `HSalsa20` with SIMD dispatch through `fearless_simd`: SSE4.2,
  AVX2, and AVX-512 on x86, NEON on aarch64.
