//! Quick salsabox vs libsodium comparison for CI.
//!
//! Checks that both produce the same boxes on this machine, then times
//! sealing and opening in place. Prints a Markdown table, appends it to
//! the GitHub job summary, and fails when salsabox is more than
//! `SALSABOX_MAX_RATIO` (default 1.5) times slower than libsodium on
//! messages of 16 KiB and up.

use std::fmt::Write as _;
use std::hint::black_box;
use std::time::{Duration, Instant};

use salsabox::SecretBox;

#[link(name = "sodium")]
unsafe extern "C" {
    fn sodium_init() -> i32;
    fn sodium_version_string() -> *const std::ffi::c_char;
    fn crypto_secretbox_detached(
        c: *mut u8,
        mac: *mut u8,
        m: *const u8,
        mlen: u64,
        n: *const u8,
        k: *const u8,
    ) -> i32;
    fn crypto_secretbox_open_detached(
        m: *mut u8,
        c: *const u8,
        mac: *const u8,
        clen: u64,
        n: *const u8,
        k: *const u8,
    ) -> i32;
}

const SIZES: [usize; 5] = [64, 1024, 16 * 1024, 256 * 1024, 1024 * 1024];
/// Sizes from here on are gated on the ratio.
const GATED_FROM: usize = 16 * 1024;

/// Seal `buf` in place with libsodium and return the tag.
fn sodium_seal(key: &[u8; 32], nonce: &[u8; 24], buf: &mut [u8]) -> [u8; 16] {
    let mut mac = [0u8; 16];
    let ptr = buf.as_mut_ptr();
    // SAFETY: `buf` and `mac` are valid for the lengths passed; libsodium
    // allows the message and ciphertext to be the same buffer.
    let rc = unsafe {
        crypto_secretbox_detached(
            ptr,
            mac.as_mut_ptr(),
            ptr,
            buf.len() as u64,
            nonce.as_ptr(),
            key.as_ptr(),
        )
    };
    assert_eq!(rc, 0);
    mac
}

/// Open `buf` in place with libsodium.
fn sodium_open(key: &[u8; 32], nonce: &[u8; 24], buf: &mut [u8], tag: &[u8; 16]) -> bool {
    let ptr = buf.as_mut_ptr();
    // SAFETY: as in `sodium_seal`; the open may also run in place.
    let rc = unsafe {
        crypto_secretbox_open_detached(
            ptr,
            ptr,
            tag.as_ptr(),
            buf.len() as u64,
            nonce.as_ptr(),
            key.as_ptr(),
        )
    };
    rc == 0
}

/// Best of five rounds of about 20 ms each, in ns per call.
fn time_ns(mut f: impl FnMut()) -> f64 {
    let start = Instant::now();
    let mut iters = 0u32;
    while start.elapsed() < Duration::from_millis(20) {
        f();
        iters += 1;
    }

    let mut best = f64::MAX;
    for _ in 0..5 {
        let round = Instant::now();
        for _ in 0..iters {
            f();
        }
        best = best.min(round.elapsed().as_nanos() as f64 / f64::from(iters));
    }
    best
}

fn simd_level() -> &'static str {
    let level = fearless_simd::Level::new();
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    {
        if level.as_avx512().is_some() {
            return "AVX-512";
        }
        if level.as_avx2().is_some() {
            return "AVX2";
        }
        if level.as_sse4_2().is_some() {
            return "SSE4.2";
        }
    }
    #[cfg(target_arch = "aarch64")]
    if level.as_neon().is_some() {
        return "NEON";
    }
    let _ = level;
    "fallback"
}

fn cpu_model() -> String {
    std::fs::read_to_string("/proc/cpuinfo")
        .ok()
        .and_then(|info| {
            info.lines()
                .find(|line| line.starts_with("model name"))
                .and_then(|line| line.split(':').nth(1))
                .map(|model| model.trim().to_owned())
        })
        .unwrap_or_else(|| "unknown".to_owned())
}

/// Seal and open with both libraries and require identical results.
fn check_interop(key: &[u8; 32], nonce: &[u8; 24], len: usize) {
    let plaintext: Vec<u8> = (0..len)
        .map(|i| (i as u8).wrapping_mul(73) ^ 0x5a)
        .collect();
    let ours = SecretBox::from_key(key);

    let mut ours_box = plaintext.clone();
    let ours_tag = ours.encrypt_detached(nonce, &mut ours_box);
    let mut sodium_box = plaintext.clone();
    let sodium_tag = sodium_seal(key, nonce, &mut sodium_box);
    assert_eq!(ours_tag, sodium_tag, "tag mismatch at {len} B");
    assert!(ours_box == sodium_box, "ciphertext mismatch at {len} B");

    assert!(
        sodium_open(key, nonce, &mut ours_box, &ours_tag),
        "libsodium rejects salsabox box at {len} B"
    );
    ours.decrypt_detached(nonce, &mut sodium_box, &sodium_tag)
        .unwrap_or_else(|_| panic!("salsabox rejects libsodium box at {len} B"));
    assert!(
        ours_box == plaintext && sodium_box == plaintext,
        "open mismatch at {len} B"
    );
}

fn main() {
    // SAFETY: plain library initialization.
    assert!(unsafe { sodium_init() } >= 0, "sodium_init failed");
    // SAFETY: libsodium returns a static NUL-terminated string.
    let sodium_version = unsafe { std::ffi::CStr::from_ptr(sodium_version_string()) }
        .to_string_lossy()
        .into_owned();
    let max_ratio: f64 =
        std::env::var("SALSABOX_MAX_RATIO").map_or(1.5, |v| v.parse().expect("SALSABOX_MAX_RATIO"));
    let label = std::env::var("PERF_LABEL").unwrap_or_else(|_| "local".to_owned());

    let key = [0x42u8; 32];
    let nonce = [0x24u8; 24];
    let ours = SecretBox::from_key(&key);

    let mut report = String::new();
    writeln!(
        report,
        "### salsabox vs libsodium {sodium_version} ({label})\n\n{}, salsabox SIMD level {}\n",
        cpu_model(),
        simd_level()
    )
    .unwrap();
    report.push_str("| bytes | seal salsabox ns | seal libsodium ns | ratio | open salsabox ns | open libsodium ns | ratio |\n");
    report.push_str("|---:|---:|---:|---:|---:|---:|---:|\n");

    let mut failures = Vec::new();
    for len in SIZES {
        check_interop(&key, &nonce, len);

        let mut buf = vec![0x5au8; len];
        let seal_ours = time_ns(|| {
            black_box(ours.encrypt_detached(&nonce, black_box(&mut buf)));
        });
        let seal_sodium = time_ns(|| {
            black_box(sodium_seal(&key, &nonce, black_box(&mut buf)));
        });

        let sealed = vec![0x5au8; len];
        let mut boxed = sealed.clone();
        let tag = ours.encrypt_detached(&nonce, &mut boxed);
        let open_ours = time_ns(|| {
            buf.copy_from_slice(&boxed);
            ours.decrypt_detached(&nonce, black_box(&mut buf), &tag)
                .unwrap();
        });
        let open_sodium = time_ns(|| {
            buf.copy_from_slice(&boxed);
            assert!(sodium_open(&key, &nonce, black_box(&mut buf), &tag));
        });

        let seal_ratio = seal_ours / seal_sodium;
        let open_ratio = open_ours / open_sodium;
        writeln!(
            report,
            "| {len} | {seal_ours:.0} | {seal_sodium:.0} | {seal_ratio:.2}x | {open_ours:.0} | {open_sodium:.0} | {open_ratio:.2}x |"
        )
        .unwrap();
        if len >= GATED_FROM {
            for (op, ratio) in [("seal", seal_ratio), ("open", open_ratio)] {
                if ratio > max_ratio {
                    failures.push(format!("{op} {len} B: {ratio:.2}x > {max_ratio}x"));
                }
            }
        }
    }

    print!("{report}");
    if let Ok(path) = std::env::var("GITHUB_STEP_SUMMARY") {
        use std::io::Write as _;
        let mut summary = std::fs::OpenOptions::new()
            .append(true)
            .create(true)
            .open(path)
            .expect("open job summary");
        writeln!(summary, "{report}").expect("write job summary");
    }

    if !failures.is_empty() {
        eprintln!("salsabox is too slow: {}", failures.join(", "));
        std::process::exit(1);
    }
}
