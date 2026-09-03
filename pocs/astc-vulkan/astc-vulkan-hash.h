#pragma once

#include <cstddef>
#include <cstdint>
#include <string>

// Stable non-cryptographic fingerprinting for ASTC PoC artifacts.
//
// Algorithm: FNV-1a 64-bit, with the canonical offset basis and prime. This
// module is suitable for deterministic cache keys, trace identifiers and the
// compact payload checksum in the binary manifest. It is deliberately *not*
// suitable for provenance, integrity against an adversary, or signatures;
// provenance uses SHA-256 instead.
//
// Reference: G. Fowler, L. Curt Noll, et al., "The FNV Non-Cryptographic Hash
// Algorithm", http://www.isthe.com/chongo/tech/comp/fnv/.
uint64_t astc_vulkan_fnv1a64(const void * data, size_t size);

// Continues an FNV-1a stream from `state`. Passing the canonical offset basis
// permits bounded-memory checks of large cache payload ranges.
uint64_t astc_vulkan_fnv1a64_update(uint64_t state, const void * data, size_t size);

// Returns the historical `fnv1a64-%016x` spelling used in latent metadata.
// Keeping this formatting in one place protects artifact replay compatibility.
std::string astc_vulkan_fnv1a64_tagged(const void * data, size_t size);
