// This file is a part of Julia. License is MIT: https://julialang.org/license

// ARM features definition
// hwcap
ARM_FEATURE_DEF(neon, 12, 0)
ARM_FEATURE_DEF(vfp3, 13, 0)
// ARM_FEATURE_DEF(vfpv3d16, 14, 0) // d16
ARM_FEATURE_DEF(vfp4, 16, 0)
ARM_FEATURE_DEF(hwdiv_arm, 17, 0)
ARM_FEATURE_DEF(hwdiv, 18, 0)
ARM_FEATURE_DEF(d32, 19, 0) // -d16

// hwcap2
ARM_FEATURE_DEF(crypto, 32 + 0, 0)
ARM_FEATURE_DEF(crc, 32 + 4, 0)
// ARM_FEATURE_DEF(ras, 32 + ???, 0)
// ARM_FEATURE_DEF(fullfp16, 32 + ???, 0)

// custom bits to match llvm model
ARM_FEATURE_DEF(v7, 32 * 2 + 0, 0)
ARM_FEATURE_DEF(v7a, 32 * 2 + 1, 0)
ARM_FEATURE_DEF(v7r, 32 * 2 + 2, 0)
// no v7m for now
ARM_FEATURE_DEF(v8, 32 * 2 + 3, 0)
ARM_FEATURE_DEF(v8a, 32 * 2 + 4, 0)
ARM_FEATURE_DEF(v8r, 32 * 2 + 5, 0)
ARM_FEATURE_DEF(v8_1a, 32 * 2 + 6, 0)
ARM_FEATURE_DEF(v8_2a, 32 * 2 + 7, 0)
