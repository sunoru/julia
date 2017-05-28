// This file is a part of Julia. License is MIT: https://julialang.org/license

// AArch64 features definition
// hwcap
AArch64_FEATURE_DEF(crypto, 3, 0)
AArch64_FEATURE_DEF(crc, 7, 0)
AArch64_FEATURE_DEF(lse, 8, 0)
AArch64_FEATURE_DEF(fullfp16, 9, 0)
AArch64_FEATURE_DEF(rdm, 12, 50000)
AArch64_FEATURE_DEF(jscvt, 13, UINT32_MAX)
AArch64_FEATURE_DEF(fcma, 14, UINT32_MAX)
AArch64_FEATURE_DEF(lrcpc, 15, UINT32_MAX)
// AArch64_FEATURE_DEF(ras, ???, 0)
