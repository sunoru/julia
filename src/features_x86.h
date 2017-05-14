// This file is a part of Julia. License is MIT: https://julialang.org/license

// X86 features definition
// EAX=1: ECX
X86_FEATURE_DEF(sse3, 0, 0)
X86_FEATURE_DEF(pclmul, 1, 0)
X86_FEATURE_DEF(ssse3, 9, 0)
X86_FEATURE_DEF(fma, 12, 0)
X86_FEATURE_DEF(cx16, 13, 307000) // cmpxchg16b in llvm 3.3
X86_FEATURE_DEF(sse41, 19, 0)
X86_FEATURE_DEF(sse42, 20, 0)
X86_FEATURE_DEF(movbe, 22, 0)
X86_FEATURE_DEF(popcnt, 23, 0)
X86_FEATURE_DEF(aes, 25, 0)
X86_FEATURE_DEF(xsave, 26, 30900)
X86_FEATURE_DEF(avx, 28, 0)
X86_FEATURE_DEF(f16c, 29, 0)
X86_FEATURE_DEF(rdrnd, 30, 30700) // rdrand in llvm 3.3

// EAX=1: EDX
X86_FEATURE_DEF(cmov, 32 + 15, 0)
X86_FEATURE_DEF(mmx, 32 + 23, 0)
X86_FEATURE_DEF(fxsr, 32 + 24, 40000)
X86_FEATURE_DEF(sse, 32 + 25, 0)
X86_FEATURE_DEF(sse2, 32 + 26, 0)

// EAX=7,ECX=0: EBX
X86_FEATURE_DEF(fsgsbase, 32 * 2 + 0, 0)
// X86_FEATURE_DEF(sgx, 32 * 2 + 2, 30900) // Disable for now since it's very hard to detect
X86_FEATURE_DEF(bmi, 32 * 2 + 3, 0)
X86_FEATURE_DEF(hle, 32 * 2 + 4, 0)
X86_FEATURE_DEF(avx2, 32 * 2 + 5, 0)
X86_FEATURE_DEF(bmi2, 32 * 2 + 8, 0)
// X86_FEATURE_DEF(invpcid, 32 * 2 + 10, 0) // Not used and gone in LLVM 5.0
X86_FEATURE_DEF(rtm, 32 * 2 + 11, 0)
X86_FEATURE_DEF(mpx, 32 * 2 + 14, 0)
X86_FEATURE_DEF(avx512f, 32 * 2 + 16, 30700)
X86_FEATURE_DEF(avx512dq, 32 * 2 + 17, 30700)
X86_FEATURE_DEF(rdseed, 32 * 2 + 18, 0)
X86_FEATURE_DEF(adx, 32 * 2 + 19, 0)
// X86_FEATURE_DEF(smap, 32 * 2 + 20, 0) // Not used and gone in LLVM 5.0
X86_FEATURE_DEF(avx512ifma, 32 * 2 + 21, 30900)
// X86_FEATURE_DEF(pcommit, 32 * 2 + 22, 30900) // Deprecated
X86_FEATURE_DEF(clflushopt, 32 * 2 + 23, 30900)
X86_FEATURE_DEF(clwb, 32 * 2 + 24, 30900)
X86_FEATURE_DEF(avx512pf, 32 * 2 + 26, 30700)
X86_FEATURE_DEF(avx512er, 32 * 2 + 27, 30700)
X86_FEATURE_DEF(avx512cd, 32 * 2 + 28, 30700)
X86_FEATURE_DEF(sha, 32 * 2 + 29, 30700)
X86_FEATURE_DEF(avx512bw, 32 * 2 + 30, 30700)
X86_FEATURE_DEF(avx512vl, 32 * 2 + 31, 30700)

// EAX=7,ECX=0: ECX
X86_FEATURE_DEF(prefetchwt1, 32 * 3 + 0, 30900)
X86_FEATURE_DEF(avx512vbmi, 32 * 3 + 1, 30900)
X86_FEATURE_DEF(pku, 32 * 3 + 4, 30900) // ospke
// X86_FEATURE_DEF(avx512vpopcntdq, 32 * 3 + 14, ?????)

// EAX=7,ECX=0: EDX
// X86_FEATURE_DEF(avx512_4vnniw, 32 * 4 + 2, ?????)
// X86_FEATURE_DEF(avx512_4fmaps, 32 * 4 + 3, ?????)

// EAX=0x80000001: ECX
X86_FEATURE_DEF(sahf, 32 * 5 + 0, 30900)
X86_FEATURE_DEF(lzcnt, 32 * 5 + 5, 0)
X86_FEATURE_DEF(sse4a, 32 * 5 + 6, 0)
X86_FEATURE_DEF(prfchw, 32 * 5 + 8, 0)
X86_FEATURE_DEF(xop, 32 * 5 + 11, 0)
X86_FEATURE_DEF(lwp, 32 * 5 + 15, 50000)
X86_FEATURE_DEF(fma4, 32 * 5 + 16, 0)
X86_FEATURE_DEF(tbm, 32 * 5 + 21, 30700)
X86_FEATURE_DEF(mwaitx, 32 * 5 + 29, 30900)

// EAX=0x80000001: EDX
// 3dnow is here but we don't care...
// X86_FEATURE_DEF(, 32 * 6 + ?, ?????)

// EAX=0xd: EAX
X86_FEATURE_DEF(xsaveopt, 32 * 7 + 0, 30900)
X86_FEATURE_DEF(xsavec, 32 * 7 + 1, 30900)
X86_FEATURE_DEF(xsaves, 32 * 7 + 3, 30900)

// EAX=0x80000008: EBX
X86_FEATURE_DEF(clzero, 32 * 8 + 0, 50000)
