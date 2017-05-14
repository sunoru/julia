// This file is a part of Julia. License is MIT: https://julialang.org/license

// Processor feature detection

#include "fix_llvm_assert.h"

#include "processor.h"

#include "julia.h"
#include "julia_internal.h"

#include <map>
#include <array>

// CPU target string is a list of strings separated by `;` each string starts with a CPU
// or architecture name and followed by an optional list of features separated by `,`.
// A "generic" or empty CPU name means the basic required feature set of the target ISA
// which is at least the architecture the C/C++ runtime is compiled with.

// CPU dispatch needs to determine the version to be used by the sysimg as well as
// the target and feature used by the JIT. Currently the only limitation on JIT target
// and feature is matching register size between the sysimg and JIT so that SIMD vectors
// can be passed correctly. This means disabling AVX and AVX2 if AVX was not enabled
// in sysimg and disabling AVX512 if it was not enabled in sysimg.
// This also possibly means that SVE needs to be disabled on AArch64 if sysimg doesn't have it
// enabled.

// CPU dispatch starts by matching the CPU name.
// If an exact match is found without any additionally enabled features, this version is used.
// If multiple exact matches are found, the one with the most features enabled that is also
// available on the current CPU is used.
// This step will query LLVM first so it can accept CPU names that is recognized by LLVM but
// not by us (yet) when LLVM is enabled.

// If no exact match is found, a feature match is performed. Known CPU names will be translated
// to feature set and unknown CPU names are ignored.
// The ones with the largest register size will be used
// (i.e. AVX512 > AVX2/AVX > SSE, SVE > ASIMD). If there's a tie, the one with the most features
// enabled will be used. If there's still a tie the one that appears earlier in the list will be
// used. (i.e. the order in the version list is significant in this case).

// Features that are not recognized will be passed to LLVM directly during codegen
// but ignored otherwise.

namespace {

// Helper functions to test/set feature bits

template<typename T1, typename T2, typename T3>
static inline bool test_bits(T1 v, T2 mask, T3 test)
{
    return T3(v & mask) == test;
}

template<typename T1, typename T2>
static inline bool test_bits(T1 v, T2 mask)
{
    return test_bits(v, mask, mask);
}

template<typename T>
static inline bool test_nbit(const uint32_t *bits, T _bitidx)
{
    auto bitidx = static_cast<uint32_t>(_bitidx);
    auto u32idx = bitidx / 32;
    auto bit = bitidx % 32;
    return (bits[u32idx] & (1 << bit)) != 0;
}

template<typename T>
static inline void unset_bits(T &bits)
{
    (void)bits;
}

template<typename T, typename T1, typename... Rest>
static inline void unset_bits(T &bits, T1 _bitidx, Rest... rest)
{
    auto bitidx = static_cast<uint32_t>(_bitidx);
    auto u32idx = bitidx / 32;
    auto bit = bitidx % 32;
    bits[u32idx] = bits[u32idx] & ~uint32_t(1 << bit);
    unset_bits(bits, rest...);
}

// Helper functions to create feature masks

static inline constexpr uint32_t add_feature_mask_u32(uint32_t mask, uint32_t u32idx)
{
    return mask;
}

template<typename T, typename... Rest>
static inline constexpr uint32_t add_feature_mask_u32(uint32_t mask, uint32_t u32idx,
                                                      T bit, Rest... args)
{
    return add_feature_mask_u32(mask | ((int(bit) >= 0 && int(bit) / 32 == (int)u32idx) ?
                                        (1 << (int(bit) % 32)) : 0),
                                u32idx, args...);
}

template<typename... Args>
static inline constexpr uint32_t get_feature_mask_u32(uint32_t u32idx, Args... args)
{
    return add_feature_mask_u32(uint32_t(0), u32idx, args...);
}

template<uint32_t... Is> struct seq{};
template<uint32_t N, uint32_t... Is>
struct gen_seq : gen_seq<N-1, N-1, Is...>{};
template<uint32_t... Is>
struct gen_seq<0, Is...> : seq<Is...>{};

template<size_t n, uint32_t... I, typename... Args>
static inline constexpr std::array<uint32_t,n>
_get_feature_mask(seq<I...>, Args... args)
{
    return std::array<uint32_t,n>{get_feature_mask_u32(I, args...)...};
}

template<size_t n, typename... Args>
static inline constexpr std::array<uint32_t,n> get_feature_masks(Args... args)
{
    return _get_feature_mask<n>(gen_seq<n>(), args...);
}

template<size_t n>
static inline void mask_features(std::array<uint32_t,n> masks, uint32_t *features)
{
    for (size_t i = 0; i < n; i++) {
        features[i] = features[i] & masks[i];
    }
}

struct TargetData {
    uint32_t cpu;
    std::string name;
    std::vector<uint32_t> features;
    jl_target_clone_t clone;
};

static inline std::vector<uint8_t> serialize_target_data(uint32_t cpu, const char *name,
                                                         uint32_t *features, uint32_t nfeature)
{
    uint32_t namelen = strlen(name);
    std::vector<uint8_t> res(4 + 4 + 4 * nfeature + 4 + namelen);
    memcpy(&res[0], &cpu, 4);
    memcpy(&res[4], &nfeature, 4);
    memcpy(&res[8], features, 4 * nfeature);
    memcpy(&res[8 + 4 * nfeature], &namelen, 4);
    memcpy(&res[12 + 4 * nfeature], name, namelen);
    return res;
}

static inline std::vector<TargetData> deserialize_target_data(const uint8_t *data)
{
    uint32_t ntarget;
    memcpy(&ntarget, data, 4);
    data += 4;
    std::vector<TargetData> res(ntarget);
    for (uint32_t i = 0; i < ntarget; i++) {
        auto &target = res[i];
        memcpy(&target.clone, data, 4);
        data += 4;
        memcpy(&target.cpu, data, 4);
        data += 4;
        uint32_t nfeature;
        memcpy(&nfeature, data, 4);
        data += 4;
        target.features.resize(nfeature);
        memcpy(&target.features[0], data, 4 * nfeature);
        data += 4 * nfeature;
        uint32_t namelen;
        memcpy(&namelen, data, 4);
        data += 4;
        target.name = std::string((const char*)data, namelen);
    }
    return res;
}

template<typename F>
static inline jl_sysimg_fptrs_t parse_sysimg(void *hdl, F &&callback)
{
    jl_sysimg_fptrs_t res = {nullptr, nullptr, 0, nullptr, nullptr};
    // TODO change this to a dummy function
    res.base = (char*)*(void**)jl_dlsym(hdl, "jl_sysimg_fptr_base");
    auto offsets = ((const int32_t*)jl_dlsym(hdl, "jl_sysimg_fptr_offsets")) + 1;
    uint32_t nfunc = ((const uint32_t*)offsets)[-1];
    res.offsets = offsets;

    void *ids = jl_dlsym_e(hdl, "jl_dispatch_target_ids");
    if (!ids)
        return res;
    int target_idx = callback(ids);

    auto got = (void**)jl_dlsym(hdl, "jl_dispatch_got");
    auto got_idxs = ((const uint32_t*)jl_dlsym(hdl, "jl_dispatch_got_idxs")) + 1;
    uint32_t num_got = got_idxs[-1];

    auto clone_offsets = (const int32_t*)jl_dlsym(hdl, "jl_dispatch_fptr_offsets");
    auto clone_idxs = (const uint32_t*)jl_dlsym(hdl, "jl_dispatch_fptr_idxs");
    uint32_t nclone = clone_idxs[0];
    clone_idxs += 1;

    // Find target
    for (int i = 0;i < target_idx;i++) {
        // TODO special case clone all to not have clone_idxs
        clone_idxs += nclone + 1;
        clone_offsets += nclone;
        nclone = clone_idxs[-1];
    }

    // Fill in return value
    if (nclone == nfunc) {
        // All function cloned
        res.offsets = clone_offsets;
        // Also assume that no GOT slots needs to be filled.
        return res;
    }
    else {
        res.nclone = nclone;
        res.clone_offsets = clone_offsets;
        res.clone_idxs = clone_idxs;
    }

    // Do relocation
    uint32_t got_i = 0;
    for (uint32_t i = 0;i < nclone;i++) {
        const uint32_t mask = 1u << 31;
        uint32_t idx = clone_idxs[i];
        if (!(idx & mask))
            continue;
        idx = idx & ~mask;
        bool found = false;
        for (;got_i < num_got;got_i++) {
            if (got_idxs[got_i] == idx) {
                found = true;
                got[got_i] = clone_offsets[i] + res.base;
                break;
            }
        }
        assert(found && "Cannot find GOT entry for cloned function.");
        (void)found;
    }

    return res;
}


/**
 * Target specific type/constant definitions, always enable.
 */

struct FeatureName {
    const char *name;
    uint32_t bit; // bit index into a `uint32_t` array;
    uint32_t min_llvm_ver; // 0 if it is available on the oldest LLVM version we support
};

template<typename CPU, typename FeatureList>
struct CPUSpec {
    const char *name;
    CPU cpu;
    CPU fallback;
    uint32_t min_llvm_ver;
    FeatureList features;
};

// Debug helper

template<typename CPU, typename FeatureList>
static inline void dump_cpu_spec(uint32_t cpu, const FeatureList &features,
                                 const FeatureName *feature_names, uint32_t nfeature_names,
                                 const CPUSpec<CPU,FeatureList> *cpus, uint32_t ncpus)
{
    bool cpu_found = false;
    for (uint32_t i = 0;i < ncpus;i++) {
        if (cpu == uint32_t(cpus[i].cpu)) {
            cpu_found = true;
            jl_safe_printf("CPU: %s\n", cpus[i].name);
            break;
        }
    }
    if (!cpu_found)
        jl_safe_printf("CPU: generic\n");
    jl_safe_printf("Features:");
    bool first = true;
    for (uint32_t i = 0;i < nfeature_names;i++) {
        if (test_nbit(&features[0], feature_names[i].bit)) {
            if (first) {
                jl_safe_printf(" %s", feature_names[i].name);
                first = false;
            }
            else {
                jl_safe_printf(", %s", feature_names[i].name);
            }
        }
    }
    jl_safe_printf("\n");
}

template<typename CPU, typename FeatureList>
static inline jl_value_t *find_cpu_name(uint32_t cpu, const CPUSpec<CPU,FeatureList> *cpus,
                                        uint32_t ncpus)
{
    const char *name = "generic";
    for (uint32_t i = 0;i < ncpus;i++) {
        if (cpu == uint32_t(cpus[i].cpu)) {
            name = cpus[i].name;
            break;
        }
    }
    return jl_cstr_to_string(name);
}

}

namespace X86 {
enum class CPU : uint32_t {
    generic = 0,
    intel_nocona,
    intel_prescott,
    intel_atom_bonnell,
    intel_atom_silvermont,
    intel_core2,
    intel_core2_penryn,
    intel_yonah,
    intel_corei7_nehalem,
    intel_corei7_westmere,
    intel_corei7_sandybridge,
    intel_corei7_ivybridge,
    intel_corei7_haswell,
    intel_corei7_broadwell,
    intel_corei7_skylake,
    intel_corei7_skylake_avx512,
    intel_corei7_cannonlake,
    intel_knights_landing,

    amd_fam10h,
    amd_athlon_fx,
    amd_athlon_64,
    amd_athlon_64_sse3,
    amd_bdver1,
    amd_bdver2,
    amd_bdver3,
    amd_bdver4,
    amd_btver1,
    amd_btver2,
    amd_k8,
    amd_k8_sse3,
    amd_opteron,
    amd_opteron_sse3,
    amd_barcelona,
    amd_znver1,
};

constexpr int feature_sz = 9;
typedef std::array<uint32_t,feature_sz> FeatureList;
constexpr FeatureName feature_names[] = {
#define X86_FEATURE_DEF(name, bit, llvmver) {#name, bit, llvmver},
#include "features_x86.h"
#undef X86_FEATURE_DEF
};
constexpr uint32_t nfeature_names = sizeof(feature_names) / sizeof(FeatureName);

template<typename... Args>
static inline constexpr std::array<uint32_t,feature_sz> get_feature_masks(Args... args)
{
    return ::get_feature_masks<feature_sz>(args...);
}

constexpr auto feature_masks = get_feature_masks(
#define X86_FEATURE_DEF(name, bit, llvmver) bit,
#include "features_x86.h"
#undef X86_FEATURE_DEF
    -1);

namespace Feature {
enum : uint32_t {
#define X86_FEATURE_DEF(name, bit, llvmver) name = bit,
#include "features_x86.h"
#undef X86_FEATURE_DEF
};

// Some of these can be simplified in c++14 by reusing the common part
// since `[]` on `std::array` is constexpr.

constexpr auto generic = get_feature_masks(mmx, cmov, sse, sse2, fxsr);
constexpr auto bonnell = get_feature_masks(mmx, cmov, sse, sse2, sse3, fxsr, cx16, movbe, sahf);
constexpr auto silvermont = get_feature_masks(mmx, cmov, sse, sse2, sse3, ssse3, sse41, sse42,
                                              fxsr, cx16, movbe, popcnt, pclmul, aes, prfchw,
                                              sahf);
constexpr auto core2 = get_feature_masks(mmx, cmov, sse, sse2, sse3, ssse3, fxsr, cx16, sahf);
constexpr auto yonah = get_feature_masks(mmx, cmov, sse, sse2, sse3, fxsr);
constexpr auto prescott = get_feature_masks(mmx, cmov, sse, sse2, sse3, fxsr);
constexpr auto nocona = get_feature_masks(mmx, cmov, sse, sse2, sse3, fxsr, cx16);
constexpr auto penryn = get_feature_masks(mmx, cmov, sse, sse2, sse3, ssse3, sse41, fxsr, cx16,
                                          sahf);
constexpr auto nehalem = get_feature_masks(mmx, cmov, sse, sse2, sse3, ssse3, sse41, sse42,
                                           fxsr, cx16, popcnt, sahf);
constexpr auto westmere = get_feature_masks(mmx, cmov, sse, sse2, sse3, ssse3, sse41, sse42,
                                            fxsr, cx16, popcnt, aes, pclmul, sahf);
constexpr auto sandybridge = get_feature_masks(mmx, cmov, sse, sse2, sse3, ssse3, sse41, sse42,
                                               avx, fxsr, cx16, popcnt, aes, pclmul, xsave,
                                               xsaveopt, sahf);
constexpr auto ivybridge = get_feature_masks(mmx, cmov, sse, sse2, sse3, ssse3, sse41, sse42,
                                             avx, fxsr, cx16, popcnt, aes, pclmul, xsave,
                                             xsaveopt, sahf, rdrnd, f16c, fsgsbase);
constexpr auto haswell = get_feature_masks(mmx, cmov, sse, sse2, sse3, ssse3, sse41, sse42,
                                           avx, fxsr, cx16, popcnt, aes, pclmul, xsave,
                                           xsaveopt, sahf, rdrnd, f16c, fsgsbase,
                                           avx2, bmi, bmi2, fma, lzcnt, movbe);
constexpr auto broadwell = get_feature_masks(mmx, cmov, sse, sse2, sse3, ssse3, sse41, sse42,
                                             avx, fxsr, cx16, popcnt, aes, pclmul, xsave,
                                             xsaveopt, sahf, rdrnd, f16c, fsgsbase,
                                             avx2, bmi, bmi2, fma, lzcnt, movbe, adx, rdseed,
                                             prfchw);
constexpr auto skylake = get_feature_masks(mmx, cmov, sse, sse2, sse3, ssse3, sse41, sse42, avx,
                                           fxsr, cx16, popcnt, aes, pclmul, xsave, xsaveopt, sahf,
                                           rdrnd, f16c, fsgsbase, avx2, bmi, bmi2, fma, lzcnt,
                                           movbe, adx, rdseed, mpx, rtm, xsavec, xsaves,
                                           clflushopt, hle, prfchw); // ignore sgx
constexpr auto knl = get_feature_masks(mmx, cmov, sse, sse2, sse3, ssse3, sse41, sse42,
                                       avx, fxsr, cx16, popcnt, aes, pclmul, xsave,
                                       xsaveopt, sahf, rdrnd, f16c, fsgsbase, avx512f, avx2,
                                       avx512er, avx512cd, avx512pf, prefetchwt1, adx, rdseed,
                                       movbe, lzcnt, bmi, bmi2, fma, prfchw);
constexpr auto skx = get_feature_masks(mmx, cmov, sse, sse2, sse3, ssse3, sse41, sse42, avx, fxsr,
                                       cx16, popcnt, aes, pclmul, xsave, xsaveopt, sahf, rdrnd,
                                       f16c, fsgsbase, avx2, bmi, bmi2, fma, lzcnt, movbe, adx,
                                       rdseed, mpx, rtm, xsavec, xsaves, clflushopt, hle, prfchw,
                                       avx512f, avx512cd, avx512dq, avx512bw, avx512vl, pku,
                                       clwb); // ignore sgx
constexpr auto cannonlake = get_feature_masks(mmx, cmov, sse, sse2, sse3, ssse3, sse41, sse42,
                                              avx, fxsr, cx16, popcnt, aes, pclmul, xsave,
                                              xsaveopt, sahf, rdrnd, f16c, fsgsbase, avx2, bmi,
                                              bmi2, fma, lzcnt, movbe, adx, rdseed, mpx, rtm,
                                              xsavec, xsaves, clflushopt, hle, prfchw, avx512f,
                                              avx512cd, avx512dq, avx512bw, avx512vl, pku, clwb,
                                              avx512vbmi, avx512ifma, sha); // ignore sgx

constexpr auto k8_sse3 = get_feature_masks(mmx, cmov, sse, sse2, sse3, fxsr, cx16);
constexpr auto amdfam10 = get_feature_masks(mmx, cmov, sse, sse2, sse3, sse4a, fxsr, cx16,
                                            lzcnt, popcnt, sahf);

constexpr auto btver1 = get_feature_masks(mmx, cmov, sse, sse2, sse3, ssse3, sse4a, fxsr, cx16,
                                          prfchw, lzcnt, popcnt, sahf);
constexpr auto btver2 = get_feature_masks(mmx, cmov, sse, sse2, sse3, ssse3, sse41, sse42, avx,
                                          fxsr, sse4a, cx16, prfchw, aes, pclmul, bmi, f16c,
                                          movbe, lzcnt, popcnt, xsave, xsaveopt, sahf);

constexpr auto bdver1 = get_feature_masks(xop, fma4, avx, cmov, sse, sse2, sse3, ssse3, sse41,
                                          sse42, cx16, aes, prfchw, pclmul, mmx, fxsr, sse4a,
                                          lzcnt, popcnt, xsave, lwp, sahf);
constexpr auto bdver2 = get_feature_masks(xop, fma4, avx, cmov, sse, sse2, sse3, ssse3, sse41,
                                          sse42, cx16, aes, prfchw, pclmul, mmx, fxsr, sse4a,
                                          f16c, lzcnt, popcnt, xsave, bmi, tbm, lwp, fma, sahf);
constexpr auto bdver3 = get_feature_masks(xop, fma4, avx, cmov, sse, sse2, sse3, ssse3, sse41,
                                          sse42, cx16, aes, prfchw, pclmul, mmx, fxsr, sse4a,
                                          f16c, lzcnt, popcnt, xsave, bmi, tbm, lwp, fma,
                                          xsaveopt, fsgsbase, sahf);
constexpr auto bdver4 = get_feature_masks(mmx, avx2, avx, cmov, sse, sse2, sse3, ssse3, sse41,
                                          sse42, fxsr, xop, fma4, cx16, aes, prfchw, pclmul,
                                          f16c, lzcnt, popcnt, xsave, bmi, bmi2, tbm, lwp, fma,
                                          xsaveopt, fsgsbase, sahf, mwaitx);

constexpr auto znver1 = get_feature_masks(adx, aes, avx2, avx, cmov, sse, sse2, sse3, ssse3, sse41,
                                          sse42, bmi, bmi2, clflushopt, clzero, cx16, f16c, fma,
                                          fsgsbase, fxsr, sahf, lzcnt, mmx, movbe, mwaitx, pclmul,
                                          popcnt, prfchw, rdrnd, rdseed, sha, sse4a, xsave,
                                          xsavec, xsaveopt, xsaves);

}

constexpr CPUSpec<CPU, FeatureList> cpus[] = {
    {"generic", CPU::generic, CPU::generic, 0, Feature::generic},
    {"bonnell", CPU::intel_atom_bonnell, CPU::generic, 0, Feature::bonnell},
    {"silvermont", CPU::intel_atom_silvermont, CPU::generic, 0, Feature::silvermont},
    {"core2", CPU::intel_core2, CPU::generic, 0, Feature::core2},
    {"yonah", CPU::intel_yonah, CPU::generic, 0, Feature::yonah},
    {"prescott", CPU::intel_prescott, CPU::generic, 0, Feature::prescott},
    {"nocona", CPU::intel_nocona, CPU::generic, 0, Feature::nocona},
    {"penryn", CPU::intel_core2_penryn, CPU::generic, 0, Feature::penryn},
    {"nehalem", CPU::intel_corei7_nehalem, CPU::generic, 0, Feature::nehalem},
    {"westmere", CPU::intel_corei7_westmere, CPU::generic, 0, Feature::westmere},
    {"sandybridge", CPU::intel_corei7_sandybridge, CPU::generic, 0, Feature::sandybridge},
    {"ivybridge", CPU::intel_corei7_ivybridge, CPU::generic, 0, Feature::ivybridge},
    {"haswell", CPU::intel_corei7_haswell, CPU::generic, 0, Feature::haswell},
    {"broadwell", CPU::intel_corei7_broadwell, CPU::intel_corei7_haswell, 30700,
     Feature::broadwell},
    {"skylake", CPU::intel_corei7_skylake, CPU::intel_corei7_haswell, 30700, Feature::skylake},
    {"knl", CPU::intel_knights_landing, CPU::intel_corei7_haswell, 30700, Feature::knl},
    {"skylake-avx512", CPU::intel_corei7_skylake_avx512, CPU::intel_corei7_haswell, 30700,
     Feature::skx},
    {"cannonlake", CPU::intel_corei7_cannonlake, CPU::intel_corei7_skylake, 40000,
     Feature::cannonlake},

    {"athlon64", CPU::amd_athlon_64, CPU::generic, 0, Feature::generic},
    {"athlon-fx", CPU::amd_athlon_fx, CPU::generic, 0, Feature::generic},
    {"k8", CPU::amd_k8, CPU::generic, 0, Feature::generic},
    {"opteron", CPU::amd_opteron, CPU::generic, 0, Feature::generic},

    {"athlon64-sse3", CPU::amd_athlon_64_sse3, CPU::generic, 0, Feature::k8_sse3},
    {"k8-sse3", CPU::amd_k8_sse3, CPU::generic, 0, Feature::k8_sse3},
    {"opteron-sse3", CPU::amd_opteron_sse3, CPU::generic, 0, Feature::k8_sse3},

    {"amdfam10", CPU::amd_fam10h, CPU::generic, 0, Feature::amdfam10},
    {"barcelona", CPU::amd_barcelona, CPU::amd_fam10h, 30900, Feature::amdfam10},

    {"btver1", CPU::amd_btver1, CPU::generic, 0, Feature::btver1},
    {"btver2", CPU::amd_btver2, CPU::generic, 0, Feature::btver2},

    {"bdver1", CPU::amd_bdver1, CPU::generic, 0, Feature::bdver1},
    {"bdver2", CPU::amd_bdver2, CPU::generic, 0, Feature::bdver2},
    {"bdver3", CPU::amd_bdver3, CPU::amd_bdver2, 30700, Feature::bdver3},
    {"bdver4", CPU::amd_bdver4, CPU::amd_bdver2, 30700, Feature::bdver4},

    {"znver1", CPU::amd_znver1, CPU::amd_btver2, 30900, Feature::znver1},
};

}

#if defined(_CPU_X86_) || defined(_CPU_X86_64_)

static inline const char *normalize_cpu_name(const char *name)
{
    if (strcmp(name, "atom") == 0)
        return "bonnell";
    if (strcmp(name, "slm") == 0)
        return "silvermont";
    if (strcmp(name, "corei7") == 0)
        return "nehalem";
    if (strcmp(name, "corei7-avx") == 0)
        return "sandybridge";
    if (strcmp(name, "core-avx-i") == 0)
        return "ivybridge";
    if (strcmp(name, "core-avx2") == 0)
        return "haswell";
    if (strcmp(name, "skx") == 0)
        return "skylake-avx512";
    return name;
}

// CPUID

extern "C" JL_DLLEXPORT void jl_cpuid(int32_t CPUInfo[4], int32_t InfoType)
{
#if defined _MSC_VER
    __cpuid(CPUInfo, InfoType);
#else
    asm volatile (
#if defined(__i386__) && defined(__PIC__)
        "xchg %%ebx, %%esi;"
        "cpuid;"
        "xchg %%esi, %%ebx;" :
        "=S" (CPUInfo[1]),
#else
        "cpuid" :
        "=b" (CPUInfo[1]),
#endif
        "=a" (CPUInfo[0]),
        "=c" (CPUInfo[2]),
        "=d" (CPUInfo[3]) :
        "a" (InfoType)
        );
#endif
}

extern "C" JL_DLLEXPORT void jl_cpuidex(int32_t CPUInfo[4], int32_t InfoType, int32_t subInfoType)
{
#if defined _MSC_VER
    __cpuidex(CPUInfo, InfoType, subInfoType);
#else
    asm volatile (
#if defined(__i386__) && defined(__PIC__)
        "xchg %%ebx, %%esi;"
        "cpuid;"
        "xchg %%esi, %%ebx;" :
        "=S" (CPUInfo[1]),
#else
        "cpuid" :
        "=b" (CPUInfo[1]),
#endif
        "=a" (CPUInfo[0]),
        "=c" (CPUInfo[2]),
        "=d" (CPUInfo[3]) :
        "a" (InfoType),
        "c" (subInfoType)
        );
#endif
}

namespace X86 {

// For CPU model and feature detection on X86

const int SIG_INTEL = 0x756e6547; // Genu
const int SIG_AMD = 0x68747541; // Auth

static uint64_t get_xcr0(void)
{
#if defined _MSC_VER
    return _xgetbv(_XCR_XFEATURE_ENABLED_MASK);
#else
    uint32_t eax, edx;
    asm volatile ("xgetbv" : "=a" (eax), "=d" (edx) : "c" (0));
    return (uint64_t(edx) << 32) | eax;
#endif
}

static CPU get_intel_processor_name(uint32_t family, uint32_t model, uint32_t brand_id,
                                    const uint32_t *features)
{
    if (brand_id != 0)
        return CPU::generic;
    switch (family) {
    case 3:
    case 4:
    case 5:
        return CPU::generic;
    case 6:
        switch (model) {
        case 0x01: // Pentium Pro processor
        case 0x03: // Intel Pentium II OverDrive processor, Pentium II processor, model 03
        case 0x05: // Pentium II processor, model 05, Pentium II Xeon processor,
            // model 05, and Intel Celeron processor, model 05
        case 0x06: // Celeron processor, model 06
        case 0x07: // Pentium III processor, model 07, and Pentium III Xeon processor, model 07
        case 0x08: // Pentium III processor, model 08, Pentium III Xeon processor,
            // model 08, and Celeron processor, model 08
        case 0x0a: // Pentium III Xeon processor, model 0Ah
        case 0x0b: // Pentium III processor, model 0Bh
        case 0x09: // Intel Pentium M processor, Intel Celeron M processor model 09.
        case 0x0d: // Intel Pentium M processor, Intel Celeron M processor, model
            // 0Dh. All processors are manufactured using the 90 nm process.
        case 0x15: // Intel EP80579 Integrated Processor and Intel EP80579
            // Integrated Processor with Intel QuickAssist Technology
            return CPU::generic;
        case 0x0e: // Intel Core Duo processor, Intel Core Solo processor, model
            // 0Eh. All processors are manufactured using the 65 nm process.
            return CPU::intel_yonah;
        case 0x0f: // Intel Core 2 Duo processor, Intel Core 2 Duo mobile
            // processor, Intel Core 2 Quad processor, Intel Core 2 Quad
            // mobile processor, Intel Core 2 Extreme processor, Intel
            // Pentium Dual-Core processor, Intel Xeon processor, model
            // 0Fh. All processors are manufactured using the 65 nm process.
        case 0x16: // Intel Celeron processor model 16h. All processors are
            // manufactured using the 65 nm process
            return CPU::intel_core2;
        case 0x17: // Intel Core 2 Extreme processor, Intel Xeon processor, model
            // 17h. All processors are manufactured using the 45 nm process.
            //
            // 45nm: Penryn , Wolfdale, Yorkfield (XE)
        case 0x1d: // Intel Xeon processor MP. All processors are manufactured using
            // the 45 nm process.
            return CPU::intel_core2_penryn;
        case 0x1a: // Intel Core i7 processor and Intel Xeon processor. All
            // processors are manufactured using the 45 nm process.
        case 0x1e: // Intel(R) Core(TM) i7 CPU         870  @ 2.93GHz.
            // As found in a Summer 2010 model iMac.
        case 0x1f:
        case 0x2e: // Nehalem EX
            return CPU::intel_corei7_nehalem;
        case 0x25: // Intel Core i7, laptop version.
        case 0x2c: // Intel Core i7 processor and Intel Xeon processor. All
            // processors are manufactured using the 32 nm process.
        case 0x2f: // Westmere EX
            return CPU::intel_corei7_westmere;
        case 0x2a: // Intel Core i7 processor. All processors are manufactured
            // using the 32 nm process.
        case 0x2d:
            return CPU::intel_corei7_sandybridge;
        case 0x3a:
        case 0x3e: // Ivy Bridge EP
            return CPU::intel_corei7_ivybridge;

            // Haswell:
        case 0x3c:
        case 0x3f:
        case 0x45:
        case 0x46:
            return CPU::intel_corei7_haswell;

            // Broadwell:
        case 0x3d:
        case 0x47:
        case 0x4f:
        case 0x56:
            return CPU::intel_corei7_broadwell;

            // Skylake:
        case 0x4e: // Skylake mobile
        case 0x5e: // Skylake desktop
        case 0x8e: // Kaby Lake mobile
        case 0x9e: // Kaby Lake desktop
            return CPU::intel_corei7_skylake;

            // Skylake Xeon:
        case 0x55:
            if (test_nbit(features, Feature::avx512f))
                return CPU::intel_corei7_skylake_avx512;
            return CPU::intel_corei7_skylake;

        case 0x1c: // Most 45 nm Intel Atom processors
        case 0x26: // 45 nm Atom Lincroft
        case 0x27: // 32 nm Atom Medfield
        case 0x35: // 32 nm Atom Midview
        case 0x36: // 32 nm Atom Midview
            return CPU::intel_atom_bonnell;

            // Atom Silvermont codes from the Intel software optimization guide.
        case 0x37:
        case 0x4a:
        case 0x4d:
        case 0x5a:
        case 0x5d:
        case 0x4c: // really airmont
            return CPU::intel_atom_silvermont;

        case 0x57:
            return CPU::intel_knights_landing;

        default:
            return CPU::generic;
        }
        break;
    case 15: {
        switch (model) {
        case 0: // Pentium 4 processor, Intel Xeon processor. All processors are
            // model 00h and manufactured using the 0.18 micron process.
        case 1: // Pentium 4 processor, Intel Xeon processor, Intel Xeon
            // processor MP, and Intel Celeron processor. All processors are
            // model 01h and manufactured using the 0.18 micron process.
        case 2: // Pentium 4 processor, Mobile Intel Pentium 4 processor - M,
            // Intel Xeon processor, Intel Xeon processor MP, Intel Celeron
            // processor, and Mobile Intel Celeron processor. All processors
            // are model 02h and manufactured using the 0.13 micron process.
        default:
            return CPU::generic;

        case 3: // Pentium 4 processor, Intel Xeon processor, Intel Celeron D
            // processor. All processors are model 03h and manufactured using
            // the 90 nm process.
        case 4: // Pentium 4 processor, Pentium 4 processor Extreme Edition,
            // Pentium D processor, Intel Xeon processor, Intel Xeon
            // processor MP, Intel Celeron D processor. All processors are
            // model 04h and manufactured using the 90 nm process.
        case 6: // Pentium 4 processor, Pentium D processor, Pentium processor
            // Extreme Edition, Intel Xeon processor, Intel Xeon processor
            // MP, Intel Celeron D processor. All processors are model 06h
            // and manufactured using the 65 nm process.
#ifdef _CPU_X86_64_
            return CPU::intel_nocona;
#else
            return CPU::intel_prescott;
#endif
        }
    }
    default:
        break; /*"generic"*/
    }
    return CPU::generic;
}

static CPU get_amd_processor_name(uint32_t family, uint32_t model, const uint32_t *features)
{
    switch (family) {
    case 4:
    case 5:
    case 6:
    default:
        return CPU::generic;
    case 15:
        if (test_nbit(features, Feature::sse3))
            return CPU::amd_k8_sse3;
        switch (model) {
        case 1:
            return CPU::amd_opteron;
        case 5:
            return CPU::amd_athlon_fx;
        default:
            return CPU::amd_athlon_64;
        }
    case 16:
        switch (model) {
        case 2:
            return CPU::amd_barcelona;
        case 4:
        case 8:
        default:
            return CPU::amd_fam10h;
        }
    case 20:
        return CPU::amd_btver1;
    case 21:
        if (!test_nbit(features, Feature::avx))
            return CPU::amd_btver1;
        if (model >= 0x50 && model <= 0x6f)
            return CPU::amd_bdver4;
        if (model >= 0x30 && model <= 0x3f)
            return CPU::amd_bdver3;
        if (model >= 0x10 && model <= 0x1f)
            return CPU::amd_bdver2;
        if (model <= 0x0f)
            return CPU::amd_bdver1;
        return CPU::amd_btver1; // fallback
    case 22:
        if (!test_nbit(features, Feature::avx))
            return CPU::amd_btver1;
        return CPU::amd_btver2;
    case 23:
        if (test_nbit(features, Feature::adx))
            return CPU::amd_znver1;
        return CPU::amd_btver1;
    }
}

const auto host_cpu = [] {
    FeatureList features = {};

    int32_t info0[4];
    jl_cpuid(info0, 0);
    int32_t info1[4];
    jl_cpuid(info1, 1);

    uint32_t maxleaf = info0[0];
    auto vendor = info0[1];
    auto brand_id = info1[1] & 0xff;

    auto family = (info1[0] >> 8) & 0xf; // Bits 8 - 11
    auto model = (info1[0] >> 4) & 0xf;  // Bits 4 - 7
    if (family == 6 || family == 0xf) {
        if (family == 0xf)
            // Examine extended family ID if family ID is F.
            family += (info1[0] >> 20) & 0xff; // Bits 20 - 27
        // Examine extended model ID if family ID is 6 or F.
        model += ((info1[0] >> 16) & 0xf) << 4; // Bits 16 - 19
    }

    // Fill in the features
    features[0] = info1[2];
    features[1] = info1[3];
    if (maxleaf >= 7) {
        int32_t info7[4];
        jl_cpuidex(info7, 7, 0);
        features[2] = info7[1];
        features[3] = info7[2];
        features[4] = info7[3];
    }
    int32_t infoex0[4];
    jl_cpuid(infoex0, 0x80000000);
    uint32_t maxexleaf = infoex0[0];
    if (maxexleaf >= 0x80000001) {
        int32_t infoex1[4];
        jl_cpuid(infoex1, 0x80000001);
        features[5] = infoex1[2];
        features[6] = infoex1[3];
    }
    if (maxleaf >= 0xd) {
        int32_t infod[4];
        jl_cpuidex(infod, 0xd, 0x1);
        features[7] = infod[0];
    }
    if (maxexleaf >= 0x80000008) {
        int32_t infoex8[4];
        jl_cpuidex(infoex8, 0x80000008, 0);
        features[8] = infoex8[1];
    }

    // Fix up AVX bits to account for OS support and match LLVM model
    uint64_t xcr0 = 0;
    const uint32_t avx_mask = (1 << 27) | (1 << 28);
    bool hasavx = test_bits(features[1], avx_mask);
    if (hasavx) {
        xcr0 = get_xcr0();
        hasavx = test_bits(xcr0, 0x6);
    }
    unset_bits(features, 32 + 27);
    if (!hasavx) {
        using namespace Feature;
        unset_bits(features, avx, Feature::fma, f16c, xsave, avx2, xop, fma4,
                   xsaveopt, xsavec, xsaves);
    }
    bool hasavx512save = hasavx && test_bits(xcr0, 0xe0);
    if (!hasavx512save) {
        using namespace Feature;
        unset_bits(features, avx512f, avx512dq, avx512ifma, avx512pf, avx512er, avx512cd,
                   avx512bw, avx512vl, avx512vbmi);
    }
    // Ignore feature bits that we are not interested in.
    mask_features(feature_masks, &features[0]);

    uint32_t cpu;
    if (vendor == SIG_INTEL) {
        cpu = uint32_t(get_intel_processor_name(family, model, brand_id, &features[0]));
    }
    else if (vendor == SIG_AMD) {
        cpu = uint32_t(get_amd_processor_name(family, model, &features[0]));
    }
    else {
        cpu = uint32_t(CPU::generic);
    }

    return std::make_pair(cpu, features);
}();

}

extern "C" JL_DLLEXPORT void jl_dump_host_cpu()
{
    using namespace X86;
    dump_cpu_spec(host_cpu.first, host_cpu.second, feature_names, nfeature_names,
                  cpus, sizeof(cpus) / sizeof(cpus[0]));
}

extern "C" JL_DLLEXPORT jl_value_t *jl_get_cpu_name(void)
{
    using namespace X86;
    return find_cpu_name(host_cpu.first, cpus, sizeof(cpus) / sizeof(cpus[0]));
}

jl_sysimg_fptrs_t jl_init_processor_sysimg(void *hdl)
{
    return parse_sysimg(hdl, [] (void *id) {
            auto targets = deserialize_target_data((const uint8_t*)id);
            // TODO
            return 0;
        });
}
std::pair<std::string,std::string> jl_get_llvm_target(uint32_t llvmver);
std::vector<jl_target_spec_t> jl_get_llvm_clone_targets(uint32_t llvmver);

extern "C" int jl_test_cpu_feature(jl_cpu_feature_t feature)
{
    using namespace X86;
    if (feature >= 32 * feature_sz)
        return 0;
    return test_nbit(&host_cpu.second[0], feature);
}

// -- set/clear the FZ/DAZ flags on x86 & x86-64 --

// Cache of information recovered from `cpuid` since executing `cpuid` it at runtime is slow.
static uint32_t subnormal_flags = [] {
    int32_t info[4];
    jl_cpuid(info, 0);
    if (info[0] >= 1) {
        jl_cpuid(info, 1);
        if (info[3] & (1 << 26)) {
            // SSE2 supports both FZ and DAZ
            return 0x00008040;
        }
        else if (info[3] & (1 << 25)) {
            // SSE supports only the FZ flag
            return 0x00008000;
        }
    }
    return 0;
}();

// Returns non-zero if subnormals go to 0; zero otherwise.
extern "C" JL_DLLEXPORT int32_t jl_get_zero_subnormals(void)
{
    return _mm_getcsr() & subnormal_flags;
}

// Return zero on success, non-zero on failure.
extern "C" JL_DLLEXPORT int32_t jl_set_zero_subnormals(int8_t isZero)
{
    uint32_t flags = subnormal_flags;
    if (flags) {
        uint32_t state = _mm_getcsr();
        if (isZero)
            state |= flags;
        else
            state &= ~flags;
        _mm_setcsr(state);
        return 0;
    }
    else {
        // Report a failure only if user is trying to enable FTZ/DAZ.
        return isZero;
    }
}

#elif defined(_CPU_AARCH64_)

// FZ, bit [24]
static const uint32_t fpcr_fz_mask = 1 << 24;

static inline uint32_t get_fpcr_aarch64(void)
{
    uint32_t fpcr;
    asm volatile("mrs %0, fpcr" : "=r"(fpcr));
    return fpcr;
}

static inline void set_fpcr_aarch64(uint32_t fpcr)
{
    asm volatile("msr fpcr, %0" :: "r"(fpcr));
}

extern "C" JL_DLLEXPORT int32_t jl_get_zero_subnormals(void)
{
    return (get_fpcr_aarch64() & fpcr_fz_mask) != 0;
}

extern "C" JL_DLLEXPORT int32_t jl_set_zero_subnormals(int8_t isZero)
{
    uint32_t fpcr = get_fpcr_aarch64();
    fpcr = isZero ? (fpcr | fpcr_fz_mask) : (fpcr & ~fpcr_fz_mask);
    set_fpcr_aarch64(fpcr);
    return 0;
}

#else

extern "C" JL_DLLEXPORT int32_t jl_get_zero_subnormals(void)
{
    return 0;
}

extern "C" JL_DLLEXPORT int32_t jl_set_zero_subnormals(int8_t isZero)
{
    return isZero;
}

#endif
