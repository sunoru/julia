// This file is a part of Julia. License is MIT: https://julialang.org/license

#include "support/dtypes.h"

#include "julia.h"

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * Related sysimg variables
 *
 * # Global function pointers
 * `jl_sysimg_fptr_base`: The base function pointers (all other pointers are stored as offsets)
 * `jl_sysimg_fptr_offsets`: The array of function offsets (`int32_t`) from the base pointer.
 *     This includes all functions in sysimg.
 *     The default implementation is used if the function is cloned.
 *
 * # Target data and dispatch slots (Only needed by runtime during loading)
 * `jl_dispatch_target_ids`: serialize target data.
 *     This contains the number of targets which is needed to decode `jl_dispatch_fptr_idxs`.
 * `jl_dispatch_got`: The runtime dispatch slots
 *     (the code in the sysimg will load from these slots)
 * `jl_dispatch_got_idxs`: The global index (`uint32_t`, into `jl_sysimg_fptr_offsets`)
 *     corresponding to each of the slots in `jl_dispatch_got`. (Sorted)
 *     The first element is the number of got slots.
 *
 * # Target functions
 * `jl_dispatch_fptr_offsets`: The function pointer offsets corresponding to each targets.
 *     This contains all the functions that are different from the default version.
 *     Not all of these needs to be filled into `jl_dispatch_got`.
 * `jl_dispatch_fptr_idxs`: The function pointer count and global index corresponding to
 *     each of the pointer offsets in `jl_dispatch_fptr_offsets`.
 *     There is one group corresponds to each target, which contains a `uint32_t` count follows
 *     by corresponding numbers of `uint32_t` indices.
 *     The count is also needed to decode `jl_dispatch_fptr_offsets`.
 *     Each of the indices arrays are sorted.
 *     The highest bit of each indices is a tag bit. When it is set, the function pointer needs
 *     To be filled into the `jl_dispatch_got`.
 */

typedef enum {
    JL_CLONE_ALL = 0,
    JL_CLONE_FMA = 1 << 0,
    JL_CLONE_LOOP = 1 << 1,
    JL_CLONE_SIMD = 1 << 2,
    JL_CLONE_SIMD_CALL = 1 << 3,
} jl_target_clone_t;

typedef enum {
#define X86_FEATURE_DEF(name, bit, llvmver) JL_X86_##name = bit,
#include "features_x86.h"
#undef X86_FEATURE_DEF
#define ARM_FEATURE_DEF(name, bit, llvmver) JL_ARM_##name = bit,
#include "features_arm.h"
#undef ARM_FEATURE_DEF
#define AArch64_FEATURE_DEF(name, bit, llvmver) JL_AArch64_##name = bit,
#include "features_aarch64.h"
#undef AArch64_FEATURE_DEF
} jl_cpu_feature_t;

int jl_test_cpu_feature(jl_cpu_feature_t feature);

typedef struct {
    char *base; // base function pointer
    const int32_t *offsets; // function pointer offsets
    uint32_t nclone; // number of cloned functions
    const int32_t *clone_offsets; // function pointer offsets of the cloned functions
    const uint32_t *clone_idxs; // sorted indices of the cloned functions (including the tag bit)
} jl_sysimg_fptrs_t;

/**
 * Initialize the processor dispatch system with sysimg `hdl` (also initialize the sysimg itself).
 * The dispatch system will find the best implementation to be used in this session.
 * The decision will be based on the host CPU and features as well as the `cpu_target`
 * option. This must be called before initializing JIT and should only be called once.
 * An error will be raised if this is called more than once or none of the implementation
 * supports the current system.
 *
 * Return the data about the function pointers selected.
 */
jl_sysimg_fptrs_t jl_init_processor_sysimg(void *hdl);

#if defined(_CPU_X86_) || defined(_CPU_X86_64_)
JL_DLLEXPORT void jl_cpuid(int32_t CPUInfo[4], int32_t InfoType);
JL_DLLEXPORT void jl_cpuidex(int32_t CPUInfo[4], int32_t InfoType, int32_t subInfoType);
#endif
JL_DLLEXPORT jl_value_t *jl_get_cpu_name(void);
JL_DLLEXPORT void jl_dump_host_cpu(void);

#ifdef __cplusplus
}

#include <utility>
#include <string>
#include <vector>

/**
 * Returns the CPU name and feature string to be used by LLVM JIT.
 *
 * If the detected/specified CPU name is not available on the LLVM version specified,
 * a fallback CPU name will be. Unsupported features will be ignored.
 */
std::pair<std::string,std::string> jl_get_llvm_target(uint32_t llvmver);

struct jl_target_spec_t {
    // LLVM target name
    std::string cpu_name;
    // LLVM feature string
    std::string cpu_features;
    // serialized identification data
    std::vector<uint8_t> data;
    // Clone condition.
    jl_target_clone_t cond;
    // Recorded clone condition. This should be a subset of `cond` and the cloning pass should
    // record if there's any function that's cloned meets this condition.
    jl_target_clone_t record;
};
/**
 * Return the list of targets to clone
 */
std::vector<jl_target_spec_t> jl_get_llvm_clone_targets(uint32_t llvmver);
#endif
