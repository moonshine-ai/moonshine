/*
 * SPDX-FileCopyrightText: Copyright 2023-2024 Arm Limited and/or its affiliates <open-source-office@arm.com>
 *
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the License); you may
 * not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an AS IS BASIS, WITHOUT
 * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

/* ----------------------------------------------------------------------
 * Project:      CMSIS NN Library
 * Title:        arm_nn_compiler.h
 * Description:  Generic compiler header
 *
 * $Date:        20 June 2024
 * $Revision:    V.1.3.0
 *
 * Target :  Arm(R) M-Profile Architecture
 * -------------------------------------------------------------------- */

#ifndef ARM_NN_COMPILER_H
#define ARM_NN_COMPILER_H

/**
 *
 * @brief Arm C-Language Extension(ACLE) Includes
 *
 */

#if defined(__ARMCC_VERSION) && (__ARMCC_VERSION >= 6010050)

    #ifndef __ASM
        #define __ASM __asm
    #endif
    #ifndef __INLINE
        #define __INLINE __inline
    #endif
    #ifndef __STATIC_INLINE
        #define __STATIC_INLINE static __inline
    #endif
    #ifndef __STATIC_FORCEINLINE
        #define __STATIC_FORCEINLINE __attribute__((always_inline)) static __inline
    #endif
    #ifndef __RESTRICT
        #define __RESTRICT __restrict
    #endif

#elif defined(__ICCARM__)

    #warning IAR support is not tested
    #ifndef __ASM
        #define __ASM __asm
    #endif
    #ifndef __INLINE
        #define __INLINE inline
    #endif
    #ifndef __STATIC_INLINE
        #define __STATIC_INLINE static inline
    #endif
    #ifndef __FORCEINLINE
        #define __FORCEINLINE _Pragma("inline=forced")
    #endif
    #ifndef __STATIC_FORCEINLINE
        #define __STATIC_FORCEINLINE __FORCEINLINE __STATIC_INLINE
    #endif
    #ifndef __RESTRICT
        #define __RESTRICT __restrict
    #endif

#elif defined(_MSC_VER)

    // Build for non Arm Cortex-M processors is not tested or supported.
    // Use this section to stub any macros or intrinsics
    #warning Unsupported compiler
    #ifndef __STATIC_FORCEINLINE
        #define __STATIC_FORCEINLINE static __forceinline
    #endif
    #ifndef __STATIC_INLINE
        #define __STATIC_INLINE static __inline
    #endif
    #ifndef __ALIGNED
        #define __ALIGNED(x) __declspec(align(x))
    #endif

#elif defined(__GNUC__)

    #ifndef __ASM
        #define __ASM __asm
    #endif
    #ifndef __INLINE
        #define __INLINE inline
    #endif
    #ifndef __STATIC_INLINE
        #define __STATIC_INLINE static inline
    #endif
    #ifndef __STATIC_FORCEINLINE
        #define __STATIC_FORCEINLINE __attribute__((always_inline)) static inline
    #endif
    #ifndef __RESTRICT
        #define __RESTRICT __restrict
    #endif

#else

    #error Unsupported compiler. Add support as needed

#endif

/**
 *
 * @brief Compiler specific diagnostic adjustment / fixes if applicable
 *
 */

// Note: __ARM_ARCH is used with M-profile architecture as the target here.
#if defined(__GNUC__)
    #if (__GNUC__ == 12 && (__GNUC_MINOR__ <= 2)) && defined(__ARM_ARCH)
        // Workaround for 'Internal Compiler Error' on Arm GNU Toolchain rel 12.2.x
        // https://gcc.gnu.org/pipermail/gcc-patches/2022-December/607963.html
        #define ARM_GCC_12_2_ICE
    #endif
#endif

#if defined(__ARM_FEATURE_MVE) && ((__ARM_FEATURE_MVE & 3) == 3) || (__ARM_FEATURE_MVE & 1)
    #include <arm_mve.h>
#endif

#if defined(__ARM_ARCH) || defined(__ARM_ACLE)
    #include <arm_acle.h>
#endif

#if defined(__GNUC__)
    #include <stdint.h>
#endif

/**
 *
 * @brief ACLE and Intrinsics
 *
 */

// Note: Have __GNUC__, that is used to check for GCC , checks at the end
// as __GNUC__ is defined by non-GCC compilers as well

/* Common intrinsics for all architectures */
#if defined(__ARMCC_VERSION) && (__ARMCC_VERSION >= 6010050) || defined(__ICCARM__)
    #define CLZ __clz
#elif defined(__GNUC__)
/**
  \brief   Count leading zeros
  \details Counts the number of leading zeros of a data value.
  \param [in]  value  Value to count the leading zeros
  \return             number of leading zeros in value
 */
__STATIC_FORCEINLINE uint8_t CLZ(uint32_t value)
{
    /* Even though __builtin_clz produces a CLZ instruction on ARM, formally
       __builtin_clz(0) is undefined behaviour, so handle this case specially.
       This guarantees Arm-compatible results if compiling on a non-Arm
       target, and ensures the compiler doesn't decide to activate any
       optimisations using the logic "value was passed to __builtin_clz, so it
       is non-zero".
       ARM GCC 7.3 and possibly earlier will optimise this test away, leaving a
       single CLZ instruction.
     */
    if (value == 0U)
    {
        return 32U;
    }
    return __builtin_clz(value);
}
#endif

// ACLE intrinsics under groups __ARM_FEATURE_QBIT, __ARM_FEATURE_DSP , __ARM_FEATURE_SAT, __ARM_FEATURE_SIMD32

// Note: Just __ARM_FEATURE_DSP is checked to collect all intrinsics from the above mentioned groups

#if (defined(__ARM_FEATURE_DSP) && (__ARM_FEATURE_DSP == 1))

    // Common intrinsics
    #define SMLABB __smlabb
    #define SMLATT __smlatt
    #define SMLALD __smlald
    #define QADD __qadd
    #define QSUB8 __qsub8
    #define QSUB16 __qsub16
    #define SADD16 __sadd16

    // Compiler specifc variants of intrinsics. Create a new section or file for IAR if needed
    #if defined(__ARMCC_VERSION) && (__ARMCC_VERSION >= 6010050) || defined(__ICCARM__)

        #define SMULBB __smulbb
        #define SMULTT __smultt
        #define ROR __ror
        #define SXTB16 __sxtb16
        #define SXTAB16 __sxtab16
        #define SXTB16_RORn(ARG1, ARG2) SXTB16(ROR(ARG1, ARG2))
        #define SXTAB16_RORn(ARG1, ARG2, ARG3) SXTAB16(ARG1, ROR(ARG2, ARG3))
        #define SMLAD __smlad
        // PKH<XY> translates into pkh<xy> on AC6
        #define PKHBT(ARG1, ARG2, ARG3)                                                                                \
            (((((uint32_t)(ARG1))) & 0x0000FFFFUL) | ((((uint32_t)(ARG2)) << (ARG3)) & 0xFFFF0000UL))
        #define PKHTB(ARG1, ARG2, ARG3)                                                                                \
            (((((uint32_t)(ARG1))) & 0xFFFF0000UL) | ((((uint32_t)(ARG2)) >> (ARG3)) & 0x0000FFFFUL))

    #elif defined(__GNUC__)

        #define PKHBT(ARG1, ARG2, ARG3)                                                                                \
            __extension__({                                                                                            \
                uint32_t __RES, __ARG1 = (ARG1), __ARG2 = (ARG2);                                                      \
                __ASM("pkhbt %0, %1, %2, lsl %3" : "=r"(__RES) : "r"(__ARG1), "r"(__ARG2), "I"(ARG3));                 \
                __RES;                                                                                                 \
            })
        #define PKHTB(ARG1, ARG2, ARG3)                                                                                \
            __extension__({                                                                                            \
                uint32_t __RES, __ARG1 = (ARG1), __ARG2 = (ARG2);                                                      \
                if (ARG3 == 0)                                                                                         \
                    __ASM("pkhtb %0, %1, %2" : "=r"(__RES) : "r"(__ARG1), "r"(__ARG2));                                \
                else                                                                                                   \
                    __ASM("pkhtb %0, %1, %2, asr %3" : "=r"(__RES) : "r"(__ARG1), "r"(__ARG2), "I"(ARG3));             \
                __RES;                                                                                                 \
            })

__STATIC_FORCEINLINE uint32_t SXTAB16(uint32_t op1, uint32_t op2)
{
    uint32_t result;

    __ASM("sxtab16 %0, %1, %2" : "=r"(result) : "r"(op1), "r"(op2));
    return (result);
}

__STATIC_FORCEINLINE uint32_t SXTB16(uint32_t op1)
{
    uint32_t result;

    __ASM("sxtb16 %0, %1" : "=r"(result) : "r"(op1));
    return (result);
}

// __smlad is defined by GCC, but results in a performance drop(Tested on Arm GNU Toolchain version 11.x and 12.x)
__STATIC_FORCEINLINE uint32_t SMLAD(uint32_t op1, uint32_t op2, uint32_t op3)
{
    uint32_t result;

    __ASM volatile("smlad %0, %1, %2, %3" : "=r"(result) : "r"(op1), "r"(op2), "r"(op3));
    return (result);
}

__STATIC_FORCEINLINE uint32_t ROR(uint32_t op1, uint32_t op2)
{
    op2 %= 32U;
    if (op2 == 0U)
    {
        return op1;
    }
    return (op1 >> op2) | (op1 << (32U - op2));
}

__STATIC_FORCEINLINE uint32_t SXTB16_RORn(uint32_t op1, uint32_t rotate)
{
    uint32_t result;
    if (__builtin_constant_p(rotate) && ((rotate == 8U) || (rotate == 16U) || (rotate == 24U)))
    {
        __ASM volatile("sxtb16 %0, %1, ROR %2" : "=r"(result) : "r"(op1), "i"(rotate));
    }
    else
    {
        result = SXTB16(ROR(op1, rotate));
    }
    return result;
}

__STATIC_FORCEINLINE uint32_t SXTAB16_RORn(uint32_t op1, uint32_t op2, uint32_t rotate)
{
    uint32_t result;
    if (__builtin_constant_p(rotate) && ((rotate == 8U) || (rotate == 16U) || (rotate == 24U)))
    {
        __ASM volatile("sxtab16 %0, %1, %2, ROR %3" : "=r"(result) : "r"(op1), "r"(op2), "i"(rotate));
    }
    else
    {
        result = SXTAB16(op1, ROR(op2, rotate));
    }
    return result;
}

// Inline assembly routines for ACLE intrinsics that are not defined by GCC toolchain
__STATIC_FORCEINLINE uint32_t SMULBB(uint32_t op1, uint32_t op2)
{
    uint32_t result;

    __ASM volatile("smulbb %0, %1, %2" : "=r"(result) : "r"(op1), "r"(op2));
    return (result);
}

__STATIC_FORCEINLINE uint32_t SMULTT(uint32_t op1, uint32_t op2)
{
    uint32_t result;

    __ASM volatile("smultt %0, %1, %2" : "=r"(result) : "r"(op1), "r"(op2));
    return (result);
}
    #endif

#endif

#endif /* #ifndef ARM_NN_COMPILER_H */
