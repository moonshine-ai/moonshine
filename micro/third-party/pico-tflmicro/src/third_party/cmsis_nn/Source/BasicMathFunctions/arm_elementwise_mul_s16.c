/*
 * SPDX-FileCopyrightText: Copyright 2022-2023 Arm Limited and/or its affiliates <open-source-office@arm.com>
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
 * Title:        arm_elementwise_mul_s16
 * Description:  Element wise multiplication
 *
 * $Date:        20 January 2023
 * $Revision:    V.2.4.0
 *
 * Target :  Arm(R) M-Profile Architecture
 *
 * -------------------------------------------------------------------- */

#include "third_party/cmsis_nn/Include/arm_nnfunctions.h"
#include "third_party/cmsis_nn/Include/arm_nnsupportfunctions.h"

/**
 *  @ingroup Public
 */

/**
 * @addtogroup groupElementwise
 * @{
 */

/**
 * @brief s16 element wise multiplication of two vectors
 *
 * @note   Refer header file for details.
 *
 */
arm_cmsis_nn_status arm_elementwise_mul_s16(const int16_t *input_1_vect,
                                            const int16_t *input_2_vect,
                                            const int32_t input_1_offset,
                                            const int32_t input_2_offset,
                                            int16_t *output,
                                            const int32_t out_offset,
                                            const int32_t out_mult,
                                            const int32_t out_shift,
                                            const int32_t out_activation_min,
                                            const int32_t out_activation_max,
                                            const int32_t block_size)
{
    (void)input_1_offset;
    (void)input_2_offset;
    (void)out_offset;
    int32_t loop_count;

#if defined(ARM_MATH_MVEI)

    loop_count = block_size;

    while (loop_count > 0)
    {
        mve_pred16_t pred = vctp32q(loop_count);

        int32x4_t input_1 = vldrhq_z_s32(input_1_vect, pred);
        int32x4_t input_2 = vldrhq_z_s32(input_2_vect, pred);

        int32x4_t res_0 = vmulq_s32(input_1, input_2);

        res_0 = arm_requantize_mve_32x4(res_0, vdupq_n_s32(out_mult), vdupq_n_s32(out_shift));

        res_0 = vmaxq_s32(res_0, vdupq_n_s32(out_activation_min));
        res_0 = vminq_s32(res_0, vdupq_n_s32(out_activation_max));

        vstrhq_p_s32(output, res_0, pred);
        input_1_vect += 4;
        input_2_vect += 4;

        output += 4;
        loop_count -= 4;
    }

#else
    int32_t input_1;
    int32_t input_2;
    int32_t mul_res;
    int32_t two_halfword_1, two_halfword_2;
    int16_t mul_1, mul_2;
    loop_count = block_size / 2;

    while (loop_count > 0)
    {
        two_halfword_1 = arm_nn_read_q15x2_ia(&input_1_vect);
        two_halfword_2 = arm_nn_read_q15x2_ia(&input_2_vect);

    #if defined(ARM_MATH_DSP)
        mul_res = SMULBB(two_halfword_1, two_halfword_2);
    #else
        input_1 = (int16_t)(two_halfword_1 & 0xFFFF);
        input_2 = (int16_t)(two_halfword_2 & 0xFFFF);
        mul_res = input_1 * input_2;
    #endif
        mul_res = arm_nn_requantize(mul_res, out_mult, out_shift);
        mul_res = MAX(mul_res, out_activation_min);
        mul_res = MIN(mul_res, out_activation_max);
        mul_1 = (int16_t)mul_res;

    #if defined(ARM_MATH_DSP)
        mul_res = SMULTT(two_halfword_1, two_halfword_2);
    #else
        input_1 = (int16_t)(two_halfword_1 >> 16);
        input_2 = (int16_t)(two_halfword_2 >> 16);
        mul_res = input_1 * input_2;
    #endif
        mul_res = arm_nn_requantize(mul_res, out_mult, out_shift);
        mul_res = MAX(mul_res, out_activation_min);
        mul_res = MIN(mul_res, out_activation_max);
        mul_2 = (int16_t)mul_res;

        arm_nn_write_q15x2_ia(&output, PACK_Q15x2_32x1(mul_1, mul_2));

        loop_count--;
    }
    loop_count = block_size & 0x1;

    while (loop_count > 0)
    {
        /* C = A * B */

        input_1 = *input_1_vect++;
        input_2 = *input_2_vect++;

        mul_res = input_1 * input_2;
        mul_res = arm_nn_requantize(mul_res, out_mult, out_shift);

        mul_res = MAX(mul_res, out_activation_min);
        mul_res = MIN(mul_res, out_activation_max);

        *output++ = (int16_t)mul_res;

        /* Decrement loop counter */
        loop_count--;
    }
#endif // #if defined(ARM_MATH_MVEI)
    return ARM_CMSIS_NN_SUCCESS;
}

/**
 * @} end of Doxygen group
 */
