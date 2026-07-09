/*
 * SPDX-FileCopyrightText: Copyright 2024 Arm Limited and/or its affiliates <open-source-office@arm.com>
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
 * Title:        arm_elementwise_mul_s16_batch_offset
 * Description:  Element wise multiplication
 *
 * $Date:        18 March 2024
 * $Revision:    V.1.0.0
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
 * @brief s16 element wise multiplication of batches of two vectors
 *
 * @note   Refer header file for details.
 *
 */
arm_cmsis_nn_status arm_elementwise_mul_s16_batch_offset(const int16_t *input_1_vect,
                                                         const int16_t *input_2_vect,
                                                         int16_t *output,
                                                         const int32_t out_offset,
                                                         const int32_t out_mult,
                                                         const int32_t out_shift,
                                                         const int32_t block_size,
                                                         const int32_t batch_size,
                                                         const int32_t batch_offset)
{

    int32_t loop_count;

    for (int i = 0; i < batch_size; i++)
    {

#if defined(ARM_MATH_MVEI)

        const int16_t *input_1_ptr = input_1_vect;
        const int16_t *input_2_ptr = input_2_vect;
        int16_t *output_ptr = output;

        loop_count = block_size;

        while (loop_count > 0)
        {
            mve_pred16_t pred = vctp32q(loop_count);

            int32x4_t input_1 = vldrhq_z_s32(input_1_ptr, pred);
            int32x4_t input_2 = vldrhq_z_s32(input_2_ptr, pred);

            int32x4_t res_0 = vmulq_s32(input_1, input_2);

            res_0 = arm_requantize_mve_32x4(res_0, vdupq_n_s32(out_mult), vdupq_n_s32(out_shift));
            res_0 = vaddq_n_s32(res_0, out_offset);

            res_0 = vmaxq_s32(res_0, vdupq_n_s32(NN_Q15_MIN));
            res_0 = vminq_s32(res_0, vdupq_n_s32(NN_Q15_MAX));

            vstrhq_p_s32(output_ptr, res_0, pred);
            input_1_ptr += 4;
            input_2_ptr += 4;

            output_ptr += 4;
            loop_count -= 4;
        }

        input_1_vect += block_size;
        input_2_vect += block_size;
        output += block_size;

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
            mul_res = arm_nn_requantize(mul_res, out_mult, out_shift) + out_offset;
            mul_res = MAX(mul_res, NN_Q15_MIN);
            mul_res = MIN(mul_res, NN_Q15_MAX);
            mul_1 = (int16_t)mul_res;

    #if defined(ARM_MATH_DSP)
            mul_res = SMULTT(two_halfword_1, two_halfword_2);
    #else
            input_1 = (int16_t)(two_halfword_1 >> 16);
            input_2 = (int16_t)(two_halfword_2 >> 16);
            mul_res = input_1 * input_2;
    #endif
            mul_res = arm_nn_requantize(mul_res, out_mult, out_shift) + out_offset;
            mul_res = MAX(mul_res, NN_Q15_MIN);
            mul_res = MIN(mul_res, NN_Q15_MAX);
            mul_2 = (int16_t)mul_res;

            arm_nn_write_q15x2_ia(&output, PACK_Q15x2_32x1(mul_1, mul_2));

            loop_count--;
        }

        if (block_size & 0x1)
        {
            /* C = A * B */

            input_1 = *input_1_vect++;
            input_2 = *input_2_vect++;

            mul_res = input_1 * input_2;
            mul_res = arm_nn_requantize(mul_res, out_mult, out_shift) + out_offset;

            mul_res = MAX(mul_res, NN_Q15_MIN);
            mul_res = MIN(mul_res, NN_Q15_MAX);

            *output++ = (int16_t)mul_res;
        }
#endif // #if defined(ARM_MATH_MVEI)

        output += (batch_offset - 1) * block_size;
    }
    return ARM_CMSIS_NN_SUCCESS;
}

/**
 * @} end of Doxygen group
 */
