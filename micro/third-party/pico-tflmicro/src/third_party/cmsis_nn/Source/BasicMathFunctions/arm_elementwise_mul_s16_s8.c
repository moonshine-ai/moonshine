/*
 * SPDX-FileCopyrightText: Copyright 2022-2024 Arm Limited and/or its affiliates <open-source-office@arm.com>
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
 * Title:        arm_elementwise_mul_s16_s8.c
 * Description:  Elementwise multiplication of 16 bit input with 8 bit output
 *
 * $Date:        20 January 2023
 * $Revision:    V.2.0.0
 *
 * Target :  Arm(R) M-Profile Architecture
 *
 * -------------------------------------------------------------------- */

#include "third_party/cmsis_nn/Include/arm_nnsupportfunctions.h"

/**
 *  @ingroup groupSupport
 */

/**
 * @addtogroup BasicMath
 * @{
 */

/*
 * s16 elementwise multiplication with s8 output
 *
 * Refer header file for details.
 *
 */
arm_cmsis_nn_status arm_elementwise_mul_s16_s8(const int16_t *input_1_vect,
                                               const int16_t *input_2_vect,
                                               int8_t *output,
                                               const int32_t out_offset,
                                               const int32_t out_mult,
                                               const int32_t out_shift,
                                               const int32_t block_size,
                                               const int32_t batch_size,
                                               const int32_t batch_offset)
{

    for (int i = 0; i < batch_size; i++)
    {
        int32_t loop_count = block_size;
#if defined(ARM_MATH_MVEI)

        const int16_t *input_1_ptr = input_1_vect;
        const int16_t *input_2_ptr = input_2_vect;
        int8_t *output_ptr = output;

        while (loop_count > 0)
        {
            mve_pred16_t pred = vctp32q(loop_count);

            int32x4_t input_1 = vldrhq_z_s32(input_1_ptr, pred);
            int32x4_t input_2 = vldrhq_z_s32(input_2_ptr, pred);

            int32x4_t res_0 = vmulq_s32(input_1, input_2);

            res_0 = arm_requantize_mve_32x4(res_0, vdupq_n_s32(out_mult), vdupq_n_s32(out_shift));
            res_0 = vaddq_n_s32(res_0, out_offset);

            res_0 = vmaxq_s32(res_0, vdupq_n_s32(NN_Q7_MIN));
            res_0 = vminq_s32(res_0, vdupq_n_s32(NN_Q7_MAX));

            vstrbq_p_s32(output_ptr, res_0, pred);
            input_1_ptr += 4;
            input_2_ptr += 4;

            output_ptr += 4;
            loop_count -= 4;
        }

        input_1_vect += block_size;
        input_2_vect += block_size;
        output += block_size;

#else
    #if defined(ARM_MATH_DSP)

        while (loop_count > 1)
        {
            int32_t input_1 = arm_nn_read_q15x2_ia(&input_1_vect);
            int32_t input_2 = arm_nn_read_q15x2_ia(&input_2_vect);

            int32_t mul_res = SMULBB(input_1, input_2);
            mul_res = arm_nn_requantize(mul_res, out_mult, out_shift) + out_offset;
            mul_res = CLAMP(mul_res, NN_Q7_MAX, NN_Q7_MIN);
            int32_t mul = (int16_t)(mul_res & 0xFF);

            mul_res = SMULTT(input_1, input_2);
            mul_res = arm_nn_requantize(mul_res, out_mult, out_shift) + out_offset;
            mul_res = CLAMP(mul_res, NN_Q7_MAX, NN_Q7_MIN);
            mul |= (int16_t)mul_res << 8;

            arm_nn_write_s8x2_ia(&output, mul);
            loop_count -= 2;
        }
    #endif
        for (int j = 0; j < loop_count; j++, input_1_vect++, input_2_vect++, output++)
        {
            /* C = A * B */
            int32_t mul_res = (*input_1_vect) * (*input_2_vect);
            mul_res = arm_nn_requantize(mul_res, out_mult, out_shift) + out_offset;

            mul_res = CLAMP(mul_res, NN_Q7_MAX, NN_Q7_MIN);

            *output = (int8_t)mul_res;
        }

#endif

        output += (batch_offset - 1) * block_size;
    }
    return ARM_CMSIS_NN_SUCCESS;
}
/**
 * @} end of BasicMath group
 */
