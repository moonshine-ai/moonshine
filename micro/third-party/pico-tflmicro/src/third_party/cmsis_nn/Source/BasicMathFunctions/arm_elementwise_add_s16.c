/*
 * SPDX-FileCopyrightText: Copyright 2022 Arm Limited and/or its affiliates <open-source-office@arm.com>
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
 * Title:        arm_elementwise_add_s16
 * Description:  Elementwise add
 *
 * $Date:        24 Oct 2022
 * $Revision:    V.2.2.0
 *
 * Target Processor:  Cortex-M CPUs
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

/*
 * s16 elementwise add
 *
 * Refer header file for details.
 *
 */

/* Note: __SHIFT is expected to be <=0 */

arm_cmsis_nn_status arm_elementwise_add_s16(const int16_t *input_1_vect,
                                            const int16_t *input_2_vect,
                                            const int32_t input_1_offset,
                                            const int32_t input_1_mult,
                                            const int32_t input_1_shift,
                                            const int32_t input_2_offset,
                                            const int32_t input_2_mult,
                                            const int32_t input_2_shift,
                                            const int32_t left_shift,
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

#if defined(ARM_MATH_MVEI)

    int32_t count = block_size;

    while (count > 0)
    {

        mve_pred16_t pred = vctp32q(count);

        int32x4_t vect_1 = vldrhq_z_s32(input_1_vect, pred);
        int32x4_t vect_2 = vldrhq_z_s32(input_2_vect, pred);

        vect_1 = vshlq_r_s32(vect_1, left_shift);
        vect_2 = vshlq_r_s32(vect_2, left_shift);

        vect_1 = arm_requantize_mve(vect_1, input_1_mult, input_1_shift);
        vect_2 = arm_requantize_mve(vect_2, input_2_mult, input_2_shift);

        vect_1 = vaddq_s32(vect_1, vect_2);
        vect_1 = arm_requantize_mve(vect_1, out_mult, out_shift);

        vect_1 = vmaxq_s32(vect_1, vdupq_n_s32(out_activation_min));
        vect_1 = vminq_s32(vect_1, vdupq_n_s32(out_activation_max));

        input_1_vect += 4;
        input_2_vect += 4;

        vstrhq_p_s32(output, vect_1, pred);

        output += 4;
        count -= 4;
    }

#else  // #if defined(ARM_MATH_MVEI)
    int32_t input_1;
    int32_t input_2;
    int32_t sum;
    int32_t two_halfword_1, two_halfword_2;
    int16_t sum_1, sum_2;
    int32_t loop_count = block_size / 2;
    while (loop_count > 0)
    {
        two_halfword_1 = arm_nn_read_q15x2_ia(&input_1_vect);
        two_halfword_2 = arm_nn_read_q15x2_ia(&input_2_vect);

        input_1 = (int16_t)(two_halfword_1 & 0xFFFF) << left_shift;
        input_1 = arm_nn_requantize(input_1, input_1_mult, input_1_shift);
        input_2 = (int16_t)(two_halfword_2 & 0xFFFF) << left_shift;
        input_2 = arm_nn_requantize(input_2, input_2_mult, input_2_shift);
        sum = input_1 + input_2;
        sum = arm_nn_requantize(sum, out_mult, out_shift);
        sum = MAX(sum, out_activation_min);
        sum = MIN(sum, out_activation_max);
        sum_1 = (int16_t)sum;

        input_1 = (int16_t)(two_halfword_1 >> 16) << left_shift;
        input_1 = arm_nn_requantize(input_1, input_1_mult, input_1_shift);
        input_2 = (int16_t)(two_halfword_2 >> 16) << left_shift;
        input_2 = arm_nn_requantize(input_2, input_2_mult, input_2_shift);
        sum = input_1 + input_2;
        sum = arm_nn_requantize(sum, out_mult, out_shift);
        sum = MAX(sum, out_activation_min);
        sum = MIN(sum, out_activation_max);
        sum_2 = (int16_t)sum;

        arm_nn_write_q15x2_ia(&output, PACK_Q15x2_32x1(sum_1, sum_2));

        loop_count--;
    }
    loop_count = block_size & 0x1;

    while (loop_count > 0)
    {
        /* C = A + B */
        input_1 = *input_1_vect++ << left_shift;
        input_2 = *input_2_vect++ << left_shift;

        input_1 = arm_nn_requantize(input_1, input_1_mult, input_1_shift);
        input_2 = arm_nn_requantize(input_2, input_2_mult, input_2_shift);

        sum = input_1 + input_2;
        sum = arm_nn_requantize(sum, out_mult, out_shift);

        sum = MAX(sum, out_activation_min);
        sum = MIN(sum, out_activation_max);

        *output++ = (int16_t)sum;

        /* Decrement loop counter */
        loop_count--;
    }
#endif // #if defined(ARM_MATH_MVEI)
    return (ARM_CMSIS_NN_SUCCESS);
}

/**
 * @} end of Doxygen group
 */