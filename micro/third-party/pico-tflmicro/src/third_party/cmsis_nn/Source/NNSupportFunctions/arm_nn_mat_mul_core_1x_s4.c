/*
 * SPDX-FileCopyrightText: Copyright 2010-2024 Arm Limited and/or its affiliates <open-source-office@arm.com>
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
 * Title:        arm_nn_mat_mul_core_1x_s4.c
 * Description:  General Matrix-multiplication function
 *
 * $Date:        10 April 2024
 * $Revision:    V.1.0.0
 *
 * Target :  Arm(R) M-Profile Architecture
 * -------------------------------------------------------------------- */

#include "third_party/cmsis_nn/Include/arm_nnsupportfunctions.h"

/**
 * @ingroup groupSupport
 */

/**
 * @addtogroup supportConvolution
 * @{
 */

/*
 * s4 matrix multiplication to process 1 row
 *
 * Refer header file for details.
 *
 */
arm_cmsis_nn_status arm_nn_mat_mul_core_1x_s4(int32_t row_elements,
                                              const int32_t skipped_row_elements,
                                              const int8_t *row_base_ref,
                                              const int8_t *col_base_ref,
                                              const int32_t out_ch,
                                              const cmsis_nn_conv_params *conv_params,
                                              const cmsis_nn_per_channel_quant_params *quant_params,
                                              const int32_t *bias,
                                              int8_t *output)
{
#if defined(ARM_MATH_MVEI)
    const int8_t *col_base = col_base_ref;
    int32_t *output_mult = quant_params->multiplier;
    int32_t *output_shift = quant_params->shift;
    const int32_t out_offset = conv_params->output_offset;
    const int32_t out_activation_min = conv_params->activation.min;
    const int32_t out_activation_max = conv_params->activation.max;

    const uint8x16_t gather_offset = {0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7};
    const mve_pred16_t lower_nibble_mask = 21845; // 0101010101010101

    int32_t acc[4];
    for (int i = 0; i < out_ch; i++)
    {
        int32_t acc_n0 = 0;
        const int8_t *row_base = row_base_ref;

        int32_t sum_tmp = 0;
        for (int j = row_elements; j > 0; j -= 16)
        {
            mve_pred16_t rmdr_mask = vctp8q((uint32_t)j);
            int8x16_t col_vec = vldrbq_gather_offset_z_s8(col_base, gather_offset, rmdr_mask);
            col_base += 8;
            col_vec = vrshlq_m_n_s8(col_vec, 4, (lower_nibble_mask & rmdr_mask));
            col_vec = vshrq_n_s8(col_vec, 4);

            sum_tmp = vaddvaq_p_s8(sum_tmp, col_vec, rmdr_mask);

            int8x16_t lhs_vec = vldrbq_z_s8(row_base, rmdr_mask);
            row_base += 16;

            acc_n0 = vmladavaq_p_s8(acc_n0, col_vec, lhs_vec, rmdr_mask);
        }

        sum_tmp *= conv_params->input_offset;
        acc_n0 += sum_tmp;

        const int32_t index = i & 0x3;
        acc[index] = acc_n0;

        if (index == 3)
        {
            int32x4_t res = vldrwq_s32(acc);
            if (bias)
            {
                res = vaddq_s32(res, vldrwq_s32(bias));
                bias += 4;
            }
            res = arm_requantize_mve_32x4(res, vldrwq_s32(output_mult), vldrwq_s32(output_shift));
            output_mult += 4;
            output_shift += 4;
            res = vaddq_n_s32(res, out_offset);
            res = vmaxq_s32(res, vdupq_n_s32(out_activation_min));
            res = vminq_s32(res, vdupq_n_s32(out_activation_max));
            vstrbq_s32(output, res);
            output += 4;
        }
        int val = (i + 1) * ((row_elements + skipped_row_elements));
        col_base = col_base_ref + (val >> 1);
    }
    // Handle left over elements
    for (int i = 0; i < (out_ch & 0x3); i++)
    {
        int32_t acc_n0 = acc[i];
        if (bias)
        {
            acc_n0 += bias[i];
        }
        acc_n0 = arm_nn_requantize(acc_n0, output_mult[i], output_shift[i]);
        acc_n0 += conv_params->output_offset;
        acc_n0 = MAX(acc_n0, conv_params->activation.min);
        acc_n0 = MIN(acc_n0, conv_params->activation.max);
        *output++ = (int8_t)acc_n0;
    }
    return ARM_CMSIS_NN_SUCCESS;

#else
    (void)row_elements;
    (void)skipped_row_elements;
    (void)row_base_ref;
    (void)col_base_ref;
    (void)out_ch;
    (void)conv_params;
    (void)quant_params;
    (void)bias;
    (void)output;
    return ARM_CMSIS_NN_NO_IMPL_ERROR;
#endif
}

/**
 * @} end of supportConvolution group
 */
