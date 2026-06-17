/*
 * SPDX-FileCopyrightText: Copyright 2010-2023 Arm Limited and/or its affiliates <open-source-office@arm.com>
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
 * Title:        arm_nn_mat_mul_core_1x_s8.c
 * Description:  General Matrix-multiplication function
 *
 * $Date:        20 January 2023
 * $Revision:    V.3.1.3
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
 * s8 matrix multiplication to process 1 row
 *
 * Refer header file for details.
 *
 */
arm_cmsis_nn_status arm_nn_mat_mul_core_1x_s8(int32_t row_elements,
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

    int32_t acc[4];
    for (int i = 0; i < out_ch; i++)
    {
        int32_t acc_n0 = 0;
        const int8_t *row_base = row_base_ref;

        int32_t sum_tmp = 0;

    #if defined(ARM_MATH_AUTOVECTORIZE)
        for (int j = 0; j < row_elements; j++)
        {
            int32_t col = col_base[j];
            sum_tmp += col;
            acc_n0 += row_base[j] * col;
        }
    #else
        __ASM volatile(" .p2align 2                             \n"
                       "  vldrb.8         q0, [%[col]], #16     \n"
                       "  wlstp.8         lr, %[cnt], 1f       \n"
                       "2:                                      \n"
                       "  vaddva.s8      %[sum], q0            \n"
                       "  vldrb.8         q1, [%[row0]], #16   \n"
                       "  vmladava.s8    %[out0], q0, q1       \n"
                       "  vldrb.8         q0, [%[col]], #16    \n"
                       "  letp            lr, 2b               \n"
                       "1:                                      \n"
                       : [col] "+r"(col_base), [sum] "+Te"(sum_tmp), [row0] "+r"(row_base), [out0] "+Te"(acc_n0)
                       : [cnt] "r"(row_elements)
                       : "q0", "q1", "memory", "r14");
    #endif

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
        col_base = col_base_ref + (i + 1) * (row_elements + skipped_row_elements);
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
