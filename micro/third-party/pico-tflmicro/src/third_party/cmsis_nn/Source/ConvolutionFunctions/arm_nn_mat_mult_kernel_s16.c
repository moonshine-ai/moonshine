/*
 * SPDX-FileCopyrightText: Copyright 2010-2024 Arm Limited and/or its affiliates
 * <open-source-office@arm.com>
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
 * Title:        arm_nn_mat_mult_kernel_s16.c
 * Description:  Matrix-multiplication function for 16 bits convolution
 *
 * $Date:        12 April 2024
 * $Revision:    V.3.0.0
 *
 * Target :  Arm(R) M-Profile Architecture
 * -------------------------------------------------------------------- */

#include "third_party/cmsis_nn/Include/arm_nnfunctions.h"
#include "third_party/cmsis_nn/Include/arm_nnsupportfunctions.h"

/**
 * @ingroup groupSupport
 */

/**
 * @addtogroup supportConvolution
 * @{
 */

/*
 * Matrix-multiplication function for convolution with per-channel requantization.
 *
 * Refer header file for details.
 *
 */
int16_t *arm_nn_mat_mult_kernel_s16(const int8_t *input_a,
                                    const int16_t *input_b,
                                    const int32_t output_ch,
                                    const int32_t *out_shift,
                                    const int32_t *out_mult,
                                    const int32_t activation_min,
                                    const int32_t activation_max,
                                    const int32_t num_col_a,
                                    const cmsis_nn_bias_data *const bias_data,
                                    int16_t *out_0)
{
#if !defined(ARM_MATH_MVEI)
    const int64_t *bias_s64 = (const int64_t *)bias_data->data;
    const int32_t *bias_s32 = (const int32_t *)bias_data->data;
    const bool is_int32_bias = bias_data->is_int32_bias;

    const int32_t num_col_a_fast = is_int32_bias ? num_col_a : (num_col_a > MAX_COL_COUNT ? MAX_COL_COUNT : num_col_a);
    const int32_t num_col_a_slow = num_col_a - MAX_COL_COUNT;

    int16_t *out_1 = out_0 + output_ch;
    int32_t row_count = output_ch / 2;
    const int8_t *ip_a0 = input_a;

    /* This loop over rows in A */
    while (row_count)
    {
        /* Setup pointers for B */
        const int16_t *ip_b0 = input_b;
        const int16_t *ip_b1 = ip_b0 + num_col_a;

        /* Align the second pointer for A */
        const int8_t *ip_a1 = ip_a0 + num_col_a;

        /* Init accumulator for channel N and N + 1 */
        int32_t ch_0_out_0 = 0;
        int32_t ch_0_out_1 = 0;
        int32_t ch_1_out_0 = 0;
        int32_t ch_1_out_1 = 0;

    #if defined(ARM_MATH_DSP)
        uint16_t col_count = num_col_a_fast / 4;

        /* Accumulate over the vector */
        while (col_count)
        {
            int32_t a01, a02, a11, a12;
            int32_t b0 = arm_nn_read_q15x2_ia(&ip_b0);
            int32_t b1 = arm_nn_read_q15x2_ia(&ip_b1);

            ip_a0 = read_and_pad(ip_a0, &a01, &a02);
            ip_a1 = read_and_pad(ip_a1, &a11, &a12);

            ch_0_out_0 = SMLAD(a01, b0, ch_0_out_0);
            ch_0_out_1 = SMLAD(a01, b1, ch_0_out_1);
            ch_1_out_0 = SMLAD(a11, b0, ch_1_out_0);
            ch_1_out_1 = SMLAD(a11, b1, ch_1_out_1);

            b0 = arm_nn_read_q15x2_ia(&ip_b0);
            b1 = arm_nn_read_q15x2_ia(&ip_b1);

            ch_0_out_0 = SMLAD(a02, b0, ch_0_out_0);
            ch_0_out_1 = SMLAD(a02, b1, ch_0_out_1);
            ch_1_out_0 = SMLAD(a12, b0, ch_1_out_0);
            ch_1_out_1 = SMLAD(a12, b1, ch_1_out_1);

            col_count--;
        }
        col_count = num_col_a_fast & 0x3;
    #else
        int32_t col_count = num_col_a_fast;
    #endif

        while (col_count)
        {
            int8_t a0 = *ip_a0++;
            int16_t b0 = *ip_b0++;
            int8_t a1 = *ip_a1++;
            int16_t b1 = *ip_b1++;

            ch_0_out_0 += a0 * b0;
            ch_0_out_1 += a0 * b1;
            ch_1_out_0 += a1 * b0;
            ch_1_out_1 += a1 * b1;
            col_count--;
        }

        if (is_int32_bias)
        {
            if (bias_s32)
            {
                ch_0_out_0 += *bias_s32;
                ch_0_out_1 += *bias_s32++;
                ch_1_out_0 += *bias_s32;
                ch_1_out_1 += *bias_s32++;
            }

            ch_0_out_0 = arm_nn_requantize(ch_0_out_0, *out_mult, *out_shift);
            ch_0_out_1 = arm_nn_requantize(ch_0_out_1, *out_mult, *out_shift);
            out_mult++;
            out_shift++;

            ch_0_out_0 = MAX(ch_0_out_0, activation_min);
            ch_0_out_0 = MIN(ch_0_out_0, activation_max);
            *out_0++ = (int16_t)ch_0_out_0;

            ch_0_out_1 = MAX(ch_0_out_1, activation_min);
            ch_0_out_1 = MIN(ch_0_out_1, activation_max);
            *out_1++ = (int16_t)ch_0_out_1;

            ch_1_out_0 = arm_nn_requantize(ch_1_out_0, *out_mult, *out_shift);
            ch_1_out_1 = arm_nn_requantize(ch_1_out_1, *out_mult, *out_shift);
            out_mult++;
            out_shift++;

            ch_1_out_0 = MAX(ch_1_out_0, activation_min);
            ch_1_out_0 = MIN(ch_1_out_0, activation_max);
            *out_0++ = (int16_t)ch_1_out_0;

            ch_1_out_1 = MAX(ch_1_out_1, activation_min);
            ch_1_out_1 = MIN(ch_1_out_1, activation_max);
            *out_1++ = (int16_t)ch_1_out_1;
        }
        else
        {
            int64_t ch_0_out_0_s64 = ch_0_out_0;
            int64_t ch_0_out_1_s64 = ch_0_out_1;
            int64_t ch_1_out_0_s64 = ch_1_out_0;
            int64_t ch_1_out_1_s64 = ch_1_out_1;

            if (num_col_a > MAX_COL_COUNT)
            {
                col_count = num_col_a_slow;
                while (col_count)
                {
                    int8_t a0 = *ip_a0++;
                    int16_t b0 = *ip_b0++;
                    int8_t a1 = *ip_a1++;
                    int16_t b1 = *ip_b1++;

                    ch_0_out_0_s64 += a0 * b0;
                    ch_0_out_1_s64 += a0 * b1;
                    ch_1_out_0_s64 += a1 * b0;
                    ch_1_out_1_s64 += a1 * b1;
                    col_count--;
                }
            }

            if (bias_s64)
            {
                ch_0_out_0_s64 += *bias_s64;
                ch_0_out_1_s64 += *bias_s64++;
                ch_1_out_0_s64 += *bias_s64;
                ch_1_out_1_s64 += *bias_s64++;
            }

            int32_t reduced_multiplier = REDUCE_MULTIPLIER(*out_mult);
            ch_0_out_0 = arm_nn_requantize_s64(ch_0_out_0_s64, reduced_multiplier, *out_shift);
            ch_0_out_1 = arm_nn_requantize_s64(ch_0_out_1_s64, reduced_multiplier, *out_shift);
            out_mult++;
            out_shift++;

            reduced_multiplier = REDUCE_MULTIPLIER(*out_mult);
            ch_1_out_0 = arm_nn_requantize_s64(ch_1_out_0_s64, reduced_multiplier, *out_shift);
            ch_1_out_1 = arm_nn_requantize_s64(ch_1_out_1_s64, reduced_multiplier, *out_shift);

            ch_0_out_0 = MAX(ch_0_out_0, activation_min);
            ch_0_out_0 = MIN(ch_0_out_0, activation_max);
            *out_0++ = (int16_t)ch_0_out_0;

            ch_0_out_1 = MAX(ch_0_out_1, activation_min);
            ch_0_out_1 = MIN(ch_0_out_1, activation_max);
            *out_1++ = (int16_t)ch_0_out_1;

            ch_1_out_0 = MAX(ch_1_out_0, activation_min);
            ch_1_out_0 = MIN(ch_1_out_0, activation_max);
            *out_0++ = (int16_t)ch_1_out_0;

            ch_1_out_1 = MAX(ch_1_out_1, activation_min);
            ch_1_out_1 = MIN(ch_1_out_1, activation_max);
            *out_1++ = (int16_t)ch_1_out_1;

            out_mult++;
            out_shift++;
        }

        /* Skip row */
        ip_a0 += num_col_a;
        row_count--;
    }

    /* Compute the last odd numbered row if any */
    if (output_ch & 0x1)
    {
        /* Setup pointers for B */
        const int16_t *ip_b0 = input_b;
        const int16_t *ip_b1 = ip_b0 + num_col_a;

        int32_t ch_0_out_0 = 0;
        int32_t ch_0_out_1 = 0;

    #if defined(ARM_MATH_DSP)
        uint16_t col_count = num_col_a_fast >> 2;
        while (col_count)
        {
            int32_t a01, a02;
            int32_t b0 = arm_nn_read_q15x2_ia(&ip_b0);
            int32_t b1 = arm_nn_read_q15x2_ia(&ip_b1);

            ip_a0 = read_and_pad(ip_a0, &a01, &a02);

            ch_0_out_0 = SMLAD(a01, b0, ch_0_out_0);
            ch_0_out_1 = SMLAD(a01, b1, ch_0_out_1);

            b0 = arm_nn_read_q15x2_ia(&ip_b0);
            b1 = arm_nn_read_q15x2_ia(&ip_b1);
            ch_0_out_0 = SMLAD(a02, b0, ch_0_out_0);
            ch_0_out_1 = SMLAD(a02, b1, ch_0_out_1);

            col_count--;
        }
        col_count = num_col_a & 0x3;
    #else
        int32_t col_count = num_col_a_fast;
    #endif
        while (col_count)
        {
            int8_t a0 = *ip_a0++;
            int16_t b0 = *ip_b0++;
            int16_t b1 = *ip_b1++;

            ch_0_out_0 += a0 * b0;
            ch_0_out_1 += a0 * b1;
            col_count--;
        }

        if (is_int32_bias)
        {
            if (bias_s32)
            {
                ch_0_out_0 += *bias_s32;
                ch_0_out_1 += *bias_s32++;
            }

            ch_0_out_0 = arm_nn_requantize(ch_0_out_0, *out_mult, *out_shift);
            ch_0_out_1 = arm_nn_requantize(ch_0_out_1, *out_mult, *out_shift);
            out_mult++;
            out_shift++;

            ch_0_out_0 = MAX(ch_0_out_0, activation_min);
            ch_0_out_0 = MIN(ch_0_out_0, activation_max);
            *out_0++ = (int16_t)ch_0_out_0;

            ch_0_out_1 = MAX(ch_0_out_1, activation_min);
            ch_0_out_1 = MIN(ch_0_out_1, activation_max);
            *out_1++ = (int16_t)ch_0_out_1;
        }
        else
        {
            int64_t ch_0_out_0_s64 = ch_0_out_0;
            int64_t ch_0_out_1_s64 = ch_0_out_1;

            if (num_col_a > MAX_COL_COUNT)
            {
                col_count = num_col_a_slow;
                while (col_count)
                {
                    int8_t a0 = *ip_a0++;
                    int16_t b0 = *ip_b0++;
                    int16_t b1 = *ip_b1++;

                    ch_0_out_0_s64 += a0 * b0;
                    ch_0_out_1_s64 += a0 * b1;
                    col_count--;
                }
            }

            if (bias_s64)
            {
                ch_0_out_0_s64 += *bias_s64;
                ch_0_out_1_s64 += *bias_s64++;
            }

            int32_t reduced_multiplier = REDUCE_MULTIPLIER(*out_mult);
            ch_0_out_0 = arm_nn_requantize_s64(ch_0_out_0_s64, reduced_multiplier, *out_shift);
            ch_0_out_1 = arm_nn_requantize_s64(ch_0_out_1_s64, reduced_multiplier, *out_shift);

            ch_0_out_0 = MAX(ch_0_out_0, activation_min);
            ch_0_out_0 = MIN(ch_0_out_0, activation_max);
            *out_0++ = (int16_t)ch_0_out_0;

            ch_0_out_1 = MAX(ch_0_out_1, activation_min);
            ch_0_out_1 = MIN(ch_0_out_1, activation_max);
            *out_1++ = (int16_t)ch_0_out_1;
            out_mult++;
            out_shift++;
        }
    }

    out_0 += output_ch;

    /* Return the new output pointer with offset */
    return out_0;
#else
    (void)input_a;
    (void)input_b;
    (void)output_ch;
    (void)out_shift;
    (void)out_mult;
    (void)activation_min;
    (void)activation_max;
    (void)num_col_a;
    (void)bias_data;
    (void)out_0;

    return NULL;
#endif
}

/**
 * @} end of Doxygen group
 */
