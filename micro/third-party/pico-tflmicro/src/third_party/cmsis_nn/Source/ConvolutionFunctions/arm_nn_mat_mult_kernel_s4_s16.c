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
 * Title:        arm_nn_mat_mult_kernel_s4_s16.c
 * Description:  Matrix-multiplication function for convolution
 *
 * $Date:        28 October 2024
 * $Revision:    V.1.1.0
 *
 * Target :  Arm(R) M-Profile Architecture
 * -------------------------------------------------------------------- */

#include "third_party/cmsis_nn/Include/arm_nnsupportfunctions.h"

/*
 * Matrix-multiplication function for convolution with per-channel requantization and 4bit weights.
 *
 * Refer header file for details.
 *
 */

int8_t *arm_nn_mat_mult_kernel_s4_s16(const int8_t *packed_input_a,
                                      const int16_t *input_b,
                                      const uint16_t output_ch,
                                      const int32_t *out_shift,
                                      const int32_t *out_mult,
                                      const int32_t out_offset,
                                      const int32_t activation_min,
                                      const int32_t activation_max,
                                      const int32_t num_col_a,
                                      const int32_t *const output_bias,
                                      int8_t *OPTIONAL_RESTRICT_KEYWORD out_0)
{

    /* set up the second output pointers */
    int8_t *out_1 = out_0 + output_ch;
    const int32_t *bias = output_bias;

    uint16_t row_count = output_ch / 4;
    const int8_t *packed_ip_a0 = packed_input_a;
    /* this loop over rows in A */
    while (row_count)
    {
        int8_t spillover0 = 0;
        int8_t spillover1 = 0;
        /* setup pointers for B */
        const int16_t *ip_b0 = input_b;
        const int16_t *ip_b1 = ip_b0 + num_col_a;

        /* Align the second pointer for A.
         * This will skip a row so that we can ensure the that spilled rows
         * don't offset the symmetry.
         */
        const int8_t *packed_ip_a1 = packed_ip_a0 + num_col_a;

        int32_t ch_0_out_0 = 0;
        int32_t ch_0_out_1 = 0;
        int32_t ch_1_out_0 = 0;
        int32_t ch_1_out_1 = 0;
        /* Init accumulator with bias for channel N and N + 1 */
        if (bias)
        {
            ch_0_out_0 = *bias;
            ch_0_out_1 = *bias;
            bias += 2;
            ch_1_out_0 = *bias;
            ch_1_out_1 = *bias--;
        }

#if defined(ARM_MATH_DSP)
        int32_t col_count = num_col_a / 4;
        /* accumulate over the vector */

        while (col_count)
        {
            int32_t a01, a02, a11, a12;
            int32_t b0 = arm_nn_read_q15x2_ia(&ip_b0);
            int32_t b1 = arm_nn_read_q15x2_ia(&ip_b1);

            read_and_pad_s4_ordered(packed_ip_a0, &a01, &a02);
            read_and_pad_s4_ordered(packed_ip_a1, &a11, &a12);
            packed_ip_a0 += 2;
            packed_ip_a1 += 2;

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
        } /* while over col_count */
        col_count = (num_col_a & 0x3) >> 1;
#else
        int32_t col_count = num_col_a >> 1;
#endif
        while (col_count)
        {
            int8_t lower_a0 = (int8_t)(packed_ip_a0[0] << 4) >> 4;
            int8_t higher_a0 = packed_ip_a0[0] >> 4;
            int16_t b0 = *ip_b0++;

            int8_t lower_a1 = (int8_t)(packed_ip_a1[0] << 4) >> 4;
            int8_t higher_a1 = packed_ip_a1[0] >> 4;
            int16_t b1 = *ip_b1++;

            packed_ip_a0++;
            packed_ip_a1++;

            ch_0_out_0 += lower_a0 * b0;
            ch_0_out_1 += lower_a0 * b1;
            ch_1_out_0 += lower_a1 * b0;
            ch_1_out_1 += lower_a1 * b1;

            b0 = *ip_b0++;
            b1 = *ip_b1++;

            ch_0_out_0 += higher_a0 * b0;
            ch_0_out_1 += higher_a0 * b1;
            ch_1_out_0 += higher_a1 * b0;
            ch_1_out_1 += higher_a1 * b1;

            col_count--;
        } /* while over col_count */
        /* left over column */
        if (num_col_a % 2)
        {
            int8_t lower_a0 = (int8_t)(packed_ip_a0[0] << 4) >> 4;
            spillover0 = packed_ip_a0[0] >> 4;
            int16_t b0 = *ip_b0++;

            int8_t lower_a1 = (int8_t)(packed_ip_a1[0] << 4) >> 4;
            spillover1 = packed_ip_a1[0] >> 4;
            int16_t b1 = *ip_b1++;

            packed_ip_a0++;
            packed_ip_a1++;

            ch_0_out_0 += lower_a0 * b0;
            ch_0_out_1 += lower_a0 * b1;
            ch_1_out_0 += lower_a1 * b0;
            ch_1_out_1 += lower_a1 * b1;
        }

        ch_0_out_0 = arm_nn_requantize(ch_0_out_0, *out_mult, *out_shift);
        ch_0_out_0 += out_offset;
        ch_0_out_0 = MAX(ch_0_out_0, activation_min);
        ch_0_out_0 = MIN(ch_0_out_0, activation_max);
        *out_0 = (int8_t)ch_0_out_0;
        out_0 += 2;

        ch_0_out_1 = arm_nn_requantize(ch_0_out_1, *out_mult, *out_shift);
        ch_0_out_1 += out_offset;
        ch_0_out_1 = MAX(ch_0_out_1, activation_min);
        ch_0_out_1 = MIN(ch_0_out_1, activation_max);
        *out_1 = (int8_t)ch_0_out_1;
        out_1 += 2;
        out_mult += 2;
        out_shift += 2;

        ch_1_out_0 = arm_nn_requantize(ch_1_out_0, *out_mult, *out_shift);
        ch_1_out_0 += out_offset;
        ch_1_out_0 = MAX(ch_1_out_0, activation_min);
        ch_1_out_0 = MIN(ch_1_out_0, activation_max);
        *out_0-- = (int8_t)ch_1_out_0;

        ch_1_out_1 = arm_nn_requantize(ch_1_out_1, *out_mult, *out_shift);
        ch_1_out_1 += out_offset;
        ch_1_out_1 = MAX(ch_1_out_1, activation_min);
        ch_1_out_1 = MIN(ch_1_out_1, activation_max);
        *out_1-- = (int8_t)ch_1_out_1;
        out_mult--;
        out_shift--;

        /* setup pointers for B */
        ip_b0 = input_b;
        ip_b1 = ip_b0 + num_col_a;

        /* Align the second pointer for A.
         * This will skip a row so that we can ensure the that spilled rows
         * don't offset the symmetry.
         */
        packed_ip_a1 = packed_ip_a0 + num_col_a;

        ch_0_out_0 = 0;
        ch_0_out_1 = 0;
        ch_1_out_0 = 0;
        ch_1_out_1 = 0;
        /* Init accumulator with bias for channel N and N + 1 */
        if (bias)
        {
            ch_0_out_0 = *bias;
            ch_0_out_1 = *bias;
            bias += 2;
            ch_1_out_0 = *bias;
            ch_1_out_1 = *bias++;
        }

        if (num_col_a % 2)
        {
            int16_t b0 = *ip_b0++;
            int16_t b1 = *ip_b1++;

            ch_0_out_0 += spillover0 * b0;
            ch_0_out_1 += spillover0 * b1;
            ch_1_out_0 += spillover1 * b0;
            ch_1_out_1 += spillover1 * b1;
        }

#if defined(ARM_MATH_DSP)
        col_count = num_col_a / 4;
        /* accumulate over the vector */
        while (col_count)
        {
            int32_t a01, a02, a11, a12;
            int32_t b0 = arm_nn_read_q15x2_ia(&ip_b0);
            int32_t b1 = arm_nn_read_q15x2_ia(&ip_b1);

            read_and_pad_s4_ordered(packed_ip_a0, &a01, &a02);
            read_and_pad_s4_ordered(packed_ip_a1, &a11, &a12);
            packed_ip_a0 += 2;
            packed_ip_a1 += 2;

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
        } /* while over col_count */
        col_count = (num_col_a & 0x3) >> 1;
#else
        col_count = num_col_a >> 1;
#endif
        while (col_count)
        {
            int8_t lower_a0 = (int8_t)(packed_ip_a0[0] << 4) >> 4;
            int8_t higher_a0 = packed_ip_a0[0] >> 4;
            int16_t b0 = *ip_b0++;

            int8_t lower_a1 = (int8_t)(packed_ip_a1[0] << 4) >> 4;
            int8_t higher_a1 = packed_ip_a1[0] >> 4;
            int16_t b1 = *ip_b1++;

            packed_ip_a0++;
            packed_ip_a1++;

            ch_0_out_0 += lower_a0 * b0;
            ch_0_out_1 += lower_a0 * b1;
            ch_1_out_0 += lower_a1 * b0;
            ch_1_out_1 += lower_a1 * b1;

            b0 = *ip_b0++;
            b1 = *ip_b1++;

            ch_0_out_0 += higher_a0 * b0;
            ch_0_out_1 += higher_a0 * b1;
            ch_1_out_0 += higher_a1 * b0;
            ch_1_out_1 += higher_a1 * b1;

            col_count--;
        } /* while over col_count */

        ch_0_out_0 = arm_nn_requantize(ch_0_out_0, *out_mult, *out_shift);
        ch_0_out_0 += out_offset;
        ch_0_out_0 = MAX(ch_0_out_0, activation_min);
        ch_0_out_0 = MIN(ch_0_out_0, activation_max);
        *out_0 = (int8_t)ch_0_out_0;
        out_0 += 2;

        ch_0_out_1 = arm_nn_requantize(ch_0_out_1, *out_mult, *out_shift);
        ch_0_out_1 += out_offset;
        ch_0_out_1 = MAX(ch_0_out_1, activation_min);
        ch_0_out_1 = MIN(ch_0_out_1, activation_max);
        *out_1 = (int8_t)ch_0_out_1;
        out_1 += 2;
        out_mult += 2;
        out_shift += 2;

        ch_1_out_0 = arm_nn_requantize(ch_1_out_0, *out_mult, *out_shift);
        ch_1_out_0 += out_offset;
        ch_1_out_0 = MAX(ch_1_out_0, activation_min);
        ch_1_out_0 = MIN(ch_1_out_0, activation_max);
        *out_0++ = (int8_t)ch_1_out_0;

        ch_1_out_1 = arm_nn_requantize(ch_1_out_1, *out_mult, *out_shift);
        ch_1_out_1 += out_offset;
        ch_1_out_1 = MAX(ch_1_out_1, activation_min);
        ch_1_out_1 = MIN(ch_1_out_1, activation_max);
        *out_1++ = (int8_t)ch_1_out_1;
        out_mult++;
        out_shift++;

        /* skip 2 rows */
        packed_ip_a0 += num_col_a;
        row_count--;
    }

    /* compute the 0 - 3 rows if any */
    int16_t left_over_rows = 0;
    while (left_over_rows < output_ch % 4)
    {
        /* setup pointers for B */
        const int16_t *ip_b0 = input_b;
        const int16_t *ip_b1 = ip_b0 + num_col_a;

        int32_t ch_0_out_0 = 0;
        int32_t ch_0_out_1 = 0;

        /* load the bias */
        if (bias)
        {
            ch_0_out_0 = *bias;
            ch_0_out_1 = *bias++;
        }

        if (left_over_rows == 1 && num_col_a % 2)
        {
            int16_t b0 = *ip_b0++;
            int16_t b1 = *ip_b1++;
            int8_t spilled_column = packed_ip_a0[0] >> 4;

            ++packed_ip_a0;

            ch_0_out_0 += spilled_column * b0;
            ch_0_out_1 += spilled_column * b1;
        }

#if defined(ARM_MATH_DSP)
        int32_t col_count = num_col_a / 4;
        while (col_count)
        {
            int32_t a01, a02;
            int32_t b0 = arm_nn_read_q15x2_ia(&ip_b0);
            int32_t b1 = arm_nn_read_q15x2_ia(&ip_b1);

            read_and_pad_s4_ordered(packed_ip_a0, &a01, &a02);
            packed_ip_a0 += 2;

            ch_0_out_0 = SMLAD(a01, b0, ch_0_out_0);
            ch_0_out_1 = SMLAD(a01, b1, ch_0_out_1);

            b0 = arm_nn_read_q15x2_ia(&ip_b0);
            b1 = arm_nn_read_q15x2_ia(&ip_b1);
            ch_0_out_0 = SMLAD(a02, b0, ch_0_out_0);
            ch_0_out_1 = SMLAD(a02, b1, ch_0_out_1);

            col_count--;
        }
        col_count = (num_col_a & 0x3) >> 1;

#else
        int32_t col_count = num_col_a >> 1;
#endif

        while (col_count)
        {
            int8_t a0 = (int8_t)(packed_ip_a0[0] << 4) >> 4;
            int8_t a1 = packed_ip_a0[0] >> 4;
            int16_t b0 = *ip_b0++;
            int16_t b1 = *ip_b1++;

            ++packed_ip_a0;

            ch_0_out_0 += a0 * b0;
            ch_0_out_1 += a0 * b1;

            b0 = *ip_b0++;
            b1 = *ip_b1++;

            ch_0_out_0 += a1 * b0;
            ch_0_out_1 += a1 * b1;

            col_count--;
        }
        if (num_col_a % 2 && left_over_rows != 1)
        {
            int8_t a0 = (int8_t)(packed_ip_a0[0] << 4) >> 4;

            int16_t b0 = *ip_b0++;
            int16_t b1 = *ip_b1++;

            ch_0_out_0 += a0 * b0;
            ch_0_out_1 += a0 * b1;
        }
        ch_0_out_0 = arm_nn_requantize(ch_0_out_0, *out_mult, *out_shift);
        ch_0_out_0 += out_offset;
        ch_0_out_0 = MAX(ch_0_out_0, activation_min);
        ch_0_out_0 = MIN(ch_0_out_0, activation_max);
        *out_0++ = (int8_t)ch_0_out_0;

        ch_0_out_1 = arm_nn_requantize(ch_0_out_1, *out_mult, *out_shift);
        ch_0_out_1 += out_offset;
        ch_0_out_1 = MAX(ch_0_out_1, activation_min);
        ch_0_out_1 = MIN(ch_0_out_1, activation_max);
        *out_1++ = (int8_t)ch_0_out_1;
        out_mult++;
        out_shift++;

        ++left_over_rows;
    }

    out_0 += output_ch;

    /* return the new output pointer with offset */
    return out_0;
}
