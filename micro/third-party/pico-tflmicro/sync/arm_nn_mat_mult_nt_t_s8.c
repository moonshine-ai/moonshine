/*
 * SPDX-FileCopyrightText: Copyright 2020-2023 Arm Limited and/or its affiliates <open-source-office@arm.com>
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
 * Title:        arm_nn_mat_mult_s8_nt_t_s8
 * Description:  Matrix multiplication support function with the right-hand-side (rhs) matrix transposed
 *
 * $Date:        22 March 2023
 * $Revision:    V.2.1.2
 *
 * Target :  Arm(R) M-Profile Architecture
 *
 * -------------------------------------------------------------------- */

#include "third_party/cmsis_nn/Include/arm_nnsupportfunctions.h"

/**
 * @ingroup groupSupport
 */

/**
 * @addtogroup supportConvolution
 * @{
 */

// Uncomment this to try the experimental dual core support on the RP2040.
#define TF_LITE_PICO_MULTICORE

#ifdef TF_LITE_PICO_MULTICORE

#include "pico/stdlib.h"
#include "pico/multicore.h"

typedef struct {
    int32_t rhs_rows_start;
    int32_t rhs_rows_end;
    const int8_t *lhs;
    const int8_t *rhs;
    const int32_t *bias;
    int8_t* dst;
    const int32_t *dst_multipliers;
    const int32_t *dst_shifts;
    int32_t lhs_rows;
    int32_t rhs_rows;
    int32_t rhs_cols;
    int32_t lhs_offset;
    int32_t dst_offset;
    int32_t activation_min;
    int32_t activation_max;
    int32_t lhs_cols_offset;
} MatMulArgs;

static MatMulArgs g_core1_mat_mul_args;

static void calculate_two_rows(
    const int8_t *lhs,
    const int8_t *rhs,
    const int32_t *bias,
    int8_t* dst,
    const int32_t *dst_multipliers,
    const int32_t *dst_shifts,
    const int32_t lhs_rows,
    const int32_t rhs_rows,
    const int32_t rhs_cols,
    const int32_t lhs_offset,
    const int32_t dst_offset,
    const int32_t activation_min,
    const int32_t activation_max,
    const int32_t lhs_cols_offset,
    const int32_t rhs_rows_idx) {

    const int8_t *lhs_ptr = &lhs[0];
    int8_t *dst_ptr = &dst[0];

    int32_t lhs_offset_contribution0 = 0;
    int32_t lhs_offset_contribution1 = 0;

    for (int32_t x = 0; x < rhs_cols; ++x)
    {
        lhs_offset_contribution0 += rhs[x];
        lhs_offset_contribution1 += rhs[x + rhs_cols];
    }

    lhs_offset_contribution0 *= lhs_offset;
    lhs_offset_contribution1 *= lhs_offset;
    if (bias)
    {
        lhs_offset_contribution0 += bias[rhs_rows_idx];
        lhs_offset_contribution1 += bias[rhs_rows_idx + 1];
    }

    int32_t lhs_rows_idx = lhs_rows >> 1;

    while (lhs_rows_idx)
    {
        const int8_t *rhs_ptr = &rhs[0];

        int32_t res00 = lhs_offset_contribution0;
        int32_t res01 = lhs_offset_contribution1;
        int32_t res10 = lhs_offset_contribution0;
        int32_t res11 = lhs_offset_contribution1;

        for (int32_t rhs_cols_idx = rhs_cols; rhs_cols_idx != 0; rhs_cols_idx--)
        {
            int8_t rhs_value0 = rhs_ptr[0];
            int8_t rhs_value1 = rhs_ptr[rhs_cols];
            int8_t lhs_value = lhs_ptr[0];

            res00 += lhs_value * rhs_value0;
            res01 += lhs_value * rhs_value1;

            lhs_value = lhs_ptr[lhs_cols_offset];
            res10 += lhs_value * rhs_value0;
            res11 += lhs_value * rhs_value1;

            ++rhs_ptr;
            ++lhs_ptr;
        }

        // Quantize down
        res00 = arm_nn_requantize(res00, dst_multipliers[rhs_rows_idx], dst_shifts[rhs_rows_idx]);
        res01 = arm_nn_requantize(res01, dst_multipliers[rhs_rows_idx + 1], dst_shifts[rhs_rows_idx + 1]);
        res10 = arm_nn_requantize(res10, dst_multipliers[rhs_rows_idx], dst_shifts[rhs_rows_idx]);
        res11 = arm_nn_requantize(res11, dst_multipliers[rhs_rows_idx + 1], dst_shifts[rhs_rows_idx + 1]);

        // Add offset
        res00 += dst_offset;
        res01 += dst_offset;
        res10 += dst_offset;
        res11 += dst_offset;

        // Clamp the result
        res00 = MAX(res00, activation_min);
        res00 = MIN(res00, activation_max);
        res01 = MAX(res01, activation_min);
        res01 = MIN(res01, activation_max);
        res10 = MAX(res10, activation_min);
        res10 = MIN(res10, activation_max);
        res11 = MAX(res11, activation_min);
        res11 = MIN(res11, activation_max);

        dst_ptr[0] = (int8_t)res00;
        dst_ptr[1] = (int8_t)res01;
        dst_ptr += rhs_rows;
        dst_ptr[0] = (int8_t)res10;
        dst_ptr[1] = (int8_t)res11;
        dst_ptr += rhs_rows;

        lhs_ptr -= rhs_cols;
        lhs_ptr += 2 * lhs_cols_offset;

        lhs_rows_idx--;
    }

    // Left-over rows
    if (lhs_rows % 2)
    {
        const int8_t *rhs_ptr = &rhs[0];

        int32_t res00 = lhs_offset_contribution0;
        int32_t res01 = lhs_offset_contribution1;

        for (int32_t rhs_cols_idx = rhs_cols; rhs_cols_idx != 0; rhs_cols_idx--)
        {
            int8_t rhs_value0 = rhs_ptr[0];
            int8_t rhs_value1 = rhs_ptr[rhs_cols];
            int8_t lhs_value = lhs_ptr[0];

            res00 += lhs_value * rhs_value0;
            res01 += lhs_value * rhs_value1;

            ++rhs_ptr;
            ++lhs_ptr;
        }

        // Quantize down
        res00 = arm_nn_requantize(res00, dst_multipliers[rhs_rows_idx], dst_shifts[rhs_rows_idx]);
        res01 = arm_nn_requantize(res01, dst_multipliers[rhs_rows_idx + 1], dst_shifts[rhs_rows_idx + 1]);

        // Add offset
        res00 += dst_offset;
        res01 += dst_offset;

        // Clamp the result
        res00 = MAX(res00, activation_min);
        res00 = MIN(res00, activation_max);
        res01 = MAX(res01, activation_min);
        res01 = MIN(res01, activation_max);

        dst_ptr[0] = (int8_t)res00;
        dst_ptr[1] = (int8_t)res01;
    }
}

static void calculate_row_range(
    int32_t rhs_rows_start,
    int32_t rhs_rows_end,
    const int8_t *lhs,
    const int8_t *rhs,
    const int32_t *bias,
    int8_t* dst,
    const int32_t *dst_multipliers,
    const int32_t *dst_shifts,
    const int32_t lhs_rows,
    const int32_t rhs_rows,
    const int32_t rhs_cols,
    const int32_t lhs_offset,
    const int32_t dst_offset,
    const int32_t activation_min,
    const int32_t activation_max,
    const int32_t lhs_cols_offset) {

    const int8_t* current_rhs = rhs + (rhs_rows_start * rhs_cols);
    int8_t* current_dst = dst + rhs_rows_start;

    for (int32_t rhs_rows_idx = rhs_rows_start; rhs_rows_idx < rhs_rows_end; rhs_rows_idx += 2)
    {
        calculate_two_rows(
            lhs,
            current_rhs,
            bias,
            current_dst,
            dst_multipliers,
            dst_shifts,
            lhs_rows,
            rhs_rows,
            rhs_cols,
            lhs_offset,
            dst_offset,
            activation_min,
            activation_max,
            lhs_cols_offset,
            rhs_rows_idx);

        current_rhs += 2 * rhs_cols;
        current_dst += 2;
    }

}

static void mat_mul_task(const MatMulArgs* args) {
    const int32_t rhs_rows_start = args->rhs_rows_start;
    const int32_t rhs_rows_end = args->rhs_rows_end;
    const int8_t *lhs = args->lhs;
    const int8_t *rhs = args->rhs;
    const int32_t *bias = args->bias;
    int8_t* dst = args->dst;
    const int32_t *dst_multipliers = args->dst_multipliers;
    const int32_t *dst_shifts = args->dst_shifts;
    int32_t lhs_rows = args->lhs_rows;
    int32_t rhs_rows = args->rhs_rows;
    int32_t rhs_cols = args->rhs_cols;
    int32_t lhs_offset = args->lhs_offset;
    int32_t dst_offset = args->dst_offset;
    int32_t activation_min = args->activation_min;
    int32_t activation_max = args->activation_max;
    int32_t lhs_cols_offset = args->lhs_cols_offset;

    calculate_row_range(
        rhs_rows_start,
        rhs_rows_end,
        lhs,
        rhs,
        bias,
        dst,
        dst_multipliers,
        dst_shifts,
        lhs_rows,
        rhs_rows,
        rhs_cols,
        lhs_offset,
        dst_offset,
        activation_min,
        activation_max,
        lhs_cols_offset);
}

static void core1_mat_mul_worker(void) {
    mat_mul_task(&g_core1_mat_mul_args);

    // Signal we're done by pushing a result of zero.
    multicore_fifo_push_blocking(ARM_CMSIS_NN_SUCCESS);
}

#endif  // TF_LITE_PICO_MULTICORE

/*
 * s8 matrix multiplication with the right-hand-side matrix transposed
 *
 * Refer header file for details.
 *
 */



arm_cmsis_nn_status arm_nn_mat_mult_nt_t_s8(const int8_t *lhs,
                                            const int8_t *rhs,
                                            const int32_t *bias,
                                            int8_t *dst,
                                            const int32_t *dst_multipliers,
                                            const int32_t *dst_shifts,
                                            const int32_t lhs_rows,
                                            const int32_t rhs_rows,
                                            const int32_t rhs_cols,
                                            const int32_t lhs_offset,
                                            const int32_t dst_offset,
                                            const int32_t activation_min,
                                            const int32_t activation_max,
                                            const int32_t row_address_offset,
                                            const int32_t lhs_cols_offset)
{

#if defined(ARM_MATH_MVEI)
    int i_items = 0;
    for (; i_items <= (lhs_rows - 4); i_items += 4)
    {
        for (int i = 0; i < rhs_rows; i++)
        {
            int32_t acc_n0 = 0;
            int32_t acc_n1 = 0;
            int32_t acc_n2 = 0;
            int32_t acc_n3 = 0;

            const int8_t *lhs_vec = lhs;
            const int8_t *ip_row_1 = lhs + lhs_cols_offset;
            const int8_t *ip_row_2 = lhs + (2 * lhs_cols_offset);
            const int8_t *ip_row_3 = lhs + (3 * lhs_cols_offset);
            const int8_t *col_base = rhs + i * rhs_cols;
            int32_t sum_tmp = 0;

    #if defined(ARM_MATH_AUTOVECTORIZE)
            for (int j = 0; j < rhs_cols; j++)
            {
                int32_t col = col_base[j];
                sum_tmp += col;
                acc_n0 += lhs_vec[j] * col;
                acc_n1 += ip_row_1[j] * col;
                acc_n2 += ip_row_2[j] * col;
                acc_n3 += ip_row_3[j] * col;
            }
    #else
            // Note: If operand initialization is moved around, use '&' constraint to
            // specify earlyclobber operands.
            __ASM volatile(" .p2align 2                             \n"
                           "   wlstp.8         lr, %[cnt], 1f       \n"
                           "   mov             %[sum], 0            \n"
                           "   mov             %[out0], 0           \n"
                           "   mov             %[out1], 0           \n"
                           "   mov             %[out2], 0           \n"
                           "   mov             %[out3], 0           \n"
                           "   vldrb.8         q0, [%[col]], #16    \n"
                           "2:                                      \n"
                           "   vaddva.s8      %[sum], q0            \n"
                           "   vldrb.8         q1, [%[row0]], #16   \n"
                           "   vmladava.s8    %[out0], q0, q1       \n"
                           "   vldrb.8         q2, [%[row1]], #16   \n"
                           "   vmladava.s8     %[out1], q0, q2      \n"
                           "   vldrb.8         q3, [%[row2]], #16   \n"
                           "   vmladava.s8     %[out2], q0, q3      \n"
                           "   vldrb.8         q4, [%[row3]], #16   \n"
                           "   vmladava.s8     %[out3], q0, q4      \n"
                           "   vldrb.8         q0, [%[col]], #16    \n"
                           "   letp            lr, 2b               \n"
                           "1:                                      \n"
                           : [col] "+r"(col_base),
                             [sum] "=Te"(sum_tmp),
                             [row0] "+r"(lhs_vec),
                             [row1] "+r"(ip_row_1),
                             [row2] "+r"(ip_row_2),
                             [row3] "+r"(ip_row_3),
                             [out0] "=Te"(acc_n0),
                             [out1] "=Te"(acc_n1),
                             [out2] "=Te"(acc_n2),
                             [out3] "=Te"(acc_n3)
                           : [cnt] "r"(rhs_cols)
                           : "q0", "q1", "q2", "q3", "q4", "memory", "r14");
    #endif
            int32x4_t res = {acc_n0, acc_n1, acc_n2, acc_n3};
            sum_tmp *= lhs_offset;
            if (bias)
            {
                sum_tmp += bias[i];
            }
            res = vaddq_n_s32(res, sum_tmp);

            res = arm_requantize_mve(res, dst_multipliers[i], dst_shifts[i]);
            res = vaddq_n_s32(res, dst_offset);

            res = vmaxq_s32(res, vdupq_n_s32(activation_min));
            res = vminq_s32(res, vdupq_n_s32(activation_max));

            const uint32x4_t scatter_offset = {0, (uint32_t)rhs_rows, (uint32_t)rhs_rows * 2, (uint32_t)rhs_rows * 3};
            vstrbq_scatter_offset_s32(dst, scatter_offset, res);
            dst++;
        }
        lhs += 4 * lhs_cols_offset;
        dst += (3 * rhs_rows);
    }

    for (; i_items < lhs_rows; i_items++)
    {
        int32_t acc[4];
        const int32_t *multipliers = dst_multipliers;
        const int32_t *shifts = dst_shifts;
        for (int i = 0; i < rhs_rows; i++)
        {
            int32_t acc_n0 = 0;
            const int8_t *lhs_vec = lhs;
            const int8_t *col_base = rhs + i * rhs_cols;
            int32_t sum_tmp = 0;

    #if defined(ARM_MATH_AUTOVECTORIZE)
            for (int j = 0; j < rhs_cols; j++)
            {
                int32_t col = col_base[j];
                sum_tmp += col;
                acc_n0 += lhs_vec[j] * col;
            }
    #else
            __ASM volatile(" .p2align 2                             \n"
                           "   wlstp.8         lr, %[cnt], 1f       \n"
                           "   mov             %[sum], 0            \n"
                           "   mov             %[out0], 0            \n"
                           "   vldrb.8         q0, [%[col]], #16    \n"
                           "2:                                      \n"
                           "   vaddva.s8      %[sum], q0            \n"
                           "   vldrb.8         q1, [%[row0]], #16   \n"
                           "   vmladava.s8    %[out0], q0, q1       \n"
                           "   vldrb.8         q0, [%[col]], #16    \n"
                           "   letp            lr, 2b               \n"
                           "1:                                      \n"
                           : [col] "+r"(col_base), [sum] "=Te"(sum_tmp), [row0] "+r"(lhs_vec), [out0] "=Te"(acc_n0)
                           : [cnt] "r"(rhs_cols)
                           : "q0", "q1", "memory", "r14");
    #endif
            sum_tmp *= lhs_offset;
            sum_tmp += acc_n0;
            if (bias)
            {
                sum_tmp += bias[i];
            }
            const int32_t index = i & 0x3;
            acc[index] = sum_tmp;

            if (index == 3)
            {
                int32x4_t res = vldrwq_s32(acc);
                res = arm_requantize_mve_32x4(res, vldrwq_s32(multipliers), vldrwq_s32(shifts));
                multipliers += 4;
                shifts += 4;
                res = vaddq_n_s32(res, dst_offset);
                res = vmaxq_s32(res, vdupq_n_s32(activation_min));
                res = vminq_s32(res, vdupq_n_s32(activation_max));
                vstrbq_s32(dst, res);
                dst += 4;
            }
        }
        lhs += lhs_cols_offset;
        const int32_t tail_rows = rhs_rows & 0x3;
        for (int i = 0; i < tail_rows; i++)
        {
            int32_t acc_n0 = acc[i];
            acc_n0 = arm_nn_requantize(acc_n0, multipliers[i], shifts[i]);
            acc_n0 += dst_offset;
            acc_n0 = MAX(acc_n0, activation_min);
            acc_n0 = MIN(acc_n0, activation_max);
            *dst++ = (int8_t)acc_n0;
        }
    }

#elif defined(ARM_MATH_DSP)
    const int32_t rhs_off0 = rhs_cols - 4;
    const int32_t lhs_off0 = lhs_cols_offset - 4;

    for (int32_t rhs_rows_idx = 0; rhs_rows_idx <= (rhs_rows - 2); rhs_rows_idx += 2)
    {
        const int8_t *lhs_ptr = &lhs[0];
        int8_t *dst_ptr = &dst[0];

        int32_t lhs_offset_contribution0 = 0;
        int32_t lhs_offset_contribution1 = 0;

        for (int32_t x = 0; x < rhs_cols; ++x)
        {
            lhs_offset_contribution0 += rhs[x];
            lhs_offset_contribution1 += rhs[x + rhs_cols];
        }

        lhs_offset_contribution0 *= lhs_offset;
        lhs_offset_contribution1 *= lhs_offset;
        if (bias)
        {
            lhs_offset_contribution0 += bias[rhs_rows_idx];
            lhs_offset_contribution1 += bias[rhs_rows_idx + 1];
        }

        int32_t lhs_rows_idx = lhs_rows >> 1;

        while (lhs_rows_idx)
        {
            const int8_t *rhs_ptr = &rhs[0];

            int32_t res00 = lhs_offset_contribution0;
            int32_t res01 = lhs_offset_contribution1;
            int32_t res10 = lhs_offset_contribution0;
            int32_t res11 = lhs_offset_contribution1;

            int32_t rhs_cols_idx = 0;

            int32_t val0, val1, val2, val3, val4, val5;

            for (; rhs_cols_idx <= (rhs_cols - 16); rhs_cols_idx += 16)
            {
                val1 = arm_nn_read_s8x4_ia((const int8_t **)&rhs_ptr);
                val2 = SXTB16(val1);
                val0 = arm_nn_read_s8x4_ia((const int8_t **)&lhs_ptr);
                val3 = SXTB16(val0);
                val4 = arm_nn_read_s8x4((const int8_t *)&rhs_ptr[rhs_off0]);
                val1 = SXTB16_RORn(val1, 8);
                val0 = SXTB16_RORn(val0, 8);

                // 4 x MAC res00, res01
                res00 = SMLAD(val3, val2, res00);
                val5 = SXTB16(val4);
                res00 = SMLAD(val0, val1, res00);
                val4 = SXTB16_RORn(val4, 8);
                res01 = SMLAD(val3, val5, res01);
                res01 = SMLAD(val0, val4, res01);

                // 4 x MAC res10, res11
                val0 = arm_nn_read_s8x4((const int8_t *)&lhs_ptr[lhs_off0]);
                val3 = SXTB16(val0);
                val0 = SXTB16_RORn(val0, 8);
                res10 = SMLAD(val3, val2, res10);
                res11 = SMLAD(val3, val5, res11);
                res10 = SMLAD(val0, val1, res10);
                val1 = arm_nn_read_s8x4_ia((const int8_t **)&rhs_ptr);
                res11 = SMLAD(val0, val4, res11);

                val4 = arm_nn_read_s8x4((const int8_t *)&rhs_ptr[rhs_off0]);
                val2 = SXTB16(val1);
                val0 = arm_nn_read_s8x4_ia((const int8_t **)&lhs_ptr);
                val3 = SXTB16(val0);
                val1 = SXTB16_RORn(val1, 8);
                val0 = SXTB16_RORn(val0, 8);

                // 4 x MAC res00, res01
                res00 = SMLAD(val3, val2, res00);
                val5 = SXTB16(val4);
                res00 = SMLAD(val0, val1, res00);
                val4 = SXTB16_RORn(val4, 8);
                res01 = SMLAD(val3, val5, res01);
                res01 = SMLAD(val0, val4, res01);

                // 4 x MAC res10, res11
                val0 = arm_nn_read_s8x4((const int8_t *)&lhs_ptr[lhs_off0]);
                val3 = SXTB16(val0);
                val0 = SXTB16_RORn(val0, 8);
                res10 = SMLAD(val3, val2, res10);
                res11 = SMLAD(val3, val5, res11);
                res10 = SMLAD(val0, val1, res10);
                val1 = arm_nn_read_s8x4_ia((const int8_t **)&rhs_ptr);
                res11 = SMLAD(val0, val4, res11);

                val4 = arm_nn_read_s8x4((const int8_t *)&rhs_ptr[rhs_off0]);
                val2 = SXTB16(val1);
                val0 = arm_nn_read_s8x4_ia((const int8_t **)&lhs_ptr);
                val3 = SXTB16(val0);
                val1 = SXTB16_RORn(val1, 8);
                val0 = SXTB16_RORn(val0, 8);

                // 4 x MAC res00, res01
                res00 = SMLAD(val3, val2, res00);
                val5 = SXTB16(val4);
                res00 = SMLAD(val0, val1, res00);
                val4 = SXTB16_RORn(val4, 8);
                res01 = SMLAD(val3, val5, res01);
                res01 = SMLAD(val0, val4, res01);

                // 4 x MAC res10, res11
                val0 = arm_nn_read_s8x4((const int8_t *)&lhs_ptr[lhs_off0]);
                val3 = SXTB16(val0);
                val0 = SXTB16_RORn(val0, 8);
                res10 = SMLAD(val3, val2, res10);
                res11 = SMLAD(val3, val5, res11);
                res10 = SMLAD(val0, val1, res10);
                val1 = arm_nn_read_s8x4_ia((const int8_t **)&rhs_ptr);
                res11 = SMLAD(val0, val4, res11);

                val4 = arm_nn_read_s8x4((const int8_t *)&rhs_ptr[rhs_off0]);
                val2 = SXTB16(val1);
                val0 = arm_nn_read_s8x4_ia((const int8_t **)&lhs_ptr);
                val3 = SXTB16(val0);
                val1 = SXTB16_RORn(val1, 8);
                val0 = SXTB16_RORn(val0, 8);

                // 4 x MAC res00, res01
                res00 = SMLAD(val3, val2, res00);
                val5 = SXTB16(val4);
                res00 = SMLAD(val0, val1, res00);
                val4 = SXTB16_RORn(val4, 8);
                res01 = SMLAD(val3, val5, res01);
                res01 = SMLAD(val0, val4, res01);

                // 4 x MAC res10, res11
                val0 = arm_nn_read_s8x4((const int8_t *)&lhs_ptr[lhs_off0]);
                val3 = SXTB16(val0);
                val0 = SXTB16_RORn(val0, 8);
                res10 = SMLAD(val3, val2, res10);
                res11 = SMLAD(val3, val5, res11);
                res10 = SMLAD(val0, val1, res10);
                res11 = SMLAD(val0, val4, res11);
            }

            for (; rhs_cols_idx <= (rhs_cols - 4); rhs_cols_idx += 4)
            {
                val1 = arm_nn_read_s8x4_ia((const int8_t **)&rhs_ptr);
                val2 = SXTB16(val1);
                val0 = arm_nn_read_s8x4_ia((const int8_t **)&lhs_ptr);
                val3 = SXTB16(val0);
                val4 = arm_nn_read_s8x4((const int8_t *)&rhs_ptr[rhs_off0]);
                val1 = SXTB16_RORn(val1, 8);
                val0 = SXTB16_RORn(val0, 8);

                // 4 x MAC res00, res01
                res00 = SMLAD(val3, val2, res00);
                val5 = SXTB16(val4);
                res00 = SMLAD(val0, val1, res00);
                val4 = SXTB16_RORn(val4, 8);
                res01 = SMLAD(val3, val5, res01);
                res01 = SMLAD(val0, val4, res01);

                // 4 x MAC res10, res11
                val0 = arm_nn_read_s8x4((const int8_t *)&lhs_ptr[lhs_off0]);
                val3 = SXTB16(val0);
                val0 = SXTB16_RORn(val0, 8);
                res10 = SMLAD(val3, val2, res10);
                res11 = SMLAD(val3, val5, res11);
                res10 = SMLAD(val0, val1, res10);
                res11 = SMLAD(val0, val4, res11);
            }

            for (; rhs_cols_idx < rhs_cols; ++rhs_cols_idx)
            {
                int8_t rhs_value0 = rhs_ptr[0];
                int8_t rhs_value1 = rhs_ptr[rhs_cols];
                int8_t lhs_value = lhs_ptr[0];

                res00 += lhs_value * rhs_value0;
                res01 += lhs_value * rhs_value1;

                lhs_value = lhs_ptr[lhs_cols_offset];
                res10 += lhs_value * rhs_value0;
                res11 += lhs_value * rhs_value1;

                ++rhs_ptr;
                ++lhs_ptr;
            }

            // Quantize down
            res00 = arm_nn_requantize(res00, dst_multipliers[rhs_rows_idx], dst_shifts[rhs_rows_idx]);
            res01 = arm_nn_requantize(res01, dst_multipliers[rhs_rows_idx + 1], dst_shifts[rhs_rows_idx + 1]);
            res10 = arm_nn_requantize(res10, dst_multipliers[rhs_rows_idx], dst_shifts[rhs_rows_idx]);
            res11 = arm_nn_requantize(res11, dst_multipliers[rhs_rows_idx + 1], dst_shifts[rhs_rows_idx + 1]);

            // Add offset
            res00 += dst_offset;
            res01 += dst_offset;
            res10 += dst_offset;
            res11 += dst_offset;

            // Clamp the result
            res00 = MAX(res00, activation_min);
            res00 = MIN(res00, activation_max);
            res01 = MAX(res01, activation_min);
            res01 = MIN(res01, activation_max);
            res10 = MAX(res10, activation_min);
            res10 = MIN(res10, activation_max);
            res11 = MAX(res11, activation_min);
            res11 = MIN(res11, activation_max);

            dst_ptr[0] = (int8_t)res00;
            dst_ptr[1] = (int8_t)res01;
            dst_ptr += rhs_rows;
            dst_ptr[0] = (int8_t)res10;
            dst_ptr[1] = (int8_t)res11;
            dst_ptr += rhs_rows;

            lhs_ptr -= rhs_cols;
            lhs_ptr += 2 * lhs_cols_offset;

            lhs_rows_idx--;
        }

        // Left-over rows
        if (lhs_rows % 2)
        {
            const int8_t *rhs_ptr = &rhs[0];

            int32_t res00 = lhs_offset_contribution0;
            int32_t res01 = lhs_offset_contribution1;

            int32_t rhs_cols_idx = 0;

            int32_t val0, val1, val2, val3, val4, val5;
            for (; rhs_cols_idx <= (rhs_cols - 16); rhs_cols_idx += 16)
            {
                val0 = arm_nn_read_s8x4_ia((const int8_t **)&rhs_ptr);
                val1 = arm_nn_read_s8x4((const int8_t *)&rhs_ptr[rhs_off0]);
                val2 = arm_nn_read_s8x4_ia((const int8_t **)&lhs_ptr);
                val3 = SXTB16(val0);
                val5 = SXTB16(val2);
                val4 = SXTB16(val1);
                val0 = SXTB16_RORn(val0, 8);
                val2 = SXTB16_RORn(val2, 8);
                val1 = SXTB16_RORn(val1, 8);

                // 4 x MAC res00, res01
                res00 = SMLAD(val5, val3, res00);
                res00 = SMLAD(val2, val0, res00);
                res01 = SMLAD(val5, val4, res01);
                res01 = SMLAD(val2, val1, res01);

                val0 = arm_nn_read_s8x4_ia((const int8_t **)&rhs_ptr);
                val1 = arm_nn_read_s8x4((const int8_t *)&rhs_ptr[rhs_off0]);
                val2 = arm_nn_read_s8x4_ia((const int8_t **)&lhs_ptr);
                val3 = SXTB16(val0);
                val5 = SXTB16(val2);
                val4 = SXTB16(val1);
                val0 = SXTB16_RORn(val0, 8);
                val2 = SXTB16_RORn(val2, 8);
                val1 = SXTB16_RORn(val1, 8);

                // 4 x MAC res00, res01
                res00 = SMLAD(val5, val3, res00);
                res00 = SMLAD(val2, val0, res00);
                res01 = SMLAD(val5, val4, res01);
                res01 = SMLAD(val2, val1, res01);

                val0 = arm_nn_read_s8x4_ia((const int8_t **)&rhs_ptr);
                val1 = arm_nn_read_s8x4((const int8_t *)&rhs_ptr[rhs_off0]);
                val2 = arm_nn_read_s8x4_ia((const int8_t **)&lhs_ptr);
                val3 = SXTB16(val0);
                val5 = SXTB16(val2);
                val4 = SXTB16(val1);
                val0 = SXTB16_RORn(val0, 8);
                val2 = SXTB16_RORn(val2, 8);
                val1 = SXTB16_RORn(val1, 8);

                // 4 x MAC res00, res01
                res00 = SMLAD(val5, val3, res00);
                res00 = SMLAD(val2, val0, res00);
                res01 = SMLAD(val5, val4, res01);
                res01 = SMLAD(val2, val1, res01);

                val0 = arm_nn_read_s8x4_ia((const int8_t **)&rhs_ptr);
                val1 = arm_nn_read_s8x4((const int8_t *)&rhs_ptr[rhs_off0]);
                val2 = arm_nn_read_s8x4_ia((const int8_t **)&lhs_ptr);
                val3 = SXTB16(val0);
                val5 = SXTB16(val2);
                val4 = SXTB16(val1);
                val0 = SXTB16_RORn(val0, 8);
                val2 = SXTB16_RORn(val2, 8);
                val1 = SXTB16_RORn(val1, 8);

                // 4 x MAC res00, res01
                res00 = SMLAD(val5, val3, res00);
                res00 = SMLAD(val2, val0, res00);
                res01 = SMLAD(val5, val4, res01);
                res01 = SMLAD(val2, val1, res01);
            }

            for (; rhs_cols_idx <= (rhs_cols - 4); rhs_cols_idx += 4)
            {
                val0 = arm_nn_read_s8x4_ia((const int8_t **)&rhs_ptr);
                val1 = arm_nn_read_s8x4((const int8_t *)&rhs_ptr[rhs_off0]);
                val2 = arm_nn_read_s8x4_ia((const int8_t **)&lhs_ptr);
                val3 = SXTB16(val0);
                val5 = SXTB16(val2);
                val4 = SXTB16(val1);
                val0 = SXTB16_RORn(val0, 8);
                val2 = SXTB16_RORn(val2, 8);
                val1 = SXTB16_RORn(val1, 8);

                // 4 x MAC res00, res01
                res00 = SMLAD(val5, val3, res00);
                res00 = SMLAD(val2, val0, res00);
                res01 = SMLAD(val5, val4, res01);
                res01 = SMLAD(val2, val1, res01);
            }

            // Left-over accumulations
            for (; rhs_cols_idx < rhs_cols; ++rhs_cols_idx)
            {
                int8_t rhs_value0 = rhs_ptr[0];
                int8_t rhs_value1 = rhs_ptr[rhs_cols];
                int8_t lhs_value = lhs_ptr[0];

                res00 += lhs_value * rhs_value0;
                res01 += lhs_value * rhs_value1;

                ++rhs_ptr;
                ++lhs_ptr;
            }

            // Quantize down
            res00 = arm_nn_requantize(res00, dst_multipliers[rhs_rows_idx], dst_shifts[rhs_rows_idx]);
            res01 = arm_nn_requantize(res01, dst_multipliers[rhs_rows_idx + 1], dst_shifts[rhs_rows_idx + 1]);

            // Add offset
            res00 += dst_offset;
            res01 += dst_offset;

            // Clamp the result
            res00 = MAX(res00, activation_min);
            res00 = MIN(res00, activation_max);
            res01 = MAX(res01, activation_min);
            res01 = MIN(res01, activation_max);

            dst_ptr[0] = (int8_t)res00;
            dst_ptr[1] = (int8_t)res01;
        }

        rhs += 2 * rhs_cols;
        dst += 2;
    }

    if (rhs_rows % 2)
    {
        const int8_t *lhs_ptr = &lhs[0];
        int8_t *dst_ptr = &dst[0];

        for (int32_t lhs_rows_idx = 0; lhs_rows_idx < lhs_rows; ++lhs_rows_idx)
        {
            const int8_t *rhs_ptr = &rhs[0];
            int32_t res00 = 0;
            if (bias)
            {
                res00 = bias[rhs_rows - 1];
            }

            for (int32_t rhs_cols_idx = 0; rhs_cols_idx < rhs_cols; ++rhs_cols_idx)
            {
                int32_t rhs_value = rhs_ptr[0];
                int32_t lhs_value = lhs_ptr[0] + lhs_offset;

                res00 += lhs_value * rhs_value;

                ++rhs_ptr;
                ++lhs_ptr;
            }
            lhs_ptr -= rhs_cols;
            lhs_ptr += lhs_cols_offset;

            // Quantize down
            res00 = arm_nn_requantize(res00, dst_multipliers[rhs_rows - 1], dst_shifts[rhs_rows - 1]);

            // Add offset
            res00 += dst_offset;

            // Clamp the result
            res00 = MAX(res00, activation_min);
            res00 = MIN(res00, activation_max);

            dst_ptr[0] = (int8_t)res00;
            dst_ptr += rhs_rows;
        }
    }
#else

#if defined(TF_LITE_PICO_MULTICORE)

    const int32_t mid_range = (rhs_rows / 4) * 2;

    MatMulArgs shared_args;
    shared_args.lhs = lhs;
    shared_args.rhs = rhs;
    shared_args.bias = bias;
    shared_args.dst = dst;
    shared_args.dst_multipliers = dst_multipliers;
    shared_args.dst_shifts = dst_shifts;
    shared_args.lhs_rows = lhs_rows;
    shared_args.rhs_rows = rhs_rows;
    shared_args.rhs_cols = rhs_cols;
    shared_args.lhs_offset = lhs_offset;
    shared_args.dst_offset = dst_offset;
    shared_args.activation_min = activation_min;
    shared_args.activation_max = activation_max;
    shared_args.lhs_cols_offset = lhs_cols_offset;

    MatMulArgs core0_args = shared_args;
    core0_args.rhs_rows_start = 0;
    core0_args.rhs_rows_end = mid_range;

    MatMulArgs core1_args = shared_args;
    core1_args.rhs_rows_start = mid_range;
    core1_args.rhs_rows_end = rhs_rows - 1;

    // Start the second core working.
    g_core1_mat_mul_args = core1_args;

    multicore_reset_core1();
    multicore_launch_core1(core1_mat_mul_worker);

    // Do the processing on the first core.
    mat_mul_task(&core0_args);

    // A result of ARM_CMSIS_NN_SUCCESS means success. Blocks until core 1 is
    // done.
    const uint32_t core1_result = multicore_fifo_pop_blocking();
    if (core1_result != ARM_CMSIS_NN_SUCCESS) {
        return core1_result;
    }

    const int32_t rows_processed = (rhs_rows / 2) * 2;
    const int8_t* new_rhs = rhs + (rows_processed * rhs_cols);
    const int8_t* new_dst = dst + rows_processed;

    rhs = new_rhs;
    dst = new_dst;


#else
    for (int32_t rhs_rows_idx = 0; rhs_rows_idx <= (rhs_rows - 2); rhs_rows_idx += 2)
    {
        const int8_t *lhs_ptr = &lhs[0];
        int8_t *dst_ptr = &dst[0];

        int32_t lhs_offset_contribution0 = 0;
        int32_t lhs_offset_contribution1 = 0;

        for (int32_t x = 0; x < rhs_cols; ++x)
        {
            lhs_offset_contribution0 += rhs[x];
            lhs_offset_contribution1 += rhs[x + rhs_cols];
        }

        lhs_offset_contribution0 *= lhs_offset;
        lhs_offset_contribution1 *= lhs_offset;
        if (bias)
        {
            lhs_offset_contribution0 += bias[rhs_rows_idx];
            lhs_offset_contribution1 += bias[rhs_rows_idx + 1];
        }

        int32_t lhs_rows_idx = lhs_rows >> 1;

        while (lhs_rows_idx)
        {
            const int8_t *rhs_ptr = &rhs[0];

            int32_t res00 = lhs_offset_contribution0;
            int32_t res01 = lhs_offset_contribution1;
            int32_t res10 = lhs_offset_contribution0;
            int32_t res11 = lhs_offset_contribution1;

            for (int32_t rhs_cols_idx = rhs_cols; rhs_cols_idx != 0; rhs_cols_idx--)
            {
                int8_t rhs_value0 = rhs_ptr[0];
                int8_t rhs_value1 = rhs_ptr[rhs_cols];
                int8_t lhs_value = lhs_ptr[0];

                res00 += lhs_value * rhs_value0;
                res01 += lhs_value * rhs_value1;

                lhs_value = lhs_ptr[lhs_cols_offset];
                res10 += lhs_value * rhs_value0;
                res11 += lhs_value * rhs_value1;

                ++rhs_ptr;
                ++lhs_ptr;
            }

            // Quantize down
            res00 = arm_nn_requantize(res00, dst_multipliers[rhs_rows_idx], dst_shifts[rhs_rows_idx]);
            res01 = arm_nn_requantize(res01, dst_multipliers[rhs_rows_idx + 1], dst_shifts[rhs_rows_idx + 1]);
            res10 = arm_nn_requantize(res10, dst_multipliers[rhs_rows_idx], dst_shifts[rhs_rows_idx]);
            res11 = arm_nn_requantize(res11, dst_multipliers[rhs_rows_idx + 1], dst_shifts[rhs_rows_idx + 1]);

            // Add offset
            res00 += dst_offset;
            res01 += dst_offset;
            res10 += dst_offset;
            res11 += dst_offset;

            // Clamp the result
            res00 = MAX(res00, activation_min);
            res00 = MIN(res00, activation_max);
            res01 = MAX(res01, activation_min);
            res01 = MIN(res01, activation_max);
            res10 = MAX(res10, activation_min);
            res10 = MIN(res10, activation_max);
            res11 = MAX(res11, activation_min);
            res11 = MIN(res11, activation_max);

            dst_ptr[0] = (int8_t)res00;
            dst_ptr[1] = (int8_t)res01;
            dst_ptr += rhs_rows;
            dst_ptr[0] = (int8_t)res10;
            dst_ptr[1] = (int8_t)res11;
            dst_ptr += rhs_rows;

            lhs_ptr -= rhs_cols;
            lhs_ptr += 2 * lhs_cols_offset;

            lhs_rows_idx--;
        }

        // Left-over rows
        if (lhs_rows % 2)
        {
            const int8_t *rhs_ptr = &rhs[0];

            int32_t res00 = lhs_offset_contribution0;
            int32_t res01 = lhs_offset_contribution1;

            for (int32_t rhs_cols_idx = rhs_cols; rhs_cols_idx != 0; rhs_cols_idx--)
            {
                int8_t rhs_value0 = rhs_ptr[0];
                int8_t rhs_value1 = rhs_ptr[rhs_cols];
                int8_t lhs_value = lhs_ptr[0];

                res00 += lhs_value * rhs_value0;
                res01 += lhs_value * rhs_value1;

                ++rhs_ptr;
                ++lhs_ptr;
            }

            // Quantize down
            res00 = arm_nn_requantize(res00, dst_multipliers[rhs_rows_idx], dst_shifts[rhs_rows_idx]);
            res01 = arm_nn_requantize(res01, dst_multipliers[rhs_rows_idx + 1], dst_shifts[rhs_rows_idx + 1]);

            // Add offset
            res00 += dst_offset;
            res01 += dst_offset;

            // Clamp the result
            res00 = MAX(res00, activation_min);
            res00 = MIN(res00, activation_max);
            res01 = MAX(res01, activation_min);
            res01 = MIN(res01, activation_max);

            dst_ptr[0] = (int8_t)res00;
            dst_ptr[1] = (int8_t)res01;
        }

        rhs += 2 * rhs_cols;
        dst += 2;
    }
#endif

    if (rhs_rows % 2)
    {
        const int8_t *lhs_ptr = &lhs[0];
        int8_t *dst_ptr = &dst[0];

        for (int32_t lhs_rows_idx = 0; lhs_rows_idx < lhs_rows; ++lhs_rows_idx)
        {
            const int8_t *rhs_ptr = &rhs[0];
            int32_t res00 = 0;
            if (bias)
            {
                res00 = bias[rhs_rows - 1];
            }

            for (int32_t rhs_cols_idx = rhs_cols; rhs_cols_idx != 0; rhs_cols_idx--)
            {
                int32_t rhs_value = rhs_ptr[0];
                int32_t lhs_value = lhs_ptr[0] + lhs_offset;

                res00 += lhs_value * rhs_value;

                ++rhs_ptr;
                ++lhs_ptr;
            }
            lhs_ptr -= rhs_cols;
            lhs_ptr += lhs_cols_offset;

            // Quantize down
            res00 = arm_nn_requantize(res00, dst_multipliers[rhs_rows - 1], dst_shifts[rhs_rows - 1]);

            // Add offset
            res00 += dst_offset;

            // Clamp the result
            res00 = MAX(res00, activation_min);
            res00 = MIN(res00, activation_max);

            dst_ptr[0] = (int8_t)res00;
            dst_ptr += rhs_rows;
        }
    }
#endif

    return ARM_CMSIS_NN_SUCCESS;
}

/**
 * @} end of Doxygen group
 */
