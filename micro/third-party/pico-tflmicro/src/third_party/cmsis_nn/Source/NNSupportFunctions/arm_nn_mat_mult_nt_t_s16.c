/*
 * SPDX-FileCopyrightText: Copyright 2020-2024 Arm Limited and/or its affiliates <open-source-office@arm.com>
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
 * Title:        arm_nn_mat_mult_nt_t_s16
 * Description:  Matrix multiplication support function with the right-hand-side (rhs) matrix transposed
 *
 * $Date:        11 April 2024
 * $Revision:    V.1.0.0
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

/*
 * s16 matrix multiplication with the right-hand-side matrix transposed
 *
 * Refer header file for details.
 *
 */
arm_cmsis_nn_status arm_nn_mat_mult_nt_t_s16(const int16_t *lhs,
                                             const int8_t *rhs,
                                             const cmsis_nn_bias_data *bias_data,
                                             int16_t *dst,
                                             const int32_t *dst_multipliers,
                                             const int32_t *dst_shifts,
                                             const int32_t lhs_rows,
                                             const int32_t rhs_rows,
                                             const int32_t rhs_cols,
                                             const int32_t activation_min,
                                             const int32_t activation_max)
{
#if defined(ARM_MATH_MVEI)

    const uint32_t rhs_rows_offset = (uint32_t)rhs_rows * sizeof(int16_t);
    const uint32x4_t scatter_offset = {
        0, (uint32_t)rhs_rows_offset, (uint32_t)rhs_rows_offset * 2, (uint32_t)rhs_rows_offset * 3};

    const int64_t *bias_s64 = (const int64_t *)bias_data->data;
    const int32_t *bias_s32 = (const int32_t *)bias_data->data;
    const bool is_int32_bias = bias_data->is_int32_bias;

    const int32_t rhs_cols_fast = is_int32_bias ? rhs_cols : (rhs_cols > MAX_COL_COUNT ? MAX_COL_COUNT : rhs_cols);
    const int32_t rhs_cols_slow = rhs_cols - MAX_COL_COUNT;

    int i_items = 0;
    for (; i_items <= (lhs_rows - 4); i_items += 4)
    {
        for (int i = 0; i < rhs_rows; i++)
        {
            int32_t acc_n0 = 0;
            int32_t acc_n1 = 0;
            int32_t acc_n2 = 0;
            int32_t acc_n3 = 0;

            const int16_t *ip_row_0 = lhs;
            const int16_t *ip_row_1 = lhs + rhs_cols;
            const int16_t *ip_row_2 = lhs + (2 * rhs_cols);
            const int16_t *ip_row_3 = lhs + (3 * rhs_cols);
            const int8_t *col_base = rhs + i * rhs_cols;

    #if defined(ARM_MATH_AUTOVECTORIZE)
            for (int j = 0; j < rhs_cols_fast; j++)
            {
                int8_t col = col_base[j];
                acc_n0 += ip_row_0[j] * col;
                acc_n1 += ip_row_1[j] * col;
                acc_n2 += ip_row_2[j] * col;
                acc_n3 += ip_row_3[j] * col;
            }
    #else
            // Note: If operand initialization is moved around, use '&' constraint to
            // specify earlyclobber operands.
            __ASM volatile(" .p2align 2                              \n"
                           "   wlstp.16        lr, %[cnt], 1f        \n"
                           "   mov             %[out0], 0            \n"
                           "   mov             %[out1], 0            \n"
                           "   mov             %[out2], 0            \n"
                           "   mov             %[out3], 0            \n"
                           "   vldrb.s16       q0, [%[col]], #8      \n"
                           "2:                                       \n"
                           "   vldrh.u16       q1, [%[row0]], #16     \n"
                           "   vmlava.s16      %[out0], q0, q1       \n"
                           "   vldrh.u16       q2, [%[row1]], #16     \n"
                           "   vmlava.s16      %[out1], q0, q2       \n"
                           "   vldrh.u16       q3, [%[row2]], #16     \n"
                           "   vmlava.s16      %[out2], q0, q3       \n"
                           "   vldrh.u16       q4, [%[row3]], #16     \n"
                           "   vmlava.s16      %[out3], q0, q4       \n"
                           "   vldrb.s16       q0, [%[col]], #8      \n"
                           "   letp            lr, 2b                \n"
                           "1:                                       \n"
                           : [col] "+l"(col_base),
                             [row0] "+l"(ip_row_0),
                             [row1] "+l"(ip_row_1),
                             [row2] "+l"(ip_row_2),
                             [row3] "+l"(ip_row_3),
                             [out0] "=Te"(acc_n0),
                             [out1] "=Te"(acc_n1),
                             [out2] "=Te"(acc_n2),
                             [out3] "=Te"(acc_n3)
                           : [cnt] "r"(rhs_cols_fast)
                           : "q0", "q1", "q2", "q3", "q4", "memory", "r14");
    #endif

            if (is_int32_bias)
            {
                int32x4_t result;

                if (bias_s32)
                {
                    acc_n0 += bias_s32[i];
                    acc_n1 += bias_s32[i];
                    acc_n2 += bias_s32[i];
                    acc_n3 += bias_s32[i];
                }

                int32x4_t res = {acc_n0, acc_n1, acc_n2, acc_n3};

                result = arm_requantize_mve(res, dst_multipliers[i], dst_shifts[i]);

                result = vmaxq_s32(result, vdupq_n_s32(activation_min));
                result = vminq_s32(result, vdupq_n_s32(activation_max));

                vstrhq_scatter_offset_s32(dst, scatter_offset, result);
            }
            else
            {
                int64_t acc_n0_s64 = acc_n0;
                int64_t acc_n1_s64 = acc_n1;
                int64_t acc_n2_s64 = acc_n2;
                int64_t acc_n3_s64 = acc_n3;

                if (rhs_cols > MAX_COL_COUNT)
                {
                    ip_row_0 = lhs + MAX_COL_COUNT;
                    ip_row_1 = lhs + rhs_cols + MAX_COL_COUNT;
                    ip_row_2 = lhs + (2 * rhs_cols) + MAX_COL_COUNT;
                    ip_row_3 = lhs + (3 * rhs_cols) + MAX_COL_COUNT;
                    col_base = rhs + i * rhs_cols + MAX_COL_COUNT;

                    for (int j = 0; j < rhs_cols_slow; j++)
                    {
                        int8_t col = col_base[j];
                        acc_n0_s64 += ip_row_0[j] * col;
                        acc_n1_s64 += ip_row_1[j] * col;
                        acc_n2_s64 += ip_row_2[j] * col;
                        acc_n3_s64 += ip_row_3[j] * col;
                    }
                }

                if (bias_s64)
                {
                    acc_n0_s64 += bias_s64[i];
                    acc_n1_s64 += bias_s64[i];
                    acc_n2_s64 += bias_s64[i];
                    acc_n3_s64 += bias_s64[i];
                }

                int32_t reduced_multiplier = REDUCE_MULTIPLIER(dst_multipliers[i]);
                int32_t shift = dst_shifts[i];

                acc_n0 = arm_nn_requantize_s64(acc_n0_s64, reduced_multiplier, shift);
                acc_n1 = arm_nn_requantize_s64(acc_n1_s64, reduced_multiplier, shift);
                acc_n2 = arm_nn_requantize_s64(acc_n2_s64, reduced_multiplier, shift);
                acc_n3 = arm_nn_requantize_s64(acc_n3_s64, reduced_multiplier, shift);
                int32x4_t res = {acc_n0, acc_n1, acc_n2, acc_n3};

                res = vmaxq_s32(res, vdupq_n_s32(activation_min));
                res = vminq_s32(res, vdupq_n_s32(activation_max));

                vstrhq_scatter_offset_s32(dst, scatter_offset, res);
            }
            dst++;
        }

        lhs += 4 * rhs_cols;
        dst += (3 * rhs_rows);
    }

    if (is_int32_bias)
    {

        for (; i_items < lhs_rows; i_items++)
        {
            int32_t acc[4];
            const int32_t *multipliers = dst_multipliers;
            const int32_t *shifts = dst_shifts;
            for (int i = 0; i < rhs_rows; i++)
            {
                int32_t acc_n0 = 0;
                const int16_t *ip_row_0 = lhs;
                const int8_t *col_base = rhs + i * rhs_cols;

    #if defined(ARM_MATH_AUTOVECTORIZE)
                for (int j = 0; j < rhs_cols; j++)
                {
                    int32_t col = col_base[j];
                    acc_n0 += ip_row_0[j] * col;
                }
    #else
                __ASM volatile(" .p2align 2                              \n"
                               "   wlstp.32        lr, %[cnt], 1f        \n"
                               "   mov             %[out0], 0            \n"
                               "2:                                       \n"
                               "   vldrb.s32       q0, [%[col]], #4      \n"
                               "   vldrh.s32       q1, [%[row0]], #8     \n"
                               "   vmlava.s32      %[out0], q0, q1       \n"
                               "   letp            lr, 2b                \n"
                               "1:                                       \n"
                               : [col] "+l"(col_base), [row0] "+l"(ip_row_0), [out0] "=Te"(acc_n0)
                               : [cnt] "r"(rhs_cols)
                               : "q0", "q1", "memory", "r14");
    #endif
                if (bias_s32)
                {
                    acc_n0 += bias_s32[i];
                }

                const int32_t index = i & 0x3;
                acc[index] = acc_n0;

                if (index == 3)
                {
                    int32x4_t res = vldrwq_s32(acc);
                    res = arm_requantize_mve_32x4(res, vldrwq_s32(multipliers), vldrwq_s32(shifts));
                    multipliers += 4;
                    shifts += 4;
                    res = vmaxq_s32(res, vdupq_n_s32(activation_min));
                    res = vminq_s32(res, vdupq_n_s32(activation_max));
                    vstrhq_s32(dst, res);
                    dst += 4;
                }
            }
            lhs += rhs_cols;

            const int32_t tail_rows = rhs_rows & 0x3;
            for (int i = 0; i < tail_rows; i++)
            {
                int32_t acc_n0 = acc[i];
                acc_n0 = arm_nn_requantize(acc_n0, multipliers[i], shifts[i]);
                acc_n0 = MAX(acc_n0, activation_min);
                acc_n0 = MIN(acc_n0, activation_max);
                *dst++ = (int16_t)acc_n0;
            }
        }
    }
    else
    {

        for (; i_items < lhs_rows; i_items++)
        {
            for (int i = 0; i < rhs_rows; i++)
            {
                int32_t acc_n0 = 0;
                int64_t acc_n0_s64 = 0;
                const int16_t *ip_row_0 = lhs;
                const int8_t *col_base = rhs + i * rhs_cols;

    #if defined(ARM_MATH_AUTOVECTORIZE)
                for (int j = 0; j < rhs_cols_fast; j++)
                {
                    int8_t col = col_base[j];
                    acc_n0 += ip_row_0[j] * col;
                }
    #else
                __ASM volatile(" .p2align 2                              \n"
                               "   wlstp.32        lr, %[cnt], 1f        \n"
                               "   mov             %[out0], 0            \n"
                               "2:                                       \n"
                               "   vldrb.s32       q0, [%[col]], #4      \n"
                               "   vldrh.s32       q1, [%[row0]], #8     \n"
                               "   vmlava.s32      %[out0], q0, q1       \n"
                               "   letp            lr, 2b                \n"
                               "1:                                       \n"
                               : [col] "+l"(col_base), [row0] "+l"(ip_row_0), [out0] "=Te"(acc_n0)
                               : [cnt] "r"(rhs_cols_fast)
                               : "q0", "q1", "memory", "r14");
    #endif

                acc_n0_s64 = acc_n0;

                if (rhs_cols > MAX_COL_COUNT)
                {
                    ip_row_0 = lhs + MAX_COL_COUNT;
                    col_base = rhs + i * rhs_cols + MAX_COL_COUNT;

                    for (int j = 0; j < rhs_cols_slow; j++)
                    {
                        int8_t col = col_base[j];
                        acc_n0_s64 += ip_row_0[j] * col;
                    }
                }

                if (bias_s64)
                {
                    acc_n0_s64 += bias_s64[i];
                }

                int32_t reduced_multiplier = REDUCE_MULTIPLIER(dst_multipliers[i]);
                int32_t shift = dst_shifts[i];

                acc_n0 = arm_nn_requantize_s64(acc_n0_s64, reduced_multiplier, shift);
                acc_n0 = MAX(acc_n0, activation_min);
                acc_n0 = MIN(acc_n0, activation_max);
                *dst++ = (int16_t)acc_n0;
            }
            lhs += rhs_cols;
        }
    }

#else
    (void)lhs;
    (void)rhs;
    (void)dst_multipliers;
    (void)dst_shifts;
    (void)dst;
    (void)activation_min;
    (void)activation_max;
    (void)bias_data;
    (void)lhs_rows;
    (void)lhs_rows;
    (void)rhs_rows;
    (void)rhs_cols;

    return ARM_CMSIS_NN_NO_IMPL_ERROR;
#endif
    return ARM_CMSIS_NN_SUCCESS;
}

/**
 * @} end of Doxygen group
 */
