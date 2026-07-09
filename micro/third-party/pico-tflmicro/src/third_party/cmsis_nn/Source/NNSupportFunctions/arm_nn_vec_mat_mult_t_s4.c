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
 * Title:        arm_nn_vec_mat_mult_t_s4
 * Description:  s4 vector by matrix (transposed) multiplication
 *
 * $Date:        26 April 2024
 * $Revision:    V.2.0.0
 *
 * Target :  Arm(R) M-Profile Architecture
 *
 * -------------------------------------------------------------------- */

#include "third_party/cmsis_nn/Include/arm_nnsupportfunctions.h"
/**
 */

/**
 * @defgroup supportFC Fully Connected
 *
 * Support functions for Fully Connected
 *
 */

/**
 * @addtogroup supportFC
 * @{
 */

/*
 * s4 vector(lhs) by matrix (transposed) multiplication
 *
 * Refer header file for details.
 *
 */
arm_cmsis_nn_status arm_nn_vec_mat_mult_t_s4(const int8_t *lhs,
                                             const int8_t *packed_rhs,
                                             const int32_t *bias,
                                             int8_t *dst,
                                             const int32_t lhs_offset,
                                             const int32_t dst_offset,
                                             const int32_t dst_multiplier,
                                             const int32_t dst_shift,
                                             const int32_t rhs_cols,
                                             const int32_t rhs_rows,
                                             const int32_t activation_min,
                                             const int32_t activation_max)
{
    const int32_t row_loop_cnt = rhs_rows / 4;
    const int rhs_offset = rhs_cols * row_loop_cnt;
    const int8_t *rhs_ptr = &packed_rhs[0];

#if defined(ARM_MATH_MVEI)
    const int rhs_cols_offset = rhs_cols % 16;
#else
    const int rhs_cols_offset = rhs_cols;
#endif

#if defined(ARM_MATH_DSP)
    const int16_t lhs_offset_s16 = (int16_t)lhs_offset;
    const uint32_t lhs_offset_s16x2 = PKHBT(lhs_offset_s16, lhs_offset_s16, 16);
#endif

    int32_t spillover0, spillover1;

#if defined(ARM_MATH_MVEI)
    const mve_pred16_t lower_nibble_mask = 21845; // 0101010101010101
    const uint8x16_t gather_offset = {0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7};
    const int32_t col_loop_cnt = rhs_cols >> 5;
    const int I6_elements_spill = rhs_cols & 0x10;

    for (int32_t i_row_loop_cnt = 0; i_row_loop_cnt < row_loop_cnt; ++i_row_loop_cnt)
    {
        const uint32x4_t scatter_offset = {0, 1, 2 * row_loop_cnt, 2 * row_loop_cnt + 1};
        const int8_t *lhs_ptr = &lhs[0];

        mve_pred16_t rmdr_mask = vctp8q(rhs_cols_offset);

        int32_t acc0 = 0;
        int32_t acc2 = 0;

        int32_t rhs_sum_0 = 0;
        int32_t rhs_sum_2 = 0;

        if (bias)
        {
            acc0 += *bias;
            acc2 += bias[2 * row_loop_cnt];
            ++bias;
        }

        for (int i = 0; i < col_loop_cnt; i++)
        {
            const int8x16x2_t inputx2 = vld2q_s8(lhs_ptr);
            const int8x16_t ker_0 = vldrbq_s8(rhs_ptr);

            int8x16_t ker_low_0 = vrshlq_n_s8(ker_0, 4);
            ker_low_0 = vshrq_n_s8(ker_low_0, 4);
            int8x16_t ker_high_0 = vshrq_n_s8(ker_0, 4);

            rhs_sum_0 = vaddvaq_s8(rhs_sum_0, ker_low_0);
            rhs_sum_0 = vaddvaq_s8(rhs_sum_0, ker_high_0);
            acc0 = vmladavaq_s8(acc0, ker_low_0, inputx2.val[0]);
            acc0 = vmladavaq_s8(acc0, ker_high_0, inputx2.val[1]);

            const int8x16_t ker_1 = vldrbq_s8(&rhs_ptr[rhs_offset]);
            int8x16_t ker_low_1 = vrshlq_n_s8(ker_1, 4);
            ker_low_1 = vshrq_n_s8(ker_low_1, 4);
            int8x16_t ker_high_1 = vshrq_n_s8(ker_1, 4);

            rhs_sum_2 = vaddvaq_s8(rhs_sum_2, ker_low_1);
            rhs_sum_2 = vaddvaq_s8(rhs_sum_2, ker_high_1);
            acc2 = vmladavaq_s8(acc2, ker_low_1, inputx2.val[0]);
            acc2 = vmladavaq_s8(acc2, ker_high_1, inputx2.val[1]);

            lhs_ptr += 32;
            rhs_ptr += 16;
        }

        if (I6_elements_spill)
        {
            const int8x16_t input = vldrbq_s8(lhs_ptr);

            int8x16_t ker_0 = vldrbq_gather_offset_s8(rhs_ptr, gather_offset);
            ker_0 = vrshlq_m_n_s8(ker_0, 4, lower_nibble_mask);
            ker_0 = vshrq_n_s8(ker_0, 4);

            rhs_sum_0 = vaddvaq_s8(rhs_sum_0, ker_0);
            acc0 = vmladavaq_s8(acc0, ker_0, input);

            int8x16_t ker_1 = vldrbq_gather_offset_s8(&rhs_ptr[rhs_offset], gather_offset);
            ker_1 = vrshlq_m_n_s8(ker_1, 4, lower_nibble_mask);
            ker_1 = vshrq_n_s8(ker_1, 4);

            rhs_sum_2 = vaddvaq_s8(rhs_sum_2, ker_1);
            acc2 = vmladavaq_s8(acc2, ker_1, input);

            lhs_ptr += 16;
            rhs_ptr += 8;
        }

        if (rmdr_mask)
        {
            const int8x16_t input = vldrbq_z_s8(lhs_ptr, rmdr_mask);

            int8x16_t ker_0 = vldrbq_gather_offset_z_s8(rhs_ptr, gather_offset, rmdr_mask);
            ker_0 = vrshlq_m_n_s8(ker_0, 4, lower_nibble_mask);
            ker_0 = vshrq_n_s8(ker_0, 4);

            rhs_sum_0 = vaddvaq_s8(rhs_sum_0, ker_0);
            acc0 = vmladavaq_s8(acc0, ker_0, input);

            int8x16_t ker_1 = vldrbq_gather_offset_z_s8(&rhs_ptr[rhs_offset], gather_offset, rmdr_mask);
            ker_1 = vrshlq_m_n_s8(ker_1, 4, lower_nibble_mask);
            ker_1 = vshrq_n_s8(ker_1, 4);

            rhs_sum_2 = vaddvaq_s8(rhs_sum_2, ker_1);
            acc2 = vmladavaq_s8(acc2, ker_1, input);

            rhs_ptr += rhs_cols_offset >> 1;
        }

        if (rhs_cols & 1)
        {
            const int32_t rhs_high0 = rhs_ptr[0] >> 4;
            const int32_t rhs_high1 = rhs_ptr[rhs_offset] >> 4;

            lhs_ptr = &lhs[0];
            const int32_t lhs_high = (int8_t)lhs_ptr[0] + lhs_offset;

            spillover0 = lhs_high * rhs_high0;
            spillover1 = lhs_high * rhs_high1;

            rmdr_mask >>= 1;

            ++lhs_ptr;
            ++rhs_ptr;
        }
        else
        {
            spillover0 = 0;
            spillover1 = 0;
            lhs_ptr = &lhs[0];
        }

        int32_t acc1 = spillover0;
        int32_t acc3 = spillover1;

        if (bias)
        {
            acc1 += *bias;
            acc3 += bias[2 * row_loop_cnt];
            ++bias;
        }

        int32_t rhs_sum_1 = 0;
        int32_t rhs_sum_3 = 0;

        for (int i = 0; i < col_loop_cnt; i++)
        {
            const int8x16x2_t inputx2 = vld2q_s8(lhs_ptr);
            int8x16_t ker_0 = vldrbq_s8(rhs_ptr);

            int8x16_t ker_low_0 = vrshlq_n_s8(ker_0, 4);
            ker_low_0 = vshrq_n_s8(ker_low_0, 4);
            int8x16_t ker_high_0 = vshrq_n_s8(ker_0, 4);

            rhs_sum_1 = vaddvaq_s8(rhs_sum_1, ker_low_0);
            rhs_sum_1 = vaddvaq_s8(rhs_sum_1, ker_high_0);
            acc1 = vmladavaq_s8(acc1, ker_low_0, inputx2.val[0]);
            acc1 = vmladavaq_s8(acc1, ker_high_0, inputx2.val[1]);

            int8x16_t ker_1 = vldrbq_s8(&rhs_ptr[rhs_offset]);
            int8x16_t ker_low_1 = vrshlq_n_s8(ker_1, 4);
            ker_low_1 = vshrq_n_s8(ker_low_1, 4);
            int8x16_t ker_high_1 = vshrq_n_s8(ker_1, 4);

            rhs_sum_3 = vaddvaq_s8(rhs_sum_3, ker_low_1);
            rhs_sum_3 = vaddvaq_s8(rhs_sum_3, ker_high_1);
            acc3 = vmladavaq_s8(acc3, ker_low_1, inputx2.val[0]);
            acc3 = vmladavaq_s8(acc3, ker_high_1, inputx2.val[1]);

            lhs_ptr += 32;
            rhs_ptr += 16;
        }

        if (I6_elements_spill)
        {
            const int8x16_t input = vldrbq_s8(lhs_ptr);

            int8x16_t ker_0 = vldrbq_gather_offset_s8(rhs_ptr, gather_offset);
            ker_0 = vrshlq_m_n_s8(ker_0, 4, lower_nibble_mask);
            ker_0 = vshrq_n_s8(ker_0, 4);

            rhs_sum_1 = vaddvaq_s8(rhs_sum_1, ker_0);
            acc1 = vmladavaq_s8(acc1, ker_0, input);

            int8x16_t ker_1 = vldrbq_gather_offset_s8(&rhs_ptr[rhs_offset], gather_offset);
            ker_1 = vrshlq_m_n_s8(ker_1, 4, lower_nibble_mask);
            ker_1 = vshrq_n_s8(ker_1, 4);

            rhs_sum_3 = vaddvaq_s8(rhs_sum_3, ker_1);
            acc3 = vmladavaq_s8(acc3, ker_1, input);

            lhs_ptr += 16;
            rhs_ptr += 8;
        }

        if (rmdr_mask)
        {
            const int8x16_t input = vldrbq_z_s8(lhs_ptr, rmdr_mask);

            int8x16_t ker_0 = vldrbq_gather_offset_z_s8(rhs_ptr, gather_offset, rmdr_mask);
            ker_0 = vrshlq_m_n_s8(ker_0, 4, lower_nibble_mask);
            ker_0 = vshrq_n_s8(ker_0, 4);

            rhs_sum_1 = vaddvaq_s8(rhs_sum_1, ker_0);
            acc1 = vmladavaq_s8(acc1, ker_0, input);

            int8x16_t ker_1 = vldrbq_gather_offset_z_s8(&rhs_ptr[rhs_offset], gather_offset, rmdr_mask);
            ker_1 = vrshlq_m_n_s8(ker_1, 4, lower_nibble_mask);
            ker_1 = vshrq_n_s8(ker_1, 4);

            rhs_sum_3 = vaddvaq_s8(rhs_sum_3, ker_1);
            acc3 = vmladavaq_s8(acc3, ker_1, input);

            rhs_ptr += rhs_cols_offset >> 1;
        }

        int32x4_t acc = {acc0, acc1, acc2, acc3};

        const int32x4_t rhs_sum = {rhs_sum_0, rhs_sum_1, rhs_sum_2, rhs_sum_3};
        acc += vdupq_n_s32(lhs_offset) * rhs_sum;

        acc = arm_requantize_mve(acc, dst_multiplier, dst_shift);
        acc = vaddq_s32(acc, vdupq_n_s32(dst_offset));
        acc = vmaxq_s32(acc, vdupq_n_s32(activation_min));
        acc = vminq_s32(acc, vdupq_n_s32(activation_max));

        vstrbq_scatter_offset_s32(dst, scatter_offset, acc);

        dst += 2;
    }

#elif defined(ARM_MATH_DSP)

    for (int32_t i_row_loop_cnt = 0; i_row_loop_cnt < row_loop_cnt; ++i_row_loop_cnt)
    {
        const int8_t *lhs_ptr = &lhs[0];
        int32_t res0 = 0;
        int32_t res1 = 0;

        if (bias)
        {
            res0 += *bias;
            res1 += bias[2 * row_loop_cnt];
            ++bias;
        }

        for (int rhs_cols_idx = 0; rhs_cols_idx < (rhs_cols / 4); ++rhs_cols_idx)
        {
            int32_t lhs_high, rhs_high0, rhs_low0, lhs_low, rhs_high1, rhs_low1;

            read_and_pad_s4(rhs_ptr, &rhs_low0, &rhs_high0);
            read_and_pad_s4((const int8_t *)&rhs_ptr[rhs_offset], &rhs_low1, &rhs_high1);
            rhs_ptr += 2;

            lhs_high = arm_nn_read_s8x4_ia((const int8_t **)&lhs_ptr);
            lhs_low = SXTAB16(lhs_offset_s16x2, lhs_high);
            lhs_high = SXTAB16_RORn(lhs_offset_s16x2, lhs_high, 8);

            res0 = SMLAD(lhs_low, rhs_low0, res0);
            res0 = SMLAD(lhs_high, rhs_high0, res0);
            res1 = SMLAD(lhs_low, rhs_low1, res1);
            res1 = SMLAD(lhs_high, rhs_high1, res1);
        }

        if (((rhs_cols % 4) == 2) || ((rhs_cols % 4) == 3))
        {
            const int32_t rhs_value0 = rhs_ptr[0];
            const int32_t lower0 = (int8_t)(rhs_value0 << 4) >> 4;
            const int32_t higher0 = rhs_value0 >> 4;

            const int32_t rhs_value1 = rhs_ptr[rhs_offset];
            const int32_t lower1 = (int8_t)(rhs_value1 << 4) >> 4;
            const int32_t higher1 = rhs_value1 >> 4;

            const int32_t lhs_value_0 = lhs_ptr[0] + lhs_offset;
            const int32_t lhs_value_1 = lhs_ptr[1] + lhs_offset;

            res0 += lhs_value_0 * lower0;
            res0 += lhs_value_1 * higher0;
            res1 += lhs_value_0 * lower1;
            res1 += lhs_value_1 * higher1;

            ++rhs_ptr;
            lhs_ptr += 2;
        }

        if (rhs_cols % 2 == 1)
        {
            const int32_t rhs_low0 = (int8_t)(rhs_ptr[0] << 4) >> 4;
            const int32_t rhs_high0 = rhs_ptr[0] >> 4;
            const int32_t rhs_low1 = (int8_t)(rhs_ptr[rhs_offset] << 4) >> 4;
            const int32_t rhs_high1 = rhs_ptr[rhs_offset] >> 4;

            const int32_t lhs_low = (int8_t)lhs_ptr[0] + lhs_offset;
            lhs_ptr = &lhs[0];
            const int32_t lhs_high = (int8_t)lhs_ptr[0] + lhs_offset;
            ++lhs_ptr;

            res0 += lhs_low * rhs_low0;
            spillover0 = lhs_high * rhs_high0;
            res1 += lhs_low * rhs_low1;
            spillover1 = lhs_high * rhs_high1;

            ++rhs_ptr;
        }
        else
        {
            spillover0 = 0;
            spillover1 = 0;
            lhs_ptr = &lhs[0];
        }

        // Quantize down
        res0 = arm_nn_requantize(res0, dst_multiplier, dst_shift);
        res1 = arm_nn_requantize(res1, dst_multiplier, dst_shift);

        // Add offset
        res0 += dst_offset;
        res1 += dst_offset;

        // Clamp the result
        res0 = MAX(res0, activation_min);
        res0 = MIN(res0, activation_max);
        res1 = MAX(res1, activation_min);
        res1 = MIN(res1, activation_max);

        *dst = (int8_t)res0;
        *(dst + 2 * row_loop_cnt) = (int8_t)res1;
        dst++;

        res0 = spillover0;
        res1 = spillover1;

        if (bias)
        {
            res0 += *bias;
            res1 += bias[2 * row_loop_cnt];
            ++bias;
        }

        for (int rhs_cols_idx = 0; rhs_cols_idx < rhs_cols / 4; ++rhs_cols_idx)
        {
            int32_t lhs_high, rhs_high0, rhs_low0, lhs_low, rhs_high1, rhs_low1;

            read_and_pad_s4(rhs_ptr, &rhs_low0, &rhs_high0);
            read_and_pad_s4((const int8_t *)&rhs_ptr[rhs_offset], &rhs_low1, &rhs_high1);
            rhs_ptr += 2;

            lhs_high = arm_nn_read_s8x4_ia((const int8_t **)&lhs_ptr);
            lhs_low = SXTAB16(lhs_offset_s16x2, lhs_high);
            lhs_high = SXTAB16_RORn(lhs_offset_s16x2, lhs_high, 8);

            res0 = SMLAD(lhs_low, rhs_low0, res0);
            res0 = SMLAD(lhs_high, rhs_high0, res0);
            res1 = SMLAD(lhs_low, rhs_low1, res1);
            res1 = SMLAD(lhs_high, rhs_high1, res1);
        }

        if (((rhs_cols % 4) == 2) || ((rhs_cols % 4) == 3))
        {
            const int32_t rhs_value0 = rhs_ptr[0];
            const int32_t lower0 = (int8_t)(rhs_value0 << 4) >> 4;
            const int32_t higher0 = rhs_value0 >> 4;

            const int32_t rhs_value1 = rhs_ptr[rhs_offset];
            const int32_t lower1 = (int8_t)(rhs_value1 << 4) >> 4;
            const int32_t higher1 = rhs_value1 >> 4;

            const int32_t lhs_value_0 = lhs_ptr[0] + lhs_offset;
            const int32_t lhs_value_1 = lhs_ptr[1] + lhs_offset;

            res0 += lhs_value_0 * lower0;
            res0 += lhs_value_1 * higher0;
            res1 += lhs_value_0 * lower1;
            res1 += lhs_value_1 * higher1;

            ++rhs_ptr;
            lhs_ptr += 2;
        }

        // Quantize down
        res0 = arm_nn_requantize(res0, dst_multiplier, dst_shift);
        res1 = arm_nn_requantize(res1, dst_multiplier, dst_shift);

        // Add offset
        res0 += dst_offset;
        res1 += dst_offset;

        // Clamp the result
        res0 = MAX(res0, activation_min);
        res0 = MIN(res0, activation_max);
        res1 = MAX(res1, activation_min);
        res1 = MIN(res1, activation_max);

        *dst = (int8_t)res0;

        *(dst + 2 * row_loop_cnt) = (int8_t)res1;
        dst++;
    }

#else

    for (int i_row_loop_cnt = 0; i_row_loop_cnt < row_loop_cnt; ++i_row_loop_cnt)
    {
        const int8_t *lhs_ptr = &lhs[0];

        int32_t res0 = 0;
        int32_t res1 = 0;

        if (bias)
        {
            res0 += *bias;
            res1 += bias[2 * row_loop_cnt];
            ++bias;
        }

        for (int32_t rhs_cols_idx = 0; rhs_cols_idx < rhs_cols / 2; ++rhs_cols_idx)
        {
            const int32_t rhs_low0 = (int8_t)(rhs_ptr[0] << 4) >> 4;
            const int32_t rhs_high0 = rhs_ptr[0] >> 4;
            const int32_t rhs_low1 = (int8_t)(rhs_ptr[rhs_offset] << 4) >> 4;
            const int32_t rhs_high1 = rhs_ptr[rhs_offset] >> 4;

            const int32_t lhs_low = (int8_t)lhs_ptr[0] + lhs_offset;
            const int32_t lhs_high = (int8_t)lhs_ptr[1] + lhs_offset;

            res0 += lhs_low * rhs_low0;
            res0 += lhs_high * rhs_high0;
            res1 += lhs_low * rhs_low1;
            res1 += lhs_high * rhs_high1;

            ++rhs_ptr;
            lhs_ptr += 2;
        }

        if (rhs_cols % 2 == 1)
        {
            const int32_t rhs_low0 = (int8_t)(rhs_ptr[0] << 4) >> 4;
            const int32_t rhs_high0 = rhs_ptr[0] >> 4;
            const int32_t rhs_low1 = (int8_t)(rhs_ptr[rhs_offset] << 4) >> 4;
            const int32_t rhs_high1 = rhs_ptr[rhs_offset] >> 4;

            const int32_t lhs_low = (int8_t)lhs_ptr[0] + lhs_offset;
            lhs_ptr = &lhs[0];
            const int32_t lhs_high = (int8_t)lhs_ptr[0] + lhs_offset;
            ++lhs_ptr;

            res0 += lhs_low * rhs_low0;
            spillover0 = lhs_high * rhs_high0;
            res1 += lhs_low * rhs_low1;
            spillover1 = lhs_high * rhs_high1;

            ++rhs_ptr;
        }
        else
        {
            spillover0 = 0;
            spillover1 = 0;
            lhs_ptr = &lhs[0];
        }

        // Quantize down
        res0 = arm_nn_requantize(res0, dst_multiplier, dst_shift);
        res1 = arm_nn_requantize(res1, dst_multiplier, dst_shift);

        // Add offset
        res0 += dst_offset;
        res1 += dst_offset;

        // Clamp the result
        res0 = MAX(res0, activation_min);
        res0 = MIN(res0, activation_max);
        res1 = MAX(res1, activation_min);
        res1 = MIN(res1, activation_max);

        *dst = (int8_t)res0;

        *(dst + 2 * row_loop_cnt) = (int8_t)res1;
        dst++;

        res0 = spillover0;
        res1 = spillover1;
        if (bias)
        {
            res0 += *bias;
            res1 += bias[2 * row_loop_cnt];
            ++bias;
        }

        for (int32_t rhs_cols_idx = 0; rhs_cols_idx < rhs_cols / 2; ++rhs_cols_idx)
        {
            const int32_t rhs_low0 = (int8_t)(rhs_ptr[0] << 4) >> 4;
            const int32_t rhs_high0 = rhs_ptr[0] >> 4;
            const int32_t rhs_low1 = (int8_t)(rhs_ptr[rhs_offset] << 4) >> 4;
            const int32_t rhs_high1 = rhs_ptr[rhs_offset] >> 4;

            const int32_t lhs_low = (int8_t)lhs_ptr[0] + lhs_offset;
            const int32_t lhs_high = (int8_t)lhs_ptr[1] + lhs_offset;

            res0 += lhs_low * rhs_low0;
            res0 += lhs_high * rhs_high0;
            res1 += lhs_low * rhs_low1;
            res1 += lhs_high * rhs_high1;

            ++rhs_ptr;
            lhs_ptr += 2;
        }

        // Quantize down
        res0 = arm_nn_requantize(res0, dst_multiplier, dst_shift);
        res1 = arm_nn_requantize(res1, dst_multiplier, dst_shift);

        // Add offset
        res0 += dst_offset;
        res1 += dst_offset;

        // Clamp the result
        res0 = MAX(res0, activation_min);
        res0 = MIN(res0, activation_max);
        res1 = MAX(res1, activation_min);
        res1 = MIN(res1, activation_max);

        *dst = (int8_t)res0;
        *(dst + 2 * row_loop_cnt) = (int8_t)res1;
        dst++;
    }

#endif

    const int8_t *lhs_ptr = &lhs[0];
    spillover0 = 0;

    for (int32_t i_row_loop_cnt = 0; i_row_loop_cnt < rhs_rows % 4; ++i_row_loop_cnt)
    {
        int32_t res0 = spillover0;
        if (bias)
        {
            res0 += bias[2 * row_loop_cnt];
            ++bias;
        }

#if defined(ARM_MATH_MVEI)
        int32_t rhs_sum_0 = 0;

        for (int i = 0; i < col_loop_cnt; i++)
        {
            const int8x16x2_t inputx2 = vld2q_s8(lhs_ptr);
            const int8x16_t ker_0 = vldrbq_s8(&rhs_ptr[rhs_offset]);

            int8x16_t ker_low_0 = vrshlq_n_s8(ker_0, 4);
            ker_low_0 = vshrq_n_s8(ker_low_0, 4);
            int8x16_t ker_high_0 = vshrq_n_s8(ker_0, 4);

            rhs_sum_0 = vaddvaq_s8(rhs_sum_0, ker_low_0);
            rhs_sum_0 = vaddvaq_s8(rhs_sum_0, ker_high_0);
            res0 = vmladavaq_s8(res0, ker_low_0, inputx2.val[0]);
            res0 = vmladavaq_s8(res0, ker_high_0, inputx2.val[1]);

            lhs_ptr += 32;
            rhs_ptr += 16;
        }

        if (I6_elements_spill)
        {
            const int8x16_t input = vldrbq_s8(lhs_ptr);

            int8x16_t ker_0 = vldrbq_gather_offset_s8(&rhs_ptr[rhs_offset], gather_offset);
            ker_0 = vrshlq_m_n_s8(ker_0, 4, lower_nibble_mask);
            ker_0 = vshrq_n_s8(ker_0, 4);

            rhs_sum_0 = vaddvaq_s8(rhs_sum_0, ker_0);
            res0 = vmladavaq_s8(res0, ker_0, input);

            lhs_ptr += 16;
            rhs_ptr += 8;
        }

        res0 += lhs_offset * rhs_sum_0;
#endif

#if defined(ARM_MATH_DSP)
        for (int32_t rhs_cols_idx = 0; rhs_cols_idx < rhs_cols_offset / 4; ++rhs_cols_idx)
        {
            int32_t lhs_high, rhs_high0, rhs_low0, lhs_low;

            read_and_pad_s4((const int8_t *)&rhs_ptr[rhs_offset], &rhs_high0, &rhs_low0);
            rhs_ptr += 2;

            lhs_high = arm_nn_read_s8x4_ia((const int8_t **)&lhs_ptr);
            lhs_low = SXTAB16(lhs_offset_s16x2, lhs_high);
            lhs_high = SXTAB16_RORn(lhs_offset_s16x2, lhs_high, 8);

            res0 = SMLAD(lhs_low, rhs_high0, res0);
            res0 = SMLAD(lhs_high, rhs_low0, res0);
        }

        if ((rhs_cols % 4) == 2 || (rhs_cols % 4 == 3))
        {
            const int32_t rhs_value0 = rhs_ptr[rhs_offset];
            const int32_t lower0 = (int8_t)(rhs_value0 << 4) >> 4;
            const int32_t higher0 = rhs_value0 >> 4;

            const int32_t lhs_value_0 = lhs_ptr[0] + lhs_offset;
            const int32_t lhs_value_1 = lhs_ptr[1] + lhs_offset;

            res0 += lhs_value_0 * lower0;
            res0 += lhs_value_1 * higher0;

            ++rhs_ptr;
            lhs_ptr += 2;
        }
#else
        for (int32_t rhs_cols_idx = 0; rhs_cols_idx < rhs_cols_offset / 2; ++rhs_cols_idx)
        {
            const int32_t rhs_low0 = (int8_t)(rhs_ptr[rhs_offset] << 4) >> 4;
            const int32_t rhs_high0 = rhs_ptr[rhs_offset] >> 4;

            const int32_t lhs_low = (int8_t)lhs_ptr[0] + lhs_offset;
            const int32_t lhs_high = (int8_t)lhs_ptr[1] + lhs_offset;

            res0 += lhs_low * rhs_low0;
            res0 += lhs_high * rhs_high0;

            ++rhs_ptr;
            lhs_ptr += 2;
        }
#endif

        if ((rhs_cols % 2 == 1) && (i_row_loop_cnt % 2 == 0))
        {
            const int32_t rhs_low0 = (int8_t)(rhs_ptr[rhs_offset] << 4) >> 4;
            const int32_t rhs_high0 = rhs_ptr[rhs_offset] >> 4;

            const int32_t lhs_low = (int8_t)lhs_ptr[0] + lhs_offset;
            lhs_ptr = &lhs[0];
            const int32_t lhs_high = (int8_t)lhs_ptr[0] + lhs_offset;
            ++lhs_ptr;

            res0 += lhs_low * rhs_low0;
            spillover0 = lhs_high * rhs_high0;
            ++rhs_ptr;
        }
        else
        {
            spillover0 = 0;
            lhs_ptr = &lhs[0];
        }

        // Quantize down
        res0 = arm_nn_requantize(res0, dst_multiplier, dst_shift);

        // Add offset
        res0 += dst_offset;

        // Clamp the result
        res0 = MAX(res0, activation_min);
        res0 = MIN(res0, activation_max);

        *(dst + 2 * row_loop_cnt) = (int8_t)res0;
        dst++;
    }

    return ARM_CMSIS_NN_SUCCESS;
}

/**
 * @} end of Doxygen group
 */
