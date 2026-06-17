/*
 * SPDX-FileCopyrightText: Copyright 2021-2023 Arm Limited and/or its affiliates <open-source-office@arm.com>
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
 * Title:        arm_nn_vec_mat_mult_t_svdf_s8
 * Description:  s8 vector by matrix (transposed) multiplication with
 *               s16 output. Targetted at SVDF operator.
 *
 * $Date:        28 March 2023
 * $Revision:    V.3.2.0
 *
 * Target :  Arm(R) M-Profile Architecture
 *
 * -------------------------------------------------------------------- */

#include "third_party/cmsis_nn/Include/arm_nnsupportfunctions.h"

/**
 * @ingroup groupSupport
 */

/**
 * @addtogroup supportFC
 * @{
 */

/*
 * s8 vector(lhs) by matrix (transposed) multiplication
 *
 * Refer header file for details.
 *
 */
arm_cmsis_nn_status arm_nn_vec_mat_mult_t_svdf_s8(const int8_t *lhs,
                                                  const int8_t *rhs,
                                                  int16_t *dst,
                                                  const int32_t lhs_offset,
                                                  const int32_t dst_offset,
                                                  const int32_t dst_multiplier,
                                                  const int32_t dst_shift,
                                                  const int32_t rhs_cols,
                                                  const int32_t rhs_rows,
                                                  const int32_t activation_min,
                                                  const int32_t activation_max)
{
    if (rhs_cols < 0 || (NN_Q31_MAX - rhs_cols) < 16 || dst_offset < 0)
    {
        return ARM_CMSIS_NN_ARG_ERROR;
    }

#if defined(ARM_MATH_MVEI)
    int32_t row_loop_cnt = rhs_rows / 3;

    for (int i_row_loop_cnt = 0; i_row_loop_cnt < row_loop_cnt; i_row_loop_cnt++)
    {
        int32_t acc_0 = 0;
        int32_t acc_1 = 0;
        int32_t acc_2 = 0;

        const int32_t col_loop_cnt = (rhs_cols + 15) / 16;

        const int8_t *lhs_vec = lhs;
        const int8_t *rhs_0 = rhs;
        const int8_t *rhs_1 = rhs + rhs_cols;
        const int8_t *rhs_2 = rhs + 2 * rhs_cols;

        int32_t rhs_sum_0 = 0;
        int32_t rhs_sum_1 = 0;
        int32_t rhs_sum_2 = 0;

        uint32_t col_cnt = (uint32_t)rhs_cols;

        for (int i = 0; i < col_loop_cnt; i++)
        {
            mve_pred16_t p = vctp8q(col_cnt);
            col_cnt -= 16;

            const int8x16_t input = vldrbq_z_s8(lhs_vec, p);

            const int8x16_t ker_0 = vldrbq_z_s8(rhs_0, p);
            rhs_sum_0 = vaddvaq_s8(rhs_sum_0, ker_0);
            acc_0 = vmladavaq_s8(acc_0, ker_0, input);

            const int8x16_t ker_1 = vldrbq_z_s8(rhs_1, p);
            rhs_sum_1 = vaddvaq_s8(rhs_sum_1, ker_1);
            acc_1 = vmladavaq_s8(acc_1, ker_1, input);

            const int8x16_t ker_2 = vldrbq_z_s8(rhs_2, p);
            rhs_sum_2 = vaddvaq_s8(rhs_sum_2, ker_2);
            acc_2 = vmladavaq_s8(acc_2, ker_2, input);

            lhs_vec += 16;
            rhs_0 += 16;
            rhs_1 += 16;
            rhs_2 += 16;
        }
        rhs += 3 * rhs_cols;

        int32x4_t acc = {acc_0, acc_1, acc_2, 0};
        const int32x4_t rhs_sum = {rhs_sum_0, rhs_sum_1, rhs_sum_2, 0};
        acc += vdupq_n_s32(lhs_offset) * rhs_sum;

        acc = arm_requantize_mve(acc, dst_multiplier, dst_shift);
        acc = vmaxq_s32(acc, vdupq_n_s32(activation_min));
        acc = vminq_s32(acc, vdupq_n_s32(activation_max));
        *(dst) = (int16_t)acc[0];
        *(dst + dst_offset) = (int16_t)acc[1];
        *(dst + 2 * dst_offset) = (int16_t)acc[2];
        dst += 3 * dst_offset;
    }

    const int loop_cnt = rhs_rows % 3;
    for (int i_row_loop_cnt = 0; i_row_loop_cnt < loop_cnt; i_row_loop_cnt++)
    {
        int32_t acc_0 = 0;
        const int32_t col_loop_cnt = (rhs_cols + 15) / 16;
        const int8_t *lhs_vec = lhs;
        const int8_t *rhs_0 = rhs;
        int32_t rhs_sum_0 = 0;
        uint32_t col_cnt = (uint32_t)rhs_cols;

        for (int i = 0; i < col_loop_cnt; i++)
        {
            mve_pred16_t p = vctp8q(col_cnt);
            col_cnt -= 16;
            const int8x16_t input = vldrbq_z_s8(lhs_vec, p);

            const int8x16_t ker_0 = vldrbq_z_s8(rhs_0, p);
            rhs_sum_0 = vaddvaq_s8(rhs_sum_0, ker_0);
            acc_0 = vmladavaq_s8(acc_0, ker_0, input);

            lhs_vec += 16;
            rhs_0 += 16;
        }
        rhs += rhs_cols;

        const int32_t offsets = rhs_sum_0 * lhs_offset;
        acc_0 = QADD(acc_0, offsets);
        acc_0 = arm_nn_requantize(acc_0, dst_multiplier, dst_shift);

        // Clamp the result
        acc_0 = MAX(acc_0, activation_min);
        *dst = (int16_t)MIN(acc_0, activation_max);
        dst += dst_offset;
    }

#elif defined(ARM_MATH_DSP)
    int32_t row_loop_cnt = rhs_rows / 2;

    const int16_t lhs_offset_s16 = lhs_offset;

    const uint32_t lhs_offset_s16x2 = PKHBT(lhs_offset_s16, lhs_offset_s16, 16);
    for (int32_t i = 0; i < row_loop_cnt; i++)
    {
        int32_t acc_0 = 0;
        int32_t acc_1 = 0;

        const int8_t *lhs_vec = lhs;
        const int8_t *rhs_0 = rhs;
        const int8_t *rhs_1 = rhs + rhs_cols;
        rhs += 2 * rhs_cols;

        int32_t rhs_cols_idx = 0;

        int32_t vec_0, vec_1, ker_0, ker_1;

    #if defined(__ARMCC_VERSION) && (__ARMCC_VERSION >= 6010050)
        #pragma clang loop unroll(disable)
    #endif
        for (; rhs_cols_idx <= (rhs_cols - 16); rhs_cols_idx += 16)
        {
            // 4 x MAC acc_0, acc1
            vec_0 = arm_nn_read_s8x4_ia(&lhs_vec);
            vec_1 = SXTAB16_RORn(lhs_offset_s16x2, (uint32_t)vec_0, 8);
            vec_0 = SXTAB16(lhs_offset_s16x2, vec_0);
            ker_0 = arm_nn_read_s8x4_ia(&rhs_0);
            ker_1 = SXTB16_RORn((uint32_t)ker_0, 8);
            ker_0 = SXTB16(ker_0);
            acc_0 = SMLAD(ker_1, vec_1, acc_0);
            acc_0 = SMLAD(ker_0, vec_0, acc_0);
            ker_0 = arm_nn_read_s8x4_ia(&rhs_1);
            ker_1 = SXTB16_RORn((uint32_t)ker_0, 8);
            ker_0 = SXTB16(ker_0);
            acc_1 = SMLAD(ker_1, vec_1, acc_1);
            acc_1 = SMLAD(ker_0, vec_0, acc_1);

            // 4 x MAC acc_0, acc1
            vec_0 = arm_nn_read_s8x4_ia(&lhs_vec);
            vec_1 = SXTAB16_RORn(lhs_offset_s16x2, (uint32_t)vec_0, 8);
            vec_0 = SXTAB16(lhs_offset_s16x2, vec_0);
            ker_0 = arm_nn_read_s8x4_ia(&rhs_0);
            ker_1 = SXTB16_RORn((uint32_t)ker_0, 8);
            ker_0 = SXTB16(ker_0);
            acc_0 = SMLAD(ker_1, vec_1, acc_0);
            acc_0 = SMLAD(ker_0, vec_0, acc_0);
            ker_0 = arm_nn_read_s8x4_ia(&rhs_1);
            ker_1 = SXTB16_RORn((uint32_t)ker_0, 8);
            ker_0 = SXTB16(ker_0);
            acc_1 = SMLAD(ker_1, vec_1, acc_1);
            acc_1 = SMLAD(ker_0, vec_0, acc_1);

            // 4 x MAC acc_0, acc1
            vec_0 = arm_nn_read_s8x4_ia(&lhs_vec);
            vec_1 = SXTAB16_RORn(lhs_offset_s16x2, (uint32_t)vec_0, 8);
            vec_0 = SXTAB16(lhs_offset_s16x2, vec_0);
            ker_0 = arm_nn_read_s8x4_ia(&rhs_0);
            ker_1 = SXTB16_RORn((uint32_t)ker_0, 8);
            ker_0 = SXTB16(ker_0);
            acc_0 = SMLAD(ker_1, vec_1, acc_0);
            acc_0 = SMLAD(ker_0, vec_0, acc_0);
            ker_0 = arm_nn_read_s8x4_ia(&rhs_1);
            ker_1 = SXTB16_RORn((uint32_t)ker_0, 8);
            ker_0 = SXTB16(ker_0);
            acc_1 = SMLAD(ker_1, vec_1, acc_1);
            acc_1 = SMLAD(ker_0, vec_0, acc_1);

            // 4 x MAC acc_0, acc1
            vec_0 = arm_nn_read_s8x4_ia(&lhs_vec);
            vec_1 = SXTAB16_RORn(lhs_offset_s16x2, (uint32_t)vec_0, 8);
            vec_0 = SXTAB16(lhs_offset_s16x2, vec_0);
            ker_0 = arm_nn_read_s8x4_ia(&rhs_0);
            ker_1 = SXTB16_RORn((uint32_t)ker_0, 8);
            ker_0 = SXTB16(ker_0);
            acc_0 = SMLAD(ker_1, vec_1, acc_0);
            acc_0 = SMLAD(ker_0, vec_0, acc_0);
            ker_0 = arm_nn_read_s8x4_ia(&rhs_1);
            ker_1 = SXTB16_RORn((uint32_t)ker_0, 8);
            ker_0 = SXTB16(ker_0);
            acc_1 = SMLAD(ker_1, vec_1, acc_1);
            acc_1 = SMLAD(ker_0, vec_0, acc_1);
        }

    #if defined(__ARMCC_VERSION) && (__ARMCC_VERSION >= 6010050)
        #pragma clang loop unroll(disable)
    #endif
        for (; rhs_cols_idx <= (rhs_cols - 4); rhs_cols_idx += 4)
        {
            vec_0 = arm_nn_read_s8x4_ia(&lhs_vec);
            vec_1 = SXTAB16_RORn(lhs_offset_s16x2, (uint32_t)vec_0, 8);

            vec_0 = SXTAB16(lhs_offset_s16x2, vec_0);

            ker_0 = arm_nn_read_s8x4_ia(&rhs_0);
            ker_1 = SXTB16_RORn((uint32_t)ker_0, 8);
            ker_0 = SXTB16(ker_0);

            acc_0 = SMLAD(ker_1, vec_1, acc_0);
            acc_0 = SMLAD(ker_0, vec_0, acc_0);

            ker_0 = arm_nn_read_s8x4_ia(&rhs_1);
            ker_1 = SXTB16_RORn((uint32_t)ker_0, 8);
            ker_0 = SXTB16(ker_0);

            acc_1 = SMLAD(ker_1, vec_1, acc_1);
            acc_1 = SMLAD(ker_0, vec_0, acc_1);
        }

    #if defined(__ARMCC_VERSION) && (__ARMCC_VERSION >= 6010050)
        #pragma clang loop unroll(disable)
    #endif
        for (; rhs_cols_idx < rhs_cols; ++rhs_cols_idx)
        {
            const int32_t lhs_temp = (*lhs_vec + lhs_offset);
            lhs_vec++;
            acc_0 += lhs_temp * (*rhs_0);
            rhs_0++;
            acc_1 += lhs_temp * (*rhs_1);
            rhs_1++;
        }

        acc_0 = arm_nn_requantize(acc_0, dst_multiplier, dst_shift);
        acc_1 = arm_nn_requantize(acc_1, dst_multiplier, dst_shift);

        // Clamp the result
        acc_0 = MAX(acc_0, activation_min);
        acc_0 = MIN(acc_0, activation_max);
        acc_1 = MAX(acc_1, activation_min);
        acc_1 = MIN(acc_1, activation_max);
        *dst = (int16_t)acc_0;
        *(dst + dst_offset) = (int16_t)acc_1;
        dst += 2 * dst_offset;
    }
    if (rhs_rows & 0x1)
    {
        int32_t acc_0 = 0;
        const int32_t col_loop_cnt = rhs_cols / 4;
        const int8_t *lhs_vec = lhs;
        const int8_t *rhs_0 = rhs;
        for (int i = col_loop_cnt; i != 0; i--)
        {
            int32_t vec_0 = arm_nn_read_s8x4_ia(&lhs_vec);
            int32_t vec_1 = SXTAB16(lhs_offset_s16x2, ROR((uint32_t)vec_0, 8));
            vec_0 = SXTAB16(lhs_offset_s16x2, vec_0);

            int32_t ker_0 = arm_nn_read_s8x4_ia(&rhs_0);
            int32_t ker_1 = SXTB16_RORn((uint32_t)ker_0, 8);
            ker_0 = SXTB16(ker_0);

            acc_0 = SMLAD(ker_1, vec_1, acc_0);
            acc_0 = SMLAD(ker_0, vec_0, acc_0);
        }
        for (int j = col_loop_cnt * 4; j < rhs_cols; j++)
        {
            const int32_t lhs_temp = (*lhs_vec + lhs_offset);
            lhs_vec++;
            acc_0 += lhs_temp * *rhs_0;
            rhs_0++;
        }
        acc_0 = arm_nn_requantize(acc_0, dst_multiplier, dst_shift);

        // Clamp the result
        acc_0 = MAX(acc_0, activation_min);
        acc_0 = MIN(acc_0, activation_max);
        *dst = (int16_t)acc_0;
        dst += dst_offset;
    }

#else

    int32_t row_loop_cnt = rhs_rows / 3;

    for (int i_row_loop_cnt = 0; i_row_loop_cnt < row_loop_cnt; i_row_loop_cnt++)
    {
        const int8_t *lhs_ptr = lhs;
        const int8_t *rhs_ptr_0 = &rhs[0];
        const int8_t *rhs_ptr_1 = &rhs[rhs_cols];
        const int8_t *rhs_ptr_2 = &rhs[rhs_cols * 2];

        int32_t res00 = 0;
        int32_t res01 = 0;
        int32_t res02 = 0;
        for (int32_t rhs_cols_idx = 0; rhs_cols_idx < rhs_cols; ++rhs_cols_idx)
        {
            const int32_t rhs_value0 = (int8_t)*rhs_ptr_0;
            const int32_t rhs_value1 = (int8_t)*rhs_ptr_1;
            const int32_t rhs_value2 = (int8_t)*rhs_ptr_2;
            const int32_t lhs_value = (int8_t)*lhs_ptr + lhs_offset;

            res00 += lhs_value * rhs_value0;
            res01 += lhs_value * rhs_value1;
            res02 += lhs_value * rhs_value2;

            ++rhs_ptr_0;
            ++rhs_ptr_1;
            ++rhs_ptr_2;
            ++lhs_ptr;
        }
        // Quantize down
        res00 = arm_nn_requantize(res00, dst_multiplier, dst_shift);
        res01 = arm_nn_requantize(res01, dst_multiplier, dst_shift);
        res02 = arm_nn_requantize(res02, dst_multiplier, dst_shift);

        // Clamp the result
        res00 = MAX(res00, activation_min);
        res00 = MIN(res00, activation_max);
        res01 = MAX(res01, activation_min);
        res01 = MIN(res01, activation_max);
        res02 = MAX(res02, activation_min);
        res02 = MIN(res02, activation_max);

        *dst = (int16_t)res00;
        *(dst + dst_offset) = (int16_t)res01;
        *(dst + 2 * dst_offset) = (int16_t)res02;
        dst += 3 * dst_offset;
        rhs += 3 * rhs_cols;
    }

    const int loop_cnt = rhs_rows % 3;

    for (int i_loop_cnt = 0; i_loop_cnt < loop_cnt; i_loop_cnt++)
    {
        const int8_t *lhs_ptr = &lhs[0];
        const int8_t *rhs_ptr = &rhs[0];

        int32_t res00 = 0;

        for (int32_t rhs_cols_idx = 0; rhs_cols_idx < rhs_cols; ++rhs_cols_idx)
        {
            int32_t rhs_value0 = (int8_t)rhs_ptr[0];
            int32_t lhs_value = (int8_t)lhs_ptr[0] + lhs_offset;

            res00 += lhs_value * rhs_value0;

            ++rhs_ptr;
            ++lhs_ptr;
        }

        // Quantize down
        res00 = arm_nn_requantize(res00, dst_multiplier, dst_shift);

        // Clamp the result
        res00 = MAX(res00, activation_min);
        res00 = MIN(res00, activation_max);

        *dst = (int16_t)res00;
        dst += dst_offset;
        rhs += rhs_cols;
    }
#endif

    return ARM_CMSIS_NN_SUCCESS;
}

/**
 * @} end of Doxygen group
 */
