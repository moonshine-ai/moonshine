/*
 * SPDX-FileCopyrightText: Copyright 2022-2024 Arm Limited and/or its affiliates <open-source-office.com>
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
 * Title:        arm_nn_vec_mat_mul_result_acc_s8_s16.c
 * Description:  Multiplies a matrix by a vector and accumulate with output.
 *
 * $Date:        19 January 2024
 * $Revision:    V.2.0.0
 *
 * Target :  Arm(R) M-Profile Architecture
 *
 * -------------------------------------------------------------------- */
#include "third_party/cmsis_nn/Include/arm_nnsupportfunctions.h"
/**
 * @ingroup groupSupport
 */

/**
 * @addtogroup supportLSTM
 * @{
 */

/*
 *  Refer to header file for details.
 */
arm_cmsis_nn_status arm_nn_vec_mat_mul_result_acc_s8_s16(const int8_t *lhs,
                                                         const int8_t *rhs,
                                                         const int32_t *effective_bias,
                                                         int16_t *dst,
                                                         const int32_t dst_multiplier,
                                                         const int32_t dst_shift,
                                                         const int32_t rhs_cols,
                                                         const int32_t rhs_rows,
                                                         const int32_t batches,
                                                         const int32_t batch_offset)
{

    for (int batch = 0; batch < batches; batch++)
    {
        const int8_t *rhs_ptr = &rhs[0];
        const int32_t *effective_bias_ptr = &effective_bias[0];

#if defined(ARM_MATH_MVEI)

        for (size_t row_loop_cnt = rhs_rows / 4; row_loop_cnt != 0; --row_loop_cnt)
        {
            const int32_t col_loop_cnt = (rhs_cols + 15) / 16;

            const int8_t *lhs_vec = lhs;
            const int8_t *rhs_0 = rhs_ptr;
            rhs_ptr += rhs_cols;
            const int8_t *rhs_1 = rhs_ptr;
            rhs_ptr += rhs_cols;
            const int8_t *rhs_2 = rhs_ptr;
            rhs_ptr += rhs_cols;
            const int8_t *rhs_3 = rhs_ptr;
            rhs_ptr += rhs_cols;

            int32_t acc_0 = *effective_bias_ptr++;
            int32_t acc_1 = *effective_bias_ptr++;
            int32_t acc_2 = *effective_bias_ptr++;
            int32_t acc_3 = *effective_bias_ptr++;

            uint32_t col_cnt = (uint32_t)rhs_cols;

            for (int i = 0; i < col_loop_cnt; i++)
            {
                mve_pred16_t p = vctp8q(col_cnt);
                col_cnt -= 16;

                const int8x16_t input = vldrbq_z_s8(lhs_vec, p);

                const int8x16_t ker_0 = vldrbq_z_s8(rhs_0, p);
                acc_0 = vmladavaq_s8(acc_0, ker_0, input);

                const int8x16_t ker_1 = vldrbq_z_s8(rhs_1, p);
                acc_1 = vmladavaq_s8(acc_1, ker_1, input);

                const int8x16_t ker_2 = vldrbq_z_s8(rhs_2, p);
                acc_2 = vmladavaq_s8(acc_2, ker_2, input);

                const int8x16_t ker_3 = vldrbq_z_s8(rhs_3, p);
                acc_3 = vmladavaq_s8(acc_3, ker_3, input);

                lhs_vec += 16;
                rhs_0 += 16;
                rhs_1 += 16;
                rhs_2 += 16;
                rhs_3 += 16;
            }

            int32x4_t acc = {acc_0, acc_1, acc_2, acc_3};

            acc = arm_requantize_mve(acc, dst_multiplier, dst_shift);
            acc = vaddq_s32(acc, vldrhq_s32(dst));

            acc = vmaxq_s32(acc, vdupq_n_s32(NN_Q15_MIN));
            acc = vminq_s32(acc, vdupq_n_s32(NN_Q15_MAX));

            vstrhq_s32(dst, acc);
            dst += 4;
        }

        for (size_t row_loop_cnt = rhs_rows % 4; row_loop_cnt != 0; --row_loop_cnt)
        {
            int32_t acc_0 = *effective_bias_ptr++;

            const int32_t col_loop_cnt = (rhs_cols + 15) / 16;
            const int8_t *lhs_vec = lhs;
            const int8_t *rhs_0 = rhs_ptr;
            uint32_t col_cnt = (uint32_t)rhs_cols;

            for (int i = 0; i < col_loop_cnt; i++)
            {
                mve_pred16_t p = vctp8q(col_cnt);
                col_cnt -= 16;
                const int8x16_t input = vldrbq_z_s8(lhs_vec, p);

                const int8x16_t ker_0 = vldrbq_z_s8(rhs_0, p);
                acc_0 = vmladavaq_s8(acc_0, ker_0, input);

                lhs_vec += 16;
                rhs_0 += 16;
            }
            rhs_ptr += rhs_cols;

            acc_0 = arm_nn_requantize(acc_0, dst_multiplier, dst_shift);
            acc_0 += *dst;

            // Clamp the result
            acc_0 = MAX(acc_0, NN_Q15_MIN);
            acc_0 = MIN(acc_0, NN_Q15_MAX);
            *dst++ = (int16_t)acc_0;
        }

#elif defined(ARM_MATH_DSP)

        for (int32_t row_loop_cnt = rhs_rows / 2; row_loop_cnt != 0; --row_loop_cnt)
        {
            int32_t acc_0 = *effective_bias_ptr++;
            int32_t acc_1 = *effective_bias_ptr++;

            const int32_t col_loop_cnt = rhs_cols / 4;

            const int8_t *lhs_vec = lhs;
            const int8_t *rhs_0 = rhs_ptr;
            rhs_ptr += rhs_cols;
            const int8_t *rhs_1 = rhs_ptr;
            rhs_ptr += rhs_cols;

            for (int j = col_loop_cnt; j != 0; j--)
            {
                int32_t vec_0 = arm_nn_read_s8x4_ia(&lhs_vec);
                int32_t vec_1 = SXTB16_RORn((uint32_t)vec_0, 8);
                vec_0 = SXTB16(vec_0);

                int32_t ker_0 = arm_nn_read_s8x4_ia(&rhs_0);
                int32_t ker_1 = SXTB16_RORn((uint32_t)ker_0, 8);
                ker_0 = SXTB16(ker_0);

                acc_0 = SMLAD(ker_1, vec_1, acc_0);
                acc_0 = SMLAD(ker_0, vec_0, acc_0);

                ker_0 = arm_nn_read_s8x4_ia(&rhs_1);
                ker_1 = SXTB16_RORn((uint32_t)ker_0, 8);
                ker_0 = SXTB16(ker_0);

                acc_1 = SMLAD(ker_1, vec_1, acc_1);
                acc_1 = SMLAD(ker_0, vec_0, acc_1);
            }

            for (int k = col_loop_cnt * 4; k < rhs_cols; k++)
            {
                const int32_t lhs_temp = (*lhs_vec);
                lhs_vec++;
                acc_0 += lhs_temp * (*rhs_0);
                rhs_0++;
                acc_1 += lhs_temp * (*rhs_1);
                rhs_1++;
            }

            acc_0 = arm_nn_requantize(acc_0, dst_multiplier, dst_shift);
            acc_1 = arm_nn_requantize(acc_1, dst_multiplier, dst_shift);

            // Add offset
            acc_0 += *dst;
            // Clamp the result
            acc_0 = MAX(acc_0, NN_Q15_MIN);
            acc_0 = MIN(acc_0, NN_Q15_MAX);
            *dst++ = (int16_t)acc_0;

            acc_1 += *dst;
            acc_1 = MAX(acc_1, NN_Q15_MIN);
            acc_1 = MIN(acc_1, NN_Q15_MAX);

            *dst++ = (int16_t)acc_1;
        }

        if (rhs_rows & 0x1)
        {
            int32_t acc_0 = *effective_bias_ptr++;
            const int32_t col_loop_cnt = rhs_cols / 4;

            const int8_t *lhs_vec = lhs;
            const int8_t *rhs_0 = rhs_ptr;

            for (int i = col_loop_cnt; i != 0; i--)
            {
                int32_t vec_0 = arm_nn_read_s8x4_ia(&lhs_vec);
                int32_t vec_1 = SXTB16_RORn((uint32_t)vec_0, 8);
                vec_0 = SXTB16(vec_0);

                int32_t ker_0 = arm_nn_read_s8x4_ia(&rhs_0);
                int32_t ker_1 = SXTB16_RORn((uint32_t)ker_0, 8);
                ker_0 = SXTB16(ker_0);

                acc_0 = SMLAD(ker_1, vec_1, acc_0);
                acc_0 = SMLAD(ker_0, vec_0, acc_0);
            }

            for (int j = col_loop_cnt * 4; j < rhs_cols; j++)
            {
                const int32_t lhs_temp = (*lhs_vec);
                lhs_vec++;
                acc_0 += lhs_temp * (*rhs_0);
                rhs_0++;
            }

            acc_0 = arm_nn_requantize(acc_0, dst_multiplier, dst_shift);

            // Accumulate
            acc_0 += dst[0];
            // Clamp the result
            acc_0 = MAX(acc_0, NN_Q15_MIN);
            acc_0 = MIN(acc_0, NN_Q15_MAX);
            *dst++ = (int16_t)acc_0;
        }

#else

        const int32_t row_loop_cnt = rhs_rows / 3;

        for (int i_row_loop_cnt = 0; i_row_loop_cnt < row_loop_cnt; i_row_loop_cnt++)
        {
            const int8_t *lhs_ptr = lhs;
            const int8_t *rhs_ptr_0 = rhs_ptr;
            rhs_ptr += rhs_cols;
            const int8_t *rhs_ptr_1 = rhs_ptr;
            rhs_ptr += rhs_cols;
            const int8_t *rhs_ptr_2 = rhs_ptr;
            rhs_ptr += rhs_cols;

            int32_t res00 = *effective_bias_ptr++;
            int32_t res01 = *effective_bias_ptr++;
            int32_t res02 = *effective_bias_ptr++;

            for (int32_t rhs_cols_idx = 0; rhs_cols_idx < rhs_cols; ++rhs_cols_idx)
            {
                const int32_t rhs_value0 = (int8_t)*rhs_ptr_0;
                const int32_t rhs_value1 = (int8_t)*rhs_ptr_1;
                const int32_t rhs_value2 = (int8_t)*rhs_ptr_2;
                const int32_t lhs_value = (int8_t)*lhs_ptr;

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

            // Add offset
            res00 += (int32_t)*dst;
            res00 = CLAMP(res00, NN_Q15_MAX, NN_Q15_MIN);
            *dst++ = (int16_t)res00;

            res01 += (int32_t)*dst;
            res01 = CLAMP(res01, NN_Q15_MAX, NN_Q15_MIN);
            *dst++ = (int16_t)res01;

            res02 += (int32_t)*dst;
            res02 = CLAMP(res02, NN_Q15_MAX, NN_Q15_MIN);
            *dst++ = (int16_t)res02;
        }

        const int loop_cnt = rhs_rows % 3;

        for (int i_loop_cnt = 0; i_loop_cnt < loop_cnt; i_loop_cnt++)
        {
            const int8_t *lhs_ptr = &lhs[0];
            const int8_t *rhs_ptr_0 = &rhs_ptr[0];

            int32_t res00 = *effective_bias_ptr++;

            for (int32_t rhs_cols_idx = 0; rhs_cols_idx < rhs_cols; ++rhs_cols_idx)
            {
                int32_t rhs_value0 = (int8_t)rhs_ptr_0[0];
                int32_t lhs_value = (int8_t)lhs_ptr[0];

                res00 += lhs_value * rhs_value0;

                ++rhs_ptr_0;
                ++lhs_ptr;
            }

            // Quantize down
            res00 = arm_nn_requantize(res00, dst_multiplier, dst_shift);

            // Accumulate
            res00 += (int32_t)dst[0];

            // Clamp the result
            res00 = CLAMP(res00, NN_Q15_MAX, NN_Q15_MIN);

            *dst++ = (int16_t)res00;
            rhs_ptr += rhs_cols;
        }
#endif

        lhs += rhs_cols * batch_offset;
    }

    return ARM_CMSIS_NN_SUCCESS;
}

/**
 * @} end of supportLSTM group
 */