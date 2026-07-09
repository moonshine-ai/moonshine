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
 * Title:        arm_nn_vec_mat_mul_result_acc_s16
 * Description:  s16 vector by matrix (transposed) multiplication
 *
 * $Date:        26 March 2023
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
 * @addtogroup supportFC
 * @{
 */

/*
 * s16 vector(lhs) by matrix (transposed) multiplication with result accumulation
 *
 * Refer header file for details.
 *
 */
arm_cmsis_nn_status arm_nn_vec_mat_mul_result_acc_s16(const int16_t *lhs,
                                                      const int8_t *rhs,
                                                      const int64_t *effective_bias,
                                                      int16_t *dst,
                                                      const int32_t dst_multiplier,
                                                      const int32_t dst_shift,
                                                      const int32_t rhs_cols,
                                                      const int32_t rhs_rows,
                                                      const int32_t batches,
                                                      const int32_t batch_offset)
{

    int32_t reduced_multiplier = REDUCE_MULTIPLIER(dst_multiplier);

    for (int batch = 0; batch < batches; batch++)
    {

        const int8_t *rhs_ptr = &rhs[0];
        const int64_t *effective_bias_ptr = &effective_bias[0];

#if defined(ARM_MATH_DSP)

        int32_t rhs_cols_fast = rhs_cols;

        if (rhs_cols > MAX_COL_COUNT)
        {
            rhs_cols_fast = MAX_COL_COUNT;
        }

    #if defined(ARM_MATH_MVEI)
        int32_t row_loop_cnt = rhs_rows / 4;
        const int32_t col_loop_cnt = (rhs_cols_fast + 7) / 8;

        for (int32_t i_row_loop_count = 0; i_row_loop_count < row_loop_cnt; i_row_loop_count++)
        {
            int32_t col_cnt = rhs_cols_fast;

            const int16_t *lhs_ptr = lhs;
            const int8_t *rhs_ptr_0 = rhs_ptr;
            const int8_t *rhs_ptr_1 = rhs_ptr + rhs_cols;
            const int8_t *rhs_ptr_2 = rhs_ptr + rhs_cols * 2;
            const int8_t *rhs_ptr_3 = rhs_ptr + rhs_cols * 3;

            int32_t result_0 = *effective_bias_ptr++;
            int32_t result_1 = *effective_bias_ptr++;
            int32_t result_2 = *effective_bias_ptr++;
            int32_t result_3 = *effective_bias_ptr++;

            for (int i_col_loop_cnt = 0; i_col_loop_cnt < col_loop_cnt; i_col_loop_cnt++)
            {
                mve_pred16_t pred = vctp16q(col_cnt);
                col_cnt -= 8;

                int16x8_t lhs_input = vldrhq_z_s16(lhs_ptr, pred);

                int16x8_t rhs_input_0 = vldrbq_z_s16(rhs_ptr_0, pred);
                int16x8_t rhs_input_1 = vldrbq_z_s16(rhs_ptr_1, pred);
                int16x8_t rhs_input_2 = vldrbq_z_s16(rhs_ptr_2, pred);
                int16x8_t rhs_input_3 = vldrbq_z_s16(rhs_ptr_3, pred);

                result_0 = vmladavaq_s16(result_0, lhs_input, rhs_input_0);
                result_1 = vmladavaq_s16(result_1, lhs_input, rhs_input_1);
                result_2 = vmladavaq_s16(result_2, lhs_input, rhs_input_2);
                result_3 = vmladavaq_s16(result_3, lhs_input, rhs_input_3);

                lhs_ptr += 8;

                rhs_ptr_0 += 8;
                rhs_ptr_1 += 8;
                rhs_ptr_2 += 8;
                rhs_ptr_3 += 8;
            }

            int64_t result_64_0 = result_0;
            int64_t result_64_1 = result_1;
            int64_t result_64_2 = result_2;
            int64_t result_64_3 = result_3;

            if (rhs_cols > MAX_COL_COUNT)
            {
                for (int i_rhs_cols = MAX_COL_COUNT; i_rhs_cols < rhs_cols; i_rhs_cols++)
                {
                    const int16_t lhs_temp = *lhs_ptr++;

                    result_64_0 += *rhs_ptr_0++ * lhs_temp;
                    result_64_1 += *rhs_ptr_1++ * lhs_temp;
                    result_64_2 += *rhs_ptr_2++ * lhs_temp;
                    result_64_3 += *rhs_ptr_3++ * lhs_temp;
                }
            }

            int32_t tmp;
            tmp = arm_nn_requantize_s64(result_64_0, reduced_multiplier, dst_shift);
            tmp += (int64_t)*dst;
            tmp = MAX(tmp, NN_Q15_MIN);
            tmp = MIN(tmp, NN_Q15_MAX);
            *dst++ = (int16_t)tmp;

            tmp = 0;
            tmp = arm_nn_requantize_s64(result_64_1, reduced_multiplier, dst_shift);
            tmp += (int64_t)*dst;
            tmp = MAX(tmp, NN_Q15_MIN);
            tmp = MIN(tmp, NN_Q15_MAX);
            *dst++ = (int16_t)tmp;

            tmp = 0;
            tmp = arm_nn_requantize_s64(result_64_2, reduced_multiplier, dst_shift);
            tmp += (int64_t)*dst;
            tmp = MAX(tmp, NN_Q15_MIN);
            tmp = MIN(tmp, NN_Q15_MAX);
            *dst++ = (int16_t)tmp;

            tmp = 0;
            tmp = arm_nn_requantize_s64(result_64_3, reduced_multiplier, dst_shift);
            tmp += (int64_t)*dst;
            tmp = MAX(tmp, NN_Q15_MIN);
            tmp = MIN(tmp, NN_Q15_MAX);
            *dst++ = (int16_t)tmp;

            rhs_ptr += 4 * rhs_cols;
        }

        for (int8_t rows_left = rhs_rows & 0x3; rows_left > 0; rows_left--)
        {
            int32_t result = *effective_bias_ptr++;

            const int16_t *lhs_ptr = lhs;
            const int8_t *rhs_ptr0 = rhs_ptr;

            int32_t col_cnt = (int32_t)rhs_cols_fast;

            for (int i_col_loop_cnt = 0; i_col_loop_cnt < col_loop_cnt; i_col_loop_cnt++)
            {
                mve_pred16_t pred = vctp16q(col_cnt);
                col_cnt -= 8;

                int16x8_t lhs_input = vldrhq_z_s16(lhs_ptr, pred);
                int16x8_t rhs_input = vldrbq_z_s16(rhs_ptr0, pred);

                result = vmladavaq_p_s16(result, lhs_input, rhs_input, pred);

                lhs_ptr += 8;
                rhs_ptr0 += 8;
            }

            int64_t result_64 = result;

            if (rhs_cols > MAX_COL_COUNT)
            {
                for (int i_rhs_cols = MAX_COL_COUNT; i_rhs_cols < rhs_cols; i_rhs_cols++)
                {
                    const int16_t lhs_temp = *lhs_ptr++;

                    result_64 += *rhs_ptr0++ * lhs_temp;
                }
            }

            int32_t tmp = 0;
            tmp = arm_nn_requantize_s64(result_64, reduced_multiplier, dst_shift);
            tmp += (int64_t)*dst;
            tmp = MAX(tmp, NN_Q15_MIN);
            tmp = MIN(tmp, NN_Q15_MAX);
            *dst++ = (int16_t)tmp;

            rhs_ptr += rhs_cols;
        }

    #else // ARM_MATH_MVEI

        const int32_t row_loop_cnt = rhs_rows / 2;

        for (int32_t i = 0; i < row_loop_cnt; i++)
        {

            int64_t acc_64_0 = 0;
            int64_t acc_64_1 = 0;
            int32_t acc_0 = 0;
            int32_t acc_1 = 0;

            const int32_t col_loop_cnt = rhs_cols_fast / 4;

            const int16_t *lhs_vec = lhs;
            const int8_t *rhs_0 = rhs_ptr;
            rhs_ptr += rhs_cols;
            const int8_t *rhs_1 = rhs_ptr;
            rhs_ptr += rhs_cols;

            for (int j = col_loop_cnt; j != 0; j--)
            {
                int32_t ker_0, ker_1, vec_part_0, vec_part_1;

                vec_part_0 = arm_nn_read_q15x2_ia(&lhs_vec);
                vec_part_1 = arm_nn_read_q15x2_ia(&lhs_vec);

                rhs_0 = read_and_pad(rhs_0, &ker_0, &ker_1);

                acc_0 = SMLAD(ker_0, vec_part_0, acc_0);
                acc_0 = SMLAD(ker_1, vec_part_1, acc_0);

                rhs_1 = read_and_pad(rhs_1, &ker_0, &ker_1);

                acc_1 = SMLAD(ker_0, vec_part_0, acc_1);
                acc_1 = SMLAD(ker_1, vec_part_1, acc_1);
            }

            acc_64_0 += acc_0;
            acc_64_1 += acc_1;

            for (int k = col_loop_cnt * 4; k < rhs_cols; k++)
            {
                const int32_t lhs_temp = (*lhs_vec);
                lhs_vec++;
                acc_64_0 += lhs_temp * (*rhs_0);
                rhs_0++;
                acc_64_1 += lhs_temp * (*rhs_1);
                rhs_1++;
            }

            acc_64_0 += *effective_bias_ptr++;
            acc_64_1 += *effective_bias_ptr++;
            int32_t tmp;

            tmp = arm_nn_requantize_s64(acc_64_0, reduced_multiplier, dst_shift);
            tmp += (int64_t)*dst;
            tmp = MAX(tmp, NN_Q15_MIN);
            tmp = MIN(tmp, NN_Q15_MAX);
            *dst++ = (int16_t)tmp;

            tmp = arm_nn_requantize_s64(acc_64_1, reduced_multiplier, dst_shift);
            tmp += (int64_t)*dst;
            tmp = MAX(tmp, NN_Q15_MIN);
            tmp = MIN(tmp, NN_Q15_MAX);
            *dst++ = (int16_t)tmp;
        }

        if (rhs_rows & 0x1)
        {
            int64_t acc_64_0 = 0;
            int32_t acc_0 = 0;
            const int32_t col_loop_cnt = rhs_cols_fast / 4;

            const int16_t *lhs_vec = lhs;
            const int8_t *rhs_0 = rhs_ptr;

            for (int i = col_loop_cnt; i != 0; i--)
            {
                int32_t ker_0, ker_1, vec;
                rhs_0 = read_and_pad(rhs_0, &ker_0, &ker_1);

                vec = arm_nn_read_q15x2_ia(&lhs_vec);
                acc_0 = SMLAD(ker_0, vec, acc_0);

                vec = arm_nn_read_q15x2_ia(&lhs_vec);
                acc_0 = SMLAD(ker_1, vec, acc_0);
            }

            acc_64_0 += acc_0;

            for (int j = col_loop_cnt * 4; j < rhs_cols; j++)
            {
                const int32_t lhs_temp = (*lhs_vec);
                lhs_vec++;
                acc_64_0 += lhs_temp * (*rhs_0);
                rhs_0++;
            }

            acc_64_0 += *effective_bias_ptr++;

            int32_t tmp;
            tmp = arm_nn_requantize_s64(acc_64_0, reduced_multiplier, dst_shift);
            tmp += (int64_t)*dst;
            tmp = MAX(tmp, NN_Q15_MIN);
            tmp = MIN(tmp, NN_Q15_MAX);
            *dst++ = (int16_t)tmp;
        }

    #endif // ARM_MATH_MVEI
#else      // ARM_MATH_DSP
        for (int i_row_loop_cnt = 0; i_row_loop_cnt < rhs_rows; i_row_loop_cnt++)
        {
            const int16_t *lhs_ptr = lhs;

            int64_t result = *effective_bias_ptr++;

            for (int32_t rhs_cols_idx = 0; rhs_cols_idx < rhs_cols; ++rhs_cols_idx)
            {
                const int64_t rhs_value0 = (int8_t)*rhs_ptr;
                const int64_t lhs_value = *lhs_ptr;

                result += lhs_value * rhs_value0;
                ++rhs_ptr;
                ++lhs_ptr;
            }

            // Quantize down
            result = arm_nn_requantize_s64(result, reduced_multiplier, dst_shift);
            result += (int64_t)*dst;

            // Clamp the result
            result = ((result) > (NN_Q15_MIN) ? (result) : (NN_Q15_MIN));
            result = ((result) < (NN_Q15_MAX) ? (result) : (NN_Q15_MAX));

            *dst++ = (int16_t)result;
        }
#endif     // ARM_MATH_DSP

        lhs += rhs_cols * batch_offset;
    }

    return ARM_CMSIS_NN_SUCCESS;
}

/**
 * @} end of Doxygen group
 */
