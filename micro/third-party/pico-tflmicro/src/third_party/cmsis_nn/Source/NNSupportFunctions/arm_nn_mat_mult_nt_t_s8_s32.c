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
 * Title:        arm_nn_mat_mult_nt_t_s8_s32
 * Description:  Matrix multiplication support function with the right-hand-side (rhs) matrix transposed
 *
 * $Date:        31 January 2024
 * $Revision:    V.1.1.0
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
 * s32 matrix multiplication with the right-hand-side matrix transposed
 *
 * Refer header file for details.
 *
 */
arm_cmsis_nn_status arm_nn_mat_mult_nt_t_s8_s32(const int8_t *lhs,
                                                const int8_t *rhs,
                                                int32_t *dst,
                                                const int32_t lhs_rows,
                                                const int32_t rhs_rows,
                                                const int32_t rhs_cols,
                                                const int32_t lhs_offset,
                                                const int32_t dst_idx_offset)
{
    int32_t rhs_rows_idx = rhs_rows;
    const int32_t dst_idx_col_offset = dst_idx_offset * rhs_cols;
#if defined(ARM_MATH_MVEI)
    for (; rhs_rows_idx >= 16; rhs_rows_idx -= 16)
    {
        int32_t *dst_ptr = &dst[0];
        const int8_t *lhs_ptr = &lhs[0];
        int32_t lhs_rows_idx = lhs_rows;

        for (; lhs_rows_idx >= 4; lhs_rows_idx -= 4)
        {
            const int8_t *rhs_ptr = &rhs[0];
            int8x16_t v_lhs0 = vldrbq_s8(lhs_ptr);
            lhs_ptr += rhs_rows;
            int8x16_t v_lhs1 = vldrbq_s8(lhs_ptr);
            lhs_ptr += rhs_rows;
            int8x16_t v_lhs2 = vldrbq_s8(lhs_ptr);
            lhs_ptr += rhs_rows;
            int8x16_t v_lhs3 = vldrbq_s8(lhs_ptr);
            lhs_ptr += rhs_rows;

            for (int32_t rhs_cols_idx = rhs_cols; rhs_cols_idx != 0; rhs_cols_idx--)
            {
                int32_t *ip_dst = dst_ptr;

                int8x16_t v_rhs0 = vldrbq_s8(rhs_ptr);
                int32_t rhs_sum = vaddvq_s8(v_rhs0);
                rhs_sum *= lhs_offset;

                *ip_dst += rhs_sum;
                *ip_dst = vmladavaq_s8(*ip_dst, v_lhs0, v_rhs0);
                ip_dst += dst_idx_col_offset;

                *ip_dst += rhs_sum;
                *ip_dst = vmladavaq_s8(*ip_dst, v_lhs1, v_rhs0);
                ip_dst += dst_idx_col_offset;

                *ip_dst += rhs_sum;
                *ip_dst = vmladavaq_s8(*ip_dst, v_lhs2, v_rhs0);
                ip_dst += dst_idx_col_offset;

                *ip_dst += rhs_sum;
                *ip_dst = vmladavaq_s8(*ip_dst, v_lhs3, v_rhs0);

                dst_ptr += dst_idx_offset;
                rhs_ptr += rhs_rows;
            }

            dst_ptr += 3 * dst_idx_col_offset;
        }
        for (; lhs_rows_idx > 0; lhs_rows_idx--)
        {
            const int8_t *rhs_ptr = &rhs[0];
            int8x16_t v_lhs0 = vldrbq_s8(lhs_ptr);

            for (int32_t rhs_cols_idx = rhs_cols; rhs_cols_idx != 0; rhs_cols_idx--)
            {
                int8x16_t v_rhs0 = vldrbq_s8(rhs_ptr);

                int32_t offset_sum = vaddvq_s8(v_rhs0);
                *dst_ptr += offset_sum * lhs_offset;

                *dst_ptr = vmladavaq_s8(*dst_ptr, v_lhs0, v_rhs0);

                dst_ptr += dst_idx_offset;
                rhs_ptr += rhs_rows;
            }
            lhs_ptr += rhs_rows;
        }

        rhs += 16;
        lhs += 16;
    }
    if (rhs_rows_idx)
    {
        mve_pred16_t rmdr = (1 << rhs_rows_idx) - 1;
        int32_t *dst_ptr = &dst[0];
        const int8_t *lhs_ptr = &lhs[0];
        int32_t lhs_rows_idx = lhs_rows;

        for (; lhs_rows_idx >= 4; lhs_rows_idx -= 4)
        {
            const int8_t *rhs_ptr = &rhs[0];
            int8x16_t v_lhs0 = vldrbq_z_s8(lhs_ptr, rmdr);
            lhs_ptr += rhs_rows;
            int8x16_t v_lhs1 = vldrbq_z_s8(lhs_ptr, rmdr);
            lhs_ptr += rhs_rows;
            int8x16_t v_lhs2 = vldrbq_z_s8(lhs_ptr, rmdr);
            lhs_ptr += rhs_rows;
            int8x16_t v_lhs3 = vldrbq_z_s8(lhs_ptr, rmdr);
            lhs_ptr += rhs_rows;

            for (int32_t rhs_cols_idx = rhs_cols; rhs_cols_idx != 0; rhs_cols_idx--)
            {
                int32_t *ip_dst = dst_ptr;
                int8x16_t v_rhs0 = vldrbq_z_s8(rhs_ptr, rmdr);

                int32_t rhs_sum = vaddvq_p_s8(v_rhs0, rmdr);
                rhs_sum *= lhs_offset;

                *ip_dst += rhs_sum;
                *ip_dst = vmladavaq_p_s8(*ip_dst, v_lhs0, v_rhs0, rmdr);
                ip_dst += dst_idx_col_offset;

                *ip_dst += rhs_sum;
                *ip_dst = vmladavaq_p_s8(*ip_dst, v_lhs1, v_rhs0, rmdr);
                ip_dst += dst_idx_col_offset;

                *ip_dst += rhs_sum;
                *ip_dst = vmladavaq_p_s8(*ip_dst, v_lhs2, v_rhs0, rmdr);
                ip_dst += dst_idx_col_offset;

                *ip_dst += rhs_sum;
                *ip_dst = vmladavaq_p_s8(*ip_dst, v_lhs3, v_rhs0, rmdr);

                dst_ptr += dst_idx_offset;
                rhs_ptr += rhs_rows;
            }

            dst_ptr += 3 * dst_idx_col_offset;
        }
        for (; lhs_rows_idx > 0; lhs_rows_idx--)
        {
            const int8_t *rhs_ptr = &rhs[0];
            int8x16_t v_lhs0 = vldrbq_z_s8(lhs_ptr, rmdr);

            for (int32_t rhs_cols_idx = rhs_cols; rhs_cols_idx != 0; rhs_cols_idx--)
            {
                int8x16_t v_rhs0 = vldrbq_z_s8(rhs_ptr, rmdr);

                int32_t rhs_sum = vaddvq_p_s8(v_rhs0, rmdr);
                *dst_ptr += rhs_sum * lhs_offset;

                *dst_ptr = vmladavaq_p_s8(*dst_ptr, v_lhs0, v_rhs0, rmdr);

                dst_ptr += dst_idx_offset;
                rhs_ptr += rhs_rows;
            }
            lhs_ptr += rhs_rows;
        }
    }

#elif defined(ARM_MATH_DSP)
    int16_t lhs_offset_s16 = (int16_t)lhs_offset;
    const uint32_t lhs_offset_s16x2 = PKHBT(lhs_offset_s16, lhs_offset_s16, 16);
    for (; rhs_rows_idx >= 8; rhs_rows_idx -= 8)
    {
        int32_t *dst_ptr = &dst[0];
        const int8_t *lhs_ptr = &lhs[0];
        int32_t lhs_rows_idx = lhs_rows >> 1;

        while (lhs_rows_idx)
        {
            const int8_t *rhs_ptr = &rhs[0];

            int32_t lhs000, lhs001, lhs010, lhs011, lhs100, lhs101, lhs110, lhs111;
            read_pad_and_add_s8(lhs_ptr, &lhs000, &lhs001, lhs_offset_s16x2);
            read_pad_and_add_s8(&lhs_ptr[4], &lhs010, &lhs011, lhs_offset_s16x2);
            read_pad_and_add_s8(&lhs_ptr[rhs_rows], &lhs100, &lhs101, lhs_offset_s16x2);
            read_pad_and_add_s8(&lhs_ptr[rhs_rows + 4], &lhs110, &lhs111, lhs_offset_s16x2);

            for (int32_t rhs_cols_idx = rhs_cols; rhs_cols_idx != 0; rhs_cols_idx--)
            {
                int32_t rhs_val00, rhs_val01;
                read_and_pad(rhs_ptr, &rhs_val00, &rhs_val01);

                dst_ptr[0] = SMLAD(lhs000, rhs_val00, dst_ptr[0]);
                dst_ptr[0] = SMLAD(lhs001, rhs_val01, dst_ptr[0]);
                dst_ptr[dst_idx_col_offset] = SMLAD(lhs100, rhs_val00, dst_ptr[dst_idx_col_offset]);
                dst_ptr[dst_idx_col_offset] = SMLAD(lhs101, rhs_val01, dst_ptr[dst_idx_col_offset]);

                read_and_pad(&rhs_ptr[4], &rhs_val00, &rhs_val01);

                dst_ptr[0] = SMLAD(lhs010, rhs_val00, dst_ptr[0]);
                dst_ptr[0] = SMLAD(lhs011, rhs_val01, dst_ptr[0]);
                dst_ptr[dst_idx_col_offset] = SMLAD(lhs110, rhs_val00, dst_ptr[dst_idx_col_offset]);
                dst_ptr[dst_idx_col_offset] = SMLAD(lhs111, rhs_val01, dst_ptr[dst_idx_col_offset]);

                dst_ptr += dst_idx_offset;
                rhs_ptr += rhs_rows;
            }
            dst_ptr += dst_idx_col_offset;

            lhs_ptr += rhs_rows << 1;

            lhs_rows_idx--;
        }
        // Left-over rows
        if (lhs_rows % 2)
        {
            const int8_t *rhs_ptr = &rhs[0];
            int32_t lhs00, lhs01, lhs10, lhs11;
            read_pad_and_add_s8(lhs_ptr, &lhs00, &lhs01, lhs_offset_s16x2);
            read_pad_and_add_s8(&lhs_ptr[4], &lhs10, &lhs11, lhs_offset_s16x2);

            for (int32_t rhs_cols_idx = rhs_cols; rhs_cols_idx != 0; rhs_cols_idx--)
            {
                int32_t rhs_val00, rhs_val01, rhs_val10, rhs_val11;
                read_and_pad(rhs_ptr, &rhs_val00, &rhs_val01);
                read_and_pad(&rhs_ptr[4], &rhs_val10, &rhs_val11);

                dst_ptr[0] = SMLAD(lhs00, rhs_val00, dst_ptr[0]);
                dst_ptr[0] = SMLAD(lhs01, rhs_val01, dst_ptr[0]);
                dst_ptr[0] = SMLAD(lhs10, rhs_val10, dst_ptr[0]);
                dst_ptr[0] = SMLAD(lhs11, rhs_val11, dst_ptr[0]);

                dst_ptr += dst_idx_offset;
                rhs_ptr += rhs_rows;
            }
        }

        rhs += 8;
        lhs += 8;
    }
    for (; rhs_rows_idx >= 4; rhs_rows_idx -= 4)
    {
        int32_t *dst_ptr = &dst[0];
        const int8_t *lhs_ptr = &lhs[0];

        int32_t lhs_rows_idx = lhs_rows >> 1;

        while (lhs_rows_idx)
        {
            const int8_t *rhs_ptr = &rhs[0];

            int32_t lhs00, lhs01, lhs10, lhs11;
            read_pad_and_add_s8(lhs_ptr, &lhs00, &lhs01, lhs_offset_s16x2);
            read_pad_and_add_s8(&lhs_ptr[rhs_rows], &lhs10, &lhs11, lhs_offset_s16x2);

            for (int32_t rhs_cols_idx = rhs_cols; rhs_cols_idx != 0; rhs_cols_idx--)
            {
                int32_t rhs_val0, rhs_val1;
                read_and_pad(rhs_ptr, &rhs_val0, &rhs_val1);

                dst_ptr[0] = SMLAD(lhs00, rhs_val0, dst_ptr[0]);
                dst_ptr[0] = SMLAD(lhs01, rhs_val1, dst_ptr[0]);
                dst_ptr[dst_idx_col_offset] = SMLAD(lhs10, rhs_val0, dst_ptr[dst_idx_col_offset]);
                dst_ptr[dst_idx_col_offset] = SMLAD(lhs11, rhs_val1, dst_ptr[dst_idx_col_offset]);
                dst_ptr += dst_idx_offset;
                rhs_ptr += rhs_rows;
            }
            dst_ptr += dst_idx_col_offset;

            lhs_ptr += rhs_rows << 1;

            lhs_rows_idx--;
        }
        // Left-over rows
        if (lhs_rows % 2)
        {
            const int8_t *rhs_ptr = &rhs[0];
            int32_t lhs00, lhs01;
            read_pad_and_add_s8(lhs_ptr, &lhs00, &lhs01, lhs_offset_s16x2);

            for (int32_t rhs_cols_idx = rhs_cols; rhs_cols_idx != 0; rhs_cols_idx--)
            {
                int32_t rhs_val0, rhs_val1;
                read_and_pad(rhs_ptr, &rhs_val0, &rhs_val1);

                dst_ptr[0] = SMLAD(lhs00, rhs_val0, dst_ptr[0]);
                dst_ptr[0] = SMLAD(lhs01, rhs_val1, dst_ptr[0]);

                dst_ptr += dst_idx_offset;
                rhs_ptr += rhs_rows;
            }
        }

        rhs += 4;
        lhs += 4;
    }
    for (; rhs_rows_idx >= 2; rhs_rows_idx -= 2)
    {
        int32_t *dst_ptr = &dst[0];
        const int8_t *lhs_ptr = &lhs[0];

        int32_t lhs_rows_idx = lhs_rows >> 1;

        while (lhs_rows_idx)
        {
            const int8_t *rhs_ptr = &rhs[0];

            int32_t lhs0, lhs1;
            read_pad_and_add_s8x2(lhs_ptr, &lhs0, lhs_offset_s16x2);
            read_pad_and_add_s8x2(&lhs_ptr[rhs_rows], &lhs1, lhs_offset_s16x2);

            for (int32_t rhs_cols_idx = rhs_cols; rhs_cols_idx != 0; rhs_cols_idx--)
            {
                int32_t rhs_val;
                read_and_pad_s8x2(rhs_ptr, &rhs_val);

                dst_ptr[0] = SMLAD(lhs0, rhs_val, dst_ptr[0]);
                dst_ptr[dst_idx_col_offset] = SMLAD(lhs1, rhs_val, dst_ptr[dst_idx_col_offset]);

                dst_ptr += dst_idx_offset;
                rhs_ptr += rhs_rows;
            }
            dst_ptr += dst_idx_col_offset;

            lhs_ptr += rhs_rows << 1;

            lhs_rows_idx--;
        }
        // Left-over rows
        if (lhs_rows % 2)
        {
            const int8_t *rhs_ptr = &rhs[0];
            const int32_t lhs_value = lhs_ptr[0] + lhs_offset;
            const int32_t lhs_value01 = lhs_ptr[1] + lhs_offset;

            for (int32_t rhs_cols_idx = rhs_cols; rhs_cols_idx != 0; rhs_cols_idx--)
            {
                const int32_t rhs_value0 = rhs_ptr[0];
                const int32_t rhs_value01 = rhs_ptr[1];

                dst_ptr[0] += lhs_value * rhs_value0;
                dst_ptr[0] += lhs_value01 * rhs_value01;
                dst_ptr += dst_idx_offset;
                rhs_ptr += rhs_rows;
            }
        }

        rhs += 2;
        lhs += 2;
    }
#else
    for (; rhs_rows_idx >= 2; rhs_rows_idx -= 2)
    {
        int32_t *dst_ptr = &dst[0];
        const int8_t *lhs_ptr = &lhs[0];

        int32_t lhs_rows_idx = lhs_rows >> 1;

        while (lhs_rows_idx)
        {
            const int8_t *rhs_ptr = &rhs[0];

            const int32_t lhs_value00 = lhs_ptr[0] + lhs_offset;
            const int32_t lhs_value01 = lhs_ptr[1] + lhs_offset;

            const int32_t lhs_value10 = lhs_ptr[rhs_rows] + lhs_offset;
            const int32_t lhs_value11 = lhs_ptr[rhs_rows + 1] + lhs_offset;

            for (int32_t rhs_cols_idx = rhs_cols; rhs_cols_idx != 0; rhs_cols_idx--)
            {
                const int32_t rhs_value0 = rhs_ptr[0];
                const int32_t rhs_value1 = rhs_ptr[1];

                dst_ptr[0] += lhs_value00 * rhs_value0;
                dst_ptr[0] += lhs_value01 * rhs_value1;

                dst_ptr[dst_idx_col_offset] += lhs_value10 * rhs_value0;
                dst_ptr[dst_idx_col_offset] += lhs_value11 * rhs_value1;
                dst_ptr += dst_idx_offset;
                rhs_ptr += rhs_rows;
            }
            dst_ptr += dst_idx_col_offset;

            lhs_ptr += rhs_rows << 1;

            lhs_rows_idx--;
        }
        // Left-over rows
        if (lhs_rows % 2)
        {
            const int8_t *rhs_ptr = &rhs[0];
            const int32_t lhs_value = lhs_ptr[0] + lhs_offset;
            const int32_t lhs_value01 = lhs_ptr[1] + lhs_offset;

            for (int32_t rhs_cols_idx = rhs_cols; rhs_cols_idx != 0; rhs_cols_idx--)
            {
                const int32_t rhs_value0 = rhs_ptr[0];
                const int32_t rhs_value01 = rhs_ptr[1];

                dst_ptr[0] += lhs_value * rhs_value0;
                dst_ptr[0] += lhs_value01 * rhs_value01;
                dst_ptr += dst_idx_offset;
                rhs_ptr += rhs_rows;
            }
        }

        rhs += 2;
        lhs += 2;
    }
#endif
#if !defined(ARM_MATH_MVEI)
    if (rhs_rows_idx)
    {
        const int8_t *lhs_ptr = &lhs[0];
        int32_t *dst_ptr = &dst[0];

        for (int32_t lhs_rows_idx = 0; lhs_rows_idx < lhs_rows; ++lhs_rows_idx)
        {
            const int8_t *rhs_ptr = &rhs[0];
            const int32_t lhs_value = lhs_ptr[0] + lhs_offset;

            for (int32_t rhs_cols_idx = rhs_cols; rhs_cols_idx != 0; rhs_cols_idx--)
            {
                const int32_t rhs_value = rhs_ptr[0];

                *dst_ptr += lhs_value * rhs_value;

                dst_ptr += dst_idx_offset;
                rhs_ptr += rhs_rows;
            }
            lhs_ptr += rhs_rows;
        }
    }
#endif
    return ARM_CMSIS_NN_SUCCESS;
}

/**
 * @} end of Doxygen group
 */
