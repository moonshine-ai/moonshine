/*
 * SPDX-FileCopyrightText: Copyright 2024 Arm Limited and/or its affiliates <open-source-office@arm.com>
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
 * Title:        arm_nn_transpose_conv_row_s8_s32
 * Description:  Transpose covolution help function.
 *
 * $Date:        22 Oct 2024
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
 *  Computation of transposed convolution for one row of input into a rolling scratch buffer.
 *
 * Refer header file for details.
 *
 */
arm_cmsis_nn_status arm_nn_transpose_conv_row_s8_s32(const int8_t *lhs,
                                                     const int8_t *rhs,
                                                     int32_t *output_start,
                                                     const int32_t output_index,
                                                     const int32_t output_max,
                                                     const int32_t rhs_rows,
                                                     const int32_t rhs_cols,
                                                     const int32_t input_channels,
                                                     const int32_t output_channels,
                                                     const int32_t lhs_offset,
                                                     const int32_t row_offset,
                                                     const int32_t input_x,
                                                     const int32_t stride_x,
                                                     const int32_t skip_rows_top,
                                                     const int32_t skip_rows_bottom)
{

    const int32_t skip_pre_rows = skip_rows_top * rhs_cols * input_channels;
    const int32_t skip_post_rows = skip_rows_bottom * rhs_cols * input_channels;
    const int32_t rhs_rows_count = rhs_rows - skip_rows_top - skip_rows_bottom;

    int32_t input_count = input_x;
    for (; input_count > 3; input_count -= 4)
    {
        const int8_t *rhs_ptr = rhs;

        for (int32_t i_out_channel = 0; i_out_channel < output_channels; i_out_channel++)
        {
            rhs_ptr += skip_pre_rows;
            int32_t index = output_index;

            for (int32_t i_row = 0; i_row < rhs_rows_count; i_row++)
            {
                int32_t *output_ptr0 = output_start + index;

                for (int32_t i_col = 0; i_col < rhs_cols; i_col++)
                {
                    const int8_t *lhs_ptr0 = lhs;

                    int32_t result0 = 0;
                    int32_t result1 = 0;
                    int32_t result2 = 0;
                    int32_t result3 = 0;

#if defined(ARM_MATH_DSP) && !defined(ARM_MATH_MVEI)
                    const int16_t lhs_offset_s16 = (int16_t)lhs_offset;
                    const uint32_t lhs_offset_s16x2 = PKHBT(lhs_offset_s16, lhs_offset_s16, 16);

                    for (int32_t channel_count = input_channels; channel_count > 3; channel_count -= 4)
                    {
                        const int8_t *lhs_temp = lhs_ptr0;
                        int32_t lhs00 = arm_nn_read_s8x4(lhs_temp);
                        int32_t lhs01 = SXTAB16_RORn(lhs_offset_s16x2, (uint32_t)lhs00, 8);
                        lhs00 = SXTAB16(lhs_offset_s16x2, lhs00);
                        lhs_temp += input_channels;
                        int32_t lhs10 = arm_nn_read_s8x4(lhs_temp);
                        int32_t lhs11 = SXTAB16_RORn(lhs_offset_s16x2, (uint32_t)lhs10, 8);
                        lhs10 = SXTAB16(lhs_offset_s16x2, lhs10);
                        lhs_temp += input_channels;
                        int32_t lhs20 = arm_nn_read_s8x4(lhs_temp);
                        int32_t lhs21 = SXTAB16_RORn(lhs_offset_s16x2, (uint32_t)lhs20, 8);
                        lhs20 = SXTAB16(lhs_offset_s16x2, lhs20);
                        lhs_temp += input_channels;
                        int32_t lhs30 = arm_nn_read_s8x4(lhs_temp);
                        int32_t lhs31 = SXTAB16_RORn(lhs_offset_s16x2, (uint32_t)lhs30, 8);
                        lhs30 = SXTAB16(lhs_offset_s16x2, lhs30);
                        lhs_ptr0 += 4;

                        int32_t rhs0 = arm_nn_read_s8x4(rhs_ptr);
                        int32_t rhs1 = SXTB16_RORn((uint32_t)rhs0, 8);
                        rhs0 = SXTB16(rhs0);
                        rhs_ptr += 4;

                        result0 = SMLAD(lhs00, rhs0, result0);
                        result0 = SMLAD(lhs01, rhs1, result0);
                        result1 = SMLAD(lhs10, rhs0, result1);
                        result1 = SMLAD(lhs11, rhs1, result1);
                        result2 = SMLAD(lhs20, rhs0, result2);
                        result2 = SMLAD(lhs21, rhs1, result2);
                        result3 = SMLAD(lhs30, rhs0, result3);
                        result3 = SMLAD(lhs31, rhs1, result3);
                    }

                    for (int32_t i = 0; i < (input_channels & 0b11); i++)
                    {
                        const int8_t *lhs_temp = lhs_ptr0;
                        const int32_t lhs_val00 = *lhs_temp + lhs_offset;
                        lhs_temp += input_channels;
                        const int32_t lhs_val10 = *lhs_temp + lhs_offset;
                        lhs_temp += input_channels;
                        const int32_t lhs_val20 = *lhs_temp + lhs_offset;
                        lhs_temp += input_channels;
                        const int32_t lhs_val30 = *lhs_temp + lhs_offset;
                        lhs_ptr0++;

                        const int32_t rhs_val0 = *rhs_ptr++;

                        result0 += lhs_val00 * rhs_val0;
                        result1 += lhs_val10 * rhs_val0;
                        result2 += lhs_val20 * rhs_val0;
                        result3 += lhs_val30 * rhs_val0;
                    }

                    int32_t *output_temp = output_ptr0;
                    *output_ptr0 += result0;
                    output_temp += stride_x * output_channels;
                    *output_temp += result1;
                    output_temp += stride_x * output_channels;
                    *output_temp += result2;
                    output_temp += stride_x * output_channels;
                    *output_temp += result3;

                    output_ptr0 += output_channels;
#else
                    int32_t rhs_sum = 0;
    #if defined(ARM_MATH_MVEI)

                    int channel_count = input_channels;
                    for (int channel_i = 0; channel_i < (input_channels + 15) / 16; channel_i++)
                    {
                        mve_pred16_t p0 = vctp8q((uint32_t)channel_count);
                        channel_count -= 16;

                        const int8_t *lhs_temp = lhs_ptr0;
                        int8x16_t v_lhs00 = vldrbq_z_s8(lhs_temp, p0);
                        lhs_temp += input_channels;
                        int8x16_t v_lhs10 = vldrbq_z_s8(lhs_temp, p0);
                        lhs_temp += input_channels;
                        int8x16_t v_lhs20 = vldrbq_z_s8(lhs_temp, p0);
                        lhs_temp += input_channels;
                        int8x16_t v_lhs30 = vldrbq_z_s8(lhs_temp, p0);

                        lhs_ptr0 += 16;
                        int8x16_t v_rhs0 = vldrbq_z_s8(rhs_ptr, p0);
                        rhs_ptr += 16;

                        result0 = vmladavaq_s8(result0, v_lhs00, v_rhs0);
                        result1 = vmladavaq_s8(result1, v_lhs10, v_rhs0);
                        result2 = vmladavaq_s8(result2, v_lhs20, v_rhs0);
                        result3 = vmladavaq_s8(result3, v_lhs30, v_rhs0);

                        rhs_sum = vaddvaq_s8(rhs_sum, v_rhs0);
                    }

                    rhs_ptr += channel_count;

    #else
                    for (int32_t channel_count = 0; channel_count < input_channels / 2; channel_count++)
                    {
                        const int8_t *lhs_temp = lhs_ptr0;
                        const int32_t lhs_val00 = *lhs_temp;
                        lhs_temp += input_channels;
                        const int32_t lhs_val10 = *lhs_temp;
                        lhs_temp += input_channels;
                        const int32_t lhs_val20 = *lhs_temp;
                        lhs_temp += input_channels;
                        const int32_t lhs_val30 = *lhs_temp;
                        lhs_ptr0++;

                        lhs_temp = lhs_ptr0;
                        const int32_t lhs_val01 = *lhs_temp;
                        lhs_temp += input_channels;
                        const int32_t lhs_val11 = *lhs_temp;
                        lhs_temp += input_channels;
                        const int32_t lhs_val21 = *lhs_temp;
                        lhs_temp += input_channels;
                        const int32_t lhs_val31 = *lhs_temp;
                        lhs_ptr0++;

                        const int32_t rhs_val0 = *rhs_ptr++;
                        const int32_t rhs_val1 = *rhs_ptr++;

                        result0 += lhs_val00 * rhs_val0;
                        result0 += lhs_val01 * rhs_val1;

                        result1 += lhs_val10 * rhs_val0;
                        result1 += lhs_val11 * rhs_val1;

                        result2 += lhs_val20 * rhs_val0;
                        result2 += lhs_val21 * rhs_val1;

                        result3 += lhs_val30 * rhs_val0;
                        result3 += lhs_val31 * rhs_val1;

                        rhs_sum += rhs_val0;
                        rhs_sum += rhs_val1;
                    }

                    // Input channel tail-handling
                    if (input_channels & 0b1)
                    {
                        const int8_t *lhs_temp = lhs_ptr0;
                        const int32_t lhs_val00 = *lhs_temp;
                        lhs_temp += input_channels;
                        const int32_t lhs_val10 = *lhs_temp;
                        lhs_temp += input_channels;
                        const int32_t lhs_val20 = *lhs_temp;
                        lhs_temp += input_channels;
                        const int32_t lhs_val30 = *lhs_temp;
                        lhs_ptr0++;

                        const int32_t rhs_val0 = *rhs_ptr++;

                        result0 += lhs_val00 * rhs_val0;
                        result1 += lhs_val10 * rhs_val0;
                        result2 += lhs_val20 * rhs_val0;
                        result3 += lhs_val30 * rhs_val0;

                        rhs_sum += rhs_val0;
                    }
    #endif
                    int32_t *output_temp = output_ptr0;
                    *output_ptr0 += result0 + rhs_sum * lhs_offset;
                    output_temp += stride_x * output_channels;
                    *output_temp += result1 + rhs_sum * lhs_offset;
                    output_temp += stride_x * output_channels;
                    *output_temp += result2 + rhs_sum * lhs_offset;
                    output_temp += stride_x * output_channels;
                    *output_temp += result3 + rhs_sum * lhs_offset;

                    output_ptr0 += output_channels;
#endif
                }

                // Next row, wrapping around the circular buffer
                index = (index + row_offset) % output_max;
            }
            // Next output_channel
            ++output_start;
            rhs_ptr += skip_post_rows;
        }

        output_start += (4 * stride_x - 1) * output_channels;
        lhs += 4 * input_channels;
    }

    // Input column tail handling
    if (input_count & 0b10)
    {
        const int8_t *rhs_ptr = rhs;

        for (int32_t i_out_channel = 0; i_out_channel < output_channels; i_out_channel++)
        {
            int32_t index = output_index;
            rhs_ptr += skip_pre_rows;

            for (int32_t i_row = 0; i_row < rhs_rows_count; i_row++)
            {
                int32_t *output_ptr0 = output_start + index;

                for (int32_t i_col = 0; i_col < rhs_cols; i_col++)
                {
                    const int8_t *lhs_ptr0 = lhs;

                    int32_t result0 = 0;
                    int32_t result1 = 0;

#if defined(ARM_MATH_DSP) && !defined(ARM_MATH_MVEI)
                    const int16_t lhs_offset_s16 = (int16_t)lhs_offset;
                    const uint32_t lhs_offset_s16x2 = PKHBT(lhs_offset_s16, lhs_offset_s16, 16);

                    for (int32_t channel_count = input_channels; channel_count > 3; channel_count -= 4)
                    {
                        const int8_t *lhs_temp = lhs_ptr0;
                        int32_t lhs00 = arm_nn_read_s8x4(lhs_temp);
                        int32_t lhs01 = SXTAB16_RORn(lhs_offset_s16x2, (uint32_t)lhs00, 8);
                        lhs00 = SXTAB16(lhs_offset_s16x2, lhs00);
                        lhs_temp += input_channels;
                        int32_t lhs10 = arm_nn_read_s8x4(lhs_temp);
                        int32_t lhs11 = SXTAB16_RORn(lhs_offset_s16x2, (uint32_t)lhs10, 8);
                        lhs10 = SXTAB16(lhs_offset_s16x2, lhs10);
                        lhs_ptr0 += 4;

                        int32_t rhs0 = arm_nn_read_s8x4(rhs_ptr);
                        int32_t rhs1 = SXTB16_RORn((uint32_t)rhs0, 8);
                        rhs0 = SXTB16(rhs0);
                        rhs_ptr += 4;

                        result0 = SMLAD(lhs00, rhs0, result0);
                        result0 = SMLAD(lhs01, rhs1, result0);
                        result1 = SMLAD(lhs10, rhs0, result1);
                        result1 = SMLAD(lhs11, rhs1, result1);
                    }

                    for (int32_t i = 0; i < (input_channels & 0b11); i++)
                    {
                        const int8_t *lhs_temp = lhs_ptr0;
                        const int32_t lhs_val00 = *lhs_temp + lhs_offset;
                        lhs_temp += input_channels;
                        const int32_t lhs_val10 = *lhs_temp + lhs_offset;
                        lhs_ptr0++;

                        const int32_t rhs_val0 = *rhs_ptr++;

                        result0 += lhs_val00 * rhs_val0;
                        result1 += lhs_val10 * rhs_val0;
                    }

                    int32_t *output_temp = output_ptr0;
                    *output_ptr0 += result0;
                    output_temp += stride_x * output_channels;
                    *output_temp += result1;

                    output_ptr0 += output_channels;
#else
                    int32_t rhs_sum = 0;
    #if defined(ARM_MATH_MVEI)
                    int channel_count = input_channels;
                    for (int channel_i = 0; channel_i < (input_channels + 15) / 16; channel_i++)
                    {
                        mve_pred16_t p0 = vctp8q((uint32_t)channel_count);
                        channel_count -= 16;

                        const int8_t *lhs_temp = lhs_ptr0;
                        int8x16_t v_lhs00 = vldrbq_z_s8(lhs_temp, p0);
                        lhs_temp += input_channels;
                        int8x16_t v_lhs10 = vldrbq_z_s8(lhs_temp, p0);
                        lhs_ptr0 += 16;
                        int8x16_t v_rhs0 = vldrbq_z_s8(rhs_ptr, p0);
                        rhs_ptr += 16;

                        result0 = vmladavaq_s8(result0, v_lhs00, v_rhs0);
                        result1 = vmladavaq_s8(result1, v_lhs10, v_rhs0);

                        rhs_sum = vaddvaq_s8(rhs_sum, v_rhs0);
                    }

                    rhs_ptr += channel_count;

    #else
                    for (int32_t channel_count = 0; channel_count < input_channels; channel_count++)
                    {
                        const int8_t *lhs_temp = lhs_ptr0;
                        const int32_t lhs_val00 = *lhs_temp;
                        lhs_temp += input_channels;
                        const int32_t lhs_val10 = *lhs_temp;
                        lhs_ptr0++;

                        const int32_t rhs_val0 = *rhs_ptr++;

                        result0 += lhs_val00 * rhs_val0;
                        result1 += lhs_val10 * rhs_val0;

                        rhs_sum += rhs_val0;
                    }
    #endif
                    int32_t *output_temp = output_ptr0;
                    *output_ptr0 += result0 + rhs_sum * lhs_offset;
                    output_temp += stride_x * output_channels;
                    *output_temp += result1 + rhs_sum * lhs_offset;

                    output_ptr0 += output_channels;
#endif
                }

                // Next row, wrapping around the circular buffer
                index = (index + row_offset) % output_max;
            }

            // Next output_channel
            ++output_start;
            rhs_ptr += skip_post_rows;
        }

        output_start += (2 * stride_x - 1) * output_channels;
        lhs += 2 * input_channels;
    }

    if (input_count & 0b1)
    {
        const int8_t *rhs_ptr = rhs;

        for (int32_t i_out_channel = 0; i_out_channel < output_channels; i_out_channel++)
        {
            int32_t index = output_index;
            rhs_ptr += skip_pre_rows;

            for (int32_t i_row = 0; i_row < rhs_rows_count; i_row++)
            {
                int32_t *output_ptr0 = output_start + index;

                for (int32_t i_col = 0; i_col < rhs_cols; i_col++)
                {
                    const int8_t *lhs_ptr0 = lhs;

                    int32_t result0 = 0;
#if defined(ARM_MATH_DSP) && !defined(ARM_MATH_MVEI)
                    const int16_t lhs_offset_s16 = (int16_t)lhs_offset;
                    const uint32_t lhs_offset_s16x2 = PKHBT(lhs_offset_s16, lhs_offset_s16, 16);

                    for (int32_t channel_count = input_channels; channel_count > 3; channel_count -= 4)
                    {
                        const int8_t *lhs_temp = lhs_ptr0;
                        int32_t lhs00 = arm_nn_read_s8x4(lhs_temp);
                        int32_t lhs01 = SXTAB16_RORn(lhs_offset_s16x2, (uint32_t)lhs00, 8);
                        lhs00 = SXTAB16(lhs_offset_s16x2, lhs00);
                        lhs_ptr0 += 4;

                        int32_t rhs0 = arm_nn_read_s8x4(rhs_ptr);
                        int32_t rhs1 = SXTB16_RORn((uint32_t)rhs0, 8);
                        rhs0 = SXTB16(rhs0);
                        rhs_ptr += 4;

                        result0 = SMLAD(lhs00, rhs0, result0);
                        result0 = SMLAD(lhs01, rhs1, result0);
                    }

                    for (int32_t i = 0; i < (input_channels & 0b11); i++)
                    {
                        const int8_t *lhs_temp = lhs_ptr0;
                        const int32_t lhs_val00 = *lhs_temp + lhs_offset;
                        lhs_ptr0++;

                        const int32_t rhs_val0 = *rhs_ptr++;

                        result0 += lhs_val00 * rhs_val0;
                    }
#else
    #if defined(ARM_MATH_MVEI)
                    int channel_count = input_channels;
                    for (int channel_i = 0; channel_i < (input_channels + 15) / 16; channel_i++)
                    {
                        mve_pred16_t p0 = vctp8q((uint32_t)channel_count);
                        channel_count -= 16;

                        int8x16_t v_lhs00 = vldrbq_z_s8(lhs_ptr0, p0);
                        lhs_ptr0 += 16;
                        int8x16_t v_rhs0 = vldrbq_z_s8(rhs_ptr, p0);
                        rhs_ptr += 16;

                        result0 = vmladavaq_s8(result0, v_lhs00, v_rhs0);

                        int32_t rhs_sum = vaddvaq_s8(0, v_rhs0);
                        result0 += rhs_sum * lhs_offset;
                    }

                    rhs_ptr += channel_count;
    #else
                    for (int32_t channel_count = 0; channel_count < input_channels; channel_count++)
                    {
                        const int32_t lhs_val00 = *lhs_ptr0;
                        lhs_ptr0++;

                        const int32_t rhs_val0 = *rhs_ptr++;

                        result0 += (lhs_val00 + lhs_offset) * rhs_val0;
                    }
    #endif
#endif
                    *output_ptr0 += result0;
                    output_ptr0 += output_channels;
                }

                // Next row, wrapping around the circular buffer
                index = (index + row_offset) % output_max;
            }

            // Next output_channel
            ++output_start;
            rhs_ptr += skip_post_rows;
        }
    }
    return ARM_CMSIS_NN_SUCCESS;
}

/**
 * @} end of Doxygen group
 */
