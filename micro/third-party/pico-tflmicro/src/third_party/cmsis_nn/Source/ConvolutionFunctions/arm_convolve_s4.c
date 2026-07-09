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
 * Title:        arm_convolve_s4.c
 * Description:  s8 version of convolution using symmetric quantization with 4 bit weights.
 *
 * $Date:        17 May 2024
 * $Revision:    V.1.2.0
 *
 * Target :  Arm(R) M-Profile Architecture
 *
 * -------------------------------------------------------------------- */

#include "third_party/cmsis_nn/Include/arm_nnfunctions.h"
#include "third_party/cmsis_nn/Include/arm_nnsupportfunctions.h"

/**
 *  @ingroup Public
 */

/**
 * @addtogroup NNConv
 * @{
 */

/*
 * Basic s8 convolution function with int4 weights.
 *
 * Refer header file for details. Optimal use case for the DSP/MVE implementation is when input and output channels
 * are multiples of 4 or at least greater than 4.
 *
 */
arm_cmsis_nn_status arm_convolve_s4(const cmsis_nn_context *ctx,
                                    const cmsis_nn_conv_params *conv_params,
                                    const cmsis_nn_per_channel_quant_params *quant_params,
                                    const cmsis_nn_dims *input_dims,
                                    const int8_t *input_data,
                                    const cmsis_nn_dims *filter_dims,
                                    const int8_t *packed_filter_data,
                                    const cmsis_nn_dims *bias_dims,
                                    const int32_t *bias_data,
                                    const cmsis_nn_dims *output_dims,
                                    int8_t *output_data)
{
    (void)bias_dims;

    if (ctx->buf == NULL)
    {
        return ARM_CMSIS_NN_ARG_ERROR;
    }
    int16_t *buffer_a = (int16_t *)ctx->buf;

    const int32_t input_batches = input_dims->n;
    const uint16_t input_x = input_dims->w;
    const uint16_t input_y = input_dims->h;
    const uint16_t input_ch = input_dims->c;
    const uint16_t kernel_x = filter_dims->w;
    const uint16_t kernel_y = filter_dims->h;
    const uint16_t output_x = output_dims->w;
    const uint16_t output_y = output_dims->h;
    const uint16_t output_ch = output_dims->c;

    const uint16_t pad_x = conv_params->padding.w;
    const uint16_t pad_y = conv_params->padding.h;
    const uint16_t stride_x = conv_params->stride.w;
    const uint16_t stride_y = conv_params->stride.h;
    const int32_t dilation_x = conv_params->dilation.w;
    const int32_t dilation_y = conv_params->dilation.h;
    const int32_t out_offset = conv_params->output_offset;
    const int32_t out_activation_min = conv_params->activation.min;
    const int32_t out_activation_max = conv_params->activation.max;
    const int32_t rhs_cols = kernel_x * kernel_y * input_ch;
    const int32_t input_offset = conv_params->input_offset;

    int32_t *output_mult = quant_params->multiplier;
    int32_t *output_shift = quant_params->shift;

    int i_batch;
    for (i_batch = 0; i_batch < input_batches; i_batch++)
    {
#if defined(ARM_MATH_MVEI)
        /* Generate up to four columns from the input tensor a GEMM computation */
        int8_t *im2col_buf = (int8_t *)buffer_a;
        const int32_t rhs_rows = output_dims->c;
        int8_t *out = output_data;
        int32_t lhs_rows = 0;

        /* This part implements the im2col function */
        for (int i_out_y = 0; i_out_y < output_y; i_out_y++)
        {
            for (int i_out_x = 0; i_out_x < output_x; i_out_x++)
            {
                const int32_t base_idx_x = stride_x * i_out_x - pad_x;
                const int32_t base_idx_y = stride_y * i_out_y - pad_y;

                for (int32_t i_ker_y = 0; i_ker_y < kernel_y; i_ker_y++)
                {
                    for (int32_t i_ker_x = 0; i_ker_x < kernel_x; i_ker_x++)
                    {
                        const int32_t k_y = base_idx_y + dilation_y * i_ker_y;
                        const int32_t k_x = base_idx_x + dilation_x * i_ker_x;

                        if (k_y < 0 || k_y >= input_y || k_x < 0 || k_x >= input_x)
                        {
                            arm_memset_s8(im2col_buf, (int8_t)-input_offset, sizeof(int8_t) * input_ch);
                        }
                        else
                        {
                            arm_memcpy_s8(im2col_buf, input_data + (k_y * input_x + k_x) * input_ch, input_ch);
                        }
                        im2col_buf += input_ch;
                    }
                }
                lhs_rows++;

                /* Computation is filed for every 4 columns */
                if (lhs_rows == 4)
                {
                    arm_nn_mat_mult_nt_t_s4((int8_t *)buffer_a,
                                            packed_filter_data,
                                            bias_data,
                                            out,
                                            output_mult,
                                            output_shift,
                                            lhs_rows,
                                            rhs_rows,
                                            rhs_cols,
                                            input_offset,
                                            out_offset,
                                            out_activation_min,
                                            out_activation_max,
                                            rhs_cols);
                    out += lhs_rows * rhs_rows;

                    lhs_rows = 0;
                    im2col_buf = (int8_t *)buffer_a;
                }
            }
        }

        /* Handle left over columns */
        if (lhs_rows != 0)
        {
            arm_nn_mat_mult_nt_t_s4((int8_t *)buffer_a,
                                    packed_filter_data,
                                    bias_data,
                                    out,
                                    output_mult,
                                    output_shift,
                                    lhs_rows,
                                    rhs_rows,
                                    rhs_cols,
                                    input_offset,
                                    out_offset,
                                    out_activation_min,
                                    out_activation_max,
                                    rhs_cols);
            out += lhs_rows * rhs_rows;
            lhs_rows = 0;
            im2col_buf = (int8_t *)buffer_a;
        }
#else // #if defined(ARM_MATH_MVEI)
        int16_t *two_column_buf = buffer_a;
        int8_t *out = output_data;
        int32_t lhs_rows = 0;

        /* This part implements the im2col function */
        for (int i_out_y = 0; i_out_y < output_y; i_out_y++)
        {
            for (int i_out_x = 0; i_out_x < output_x; i_out_x++)
            {
                const int32_t base_idx_x = stride_x * i_out_x - pad_x;
                const int32_t base_idx_y = stride_y * i_out_y - pad_y;

                for (int32_t i_ker_y = 0; i_ker_y < kernel_y; i_ker_y++)
                {
                    for (int32_t i_ker_x = 0; i_ker_x < kernel_x; i_ker_x++)
                    {
                        const int32_t k_y = base_idx_y + dilation_y * i_ker_y;
                        const int32_t k_x = base_idx_x + dilation_x * i_ker_x;

                        if (k_y < 0 || k_y >= input_y || k_x < 0 || k_x >= input_x)
                        {
                            /* Filling 0 for out-of-bound paddings */
                            memset(two_column_buf, 0, sizeof(int16_t) * input_ch);
                        }
                        else
                        {
                            /* Copying the pixel data to column */
                            arm_q7_to_q15_with_offset(
                                input_data + (k_y * input_x + k_x) * input_ch, two_column_buf, input_ch, input_offset);
                        }
                        two_column_buf += input_ch;
                    }
                }
                lhs_rows++;
                /* Computation is filed for every 2 columns */
                if (lhs_rows == 2)
                {
                    out = arm_nn_mat_mult_kernel_s4_s16(packed_filter_data,
                                                        buffer_a,
                                                        output_ch,
                                                        output_shift,
                                                        output_mult,
                                                        out_offset,
                                                        out_activation_min,
                                                        out_activation_max,
                                                        rhs_cols,
                                                        bias_data,
                                                        out);

                    /* counter reset */
                    two_column_buf = buffer_a;
                    lhs_rows = 0;
                }
            }

            if (out == NULL)
            {
                return ARM_CMSIS_NN_NO_IMPL_ERROR;
            }
        }

        /* Handle left over columns */
        if (lhs_rows != 0)
        {
            const int8_t *ker_a_ptr = packed_filter_data;
            int i;
            int8_t spilled_ker_a = 0;
            for (i = 0; i < output_ch; i++)
            {
                /* Load the accumulator with bias first */
                int32_t sum = 0;
                if (bias_data)
                {
                    sum = bias_data[i];
                }

                const int16_t *ip_as_col = buffer_a;

                if (rhs_cols % 2 && (i % 2))
                {
                    int16_t ip_b0 = *ip_as_col++;
                    sum += spilled_ker_a * ip_b0;
                }

    #if defined(ARM_MATH_DSP)
                /* 4 multiply and accumulates are done in one loop. */
                uint16_t col_count = rhs_cols / 4;
                while (col_count)
                {
                    int32_t ker_a1, ker_a2;
                    int32_t ip_b1, ip_b2;

                    read_and_pad_s4_ordered(ker_a_ptr, &ker_a1, &ker_a2);
                    ker_a_ptr += 2;
                    ip_b1 = arm_nn_read_q15x2_ia(&ip_as_col);
                    sum = SMLAD(ker_a1, ip_b1, sum);
                    ip_b2 = arm_nn_read_q15x2_ia(&ip_as_col);
                    sum = SMLAD(ker_a2, ip_b2, sum);

                    col_count--;
                }
                col_count = (rhs_cols & 0x3) >> 1;
    #else
                uint16_t col_count = rhs_cols >> 1;
    #endif

                while (col_count)
                {
                    int8_t ker_a0 = (int8_t)(*ker_a_ptr << 4) >> 4;
                    int8_t ker_a1 = *ker_a_ptr >> 4;
                    ker_a_ptr++;

                    int16_t ip_b0 = *ip_as_col++;
                    sum += ker_a0 * ip_b0;

                    ip_b0 = *ip_as_col++;
                    sum += ker_a1 * ip_b0;

                    col_count--;
                }

                if (rhs_cols % 2 && !(i % 2))
                {
                    int8_t ker_a0 = (int8_t)(*ker_a_ptr << 4) >> 4;
                    spilled_ker_a = *ker_a_ptr >> 4;
                    ker_a_ptr++;
                    int16_t ip_b0 = *ip_as_col;

                    sum += ker_a0 * ip_b0;
                }

                sum = arm_nn_requantize(sum, output_mult[i], output_shift[i]);
                sum += out_offset;
                sum = MAX(sum, out_activation_min);
                sum = MIN(sum, out_activation_max);
                *out++ = (int8_t)sum;
            }
        }
#endif
        /* Advance to the next batch */
        input_data += (input_x * input_y * input_ch);
        output_data += (output_x * output_y * output_ch);
    }

    /* Return to application */
    return ARM_CMSIS_NN_SUCCESS;
}

/**
 * @} end of NNConv group
 */
