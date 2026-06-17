/*
 * SPDX-FileCopyrightText: Copyright 2010-2024 Arm Limited and/or its affiliates <open-source-office@arm.com>
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
 * Title:        arm_convolve_s8.c
 * Description:  s8 version of convolution using symmetric quantization.
 *
 * $Date:        04 November 2024
 * $Revision:    V.4.0.0
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
 * Basic s8 convolution function.
 *
 * Refer header file for details. Optimal use case for the DSP/MVE implementation is when input and output channels
 * are multiples of 4 or atleast greater than 4.
 *
 */
arm_cmsis_nn_status arm_convolve_s8(const cmsis_nn_context *ctx,
                                    const cmsis_nn_conv_params *conv_params,
                                    const cmsis_nn_per_channel_quant_params *quant_params,
                                    const cmsis_nn_dims *input_dims,
                                    const int8_t *input_data,
                                    const cmsis_nn_dims *filter_dims,
                                    const int8_t *filter_data,
                                    const cmsis_nn_dims *bias_dims,
                                    const int32_t *bias_data,
                                    const cmsis_nn_dims *upscale_dims,
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
    const uint16_t kernel_ch = filter_dims->c;
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
    const int32_t input_offset = conv_params->input_offset;

    const int32_t groups = input_ch / kernel_ch;
    const int32_t rhs_cols = kernel_x * kernel_y * kernel_ch;
    const int32_t output_ch_per_group = output_ch / groups;

    const int32_t *output_mult = quant_params->multiplier;
    const int32_t *output_shift = quant_params->shift;

    if (input_ch % groups != 0 || output_ch % groups != 0)
    {
        return ARM_CMSIS_NN_ARG_ERROR;
    }

    // For upscale_dims == 2, the actual index of the input data is the index of the upscaled input divided by two. In
    // the ordinary case, there is no difference. The division is implemented as a rshift for optimization purposes.
    uint32_t y_rshift = 0;
    uint32_t x_rshift = 0;

    if (upscale_dims)
    {
        y_rshift = upscale_dims->h == 2 ? 1 : 0;
        x_rshift = upscale_dims->w == 2 ? 1 : 0;
    }

    const int32_t input_x_rshifted = input_x >> x_rshift;
    const int32_t input_y_rshifted = input_y >> y_rshift;

    const int32_t remainder = rhs_cols % 4;
    const int32_t aligned_rhs_cols = remainder != 0 ? rhs_cols + 4 - remainder : rhs_cols;

    for (int i_batch = 0; i_batch < input_batches; i_batch++)
    {

#if defined(ARM_MATH_MVEI)
        const int32_t aligned_rhs_cols_offset = aligned_rhs_cols - rhs_cols;

        /* Generate up to four columns from the input tensor a GEMM computation */
        int8_t *im2col_buf = (int8_t *)buffer_a;
#else
        /* Use as a ping-pong buffer for unordered elements */
        int8_t *im2col_buf = (int8_t *)buffer_a + aligned_rhs_cols * 2;
        int16_t *im2col_buf_start_s16 = buffer_a;
#endif
        int32_t lhs_rows = 0;

        const int8_t *filter_data_ptr = &filter_data[0];
        const int32_t *bias_data_ptr = &bias_data[0];
        const int32_t *output_mult_ptr = &output_mult[0];
        const int32_t *output_shift_ptr = &output_shift[0];

        /* This part implements the im2col function */
        for (int32_t i_group = 0; i_group < groups; i_group++)
        {
            int8_t *out = output_data + i_group * output_ch_per_group;
            for (int i_out_y = 0; i_out_y < output_y; i_out_y++)
            {
                for (int i_out_x = 0; i_out_x < output_x; i_out_x++)
                {
                    const int32_t base_idx_x = stride_x * i_out_x - pad_x;
                    const int32_t base_idx_y = stride_y * i_out_y - pad_y;

                    if (y_rshift == 1 || x_rshift == 1)
                    {
                        // Fill complete buf with -input_offset
                        arm_memset_s8(
                            im2col_buf, (int8_t)-input_offset, sizeof(int8_t) * kernel_ch * kernel_x * kernel_y);
                        for (int32_t i_ker_y = 0; i_ker_y < kernel_y; i_ker_y++)
                        {
                            const int32_t k_y = base_idx_y + dilation_y * i_ker_y;

                            //  Don't copy data when padding, or for every second row if stride_y == 2
                            if ((k_y < 0 || k_y >= input_y) || (k_y % 2 && y_rshift == 1))
                            {
                                im2col_buf += kernel_ch * kernel_x;
                            }
                            else
                            {
                                const int32_t k_y_rshifted = k_y >> y_rshift;
                                for (int32_t i_ker_x = 0; i_ker_x < kernel_x; i_ker_x++)
                                {
                                    const int32_t k_x = base_idx_x + dilation_x * i_ker_x;

                                    // Don't copy data when padding, or for every second element if stride_x == 2
                                    if ((k_x >= 0 && k_x < input_x) && ((k_x % 2 == 0) || x_rshift == 0))
                                    {
                                        const int32_t k_x_rshifted = k_x >> x_rshift;
                                        arm_memcpy_s8(im2col_buf,
                                                      input_data +
                                                          (k_y_rshifted * input_x_rshifted + k_x_rshifted) * input_ch,
                                                      sizeof(int8_t) * kernel_ch);
                                    }
                                    im2col_buf += kernel_ch;
                                }
                            }
                        }
                    }
                    else
                    {
                        for (int32_t i_ker_y = 0; i_ker_y < kernel_y; i_ker_y++)
                        {
                            for (int32_t i_ker_x = 0; i_ker_x < kernel_x; i_ker_x++)
                            {
                                const int32_t k_y = base_idx_y + dilation_y * i_ker_y;
                                const int32_t k_x = base_idx_x + dilation_x * i_ker_x;

                                if (k_y < 0 || k_y >= input_y || k_x < 0 || k_x >= input_x)
                                {
                                    arm_memset_s8(im2col_buf, (int8_t)-input_offset, sizeof(int8_t) * kernel_ch);
                                }
                                else
                                {
                                    arm_memcpy_s8(im2col_buf,
                                                  input_data + (k_y * input_x + k_x) * input_ch + i_group * kernel_ch,
                                                  sizeof(int8_t) * kernel_ch);
                                }
                                im2col_buf += kernel_ch;
                            }
                        }
                    }
                    lhs_rows++;

#if defined(ARM_MATH_MVEI)
                    im2col_buf += aligned_rhs_cols_offset;

                    /* Computation is filed for every 4 columns */
                    if (lhs_rows == 4)
                    {
                        arm_nn_mat_mult_nt_t_s8((int8_t *)buffer_a,
                                                filter_data_ptr,
                                                bias_data_ptr,
                                                out,
                                                output_mult_ptr,
                                                output_shift_ptr,
                                                lhs_rows,
                                                output_ch_per_group,
                                                rhs_cols,
                                                input_offset,
                                                out_offset,
                                                out_activation_min,
                                                out_activation_max,
                                                output_ch,
                                                aligned_rhs_cols);

                        out += lhs_rows * output_ch;

                        lhs_rows = 0;
                        im2col_buf = (int8_t *)buffer_a;
                    }
#else
    #if defined(ARM_MATH_DSP)
                    /* Copy one column with input offset and no ordering */
                    arm_s8_to_s16_unordered_with_offset(
                        im2col_buf - rhs_cols, im2col_buf_start_s16, rhs_cols, (int16_t)input_offset);
    #else

                    arm_q7_to_q15_with_offset(
                        im2col_buf - rhs_cols, im2col_buf_start_s16, rhs_cols, (int16_t)input_offset);

    #endif
                    im2col_buf_start_s16 += aligned_rhs_cols;

                    if (lhs_rows == 2)
                    {
                        if (groups > 1)
                        {
                            out = arm_nn_mat_mult_kernel_row_offset_s8_s16(filter_data_ptr,
                                                                           buffer_a,
                                                                           output_ch_per_group,
                                                                           output_shift_ptr,
                                                                           output_mult_ptr,
                                                                           out_offset,
                                                                           out_activation_min,
                                                                           out_activation_max,
                                                                           rhs_cols,
                                                                           aligned_rhs_cols,
                                                                           bias_data_ptr,
                                                                           output_ch,
                                                                           out);
                        }
                        else
                        {
                            out = arm_nn_mat_mult_kernel_s8_s16(filter_data_ptr,
                                                                buffer_a,
                                                                output_ch_per_group,
                                                                output_shift_ptr,
                                                                output_mult_ptr,
                                                                out_offset,
                                                                out_activation_min,
                                                                out_activation_max,
                                                                rhs_cols,
                                                                aligned_rhs_cols,
                                                                bias_data_ptr,
                                                                out);
                        }

                        /* counter reset */
                        im2col_buf_start_s16 = buffer_a;
                        im2col_buf = (int8_t *)buffer_a + aligned_rhs_cols * 2;
                        lhs_rows = 0;
                    }
#endif
                }
            }

            if (out == NULL)
            {
                return ARM_CMSIS_NN_NO_IMPL_ERROR;
            }

            /* Handle left over columns */
            if (lhs_rows != 0)
            {
#if defined(ARM_MATH_MVEI)
                arm_nn_mat_mult_nt_t_s8((int8_t *)buffer_a,
                                        filter_data_ptr,
                                        bias_data_ptr,
                                        out,
                                        output_mult_ptr,
                                        output_shift_ptr,
                                        lhs_rows,
                                        output_ch_per_group,
                                        rhs_cols,
                                        input_offset,
                                        out_offset,
                                        out_activation_min,
                                        out_activation_max,
                                        output_ch,
                                        aligned_rhs_cols);

                out += lhs_rows * output_ch;
                lhs_rows = 0;
                im2col_buf = (int8_t *)buffer_a;
#else // #if defined(ARM_MATH_MVEI)

                const int8_t *ker_a = filter_data_ptr;
                int i;

                for (i = 0; i < output_ch_per_group; i++)
                {
                    /* Load the accumulator with bias first */
                    int32_t sum = 0;
                    if (bias_data_ptr)
                    {
                        sum = bias_data_ptr[i];
                    }

                    const int16_t *ip_as_col = buffer_a;

    #if defined(ARM_MATH_DSP)
                    /* 4 multiply and accumulates are done in one loop. */
                    uint16_t col_count = rhs_cols / 4;
                    while (col_count)
                    {
                        int32_t ker_a1, ker_a2;
                        int32_t ip_b1, ip_b2;

                        ker_a = read_and_pad_reordered(ker_a, &ker_a1, &ker_a2);

                        ip_b1 = arm_nn_read_q15x2_ia(&ip_as_col);
                        sum = SMLAD(ker_a1, ip_b1, sum);
                        ip_b2 = arm_nn_read_q15x2_ia(&ip_as_col);
                        sum = SMLAD(ker_a2, ip_b2, sum);

                        col_count--;
                    }
                    /* Handle left over mac */
                    col_count = rhs_cols & 0x3;
    #else
                    uint16_t col_count = rhs_cols;

    #endif
                    while (col_count)
                    {
                        int8_t ker_a1 = *ker_a++;
                        int16_t ip_b1 = *ip_as_col++;

                        sum += ker_a1 * ip_b1;
                        col_count--;
                    }

                    sum = arm_nn_requantize(sum, output_mult_ptr[i], output_shift_ptr[i]);
                    sum += out_offset;
                    sum = MAX(sum, out_activation_min);
                    sum = MIN(sum, out_activation_max);
                    *out++ = (int8_t)sum;
                }

                im2col_buf_start_s16 = buffer_a;
                im2col_buf = (int8_t *)buffer_a + aligned_rhs_cols * 2;
                lhs_rows = 0;
#endif // #if defined(ARM_MATH_MVEI)
            }
            filter_data_ptr += output_ch_per_group * rhs_cols;
            bias_data_ptr += output_ch_per_group;
            output_mult_ptr += output_ch_per_group;
            output_shift_ptr += output_ch_per_group;
        }
        /* Advance to the next batch */
        input_data += (input_x_rshifted * input_y_rshifted * input_ch);
        output_data += (output_x * output_y * output_ch);
    }

    /* Return to application */
    return ARM_CMSIS_NN_SUCCESS;
}

/**
 * @} end of NNConv group
 */
