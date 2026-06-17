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
 * Title:        arm_convolve_s16.c
 * Description:  s16 version of convolution.
 *
 * $Date:        22 April 2024
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
 * Basic s16 convolution function.
 *
 * Refer header file for details. Optimal use case for the DSP/MVE implementation is when input and output channels
 * are multiples of 4 or atleast greater than 4.
 *
 */
arm_cmsis_nn_status arm_convolve_s16(const cmsis_nn_context *ctx,
                                     const cmsis_nn_conv_params *conv_params,
                                     const cmsis_nn_per_channel_quant_params *quant_params,
                                     const cmsis_nn_dims *input_dims,
                                     const int16_t *input_data,
                                     const cmsis_nn_dims *filter_dims,
                                     const int8_t *filter_data,
                                     const cmsis_nn_dims *bias_dims,
                                     const cmsis_nn_bias_data *bias_data,
                                     const cmsis_nn_dims *output_dims,
                                     int16_t *output_data)
{
    (void)bias_dims;

    if (ctx->buf == NULL)
    {
        return ARM_CMSIS_NN_ARG_ERROR;
    }
    int16_t *buffer_a = (int16_t *)ctx->buf;

    const int32_t input_batches = input_dims->n;
    const int32_t input_x = input_dims->w;
    const int32_t input_y = input_dims->h;
    const int32_t input_ch = input_dims->c;
    const int32_t kernel_x = filter_dims->w;
    const int32_t kernel_y = filter_dims->h;
    const int32_t output_x = output_dims->w;
    const int32_t output_y = output_dims->h;
    const int32_t output_ch = output_dims->c;
    const int32_t rhs_cols = input_ch * kernel_y * kernel_x;

    const int32_t dilation_x = conv_params->dilation.w;
    const int32_t dilation_y = conv_params->dilation.h;
    const int32_t pad_x = conv_params->padding.w;
    const int32_t pad_y = conv_params->padding.h;
    const int32_t stride_x = conv_params->stride.w;
    const int32_t stride_y = conv_params->stride.h;

    const int32_t out_activation_min = conv_params->activation.min;
    const int32_t out_activation_max = conv_params->activation.max;
    int32_t *output_mult = quant_params->multiplier;
    int32_t *output_shift = quant_params->shift;

#if defined(ARM_MATH_MVEI)
    const int32_t rhs_rows = output_dims->c;
#endif

    for (int i_batch = 0; i_batch < input_batches; i_batch++)
    {
        int16_t *im2col = buffer_a;
        int16_t *out = output_data;

        int32_t lhs_rows = 0;

        /* This part implements the im2col function */
        for (int32_t i_out_y = 0; i_out_y < output_y; i_out_y++)
        {
            for (int32_t i_out_x = 0; i_out_x < output_x; i_out_x++)
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
                            arm_memset_s8((int8_t *)im2col, 0, sizeof(int16_t) * (uint32_t)input_ch);
                        }
                        else
                        {
                            arm_memcpy_s8((int8_t *)im2col,
                                          (const int8_t *)(input_data + (k_y * input_x + k_x) * input_ch),
                                          (uint32_t)input_ch * sizeof(int16_t));
                        }
                        im2col += input_ch;
                    }
                }

                lhs_rows++;
#if defined(ARM_MATH_MVEI)
                /* Computation is filed for every 4 columns */
                if (lhs_rows == 4)
                {
                    arm_nn_mat_mult_nt_t_s16(buffer_a,
                                             filter_data,
                                             bias_data,
                                             out,
                                             output_mult,
                                             output_shift,
                                             lhs_rows,
                                             rhs_rows,
                                             rhs_cols,
                                             out_activation_min,
                                             out_activation_max);
                    out += lhs_rows * output_ch;

                    lhs_rows = 0;
                    im2col = buffer_a;
                }
#else
                /* Computation is filed for every 2 columns */
                if (lhs_rows == 2)
                {
                    out = arm_nn_mat_mult_kernel_s16(filter_data,
                                                     buffer_a,
                                                     output_ch,
                                                     output_shift,
                                                     output_mult,
                                                     out_activation_min,
                                                     out_activation_max,
                                                     rhs_cols,
                                                     bias_data,
                                                     out);

                    /* Counter reset */
                    im2col = buffer_a;
                    lhs_rows = 0;
                }
#endif
            }

            if (out == NULL)
            {
                return ARM_CMSIS_NN_NO_IMPL_ERROR;
            }
        }

        /* Handle left over columns */
        if (lhs_rows != 0)
        {
#if defined(ARM_MATH_MVEI)
            arm_nn_mat_mult_nt_t_s16(buffer_a,
                                     filter_data,
                                     bias_data,
                                     out,
                                     output_mult,
                                     output_shift,
                                     lhs_rows,
                                     rhs_rows,
                                     rhs_cols,
                                     out_activation_min,
                                     out_activation_max);
            out += lhs_rows * rhs_rows;
            lhs_rows = 0;
            im2col = buffer_a;
#else // #if defined(ARM_MATH_MVEI)

            const int64_t *bias_s64 = (const int64_t *)bias_data->data;
            const int32_t *bias_s32 = (const int32_t *)bias_data->data;
            const bool is_int32_bias = bias_data->is_int32_bias;
            const int8_t *ker_a = filter_data;
            int i;

            for (i = 0; i < output_ch; i++)
            {
                /* Init the accumulator*/
                int32_t sum = 0;

                /* Point to the beginning of the im2col buffer where the input is available as a rearranged column */
                const int16_t *ip_as_col = buffer_a;

    #if defined(ARM_MATH_DSP)
                /* 4 multiply and accumulates are done in one loop. */
                int32_t col_count = rhs_cols >> 2;

                while (col_count)
                {
                    int32_t ker_a1, ker_a2;
                    int32_t ip_b1, ip_b2;

                    ker_a = read_and_pad(ker_a, &ker_a1, &ker_a2);

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

                if (is_int32_bias)
                {
                    if (bias_s32)
                    {
                        sum += bias_s32[i];
                    }

                    sum = arm_nn_requantize(sum, output_mult[i], output_shift[i]);
                }
                else
                {
                    int64_t acc_64 = sum;

                    if (bias_s64)
                    {
                        acc_64 += bias_s64[i];
                    }

                    int32_t reduced_multiplier = REDUCE_MULTIPLIER(output_mult[i]);
                    sum = arm_nn_requantize_s64(acc_64, reduced_multiplier, output_shift[i]);
                }

                sum = MAX(sum, out_activation_min);
                sum = MIN(sum, out_activation_max);
                *out++ = (int16_t)sum;
            }
            lhs_rows = 0;

#endif // #if defined(ARM_MATH_MVEI)
        }

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
