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
 * Title:        arm_convolve_even_s4.c
 * Description:  s8 version of convolution using symmetric quantization with 4 bit weights.
 *
 * $Date:        05 Jun 2024
 * $Revision:    V.1.0.0
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
 * Basic s8 convolution function with int4 packed RHS (weights) and even RHS columns,
 *
 * Refer header file for details.
 *
 */
arm_cmsis_nn_status arm_convolve_even_s4(const cmsis_nn_context *ctx,
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

#if defined(ARM_MATH_MVEI)

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

    if (rhs_cols & 0x1)
    {
        return ARM_CMSIS_NN_ARG_ERROR;
    }

    const int32_t blk_cnt = rhs_cols >> 5;

    int32_t *output_mult = quant_params->multiplier;
    int32_t *output_shift = quant_params->shift;

    int i_batch;

    for (i_batch = 0; i_batch < input_batches; i_batch++)
    {
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

                /* Reformat most of the buffer by interleaving it */
                int8_t *im2col_buf_interleaved = (int8_t *)buffer_a + lhs_rows * rhs_cols;
                for (int j = blk_cnt; j > 0; --j)
                {
                    int8x16x2_t x2 = vld2q_s8(im2col_buf_interleaved);

                    vstrbq_s8(im2col_buf_interleaved, x2.val[1]);
                    im2col_buf_interleaved += 16;

                    vstrbq_s8(im2col_buf_interleaved, x2.val[0]);
                    im2col_buf_interleaved += 16;
                }

                lhs_rows++;

                /* Computation is filed for every 4 columns */
                if (lhs_rows == 4)
                {
                    arm_nn_mat_mult_nt_interleaved_t_even_s4((int8_t *)buffer_a,
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
            arm_nn_mat_mult_nt_interleaved_t_even_s4((int8_t *)buffer_a,
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

        /* Advance to the next batch */
        input_data += (input_x * input_y * input_ch);
        output_data += (output_x * output_y * output_ch);
    }
#else
    (void)ctx;
    (void)conv_params;
    (void)quant_params;
    (void)input_dims;
    (void)input_data;
    (void)filter_dims;
    (void)packed_filter_data;
    (void)bias_data;
    (void)output_dims;
    (void)output_data;

    return ARM_CMSIS_NN_NO_IMPL_ERROR;

#endif // #if defined(ARM_MATH_MVEI)

    /* Return to application */
    return ARM_CMSIS_NN_SUCCESS;
}

/**
 * @} end of NNConv group
 */
