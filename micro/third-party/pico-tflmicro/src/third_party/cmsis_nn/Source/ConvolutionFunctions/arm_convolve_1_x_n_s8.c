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
 * Title:        arm_convolve_1_x_n_s8.c
 * Description:  s8 version of 1xN convolution using symmetric quantization.
 *
 * $Date:        04 November 2024
 * $Revision:    V.3.6.1
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
 * 1xN s8 convolution function.
 *
 * Refer header file for details.
 *
 */
arm_cmsis_nn_status arm_convolve_1_x_n_s8(const cmsis_nn_context *ctx,
                                          const cmsis_nn_conv_params *conv_params,
                                          const cmsis_nn_per_channel_quant_params *quant_params,
                                          const cmsis_nn_dims *input_dims,
                                          const int8_t *input_data,
                                          const cmsis_nn_dims *filter_dims,
                                          const int8_t *filter_data,
                                          const cmsis_nn_dims *bias_dims,
                                          const int32_t *bias_data,
                                          const cmsis_nn_dims *output_dims,
                                          int8_t *output_data)
{
    arm_cmsis_nn_status status = ARM_CMSIS_NN_SUCCESS;

    /* The wrapper API is the ultimate reference for argument check */
    if ((input_dims->h != 1) || conv_params->dilation.w != 1 || ctx->buf == NULL || conv_params->stride.w == 0 ||
        (conv_params->stride.w * input_dims->c % 4 != 0))
    {
        return ARM_CMSIS_NN_ARG_ERROR;
    }

#if defined(ARM_MATH_MVEI)
    (void)bias_dims;

    const int32_t input_x = input_dims->w;
    const int32_t kernel_x = filter_dims->w;
    const int32_t output_x = output_dims->w;
    const int32_t input_ch = input_dims->c;
    const int32_t pad_x = conv_params->padding.w;
    const int32_t stride_x = conv_params->stride.w;

    // Total pad for dilation of 1
    const int32_t total_pad = ((output_x - 1) * stride_x + kernel_x - input_x);
    const int32_t asym_pad = total_pad % 2;

    if (pad_x * 2 + asym_pad != total_pad)
    {
        return ARM_CMSIS_NN_FAILURE;
    }

    const int32_t right_pad_num = pad_x + asym_pad != 0 ? MAX(1, (pad_x + asym_pad + stride_x - 1) / stride_x) : 0;
    const int32_t left_pad_num = pad_x != 0 ? MAX(1, (pad_x + stride_x - 1) / stride_x) : 0;
    const int32_t no_pad_num = MAX(output_x - (right_pad_num + left_pad_num), 0);

    const int32_t pad_size_left = pad_x * input_ch;
    const int32_t pad_size_right = asym_pad ? right_pad_num * input_ch : pad_size_left;

    const int32_t rhs_cols = kernel_x * input_ch;
    const int32_t rhs_rows = output_dims->c;
    const int32_t lhs_offset = input_ch * stride_x;

    if (right_pad_num + no_pad_num + left_pad_num != output_x)
    {
        return arm_convolve_s8(ctx,
                               conv_params,
                               quant_params,
                               input_dims,
                               input_data,
                               filter_dims,
                               filter_data,
                               bias_dims,
                               bias_data,
                               NULL,
                               output_dims,
                               output_data);
    }

    const uint32_t num_elem_left = kernel_x * input_ch;
    const uint32_t num_elem_right = num_elem_left - input_ch;

    for (int i_batch = 0; i_batch < input_dims->n; i_batch++)
    {
        /* Handle left padded sections */
        int32_t lhs_rows = left_pad_num;
        int8_t *im2col = ctx->buf;

        arm_memset_s8(im2col, (int8_t)-conv_params->input_offset, sizeof(int8_t) * (uint32_t)pad_size_left);
        im2col += pad_size_left;
        arm_memcpy_s8(im2col, input_data, sizeof(int8_t) * num_elem_left);

        arm_nn_mat_mult_nt_t_s8((int8_t *)ctx->buf,
                                filter_data,
                                bias_data,
                                output_data,
                                quant_params->multiplier,
                                quant_params->shift,
                                lhs_rows,
                                rhs_rows,
                                rhs_cols,
                                conv_params->input_offset,
                                conv_params->output_offset,
                                conv_params->activation.min,
                                conv_params->activation.max,
                                rhs_rows,
                                lhs_offset);

        output_data += lhs_rows * rhs_rows;

        /* Non padded elements */
        int32_t out_idx = lhs_rows;
        int32_t input_start = stride_x * lhs_rows - pad_x;

        if (input_start < 0)
        {
            return ARM_CMSIS_NN_FAILURE;
        }

        input_start *= input_ch;
        lhs_rows = no_pad_num;

        arm_nn_mat_mult_nt_t_s8(input_data + input_start,
                                filter_data,
                                bias_data,
                                output_data,
                                quant_params->multiplier,
                                quant_params->shift,
                                lhs_rows,
                                rhs_rows,
                                rhs_cols,
                                conv_params->input_offset,
                                conv_params->output_offset,
                                conv_params->activation.min,
                                conv_params->activation.max,
                                rhs_rows,
                                lhs_offset);

        output_data += lhs_rows * rhs_rows;
        out_idx += lhs_rows;

        /* Right padded elements */
        lhs_rows = output_x - out_idx;

        if (lhs_rows < 0)
        {
            return ARM_CMSIS_NN_FAILURE;
        }

        im2col = ctx->buf;
        input_start = (stride_x * (left_pad_num + no_pad_num) - pad_x) * input_ch;

        arm_memcpy_s8(im2col, input_data + input_start, sizeof(int8_t) * num_elem_right);
        im2col += num_elem_right;
        arm_memset_s8(im2col, (int8_t)-conv_params->input_offset, sizeof(int8_t) * (uint32_t)pad_size_right);

        arm_nn_mat_mult_nt_t_s8((int8_t *)ctx->buf,
                                filter_data,
                                bias_data,
                                output_data,
                                quant_params->multiplier,
                                quant_params->shift,
                                lhs_rows,
                                rhs_rows,
                                rhs_cols,
                                conv_params->input_offset,
                                conv_params->output_offset,
                                conv_params->activation.min,
                                conv_params->activation.max,
                                rhs_rows,
                                lhs_offset);

        output_data += lhs_rows * rhs_rows;

        /* Advance to the next batch */
        input_data += (input_x * input_ch);
    }
#else
    status = arm_convolve_s8(ctx,
                             conv_params,
                             quant_params,
                             input_dims,
                             input_data,
                             filter_dims,
                             filter_data,
                             bias_dims,
                             bias_data,
                             NULL,
                             output_dims,
                             output_data);

#endif

    /* Return to application */
    return status;
}

/**
 * @} end of NNConv group
 */
