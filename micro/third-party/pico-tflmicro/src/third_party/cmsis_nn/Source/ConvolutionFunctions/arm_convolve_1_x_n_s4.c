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
 * Title:        arm_convolve_1_x_n_s4.c
 * Description:  s4 version of 1xN convolution using symmetric quantization.
 *
 * $Date:        10 April 2024
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
 * 1xN s4 convolution function.
 *
 * Refer header file for details.
 *
 */

arm_cmsis_nn_status arm_convolve_1_x_n_s4(const cmsis_nn_context *ctx,
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
    int32_t buffer_size = arm_convolve_1_x_n_s4_get_buffer_size(conv_params, input_dims, filter_dims, output_dims);
    /* The wrapper API is the ultimate reference for argument check */
    if ((input_dims->h != 1) || conv_params->dilation.w != 1 || (buffer_size != 0 && ctx->buf == NULL) ||
        conv_params->stride.w == 0 || (conv_params->stride.w * input_dims->c % 4 != 0))
    {
        status = ARM_CMSIS_NN_ARG_ERROR;
        goto out;
    }

#if defined(ARM_MATH_MVEI)
    (void)bias_dims;
    const uint16_t input_x = input_dims->w;
    const uint16_t kernel_x = filter_dims->w;
    const uint16_t output_x = output_dims->w;
    const uint16_t output_ch = output_dims->c;
    const uint16_t input_ch = input_dims->c;
    const uint16_t pad_x = conv_params->padding.w;
    const uint16_t stride_x = conv_params->stride.w;

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

    if (right_pad_num + no_pad_num + left_pad_num != output_x)
    {
        return ARM_CMSIS_NN_FAILURE;
    }

    for (int i_batch = 0; i_batch < input_dims->n; i_batch++)
    {
        // Handle left padded sections
        int32_t lhs_rows = left_pad_num;
        const int32_t rhs_cols = kernel_x * input_dims->c;
        const int32_t rhs_rows = output_dims->c;
        const int32_t lhs_offset = input_ch * stride_x;

        int32_t out_idx = 0;

        for (int i = 0; i < lhs_rows; i++)
        {
            const int32_t est_input_x_idx = stride_x * i - pad_x;
            const int32_t ker_begin_idx = -est_input_x_idx;
            const int32_t actual_kernel_len = kernel_x - ker_begin_idx;
            status = arm_nn_mat_mul_core_1x_s4(actual_kernel_len * input_ch,
                                               ker_begin_idx * input_ch,
                                               input_data,
                                               filter_data + ((ker_begin_idx * input_ch) >> 1),
                                               output_ch,
                                               conv_params,
                                               quant_params,
                                               bias_data,
                                               output_data);
            output_data += output_ch;
        }

        out_idx += lhs_rows;
        int32_t input_start = stride_x * lhs_rows - pad_x;

        if (input_start < 0)
        {
            return ARM_CMSIS_NN_FAILURE;
        }
        /* Non padded elements */
        input_start *= input_ch;
        lhs_rows = no_pad_num;
        arm_nn_mat_mult_nt_t_s4(input_data + input_start,
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
                                lhs_offset);

        output_data += lhs_rows * rhs_rows;
        /* Right padded elements */
        out_idx += lhs_rows;
        lhs_rows = output_x - out_idx;

        if (lhs_rows < 0)
        {
            return ARM_CMSIS_NN_FAILURE;
        }

        for (int i = out_idx; i < output_x; i++)
        {
            const int32_t est_input_x_idx = stride_x * i - pad_x;
            const int32_t ker_end_idx = MIN(kernel_x, input_x - est_input_x_idx);
            status = arm_nn_mat_mul_core_1x_s4(ker_end_idx * input_ch,
                                               (kernel_x - ker_end_idx) * input_ch,
                                               input_data + est_input_x_idx * input_ch,
                                               filter_data,
                                               output_ch,
                                               conv_params,
                                               quant_params,
                                               bias_data,
                                               output_data);
            output_data += output_ch;
        }
        /* Advance to the next batch */
        input_data += (input_x * input_ch);
    }
#else
    status = arm_convolve_s4(ctx,
                             conv_params,
                             quant_params,
                             input_dims,
                             input_data,
                             filter_dims,
                             filter_data,
                             bias_dims,
                             bias_data,
                             output_dims,
                             output_data);

#endif

out:
    /* Return to application */
    return status;
}

/**
 * @} end of NNConv group
 */
