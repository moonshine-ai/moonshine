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
 * Title:        arm_transpose_conv_wrapper_s8.c
 * Description:  Wrapper API to select appropriate transpose conv API based
 *               on dimensions.
 *
 * $Date:        16 October 2024
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
 *  s8 Transpose conv wrapper function
 *
 *  Refer header file for details.
 *
 */
arm_cmsis_nn_status arm_transpose_conv_wrapper_s8(const cmsis_nn_context *ctx,
                                                  const cmsis_nn_context *reverse_conv_ctx,
                                                  const cmsis_nn_transpose_conv_params *transpose_conv_params,
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

    if (ctx->buf == NULL)
    {
        return ARM_CMSIS_NN_ARG_ERROR;
    }

    const bool reverse_conv_possible =
        ((transpose_conv_params->stride.w <= 2) && (transpose_conv_params->stride.h <= 2));
    const bool reverse_conv_efficient = (input_dims->c > REVERSE_TCOL_EFFICIENT_THRESHOLD);

    if (reverse_conv_possible && reverse_conv_efficient)
    {

        if (reverse_conv_ctx->buf == NULL)
        {
            return ARM_CMSIS_NN_ARG_ERROR;
        }

        const int32_t stride_w = transpose_conv_params->stride.w;
        const int32_t stride_h = transpose_conv_params->stride.h;
        const int32_t filter_h = filter_dims->h;
        const int32_t filter_w = filter_dims->w;
        const int32_t output_c = output_dims->c;
        const int32_t input_n = input_dims->n;
        const int32_t input_h = input_dims->h;
        const int32_t input_w = input_dims->w;
        const int32_t input_c = input_dims->c;
        const int32_t padding_w = transpose_conv_params->padding.w;
        const int32_t padding_h = transpose_conv_params->padding.h;

        cmsis_nn_conv_params conv_params;
        conv_params.padding.h = filter_h - 1 - padding_h;
        conv_params.padding.w = filter_w - 1 - padding_w;
        conv_params.input_offset = transpose_conv_params->input_offset;
        conv_params.output_offset = transpose_conv_params->output_offset;
        conv_params.stride.h = 1;
        conv_params.stride.w = 1;
        conv_params.dilation.h = 1;
        conv_params.dilation.w = 1;
        conv_params.activation = transpose_conv_params->activation;

        const cmsis_nn_dims transposed_input_dims = {input_n, input_h * stride_h, input_w * stride_w, input_c};
        const cmsis_nn_dims upscale_dims = {0, stride_h, stride_w, 0};

        // Reverse filter in x and y-dimensions
        int8_t *reversed_filter = reverse_conv_ctx->buf;
        const int8_t *in_ptr = filter_data;
        int8_t *out_ptr = reversed_filter;
        const int32_t filter_size = filter_h * filter_w * input_c;

        out_ptr += filter_size;
        for (int32_t i = 0; i < output_c; i++)
        {
            for (int32_t y = 0; y < filter_h; y++)
            {
                for (int32_t x = 0; x < filter_w; x++)
                {
                    out_ptr -= input_c;
                    arm_memcpy_s8(out_ptr, in_ptr, input_c * sizeof(int8_t));
                    in_ptr += input_c;
                }
            }
            out_ptr += 2 * filter_size;
        }

        return arm_convolve_s8(ctx,
                               &conv_params,
                               quant_params,
                               &transposed_input_dims,
                               input_data,
                               filter_dims,
                               reversed_filter,
                               bias_dims,
                               bias_data,
                               &upscale_dims,
                               output_dims,
                               output_data);
    }
    else
    {

        return arm_transpose_conv_s8(ctx,
                                     reverse_conv_ctx,
                                     transpose_conv_params,
                                     quant_params,
                                     input_dims,
                                     input_data,
                                     filter_dims,
                                     filter_data,
                                     bias_dims,
                                     bias_data,
                                     output_dims,
                                     output_data);
    }
}

/**
 * @} end of NNconv group
 */
