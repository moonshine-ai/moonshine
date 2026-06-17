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
 * Title:        arm_depthwise_conv_wrapper_s8.c
 * Description:  Wrapper API to select appropriate depthwise conv API based
 *               on dimensions.
 *
 * $Date:        04 November 2024
 * $Revision:    V.2.2.0
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

#if defined(ARM_MATH_MVEI)
static arm_cmsis_nn_status arm_depthwise_conv_to_conv_s8(const cmsis_nn_context *ctx,
                                                         const cmsis_nn_dw_conv_params *dw_conv_params,
                                                         const cmsis_nn_per_channel_quant_params *quant_params,
                                                         const cmsis_nn_dims *input_dims,
                                                         const int8_t *input,
                                                         const cmsis_nn_dims *filter_dims,
                                                         const int8_t *filter,
                                                         const cmsis_nn_dims *bias_dims,
                                                         const int32_t *bias,
                                                         const cmsis_nn_dims *output_dims,
                                                         int8_t *output)
{
    const cmsis_nn_conv_params conv_params = {dw_conv_params->input_offset,
                                              dw_conv_params->output_offset,
                                              dw_conv_params->stride,
                                              dw_conv_params->padding,
                                              dw_conv_params->dilation,
                                              dw_conv_params->activation};
    const cmsis_nn_dims filter_output_dims = {filter_dims->c, filter_dims->h, filter_dims->w, filter_dims->n};
    int8_t *w_buf =
        ctx->buf + arm_convolve_wrapper_s8_get_buffer_size(&conv_params, input_dims, &filter_output_dims, output_dims);
    const uint32_t perm[4] = {3, 1, 2, 0};
    const cmsis_nn_transpose_params transpose_params = {4, perm};

    arm_cmsis_nn_status status = arm_transpose_s8(filter, w_buf, filter_dims, &filter_output_dims, &transpose_params);

    if (status == ARM_CMSIS_NN_SUCCESS)
    {
        status = arm_convolve_wrapper_s8(ctx,
                                         &conv_params,
                                         quant_params,
                                         input_dims,
                                         input,
                                         &filter_output_dims,
                                         (const int8_t *)w_buf,
                                         bias_dims,
                                         bias,
                                         output_dims,
                                         output);
    }
    return status;
}
#endif

/*
 *  s8 Depthwise conv wrapper function
 *
 *  Refer header file for details.
 *
 */
arm_cmsis_nn_status arm_depthwise_conv_wrapper_s8(const cmsis_nn_context *ctx,
                                                  const cmsis_nn_dw_conv_params *dw_conv_params,
                                                  const cmsis_nn_per_channel_quant_params *quant_params,
                                                  const cmsis_nn_dims *input_dims,
                                                  const int8_t *input,
                                                  const cmsis_nn_dims *filter_dims,
                                                  const int8_t *filter,
                                                  const cmsis_nn_dims *bias_dims,
                                                  const int32_t *bias,
                                                  const cmsis_nn_dims *output_dims,
                                                  int8_t *output)
{
    arm_cmsis_nn_status status = ARM_CMSIS_NN_SUCCESS;

#if defined(ARM_MATH_MVEI)
    if (input_dims->c == 1 && output_dims->c > CONVERT_DW_CONV_WITH_ONE_INPUT_CH_AND_OUTPUT_CH_ABOVE_THRESHOLD)
    {
        return arm_depthwise_conv_to_conv_s8(ctx,
                                             dw_conv_params,
                                             quant_params,
                                             input_dims,
                                             input,
                                             filter_dims,
                                             filter,
                                             bias_dims,
                                             bias,
                                             output_dims,
                                             output);
    }
#endif

    if (1 == dw_conv_params->ch_mult && input_dims->n == 1 && dw_conv_params->dilation.w == 1 &&
        dw_conv_params->dilation.h == 1)
    {
#if !defined(ARM_MATH_MVEI)
        if (filter_dims->w == 3 && filter_dims->h == 3 && dw_conv_params->padding.h <= 1 &&
            dw_conv_params->padding.w <= 1)
        {
            status = arm_depthwise_conv_3x3_s8(ctx,
                                               dw_conv_params,
                                               quant_params,
                                               input_dims,
                                               input,
                                               filter_dims,
                                               filter,
                                               bias_dims,
                                               bias,
                                               output_dims,
                                               output);
        }
        else
#endif
        {
            status = arm_depthwise_conv_s8_opt(ctx,
                                               dw_conv_params,
                                               quant_params,
                                               input_dims,
                                               input,
                                               filter_dims,
                                               filter,
                                               bias_dims,
                                               bias,
                                               output_dims,
                                               output);
        }
    }
    else
    {
        status = arm_depthwise_conv_s8(ctx,
                                       dw_conv_params,
                                       quant_params,
                                       input_dims,
                                       input,
                                       filter_dims,
                                       filter,
                                       bias_dims,
                                       bias,
                                       output_dims,
                                       output);
    }

    /* Return to application */
    return status;
}

/**
 * @} end of NNConv group
 */
