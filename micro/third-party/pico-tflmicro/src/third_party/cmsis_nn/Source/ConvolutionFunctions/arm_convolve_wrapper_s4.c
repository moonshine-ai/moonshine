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
 * Title:        arm_convolve_wrapper_s4.c
 * Description:  s4 convolution layer wrapper function with the main purpose to call the optimal kernel available in
 *               cmsis-nn to perform the convolution. See header files for details.
 *
 * $Date:        27 May 2024
 * $Revision:    V.1.2.0
 *
 * Target :  Arm(R) M-Profile Architecture
 *
 * -------------------------------------------------------------------- */

#include "third_party/cmsis_nn/Include/arm_nnfunctions.h"

/**
 *  @ingroup Public
 */

/**
 * @addtogroup NNConv
 * @{
 */

/*
 * Convolution layer
 *
 * Refer header file for details.
 *
 */

arm_cmsis_nn_status arm_convolve_wrapper_s4(const cmsis_nn_context *ctx,
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
    if ((conv_params->padding.w == 0) && (conv_params->padding.h == 0) && (filter_dims->w == 1) &&
        (filter_dims->h == 1) && (conv_params->dilation.w == 1 && conv_params->dilation.h == 1))
    {
        if ((conv_params->stride.w == 1) && (conv_params->stride.h == 1))
        {
            return arm_convolve_1x1_s4_fast(ctx,
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
        }
        else
        {
            return arm_convolve_1x1_s4(ctx,
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
        }
    }
    else if ((input_dims->h == 1) && conv_params->dilation.w == 1 && (filter_dims->h == 1) &&
             ((conv_params->stride.w * input_dims->c) % 4 == 0) && (input_dims->c == filter_dims->c))
    {
        return arm_convolve_1_x_n_s4(ctx,
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
    }
#if defined(ARM_MATH_MVEI)
    else if (((filter_dims->h * filter_dims->w * input_dims->c) & 0x1) == 0)
    {
        return arm_convolve_even_s4(ctx,
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
    }
#endif
    else
    {
        return arm_convolve_s4(ctx,
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
    }
}
/**
 * @} end of NNConv group
 */
