/*
 * SPDX-FileCopyrightText: Copyright 2022-2024 Arm Limited and/or its affiliates <open-source-office@arm.com>
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
 * Title:        arm_convolve_1x1_s8.c
 * Description:  Generic s8 version of 1x1 convolution
 *
 * $Date:        04 January 2024
 * $Revision:    V.1.1.0
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
 * A more generic version of s8 1x1 convolution intended for non-unity strides. This is slower
 * than the _fast() version if used for unity stride values.
 *
 * Refer header file for details.
 *
 */
arm_cmsis_nn_status arm_convolve_1x1_s8(const cmsis_nn_context *ctx,
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
    (void)ctx;
    (void)filter_dims;
    (void)bias_dims;
    if (conv_params->padding.w != 0 || conv_params->padding.h != 0)
    {
        return ARM_CMSIS_NN_ARG_ERROR;
    }

    const int32_t lhs_rows = output_dims->w;
    const int32_t rhs_rows = output_dims->c;
    const int32_t rhs_cols = input_dims->c;
    const int32_t stride_w = conv_params->stride.w;
    const int32_t input_inc = input_dims->w * conv_params->stride.h * rhs_cols;
    const int32_t output_inc = output_dims->w * rhs_rows;
    const int32_t output_h = output_dims->h;
    const int32_t batch = input_dims->n;
    const int8_t *input_data_ref = input_data;

    for (int i_batch = 0; i_batch < batch; i_batch++)
    {
        input_data = input_data_ref + (i_batch * rhs_cols * input_dims->w * input_dims->h);
        for (int i_output_h = 0; i_output_h < output_h; i_output_h++)
        {
            // Process one input row
            arm_cmsis_nn_status result = arm_nn_mat_mult_nt_t_s8(input_data,
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
                                                                 rhs_cols * stride_w);
            if (result != ARM_CMSIS_NN_SUCCESS)
            {
                return result;
            }
            input_data += input_inc;
            output_data += output_inc;
        }
    }

    /* Return to application */
    return ARM_CMSIS_NN_SUCCESS;
}

/**
 * @} end of NNConv group
 */
