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
 * Title:        arm_depthwise_conv_get_buffer_sizes_s8.c
 * Description:  Collection of get buffer size functions for the various s8 convolution layer functions.
 *
 * $Date:        1 November 2024
 * $Revision:    V.1.3.0
 *
 * Target :  Arm(R) M-Profile Architecture
 *
 * -------------------------------------------------------------------- */

#include "third_party/cmsis_nn/Include/arm_nnfunctions.h"
#include "third_party/cmsis_nn/Include/arm_nnsupportfunctions.h"

/**
 *  @ingroup NNConv
 */

/**
 * @addtogroup GetBufferSizeNNConv
 * @{
 */

__STATIC_INLINE int32_t
arm_deptwise_conv_s8_one_in_ch_get_buffer_size_mve(const cmsis_nn_dw_conv_params *dw_conv_params,
                                                   const cmsis_nn_dims *input_dims,
                                                   const cmsis_nn_dims *filter_dims,
                                                   const cmsis_nn_dims *output_dims)
{
    const cmsis_nn_dims filter_conv_dims = {filter_dims->c, filter_dims->h, filter_dims->w, filter_dims->n};
    const cmsis_nn_conv_params conv_params = {dw_conv_params->input_offset,
                                              dw_conv_params->output_offset,
                                              dw_conv_params->stride,
                                              dw_conv_params->padding,
                                              dw_conv_params->dilation,
                                              dw_conv_params->activation};

    int32_t size =
        arm_convolve_wrapper_s8_get_buffer_size_mve(&conv_params, input_dims, &filter_conv_dims, output_dims);
    size += filter_dims->c * filter_dims->h * filter_dims->w * filter_dims->n;

    return size;
}

int32_t arm_depthwise_conv_s8_opt_get_buffer_size_mve(const cmsis_nn_dims *input_dims, const cmsis_nn_dims *filter_dims)
{
    (void)input_dims;
    return (4 * CH_IN_BLOCK_MVE * filter_dims->w * filter_dims->h) * (int32_t)sizeof(int8_t);
}

int32_t arm_depthwise_conv_s8_opt_get_buffer_size_dsp(const cmsis_nn_dims *input_dims, const cmsis_nn_dims *filter_dims)
{
    return (input_dims->c * filter_dims->w * filter_dims->h) * sizeof(int16_t);
}

int32_t arm_depthwise_conv_s8_opt_get_buffer_size(const cmsis_nn_dims *input_dims, const cmsis_nn_dims *filter_dims)
{
#if defined(ARM_MATH_MVEI)
    return arm_depthwise_conv_s8_opt_get_buffer_size_mve(input_dims, filter_dims);
#elif defined(ARM_MATH_DSP)
    return arm_depthwise_conv_s8_opt_get_buffer_size_dsp(input_dims, filter_dims);
#else
    (void)input_dims;
    (void)filter_dims;
    return 0;
#endif
}

int32_t arm_depthwise_conv_wrapper_s8_get_buffer_size(const cmsis_nn_dw_conv_params *dw_conv_params,
                                                      const cmsis_nn_dims *input_dims,
                                                      const cmsis_nn_dims *filter_dims,
                                                      const cmsis_nn_dims *output_dims)
{
    int32_t size = 0;

#if defined(ARM_MATH_MVEI)
    if (input_dims->c == 1 && output_dims->c > CONVERT_DW_CONV_WITH_ONE_INPUT_CH_AND_OUTPUT_CH_ABOVE_THRESHOLD)
    {
        return arm_deptwise_conv_s8_one_in_ch_get_buffer_size_mve(dw_conv_params, input_dims, filter_dims, output_dims);
    }
#endif

    if (input_dims->c == output_dims->c && input_dims->n == 1 && dw_conv_params->dilation.w == 1 &&
        dw_conv_params->dilation.h == 1)
    {
#if !defined(ARM_MATH_MVEI)
        if (filter_dims->w == 3 && filter_dims->h == 3 && dw_conv_params->padding.h <= 1 &&
            dw_conv_params->padding.w <= 1)
        {
            return size;
        }
#endif
        size = arm_depthwise_conv_s8_opt_get_buffer_size(input_dims, filter_dims);
    }

    return size;
}

int32_t arm_depthwise_conv_wrapper_s8_get_buffer_size_dsp(const cmsis_nn_dw_conv_params *dw_conv_params,
                                                          const cmsis_nn_dims *input_dims,
                                                          const cmsis_nn_dims *filter_dims,
                                                          const cmsis_nn_dims *output_dims)
{
    int32_t size = 0;

    if (input_dims->c == output_dims->c && input_dims->n == 1 && dw_conv_params->dilation.w == 1 &&
        dw_conv_params->dilation.h == 1)
    {
        if (filter_dims->w == 3 && filter_dims->h == 3 && dw_conv_params->padding.h <= 1 &&
            dw_conv_params->padding.w <= 1)
        {
            return size;
        }
        size = arm_depthwise_conv_s8_opt_get_buffer_size_dsp(input_dims, filter_dims);
    }

    return size;
}

int32_t arm_depthwise_conv_wrapper_s8_get_buffer_size_mve(const cmsis_nn_dw_conv_params *dw_conv_params,
                                                          const cmsis_nn_dims *input_dims,
                                                          const cmsis_nn_dims *filter_dims,
                                                          const cmsis_nn_dims *output_dims)
{
    int32_t size = 0;

    if (input_dims->c == output_dims->c && input_dims->n == 1 && dw_conv_params->dilation.w == 1 &&
        dw_conv_params->dilation.h == 1)
    {
        size = arm_depthwise_conv_s8_opt_get_buffer_size_mve(input_dims, filter_dims);
    }

    if (input_dims->c == 1 && output_dims->c > CONVERT_DW_CONV_WITH_ONE_INPUT_CH_AND_OUTPUT_CH_ABOVE_THRESHOLD)
    {
        const int32_t to_conv_size =
            arm_deptwise_conv_s8_one_in_ch_get_buffer_size_mve(dw_conv_params, input_dims, filter_dims, output_dims);

        /* Special case since this is compiler dependent.
           Note it is recommended to use arm_depthwise_conv_wrapper_s8_get_buffer_size() instead. */
        if (to_conv_size > size)
        {
            return to_conv_size;
        }
    }

    return size;
}

/**
 * @} end of GetBufferSizeNNConv group
 */
