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
 * Title:        arm_transpose_conv_get_buffer_sizes_s8.c
 * Description:  Collection of get buffer size functions for the transpose convolution layer functions.
 *
 * $Date:        29 October 2024
 * $Revision:    V.2.0.0
 *
 * Target :  Arm(R) M-Profile Architecture
 *
 * -------------------------------------------------------------------- */

#include "third_party/cmsis_nn/Include/Internal/arm_nn_compiler.h"
#include "third_party/cmsis_nn/Include/arm_nnfunctions.h"
#include "third_party/cmsis_nn/Include/arm_nnsupportfunctions.h"

/**
 *  @ingroup NNConv
 */

/**
 * @addtogroup GetBufferSizeNNConv
 * @{
 */

/*
 * Get the required buffer size for arm_transpose_conv_s8. This is the recommended transpose conv s8 get buffer size
 * function.
 *
 * Refer to header file for details.
 *
 */
int32_t arm_transpose_conv_s8_get_buffer_size(const cmsis_nn_transpose_conv_params *transpose_conv_params,
                                              const cmsis_nn_dims *input_dims,
                                              const cmsis_nn_dims *filter_dims,
                                              const cmsis_nn_dims *out_dims)
{

    const bool reverse_conv_possible =
        ((transpose_conv_params->stride.w <= 2) && (transpose_conv_params->stride.h <= 2));
    const bool reverse_conv_efficient = (input_dims->c > REVERSE_TCOL_EFFICIENT_THRESHOLD);

    if (reverse_conv_possible && reverse_conv_efficient)
    {
        const cmsis_nn_dims reverse_conv_input_dims = {input_dims->n,
                                                       input_dims->h * transpose_conv_params->stride.h,
                                                       input_dims->w * transpose_conv_params->stride.w,
                                                       input_dims->c};
        return arm_convolve_s8_get_buffer_size(&reverse_conv_input_dims, filter_dims);
    }
    else
    {
        const int32_t buf_x = ((input_dims->w - 1) * transpose_conv_params->stride.w +
                               MAX(filter_dims->w, transpose_conv_params->stride.h)) *
            out_dims->c;
        const int32_t buf_y = MAX(filter_dims->h, transpose_conv_params->stride.h);
        return buf_x * buf_y * sizeof(int32_t);
    }
}

int32_t arm_transpose_conv_s8_get_reverse_conv_buffer_size(const cmsis_nn_transpose_conv_params *transpose_conv_params,
                                                           const cmsis_nn_dims *input_dims,
                                                           const cmsis_nn_dims *filter_dims)
{
    const bool reverse_conv_possible =
        ((transpose_conv_params->stride.w <= 2) && (transpose_conv_params->stride.h <= 2));
    const bool reverse_conv_efficient = (input_dims->c > REVERSE_TCOL_EFFICIENT_THRESHOLD);

    if (reverse_conv_possible && reverse_conv_efficient)
    {
        return input_dims->c * filter_dims->w * filter_dims->h * filter_dims->n;
    }
    else
    {
        return 0;
    }
}

/**
 * @} end of GetBufferSizeNNConv group
 */
