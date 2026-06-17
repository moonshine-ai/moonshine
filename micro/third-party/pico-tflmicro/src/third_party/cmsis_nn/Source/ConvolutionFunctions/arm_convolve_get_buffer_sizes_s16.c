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
 * Title:        arm_convolve_get_buffer_sizes_s16.c
 * Description:  Collection of get buffer size functions for the various s16 convolution layer functions.
 *
 * $Date:        20 March 2024
 * $Revision:    V.2.0.0
 *
 * Target :  Arm(R) M-Profile Architecture
 *
 * -------------------------------------------------------------------- */

#include "third_party/cmsis_nn/Include/Internal/arm_nn_compiler.h"
#include "third_party/cmsis_nn/Include/arm_nnfunctions.h"

/**
 *  @ingroup NNConv
 */

/**
 * @addtogroup GetBufferSizeNNConv
 * @{
 */

__STATIC_INLINE int32_t arm_convolve_s16_get_buffer_size_mve(const cmsis_nn_dims *input_dims,
                                                             const cmsis_nn_dims *filter_dims)
{
    int32_t col_length = input_dims->c * filter_dims->w * filter_dims->h;
    // Get number of complete lanes with int16 elements (multiple of 8) for given col_length. This is dependent on
    // implementation of arm_nn_mat_mult_nt_t_s16
    col_length = (col_length + 7) / 8;
    // 4 -> number of im2col buffers, 8 -> 8 elements per Q register
    return 4 * col_length * 8 * (int32_t)sizeof(int16_t);
}

int32_t arm_convolve_s16_get_buffer_size(const cmsis_nn_dims *input_dims, const cmsis_nn_dims *filter_dims)
{
#if defined(ARM_MATH_MVEI)
    return arm_convolve_s16_get_buffer_size_mve(input_dims, filter_dims);
#else
    return (2 * input_dims->c * filter_dims->w * filter_dims->h) * (int32_t)sizeof(int16_t);
#endif
}

/*
 * Get the required buffer size for arm_convolve_wrapper_s16. This is the recommended function convolve wrapper s16
 * function.
 *
 * Refer to header file for details.
 *
 */
int32_t arm_convolve_wrapper_s16_get_buffer_size(const cmsis_nn_conv_params *conv_params,
                                                 const cmsis_nn_dims *input_dims,
                                                 const cmsis_nn_dims *filter_dims,
                                                 const cmsis_nn_dims *output_dims)
{
    (void)conv_params;
    (void)output_dims;

    return arm_convolve_s16_get_buffer_size(input_dims, filter_dims);
}

int32_t arm_convolve_wrapper_s16_get_buffer_size_dsp(const cmsis_nn_conv_params *conv_params,
                                                     const cmsis_nn_dims *input_dims,
                                                     const cmsis_nn_dims *filter_dims,
                                                     const cmsis_nn_dims *output_dims)
{
    return arm_convolve_wrapper_s16_get_buffer_size(conv_params, input_dims, filter_dims, output_dims);
}

int32_t arm_convolve_wrapper_s16_get_buffer_size_mve(const cmsis_nn_conv_params *conv_params,
                                                     const cmsis_nn_dims *input_dims,
                                                     const cmsis_nn_dims *filter_dims,
                                                     const cmsis_nn_dims *output_dims)
{
    (void)conv_params;
    (void)output_dims;

    return arm_convolve_s16_get_buffer_size_mve(input_dims, filter_dims);
}

/**
 * @} end of GetBufferSizeNNConv group
 */
