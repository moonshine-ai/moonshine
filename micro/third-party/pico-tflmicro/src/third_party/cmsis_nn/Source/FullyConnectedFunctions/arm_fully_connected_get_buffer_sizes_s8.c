/*
 * SPDX-FileCopyrightText: Copyright 2023 Arm Limited and/or its affiliates <open-source-office@arm.com>
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
 * Title:        arm_fully_connected_get_buffer_sizes_s8.c
 * Description:  Collection of get buffer size functions for fully connected s8 layer function.
 *
 * $Date:        15 August 2023
 * $Revision:    V.1.1.0
 *
 * Target :  Arm(R) M-Profile Architecture
 *
 * -------------------------------------------------------------------- */

#include "third_party/cmsis_nn/Include/arm_nnfunctions.h"

/**
 *  @ingroup FC
 */

/**
 * @addtogroup GetBufferSizeFC
 * @{
 */

int32_t arm_fully_connected_s8_get_buffer_size_dsp(const cmsis_nn_dims *filter_dims)
{
    (void)filter_dims;
    return 0;
}

int32_t arm_fully_connected_s8_get_buffer_size_mve(const cmsis_nn_dims *filter_dims)
{
    return filter_dims->c * sizeof(int32_t);
}

int32_t arm_fully_connected_s8_get_buffer_size(const cmsis_nn_dims *filter_dims)
{
#if defined(ARM_MATH_MVEI)
    return arm_fully_connected_s8_get_buffer_size_mve(filter_dims);
#else
    return arm_fully_connected_s8_get_buffer_size_dsp(filter_dims);
#endif
}

/**
 * @} end of GetBufferSizeFC group
 */
