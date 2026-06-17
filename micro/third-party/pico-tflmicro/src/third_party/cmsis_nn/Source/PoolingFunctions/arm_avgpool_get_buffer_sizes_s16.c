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
 * Title:        arm_avgpool_get_buffer_sizes_s16.c
 * Description:  Collection of get buffer size functions for avgpool s16 layer function.
 *
 * $Date:        13 January 2023
 * $Revision:    V.1.0.0
 *
 * Target :  Arm(R) M-Profile Architecture
 *
 * -------------------------------------------------------------------- */

#include "third_party/cmsis_nn/Include/arm_nnfunctions.h"

/**
 *  @ingroup Pooling
 */

/**
 * @addtogroup GetBufferSizePooling
 * @{
 */

int32_t arm_avgpool_s16_get_buffer_size(const int output_x, const int ch_src)
{
#if defined(ARM_MATH_MVEI)
    return arm_avgpool_s16_get_buffer_size_mve(output_x, ch_src);
#elif defined(ARM_MATH_DSP) && !defined(ARM_MATH_MVEI)
    return arm_avgpool_s16_get_buffer_size_dsp(output_x, ch_src);
#else
    (void)output_x;
    (void)ch_src;
    return 0;
#endif
}

int32_t arm_avgpool_s16_get_buffer_size_dsp(const int output_x, const int ch_src)
{
    (void)output_x;
    return (ch_src * sizeof(int32_t));
}

int32_t arm_avgpool_s16_get_buffer_size_mve(const int output_x, const int ch_src)
{
    (void)output_x;
    (void)ch_src;

    return 0;
}

/**
 * @} end of GetBufferSizePooling group
 */
