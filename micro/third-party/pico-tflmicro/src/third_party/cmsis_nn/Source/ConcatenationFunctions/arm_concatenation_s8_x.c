/*
 * SPDX-FileCopyrightText: Copyright 2010-2022 Arm Limited and/or its affiliates <open-source-office@arm.com>
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
 * Title:        arm_concatenation_s8_x.c
 * Description:  s8 version of concatenation along the X axis
 *
 * $Date:        26 October 2022
 * $Revision:    V.1.0.2
 *
 * Target Processor:  Cortex-M cores
 *
 * -------------------------------------------------------------------- */

#include "third_party/cmsis_nn/Include/arm_nnfunctions.h"
#include "third_party/cmsis_nn/Include/arm_nnsupportfunctions.h"

/**
 *  @ingroup Public
 */

/**
 * @addtogroup Concatenation
 * @{
 */

/*
 *  s8 version of concatenation along the X axis
 *
 * Refer to header file for details.
 *
 */
void arm_concatenation_s8_x(const int8_t *input,
                            const uint16_t input_x,
                            const uint16_t input_y,
                            const uint16_t input_z,
                            const uint16_t input_w,
                            int8_t *output,
                            const uint16_t output_x,
                            const uint32_t offset_x)
{
    const uint32_t num_iterations = input_y * input_z * input_w;

    output += offset_x;

    uint32_t i;

    // Copy per row
    for (i = 0; i < num_iterations; ++i)
    {
        arm_memcpy_s8(output, input, input_x);
        input += input_x;
        output += output_x;
    }
}

/**
 * @} end of Concatenation group
 */
