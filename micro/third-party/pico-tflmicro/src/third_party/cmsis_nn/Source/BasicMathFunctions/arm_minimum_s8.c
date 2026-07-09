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
 * Title:        arm_minimum_s8
 * Description:  Minimum and Maximum
 *
 * $Date:        08 October 2024
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
 * @addtogroup minimumMaximum
 * @{
 */

static arm_cmsis_nn_status
arm_min_no_broadcast_s8(const int8_t *input_1, const int8_t *input_2, int8_t *output, int32_t flat_size)
{
#if defined(ARM_MATH_MVEI)
    while (flat_size > 0)
    {
        mve_pred16_t p = vctp8q(flat_size);

        int8x16_t vec1 = vldrbq_z_s8(input_1, p);
        input_1 += 16;
        int8x16_t vec2 = vldrbq_z_s8(input_2, p);
        input_2 += 16;

        vstrbq_p_s8(output, vminq_s8(vec1, vec2), p);
        output += 16;
        flat_size -= 16;
    }
#else
    while (flat_size > 0)
    {
        int8_t in1 = *input_1++;
        int8_t in2 = *input_2++;
        *output++ = in1 >= in2 ? in2 : in1;
        --flat_size;
    }
#endif

    return ARM_CMSIS_NN_SUCCESS;
}

static arm_cmsis_nn_status
arm_min_scalar_s8(const int8_t *input_1, const int8_t *input_2, int8_t *output, int32_t flat_size)
{
#if defined(ARM_MATH_MVEI)
    int8x16_t scalar_vec = vdupq_n_s8(*input_1);

    while (flat_size > 0)
    {
        mve_pred16_t p = vctp8q(flat_size);

        int8x16_t vec = vldrbq_z_s8(input_2, p);
        input_2 += 16;

        vstrbq_p_s8(output, vminq_s8(scalar_vec, vec), p);
        output += 16;
        flat_size -= 16;
    }
#else
    int8_t in1 = *input_1;
    while (flat_size > 0)
    {
        int8_t in2 = *input_2++;
        *output++ = in1 >= in2 ? in2 : in1;
        --flat_size;
    }
#endif
    return ARM_CMSIS_NN_SUCCESS;
}

/*
 * s8 minimum
 *
 * Refer header file for details.
 *
 */
arm_cmsis_nn_status arm_minimum_s8(const cmsis_nn_context *ctx,
                                   const int8_t *input_1_data,
                                   const cmsis_nn_dims *input_1_dims,
                                   const int8_t *input_2_data,
                                   const cmsis_nn_dims *input_2_dims,
                                   int8_t *output_data,
                                   const cmsis_nn_dims *output_dims)
{
    (void)ctx;
    const int32_t output_batch = output_dims->n;
    const int32_t output_height = output_dims->h;
    const int32_t output_width = output_dims->w;

    const int32_t input_1_batch = input_1_dims->n;
    const int32_t input_1_height = input_1_dims->h;
    const int32_t input_1_width = input_1_dims->w;
    const int32_t input_1_channels = input_1_dims->c;

    const int32_t input_2_batch = input_2_dims->n;
    const int32_t input_2_height = input_2_dims->h;
    const int32_t input_2_width = input_2_dims->w;
    const int32_t input_2_channels = input_2_dims->c;

    int32_t flat_size_1 = input_1_batch * input_1_height * input_1_width * input_1_channels;
    int32_t flat_size_2 = input_2_batch * input_2_height * input_2_width * input_2_channels;

    if (arm_check_broadcast_required(input_1_dims, input_2_dims))
    {
        if (flat_size_1 == 1)
        {
            // arm_min_scalar expects the tensor with the scalar value to be provided first
            arm_min_scalar_s8(input_1_data, input_2_data, output_data, flat_size_2);
        }
        else if (flat_size_2 == 1)
        {
            // arm_min_scalar expects the tensor with the scalar value to be provided first
            arm_min_scalar_s8(input_2_data, input_1_data, output_data, flat_size_1);
        }
        else
        {
            int32_t width_1_diff = input_1_width >= input_2_width ? 0 : input_1_channels;
            int32_t width_2_diff = input_2_width >= input_1_width ? 0 : input_2_channels;

            int32_t height_1_diff =
                input_1_height >= input_2_height ? width_1_diff : -input_1_width * (input_1_channels - width_1_diff);
            int32_t height_2_diff =
                input_2_height >= input_1_height ? width_2_diff : -input_2_width * (input_2_channels - width_2_diff);

            int32_t batch_1_diff =
                input_1_batch >= input_2_batch ? input_1_channels * input_1_width * input_1_height : 0;
            int32_t batch_2_diff =
                input_2_batch >= input_1_batch ? input_2_channels * input_2_width * input_2_height : 0;

            for (int32_t i_out_batch = 0; i_out_batch < output_batch; i_out_batch++)
            {
                const int8_t *input_1_ptr = input_1_data;
                const int8_t *input_2_ptr = input_2_data;
                flat_size_1 = input_1_height * input_1_width * input_1_channels;
                flat_size_2 = input_2_height * input_2_width * input_2_channels;
                if (input_1_height == input_2_height && input_1_width == input_2_width &&
                    input_1_channels == input_2_channels)
                {
                    arm_min_no_broadcast_s8(input_1_ptr, input_2_ptr, output_data, flat_size_1);
                    output_data += flat_size_1;
                }
                else if (flat_size_1 == 1)
                {
                    arm_min_scalar_s8(input_1_ptr, input_2_ptr, output_data, flat_size_2);
                    output_data += flat_size_2;
                }
                else if (flat_size_2 == 1)
                {
                    arm_min_scalar_s8(input_2_ptr, input_1_ptr, output_data, flat_size_1);
                    output_data += flat_size_1;
                }
                else
                {
                    flat_size_1 = input_1_width * input_1_channels;
                    flat_size_2 = input_2_width * input_2_channels;
                    for (int32_t i_out_height = 0; i_out_height < output_height; i_out_height++)
                    {
                        if (input_1_width == input_2_width && input_1_channels == input_2_channels)
                        {
                            arm_min_no_broadcast_s8(input_1_ptr, input_2_ptr, output_data, flat_size_1);
                            output_data += flat_size_1;
                            input_1_ptr += flat_size_1;
                            input_2_ptr += flat_size_1;
                        }
                        else if (flat_size_1 == 1)
                        {
                            // arm_min_scalar expects the tensor with the scalar value to be provided first
                            arm_min_scalar_s8(input_1_ptr, input_2_ptr, output_data, flat_size_2);
                            output_data += flat_size_2;
                            ++input_1_ptr;
                            input_2_ptr += flat_size_2;
                        }
                        else if (flat_size_2 == 1)
                        {
                            // arm_min_scalar expects the tensor with the scalar value to be provided first
                            arm_min_scalar_s8(input_2_ptr, input_1_ptr, output_data, flat_size_1);
                            output_data += flat_size_1;
                            ++input_2_ptr;
                            input_1_ptr += flat_size_1;
                        }
                        else
                        {
                            for (int32_t i_out_width = 0; i_out_width < output_width; i_out_width++)
                            {
                                if (input_1_channels == input_2_channels)
                                {
                                    arm_min_no_broadcast_s8(input_1_ptr, input_2_ptr, output_data, input_1_channels);
                                    output_data += input_1_channels;
                                    input_1_ptr += input_1_channels;
                                    input_2_ptr += input_1_channels;
                                }
                                else if (input_1_channels == 1)
                                {
                                    // arm_min_scalar expects the tensor with the scalar value to be provided first
                                    arm_min_scalar_s8(input_1_ptr, input_2_ptr, output_data, input_2_channels);
                                    output_data += input_2_channels;
                                    input_1_ptr++;
                                    input_2_ptr += input_2_channels;
                                }
                                else if (input_2_channels == 1)
                                {
                                    // arm_min_scalar expects the tensor with the scalar value to be provided first
                                    arm_min_scalar_s8(input_2_ptr, input_1_ptr, output_data, input_1_channels);
                                    output_data += input_1_channels;
                                    input_1_ptr += input_1_channels;
                                    input_2_ptr++;
                                }
                                input_1_ptr -= width_1_diff;
                                input_2_ptr -= width_2_diff;
                            }
                        }
                        input_1_ptr += height_1_diff;
                        input_2_ptr += height_2_diff;
                    }
                }
                input_1_data += batch_1_diff;
                input_2_data += batch_2_diff;
            }
        }
    }
    else
    {
        arm_min_no_broadcast_s8(input_1_data, input_2_data, output_data, flat_size_1);
    }

    return (ARM_CMSIS_NN_SUCCESS);
}

/**
 * @} end of Doxygen group
 */
