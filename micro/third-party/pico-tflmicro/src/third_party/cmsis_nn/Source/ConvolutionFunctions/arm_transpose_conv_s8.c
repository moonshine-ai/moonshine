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
 * Title:        arm_transpose_conv_s8.c
 * Description:  s8 version of transposed convolution using symmetric quantization.
 *
 * $Date:        29 October 2024
 * $Revision:    V.2.0.0
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
 * Basic s8 transpose convolution function.
 *
 * Refer header file for details.
 *
 */
arm_cmsis_nn_status arm_transpose_conv_s8(const cmsis_nn_context *ctx,
                                          const cmsis_nn_context *output_ctx,
                                          const cmsis_nn_transpose_conv_params *transpose_conv_params,
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
    (void)bias_dims;
    (void)output_ctx;

    const int32_t activation_min = transpose_conv_params->activation.min;
    const int32_t activation_max = transpose_conv_params->activation.max;

    const int32_t input_ch = input_dims->c;
    const int32_t input_x = input_dims->w;
    const int32_t input_y = input_dims->h;

    const int32_t output_x = output_dims->w;
    const int32_t output_y = output_dims->h;

    const int32_t output_ch = output_dims->c;

    const int32_t filter_x = filter_dims->w;
    const int32_t filter_y = filter_dims->h;

    const int32_t pad_x = transpose_conv_params->padding.w;
    const int32_t pad_y = transpose_conv_params->padding.h;

    const int32_t stride_x = transpose_conv_params->stride.w;
    const int32_t stride_y = transpose_conv_params->stride.h;

    const int32_t *output_multiplier = quant_params->multiplier;
    const int32_t *output_shift = quant_params->shift;

    const int32_t out_offset = transpose_conv_params->output_offset;
    const int32_t input_offset = transpose_conv_params->input_offset;

    const int32_t buf_x_elements = ((input_x - 1) * stride_x + MAX(filter_x, stride_x));
    const int32_t buf_x = buf_x_elements * output_ch;
    const int32_t buf_y = MAX(filter_y, stride_y);
    const int32_t buf_size = buf_y * buf_x;
    int32_t *buf = ctx->buf;
    int32_t batch_cnt = input_dims->n;

    const int8_t *filter = filter_data;
    const int8_t *input = input_data;
    int8_t *output = output_data;

    while (batch_cnt)
    {
        // Reset buf
        if (bias_data)
        {
            for (int x = 0; x < buf_x_elements * buf_y; x++)
            {
                arm_memcpy_s8((int8_t *)(buf + x * output_ch), (const int8_t *)bias_data, output_ch * sizeof(int32_t));
            }
        }
        else
        {
            arm_memset_s8((int8_t *)buf, 0, buf_size * sizeof(int32_t));
        }

        int32_t buf_row = 0;
        for (int j = 0; j < input_y; j++)
        {
            int skip_rows_top = MAX(0, pad_y - j * stride_y);
            int skip_rows_bottom = MAX(0, (j * stride_y + filter_y) - (pad_y + output_y) - 1);

            // Compute output for one row of input
            arm_nn_transpose_conv_row_s8_s32(input,
                                             filter,
                                             buf,
                                             buf_row,
                                             buf_size,
                                             filter_y,
                                             filter_x,
                                             input_ch,
                                             output_ch,
                                             input_offset,
                                             buf_x,
                                             input_x,
                                             stride_x,
                                             skip_rows_top,
                                             skip_rows_bottom);
            input += input_ch * input_x;

            if (skip_rows_top == 0)
            {
                for (int y = 0; y < stride_y; y++)
                {
                    int32_t *buf_out = buf + buf_row;
                    buf_out += output_ch * pad_x;

#if defined(ARM_MATH_MVEI)
                    for (int x = 0; x < output_x; x++)
                    {
                        const int32_t *mult_ptr = output_multiplier;
                        const int32_t *shift_ptr = output_shift;

                        int channel_count = output_ch;
                        for (; channel_count > 0; channel_count -= 4)
                        {
                            mve_pred16_t p = vctp32q((uint32_t)channel_count);

                            int32x4_t result = vldrwq_z_s32(buf_out, p);
                            buf_out += 4;
                            result =
                                arm_requantize_mve_32x4(result, vldrwq_z_s32(mult_ptr, p), vldrwq_z_s32(shift_ptr, p));
                            mult_ptr += 4;
                            shift_ptr += 4;
                            result = vaddq_n_s32(result, out_offset);
                            result = vmaxq_s32(result, vdupq_n_s32(activation_min));
                            result = vminq_s32(result, vdupq_n_s32(activation_max));
                            vstrbq_p_s32(output, result, p);
                            output += 4;
                        }

                        // Correct pointer overshoot due to predication
                        buf_out += channel_count;
                        output += channel_count;
                    }
#else

                    for (int x = 0; x < output_x; x++)
                    {
                        const int32_t *output_multiplier_ptr = output_multiplier;
                        const int32_t *output_shift_ptr = output_shift;
                        for (int z = 0; z < output_ch; z++)
                        {
                            int32_t result = *buf_out++;
                            result = arm_nn_requantize(result, *output_multiplier_ptr++, *output_shift_ptr++);
                            result += out_offset;
                            result = MAX(result, activation_min);
                            result = MIN(result, activation_max);
                            *output++ = result;
                        }
                    }
#endif

                    // Reset the buffer which was just written
                    if (bias_data)
                    {
                        for (int x = 0; x < buf_x_elements; x++)
                        {
                            arm_memcpy_s8((int8_t *)(buf + buf_row + x * output_ch),
                                          (const int8_t *)bias_data,
                                          output_ch * sizeof(int32_t));
                        }
                    }
                    else
                    {
                        arm_memset_s8((int8_t *)(buf + buf_row), 0, buf_x * sizeof(int32_t));
                    }

                    // Next row in the rolling buffer
                    buf_row = (buf_row + buf_x) % buf_size;
                }
            }
        }

        // Write leftover rows
        for (int y = 0; y < filter_y - stride_y; y++)
        {
            int32_t *buf_out = buf + buf_row;
            if ((input_y * stride_y + y >= pad_y) && (input_y * stride_y + y < pad_y + output_y))
            {
                buf_out += output_ch * pad_x;
#if defined(ARM_MATH_MVEI)
                for (int x = 0; x < output_x; x++)
                {
                    const int32_t *mult_ptr = output_multiplier;
                    const int32_t *shift_ptr = output_shift;

                    int channel_count = output_ch;
                    for (; channel_count > 0; channel_count -= 4)
                    {
                        mve_pred16_t p = vctp32q((uint32_t)channel_count);

                        int32x4_t result = vldrwq_z_s32(buf_out, p);
                        buf_out += 4;
                        result = arm_requantize_mve_32x4(result, vldrwq_z_s32(mult_ptr, p), vldrwq_z_s32(shift_ptr, p));
                        mult_ptr += 4;
                        shift_ptr += 4;
                        result = vaddq_n_s32(result, out_offset);
                        result = vmaxq_s32(result, vdupq_n_s32(activation_min));
                        result = vminq_s32(result, vdupq_n_s32(activation_max));
                        vstrbq_p_s32(output, result, p);
                        output += 4;
                    }

                    // Correct pointer overshoot due to predication
                    buf_out += channel_count;
                    output += channel_count;
                }
#else
                for (int x = 0; x < output_x; x++)
                {
                    const int32_t *output_multiplier_ptr = output_multiplier;
                    const int32_t *output_shift_ptr = output_shift;

                    for (int z = 0; z < output_ch; z++)
                    {
                        int32_t result = *buf_out++;

                        result = arm_nn_requantize(result, *output_multiplier_ptr++, *output_shift_ptr++);
                        result += out_offset;
                        result = MAX(result, activation_min);
                        result = MIN(result, activation_max);
                        *output++ = result;
                    }
                }
#endif
            }
            buf_row = (buf_row + buf_x) % buf_size;
        }

        batch_cnt--;
    }

    /* Return to application */
    return ARM_CMSIS_NN_SUCCESS;
}

/**
 * @} end of NNConv group
 */
