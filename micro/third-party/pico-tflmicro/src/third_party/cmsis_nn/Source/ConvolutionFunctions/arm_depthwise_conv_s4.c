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
 * Title:        arm_depthwise_conv_s4.c
 * Description:  s4 version of depthwise convolution.
 *
 * $Date:        13 February 2024
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

static void depthwise_conv_s4_generic(const int8_t *input,
                                      const int32_t input_batches,
                                      const int32_t input_x,
                                      const int32_t input_y,
                                      const int32_t input_ch,
                                      const int8_t *kernel,
                                      const int32_t output_ch,
                                      const int32_t ch_mult,
                                      const int32_t kernel_x,
                                      const int32_t kernel_y,
                                      const int32_t pad_x,
                                      const int32_t pad_y,
                                      const int32_t stride_x,
                                      const int32_t stride_y,
                                      const int32_t *bias,
                                      int8_t *output,
                                      const int32_t *output_shift,
                                      const int32_t *output_mult,
                                      const int32_t output_x,
                                      const int32_t output_y,
                                      const int32_t output_offset,
                                      const int32_t input_offset,
                                      const int32_t output_activation_min,
                                      const int32_t output_activation_max,
                                      const int32_t dilation_x,
                                      const int32_t dilation_y)

{
    (void)output_ch;
    int i_out = 0;
    int i_batch;

    const int32_t kernel_index_offset = input_ch >> 1;
    if (!(input_ch % 2))
    {
        for (i_batch = 0; i_batch < input_batches; i_batch++)
        {
            for (int i_out_y = 0; i_out_y < output_y; i_out_y++)
            {
                const int16_t base_idx_y = (i_out_y * stride_y) - pad_y;
                for (int i_out_x = 0; i_out_x < output_x; i_out_x++)
                {
                    const int16_t base_idx_x = (i_out_x * stride_x) - pad_x;
                    int idx_out_ch_s4 = 0;
                    int get_low_nibble = 1;

                    // If ch_mult is 1 we can process 2 outputs at a time by doing 2 input_ch iterations
                    if (ch_mult == 1)
                    {
                        for (int i_input_ch = 0; i_input_ch < input_ch; i_input_ch += 2, idx_out_ch_s4++)
                        {
                            int32_t acc_0 = 0;
                            int32_t acc_1 = 0;

                            int ker_y_start;
                            int ker_x_start;
                            int ker_y_end;
                            int ker_x_end;

                            if (dilation_x > 1)
                            {
                                const int32_t start_x_max = (-base_idx_x + dilation_x - 1) / dilation_x;
                                ker_x_start = MAX(0, start_x_max);
                                const int32_t end_min_x = (input_x - base_idx_x + dilation_x - 1) / dilation_x;
                                ker_x_end = MIN(kernel_x, end_min_x);
                            }
                            else
                            {
                                ker_x_start = MAX(0, -base_idx_x);
                                ker_x_end = MIN(kernel_x, input_x - base_idx_x);
                            }

                            if (dilation_y > 1)
                            {
                                const int32_t start_y_max = (-base_idx_y + dilation_y - 1) / dilation_y;
                                ker_y_start = MAX(0, start_y_max);
                                const int32_t end_min_y = (input_y - base_idx_y + dilation_y - 1) / dilation_y;
                                ker_y_end = MIN(kernel_y, end_min_y);
                            }
                            else
                            {
                                ker_y_start = MAX(0, -base_idx_y);
                                ker_y_end = MIN(kernel_y, input_y - base_idx_y);
                            }

                            if (bias)
                            {
                                acc_0 = bias[i_input_ch];
                                acc_1 = bias[i_input_ch + 1];
                            }

                            int32_t idx_y = base_idx_y + dilation_y * ker_y_start;
                            for (int i_ker_y = ker_y_start; i_ker_y < ker_y_end; i_ker_y++)
                            {
                                int32_t idx_x = base_idx_x + dilation_x * ker_x_start;
                                int32_t idx_0 = (idx_y * input_x + idx_x) * input_ch + i_input_ch;

                                int32_t ker_idx_0 =
                                    (i_ker_y * kernel_x + ker_x_start) * kernel_index_offset + idx_out_ch_s4;

                                for (int i_ker_x = ker_x_start; i_ker_x < ker_x_end; i_ker_x++)
                                {
                                    int8_t ker_val0, ker_val1;

                                    ker_val0 = ((int8_t)(kernel[ker_idx_0] << 4) >> 4);
                                    ker_val1 = (kernel[ker_idx_0] >> 4);

                                    acc_0 += (input[idx_0] + input_offset) * ker_val0;
                                    acc_1 += (input[idx_0 + 1] + input_offset) * ker_val1;

                                    idx_0 += dilation_x * input_ch;
                                    idx_x += dilation_x;
                                    ker_idx_0 += kernel_index_offset;
                                }
                                idx_y += dilation_y;
                            }

                            /* Requantize and clamp output to provided range */
                            acc_0 = arm_nn_requantize(acc_0, output_mult[i_input_ch], output_shift[i_input_ch]);
                            acc_0 += output_offset;
                            acc_0 = MAX(acc_0, output_activation_min);
                            acc_0 = MIN(acc_0, output_activation_max);
                            output[i_out++] = acc_0;

                            acc_1 = arm_nn_requantize(acc_1, output_mult[i_input_ch + 1], output_shift[i_input_ch + 1]);
                            acc_1 += output_offset;
                            acc_1 = MAX(acc_1, output_activation_min);
                            acc_1 = MIN(acc_1, output_activation_max);
                            output[i_out++] = acc_1;
                        }
                    }
                    // if ch_mult is odd and greater than 1, we need to continue to process 1 output at a time
                    else if (ch_mult % 2)
                    {
                        for (int i_input_ch = 0; i_input_ch < input_ch; i_input_ch++)
                        {
                            for (int i_ch_mult = 0; i_ch_mult < ch_mult; i_ch_mult++)
                            {
                                const int idx_out_ch = i_ch_mult + i_input_ch * ch_mult;
                                if (idx_out_ch && (idx_out_ch % 2 == 0))
                                {
                                    idx_out_ch_s4++;
                                }

                                int32_t acc_0 = 0;

                                int ker_y_start;
                                int ker_x_start;
                                int ker_y_end;
                                int ker_x_end;

                                if (dilation_x > 1)
                                {
                                    const int32_t start_x_max = (-base_idx_x + dilation_x - 1) / dilation_x;
                                    ker_x_start = MAX(0, start_x_max);
                                    const int32_t end_min_x = (input_x - base_idx_x + dilation_x - 1) / dilation_x;
                                    ker_x_end = MIN(kernel_x, end_min_x);
                                }
                                else
                                {
                                    ker_x_start = MAX(0, -base_idx_x);
                                    ker_x_end = MIN(kernel_x, input_x - base_idx_x);
                                }

                                if (dilation_y > 1)
                                {
                                    const int32_t start_y_max = (-base_idx_y + dilation_y - 1) / dilation_y;
                                    ker_y_start = MAX(0, start_y_max);
                                    const int32_t end_min_y = (input_y - base_idx_y + dilation_y - 1) / dilation_y;
                                    ker_y_end = MIN(kernel_y, end_min_y);
                                }
                                else
                                {
                                    ker_y_start = MAX(0, -base_idx_y);
                                    ker_y_end = MIN(kernel_y, input_y - base_idx_y);
                                }

                                if (bias)
                                {
                                    acc_0 = bias[idx_out_ch];
                                }

                                int32_t idx_y = base_idx_y + dilation_y * ker_y_start;
                                for (int i_ker_y = ker_y_start; i_ker_y < ker_y_end; i_ker_y++)
                                {
                                    int32_t idx_x = base_idx_x + dilation_x * ker_x_start;
                                    int32_t idx_0 = (idx_y * input_x + idx_x) * input_ch + i_input_ch;

                                    int32_t ker_idx_0 =
                                        (i_ker_y * kernel_x + ker_x_start) * (kernel_index_offset * ch_mult) +
                                        idx_out_ch_s4;

                                    for (int i_ker_x = ker_x_start; i_ker_x < ker_x_end; i_ker_x++)
                                    {
                                        int8_t ker_val0;

                                        if (get_low_nibble)
                                        {
                                            ker_val0 = ((int8_t)(kernel[ker_idx_0] << 4) >> 4);
                                        }
                                        else
                                        {
                                            ker_val0 = (kernel[ker_idx_0] >> 4);
                                        }

                                        acc_0 += (input[idx_0] + input_offset) * ker_val0;

                                        idx_0 += dilation_x * input_ch;
                                        idx_x += dilation_x;
                                        ker_idx_0 += (kernel_index_offset * ch_mult);
                                    }
                                    idx_y += dilation_y;
                                }
                                get_low_nibble = !get_low_nibble;

                                /* Requantize and clamp output to provided range */
                                acc_0 = arm_nn_requantize(acc_0, output_mult[idx_out_ch], output_shift[idx_out_ch]);
                                acc_0 += output_offset;
                                acc_0 = MAX(acc_0, output_activation_min);
                                acc_0 = MIN(acc_0, output_activation_max);
                                output[i_out++] = acc_0;
                            }
                        }
                    }
                    // if ch_mult is even then we can do 2 outputs at a time by processing 2 ch_mult iterations
                    else
                    {
                        for (int i_input_ch = 0; i_input_ch < input_ch; i_input_ch++)
                        {
                            // ch_mult is limited to being a multiple of input_ch.
                            // This means that we can assume ch_mult is a multiple of 2 given that input_ch is even
                            for (int i_ch_mult = 0; i_ch_mult < ch_mult; i_ch_mult += 2, idx_out_ch_s4++)
                            {
                                const int idx_out_ch = i_ch_mult + i_input_ch * ch_mult;

                                int32_t acc_0 = 0;
                                int32_t acc_1 = 0;

                                int ker_y_start;
                                int ker_x_start;
                                int ker_y_end;
                                int ker_x_end;

                                if (dilation_x > 1)
                                {
                                    const int32_t start_x_max = (-base_idx_x + dilation_x - 1) / dilation_x;
                                    ker_x_start = MAX(0, start_x_max);
                                    const int32_t end_min_x = (input_x - base_idx_x + dilation_x - 1) / dilation_x;
                                    ker_x_end = MIN(kernel_x, end_min_x);
                                }
                                else
                                {
                                    ker_x_start = MAX(0, -base_idx_x);
                                    ker_x_end = MIN(kernel_x, input_x - base_idx_x);
                                }

                                if (dilation_y > 1)
                                {
                                    const int32_t start_y_max = (-base_idx_y + dilation_y - 1) / dilation_y;
                                    ker_y_start = MAX(0, start_y_max);
                                    const int32_t end_min_y = (input_y - base_idx_y + dilation_y - 1) / dilation_y;
                                    ker_y_end = MIN(kernel_y, end_min_y);
                                }
                                else
                                {
                                    ker_y_start = MAX(0, -base_idx_y);
                                    ker_y_end = MIN(kernel_y, input_y - base_idx_y);
                                }

                                if (bias)
                                {
                                    acc_0 = bias[idx_out_ch];
                                    acc_1 = bias[idx_out_ch + 1];
                                }

                                int32_t idx_y = base_idx_y + dilation_y * ker_y_start;
                                for (int i_ker_y = ker_y_start; i_ker_y < ker_y_end; i_ker_y++)
                                {
                                    int32_t idx_x = base_idx_x + dilation_x * ker_x_start;
                                    int32_t idx_0 = (idx_y * input_x + idx_x) * input_ch + i_input_ch;

                                    int32_t ker_idx_0 =
                                        (i_ker_y * kernel_x + ker_x_start) * (kernel_index_offset * ch_mult) +
                                        idx_out_ch_s4;

                                    for (int i_ker_x = ker_x_start; i_ker_x < ker_x_end; i_ker_x++)
                                    {
                                        int8_t ker_val0, ker_val1;

                                        ker_val0 = ((int8_t)(kernel[ker_idx_0] << 4) >> 4);
                                        ker_val1 = (kernel[ker_idx_0] >> 4);

                                        acc_0 += (input[idx_0] + input_offset) * ker_val0;
                                        acc_1 += (input[idx_0] + input_offset) * ker_val1;

                                        idx_0 += dilation_x * input_ch;
                                        idx_x += dilation_x;
                                        ker_idx_0 += (kernel_index_offset * ch_mult);
                                    }
                                    idx_y += dilation_y;
                                }

                                /* Requantize and clamp output to provided range */
                                acc_0 = arm_nn_requantize(acc_0, output_mult[idx_out_ch], output_shift[idx_out_ch]);
                                acc_0 += output_offset;
                                acc_0 = MAX(acc_0, output_activation_min);
                                acc_0 = MIN(acc_0, output_activation_max);
                                output[i_out++] = acc_0;

                                acc_1 =
                                    arm_nn_requantize(acc_1, output_mult[idx_out_ch + 1], output_shift[idx_out_ch + 1]);
                                acc_1 += output_offset;
                                acc_1 = MAX(acc_1, output_activation_min);
                                acc_1 = MIN(acc_1, output_activation_max);
                                output[i_out++] = acc_1;
                            }
                        }
                    }
                }
            }
            /* Advance to the next batch */
            input += (input_x * input_y * input_ch);
        }
    }
    else
    {
        for (i_batch = 0; i_batch < input_batches; i_batch++)
        {
            for (int i_out_y = 0; i_out_y < output_y; i_out_y++)
            {
                const int16_t base_idx_y = (i_out_y * stride_y) - pad_y;
                for (int i_out_x = 0; i_out_x < output_x; i_out_x++)
                {
                    const int16_t base_idx_x = (i_out_x * stride_x) - pad_x;
                    int idx_out_ch_s4 = 0;
                    int get_low_nibble = 1;

                    for (int i_input_ch = 0; i_input_ch < input_ch; i_input_ch++)
                    {
                        for (int i_ch_mult = 0; i_ch_mult < ch_mult; i_ch_mult++)
                        {
                            const int idx_out_ch = i_ch_mult + i_input_ch * ch_mult;
                            if (idx_out_ch && (idx_out_ch % 2 == 0))
                            {
                                idx_out_ch_s4++;
                            }

                            int16_t kernel_index_offset_uneven = 0;
                            int32_t acc_0 = 0;

                            int ker_y_start;
                            int ker_x_start;
                            int ker_y_end;
                            int ker_x_end;

                            if (dilation_x > 1)
                            {
                                const int32_t start_x_max = (-base_idx_x + dilation_x - 1) / dilation_x;
                                ker_x_start = MAX(0, start_x_max);
                                const int32_t end_min_x = (input_x - base_idx_x + dilation_x - 1) / dilation_x;
                                ker_x_end = MIN(kernel_x, end_min_x);
                            }
                            else
                            {
                                ker_x_start = MAX(0, -base_idx_x);
                                ker_x_end = MIN(kernel_x, input_x - base_idx_x);
                            }

                            if (dilation_y > 1)
                            {
                                const int32_t start_y_max = (-base_idx_y + dilation_y - 1) / dilation_y;
                                ker_y_start = MAX(0, start_y_max);
                                const int32_t end_min_y = (input_y - base_idx_y + dilation_y - 1) / dilation_y;
                                ker_y_end = MIN(kernel_y, end_min_y);
                            }
                            else
                            {
                                ker_y_start = MAX(0, -base_idx_y);
                                ker_y_end = MIN(kernel_y, input_y - base_idx_y);
                            }

                            if (bias)
                            {
                                acc_0 = bias[idx_out_ch];
                            }
                            int32_t idx_y = base_idx_y + dilation_y * ker_y_start;
                            for (int i_ker_y = ker_y_start; i_ker_y < ker_y_end; i_ker_y++)
                            {
                                int32_t idx_x = base_idx_x + dilation_x * ker_x_start;
                                int32_t idx_0 = (idx_y * input_x + idx_x) * input_ch + i_input_ch;

                                int32_t ker_idx_0 =
                                    (i_ker_y * kernel_x + ker_x_start) * (kernel_index_offset * ch_mult) +
                                    idx_out_ch_s4 + kernel_index_offset_uneven;

                                for (int i_ker_x = ker_x_start; i_ker_x < ker_x_end; i_ker_x++)
                                {
                                    int8_t ker_val;

                                    if (get_low_nibble)
                                    {
                                        get_low_nibble = 0;
                                        ker_val = ((int8_t)(kernel[ker_idx_0] << 4) >> 4);
                                    }
                                    else
                                    {
                                        ker_val = (kernel[ker_idx_0] >> 4);
                                        get_low_nibble = 1;
                                        kernel_index_offset_uneven++;
                                    }

                                    acc_0 += (input[idx_0] + input_offset) * ker_val;
                                    idx_0 += dilation_x * input_ch;
                                    idx_x += dilation_x;
                                    ker_idx_0 += (kernel_index_offset * ch_mult) + get_low_nibble;
                                }
                                idx_y += dilation_y;
                            }
                            if ((kernel_x * kernel_y) % 2)
                            {
                                get_low_nibble = !get_low_nibble;
                            }
                            get_low_nibble = !get_low_nibble;

                            /* Requantize and clamp output to provided range */
                            acc_0 = arm_nn_requantize(acc_0, output_mult[idx_out_ch], output_shift[idx_out_ch]);
                            acc_0 += output_offset;
                            acc_0 = MAX(acc_0, output_activation_min);
                            acc_0 = MIN(acc_0, output_activation_max);

                            output[i_out++] = acc_0;
                        }
                    }
                }
            }

            /* Advance to the next batch */
            input += (input_x * input_y * input_ch);
        }
    }
}

/*
 *  Basic s4 depthwise convolution function.
 *
 *  Refer header file for details.
 *  Optimization using DSP extension is not available for the generic case where channel multiplier is > 1.
 *
 */
arm_cmsis_nn_status arm_depthwise_conv_s4(const cmsis_nn_context *ctx,
                                          const cmsis_nn_dw_conv_params *dw_conv_params,
                                          const cmsis_nn_per_channel_quant_params *quant_params,
                                          const cmsis_nn_dims *input_dims,
                                          const int8_t *input,
                                          const cmsis_nn_dims *filter_dims,
                                          const int8_t *kernel,
                                          const cmsis_nn_dims *bias_dims,
                                          const int32_t *bias,
                                          const cmsis_nn_dims *output_dims,
                                          int8_t *output)
{
    (void)bias_dims;
    (void)ctx;

    const int32_t dilation_x = dw_conv_params->dilation.w;
    const int32_t dilation_y = dw_conv_params->dilation.h;
    depthwise_conv_s4_generic(input,
                              input_dims->n,
                              input_dims->w,
                              input_dims->h,
                              input_dims->c,
                              kernel,
                              output_dims->c,
                              dw_conv_params->ch_mult,
                              filter_dims->w,
                              filter_dims->h,
                              dw_conv_params->padding.w,
                              dw_conv_params->padding.h,
                              dw_conv_params->stride.w,
                              dw_conv_params->stride.h,
                              bias,
                              output,
                              quant_params->shift,
                              quant_params->multiplier,
                              output_dims->w,
                              output_dims->h,
                              dw_conv_params->output_offset,
                              dw_conv_params->input_offset,
                              dw_conv_params->activation.min,
                              dw_conv_params->activation.max,
                              dilation_x,
                              dilation_y);
    /* Return to application */
    return ARM_CMSIS_NN_SUCCESS;
}

/**
 * @} end of NNConv group
 */
