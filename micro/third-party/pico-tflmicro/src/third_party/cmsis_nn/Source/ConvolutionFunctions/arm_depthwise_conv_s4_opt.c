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
 * Title:        arm_depthwise_conv_s4_opt.c
 * Description:  Optimized s4 depthwise separable convolution function for
 *               channel multiplier of 1.
 *
 * $Date:        17 April 2024
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
 * Optimized s4 depthwise convolution function with constraint that in_channel equals out_channel
 *
 *  Refer prototype header file for details.
 *
 */

arm_cmsis_nn_status arm_depthwise_conv_s4_opt(const cmsis_nn_context *ctx,
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

    const int32_t input_ch = input_dims->c;
    const int32_t output_ch = output_dims->c;

    /* Check depth multiplier is 1 */
    if (input_ch != output_ch)
    {
        return ARM_CMSIS_NN_ARG_ERROR;
    }

    if (ctx->buf == NULL)
    {
        return ARM_CMSIS_NN_ARG_ERROR;
    }

    const int32_t input_x = input_dims->w;
    const int32_t input_y = input_dims->h;
    const int32_t kernel_x = filter_dims->w;
    const int32_t kernel_y = filter_dims->h;
    const int32_t pad_x = dw_conv_params->padding.w;
    const int32_t pad_y = dw_conv_params->padding.h;
    const int32_t stride_x = dw_conv_params->stride.w;
    const int32_t stride_y = dw_conv_params->stride.h;
    const int32_t *output_shift = quant_params->shift;
    const int32_t *output_mult = quant_params->multiplier;
    const int32_t output_x = output_dims->w;
    const int32_t output_y = output_dims->h;
    const int32_t output_offset = dw_conv_params->output_offset;
    const int32_t input_offset = dw_conv_params->input_offset;
    const int32_t output_activation_min = dw_conv_params->activation.min;
    const int32_t output_activation_max = dw_conv_params->activation.max;
    int16_t *buffer_a = (int16_t *)ctx->buf;

#ifdef ARM_MATH_MVEI
    /* Generate two columns from the input tensor */
    int8_t *lhs_buffer = (int8_t *)buffer_a;
    int8_t *out = output;
    int buffer_count = 0;
    const int32_t kernel_size = kernel_x * kernel_y;

    const int32_t ch_loop = (input_ch + (S4_CH_IN_BLOCK_MVE - 1)) / S4_CH_IN_BLOCK_MVE;
    int32_t remaining_ch = output_ch;
    int32_t active_ch = MIN(S4_CH_IN_BLOCK_MVE, remaining_ch);
    remaining_ch -= S4_CH_IN_BLOCK_MVE;

    for (int i_ch = 0; i_ch < ch_loop; i_ch++)
    {
        out = output + i_ch * S4_CH_IN_BLOCK_MVE;
        const int8_t *input_slice = input + (i_ch * S4_CH_IN_BLOCK_MVE);

        for (int i_out_y = 0, base_idx_y = -pad_y; i_out_y < output_y; base_idx_y += stride_y, i_out_y++)
        {
            for (int i_out_x = 0, base_idx_x = -pad_x; i_out_x < output_x; base_idx_x += stride_x, i_out_x++)
            {
                for (int i_ker_y = base_idx_y; i_ker_y < base_idx_y + kernel_y; i_ker_y++)
                {
                    for (int i_ker_x = base_idx_x; i_ker_x < base_idx_x + kernel_x; i_ker_x++)
                    {
                        if (i_ker_y < 0 || i_ker_y >= input_y || i_ker_x < 0 || i_ker_x >= input_x)
                        {
                            arm_memset_s8(lhs_buffer, (int8_t)-input_offset, (uint32_t)active_ch);
                        }
                        else
                        {
                            arm_memcpy_s8(lhs_buffer,
                                          input_slice + (i_ker_y * input_x + i_ker_x) * input_ch,
                                          (uint32_t)active_ch);
                        }
                        lhs_buffer += S4_CH_IN_BLOCK_MVE;
                    }
                }
                buffer_count++;

                if (buffer_count == 4)
                {
                    const int32_t block_offset = i_ch * S4_CH_IN_BLOCK_MVE;
                    lhs_buffer = (int8_t *)buffer_a;
                    arm_nn_depthwise_conv_nt_t_s4(lhs_buffer,
                                                  kernel + (block_offset >> 1),
                                                  input_offset,
                                                  active_ch,
                                                  input_ch,
                                                  output_shift + block_offset,
                                                  output_mult + block_offset,
                                                  output_offset,
                                                  output_activation_min,
                                                  output_activation_max,
                                                  kernel_size,
                                                  bias + block_offset,
                                                  out);

                    out += (4 * input_ch);
                    buffer_count = 0;
                }
            }
        }
        /* Handle left over buffers */
        lhs_buffer = (int8_t *)buffer_a;

        int8_t *out_base = out;
        const uint32x4_t gather_offset = {0, 0, 1, 1};
        const mve_pred16_t lower_nibble_mask = 3855; // 0000111100001111
        for (int i_buf = 0; i_buf < buffer_count; i_buf++)
        {
            int32_t loop_count = (active_ch + 3) / 4;
            int32_t num_ch_to_process = active_ch;
            out = out_base + (i_buf * input_ch);
            for (int i_loop_cnt = 0, offset = i_ch * S4_CH_IN_BLOCK_MVE; i_loop_cnt < loop_count;
                 num_ch_to_process -= 4, offset += 4, i_loop_cnt++)
            {
                const int8_t *col_0 = lhs_buffer + (kernel_size * S4_CH_IN_BLOCK_MVE * i_buf) + (i_loop_cnt * 4);
                const int8_t *row_0 = kernel + (offset >> 1);
                int32x4_t out_0 = vdupq_n_s32(0);
                if (bias)
                {
                    out_0 = vldrwq_s32(&bias[offset]);
                }

                if (input_ch % 2)
                {
                    int get_low_nibble = 1;
                    for (int i_ker = 0; i_ker < kernel_size; i_ker++)
                    {
                        int32x4_t ker_0;
                        if (get_low_nibble)
                        {
                            ker_0 = vldrbq_gather_offset_s32(row_0, gather_offset);
                            ker_0 = vrshlq_m_n_s32(ker_0, 28, lower_nibble_mask);
                            ker_0 = vshrq_m_n_s32(ker_0, ker_0, 24, lower_nibble_mask);

                            ker_0 = vshrq_n_s32(ker_0, 4);
                        }
                        else
                        {
                            int8_t temp[] = {row_0[0] >> 4,
                                             (int8_t)(row_0[1] << 4) >> 4,
                                             row_0[1] >> 4,
                                             (int8_t)(row_0[2] << 4) >> 4};
                            ker_0 = vldrbq_s32(temp);
                        }

                        int32x4_t ip_0 = vldrbq_s32(col_0);
                        ip_0 = vaddq_n_s32(ip_0, input_offset);
                        out_0 += vmulq_s32(ip_0, ker_0);

                        get_low_nibble = !get_low_nibble;
                        col_0 += S4_CH_IN_BLOCK_MVE;
                        row_0 += (input_ch >> 1) + get_low_nibble;
                    }
                }
                else
                {
                    for (int i_ker = 0; i_ker < kernel_size; i_ker++)
                    {
                        int32x4_t ker_0 = vldrbq_gather_offset_s32(row_0, gather_offset);
                        ker_0 = vrshlq_m_n_s32(ker_0, 28, lower_nibble_mask);
                        ker_0 = vshrq_m_n_s32(ker_0, ker_0, 24, lower_nibble_mask);

                        ker_0 = vshrq_n_s32(ker_0, 4);

                        int32x4_t ip_0 = vldrbq_s32(col_0);
                        ip_0 = vaddq_n_s32(ip_0, input_offset);
                        out_0 += vmulq_s32(ip_0, ker_0);

                        col_0 += S4_CH_IN_BLOCK_MVE;
                        row_0 += input_ch >> 1;
                    }
                }

                const int32x4_t mult = vldrwq_s32(&output_mult[offset]);
                const int32x4_t shift = vldrwq_s32(&output_shift[offset]);

                out_0 = arm_requantize_mve_32x4(out_0, mult, shift);
                out_0 = vaddq_n_s32(out_0, output_offset);
                out_0 = vmaxq_s32(out_0, vdupq_n_s32(output_activation_min));
                out_0 = vminq_s32(out_0, vdupq_n_s32(output_activation_max));
                mve_pred16_t p = vctp32q((uint32_t)num_ch_to_process);
                vstrbq_p_s32(out, out_0, p);

                out += 4;
            }
        }
        buffer_count = 0;

        active_ch = MIN(S4_CH_IN_BLOCK_MVE, remaining_ch);
        remaining_ch -= S4_CH_IN_BLOCK_MVE;
    }
#else
    int16_t *const col_buffer_start = buffer_a;
    int16_t *col_buffer = col_buffer_start;
    const int32_t *const bias_start_pos = bias;
    const int32_t *const out_mult_start_pos = output_mult;
    const int32_t *const out_shift_start_pos = output_shift;
    const uint16_t num_cols = kernel_x * kernel_y;
    uint16_t row_count;
    uint16_t row_shift = 0;
    uint16_t col_shift = 0;

    for (int i_out_y = 0; i_out_y < output_y; i_out_y++)
    {
        const int16_t base_idx_y = (i_out_y * stride_y) - pad_y;
        for (int i_out_x = 0; i_out_x < output_x; i_out_x++)
        {
            const int16_t base_idx_x = (i_out_x * stride_x) - pad_x;

            /* Out of bounds is only considered for the y axis as it provides a contiguous zero'ing opportunity than
               along the x axis */
            const int ker_y_start = MAX(0, -base_idx_y);
            /* Condition for kernel end dimension: (base_idx_y + ker_y_end) < input_y */
            const int ker_y_end = MIN(kernel_y, input_y - base_idx_y);

            int32_t index = 0;
            if (ker_y_start != 0)
            {
                memset(&col_buffer[index], 0, (kernel_x * input_ch) * ker_y_start * sizeof(int16_t));
                index += (kernel_x * input_ch) * ker_y_start;
            }

            for (int i_ker_y = ker_y_start; i_ker_y < ker_y_end; i_ker_y++)
            {
                const int32_t idx_y = base_idx_y + i_ker_y;

                for (int i_ker_x = 0; i_ker_x < kernel_x; i_ker_x++)
                {
                    const int32_t idx_x = base_idx_x + i_ker_x;
                    if (idx_x < 0 || idx_x >= input_x)
                    {
                        memset(&col_buffer[index], 0, input_ch * sizeof(int16_t));
                    }
                    else
                    {
                        arm_q7_to_q15_with_offset((int8_t *)input + (idx_y * input_x + idx_x) * input_ch,
                                                  &col_buffer[index],
                                                  input_ch,
                                                  (int16_t)input_offset);
                    }
                    index += input_ch;
                }
            }

            const int diff = kernel_y - ker_y_end;
            if (diff != 0)
            {
                memset(&col_buffer[index], 0, (kernel_x * input_ch) * diff * sizeof(int16_t));
            }

            row_count = output_ch / 4;
            row_shift = 0;
            col_shift = 0;
            bias = bias_start_pos;
            output_mult = out_mult_start_pos;
            output_shift = out_shift_start_pos;

            if (output_ch % 2) /* Uneven number of channels */
            {
                int get_low_nibble = 1;

                while (row_count)
                {
                    int32_t sum = 0;
                    int32_t sum_2 = 0;
                    int32_t sum_3 = 0;
                    int32_t sum_4 = 0;
                    if (bias)
                    {
                        sum = *bias++;
                        sum_2 = *bias++;
                        sum_3 = *bias++;
                        sum_4 = *bias++;
                    }

                    uint16_t col_count = num_cols / 2;
                    int16_t *col_pos = col_buffer_start + col_shift;
                    const int8_t *row_pos = kernel + row_shift;

                    row_shift += 2;
                    col_shift += 4;

                    while (col_count)
                    {
    #ifdef ARM_MATH_DSP
                        /* General idea is to read 4 + 4 (input, kernel) pair and re-arrange them in the right order to
                           use in a SMLAD instruction . One run of this loop produces 4 partial outputs with 8 MACs. */
                        /* Note: variable names can be improved here to align with rows and columns. */
                        int32_t ip_a1, ip_a2, ip_b1, ip_b2, op_a, op_b, op_c;

                        /* Read 4 weights */
                        read_and_pad_s4(row_pos, &ip_a2, &ip_b1);
                        read_and_pad_s4_uneven(row_pos + (input_ch >> 1), &ip_a1, &ip_b2);

                        op_a = arm_nn_read_s16x2(col_pos);
                        op_b = arm_nn_read_s16x2(col_pos + input_ch);

                        op_c = PKHBT(op_b, op_a, 16);
                        op_a = PKHTB(op_b, op_a, 16);
                        op_b = PKHBT(ip_b2, ip_a2, 16);
                        sum = SMLAD(op_c, op_b, sum);

                        op_b = PKHBT(ip_b1, ip_a1, 16);

                        sum_2 = SMLAD(op_a, op_b, sum_2);

                        op_a = arm_nn_read_s16x2(col_pos + 2);
                        op_b = arm_nn_read_s16x2(col_pos + input_ch + 2);

                        op_c = PKHBT(op_b, op_a, 16);
                        op_a = PKHTB(op_b, op_a, 16);
                        op_b = PKHTB(ip_a2, ip_b2, 16);
                        sum_3 = SMLAD(op_c, op_b, sum_3);

                        op_b = PKHTB(ip_a1, ip_b1, 16);
                        sum_4 = SMLAD(op_a, op_b, sum_4);

    #else
                        int8_t ker0, ker1, ker2, ker3, ker00, ker11;

                        ker00 = row_pos[0];
                        ker11 = row_pos[1];
                        ker0 = (int8_t)(ker00 << 4) >> 4;
                        ker1 = ker00 >> 4;
                        ker2 = (int8_t)(ker11 << 4) >> 4;
                        ker3 = ker11 >> 4;

                        sum += ker0 * col_pos[0];
                        sum_2 += ker1 * col_pos[1];
                        sum_3 += ker2 * col_pos[2];
                        sum_4 += ker3 * col_pos[3];

                        ker11 = row_pos[1 + (input_ch >> 1)];
                        ker0 = row_pos[0 + (input_ch >> 1)] >> 4;
                        ker1 = (int8_t)(ker11 << 4) >> 4;
                        ker2 = ker11 >> 4;
                        ker3 = (int8_t)(row_pos[2 + (input_ch >> 1)] << 4) >> 4;

                        sum += ker0 * col_pos[0 + input_ch];
                        sum_2 += ker1 * col_pos[1 + input_ch];
                        sum_3 += ker2 * col_pos[2 + input_ch];
                        sum_4 += ker3 * col_pos[3 + input_ch];

    #endif
                        row_pos += (input_ch);
                        col_pos += input_ch << 1;

                        col_count--;
                    }

                    col_count = num_cols & 0x1;

                    while (col_count)
                    {
                        int8_t ker0, ker1, ker2, ker3, ker00, ker11;

                        ker00 = row_pos[0];
                        ker11 = row_pos[1];

                        ker0 = (int8_t)(ker00 << 4) >> 4;
                        ker1 = ker00 >> 4;

                        ker2 = (int8_t)(ker11 << 4) >> 4;
                        ker3 = ker11 >> 4;

                        sum += ker0 * col_pos[0];
                        sum_2 += ker1 * col_pos[1];
                        sum_3 += ker2 * col_pos[2];
                        sum_4 += ker3 * col_pos[3];

                        row_pos += input_ch >> 1;
                        col_pos += input_ch;

                        col_count--;
                    }

                    sum = arm_nn_requantize(sum, *output_mult++, *output_shift++);
                    sum += output_offset;
                    sum = MAX(sum, output_activation_min);
                    sum = MIN(sum, output_activation_max);
                    *output++ = (int8_t)sum;

                    sum_2 = arm_nn_requantize(sum_2, *output_mult++, *output_shift++);
                    sum_2 += output_offset;
                    sum_2 = MAX(sum_2, output_activation_min);
                    sum_2 = MIN(sum_2, output_activation_max);
                    *output++ = (int8_t)sum_2;
                    sum_3 = arm_nn_requantize(sum_3, *output_mult++, *output_shift++);
                    sum_3 += output_offset;
                    sum_3 = MAX(sum_3, output_activation_min);
                    sum_3 = MIN(sum_3, output_activation_max);
                    *output++ = (int8_t)sum_3;

                    sum_4 = arm_nn_requantize(sum_4, *output_mult++, *output_shift++);
                    sum_4 += output_offset;
                    sum_4 = MAX(sum_4, output_activation_min);
                    sum_4 = MIN(sum_4, output_activation_max);
                    *output++ = (int8_t)sum_4;

                    row_count--;
                }

                row_count = output_ch & 0x3;

                while (row_count)
                {
                    const int16_t *col_pos = col_buffer_start + col_shift;
                    const int8_t *row_pos = kernel + row_shift;
                    int32_t sum = 0;
                    int col_index = 0;

                    if (bias)
                    {
                        sum = *bias++;
                    }

                    col_shift += 1;

                    for (int i = 0; i < num_cols; i++)
                    {
                        int8_t rhs = row_pos[i * (input_ch >> 1) + col_index];
                        int8_t rhs0;
                        int16_t lhs0 = col_pos[i * input_ch];

                        if (get_low_nibble)
                        {
                            rhs0 = (int8_t)(rhs << 4) >> 4;
                            get_low_nibble = 0;
                        }
                        else
                        {
                            rhs0 = rhs >> 4;
                            get_low_nibble = 1;
                            col_index++;
                        }

                        sum += rhs0 * lhs0;
                    }

                    if (num_cols % 2 == 0)
                    {
                        get_low_nibble = !get_low_nibble;
                    }

                    sum = arm_nn_requantize(sum, *output_mult++, *output_shift++);
                    sum += output_offset;
                    sum = MAX(sum, output_activation_min);
                    sum = MIN(sum, output_activation_max);
                    *output++ = (int8_t)sum;

                    row_count--;

                    /* Last row */
                    if (row_count == 1)
                    {
                        row_shift += 1;
                    }
                }
            }
            else /* Even number of channels */
            {
                while (row_count)
                {
                    int32_t sum = 0;
                    int32_t sum_2 = 0;
                    int32_t sum_3 = 0;
                    int32_t sum_4 = 0;
                    if (bias)
                    {
                        sum = *bias++;
                        sum_2 = *bias++;
                        sum_3 = *bias++;
                        sum_4 = *bias++;
                    }

                    uint16_t col_count = num_cols / 2;
                    int16_t *col_pos = col_buffer_start + col_shift;
                    const int8_t *row_pos = kernel + row_shift;

                    row_shift += 2;
                    col_shift += 4;

    #ifdef ARM_MATH_DSP
                    while (col_count)
                    {
                        /* General idea is to read 4 + 4 (input, kernel) pair and re-arrange them in the right order to
                           use in a SMLAD instruction . One run of this loop produces 4 partial outputs with 8 MACs. */
                        /* Note: variable names can be improved here to align with rows and columns. */
                        int32_t ip_a1, ip_a2, ip_b1, ip_b2, op_a, op_b, op_c;

                        /* Read 4 weights */
                        read_and_pad_s4(row_pos, &ip_a2, &ip_b1);
                        read_and_pad_s4(row_pos + (input_ch >> 1), &ip_b2, &ip_a1);

                        op_a = arm_nn_read_s16x2(col_pos);
                        op_b = arm_nn_read_s16x2(col_pos + input_ch);

                        op_c = PKHBT(op_b, op_a, 16);
                        op_a = PKHTB(op_b, op_a, 16);
                        op_b = PKHBT(ip_b2, ip_a2, 16);
                        sum = SMLAD(op_c, op_b, sum);

                        op_b = PKHBT(ip_b1, ip_a1, 16);

                        sum_2 = SMLAD(op_a, op_b, sum_2);

                        op_a = arm_nn_read_s16x2(col_pos + 2);
                        op_b = arm_nn_read_s16x2(col_pos + input_ch + 2);

                        op_c = PKHBT(op_b, op_a, 16);
                        op_a = PKHTB(op_b, op_a, 16);
                        op_b = PKHTB(ip_a2, ip_b2, 16);
                        sum_3 = SMLAD(op_c, op_b, sum_3);

                        op_b = PKHTB(ip_a1, ip_b1, 16);
                        sum_4 = SMLAD(op_a, op_b, sum_4);

                        row_pos += (input_ch);
                        col_pos += input_ch << 1;

                        col_count--;
                    }

                    col_count = num_cols & 0x1;
    #else
                    col_count = num_cols;
    #endif
                    while (col_count)
                    {
                        int8_t ker0, ker1, ker2, ker3, ker00, ker11;

                        ker00 = row_pos[0];
                        ker11 = row_pos[1];

                        ker0 = (int8_t)(ker00 << 4) >> 4;
                        ker1 = ker00 >> 4;

                        ker2 = (int8_t)(ker11 << 4) >> 4;
                        ker3 = ker11 >> 4;

                        sum += ker0 * col_pos[0];
                        sum_2 += ker1 * col_pos[1];
                        sum_3 += ker2 * col_pos[2];
                        sum_4 += ker3 * col_pos[3];

                        row_pos += input_ch >> 1;
                        col_pos += input_ch;

                        col_count--;
                    }

                    sum = arm_nn_requantize(sum, *output_mult++, *output_shift++);
                    sum += output_offset;
                    sum = MAX(sum, output_activation_min);
                    sum = MIN(sum, output_activation_max);
                    *output++ = (int8_t)sum;

                    sum_2 = arm_nn_requantize(sum_2, *output_mult++, *output_shift++);
                    sum_2 += output_offset;
                    sum_2 = MAX(sum_2, output_activation_min);
                    sum_2 = MIN(sum_2, output_activation_max);
                    *output++ = (int8_t)sum_2;
                    sum_3 = arm_nn_requantize(sum_3, *output_mult++, *output_shift++);
                    sum_3 += output_offset;
                    sum_3 = MAX(sum_3, output_activation_min);
                    sum_3 = MIN(sum_3, output_activation_max);
                    *output++ = (int8_t)sum_3;

                    sum_4 = arm_nn_requantize(sum_4, *output_mult++, *output_shift++);
                    sum_4 += output_offset;
                    sum_4 = MAX(sum_4, output_activation_min);
                    sum_4 = MIN(sum_4, output_activation_max);
                    *output++ = (int8_t)sum_4;

                    row_count--;
                }

                if (output_ch & 0x2)
                {
                    const int16_t *col_pos = col_buffer_start + col_shift;
                    const int16_t *col_pos_2 = col_buffer_start + col_shift + 1;
                    const int8_t *row_pos = kernel + row_shift;
                    int32_t sum = 0;
                    int32_t sum2 = 0;

                    if (bias)
                    {
                        sum = *bias++;
                        sum2 = *bias++;
                    }

                    for (int i = 0; i < num_cols; i++)
                    {
                        int8_t rhs = row_pos[i * (input_ch >> 1)];

                        int8_t rhs_low = (int8_t)(rhs << 4) >> 4;
                        int8_t rhs_high = rhs >> 4;

                        int16_t lhs0 = col_pos[i * input_ch];
                        int16_t lhs1 = col_pos_2[i * input_ch];

                        sum += rhs_low * lhs0;
                        sum2 += rhs_high * lhs1;
                    }

                    sum = arm_nn_requantize(sum, *output_mult++, *output_shift++);
                    sum += output_offset;
                    sum = MAX(sum, output_activation_min);
                    sum = MIN(sum, output_activation_max);
                    *output++ = (int8_t)sum;
                    sum2 = arm_nn_requantize(sum2, *output_mult++, *output_shift++);
                    sum2 += output_offset;
                    sum2 = MAX(sum2, output_activation_min);
                    sum2 = MIN(sum2, output_activation_max);
                    *output++ = (int8_t)sum2;
                }
            }

            /* Clear counter and pointers */
            col_buffer = col_buffer_start;
        }
    }
#endif // ARM_MATH_MVEI
    /* Return to application */
    return ARM_CMSIS_NN_SUCCESS;
}

/**
 * @} end of NNConv group
 */
