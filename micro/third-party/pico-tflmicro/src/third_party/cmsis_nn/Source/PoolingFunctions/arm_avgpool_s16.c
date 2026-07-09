/*
 * SPDX-FileCopyrightText: Copyright 2022-2023 Arm Limited and/or its affiliates <open-source-office@arm.com>
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
 * Title:        arm_avgpool_s16.c
 * Description:  Pooling function implementations
 *
 * $Date:        27 November 2023
 * $Revision:    V.2.5.0
 *
 * Target :  Arm(R) M-Profile Architecture
 *
 * -------------------------------------------------------------------- */

#include "third_party/cmsis_nn/Include/arm_nnfunctions.h"
#include "third_party/cmsis_nn/Include/arm_nnsupportfunctions.h"

#if defined(ARM_MATH_DSP) && !defined(ARM_MATH_MVEI)

static void scale_q31_to_q15_and_clamp(const int32_t *buffer,
                                       int16_t *target,
                                       int32_t length,
                                       const int32_t count,
                                       const int act_min,
                                       const int act_max)
{
    const int half_count = count / 2;

    for (int i = 0; i < length; i++)
    {
        int32_t sum = buffer[i] > 0 ? (buffer[i] + half_count) : (buffer[i] - half_count);
        sum = sum / count;
        sum = MAX(sum, act_min);
        sum = MIN(sum, act_max);

        target[i] = (int16_t)sum;
    }
}
#endif

/**
 *  @ingroup Public

 */

/**
 * @addtogroup Pooling
 * @{
 */

/*
 * s16 average pooling function
 *
 * Refer to header file for details.
 *
 */
arm_cmsis_nn_status arm_avgpool_s16(const cmsis_nn_context *ctx,
                                    const cmsis_nn_pool_params *pool_params,
                                    const cmsis_nn_dims *input_dims,
                                    const int16_t *src,
                                    const cmsis_nn_dims *filter_dims,
                                    const cmsis_nn_dims *output_dims,
                                    int16_t *dst)
{
    const int32_t input_y = input_dims->h;
    const int32_t input_x = input_dims->w;
    const int32_t output_y = output_dims->h;
    const int32_t output_x = output_dims->w;
    const int32_t stride_y = pool_params->stride.h;
    const int32_t stride_x = pool_params->stride.w;
    const int32_t kernel_y = filter_dims->h;
    const int32_t kernel_x = filter_dims->w;
    const int32_t pad_y = pool_params->padding.h;
    const int32_t pad_x = pool_params->padding.w;
    const int32_t act_min = pool_params->activation.min;
    const int32_t act_max = pool_params->activation.max;
    const int32_t ch_src = input_dims->c;
    const int32_t batch_input = input_x * input_y * ch_src;
    int32_t batch_cnt = input_dims->n;

    if (batch_cnt < 1)
    {
        return ARM_CMSIS_NN_ARG_ERROR;
    }

#if defined(ARM_MATH_MVEI)
    (void)ctx;

    const int32_t batch_output = output_x * output_y * ch_src;

    while (batch_cnt)
    {
        for (int i_y = 0; i_y < output_y; i_y++)
        {
            for (int i_x = 0; i_x < output_x; i_x++)
            {
                const int32_t k_y_start = MAX(0, i_y * stride_y - pad_y);
                const int32_t k_y_end = MIN(i_y * stride_y - pad_y + kernel_y, input_y);

                const int32_t k_x_start = MAX(0, i_x * stride_x - pad_x);
                const int32_t k_x_end = MIN(i_x * stride_x - pad_x + kernel_x, input_x);

                const int16_t *src_base = src;
                int16_t *out = &dst[ch_src * (i_x + i_y * output_x)];

                int32_t ch_count = (ch_src + 7) / 8;
                int32_t channels = ch_src;

                while (ch_count > 0)
                {
                    int32_t count = 0;

                    int32x4_t sum_1 = vdupq_n_s32(0);
                    int32x4_t sum_2 = vdupq_n_s32(0);
                    // Load store tail predicate
                    const mve_pred16_t ld_st_p = vctp16q(channels);
                    channels -= 8;

                    for (int k_y = k_y_start; k_y < k_y_end; k_y++)
                    {
                        for (int k_x = k_x_start; k_x < k_x_end; k_x++)
                        {
                            const int16_t *src_inner = src_base + (ch_src * (k_x + k_y * input_x));
                            const int16x8_t temp = vldrhq_z_s16(src_inner, ld_st_p);

                            const int32x4_t temp_lo = vmovlbq_s16(temp);
                            const int32x4_t temp_hi = vmovltq_s16(temp);

                            sum_1 = vaddq_s32(sum_1, temp_lo);
                            sum_2 = vaddq_s32(sum_2, temp_hi);

                            count++;
                        }
                    }

                    // Prevent static code issue DIVIDE_BY_ZERO.
                    if (count == 0)
                    {
                        return ARM_CMSIS_NN_ARG_ERROR;
                    }

                    // Perform the following operation
                    // sum = sum > 0 ? (sum + count / 2) / count : (sum - count / 2) / count;
                    const int32_t half_count = count / 2;
                    // Predicate for 'sum > 0' operation
                    mve_pred16_t p = vcmpgtq_n_s32(sum_1, 0);
                    sum_1 = vaddq_m_n_s32(sum_1, sum_1, half_count, p);
                    sum_1 = vsubq_m_n_s32(sum_1, sum_1, half_count, ~p);

                    p = vcmpgtq_n_s32(sum_2, 0);
                    sum_2 = vaddq_m_n_s32(sum_2, sum_2, half_count, p);
                    sum_2 = vsubq_m_n_s32(sum_2, sum_2, half_count, ~p);

                    for (int i = 0; i < 4; i++)
                    {
                        sum_1[i] = sum_1[i] / count;
                        sum_2[i] = sum_2[i] / count;
                    }

                    sum_1 = vmaxq_s32(sum_1, vdupq_n_s32(act_min));
                    sum_1 = vminq_s32(sum_1, vdupq_n_s32(act_max));

                    sum_2 = vmaxq_s32(sum_2, vdupq_n_s32(act_min));
                    sum_2 = vminq_s32(sum_2, vdupq_n_s32(act_max));

                    int16x8_t temp = vdupq_n_s16(0);
                    temp = vmovnbq_s32(temp, sum_1);
                    temp = vmovntq_s32(temp, sum_2);

                    vstrhq_p_s16(out, temp, ld_st_p);

                    out += 8;
                    ch_count--;
                    src_base += 8;
                }
            }
        }
        src += batch_input;
        dst += batch_output;

        batch_cnt--;
    }

#elif defined(ARM_MATH_DSP)
    /* Run the following code for CPU's with DSP extension
     */
    int32_t *buffer = (int32_t *)ctx->buf;

    if (buffer == NULL)
    {
        return ARM_CMSIS_NN_ARG_ERROR;
    }

    while (batch_cnt)
    {

        for (int i_y = 0, idx_y = -pad_y; i_y < output_y; idx_y += stride_y, i_y++)
        {
            for (int i_x = 0, idx_x = -pad_x; i_x < output_x; idx_x += stride_x, i_x++)
            {
                /* Condition for kernel start dimension:
                   (base_idx_<x,y> + kernel_<x,y>_start) >= 0 */
                const int32_t kernel_y_start = MAX(0, -idx_y);
                const int32_t kernel_x_start = MAX(0, -idx_x);

                /* Condition for kernel end dimension:
                   (base_idx_<x,y> + kernel_<x,y>_end) < dim_src_<width,height> */
                const int32_t kernel_y_end = MIN(kernel_y, input_y - idx_y);
                const int32_t kernel_x_end = MIN(kernel_x, input_x - idx_x);

                int count = 0;

                for (int k_y = kernel_y_start; k_y < kernel_y_end; k_y++)
                {
                    for (int k_x = kernel_x_start; k_x < kernel_x_end; k_x++)
                    {
                        const int16_t *start = src + ch_src * (k_x + idx_x + (k_y + idx_y) * input_x);

                        if (count == 0)
                        {
                            for (int i = 0; i < ch_src; i++)
                            {
                                buffer[i] = start[i];
                            }
                        }
                        else
                        {
                            for (int i = 0; i < ch_src; i++)
                            {
                                buffer[i] = QADD(start[i], buffer[i]);
                            }
                        }
                        count++;
                    }
                }

                // Prevent static code issue DIVIDE_BY_ZERO.
                if (count == 0)
                {
                    return ARM_CMSIS_NN_ARG_ERROR;
                }

                scale_q31_to_q15_and_clamp(buffer, dst, ch_src, count, act_min, act_max);
                dst += ch_src;
            }
        }
        src += batch_input;

        batch_cnt--;
    }

#else
    /* Reference C code adapted from CMSIS-NN arm_avgpool_s8.c.
     */
    const int32_t batch_output = output_x * output_y * ch_src;
    (void)ctx;

    while (batch_cnt)
    {
        for (int i_y = 0, base_idx_y = -pad_y; i_y < output_y; base_idx_y += stride_y, i_y++)
        {
            for (int i_x = 0, base_idx_x = -pad_x; i_x < output_x; base_idx_x += stride_x, i_x++)
            {
                /* Condition for kernel start dimension: (base_idx_<x,y> + kernel_<x,y>_start) >= 0 */
                const int32_t ker_y_start = MAX(0, -base_idx_y);
                const int32_t ker_x_start = MAX(0, -base_idx_x);

                /* Condition for kernel end dimension: (base_idx_<x,y> + kernel_<x,y>_end) < dim_src_<width,height> */
                const int32_t kernel_y_end = MIN(kernel_y, input_y - base_idx_y);
                const int32_t kernel_x_end = MIN(kernel_x, input_x - base_idx_x);

                for (int i_ch_in = 0; i_ch_in < ch_src; i_ch_in++)
                {
                    int sum = 0;
                    int count = 0;

                    for (int k_y = ker_y_start; k_y < kernel_y_end; k_y++)
                    {
                        for (int k_x = ker_x_start; k_x < kernel_x_end; k_x++)
                        {
                            sum += src[i_ch_in + ch_src * (k_x + base_idx_x + (k_y + base_idx_y) * input_x)];
                            count++;
                        }
                    }

                    // Prevent static code issue DIVIDE_BY_ZERO.
                    if (count == 0)
                    {
                        return ARM_CMSIS_NN_ARG_ERROR;
                    }

                    sum = sum > 0 ? (sum + count / 2) / count : (sum - count / 2) / count;
                    sum = MAX(sum, act_min);
                    sum = MIN(sum, act_max);

                    dst[i_ch_in + ch_src * (i_x + i_y * output_x)] = sum;
                }
            }
        }
        src += batch_input;
        dst += batch_output;

        batch_cnt--;
    }
#endif

    return ARM_CMSIS_NN_SUCCESS;
}

/**
 * @} end of Pooling group
 */
