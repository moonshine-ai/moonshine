/*
 * SPDX-FileCopyrightText: Copyright 2010-2024 Arm Limited and/or its affiliates
 * <open-source-office@arm.com>
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
 * Title:        arm_nn_mat_mult_kernel_s16.c
 * Description:  Matrix-multiplication function for 16 bits convolution
 *
 * $Date:        12 April 2024
 * $Revision:    V.3.0.0
 *
 * Target :  Arm(R) M-Profile Architecture
 * -------------------------------------------------------------------- */

#include "third_party/cmsis_nn/Include/arm_nnfunctions.h"
#include "third_party/cmsis_nn/Include/arm_nnsupportfunctions.h"

/*
 * moonshine-micro RP2350 modifications (NOT upstream CMSIS-NN):
 *
 *  1. COLUMN BLOCKING (arm_nn_mat_mult_kernel_s16_block): upstream walks
 *     the whole int8 weight matrix for every 2 im2col columns. On the
 *     RP2350 the weights stream from QSPI flash (they far exceed the
 *     16 KiB XIP cache), so a 64-position conv layer re-reads its weight
 *     matrix 32 times per Invoke -- flash bandwidth, not compute, was the
 *     bottleneck. The block kernel loops channel-pairs OUTER and column-
 *     pairs INNER over up to SPELLING_S16_COL_BLOCK buffered columns, so
 *     one channel-pair's weight rows (2 * num_col_a bytes, XIP-cache-hot
 *     after the first pass) serve every column in the block: flash
 *     traffic drops by block/2.
 *  2. DUAL-CORE: with SPELLING_TINY_MULTICORE the output-channel range is
 *     split across both M33 cores via the generic core-1 dispatch in
 *     spelling_multicore.h (implemented in arm_nn_mat_mult_nt_t_s8.c).
 *  3. __not_in_flash_func keeps the hot loops in SRAM so instruction
 *     fetch doesn't contend with weight reads on the XIP bus.
 */
#if defined(PICO_RP2040) || defined(PICO_RP2350) || defined(PICO_BUILD)
#include "pico.h"
#define S16_MM_RAM_FUNC(f) __not_in_flash_func(f)
#else
#define S16_MM_RAM_FUNC(f) f
#endif

#if defined(SPELLING_TINY_MULTICORE) && defined(ARM_MATH_DSP)
#include "third_party/cmsis_nn/Include/spelling_multicore.h"
#ifndef SPELLING_TINY_MC_MIN_ROWS
#define SPELLING_TINY_MC_MIN_ROWS 8
#endif
#endif

/**
 * @ingroup groupSupport
 */

/**
 * @addtogroup supportConvolution
 * @{
 */

#if !defined(ARM_MATH_MVEI)

/*
 * Accumulate one (channel pair) x (column pair) tile: the upstream SMLAD
 * quad loop. acc[0..3] = {ch0*col0, ch0*col1, ch1*col0, ch1*col1} raw
 * int32 sums over the first num_col_fast elements (int32-safe range).
 */
static inline void s16_mm_acc_2x2(const int8_t *ip_a0,
                                  const int16_t *ip_b0,
                                  const int32_t num_col_a,
                                  const int32_t num_col_fast,
                                  int32_t *acc)
{
    const int8_t *ip_a1 = ip_a0 + num_col_a;
    const int16_t *ip_b1 = ip_b0 + num_col_a;

    int32_t ch_0_out_0 = 0;
    int32_t ch_0_out_1 = 0;
    int32_t ch_1_out_0 = 0;
    int32_t ch_1_out_1 = 0;

#if defined(ARM_MATH_DSP)
    uint16_t col_count = (uint16_t)(num_col_fast / 4);

    while (col_count)
    {
        int32_t a01, a02, a11, a12;
        int32_t b0 = arm_nn_read_q15x2_ia(&ip_b0);
        int32_t b1 = arm_nn_read_q15x2_ia(&ip_b1);

        ip_a0 = read_and_pad(ip_a0, &a01, &a02);
        ip_a1 = read_and_pad(ip_a1, &a11, &a12);

        ch_0_out_0 = SMLAD(a01, b0, ch_0_out_0);
        ch_0_out_1 = SMLAD(a01, b1, ch_0_out_1);
        ch_1_out_0 = SMLAD(a11, b0, ch_1_out_0);
        ch_1_out_1 = SMLAD(a11, b1, ch_1_out_1);

        b0 = arm_nn_read_q15x2_ia(&ip_b0);
        b1 = arm_nn_read_q15x2_ia(&ip_b1);

        ch_0_out_0 = SMLAD(a02, b0, ch_0_out_0);
        ch_0_out_1 = SMLAD(a02, b1, ch_0_out_1);
        ch_1_out_0 = SMLAD(a12, b0, ch_1_out_0);
        ch_1_out_1 = SMLAD(a12, b1, ch_1_out_1);

        col_count--;
    }
    col_count = num_col_fast & 0x3;
#else
    int32_t col_count = num_col_fast;
#endif

    while (col_count)
    {
        int8_t a0 = *ip_a0++;
        int16_t b0 = *ip_b0++;
        int8_t a1 = *ip_a1++;
        int16_t b1 = *ip_b1++;

        ch_0_out_0 += a0 * b0;
        ch_0_out_1 += a0 * b1;
        ch_1_out_0 += a1 * b0;
        ch_1_out_1 += a1 * b1;
        col_count--;
    }

    acc[0] = ch_0_out_0;
    acc[1] = ch_0_out_1;
    acc[2] = ch_1_out_0;
    acc[3] = ch_1_out_1;
}

/* Single (channel) x (column) dot product, raw int32 over num_col_fast. */
static inline int32_t s16_mm_acc_1x1(const int8_t *ip_a0,
                                     const int16_t *ip_b0,
                                     const int32_t num_col_fast)
{
    int32_t sum = 0;

#if defined(ARM_MATH_DSP)
    uint16_t col_count = (uint16_t)(num_col_fast / 4);
    while (col_count)
    {
        int32_t a01, a02;
        int32_t b0 = arm_nn_read_q15x2_ia(&ip_b0);
        ip_a0 = read_and_pad(ip_a0, &a01, &a02);
        sum = SMLAD(a01, b0, sum);
        b0 = arm_nn_read_q15x2_ia(&ip_b0);
        sum = SMLAD(a02, b0, sum);
        col_count--;
    }
    col_count = num_col_fast & 0x3;
#else
    int32_t col_count = num_col_fast;
#endif
    while (col_count)
    {
        sum += *ip_a0++ * *ip_b0++;
        col_count--;
    }
    return sum;
}

/* int64 scalar continuation for cols past MAX_COL_COUNT (int64-bias path
 * only, where the int32 accumulator could overflow). */
static inline int64_t s16_mm_acc_slow(const int8_t *ip_a,
                                      const int16_t *ip_b,
                                      const int32_t n)
{
    int64_t sum = 0;
    for (int32_t j = 0; j < n; j++)
    {
        sum += ip_a[j] * ip_b[j];
    }
    return sum;
}

typedef struct
{
    const int8_t *input_a;
    const int16_t *input_b;
    int32_t ch_start;
    int32_t ch_end;
    int32_t output_ch;
    const int32_t *out_shift;
    const int32_t *out_mult;
    int32_t activation_min;
    int32_t activation_max;
    int32_t num_col_a;
    int32_t n_cols;
    const cmsis_nn_bias_data *bias_data;
    int16_t *out_base;
} s16_mm_range_args;

/*
 * Compute output channels [ch_start, ch_end) for all n_cols buffered
 * im2col columns. Channel-pair outer, column-pair inner: one weight-row
 * pair is re-read n_cols/2 times back-to-back, hitting the XIP cache
 * after the first (flash) pass. Output layout: out_base[col * output_ch
 * + ch]. Handles the odd trailing channel and odd trailing column.
 */
static void S16_MM_RAM_FUNC(s16_mm_range)(const s16_mm_range_args *a)
{
    const int64_t *bias_s64 = (const int64_t *)a->bias_data->data;
    const int32_t *bias_s32 = (const int32_t *)a->bias_data->data;
    const bool is_int32_bias = a->bias_data->is_int32_bias;
    const int32_t num_col_a = a->num_col_a;
    const int32_t num_col_fast =
        is_int32_bias ? num_col_a : (num_col_a > MAX_COL_COUNT ? MAX_COL_COUNT : num_col_a);
    const int32_t num_col_slow = num_col_a - num_col_fast;
    const int32_t output_ch = a->output_ch;
    const int32_t n_cols = a->n_cols;
    const int32_t activation_min = a->activation_min;
    const int32_t activation_max = a->activation_max;

    int32_t ch = a->ch_start;

    for (; ch + 2 <= a->ch_end; ch += 2)
    {
        const int8_t *w_row = a->input_a + ch * num_col_a;
        const int32_t mult0 = a->out_mult[ch];
        const int32_t mult1 = a->out_mult[ch + 1];
        const int32_t shift0 = a->out_shift[ch];
        const int32_t shift1 = a->out_shift[ch + 1];

        int32_t col = 0;
        for (; col + 2 <= n_cols; col += 2)
        {
            const int16_t *b_cols = a->input_b + col * num_col_a;
            int32_t acc[4];
            s16_mm_acc_2x2(w_row, b_cols, num_col_a, num_col_fast, acc);

            int16_t *out_c0 = a->out_base + col * output_ch + ch;
            int16_t *out_c1 = out_c0 + output_ch;

            if (is_int32_bias)
            {
                int32_t r00 = acc[0], r01 = acc[1], r10 = acc[2], r11 = acc[3];
                if (bias_s32)
                {
                    r00 += bias_s32[ch];
                    r01 += bias_s32[ch];
                    r10 += bias_s32[ch + 1];
                    r11 += bias_s32[ch + 1];
                }
                r00 = arm_nn_requantize(r00, mult0, shift0);
                r01 = arm_nn_requantize(r01, mult0, shift0);
                r10 = arm_nn_requantize(r10, mult1, shift1);
                r11 = arm_nn_requantize(r11, mult1, shift1);

                r00 = MAX(r00, activation_min); r00 = MIN(r00, activation_max);
                r01 = MAX(r01, activation_min); r01 = MIN(r01, activation_max);
                r10 = MAX(r10, activation_min); r10 = MIN(r10, activation_max);
                r11 = MAX(r11, activation_min); r11 = MIN(r11, activation_max);

                out_c0[0] = (int16_t)r00;
                out_c1[0] = (int16_t)r01;
                out_c0[1] = (int16_t)r10;
                out_c1[1] = (int16_t)r11;
            }
            else
            {
                int64_t s00 = acc[0], s01 = acc[1], s10 = acc[2], s11 = acc[3];
                if (num_col_slow > 0)
                {
                    const int8_t *wa0 = w_row + num_col_fast;
                    const int8_t *wa1 = wa0 + num_col_a;
                    const int16_t *bb0 = b_cols + num_col_fast;
                    const int16_t *bb1 = bb0 + num_col_a;
                    s00 += s16_mm_acc_slow(wa0, bb0, num_col_slow);
                    s01 += s16_mm_acc_slow(wa0, bb1, num_col_slow);
                    s10 += s16_mm_acc_slow(wa1, bb0, num_col_slow);
                    s11 += s16_mm_acc_slow(wa1, bb1, num_col_slow);
                }
                if (bias_s64)
                {
                    s00 += bias_s64[ch];
                    s01 += bias_s64[ch];
                    s10 += bias_s64[ch + 1];
                    s11 += bias_s64[ch + 1];
                }
                const int32_t red0 = REDUCE_MULTIPLIER(mult0);
                const int32_t red1 = REDUCE_MULTIPLIER(mult1);
                int32_t r00 = arm_nn_requantize_s64(s00, red0, shift0);
                int32_t r01 = arm_nn_requantize_s64(s01, red0, shift0);
                int32_t r10 = arm_nn_requantize_s64(s10, red1, shift1);
                int32_t r11 = arm_nn_requantize_s64(s11, red1, shift1);

                r00 = MAX(r00, activation_min); r00 = MIN(r00, activation_max);
                r01 = MAX(r01, activation_min); r01 = MIN(r01, activation_max);
                r10 = MAX(r10, activation_min); r10 = MIN(r10, activation_max);
                r11 = MAX(r11, activation_min); r11 = MIN(r11, activation_max);

                out_c0[0] = (int16_t)r00;
                out_c1[0] = (int16_t)r01;
                out_c0[1] = (int16_t)r10;
                out_c1[1] = (int16_t)r11;
            }
        }

        /* odd trailing column */
        for (; col < n_cols; col++)
        {
            const int16_t *b_col = a->input_b + col * num_col_a;
            int16_t *out_c = a->out_base + col * output_ch + ch;
            for (int32_t k = 0; k < 2; k++)
            {
                const int8_t *wk = w_row + k * num_col_a;
                const int32_t multk = a->out_mult[ch + k];
                const int32_t shiftk = a->out_shift[ch + k];
                int32_t r;
                if (is_int32_bias)
                {
                    int32_t s = s16_mm_acc_1x1(wk, b_col, num_col_fast);
                    if (bias_s32)
                    {
                        s += bias_s32[ch + k];
                    }
                    r = arm_nn_requantize(s, multk, shiftk);
                }
                else
                {
                    int64_t s = s16_mm_acc_1x1(wk, b_col, num_col_fast);
                    if (num_col_slow > 0)
                    {
                        s += s16_mm_acc_slow(wk + num_col_fast, b_col + num_col_fast, num_col_slow);
                    }
                    if (bias_s64)
                    {
                        s += bias_s64[ch + k];
                    }
                    r = arm_nn_requantize_s64(s, REDUCE_MULTIPLIER(multk), shiftk);
                }
                r = MAX(r, activation_min);
                r = MIN(r, activation_max);
                out_c[k] = (int16_t)r;
            }
        }
    }

    /* odd trailing channel */
    for (; ch < a->ch_end; ch++)
    {
        const int8_t *wk = a->input_a + ch * num_col_a;
        const int32_t multk = a->out_mult[ch];
        const int32_t shiftk = a->out_shift[ch];
        for (int32_t col = 0; col < n_cols; col++)
        {
            const int16_t *b_col = a->input_b + col * num_col_a;
            int32_t r;
            if (is_int32_bias)
            {
                int32_t s = s16_mm_acc_1x1(wk, b_col, num_col_fast);
                if (bias_s32)
                {
                    s += bias_s32[ch];
                }
                r = arm_nn_requantize(s, multk, shiftk);
            }
            else
            {
                int64_t s = s16_mm_acc_1x1(wk, b_col, num_col_fast);
                if (num_col_slow > 0)
                {
                    s += s16_mm_acc_slow(wk + num_col_fast, b_col + num_col_fast, num_col_slow);
                }
                if (bias_s64)
                {
                    s += bias_s64[ch];
                }
                r = arm_nn_requantize_s64(s, REDUCE_MULTIPLIER(multk), shiftk);
            }
            r = MAX(r, activation_min);
            r = MIN(r, activation_max);
            a->out_base[col * output_ch + ch] = (int16_t)r;
        }
    }
}

#if defined(SPELLING_TINY_MULTICORE) && defined(ARM_MATH_DSP)
static void S16_MM_RAM_FUNC(s16_mm_range_task)(void *arg)
{
    s16_mm_range((const s16_mm_range_args *)arg);
}
#endif

#endif /* !defined(ARM_MATH_MVEI) */

/*
 * Column-blocked s16 matrix multiplication: n_cols im2col columns
 * (contiguous in input_b, each num_col_a int16s) against all output_ch
 * weight rows. Returns the output pointer advanced past the block.
 */
int16_t *arm_nn_mat_mult_kernel_s16_block(const int8_t *input_a,
                                          const int16_t *input_b,
                                          const int32_t output_ch,
                                          const int32_t *out_shift,
                                          const int32_t *out_mult,
                                          const int32_t activation_min,
                                          const int32_t activation_max,
                                          const int32_t num_col_a,
                                          const int32_t n_cols,
                                          const cmsis_nn_bias_data *const bias_data,
                                          int16_t *out_0)
{
#if !defined(ARM_MATH_MVEI)
    s16_mm_range_args args = {input_a,        input_b,   0,
                              output_ch,      output_ch, out_shift,
                              out_mult,       activation_min, activation_max,
                              num_col_a,      n_cols,    bias_data,
                              out_0};
    args.ch_start = 0;
    args.ch_end = output_ch;

#if defined(SPELLING_TINY_MULTICORE) && defined(ARM_MATH_DSP)
    if (output_ch >= SPELLING_TINY_MC_MIN_ROWS)
    {
        /* Split the channel range at an even boundary; core 0 takes the
         * lower (ceil) half. `core1` lives on this frame, valid until
         * wait(). */
        const int32_t num_pairs = output_ch / 2;
        const int32_t mid = (num_pairs - num_pairs / 2) * 2;
        s16_mm_range_args core1 = args;
        core1.ch_start = mid;
        core1.ch_end = output_ch;
        args.ch_start = 0;
        args.ch_end = mid;
        spelling_mc_dispatch_task(s16_mm_range_task, &core1);
        s16_mm_range(&args);
        spelling_mc_wait();
    }
    else
#endif
    {
        s16_mm_range(&args);
    }

    return out_0 + n_cols * output_ch;
#else
    (void)input_a;
    (void)input_b;
    (void)output_ch;
    (void)out_shift;
    (void)out_mult;
    (void)activation_min;
    (void)activation_max;
    (void)num_col_a;
    (void)n_cols;
    (void)bias_data;
    (void)out_0;

    return NULL;
#endif
}

/*
 * Matrix-multiplication function for convolution with per-channel
 * requantization (upstream 2-column entry point, kept for other callers).
 *
 * Refer header file for details.
 */
int16_t *arm_nn_mat_mult_kernel_s16(const int8_t *input_a,
                                    const int16_t *input_b,
                                    const int32_t output_ch,
                                    const int32_t *out_shift,
                                    const int32_t *out_mult,
                                    const int32_t activation_min,
                                    const int32_t activation_max,
                                    const int32_t num_col_a,
                                    const cmsis_nn_bias_data *const bias_data,
                                    int16_t *out_0)
{
    return arm_nn_mat_mult_kernel_s16_block(input_a,
                                            input_b,
                                            output_ch,
                                            out_shift,
                                            out_mult,
                                            activation_min,
                                            activation_max,
                                            num_col_a,
                                            2,
                                            bias_data,
                                            out_0);
}

/**
 * @} end of Doxygen group
 */
