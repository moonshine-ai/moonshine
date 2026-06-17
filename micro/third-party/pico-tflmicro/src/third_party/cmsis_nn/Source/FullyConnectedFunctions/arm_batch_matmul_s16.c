/*
 * SPDX-FileCopyrightText: Copyright 2024 Arm Limited and/or its affiliates <open-source-office.com>
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
 * Title:        arm_nn_batch_matmul_s16.c
 * Description:  Batch matrix multiplication. Does not perform transposes, see header file for details.
 *
 * $Date:        19 June 2024
 * $Revision:    V.1.0.0
 *
 * Target :  Arm(R) M-Profile Architecture
 *
 * -------------------------------------------------------------------- */
#include "third_party/cmsis_nn/Include/arm_nnfunctions.h"
#include "third_party/cmsis_nn/Include/arm_nnsupportfunctions.h"

/**
 * @ingroup Public
 */

/**
 * @addtogroup FC
 * @{
 */

/*
 * s16 batch matrix multiplication
 * Refer to header file for details.
 */
arm_cmsis_nn_status arm_batch_matmul_s16(const cmsis_nn_context *ctx,
                                         const cmsis_nn_bmm_params *bmm_params,
                                         const cmsis_nn_per_tensor_quant_params *quant_params,
                                         const cmsis_nn_dims *input_lhs_dims,
                                         const int16_t *input_lhs,
                                         const cmsis_nn_dims *input_rhs_dims,
                                         const int16_t *input_rhs,
                                         const cmsis_nn_dims *output_dims,
                                         int16_t *output)
{
    (void)ctx;
    const int32_t output_batch = output_dims->n;
    const int32_t output_height = output_dims->h;
    const int32_t lhs_rows = input_lhs_dims->w;
    const int32_t rhs_rows = input_rhs_dims->w;
    const int32_t rhs_cols = input_rhs_dims->c;

    const int32_t inner_lhs_diff = input_lhs_dims->h >= input_rhs_dims->h ? 0 : lhs_rows * rhs_cols;
    const int32_t inner_rhs_diff = input_rhs_dims->h >= input_lhs_dims->h ? rhs_rows * rhs_cols : 0;
    const int32_t outer_lhs_diff = input_lhs_dims->n >= input_rhs_dims->n
        ? inner_lhs_diff
        : -((lhs_rows * rhs_cols) - inner_lhs_diff) * input_lhs_dims->h;
    const int32_t outer_rhs_diff = input_rhs_dims->n >= input_lhs_dims->n ? (rhs_rows * rhs_cols) - inner_rhs_diff
                                                                          : -inner_rhs_diff * input_rhs_dims->h;

    const int32_t reduced_multiplier = REDUCE_MULTIPLIER(quant_params->multiplier);

    for (int i_out_batch = 0; i_out_batch < output_batch; i_out_batch++)
    {
        for (int i_out_height = 0; i_out_height < output_height; i_out_height++)
        {

            for (int j = 0; j < lhs_rows; j++)
            {
                arm_nn_vec_mat_mult_t_s16_s16(input_lhs,
                                              input_rhs,
                                              NULL,
                                              output,
                                              reduced_multiplier,
                                              quant_params->shift,
                                              rhs_cols,
                                              rhs_rows,
                                              bmm_params->fc_params.activation.min,
                                              bmm_params->fc_params.activation.max);
                input_lhs += rhs_cols;
                output += rhs_rows;
            }
            input_lhs -= inner_lhs_diff;
            input_rhs += inner_rhs_diff;
        }
        input_lhs += outer_lhs_diff;
        input_rhs += outer_rhs_diff;
    }

    return ARM_CMSIS_NN_SUCCESS;
}

/**
 * @} end of Doxygen group
 */
