/*
 * SPDX-FileCopyrightText: Copyright 2010-2024 Arm Limited and/or its affiliates <open-source-office@arm.com>
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
 * Title:        arm_fully_connected_s8
 * Description:  Fully connected function compatible with TF Lite.
 *
 * $Date:        6 February 2024
 * $Revision:    V.5.3.0
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
 * @addtogroup FC
 * @{
 */

/*
 * S8 basic fully-connected and matrix multiplication layer function for TensorFlow Lite
 *
 * Refer header file for details.
 *
 */

arm_cmsis_nn_status arm_fully_connected_s8(const cmsis_nn_context *ctx,
                                           const cmsis_nn_fc_params *fc_params,
                                           const cmsis_nn_per_tensor_quant_params *quant_params,
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

    int32_t batch_cnt = input_dims->n;

#if defined(ARM_MATH_MVEI)
    if (ctx->buf == NULL)
    {
        return (ARM_CMSIS_NN_ARG_ERROR);
    }
#endif

    const int32_t *kernel_sum = (const int32_t *)ctx->buf;

    while (batch_cnt)
    {

        arm_nn_vec_mat_mult_t_s8(input,
                                 kernel,
                                 kernel_sum,
                                 bias,
                                 output,
                                 fc_params->input_offset,
                                 fc_params->output_offset,
                                 quant_params->multiplier,
                                 quant_params->shift,
                                 filter_dims->n, /* col_dim or accum_depth */
                                 output_dims->c, /* row_dim or output_depth */
                                 fc_params->activation.min,
                                 fc_params->activation.max,
                                 1L,
                                 fc_params->filter_offset);

        input += filter_dims->n;
        output += output_dims->c;
        batch_cnt--;
    }
    return (ARM_CMSIS_NN_SUCCESS);
}

/**
 * @} end of FC group
 */
