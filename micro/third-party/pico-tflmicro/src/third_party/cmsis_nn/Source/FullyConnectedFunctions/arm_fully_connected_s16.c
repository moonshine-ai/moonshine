/*
 * SPDX-FileCopyrightText: Copyright 2010-2023 Arm Limited and/or its affiliates <open-source-office@arm.com>
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
 * Title:        arm_fully_connected_s16
 * Description:  Fully connected function compatible with TF Lite.
 *
 * $Date:        13 January 2023
 * $Revision:    V.2.1.0
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
 * S16 basic fully-connected and matrix multiplication layer function for TensorFlow Lite
 *
 * Refer header file for details.
 *
 */
arm_cmsis_nn_status arm_fully_connected_s16(const cmsis_nn_context *ctx,
                                            const cmsis_nn_fc_params *fc_params,
                                            const cmsis_nn_per_tensor_quant_params *quant_params,
                                            const cmsis_nn_dims *input_dims,
                                            const int16_t *input,
                                            const cmsis_nn_dims *filter_dims,
                                            const int8_t *kernel,
                                            const cmsis_nn_dims *bias_dims,
                                            const int64_t *bias,
                                            const cmsis_nn_dims *output_dims,
                                            int16_t *output)
{
    (void)bias_dims;
    (void)ctx;
    (void)fc_params->filter_offset;

    int32_t batch_cnt = input_dims->n;

    const int32_t reduced_multiplier = REDUCE_MULTIPLIER(quant_params->multiplier);

    while (batch_cnt)
    {
        arm_nn_vec_mat_mult_t_s16(input,
                                  kernel,
                                  bias,
                                  output,
                                  reduced_multiplier,
                                  quant_params->shift,
                                  filter_dims->n, /* col_dim or accum_depth */
                                  output_dims->c, /* row_dim or output_depth */
                                  fc_params->activation.min,
                                  fc_params->activation.max);
        input += filter_dims->n;
        output += output_dims->c;
        batch_cnt--;
    }

    return (ARM_CMSIS_NN_SUCCESS);
}

/**
 * @} end of FC group
 */
