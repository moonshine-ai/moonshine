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
 * $Date:        19 Aug 2024
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
 * @addtogroup FC
 * @{
 */

/*
 * S8 basic fully-connected and matrix multiplication layer function for TensorFlow Lite
 *
 * Refer header file for details.
 *
 */

arm_cmsis_nn_status arm_fully_connected_wrapper_s8(const cmsis_nn_context *ctx,
                                                   const cmsis_nn_fc_params *fc_params,
                                                   const cmsis_nn_quant_params *quant_params,
                                                   const cmsis_nn_dims *input_dims,
                                                   const int8_t *input_data,
                                                   const cmsis_nn_dims *filter_dims,
                                                   const int8_t *filter_data,
                                                   const cmsis_nn_dims *bias_dims,
                                                   const int32_t *bias_data,
                                                   const cmsis_nn_dims *output_dims,
                                                   int8_t *output_data)
{

    if (quant_params->is_per_channel)
    {
        const cmsis_nn_per_channel_quant_params per_channel_quant_params = {quant_params->multiplier,
                                                                            quant_params->shift};

        return arm_fully_connected_per_channel_s8(ctx,
                                                  fc_params,
                                                  &per_channel_quant_params,
                                                  input_dims,
                                                  input_data,
                                                  filter_dims,
                                                  filter_data,
                                                  bias_dims,
                                                  bias_data,
                                                  output_dims,
                                                  output_data);
    }
    else
    {
        const cmsis_nn_per_tensor_quant_params per_tensor_quant_params = {*quant_params->multiplier,
                                                                          *quant_params->shift};
        return arm_fully_connected_s8(ctx,
                                      fc_params,
                                      &per_tensor_quant_params,
                                      input_dims,
                                      input_data,
                                      filter_dims,
                                      filter_data,
                                      bias_dims,
                                      bias_data,
                                      output_dims,
                                      output_data);
    }
}

/**
 * @} end of FC group
 */
