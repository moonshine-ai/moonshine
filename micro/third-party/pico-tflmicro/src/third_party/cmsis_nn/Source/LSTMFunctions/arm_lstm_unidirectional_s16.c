/*
 * SPDX-FileCopyrightText: Copyright 2024, Arm Limited and/or its affiliates <open-source-office@arm.com>
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
 * Title:        arm_lstm_unidirectional_s16.c
 * Description:  S16 LSTM function with S16 gate output
 *
 * $Date:        26 March 2024
 * $Revision:    V.1.0.0
 *
 * Target Processor:  Cortex-M processors
 *
 * -------------------------------------------------------------------- */

#include "third_party/cmsis_nn/Include/arm_nnfunctions.h"
#include "third_party/cmsis_nn/Include/arm_nnsupportfunctions.h"
/**
 * @ingroup Public
 */

/**
 * @addtogroup LSTM
 * @{
 */

/*
 * S16 LSTM function for TensorFlow Lite with S16 gate output
 *
 * Refer to header file for details.
 *
 */

arm_cmsis_nn_status arm_lstm_unidirectional_s16(const int16_t *input,
                                                int16_t *output,
                                                const cmsis_nn_lstm_params *params,
                                                cmsis_nn_lstm_context *buffers)
{

    int16_t *hidden_in = NULL;
    memset(buffers->cell_state, 0, params->batch_size * params->hidden_size * sizeof(int16_t));
    if (params->time_major)
    {
        // First dimension is time, input/output for each time step is stored continously in memory
        for (int t = 0; t < params->time_steps; t++)
        {
            const int16_t *data_in = input + (t * params->batch_size * params->input_size);
            int16_t *hidden_out = output + (t * params->batch_size * params->hidden_size);
            arm_cmsis_nn_status status = arm_nn_lstm_step_s16(data_in, hidden_in, hidden_out, params, buffers, 1);
            if (status != ARM_CMSIS_NN_SUCCESS)
            {
                return status;
            }
            // Output is used as recurrent input/hidden state for the next timestep.
            hidden_in = &hidden_out[0];
        }
    }
    else
    {
        // First dimension is time, add batch_offset to jump in memory for each batch
        for (int t = 0; t < params->time_steps; t++)
        {
            const int16_t *data_in = input + (t * params->input_size);
            int16_t *hidden_out = output + (t * params->hidden_size);
            arm_cmsis_nn_status status =
                arm_nn_lstm_step_s16(data_in, hidden_in, hidden_out, params, buffers, params->time_steps);
            if (status != ARM_CMSIS_NN_SUCCESS)
            {
                return status;
            }
            // Output is used as recurrent input/hidden state for the next timestep.
            hidden_in = &hidden_out[0];
        }
    }
    return ARM_CMSIS_NN_SUCCESS;
}

/**
 * @} end of LSTM group
 */
