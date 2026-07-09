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
 * Title:        arm_nn_tables.h
 * Description:  Extern declaration for NN tables
 *
 * $Date:        8 December 2022
 * $Revision:    V.2.1.1
 *
 * Target Processor:  Cortex-M cores
 * -------------------------------------------------------------------- */

#ifndef ARM_NN_TABLES_H
#define ARM_NN_TABLES_H

#include "third_party/cmsis_nn/Include/arm_nn_math_types.h"

/**
 * @brief tables for various activation functions
 *
 */

extern const uint16_t sigmoid_table_uint16[256];

#endif /*  ARM_NN_TABLES_H */
