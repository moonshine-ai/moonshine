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
 * Title:        arm_nnfunctions.h
 * Description:  Public header file for CMSIS NN Library
 *
 * $Date:        04 November 2024
 * $Revision:    V.18.0.0
 *
 * Target :  Arm(R) M-Profile Architecture
 * -------------------------------------------------------------------- */

/**
 * @defgroup Public Public
 * A collection of functions to perform basic operations for neural network layers. Functions with a _s8 suffix support
 * TensorFlow Lite framework.
 */

#ifndef ARM_NNFUNCTIONS_H
#define ARM_NNFUNCTIONS_H

#include "third_party/cmsis_nn/Include/arm_nn_math_types.h"
#include "third_party/cmsis_nn/Include/arm_nn_types.h"

#define USE_INTRINSIC

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @defgroup NNConv Convolution Functions
 *
 * Collection of convolution, depthwise convolution functions and their variants.
 *
 * The convolution is implemented in 2 steps: im2col and General Matrix Multiplication(GEMM)
 *
 * im2col is a process of converting each patch of image data into
 * a column. After im2col, the convolution is computed as matrix-matrix
 * multiplication.
 *
 * To reduce the memory footprint, the im2col is performed partially.
 * Each iteration, only a few column (i.e., patches) are generated followed
 * by GEMM.
 *
 */

/**
 * @brief s4 convolution layer wrapper function with the main purpose to call the optimal kernel available in
 *        cmsis-nn  to perform the convolution.
 *
 * @param[in, out] ctx            Function context that contains the additional buffer if required by the function.
 *                                arm_convolve_wrapper_s4_get_buffer_size will return the buffer_size if required.
 *                                The caller is expected to clear the buffer ,if applicable, for security reasons.
 * @param[in]      conv_params    Convolution parameters (e.g. strides, dilations, pads,...).
 *                                Range of conv_params->input_offset  : [-127, 128]
 *                                Range of conv_params->output_offset : [-128, 127]
 * @param[in]      quant_params   Per-channel quantization info.
 *                                It contains the multiplier and shift values to be applied to each output channel
 * @param[in]      input_dims     Input (activation) tensor dimensions. Format: [N, H, W, C_IN]
 * @param[in]      input_data     Input (activation) data pointer. Data type: int8
 * @param[in]      filter_dims    Filter tensor dimensions. Format: [C_OUT, HK, WK, C_IN] where HK and WK are the
 *                                spatial filter dimensions
 * @param[in]      filter_data    Filter data pointer. Data type: int8 packed with 2x int4
 * @param[in]      bias_dims      Bias tensor dimensions. Format: [C_OUT]
 * @param[in]      bias_data      Bias data pointer. Data type: int32
 * @param[in]      output_dims    Output tensor dimensions. Format: [N, H, W, C_OUT]
 * @param[out]     output_data    Output data pointer. Data type: int8
 *
 * @return     The function returns either
 *                  <code>ARM_CMSIS_NN_ARG_ERROR</code> if argument constraints fail. or,
 *                  <code>ARM_CMSIS_NN_SUCCESS</code> on successful completion.
 *
 */
arm_cmsis_nn_status arm_convolve_wrapper_s4(const cmsis_nn_context *ctx,
                                            const cmsis_nn_conv_params *conv_params,
                                            const cmsis_nn_per_channel_quant_params *quant_params,
                                            const cmsis_nn_dims *input_dims,
                                            const int8_t *input_data,
                                            const cmsis_nn_dims *filter_dims,
                                            const int8_t *filter_data,
                                            const cmsis_nn_dims *bias_dims,
                                            const int32_t *bias_data,
                                            const cmsis_nn_dims *output_dims,
                                            int8_t *output_data);

/**
 * @brief Get the required buffer size for arm_convolve_wrapper_s4
 *
 * @param[in]      conv_params    Convolution parameters (e.g. strides, dilations, pads,...).
 *                                Range of conv_params->input_offset  : [-127, 128]
 *                                Range of conv_params->output_offset : [-128, 127]
 * @param[in]      input_dims     Input (activation) dimensions. Format: [N, H, W, C_IN]
 * @param[in]      filter_dims    Filter dimensions. Format: [C_OUT, HK, WK, C_IN] where HK and WK are the spatial
 *                                filter dimensions
 * @param[in]      output_dims    Output tensor dimensions. Format: [N, H, W, C_OUT]
 *
 * @return         The function returns required buffer size(bytes)
 *
 */
int32_t arm_convolve_wrapper_s4_get_buffer_size(const cmsis_nn_conv_params *conv_params,
                                                const cmsis_nn_dims *input_dims,
                                                const cmsis_nn_dims *filter_dims,
                                                const cmsis_nn_dims *output_dims);

/**
 * @brief Get the required buffer size for arm_convolve_wrapper_s4 for Arm(R) Helium Architecture case.
 *        Refer to arm_convolve_wrapper_s4_get_buffer_size() for function argument details.
 *
 * @note       Intended for compilation on Host. If compiling for an Arm target, use
 *             arm_convolve_wrapper_s4_get_buffer_size(). Currently this operator does not have an
 *             mve implementation, so dsp will be used.
 *
 */
int32_t arm_convolve_wrapper_s4_get_buffer_size_mve(const cmsis_nn_conv_params *conv_params,
                                                    const cmsis_nn_dims *input_dims,
                                                    const cmsis_nn_dims *filter_dims,
                                                    const cmsis_nn_dims *output_dims);

/**
 * @brief Get the required buffer size for arm_convolve_wrapper_s4 for processors with DSP extension.
 *        Refer to arm_convolve_wrapper_s4_get_buffer_size() for function argument details.
 *
 * @note       Intended for compilation on Host. If compiling for an Arm target, use
 *             arm_convolve_wrapper_s4_get_buffer_size().
 *
 */
int32_t arm_convolve_wrapper_s4_get_buffer_size_dsp(const cmsis_nn_conv_params *conv_params,
                                                    const cmsis_nn_dims *input_dims,
                                                    const cmsis_nn_dims *filter_dims,
                                                    const cmsis_nn_dims *output_dims);

/**
 * @brief s8 convolution layer wrapper function with the main purpose to call the optimal kernel available in
 *        cmsis-nn  to perform the convolution.
 *
 * @param[in, out] ctx            Function context that contains the additional buffer if required by the function.
 *                                arm_convolve_wrapper_s8_get_buffer_size will return the buffer_size if required.
 *                                The caller is expected to clear the buffer, if applicable, for security reasons.
 * @param[in]      conv_params    Convolution parameters (e.g. strides, dilations, pads,...).
 *                                Range of conv_params->input_offset  : [-127, 128]
 *                                Range of conv_params->output_offset : [-128, 127]
 * @param[in]      quant_params   Per-channel quantization info.
 *                                It contains the multiplier and shift values to be applied to each output channel
 * @param[in]      input_dims     Input (activation) tensor dimensions. Format: [N, H, W, C_IN]
 * @param[in]      input_data     Input (activation) data pointer. Data type: int8
 * @param[in]      filter_dims    Filter tensor dimensions. Format: [C_OUT, HK, WK, C_IN] where HK and WK are the
 *                                spatial filter dimensions
 * @param[in]      filter_data    Filter data pointer. Data type: int8
 * @param[in]      bias_dims      Bias tensor dimensions. Format: [C_OUT]
 * @param[in]      bias_data      Bias data pointer. Data type: int32
 * @param[in]      output_dims    Output tensor dimensions. Format: [N, H, W, C_OUT]
 * @param[out]     output_data    Output data pointer. Data type: int8
 *
 * @return     The function returns either
 *                  <code>ARM_CMSIS_NN_ARG_ERROR</code> if argument constraints fail. or,
 *                  <code>ARM_CMSIS_NN_SUCCESS</code> on successful completion.
 *
 */
arm_cmsis_nn_status arm_convolve_wrapper_s8(const cmsis_nn_context *ctx,
                                            const cmsis_nn_conv_params *conv_params,
                                            const cmsis_nn_per_channel_quant_params *quant_params,
                                            const cmsis_nn_dims *input_dims,
                                            const int8_t *input_data,
                                            const cmsis_nn_dims *filter_dims,
                                            const int8_t *filter_data,
                                            const cmsis_nn_dims *bias_dims,
                                            const int32_t *bias_data,
                                            const cmsis_nn_dims *output_dims,
                                            int8_t *output_data);

/**
 * @brief Get the required buffer size for arm_convolve_wrapper_s8
 *
 * @param[in]      conv_params    Convolution parameters (e.g. strides, dilations, pads,...).
 *                                Range of conv_params->input_offset  : [-127, 128]
 *                                Range of conv_params->output_offset : [-128, 127]
 * @param[in]      input_dims     Input (activation) dimensions. Format: [N, H, W, C_IN]
 * @param[in]      filter_dims    Filter dimensions. Format: [C_OUT, HK, WK, C_IN] where HK and WK are the spatial
 *                                filter dimensions
 * @param[in]      output_dims    Output tensor dimensions. Format: [N, H, W, C_OUT]
 *
 * @return         The function returns required buffer size(bytes)
 *
 */
int32_t arm_convolve_wrapper_s8_get_buffer_size(const cmsis_nn_conv_params *conv_params,
                                                const cmsis_nn_dims *input_dims,
                                                const cmsis_nn_dims *filter_dims,
                                                const cmsis_nn_dims *output_dims);

/**
 * @brief Get the required buffer size for arm_convolve_wrapper_s8 for Arm(R) Helium Architecture case.
 *        Refer to arm_convolve_wrapper_s8_get_buffer_size() for function argument details.
 *
 * @note       Intended for compilation on Host. If compiling for an Arm target, use
 *             arm_convolve_wrapper_s8_get_buffer_size().
 *
 */
int32_t arm_convolve_wrapper_s8_get_buffer_size_mve(const cmsis_nn_conv_params *conv_params,
                                                    const cmsis_nn_dims *input_dims,
                                                    const cmsis_nn_dims *filter_dims,
                                                    const cmsis_nn_dims *output_dims);

/**
 * @brief Get the required buffer size for arm_convolve_wrapper_s8 for processors with DSP extension.
 *        Refer to arm_convolve_wrapper_s8_get_buffer_size() for function argument details.
 *
 * @note       Intended for compilation on Host. If compiling for an Arm target, use
 *             arm_convolve_wrapper_s8_get_buffer_size().
 *
 */
int32_t arm_convolve_wrapper_s8_get_buffer_size_dsp(const cmsis_nn_conv_params *conv_params,
                                                    const cmsis_nn_dims *input_dims,
                                                    const cmsis_nn_dims *filter_dims,
                                                    const cmsis_nn_dims *output_dims);

/**
 * @brief s16 convolution layer wrapper function with the main purpose to call the optimal kernel available in
 *        cmsis-nn to perform the convolution.
 *
 * @param[in, out] ctx            Function context that contains the additional buffer if required by the function.
 *                                arm_convolve_wrapper_s8_get_buffer_size will return the buffer_size if required
 *                                The caller is expected to clear the buffer, if applicable, for security reasons.
 * @param[in]      conv_params    Convolution parameters (e.g. strides, dilations, pads,...).
 *                                conv_params->input_offset  : Not used
 *                                conv_params->output_offset : Not used
 * @param[in]      quant_params   Per-channel quantization info.
 *                                It contains the multiplier and shift values to be applied to each output channel
 * @param[in]      input_dims     Input (activation) tensor dimensions. Format: [N, H, W, C_IN]
 * @param[in]      input_data     Input (activation) data pointer. Data type: int16
 * @param[in]      filter_dims    Filter tensor dimensions. Format: [C_OUT, HK, WK, C_IN] where HK and WK are the
 *                                spatial filter dimensions
 * @param[in]      filter_data    Filter data pointer. Data type: int8
 * @param[in]      bias_dims      Bias tensor dimensions. Format: [C_OUT]
 * @param[in]      bias_data      Struct with optional bias data pointer. Bias data type can be int64 or int32 depending
 *                                flag in struct.
 * @param[in]      output_dims    Output tensor dimensions. Format: [N, H, W, C_OUT]
 * @param[out]     output_data    Output data pointer. Data type: int16
 *
 * @return     The function returns either
 *                  <code>ARM_CMSIS_NN_ARG_ERROR</code> if argument constraints fail. or,
 *                  <code>ARM_CMSIS_NN_SUCCESS</code> on successful completion.
 *
 */
arm_cmsis_nn_status arm_convolve_wrapper_s16(const cmsis_nn_context *ctx,
                                             const cmsis_nn_conv_params *conv_params,
                                             const cmsis_nn_per_channel_quant_params *quant_params,
                                             const cmsis_nn_dims *input_dims,
                                             const int16_t *input_data,
                                             const cmsis_nn_dims *filter_dims,
                                             const int8_t *filter_data,
                                             const cmsis_nn_dims *bias_dims,
                                             const cmsis_nn_bias_data *bias_data,
                                             const cmsis_nn_dims *output_dims,
                                             int16_t *output_data);

/**
 * @brief Get the required buffer size for arm_convolve_wrapper_s16.
 *
 * @param[in]      conv_params    Convolution parameters (e.g. strides, dilations, pads,...).
 *                                conv_params->input_offset  : Not used
 *                                conv_params->output_offset : Not used
 * @param[in]      input_dims     Input (activation) dimensions. Format: [N, H, W, C_IN]
 * @param[in]      filter_dims    Filter dimensions. Format: [C_OUT, HK, WK, C_IN] where HK and WK are the spatial
 *                                filter dimensions
 * @param[in]      output_dims    Output tensor dimensions. Format: [N, H, W, C_OUT]
 *
 * @return         The function returns required buffer size(bytes)
 *
 */
int32_t arm_convolve_wrapper_s16_get_buffer_size(const cmsis_nn_conv_params *conv_params,
                                                 const cmsis_nn_dims *input_dims,
                                                 const cmsis_nn_dims *filter_dims,
                                                 const cmsis_nn_dims *output_dims);

/**
 * @brief Get the required buffer size for arm_convolve_wrapper_s16 for for processors with DSP extension.
 *        Refer to arm_convolve_wrapper_s16_get_buffer_size() for function argument details.
 *
 * @note       Intended for compilation on Host. If compiling for an Arm target, use
 *             arm_convolve_wrapper_s16_get_buffer_size().
 *
 */
int32_t arm_convolve_wrapper_s16_get_buffer_size_dsp(const cmsis_nn_conv_params *conv_params,
                                                     const cmsis_nn_dims *input_dims,
                                                     const cmsis_nn_dims *filter_dims,
                                                     const cmsis_nn_dims *output_dims);

/**
 * @brief Get the required buffer size for arm_convolve_wrapper_s16 for Arm(R) Helium Architecture case.
 *        Refer to arm_convolve_wrapper_s16_get_buffer_size() for function argument details.
 *
 * @note       Intended for compilation on Host. If compiling for an Arm target, use
 *             arm_convolve_wrapper_s16_get_buffer_size().
 *
 */
int32_t arm_convolve_wrapper_s16_get_buffer_size_mve(const cmsis_nn_conv_params *conv_params,
                                                     const cmsis_nn_dims *input_dims,
                                                     const cmsis_nn_dims *filter_dims,
                                                     const cmsis_nn_dims *output_dims);

/**
 * @brief Basic s4 convolution function
 * @param[in, out] ctx            Function context that contains the additional buffer if required by the function.
 *                                arm_convolve_s4_get_buffer_size will return the buffer_size if required.
 *                                The caller is expected to clear the buffer ,if applicable, for security reasons.
 * @param[in]      conv_params    Convolution parameters (e.g. strides, dilations, pads,...).
 *                                Range of conv_params->input_offset  : [-127, 128]
 *                                Range of conv_params->output_offset : [-128, 127]
 * @param[in]      quant_params   Per-channel quantization info.
 *                                It contains the multiplier and shift values to be applied to each output channel
 * @param[in]      input_dims     Input (activation) tensor dimensions. Format: [N, H, W, C_IN]
 * @param[in]      input_data     Input (activation) data pointer. Data type: int8
 * @param[in]      filter_dims    Filter tensor dimensions. Format: [C_OUT, HK, WK, C_IN] where HK and WK are the
 *                                spatial filter dimensions
 * @param[in]      filter_data    Packed Filter data pointer. Data type: int8 packed with 2x int4
 * @param[in]      bias_dims      Bias tensor dimensions. Format: [C_OUT]
 * @param[in]      bias_data      Optional bias data pointer. Data type: int32
 * @param[in]      output_dims    Output tensor dimensions. Format: [N, H, W, C_OUT]
 * @param[out]     output_data    Output data pointer. Data type: int8

 * @return     The function returns <code>ARM_CMSIS_NN_SUCCESS</code>
 *
 * @details
 *    1. Supported framework: TensorFlow Lite micro
 *    2. Additional memory is required for optimization. Refer to argument 'ctx' for details.
 *
 */
arm_cmsis_nn_status arm_convolve_s4(const cmsis_nn_context *ctx,
                                    const cmsis_nn_conv_params *conv_params,
                                    const cmsis_nn_per_channel_quant_params *quant_params,
                                    const cmsis_nn_dims *input_dims,
                                    const int8_t *input_data,
                                    const cmsis_nn_dims *filter_dims,
                                    const int8_t *filter_data,
                                    const cmsis_nn_dims *bias_dims,
                                    const int32_t *bias_data,
                                    const cmsis_nn_dims *output_dims,
                                    int8_t *output_data);

/**
 * @brief Basic s4 convolution function with a requirement of even number of kernels.
 * @param[in, out] ctx            Function context that contains the additional buffer if required by the function.
 *                                arm_convolve_s4_get_buffer_size will return the buffer_size if required.
 *                                The caller is expected to clear the buffer ,if applicable, for security reasons.
 * @param[in]      conv_params    Convolution parameters (e.g. strides, dilations, pads,...).
 *                                Range of conv_params->input_offset  : [-127, 128]
 *                                Range of conv_params->output_offset : [-128, 127]
 * @param[in]      quant_params   Per-channel quantization info.
 *                                It contains the multiplier and shift values to be applied to each output channel
 * @param[in]      input_dims     Input (activation) tensor dimensions. Format: [N, H, W, C_IN]
 * @param[in]      input_data     Input (activation) data pointer. Data type: int8
 * @param[in]      filter_dims    Filter tensor dimensions. Format: [C_OUT, HK, WK, C_IN] where HK and WK are the
 *                                spatial filter dimensions. Note the product must be even.
 * @param[in]      filter_data    Packed Filter data pointer. Data type: int8 packed with 2x int4
 * @param[in]      bias_dims      Bias tensor dimensions. Format: [C_OUT]
 * @param[in]      bias_data      Optional bias data pointer. Data type: int32
 * @param[in]      output_dims    Output tensor dimensions. Format: [N, H, W, C_OUT]
 * @param[out]     output_data    Output data pointer. Data type: int8
 *
 * @return     The function returns <code>ARM_CMSIS_NN_SUCCESS</code> if successful or
 *                                  <code>ARM_CMSIS_NN_ARG_ERROR</code> if incorrect arguments or
 *                                  <code>ARM_CMSIS_NN_NO_IMPL_ERROR</code> if not for MVE
 *
 * @details
 *    1. Supported framework: TensorFlow Lite micro
 *    2. Additional memory is required for optimization. Refer to argument 'ctx' for details.
 *
 */
arm_cmsis_nn_status arm_convolve_even_s4(const cmsis_nn_context *ctx,
                                         const cmsis_nn_conv_params *conv_params,
                                         const cmsis_nn_per_channel_quant_params *quant_params,
                                         const cmsis_nn_dims *input_dims,
                                         const int8_t *input_data,
                                         const cmsis_nn_dims *filter_dims,
                                         const int8_t *filter_data,
                                         const cmsis_nn_dims *bias_dims,
                                         const int32_t *bias_data,
                                         const cmsis_nn_dims *output_dims,
                                         int8_t *output_data);

/**
 * @brief Basic s8 convolution function
 * @param[in, out] ctx            Function context that contains the additional buffer if required by the function.
 *                                arm_convolve_s8_get_buffer_size will return the buffer_size if required.
 *                                The caller is expected to clear the buffer, if applicable, for security reasons.
 * @param[in]      conv_params    Convolution parameters (e.g. strides, dilations, pads,...).
 *                                Range of conv_params->input_offset  : [-127, 128]
 *                                Range of conv_params->output_offset : [-128, 127]
 * @param[in]      quant_params   Per-channel quantization info.
 *                                It contains the multiplier and shift values to be applied to each output channel
 * @param[in]      input_dims     Input (activation) tensor dimensions. Format: [N, H, W, C_IN]
 * @param[in]      input_data     Input (activation) data pointer. Data type: int8
 * @param[in]      filter_dims    Filter tensor dimensions. Format: [C_OUT, HK, WK, CK] where HK, WK and CK are the
 *                                spatial filter dimensions. CK != C_IN is used for grouped convolution, in which
 *                                case the required conditions are C_IN = N * CK and C_OUT = N * M for N groups of
 *                                size M.
 * @param[in]      filter_data    Filter data pointer. Data type: int8
 * @param[in]      bias_dims      Bias tensor dimensions. Format: [C_OUT]
 * @param[in]      bias_data      Optional bias data pointer. Data type: int32
 * @param[in]      upscale_dims   Inserts zeroes to upscale the input in h/w dimensions if set to 2. This is used for
 * tranposed convolution.
 * @param[in]      output_dims    Output tensor dimensions. Format: [N, H, W, C_OUT]
 * @param[out]     output_data    Output data pointer. Data type: int8
 *
 * @return     The function returns <code>ARM_CMSIS_NN_SUCCESS</code> if successful or
 *                                  <code>ARM_CMSIS_NN_ARG_ERROR</code> if incorrect arguments or
 *                                  <code>ARM_CMSIS_NN_NO_IMPL_ERROR</code>
 *
 * @details
 *    1. Supported framework: TensorFlow Lite micro
 *    2. Additional memory is required for optimization. Refer to argument 'ctx' for details.
 *
 */
arm_cmsis_nn_status arm_convolve_s8(const cmsis_nn_context *ctx,
                                    const cmsis_nn_conv_params *conv_params,
                                    const cmsis_nn_per_channel_quant_params *quant_params,
                                    const cmsis_nn_dims *input_dims,
                                    const int8_t *input_data,
                                    const cmsis_nn_dims *filter_dims,
                                    const int8_t *filter_data,
                                    const cmsis_nn_dims *bias_dims,
                                    const int32_t *bias_data,
                                    const cmsis_nn_dims *upscale_dims,
                                    const cmsis_nn_dims *output_dims,
                                    int8_t *output_data);

/**
 * @brief Get the required buffer size for s4 convolution function
 *
 * @param[in]       input_dims            Input (activation) tensor dimensions. Format: [N, H, W, C_IN]
 * @param[in]       filter_dims           Filter tensor dimensions. Format: [C_OUT, HK, WK, C_IN] where HK and WK
 * are the spatial filter dimensions
 * @return          The function returns required buffer size(bytes)
 *
 */
int32_t arm_convolve_s4_get_buffer_size(const cmsis_nn_dims *input_dims, const cmsis_nn_dims *filter_dims);

/**
 * @brief Get the required buffer size for s8 convolution function
 *
 * @param[in]       input_dims            Input (activation) tensor dimensions. Format: [N, H, W, C_IN]
 * @param[in]       filter_dims           Filter tensor dimensions. Format: [C_OUT, HK, WK, C_IN] where HK and WK
 * are the spatial filter dimensions
 * @return          The function returns required buffer size(bytes)
 *
 */
int32_t arm_convolve_s8_get_buffer_size(const cmsis_nn_dims *input_dims, const cmsis_nn_dims *filter_dims);

/**
 * @brief Wrapper to select optimal transposed convolution algorithm depending on parameters.
 * @param[in, out] ctx                   Function context that contains the additional buffer if required by the
 *                                       function.
 *                                       arm_transpose_conv_s8_get_buffer_size will return the buffer_size if required.
 *                                       The caller is expected to clear the buffer, if applicable, for security
 reasons.
 * @param[in, out] output_ctx            Temporary scratch buffer.
 *                                       The size required size is: output width * output height * output channel * 4
 *                                       The caller is expected to clear the buffer, if applicable, for security
 *                                        reasons.
 * @param[in]      transpose_conv_params Convolution parameters (e.g. strides, dilations, pads,...).
 *                                       Range of transpose_conv_params->input_offset  : [-127, 128]
 *                                       Range of transpose_conv_params->output_offset : [-128, 127]
 * @param[in]      quant_params          Per-channel quantization info.
 *                                       It contains the multiplier and shift values to be applied to each out channel.
 * @param[in]      input_dims            Input (activation) tensor dimensions. Format: [N, H, W, C_IN]
 * @param[in]      input_data            Input (activation) data pointer. Data type: int8
 * @param[in]      filter_dims           Filter tensor dimensions. Format: [C_OUT, HK, WK, C_IN] where HK and WK are the
 *                                       spatial filter dimensions
 * @param[in]      filter_data           Filter data pointer. Data type: int8
 * @param[in]      bias_dims             Bias tensor dimensions. Format: [C_OUT]
 * @param[in]      bias_data             Optional bias data pointer. Data type: int32
 * @param[in]      output_dims           Output tensor dimensions. Format: [N, H, W, C_OUT]
 * @param[out]     output_data           Output data pointer. Data type: int8

 * @return     The function returns either
 *                  <code>ARM_CMSIS_NN_ARG_ERROR</code> if argument constraints fail. or,
 *                  <code>ARM_CMSIS_NN_SUCCESS</code> on successful completion.
 *
 * @details
 *    1. Supported framework: TensorFlow Lite micro
 *    2. Additional memory is required for optimization. Refer to arguments 'ctx' and 'output_ctx' for details.
 *
 */
arm_cmsis_nn_status arm_transpose_conv_wrapper_s8(const cmsis_nn_context *ctx,
                                                  const cmsis_nn_context *output_ctx,
                                                  const cmsis_nn_transpose_conv_params *transpose_conv_params,
                                                  const cmsis_nn_per_channel_quant_params *quant_params,
                                                  const cmsis_nn_dims *input_dims,
                                                  const int8_t *input_data,
                                                  const cmsis_nn_dims *filter_dims,
                                                  const int8_t *filter_data,
                                                  const cmsis_nn_dims *bias_dims,
                                                  const int32_t *bias_data,
                                                  const cmsis_nn_dims *output_dims,
                                                  int8_t *output_data);

/**
 * @brief Basic s8 transpose convolution function
 * @param[in, out] ctx                   Function context that contains the additional buffer if required by the
 *                                       function.
 *                                       arm_transpose_conv_s8_get_buffer_size will return the buffer_size if required.
 *                                       The caller is expected to clear the buffer, if applicable, for security
 reasons.
 * @param[in, out] output_ctx            Temporary scratch buffer.
 *                                       The size required size is: output width * output height * output channel * 4
 *                                       The caller is expected to clear the buffer, if applicable, for security
 *                                        reasons.
 * @param[in]      transpose_conv_params Convolution parameters (e.g. strides, dilations, pads,...).
 *                                       Range of transpose_conv_params->input_offset  : [-127, 128]
 *                                       Range of transpose_conv_params->output_offset : [-128, 127]
 * @param[in]      quant_params          Per-channel quantization info.
 *                                       It contains the multiplier and shift values to be applied to each out channel.
 * @param[in]      input_dims            Input (activation) tensor dimensions. Format: [N, H, W, C_IN]
 * @param[in]      input_data            Input (activation) data pointer. Data type: int8
 * @param[in]      filter_dims           Filter tensor dimensions. Format: [C_OUT, HK, WK, C_IN] where HK and WK are the
 *                                       spatial filter dimensions
 * @param[in]      filter_data           Filter data pointer. Data type: int8
 * @param[in]      bias_dims             Bias tensor dimensions. Format: [C_OUT]
 * @param[in]      bias_data             Optional bias data pointer. Data type: int32
 * @param[in]      output_dims           Output tensor dimensions. Format: [N, H, W, C_OUT]
 * @param[out]     output_data           Output data pointer. Data type: int8

 * @return     The function returns either
 *                  <code>ARM_CMSIS_NN_ARG_ERROR</code> if argument constraints fail. or,
 *                  <code>ARM_CMSIS_NN_SUCCESS</code> on successful completion.
 *
 * @details
 *    1. Supported framework: TensorFlow Lite micro
 *    2. Additional memory is required for optimization. Refer to arguments 'ctx' and 'output_ctx' for details.
 *
 */
arm_cmsis_nn_status arm_transpose_conv_s8(const cmsis_nn_context *ctx,
                                          const cmsis_nn_context *output_ctx,
                                          const cmsis_nn_transpose_conv_params *transpose_conv_params,
                                          const cmsis_nn_per_channel_quant_params *quant_params,
                                          const cmsis_nn_dims *input_dims,
                                          const int8_t *input_data,
                                          const cmsis_nn_dims *filter_dims,
                                          const int8_t *filter_data,
                                          const cmsis_nn_dims *bias_dims,
                                          const int32_t *bias_data,
                                          const cmsis_nn_dims *output_dims,
                                          int8_t *output_data);

/**
 * @brief Get the required buffer size for ctx in s8 transpose conv function
 *
 * @param[in]       transposed_conv_params  Transposed convolution parameters
 * @param[in]       input_dims              Input (activation) tensor dimensions. Format: [N, H, W, C_IN]
 * @param[in]       filter_dims             Filter tensor dimensions. Format: [C_OUT, HK, WK, C_IN] where HK and WK
 *                                          are the spatial filter dimensions
 * @param[in]       out_dims                Output tensor dimensions. Format: [N, H, W, C_OUT]
 * @return          The function returns required buffer size(bytes)
 *
 */
int32_t arm_transpose_conv_s8_get_buffer_size(const cmsis_nn_transpose_conv_params *transposed_conv_params,
                                              const cmsis_nn_dims *input_dims,
                                              const cmsis_nn_dims *filter_dims,
                                              const cmsis_nn_dims *out_dims);

/**
 * @brief Get the required buffer size for output_ctx in s8 transpose conv function
 *
 * @param[in]       transposed_conv_params  Transposed convolution parameters
 * @param[in]       input_dims              Input (activation) tensor dimensions. Format: [N, H, W, C_IN]
 * @param[in]       filter_dims             Filter tensor dimensions. Format: [C_OUT, HK, WK, C_IN] where HK and WK
 *                                        are the spatial filter dimensions
 * @return          The function returns required buffer size(bytes)
 *
 */
int32_t arm_transpose_conv_s8_get_reverse_conv_buffer_size(const cmsis_nn_transpose_conv_params *transposed_conv_params,
                                                           const cmsis_nn_dims *input_dims,
                                                           const cmsis_nn_dims *filter_dims);

/**
 * @brief Get size of additional buffer required by arm_transpose_conv_s8() for processors with DSP extension.
 *        Refer to arm_transpose_conv_s8_get_buffer_size() for function argument details.
 *
 * @note       Intended for compilation on Host. If compiling for an Arm target, use
 *             arm_transpose_conv_s8_get_buffer_size().
 *
 */
int32_t arm_transpose_conv_s8_get_buffer_size_dsp(const cmsis_nn_dims *input_dims,
                                                  const cmsis_nn_dims *filter_dims,
                                                  const cmsis_nn_dims *out_dims);

/**
 * @brief Get size of additional buffer required by arm_transpose_conv_s8() for Arm(R) Helium Architecture case.
 *        Refer to arm_transpose_conv_s8_get_buffer_size() for function argument details.
 *
 * @note       Intended for compilation on Host. If compiling for an Arm target, use
 *             arm_transpose_conv_s8_get_buffer_size().
 *
 */
int32_t arm_transpose_conv_s8_get_buffer_size_mve(const cmsis_nn_dims *input_dims,
                                                  const cmsis_nn_dims *filter_dims,
                                                  const cmsis_nn_dims *out_dims);

/**
 * @brief Basic s16 convolution function
 * @param[in, out] ctx            Function context that contains the additional buffer if required by the function.
 *                                arm_convolve_s16_get_buffer_size will return the buffer_size if required.
 *                                The caller is expected to clear the buffer, if applicable, for security reasons.
 * @param[in]      conv_params    Convolution parameters (e.g. strides, dilations, pads,...).
 *                                conv_params->input_offset  : Not used
 *                                conv_params->output_offset : Not used
 * @param[in]      quant_params   Per-channel quantization info.
 *                                It contains the multiplier and shift values to be applied to each output channel
 * @param[in]      input_dims     Input (activation) tensor dimensions. Format: [N, H, W, C_IN]
 * @param[in]      input_data     Input (activation) data pointer. Data type: int16
 * @param[in]      filter_dims    Filter tensor dimensions. Format: [C_OUT, HK, WK, C_IN] where HK and WK are the
 *                                spatial filter dimensions
 * @param[in]      filter_data    Filter data pointer. Data type: int8
 * @param[in]      bias_dims      Bias tensor dimensions. Format: [C_OUT]
 * @param[in]      bias_data      Struct with optional bias data pointer. Bias data type can be int64 or int32 depending
 *                                flag in struct.
 * @param[in]      output_dims    Output tensor dimensions. Format: [N, H, W, C_OUT]
 * @param[out]     output_data    Output data pointer. Data type: int16
 *
 * @return     The function returns <code>ARM_CMSIS_NN_SUCCESS</code> if successful or
 *                                  <code>ARM_CMSIS_NN_ARG_ERROR</code> if incorrect arguments or
 *                                  <code>ARM_CMSIS_NN_NO_IMPL_ERROR</code>
 *
 * @details
 *    1. Supported framework: TensorFlow Lite micro
 *    2. Additional memory is required for optimization. Refer to argument 'ctx' for details.
 *
 */
arm_cmsis_nn_status arm_convolve_s16(const cmsis_nn_context *ctx,
                                     const cmsis_nn_conv_params *conv_params,
                                     const cmsis_nn_per_channel_quant_params *quant_params,
                                     const cmsis_nn_dims *input_dims,
                                     const int16_t *input_data,
                                     const cmsis_nn_dims *filter_dims,
                                     const int8_t *filter_data,
                                     const cmsis_nn_dims *bias_dims,
                                     const cmsis_nn_bias_data *bias_data,
                                     const cmsis_nn_dims *output_dims,
                                     int16_t *output_data);

/**
 * @brief Get the required buffer size for s16 convolution function
 *
 * @param[in]       input_dims    Input (activation) tensor dimensions. Format: [N, H, W, C_IN]
 * @param[in]       filter_dims   Filter tensor dimensions. Format: [C_OUT, HK, WK, C_IN] where HK and WK
 *                                are the spatial filter dimensions
 * @return          The function returns required buffer size(bytes)
 *
 */
int32_t arm_convolve_s16_get_buffer_size(const cmsis_nn_dims *input_dims, const cmsis_nn_dims *filter_dims);

/**
 * @brief Fast s4 version for 1x1 convolution (non-square shape)
 *
 * @param[in, out] ctx           Function context that contains the additional buffer if required by the function.
 *                               arm_convolve_1x1_s4_fast_get_buffer_size will return the buffer_size if required.
 *                               The caller is expected to clear the buffer ,if applicable, for security reasons.
 * @param[in]      conv_params   Convolution parameters (e.g. strides, dilations, pads,...).
 *                               Range of conv_params->input_offset  : [-127, 128]
 *                               Range of conv_params->output_offset : [-128, 127]
 * @param[in]      quant_params  Per-channel quantization info.
 *                               It contains the multiplier and shift values to be applied to each output channel
 * @param[in]      input_dims    Input (activation) tensor dimensions. Format: [N, H, W, C_IN]
 * @param[in]      input_data    Input (activation) data pointer. Data type: int8
 * @param[in]      filter_dims   Filter tensor dimensions. Format: [C_OUT, 1, 1, C_IN]
 * @param[in]      filter_data   Filter data pointer. Data type: int8 packed with 2x int4
 * @param[in]      bias_dims     Bias tensor dimensions. Format: [C_OUT]
 * @param[in]      bias_data     Optional bias data pointer. Data type: int32
 * @param[in]      output_dims   Output tensor dimensions. Format: [N, H, W, C_OUT]
 * @param[out]     output_data   Output data pointer. Data type: int8
 *
 * @return     The function returns either
 *                  <code>ARM_CMSIS_NN_ARG_ERROR</code> if argument constraints fail. or,
 *                  <code>ARM_CMSIS_NN_SUCCESS</code> on successful completion.
 *
 * @details
 *   - Supported framework : TensorFlow Lite Micro
 *   - The following constrains on the arguments apply
 *      -# conv_params->padding.w = conv_params->padding.h = 0
 *      -# conv_params->stride.w = conv_params->stride.h = 1
 *
 */
arm_cmsis_nn_status arm_convolve_1x1_s4_fast(const cmsis_nn_context *ctx,
                                             const cmsis_nn_conv_params *conv_params,
                                             const cmsis_nn_per_channel_quant_params *quant_params,
                                             const cmsis_nn_dims *input_dims,
                                             const int8_t *input_data,
                                             const cmsis_nn_dims *filter_dims,
                                             const int8_t *filter_data,
                                             const cmsis_nn_dims *bias_dims,
                                             const int32_t *bias_data,
                                             const cmsis_nn_dims *output_dims,
                                             int8_t *output_data);

/**
 * @brief s4 version for 1x1 convolution with support for non-unity stride values
 *
 * @param[in, out] ctx           Function context that contains the additional buffer if required by the function.
 *                               None is required by this function.
 * @param[in]      conv_params   Convolution parameters (e.g. strides, dilations, pads,...).
 *                               Range of conv_params->input_offset  : [-127, 128]
 *                               Range of conv_params->output_offset : [-128, 127]
 * @param[in]      quant_params  Per-channel quantization info.
 *                               It contains the multiplier and shift values to be applied to each output channel
 * @param[in]      input_dims    Input (activation) tensor dimensions. Format: [N, H, W, C_IN]
 * @param[in]      input_data    Input (activation) data pointer. Data type: int8
 * @param[in]      filter_dims   Filter tensor dimensions. Format: [C_OUT, 1, 1, C_IN]
 * @param[in]      filter_data   Filter data pointer. Data type: int8 packed with 2x int4
 * @param[in]      bias_dims     Bias tensor dimensions. Format: [C_OUT]
 * @param[in]      bias_data     Optional bias data pointer. Data type: int32
 * @param[in]      output_dims   Output tensor dimensions. Format: [N, H, W, C_OUT]
 * @param[out]     output_data   Output data pointer. Data type: int8
 *
 * @return     The function returns either
 *                  <code>ARM_CMSIS_NN_ARG_ERROR</code> if argument constraints fail. or,
 *                  <code>ARM_CMSIS_NN_SUCCESS</code> on successful completion.
 * @details
 *   - Supported framework : TensorFlow Lite Micro
 *   - The following constrains on the arguments apply
 *      -# conv_params->padding.w = conv_params->padding.h = 0
 *
 */
arm_cmsis_nn_status arm_convolve_1x1_s4(const cmsis_nn_context *ctx,
                                        const cmsis_nn_conv_params *conv_params,
                                        const cmsis_nn_per_channel_quant_params *quant_params,
                                        const cmsis_nn_dims *input_dims,
                                        const int8_t *input_data,
                                        const cmsis_nn_dims *filter_dims,
                                        const int8_t *filter_data,
                                        const cmsis_nn_dims *bias_dims,
                                        const int32_t *bias_data,
                                        const cmsis_nn_dims *output_dims,
                                        int8_t *output_data);

/**
 * @brief Fast s8 version for 1x1 convolution (non-square shape)
 *
 * @param[in, out] ctx           Function context that contains the additional buffer if required by the function.
 *                               arm_convolve_1x1_s8_fast_get_buffer_size will return the buffer_size if required.
 *                               The caller is expected to clear the buffer, if applicable, for security reasons.
 * @param[in]      conv_params   Convolution parameters (e.g. strides, dilations, pads,...).
 *                               Range of conv_params->input_offset  : [-127, 128]
 *                               Range of conv_params->output_offset : [-128, 127]
 * @param[in]      quant_params  Per-channel quantization info.
 *                               It contains the multiplier and shift values to be applied to each output channel
 * @param[in]      input_dims    Input (activation) tensor dimensions. Format: [N, H, W, C_IN]
 * @param[in]      input_data    Input (activation) data pointer. Data type: int8
 * @param[in]      filter_dims   Filter tensor dimensions. Format: [C_OUT, 1, 1, C_IN]
 * @param[in]      filter_data   Filter data pointer. Data type: int8
 * @param[in]      bias_dims     Bias tensor dimensions. Format: [C_OUT]
 * @param[in]      bias_data     Optional bias data pointer. Data type: int32
 * @param[in]      output_dims   Output tensor dimensions. Format: [N, H, W, C_OUT]
 * @param[out]     output_data   Output data pointer. Data type: int8
 *
 * @return     The function returns either
 *                  <code>ARM_CMSIS_NN_ARG_ERROR</code> if argument constraints fail. or,
 *                  <code>ARM_CMSIS_NN_SUCCESS</code> on successful completion.
 *
 * @details
 *   - Supported framework : TensorFlow Lite Micro
 *   - The following constrains on the arguments apply
 *      -# conv_params->padding.w = conv_params->padding.h = 0
 *      -# conv_params->stride.w = conv_params->stride.h = 1
 *
 */
arm_cmsis_nn_status arm_convolve_1x1_s8_fast(const cmsis_nn_context *ctx,
                                             const cmsis_nn_conv_params *conv_params,
                                             const cmsis_nn_per_channel_quant_params *quant_params,
                                             const cmsis_nn_dims *input_dims,
                                             const int8_t *input_data,
                                             const cmsis_nn_dims *filter_dims,
                                             const int8_t *filter_data,
                                             const cmsis_nn_dims *bias_dims,
                                             const int32_t *bias_data,
                                             const cmsis_nn_dims *output_dims,
                                             int8_t *output_data);

/**
 * @brief Get the required buffer size for arm_convolve_1x1_s4_fast
 *
 * @param[in]       input_dims            Input (activation) dimensions
 * @return          The function returns the required buffer size in bytes
 *
 */
int32_t arm_convolve_1x1_s4_fast_get_buffer_size(const cmsis_nn_dims *input_dims);

/**
 * @brief Get the required buffer size for arm_convolve_1x1_s8_fast
 *
 * @param[in]       input_dims            Input (activation) dimensions
 * @return          The function returns the required buffer size in bytes
 *
 */
int32_t arm_convolve_1x1_s8_fast_get_buffer_size(const cmsis_nn_dims *input_dims);

/**
 * @brief s8 version for 1x1 convolution with support for non-unity stride values
 *
 * @param[in, out] ctx           Function context that contains the additional buffer if required by the function.
 *                               None is required by this function.
 * @param[in]      conv_params   Convolution parameters (e.g. strides, dilations, pads,...).
 *                               Range of conv_params->input_offset  : [-127, 128]
 *                               Range of conv_params->output_offset : [-128, 127]
 * @param[in]      quant_params  Per-channel quantization info.
 *                               It contains the multiplier and shift values to be applied to each output channel
 * @param[in]      input_dims    Input (activation) tensor dimensions. Format: [N, H, W, C_IN]
 * @param[in]      input_data    Input (activation) data pointer. Data type: int8
 * @param[in]      filter_dims   Filter tensor dimensions. Format: [C_OUT, 1, 1, C_IN]
 * @param[in]      filter_data   Filter data pointer. Data type: int8
 * @param[in]      bias_dims     Bias tensor dimensions. Format: [C_OUT]
 * @param[in]      bias_data     Optional bias data pointer. Data type: int32
 * @param[in]      output_dims   Output tensor dimensions. Format: [N, H, W, C_OUT]
 * @param[out]     output_data   Output data pointer. Data type: int8
 *
 * @return     The function returns either
 *                  <code>ARM_CMSIS_NN_ARG_ERROR</code> if argument constraints fail. or,
 *                  <code>ARM_CMSIS_NN_SUCCESS</code> on successful completion.
 * @details
 *   - Supported framework : TensorFlow Lite Micro
 *   - The following constrains on the arguments apply
 *      -# conv_params->padding.w = conv_params->padding.h = 0
 *
 */
arm_cmsis_nn_status arm_convolve_1x1_s8(const cmsis_nn_context *ctx,
                                        const cmsis_nn_conv_params *conv_params,
                                        const cmsis_nn_per_channel_quant_params *quant_params,
                                        const cmsis_nn_dims *input_dims,
                                        const int8_t *input_data,
                                        const cmsis_nn_dims *filter_dims,
                                        const int8_t *filter_data,
                                        const cmsis_nn_dims *bias_dims,
                                        const int32_t *bias_data,
                                        const cmsis_nn_dims *output_dims,
                                        int8_t *output_data);

/**
 * @brief 1xn convolution
 *
 * @param[in, out] ctx           Function context that contains the additional buffer if required by the function.
 *                               arm_convolve_1_x_n_s8_get_buffer_size will return the buffer_size if required
 *                               The caller is expected to clear the buffer, if applicable, for security reasons.
 * @param[in]      conv_params   Convolution parameters (e.g. strides, dilations, pads,...).
 *                               Range of conv_params->input_offset  : [-127, 128]
 *                               Range of conv_params->output_offset : [-128, 127]
 * @param[in]      quant_params  Per-channel quantization info.
 *                               It contains the multiplier and shift values to be applied to each output channel
 * @param[in]      input_dims    Input (activation) tensor dimensions. Format: [N, H, W, C_IN]
 * @param[in]      input_data    Input (activation) data pointer. Data type: int8
 * @param[in]      filter_dims   Filter tensor dimensions. Format: [C_OUT, 1, WK, C_IN] where WK is the horizontal
 *                               spatial filter dimension
 * @param[in]      filter_data   Filter data pointer. Data type: int8
 * @param[in]      bias_dims     Bias tensor dimensions. Format: [C_OUT]
 * @param[in]      bias_data     Optional bias data pointer. Data type: int32
 * @param[in]      output_dims   Output tensor dimensions. Format: [N, H, W, C_OUT]
 * @param[out]     output_data   Output data pointer. Data type: int8
 *
 * @return     The function returns either
 *                  <code>ARM_CMSIS_NN_ARG_ERROR</code> if argument constraints fail. or,
 *                  <code>ARM_CMSIS_NN_SUCCESS</code> on successful completion.
 *
 * @details
 *   - Supported framework : TensorFlow Lite Micro
 *   - The following constrains on the arguments apply
 *      -# input_dims->n equals 1
 *      -# ouput_dims->w is a multiple of 4
 *      -# Explicit constraints(since it is for 1xN convolution)
 *      -## input_dims->h equals 1
 *      -## output_dims->h equals 1
 *      -## filter_dims->h equals 1
 *@todo  Remove constraint on output_dims->w to make the function generic.
 *
 */
arm_cmsis_nn_status arm_convolve_1_x_n_s8(const cmsis_nn_context *ctx,
                                          const cmsis_nn_conv_params *conv_params,
                                          const cmsis_nn_per_channel_quant_params *quant_params,
                                          const cmsis_nn_dims *input_dims,
                                          const int8_t *input_data,
                                          const cmsis_nn_dims *filter_dims,
                                          const int8_t *filter_data,
                                          const cmsis_nn_dims *bias_dims,
                                          const int32_t *bias_data,
                                          const cmsis_nn_dims *output_dims,
                                          int8_t *output_data);

/**
 * @brief 1xn convolution for s4 weights
 *
 * @param[in, out] ctx           Function context that contains the additional buffer if required by the function.
 *                               arm_convolve_1_x_n_s4_get_buffer_size will return the buffer_size if required
 *                               The caller is expected to clear the buffer, if applicable, for security reasons.
 * @param[in]      conv_params   Convolution parameters (e.g. strides, dilations, pads,...).
 *                               Range of conv_params->input_offset  : [-127, 128]
 *                               Range of conv_params->output_offset : [-128, 127]
 * @param[in]      quant_params  Per-channel quantization info.
 *                               It contains the multiplier and shift values to be applied to each output channel
 * @param[in]      input_dims    Input (activation) tensor dimensions. Format: [N, H, W, C_IN]
 * @param[in]      input_data    Input (activation) data pointer. Data type: int8
 * @param[in]      filter_dims   Filter tensor dimensions. Format: [C_OUT, 1, WK, C_IN] where WK is the horizontal
 *                               spatial filter dimension
 * @param[in]      filter_data   Filter data pointer. Data type: int8 as packed int4
 * @param[in]      bias_dims     Bias tensor dimensions. Format: [C_OUT]
 * @param[in]      bias_data     Optional bias data pointer. Data type: int32
 * @param[in]      output_dims   Output tensor dimensions. Format: [N, H, W, C_OUT]
 * @param[out]     output_data   Output data pointer. Data type: int8
 *
 * @return     The function returns either
 *                  <code>ARM_CMSIS_NN_ARG_ERROR</code> if argument constraints fail. or,
 *                  <code>ARM_CMSIS_NN_SUCCESS</code> on successful completion.
 *
 * @details
 *   - Supported framework : TensorFlow Lite Micro
 *   - The following constrains on the arguments apply
 *      -# stride.w * input_dims->c is a multiple of 4
 *      -# Explicit constraints(since it is for 1xN convolution)
 *      -## input_dims->h equals 1
 *      -## output_dims->h equals 1
 *      -## filter_dims->h equals 1
 *@todo  Remove constraint on output_dims->w to make the function generic.
 *
 */
arm_cmsis_nn_status arm_convolve_1_x_n_s4(const cmsis_nn_context *ctx,
                                          const cmsis_nn_conv_params *conv_params,
                                          const cmsis_nn_per_channel_quant_params *quant_params,
                                          const cmsis_nn_dims *input_dims,
                                          const int8_t *input_data,
                                          const cmsis_nn_dims *filter_dims,
                                          const int8_t *filter_data,
                                          const cmsis_nn_dims *bias_dims,
                                          const int32_t *bias_data,
                                          const cmsis_nn_dims *output_dims,
                                          int8_t *output_data);

/**
 * @brief Get the required additional buffer size for 1xn convolution
 *
 * @param[in]       conv_params           Convolution parameters (e.g. strides, dilations, pads,...).
 *                                        Range of conv_params->input_offset  : [-127, 128]
 *                                        Range of conv_params->output_offset : [-128, 127]
 * @param[in]       input_dims            Input (activation) tensor dimensions. Format: [N, H, W, C_IN]
 * @param[in]       filter_dims           Filter tensor dimensions. Format: [C_OUT, 1, WK, C_IN] where WK is the
 *                                        horizontal spatial filter dimension
 * @param[in]       output_dims           Output tensor dimensions. Format: [N, H, W, C_OUT]
 *
 * @return          The function returns required buffer size(bytes)
 *
 */
int32_t arm_convolve_1_x_n_s8_get_buffer_size(const cmsis_nn_conv_params *conv_params,
                                              const cmsis_nn_dims *input_dims,
                                              const cmsis_nn_dims *filter_dims,
                                              const cmsis_nn_dims *output_dims);

/**
 * @brief Get the required additional buffer size for 1xn convolution
 *
 * @param[in]       conv_params           Convolution parameters (e.g. strides, dilations, pads,...).
 *                                        Range of conv_params->input_offset  : [-127, 128]
 *                                        Range of conv_params->output_offset : [-128, 127]
 * @param[in]       input_dims            Input (activation) tensor dimensions. Format: [N, H, W, C_IN]
 * @param[in]       filter_dims           Filter tensor dimensions. Format: [C_OUT, 1, WK, C_IN] where WK is the
 *                                        horizontal spatial filter dimension
 * @param[in]       output_dims           Output tensor dimensions. Format: [N, H, W, C_OUT]
 *
 * @return          The function returns required buffer size(bytes)
 *
 */
int32_t arm_convolve_1_x_n_s4_get_buffer_size(const cmsis_nn_conv_params *conv_params,
                                              const cmsis_nn_dims *input_dims,
                                              const cmsis_nn_dims *filter_dims,
                                              const cmsis_nn_dims *output_dims);

/**
 * @brief Wrapper function to pick the right optimized s8 depthwise convolution function
 *
 * @param[in, out] ctx             Function context (e.g. temporary buffer). Check the function
 *                                 definition file to see if an additional buffer is required.
 *                                 Optional function {API}_get_buffer_size() provides the buffer
 *                                 size if required.
 *                                 The caller is expected to clear the buffer, if applicable, for security reasons.
 * @param[in]      dw_conv_params  Depthwise convolution parameters (e.g. strides, dilations, pads,...)
 *                                 dw_conv_params->dilation is not used.
 *                                 Range of dw_conv_params->input_offset : [-127, 128]
 *                                 Range of dw_conv_params->output_offset : [-128, 127]
 * @param[in]      quant_params    Per-channel quantization info.
 *                                 It contains the multiplier and shift values to be applied to each
 *                                 output channel
 * @param[in]      input_dims      Input (activation) tensor dimensions. Format: [H, W, C_IN]
 *                                 Batch argument N is not used and assumed to be 1.
 * @param[in]      input_data      Input (activation) data pointer. Data type: int8
 * @param[in]      filter_dims     Filter tensor dimensions. Format: [1, H, W, C_OUT]
 * @param[in]      filter_data     Filter data pointer. Data type: int8
 * @param[in]      bias_dims       Bias tensor dimensions. Format: [C_OUT]
 * @param[in]      bias_data       Bias data pointer. Data type: int32
 * @param[in]      output_dims     Output tensor dimensions. Format: [1, H, W, C_OUT]
 * @param[in, out] output_data     Output data pointer. Data type: int8
 * @return     The function returns
 *                <code>ARM_CMSIS_NN_SUCCESS</code>   -  Successful completion.
 *
 * @details
 *    - Supported framework: TensorFlow Lite
 *    - Picks one of the the following functions
 *        -# arm_depthwise_conv_s8()
 *        -# arm_depthwise_conv_3x3_s8() - Cortex-M CPUs with DSP extension only
 *        -# arm_depthwise_conv_s8_opt()
 *    - Check details of arm_depthwise_conv_s8_opt() for potential data that can be accessed outside of the
 * boundary.
 */
arm_cmsis_nn_status arm_depthwise_conv_wrapper_s8(const cmsis_nn_context *ctx,
                                                  const cmsis_nn_dw_conv_params *dw_conv_params,
                                                  const cmsis_nn_per_channel_quant_params *quant_params,
                                                  const cmsis_nn_dims *input_dims,
                                                  const int8_t *input_data,
                                                  const cmsis_nn_dims *filter_dims,
                                                  const int8_t *filter_data,
                                                  const cmsis_nn_dims *bias_dims,
                                                  const int32_t *bias_data,
                                                  const cmsis_nn_dims *output_dims,
                                                  int8_t *output_data);

/**
 * @brief Wrapper function to pick the right optimized s4 depthwise convolution function
 *
 * @param[in, out] ctx             Function context (e.g. temporary buffer). Check the function
 *                                 definition file to see if an additional buffer is required.
 *                                 Optional function {API}_get_buffer_size() provides the buffer
 *                                 size if required.
 *                                 The caller is expected to clear the buffer ,if applicable, for security reasons.
 * @param[in]      dw_conv_params  Depthwise convolution parameters (e.g. strides, dilations, pads,...)
 *                                 dw_conv_params->dilation is not used.
 *                                 Range of dw_conv_params->input_offset : [-127, 128]
 *                                 Range of dw_conv_params->output_offset : [-128, 127]
 * @param[in]      quant_params    Per-channel quantization info.
 *                                 It contains the multiplier and shift values to be applied to each
 *                                 output channel
 * @param[in]      input_dims      Input (activation) tensor dimensions. Format: [H, W, C_IN]
 *                                 Batch argument N is not used and assumed to be 1.
 * @param[in]      input_data      Input (activation) data pointer. Data type: int8
 * @param[in]      filter_dims     Filter tensor dimensions. Format: [1, H, W, C_OUT]
 * @param[in]      filter_data     Filter data pointer. Data type: int8_t packed 4-bit weights, e.g four sequential
 *                                 weights [0x1, 0x2, 0x3, 0x4]  packed as [0x21, 0x43].
 * @param[in]      bias_dims       Bias tensor dimensions. Format: [C_OUT]
 * @param[in]      bias_data       Bias data pointer. Data type: int32
 * @param[in]      output_dims     Output tensor dimensions. Format: [1, H, W, C_OUT]
 * @param[in, out] output_data     Output data pointer. Data type: int8
 * @return     The function returns
 *                <code>ARM_CMSIS_NN_SUCCESS</code>   -  Successful completion.
 *
 * @details
 *    - Supported framework: TensorFlow Lite
 */
arm_cmsis_nn_status arm_depthwise_conv_wrapper_s4(const cmsis_nn_context *ctx,
                                                  const cmsis_nn_dw_conv_params *dw_conv_params,
                                                  const cmsis_nn_per_channel_quant_params *quant_params,
                                                  const cmsis_nn_dims *input_dims,
                                                  const int8_t *input_data,
                                                  const cmsis_nn_dims *filter_dims,
                                                  const int8_t *filter_data,
                                                  const cmsis_nn_dims *bias_dims,
                                                  const int32_t *bias_data,
                                                  const cmsis_nn_dims *output_dims,
                                                  int8_t *output_data);

/**
 * @brief Get size of additional buffer required by arm_depthwise_conv_wrapper_s8()
 *
 * @param[in]      dw_conv_params  Depthwise convolution parameters (e.g. strides, dilations, pads,...)
 *                                 Range of dw_conv_params->input_offset : [-127, 128]
 *                                 Range of dw_conv_params->input_offset : [-128, 127]
 * @param[in]      input_dims      Input (activation) tensor dimensions. Format: [H, W, C_IN]
 *                                 Batch argument N is not used and assumed to be 1.
 * @param[in]      filter_dims     Filter tensor dimensions. Format: [1, H, W, C_OUT]
 * @param[in]      output_dims     Output tensor dimensions. Format: [1, H, W, C_OUT]
 * @return                         Size of additional memory required for optimizations in bytes.
 *
 */
int32_t arm_depthwise_conv_wrapper_s8_get_buffer_size(const cmsis_nn_dw_conv_params *dw_conv_params,
                                                      const cmsis_nn_dims *input_dims,
                                                      const cmsis_nn_dims *filter_dims,
                                                      const cmsis_nn_dims *output_dims);

/**
 * @brief Get size of additional buffer required by arm_depthwise_conv_wrapper_s8() for processors with DSP extension.
 *        Refer to arm_depthwise_conv_wrapper_s8_get_buffer_size() for function argument details.
 *
 * @note       Intended for compilation on Host. If compiling for an Arm target, use
 *             arm_depthwise_conv_wrapper_s8_get_buffer_size().
 *
 */
int32_t arm_depthwise_conv_wrapper_s8_get_buffer_size_dsp(const cmsis_nn_dw_conv_params *dw_conv_params,
                                                          const cmsis_nn_dims *input_dims,
                                                          const cmsis_nn_dims *filter_dims,
                                                          const cmsis_nn_dims *output_dims);

/**
 * @brief Get size of additional buffer required by arm_depthwise_conv_wrapper_s8() for Arm(R) Helium Architecture case.
 *        Refer to arm_depthwise_conv_wrapper_s8_get_buffer_size() for function argument details.
 *
 * @note       Intended for compilation on Host. If compiling for an Arm target, use
 *             arm_depthwise_conv_wrapper_s8_get_buffer_size().
 *
 */
int32_t arm_depthwise_conv_wrapper_s8_get_buffer_size_mve(const cmsis_nn_dw_conv_params *dw_conv_params,
                                                          const cmsis_nn_dims *input_dims,
                                                          const cmsis_nn_dims *filter_dims,
                                                          const cmsis_nn_dims *output_dims);

/**
 * @brief Get size of additional buffer required by arm_depthwise_conv_wrapper_s4()
 *
 * @param[in]      dw_conv_params  Depthwise convolution parameters (e.g. strides, dilations, pads,...)
 *                                 Range of dw_conv_params->input_offset : [-127, 128]
 *                                 Range of dw_conv_params->input_offset : [-128, 127]
 * @param[in]      input_dims      Input (activation) tensor dimensions. Format: [H, W, C_IN]
 *                                 Batch argument N is not used and assumed to be 1.
 * @param[in]      filter_dims     Filter tensor dimensions. Format: [1, H, W, C_OUT]
 * @param[in]      output_dims     Output tensor dimensions. Format: [1, H, W, C_OUT]
 * @return                         Size of additional memory required for optimizations in bytes.
 *
 */
int32_t arm_depthwise_conv_wrapper_s4_get_buffer_size(const cmsis_nn_dw_conv_params *dw_conv_params,
                                                      const cmsis_nn_dims *input_dims,
                                                      const cmsis_nn_dims *filter_dims,
                                                      const cmsis_nn_dims *output_dims);

/**
 * @brief Get size of additional buffer required by arm_depthwise_conv_wrapper_s4() for processors with DSP extension.
 *        Refer to arm_depthwise_conv_wrapper_s4_get_buffer_size() for function argument details.
 *
 * @note       Intended for compilation on Host. If compiling for an Arm target, use
 *             arm_depthwise_conv_wrapper_s4_get_buffer_size().
 *
 */
int32_t arm_depthwise_conv_wrapper_s4_get_buffer_size_dsp(const cmsis_nn_dw_conv_params *dw_conv_params,
                                                          const cmsis_nn_dims *input_dims,
                                                          const cmsis_nn_dims *filter_dims,
                                                          const cmsis_nn_dims *output_dims);

/**
 * @brief Get size of additional buffer required by arm_depthwise_conv_wrapper_s4() for Arm(R) Helium Architecture case.
 *        Refer to arm_depthwise_conv_wrapper_s4_get_buffer_size() for function argument details.
 *
 * @note       Intended for compilation on Host. If compiling for an Arm target, use
 *             arm_depthwise_conv_wrapper_s4_get_buffer_size().
 *
 */
int32_t arm_depthwise_conv_wrapper_s4_get_buffer_size_mve(const cmsis_nn_dw_conv_params *dw_conv_params,
                                                          const cmsis_nn_dims *input_dims,
                                                          const cmsis_nn_dims *filter_dims,
                                                          const cmsis_nn_dims *output_dims);

/**
 * @brief Basic s8 depthwise convolution function that doesn't have any constraints on the input dimensions.
 *
 * @param[in, out] ctx             Function context (e.g. temporary buffer). Check the function
 *                                 definition file to see if an additional buffer is required.
 *                                 Optional function {API}_get_buffer_size() provides the buffer
 *                                 size if an additional buffer is required exists if additional memory is.
 *                                 The caller is expected to clear the buffer, if applicable, for security reasons.
 * @param[in]      dw_conv_params  Depthwise convolution parameters (e.g. strides, dilations, pads,...)
 *                                 dw_conv_params->dilation is not used.
 *                                 Range of dw_conv_params->input_offset : [-127, 128]
 *                                 Range of dw_conv_params->input_offset : [-128, 127]
 * @param[in]      quant_params    Per-channel quantization info.
 *                                 It contains the multiplier and shift values to be applied to each
 *                                 output channel
 * @param[in]      input_dims      Input (activation) tensor dimensions. Format: [N, H, W, C_IN]
 *                                 Batch argument N is not used.
 * @param[in]      input_data      Input (activation) data pointer. Data type: int8
 * @param[in]      filter_dims     Filter tensor dimensions. Format: [1, H, W, C_OUT]
 * @param[in]      filter_data     Filter data pointer. Data type: int8
 * @param[in]      bias_dims       Bias tensor dimensions. Format: [C_OUT]
 * @param[in]      bias_data       Bias data pointer. Data type: int32
 * @param[in]      output_dims     Output tensor dimensions. Format: [N, H, W, C_OUT]
 * @param[in, out] output_data     Output data pointer. Data type: int8
 * @return     The function returns <code>ARM_CMSIS_NN_SUCCESS</code>
 *
 * @details
 *    - Supported framework: TensorFlow Lite
 */
arm_cmsis_nn_status arm_depthwise_conv_s8(const cmsis_nn_context *ctx,
                                          const cmsis_nn_dw_conv_params *dw_conv_params,
                                          const cmsis_nn_per_channel_quant_params *quant_params,
                                          const cmsis_nn_dims *input_dims,
                                          const int8_t *input_data,
                                          const cmsis_nn_dims *filter_dims,
                                          const int8_t *filter_data,
                                          const cmsis_nn_dims *bias_dims,
                                          const int32_t *bias_data,
                                          const cmsis_nn_dims *output_dims,
                                          int8_t *output_data);

/**
 * @brief Basic s4 depthwise convolution function that doesn't have any constraints on the input dimensions.
 *
 * @param[in, out] ctx             Function context (e.g. temporary buffer). Check the function
 *                                 definition file to see if an additional buffer is required.
 *                                 Optional function {API}_get_buffer_size() provides the buffer
 *                                 size if an additional buffer is required exists if additional memory is.
 *                                 The caller is expected to clear the buffer ,if applicable, for security reasons.
 * @param[in]      dw_conv_params  Depthwise convolution parameters (e.g. strides, dilations, pads,...)
 *                                 dw_conv_params->dilation is not used.
 *                                 Range of dw_conv_params->input_offset : [-127, 128]
 *                                 Range of dw_conv_params->input_offset : [-128, 127]
 * @param[in]      quant_params    Per-channel quantization info.
 *                                 It contains the multiplier and shift values to be applied to each
 *                                 output channel
 * @param[in]      input_dims      Input (activation) tensor dimensions. Format: [N, H, W, C_IN]
 *                                 Batch argument N is not used.
 * @param[in]      input           Input (activation) data pointer. Data type: int8
 * @param[in]      filter_dims     Filter tensor dimensions. Format: [1, H, W, C_OUT]
 * @param[in]      kernel          Filter data pointer. Data type: int8_t packed 4-bit weights, e.g four sequential
 *                                 weights [0x1, 0x2, 0x3, 0x4]  packed as [0x21, 0x43].
 * @param[in]      bias_dims       Bias tensor dimensions. Format: [C_OUT]
 * @param[in]      bias            Bias data pointer. Data type: int32
 * @param[in]      output_dims     Output tensor dimensions. Format: [N, H, W, C_OUT]
 * @param[in, out] output          Output data pointer. Data type: int8
 * @return     The function returns <code>ARM_CMSIS_NN_SUCCESS</code>
 *
 * @details
 *    - Supported framework: TensorFlow Lite
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
                                          int8_t *output);

/**
 * @brief Basic s16 depthwise convolution function that doesn't have any constraints on the input dimensions.
 *
 * @param[in, out] ctx             Function context (e.g. temporary buffer). Check the function
 *                                 definition file to see if an additional buffer is required.
 *                                 Optional function {API}_get_buffer_size() provides the buffer
 *                                 size if an additional buffer is required.
 *                                 exists if additional memory is.
 *                                 The caller is expected to clear the buffer, if applicable, for security reasons.
 * @param[in]      dw_conv_params  Depthwise convolution parameters (e.g. strides, dilations, pads,...)
 *                                 conv_params->input_offset  : Not used
 *                                 conv_params->output_offset : Not used
 * @param[in]      quant_params    Per-channel quantization info.
 *                                 It contains the multiplier and shift values to be applied to each
 *                                 output channel
 * @param[in]      input_dims      Input (activation) tensor dimensions. Format: [N, H, W, C_IN]
 *                                 Batch argument N is not used.
 * @param[in]      input_data      Input (activation) data pointer. Data type: int8
 * @param[in]      filter_dims     Filter tensor dimensions. Format: [1, H, W, C_OUT]
 * @param[in]      filter_data     Filter data pointer. Data type: int8
 * @param[in]      bias_dims       Bias tensor dimensions. Format: [C_OUT]
 * @param[in]      bias_data       Bias data pointer. Data type: int64
 * @param[in]      output_dims     Output tensor dimensions. Format: [N, H, W, C_OUT]
 * @param[in, out] output_data     Output data pointer. Data type: int16
 * @return     The function returns <code>ARM_CMSIS_NN_SUCCESS</code>
 *
 * @details
 *    - Supported framework: TensorFlow Lite
 */
arm_cmsis_nn_status arm_depthwise_conv_s16(const cmsis_nn_context *ctx,
                                           const cmsis_nn_dw_conv_params *dw_conv_params,
                                           const cmsis_nn_per_channel_quant_params *quant_params,
                                           const cmsis_nn_dims *input_dims,
                                           const int16_t *input_data,
                                           const cmsis_nn_dims *filter_dims,
                                           const int8_t *filter_data,
                                           const cmsis_nn_dims *bias_dims,
                                           const int64_t *bias_data,
                                           const cmsis_nn_dims *output_dims,
                                           int16_t *output_data);

/**
 * @brief Wrapper function to pick the right optimized s16 depthwise convolution function
 *
 * @param[in, out] ctx             Function context (e.g. temporary buffer). Check the function
 *                                 definition file to see if an additional buffer is required.
 *                                 Optional function {API}_get_buffer_size() provides the buffer
 *                                 size if required.
 *                                 The caller is expected to clear the buffer, if applicable, for security reasons.
 * @param[in]      dw_conv_params  Depthwise convolution parameters (e.g. strides, dilations, pads,...)
 *                                 dw_conv_params->dilation is not used.
 *                                 Range of dw_conv_params->input_offset : Not used
 *                                 Range of dw_conv_params->output_offset : Not used
 * @param[in]      quant_params    Per-channel quantization info.
 *                                 It contains the multiplier and shift values to be applied to each
 *                                 output channel
 * @param[in]      input_dims      Input (activation) tensor dimensions. Format: [H, W, C_IN]
 *                                 Batch argument N is not used and assumed to be 1.
 * @param[in]      input_data      Input (activation) data pointer. Data type: int16
 * @param[in]      filter_dims     Filter tensor dimensions. Format: [1, H, W, C_OUT]
 * @param[in]      filter_data     Filter data pointer. Data type: int8
 * @param[in]      bias_dims       Bias tensor dimensions. Format: [C_OUT]
 * @param[in]      bias_data       Bias data pointer. Data type: int64
 * @param[in]      output_dims     Output tensor dimensions. Format: [1, H, W, C_OUT]
 * @param[in, out] output_data     Output data pointer. Data type: int16
 * @return     The function returns
 *                <code>ARM_CMSIS_NN_SUCCESS</code>   -  Successful completion.
 *
 * @details
 *    - Supported framework: TensorFlow Lite
 *    - Picks one of the the following functions
 *        -# arm_depthwise_conv_s16()
 *        -# arm_depthwise_conv_fast_s16()  - Cortex-M CPUs with DSP extension only
 */
arm_cmsis_nn_status arm_depthwise_conv_wrapper_s16(const cmsis_nn_context *ctx,
                                                   const cmsis_nn_dw_conv_params *dw_conv_params,
                                                   const cmsis_nn_per_channel_quant_params *quant_params,
                                                   const cmsis_nn_dims *input_dims,
                                                   const int16_t *input_data,
                                                   const cmsis_nn_dims *filter_dims,
                                                   const int8_t *filter_data,
                                                   const cmsis_nn_dims *bias_dims,
                                                   const int64_t *bias_data,
                                                   const cmsis_nn_dims *output_dims,
                                                   int16_t *output_data);

/**
 * @brief Get size of additional buffer required by arm_depthwise_conv_wrapper_s16()
 *
 * @param[in]      dw_conv_params  Depthwise convolution parameters (e.g. strides, dilations, pads,...)
 *                                 Range of dw_conv_params->input_offset : Not used
 *                                 Range of dw_conv_params->input_offset : Not used
 * @param[in]      input_dims      Input (activation) tensor dimensions. Format: [H, W, C_IN]
 *                                 Batch argument N is not used and assumed to be 1.
 * @param[in]      filter_dims     Filter tensor dimensions. Format: [1, H, W, C_OUT]
 * @param[in]      output_dims     Output tensor dimensions. Format: [1, H, W, C_OUT]
 * @return                         Size of additional memory required for optimizations in bytes.
 *
 */
int32_t arm_depthwise_conv_wrapper_s16_get_buffer_size(const cmsis_nn_dw_conv_params *dw_conv_params,
                                                       const cmsis_nn_dims *input_dims,
                                                       const cmsis_nn_dims *filter_dims,
                                                       const cmsis_nn_dims *output_dims);

/**
 * @brief Get size of additional buffer required by arm_depthwise_conv_wrapper_s16() for processors with DSP extension.
 *        Refer to arm_depthwise_conv_wrapper_s16_get_buffer_size() for function argument details.
 *
 * @note       Intended for compilation on Host. If compiling for an Arm target, use
 *             arm_depthwise_conv_wrapper_s16_get_buffer_size().
 *
 */
int32_t arm_depthwise_conv_wrapper_s16_get_buffer_size_dsp(const cmsis_nn_dw_conv_params *dw_conv_params,
                                                           const cmsis_nn_dims *input_dims,
                                                           const cmsis_nn_dims *filter_dims,
                                                           const cmsis_nn_dims *output_dims);

/**
 * @brief Get size of additional buffer required by arm_depthwise_conv_wrapper_s16() for Arm(R) Helium Architecture
 * case. Refer to arm_depthwise_conv_wrapper_s16_get_buffer_size() for function argument details.
 *
 * @note       Intended for compilation on Host. If compiling for an Arm target, use
 *             arm_depthwise_conv_wrapper_s16_get_buffer_size().
 *
 */
int32_t arm_depthwise_conv_wrapper_s16_get_buffer_size_mve(const cmsis_nn_dw_conv_params *dw_conv_params,
                                                           const cmsis_nn_dims *input_dims,
                                                           const cmsis_nn_dims *filter_dims,
                                                           const cmsis_nn_dims *output_dims);

/**
 * @brief Optimized s16 depthwise convolution function with constraint that in_channel equals out_channel.
 *        Refer arm_depthwise_conv_s16() for function argument details.
 *
 * @return     The function returns one of the following
 *                <code>ARM_CMSIS_NN_ARG_ERROR</code> - ctx-buff == NULL and
 *                                                      arm_depthwise_conv_fast_s16_get_buffer_size() > 0 or
 *                                                      input channel != output channel or
 *                                                      ch_mult != 1
 *
 *                <code>ARM_CMSIS_NN_SUCCESS</code> - Successful operation
 *
 * @details
 *    - Supported framework: TensorFlow Lite
 *    - The following constrains on the arguments apply
 *        -# Number of input channel equals number of output channels or ch_mult equals 1
 *    - Reccomended when number of channels is 4 or greater.
 *
 */
arm_cmsis_nn_status arm_depthwise_conv_fast_s16(const cmsis_nn_context *ctx,
                                                const cmsis_nn_dw_conv_params *dw_conv_params,
                                                const cmsis_nn_per_channel_quant_params *quant_params,
                                                const cmsis_nn_dims *input_dims,
                                                const int16_t *input_data,
                                                const cmsis_nn_dims *filter_dims,
                                                const int8_t *filter_data,
                                                const cmsis_nn_dims *bias_dims,
                                                const int64_t *bias_data,
                                                const cmsis_nn_dims *output_dims,
                                                int16_t *output_data);

/**
 * @brief Get the required buffer size for optimized s16 depthwise convolution
 * function with constraint that in_channel equals out_channel.
 * @param[in]       input_dims   Input (activation) tensor dimensions. Format: [1, H, W, C_IN]
 *                               Batch argument N is not used.
 * @param[in]       filter_dims  Filter tensor dimensions. Format: [1, H, W, C_OUT]
 * @return          The function returns required buffer size in bytes
 *
 */
int32_t arm_depthwise_conv_fast_s16_get_buffer_size(const cmsis_nn_dims *input_dims, const cmsis_nn_dims *filter_dims);

/**
 * @brief Optimized s8 depthwise convolution function for 3x3 kernel size with some constraints on
 *        the input arguments(documented below). Refer arm_depthwise_conv_s8() for function
 *        argument details.
 *
 * @return     The function returns one of the following
 *                <code>ARM_CMSIS_NN_ARG_ERROR</code> - Unsupported dimension of tensors
 *                                                    - Unsupported pad size along the x axis
 *                <code>ARM_CMSIS_NN_SUCCESS</code> - Successful operation
 *
 * @details
 *   - Supported framework : TensorFlow Lite Micro
 *   - The following constrains on the arguments apply
 *      -# Number of input channel equals number of output channels
 *      -# Filter height and width equals 3
 *      -# Padding along x is either 0 or 1.
 *
 */
arm_cmsis_nn_status arm_depthwise_conv_3x3_s8(const cmsis_nn_context *ctx,
                                              const cmsis_nn_dw_conv_params *dw_conv_params,
                                              const cmsis_nn_per_channel_quant_params *quant_params,
                                              const cmsis_nn_dims *input_dims,
                                              const int8_t *input_data,
                                              const cmsis_nn_dims *filter_dims,
                                              const int8_t *filter_data,
                                              const cmsis_nn_dims *bias_dims,
                                              const int32_t *bias_data,
                                              const cmsis_nn_dims *output_dims,
                                              int8_t *output_data);

/**
 * @brief Optimized s8 depthwise convolution function with constraint that in_channel equals out_channel.
 *        Refer arm_depthwise_conv_s8() for function argument details.
 *
 * @return     The function returns one of the following
 *                <code>ARM_CMSIS_NN_ARG_ERROR</code> - input channel != output channel or
 *                                                      ch_mult != 1
 *                <code>ARM_CMSIS_NN_SUCCESS</code> - Successful operation
 *
 * @note       If number of channels is not a multiple of 4, upto 3 elements outside the boundary will be read out
 *             for the following if MVE optimizations(Arm Helium Technology) are used.
 *               - Output shift
 *               - Output multiplier
 *               - Output bias
 *               - kernel
 * @details
 *    - Supported framework: TensorFlow Lite
 *    - The following constrains on the arguments apply
 *        -# Number of input channel equals number of output channels or ch_mult equals 1
 *    - Reccomended when number of channels is 4 or greater.
 *
 */
arm_cmsis_nn_status arm_depthwise_conv_s8_opt(const cmsis_nn_context *ctx,
                                              const cmsis_nn_dw_conv_params *dw_conv_params,
                                              const cmsis_nn_per_channel_quant_params *quant_params,
                                              const cmsis_nn_dims *input_dims,
                                              const int8_t *input_data,
                                              const cmsis_nn_dims *filter_dims,
                                              const int8_t *filter_data,
                                              const cmsis_nn_dims *bias_dims,
                                              const int32_t *bias_data,
                                              const cmsis_nn_dims *output_dims,
                                              int8_t *output_data);

/**
 * @brief Optimized s4 depthwise convolution function with constraint that in_channel equals out_channel.
 *        Refer arm_depthwise_conv_s4() for function argument details.
 *
 * @return     The function returns one of the following
 *                <code>ARM_CMSIS_NN_ARG_ERROR</code> - input channel != output channel or
 *                                                      ch_mult != 1
 *                <code>ARM_CMSIS_NN_SUCCESS</code> - Successful operation
 *
 * @note       If number of channels is not a multiple of 4, upto 3 elements outside the boundary will be read out
 *             for the following if MVE optimizations(Arm Helium Technology) are used.
 *               - Output shift
 *               - Output multiplier
 *               - Output bias
 *               - kernel
 * @details
 *    - Supported framework: TensorFlow Lite
 *    - The following constrains on the arguments apply
 *        -# Number of input channel equals number of output channels or ch_mult equals 1
 *    - Reccomended when number of channels is 4 or greater.
 *
 */
arm_cmsis_nn_status arm_depthwise_conv_s4_opt(const cmsis_nn_context *ctx,
                                              const cmsis_nn_dw_conv_params *dw_conv_params,
                                              const cmsis_nn_per_channel_quant_params *quant_params,
                                              const cmsis_nn_dims *input_dims,
                                              const int8_t *input_data,
                                              const cmsis_nn_dims *filter_dims,
                                              const int8_t *filter_data,
                                              const cmsis_nn_dims *bias_dims,
                                              const int32_t *bias_data,
                                              const cmsis_nn_dims *output_dims,
                                              int8_t *output_data);

/**
 * @brief Get the required buffer size for optimized s8 depthwise convolution
 * function with constraint that in_channel equals out_channel.
 * @param[in]       input_dims   Input (activation) tensor dimensions. Format: [1, H, W, C_IN]
 *                               Batch argument N is not used.
 * @param[in]       filter_dims  Filter tensor dimensions. Format: [1, H, W, C_OUT]
 * @return          The function returns required buffer size in bytes
 *
 */
int32_t arm_depthwise_conv_s8_opt_get_buffer_size(const cmsis_nn_dims *input_dims, const cmsis_nn_dims *filter_dims);

/**
 * @brief Get the required buffer size for optimized s4 depthwise convolution
 * function with constraint that in_channel equals out_channel.
 * @param[in]       input_dims   Input (activation) tensor dimensions. Format: [1, H, W, C_IN]
 *                               Batch argument N is not used.
 * @param[in]       filter_dims  Filter tensor dimensions. Format: [1, H, W, C_OUT]
 * @return          The function returns required buffer size in bytes
 *
 */
int32_t arm_depthwise_conv_s4_opt_get_buffer_size(const cmsis_nn_dims *input_dims, const cmsis_nn_dims *filter_dims);

/**
 * @defgroup FC Fully-connected Layer Functions
 *
 * Collection of fully-connected and matrix multiplication functions.
 *
 * Fully-connected layer is basically a matrix-vector multiplication
 * with bias. The matrix is the weights and the input/output vectors
 * are the activation values. Supported {weight, activation} precisions
 * include {8-bit, 8-bit} and {8-bit, 16-bit}
 *
 *
 */

/**
 * @brief Basic s4 Fully Connected function.
 *
 * @param[in, out] ctx           Function context (e.g. temporary buffer). Check the function
 *                               definition file to see if an additional buffer is required.
 *                               Optional function {API}_get_buffer_size() provides the buffer
 *                               size if an additional buffer is required.
 *                               The caller is expected to clear the buffer ,if applicable, for security reasons.
 * @param[in]      fc_params     Fully Connected layer parameters.
 *                               Range of fc_params->input_offset  : [-127, 128]
 *                               fc_params->filter_offset : 0
 *                               Range of fc_params->output_offset : [-128, 127]
 * @param[in]      quant_params  Per-tensor quantization info.
 *                               It contains the multiplier and shift value to be applied to the output tensor.
 * @param[in]      input_dims    Input (activation) tensor dimensions. Format: [N, H, W, C_IN]
 *                               Input dimension is taken as Nx(H * W * C_IN)
 * @param[in]      input_data    Input (activation) data pointer. Data type: int8
 * @param[in]      filter_dims   Two dimensional filter dimensions. Format: [N, C]
 *                               N : accumulation depth and equals (H * W * C_IN) from input_dims
 *                               C : output depth and equals C_OUT in output_dims
 *                               H & W : Not used
 * @param[in]      filter_data   Filter data pointer. Data type: int8_t packed 4-bit weights, e.g four sequential
 *                               weights [0x1, 0x2, 0x3, 0x4]  packed as [0x21, 0x43].
 * @param[in]      bias_dims     Bias tensor dimensions. Format: [C_OUT]
 *                               N, H, W : Not used
 * @param[in]      bias_data     Bias data pointer. Data type: int32
 * @param[in]      output_dims   Output tensor dimensions. Format: [N, C_OUT]
 *                               N : Batches
 *                               C_OUT : Output depth
 *                               H & W : Not used.
 * @param[in, out] output_data    Output data pointer. Data type: int8
 * @return     The function returns <code>ARM_CMSIS_NN_SUCCESS</code>
 *
 * @details
 *    - Supported framework: TensorFlow Lite
 */
arm_cmsis_nn_status arm_fully_connected_s4(const cmsis_nn_context *ctx,
                                           const cmsis_nn_fc_params *fc_params,
                                           const cmsis_nn_per_tensor_quant_params *quant_params,
                                           const cmsis_nn_dims *input_dims,
                                           const int8_t *input_data,
                                           const cmsis_nn_dims *filter_dims,
                                           const int8_t *filter_data,
                                           const cmsis_nn_dims *bias_dims,
                                           const int32_t *bias_data,
                                           const cmsis_nn_dims *output_dims,
                                           int8_t *output_data);

/**
 * @brief Basic s8 Fully Connected function.
 *
 * @param[in, out] ctx           Function context (e.g. temporary buffer). Check the function
 *                               definition file to see if an additional buffer is required.
 *                               Optional function {API}_get_buffer_size() provides the buffer
 *                               size if an additional buffer is required.
 *                               The caller is expected to clear the buffer, if applicable, for security reasons.
 * @param[in]      fc_params     Fully Connected layer parameters.
 *                               Range of fc_params->input_offset  : [-127, 128]
 *                               fc_params->filter_offset : 0
 *                               Range of fc_params->output_offset : [-128, 127]
 * @param[in]      quant_params  Per-tensor quantization info.
 *                               It contains the multiplier and shift value to be applied to the output tensor.
 * @param[in]      input_dims    Input (activation) tensor dimensions. Format: [N, H, W, C_IN]
 *                               Input dimension is taken as Nx(H * W * C_IN)
 * @param[in]      input_data    Input (activation) data pointer. Data type: int8
 * @param[in]      filter_dims   Two dimensional filter dimensions. Format: [N, C]
 *                               N : accumulation depth and equals (H * W * C_IN) from input_dims
 *                               C : output depth and equals C_OUT in output_dims
 *                               H & W : Not used
 * @param[in]      filter_data   Filter data pointer. Data type: int8
 * @param[in]      bias_dims     Bias tensor dimensions. Format: [C_OUT]
 *                               N, H, W : Not used
 * @param[in]      bias_data     Bias data pointer. Data type: int32
 * @param[in]      output_dims   Output tensor dimensions. Format: [N, C_OUT]
 *                               N : Batches
 *                               C_OUT : Output depth
 *                               H & W : Not used.
 * @param[in, out] output_data    Output data pointer. Data type: int8
 *
 * @return     The function returns either
 *                  <code>ARM_CMSIS_NN_ARG_ERROR</code> if argument constraints fail. or,
 *                  <code>ARM_CMSIS_NN_SUCCESS</code> on successful completion.
 *
 * @details
 *    - Supported framework: TensorFlow Lite
 */
arm_cmsis_nn_status arm_fully_connected_s8(const cmsis_nn_context *ctx,
                                           const cmsis_nn_fc_params *fc_params,
                                           const cmsis_nn_per_tensor_quant_params *quant_params,
                                           const cmsis_nn_dims *input_dims,
                                           const int8_t *input_data,
                                           const cmsis_nn_dims *filter_dims,
                                           const int8_t *filter_data,
                                           const cmsis_nn_dims *bias_dims,
                                           const int32_t *bias_data,
                                           const cmsis_nn_dims *output_dims,
                                           int8_t *output_data);

/**
 * @brief Basic s8 Fully Connected function using per channel quantization.
 *
 * @param[in, out] ctx           Function context (e.g. temporary buffer). Check the function
 *                               definition file to see if an additional buffer is required.
 *                               Optional function {API}_get_buffer_size() provides the buffer
 *                               size if an additional buffer is required.
 *                               The caller is expected to clear the buffer, if applicable, for security reasons.
 * @param[in]      fc_params     Fully Connected layer parameters.
 *                               Range of fc_params->input_offset  : [-127, 128]
 *                               fc_params->filter_offset : 0
 *                               Range of fc_params->output_offset : [-128, 127]
 * @param[in]      quant_params  Per-channel quantization info.
 *                               It contains the multiplier and shift values to be applied to each output channel
 * @param[in]      input_dims    Input (activation) tensor dimensions. Format: [N, H, W, C_IN]
 *                               Input dimension is taken as Nx(H * W * C_IN)
 * @param[in]      input_data    Input (activation) data pointer. Data type: int8
 * @param[in]      filter_dims   Two dimensional filter dimensions. Format: [N, C]
 *                               N : accumulation depth and equals (H * W * C_IN) from input_dims
 *                               C : output depth and equals C_OUT in output_dims
 *                               H & W : Not used
 * @param[in]      filter_data   Filter data pointer. Data type: int8
 * @param[in]      bias_dims     Bias tensor dimensions. Format: [C_OUT]
 *                               N, H, W : Not used
 * @param[in]      bias_data     Bias data pointer. Data type: int32
 * @param[in]      output_dims   Output tensor dimensions. Format: [N, C_OUT]
 *                               N : Batches
 *                               C_OUT : Output depth
 *                               H & W : Not used.
 * @param[in, out] output_data    Output data pointer. Data type: int8
 *
 * @return     The function returns either
 *                  <code>ARM_CMSIS_NN_ARG_ERROR</code> if argument constraints fail. or,
 *                  <code>ARM_CMSIS_NN_SUCCESS</code> on successful completion.
 *
 * @details
 *    - Supported framework: TensorFlow Lite
 */
arm_cmsis_nn_status arm_fully_connected_per_channel_s8(const cmsis_nn_context *ctx,
                                                       const cmsis_nn_fc_params *fc_params,
                                                       const cmsis_nn_per_channel_quant_params *quant_params,
                                                       const cmsis_nn_dims *input_dims,
                                                       const int8_t *input_data,
                                                       const cmsis_nn_dims *filter_dims,
                                                       const int8_t *filter_data,
                                                       const cmsis_nn_dims *bias_dims,
                                                       const int32_t *bias_data,
                                                       const cmsis_nn_dims *output_dims,
                                                       int8_t *output_data);

/**
 * @brief s8 Fully Connected layer wrapper function
 *
 * @param[in, out] ctx           Function context (e.g. temporary buffer). Check the function
 *                               definition file to see if an additional buffer is required.
 *                               Optional function {API}_get_buffer_size() provides the buffer
 *                               size if an additional buffer is required.
 *                               The caller is expected to clear the buffer, if applicable, for security reasons.
 * @param[in]      fc_params     Fully Connected layer parameters.
 *                               Range of fc_params->input_offset  : [-127, 128]
 *                               fc_params->filter_offset : 0
 *                               Range of fc_params->output_offset : [-128, 127]
 * @param[in]      quant_params  Per-channel or per-tensor quantization info. Check struct defintion for details.
 *                               It contains the multiplier and shift value(s) to be applied to each output channel
 * @param[in]      input_dims    Input (activation) tensor dimensions. Format: [N, H, W, C_IN]
 *                               Input dimension is taken as Nx(H * W * C_IN)
 * @param[in]      input_data    Input (activation) data pointer. Data type: int8
 * @param[in]      filter_dims   Two dimensional filter dimensions. Format: [N, C]
 *                               N : accumulation depth and equals (H * W * C_IN) from input_dims
 *                               C : output depth and equals C_OUT in output_dims
 *                               H & W : Not used
 * @param[in]      filter_data   Filter data pointer. Data type: int8
 * @param[in]      bias_dims     Bias tensor dimensions. Format: [C_OUT]
 *                               N, H, W : Not used
 * @param[in]      bias_data     Bias data pointer. Data type: int32
 * @param[in]      output_dims   Output tensor dimensions. Format: [N, C_OUT]
 *                               N : Batches
 *                               C_OUT : Output depth
 *                               H & W : Not used.
 * @param[in, out] output_data    Output data pointer. Data type: int8
 *
 * @return     The function returns either
 *                  <code>ARM_CMSIS_NN_ARG_ERROR</code> if argument constraints fail. or,
 *                  <code>ARM_CMSIS_NN_SUCCESS</code> on successful completion.
 *
 * @details
 *    - Supported framework: TensorFlow Lite
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
                                                   int8_t *output_data);

/**
 * @brief Calculate the sum of each row in vector_data, multiply by lhs_offset and optionally add s32 bias_data.
 * @param[in, out]      vector_sum_buf              Buffer for vector sums
 * @param[in]           vector_cols                 Number of vector columns
 * @param[in]           vector_rows                 Number of vector rows
 * @param[in]           vector_data                 Vector of weigths data
 * @param[in]           lhs_offset                  Constant multiplied with each sum
 * @param[in]           rhs_offset                  Constant added to each vector element before sum
 * @param[in]           bias_data                   Vector of bias data, added to each sum.
 * @return              The function returns
 *                         <code>ARM_CMSIS_NN_SUCCESS</code> - Successful operation
 */
arm_cmsis_nn_status arm_vector_sum_s8(int32_t *vector_sum_buf,
                                      const int32_t vector_cols,
                                      const int32_t vector_rows,
                                      const int8_t *vector_data,
                                      const int32_t lhs_offset,
                                      const int32_t rhs_offset,
                                      const int32_t *bias_data);

/**
 * @brief Calculate the sum of each row in vector_data, multiply by lhs_offset and optionally add s64 bias_data.
 * @param[in, out]      vector_sum_buf              Buffer for vector sums
 * @param[in]           vector_cols                 Number of vector columns
 * @param[in]           vector_rows                 Number of vector rows
 * @param[in]           vector_data                 Vector of weigths data
 * @param[in]           lhs_offset                  Constant multiplied with each sum
 * @param[in]           bias_data                   Vector of bias data, added to each sum.
 * @return              The function returns
 *                         <code>ARM_CMSIS_NN_SUCCESS</code> - Successful operation
 */
arm_cmsis_nn_status arm_vector_sum_s8_s64(int64_t *vector_sum_buf,
                                          const int32_t vector_cols,
                                          const int32_t vector_rows,
                                          const int8_t *vector_data,
                                          const int32_t lhs_offset,
                                          const int64_t *bias_data);

/**
 * @brief Get size of additional buffer required by arm_fully_connected_s8().
 *        See also arm_vector_sum_s8, which is required if buffer size is > 0.
 * @param[in]      filter_dims             dimension of filter
 * @return         The function returns    required buffer size in bytes
 *
 */
int32_t arm_fully_connected_s8_get_buffer_size(const cmsis_nn_dims *filter_dims);

/**
 * @brief Get size of additional buffer required by arm_fully_connected_s8() for processors with DSP extension.
 *        Refer to arm_fully_connected_s8_get_buffer_size() for function argument details.
 *
 * @note       Intended for compilation on Host. If compiling for an Arm target, use
 *             arm_fully_connected_s8_get_buffer_size().
 *
 */
int32_t arm_fully_connected_s8_get_buffer_size_dsp(const cmsis_nn_dims *filter_dims);

/**
 * @brief Get size of additional buffer required by arm_fully_connected_s8() for Arm(R) Helium Architecture case.
 *        Refer to arm_fully_connected_s8_get_buffer_size() for function argument details.
 *
 * @note       Intended for compilation on Host. If compiling for an Arm target, use
 *             arm_fully_connected_s8_get_buffer_size().
 *
 */
int32_t arm_fully_connected_s8_get_buffer_size_mve(const cmsis_nn_dims *filter_dims);

/**
 * @brief Basic s16 Fully Connected function.
 *
 * @param[in, out] ctx           Function context (e.g. temporary buffer). Check the function
 *                               definition file to see if an additional buffer is required.
 *                               Optional function {API}_get_buffer_size() provides the buffer
 *                               size if an additional buffer is required.
 *                               The caller is expected to clear the buffer, if applicable, for security reasons.
 * @param[in]      fc_params     Fully Connected layer parameters.
 *                               fc_params->input_offset  : 0
 *                               fc_params->filter_offset : 0
 *                               fc_params->output_offset : 0
 * @param[in]      quant_params  Per-tensor quantization info.
 *                               It contains the multiplier and shift value to be applied to the output tensor.
 * @param[in]      input_dims    Input (activation) tensor dimensions. Format: [N, H, W, C_IN]
 *                               Input dimension is taken as Nx(H * W * C_IN)
 * @param[in]      input_data    Input (activation) data pointer. Data type: int16
 * @param[in]      filter_dims   Two dimensional filter dimensions. Format: [N, C]
 *                               N : accumulation depth and equals (H * W * C_IN) from input_dims
 *                               C : output depth and equals C_OUT in output_dims
 *                               H & W : Not used
 * @param[in]      filter_data   Filter data pointer. Data type: int8
 * @param[in]      bias_dims     Bias tensor dimensions. Format: [C_OUT]
 *                               N, H, W : Not used
 * @param[in]      bias_data     Bias data pointer. Data type: int64
 * @param[in]      output_dims   Output tensor dimensions. Format: [N, C_OUT]
 *                               N : Batches
 *                               C_OUT : Output depth
 *                               H & W : Not used.
 * @param[in, out] output_data    Output data pointer. Data type: int16
 * @return     The function returns <code>ARM_CMSIS_NN_SUCCESS</code>
 *
 * @details
 *    - Supported framework: TensorFlow Lite
 */
arm_cmsis_nn_status arm_fully_connected_s16(const cmsis_nn_context *ctx,
                                            const cmsis_nn_fc_params *fc_params,
                                            const cmsis_nn_per_tensor_quant_params *quant_params,
                                            const cmsis_nn_dims *input_dims,
                                            const int16_t *input_data,
                                            const cmsis_nn_dims *filter_dims,
                                            const int8_t *filter_data,
                                            const cmsis_nn_dims *bias_dims,
                                            const int64_t *bias_data,
                                            const cmsis_nn_dims *output_dims,
                                            int16_t *output_data);

/**
 * @brief Get size of additional buffer required by arm_fully_connected_s16().
 * @param[in]      filter_dims             dimension of filter
 * @return         The function returns    required buffer size in bytes
 *
 */
int32_t arm_fully_connected_s16_get_buffer_size(const cmsis_nn_dims *filter_dims);

/**
 * @brief Get size of additional buffer required by arm_fully_connected_s16() for processors with DSP extension.
 *        Refer to arm_fully_connected_s16_get_buffer_size() for function argument details.
 *
 * @note       Intended for compilation on Host. If compiling for an Arm target, use
 *             arm_fully_connected_s16_get_buffer_size().
 *
 */
int32_t arm_fully_connected_s16_get_buffer_size_dsp(const cmsis_nn_dims *filter_dims);

/**
 * @brief Get size of additional buffer required by arm_fully_connected_s16() for Arm(R) Helium Architecture case.
 *        Refer to arm_fully_connected_s16_get_buffer_size() for function argument details.
 *
 * @note       Intended for compilation on Host. If compiling for an Arm target, use
 *             arm_fully_connected_s16_get_buffer_size().
 *
 */
int32_t arm_fully_connected_s16_get_buffer_size_mve(const cmsis_nn_dims *filter_dims);

/**
 * @defgroup groupElementwise Elementwise Functions
 *
 * Elementwise add and multiplication functions.
 *
 */

/**
 * @brief s8 elementwise add of two vectors
 * @param[in]       input_1_vect        pointer to input vector 1
 * @param[in]       input_2_vect        pointer to input vector 2
 * @param[in]       input_1_offset      offset for input 1. Range: -127 to 128
 * @param[in]       input_1_mult        multiplier for input 1
 * @param[in]       input_1_shift       shift for input 1
 * @param[in]       input_2_offset      offset for input 2. Range: -127 to 128
 * @param[in]       input_2_mult        multiplier for input 2
 * @param[in]       input_2_shift       shift for input 2
 * @param[in]       left_shift          input left shift
 * @param[in,out]   output              pointer to output vector
 * @param[in]       out_offset          output offset.  Range: -128 to 127
 * @param[in]       out_mult            output multiplier
 * @param[in]       out_shift           output shift
 * @param[in]       out_activation_min  minimum value to clamp output to. Min: -128
 * @param[in]       out_activation_max  maximum value to clamp output to. Max: 127
 * @param[in]       block_size          number of samples
 * @return          The function returns    ARM_CMSIS_NN_SUCCESS
 */
arm_cmsis_nn_status arm_elementwise_add_s8(const int8_t *input_1_vect,
                                           const int8_t *input_2_vect,
                                           const int32_t input_1_offset,
                                           const int32_t input_1_mult,
                                           const int32_t input_1_shift,
                                           const int32_t input_2_offset,
                                           const int32_t input_2_mult,
                                           const int32_t input_2_shift,
                                           const int32_t left_shift,
                                           int8_t *output,
                                           const int32_t out_offset,
                                           const int32_t out_mult,
                                           const int32_t out_shift,
                                           const int32_t out_activation_min,
                                           const int32_t out_activation_max,
                                           const int32_t block_size);

/**
 * @brief s16 elementwise add of two vectors
 * @param[in]       input_1_vect        pointer to input vector 1
 * @param[in]       input_2_vect        pointer to input vector 2
 * @param[in]       input_1_offset      offset for input 1. Not used.
 * @param[in]       input_1_mult        multiplier for input 1
 * @param[in]       input_1_shift       shift for input 1
 * @param[in]       input_2_offset      offset for input 2. Not used.
 * @param[in]       input_2_mult        multiplier for input 2
 * @param[in]       input_2_shift       shift for input 2
 * @param[in]       left_shift          input left shift
 * @param[in,out]   output              pointer to output vector
 * @param[in]       out_offset          output offset. Not used.
 * @param[in]       out_mult            output multiplier
 * @param[in]       out_shift           output shift
 * @param[in]       out_activation_min  minimum value to clamp output to. Min: -32768
 * @param[in]       out_activation_max  maximum value to clamp output to. Max: 32767
 * @param[in]       block_size          number of samples
 * @return          The function returns  ARM_CMSIS_NN_SUCCESS
 */
arm_cmsis_nn_status arm_elementwise_add_s16(const int16_t *input_1_vect,
                                            const int16_t *input_2_vect,
                                            const int32_t input_1_offset,
                                            const int32_t input_1_mult,
                                            const int32_t input_1_shift,
                                            const int32_t input_2_offset,
                                            const int32_t input_2_mult,
                                            const int32_t input_2_shift,
                                            const int32_t left_shift,
                                            int16_t *output,
                                            const int32_t out_offset,
                                            const int32_t out_mult,
                                            const int32_t out_shift,
                                            const int32_t out_activation_min,
                                            const int32_t out_activation_max,
                                            const int32_t block_size);

/**
 * @brief s8 elementwise multiplication
 * @param[in]       input_1_vect        pointer to input vector 1
 * @param[in]       input_2_vect        pointer to input vector 2
 * @param[in]       input_1_offset      offset for input 1. Range: -127 to 128
 * @param[in]       input_2_offset      offset for input 2. Range: -127 to 128
 * @param[in,out]   output              pointer to output vector
 * @param[in]       out_offset          output offset. Range: -128 to 127
 * @param[in]       out_mult            output multiplier
 * @param[in]       out_shift           output shift
 * @param[in]       out_activation_min  minimum value to clamp output to. Min: -128
 * @param[in]       out_activation_max  maximum value to clamp output to. Max: 127
 * @param[in]       block_size          number of samples
 * @return          The function returns ARM_CMSIS_NN_SUCCESS
 *
 * @details   Supported framework: TensorFlow Lite micro
 */
arm_cmsis_nn_status arm_elementwise_mul_s8(const int8_t *input_1_vect,
                                           const int8_t *input_2_vect,
                                           const int32_t input_1_offset,
                                           const int32_t input_2_offset,
                                           int8_t *output,
                                           const int32_t out_offset,
                                           const int32_t out_mult,
                                           const int32_t out_shift,
                                           const int32_t out_activation_min,
                                           const int32_t out_activation_max,
                                           const int32_t block_size);

/**
 * @brief s16 elementwise multiplication
 * @param[in]       input_1_vect        pointer to input vector 1
 * @param[in]       input_2_vect        pointer to input vector 2
 * @param[in]       input_1_offset      offset for input 1. Not used.
 * @param[in]       input_2_offset      offset for input 2. Not used.
 * @param[in,out]   output              pointer to output vector
 * @param[in]       out_offset          output offset. Not used.
 * @param[in]       out_mult            output multiplier
 * @param[in]       out_shift           output shift
 * @param[in]       out_activation_min  minimum value to clamp output to. Min: -32768
 * @param[in]       out_activation_max  maximum value to clamp output to. Max: 32767
 * @param[in]       block_size          number of samples
 * @return          The function returns ARM_CMSIS_NN_SUCCESS
 *
 * @details   Supported framework: TensorFlow Lite micro
 */
arm_cmsis_nn_status arm_elementwise_mul_s16(const int16_t *input_1_vect,
                                            const int16_t *input_2_vect,
                                            const int32_t input_1_offset,
                                            const int32_t input_2_offset,
                                            int16_t *output,
                                            const int32_t out_offset,
                                            const int32_t out_mult,
                                            const int32_t out_shift,
                                            const int32_t out_activation_min,
                                            const int32_t out_activation_max,
                                            const int32_t block_size);

/**
 * @defgroup Acti Activation Functions
 *
 * Perform activation layers, including ReLU (Rectified Linear Unit),
 * sigmoid and tanh
 *
 */

/**
 * @brief Q7 RELU function
 * @param[in,out]   data        pointer to input
 * @param[in]       size        number of elements
 */
void arm_relu_q7(int8_t *data, uint16_t size);

/**
 * @brief s8 ReLU6 function
 * @param[in,out]   data        pointer to input
 * @param[in]       size        number of elements
 */
void arm_relu6_s8(int8_t *data, uint16_t size);

/**
 * @brief Q15 RELU function
 * @param[in,out]   data        pointer to input
 * @param[in]       size        number of elements
 */
void arm_relu_q15(int16_t *data, uint16_t size);

/**
 * @brief s16 neural network activation function using direct table look-up
 * @param[in]       input       pointer to input data
 * @param[out]      output      pointer to output
 * @param[in]       size        number of elements
 * @param[in]       left_shift  bit-width of the integer part, assumed to be smaller than 3.
 * @param[in]       type        type of activation functions
 * @return                      The function returns <code>ARM_CMSIS_NN_SUCCESS</code>

 *
 * @details Supported framework: TensorFlow Lite for Microcontrollers.
 * This activation function must be bit precise congruent with the corresponding TFLM tanh and sigmoid activation
 * functions
 */
arm_cmsis_nn_status arm_nn_activation_s16(const int16_t *input,
                                          int16_t *output,
                                          const int32_t size,
                                          const int32_t left_shift,
                                          const arm_nn_activation_type type);

/**
 * @defgroup Pooling Pooling Functions
 *
 * Perform max and average pooling operations
 *
 */

/**
 * @brief s8 average pooling function.
 *
 * @param[in, out] ctx          Function context (e.g. temporary buffer). Check the function
 *                              definition file to see if an additional buffer is required.
 *                              Optional function {API}_get_buffer_size() provides the buffer
 *                              size if an additional buffer is required.
 *                              The caller is expected to clear the buffer, if applicable, for security reasons.
 * @param[in]      pool_params  Pooling parameters
 * @param[in]      input_dims   Input (activation) tensor dimensions. Format: [H, W, C_IN]
 * @param[in]      input_data   Input (activation) data pointer. Data type: int8
 * @param[in]      filter_dims  Filter tensor dimensions. Format: [H, W]
 *                              Argument N and C are not used.
 * @param[in]      output_dims  Output tensor dimensions. Format: [H, W, C_OUT]
 *                              Argument N is not used.
 *                              C_OUT equals C_IN.
 * @param[in, out] output_data Output data pointer. Data type: int8
 *
 * @return     The function returns either
 *                  <code>ARM_CMSIS_NN_ARG_ERROR</code> if argument constraints fail. or,
 *                  <code>ARM_CMSIS_NN_SUCCESS</code> on successful completion.
 *
 * @details
 *    - Supported Framework: TensorFlow Lite
 *
 */
arm_cmsis_nn_status arm_avgpool_s8(const cmsis_nn_context *ctx,
                                   const cmsis_nn_pool_params *pool_params,
                                   const cmsis_nn_dims *input_dims,
                                   const int8_t *input_data,
                                   const cmsis_nn_dims *filter_dims,
                                   const cmsis_nn_dims *output_dims,
                                   int8_t *output_data);

/**
 * @brief Get the required buffer size for S8 average pooling function
 * @param[in]       dim_dst_width         output tensor dimension
 * @param[in]       ch_src                number of input tensor channels
 * @return          The function returns required buffer size in bytes
 *
 */
int32_t arm_avgpool_s8_get_buffer_size(const int dim_dst_width, const int ch_src);

/**
 * @brief Get the required buffer size for S8 average pooling function for processors with DSP extension.
 *        Refer to arm_avgpool_s8_get_buffer_size() for function argument details.
 *
 * @note       Intended for compilation on Host. If compiling for an Arm target, use
 *             arm_avgpool_s8_get_buffer_size().
 *
 */
int32_t arm_avgpool_s8_get_buffer_size_dsp(const int dim_dst_width, const int ch_src);

/**
 * @brief Get the required buffer size for S8 average pooling function for Arm(R) Helium Architecture case.
 *        Refer to arm_avgpool_s8_get_buffer_size() for function argument details.
 *
 * @note       Intended for compilation on Host. If compiling for an Arm target, use
 *             arm_avgpool_s8_get_buffer_size().
 *
 */
int32_t arm_avgpool_s8_get_buffer_size_mve(const int dim_dst_width, const int ch_src);

/**
 * @brief s16 average pooling function.
 *
 * @param[in, out] ctx          Function context (e.g. temporary buffer). Check the function
 *                              definition file to see if an additional buffer is required.
 *                              Optional function {API}_get_buffer_size() provides the buffer
 *                              size if an additional buffer is required.
 *                              The caller is expected to clear the buffer, if applicable, for security reasons.
 * @param[in]      pool_params  Pooling parameters
 * @param[in]      input_dims   Input (activation) tensor dimensions. Format: [H, W, C_IN]
 * @param[in]      input_data   Input (activation) data pointer. Data type: int16
 * @param[in]      filter_dims  Filter tensor dimensions. Format: [H, W]
 *                              Argument N and C are not used.
 * @param[in]      output_dims  Output tensor dimensions. Format: [H, W, C_OUT]
 *                              Argument N is not used.
 *                              C_OUT equals C_IN.
 * @param[in, out] output_data  Output data pointer. Data type: int16
 *
 * @return                        The function returns
 *                                    <code>ARM_CMSIS_NN_SUCCESS</code> - Successful operation
 *                                    <code>ARM_CMSIS_NN_ARG_ERROR</code> - In case of invalid arguments
 *
 * @details
 *    - Supported Framework: TensorFlow Lite
 *
 */
arm_cmsis_nn_status arm_avgpool_s16(const cmsis_nn_context *ctx,
                                    const cmsis_nn_pool_params *pool_params,
                                    const cmsis_nn_dims *input_dims,
                                    const int16_t *input_data,
                                    const cmsis_nn_dims *filter_dims,
                                    const cmsis_nn_dims *output_dims,
                                    int16_t *output_data);

/**
 * @brief Get the required buffer size for S16 average pooling function
 * @param[in]       dim_dst_width         output tensor dimension
 * @param[in]       ch_src                number of input tensor channels
 * @return          The function returns required buffer size in bytes
 *
 */
int32_t arm_avgpool_s16_get_buffer_size(const int dim_dst_width, const int ch_src);

/**
 * @brief Get the required buffer size for S16 average pooling function for processors with DSP extension.
 *        Refer to arm_avgpool_s16_get_buffer_size() for function argument details.
 *
 * @note       Intended for compilation on Host. If compiling for an Arm target, use
 *             arm_avgpool_s16_get_buffer_size().
 *
 */
int32_t arm_avgpool_s16_get_buffer_size_dsp(const int dim_dst_width, const int ch_src);

/**
 * @brief Get the required buffer size for S16 average pooling function for Arm(R) Helium Architecture case.
 *        Refer to arm_avgpool_s16_get_buffer_size() for function argument details.
 *
 * @note       Intended for compilation on Host. If compiling for an Arm target, use
 *             arm_avgpool_s16_get_buffer_size().
 *
 */
int32_t arm_avgpool_s16_get_buffer_size_mve(const int dim_dst_width, const int ch_src);

/**
 * @brief s8 max pooling function.
 *
 * @param[in, out] ctx          Function context (e.g. temporary buffer). Check the function
 *                              definition file to see if an additional buffer is required.
 *                              Optional function {API}_get_buffer_size() provides the buffer
 *                              size if an additional buffer is required.
 *                              The caller is expected to clear the buffer, if applicable, for security reasons.
 * @param[in]      pool_params  Pooling parameters
 * @param[in]      input_dims   Input (activation) tensor dimensions. Format: [H, W, C_IN]
 * @param[in]      input_data   Input (activation) data pointer. The input tensor must not
 *                              overlap with the output tensor. Data type: int8
 * @param[in]      filter_dims  Filter tensor dimensions. Format: [H, W]
 *                              Argument N and C are not used.
 * @param[in]      output_dims  Output tensor dimensions. Format: [H, W, C_OUT]
 *                              Argument N is not used.
 *                              C_OUT equals C_IN.
 * @param[in, out] output_data    Output data pointer. Data type: int8
 *
 * @return     The function returns either
 *                  <code>ARM_CMSIS_NN_ARG_ERROR</code> if argument constraints fail. or,
 *                  <code>ARM_CMSIS_NN_SUCCESS</code> on successful completion.
 *
 * @details
 *    - Supported Framework: TensorFlow Lite
 *
 */
arm_cmsis_nn_status arm_max_pool_s8(const cmsis_nn_context *ctx,
                                    const cmsis_nn_pool_params *pool_params,
                                    const cmsis_nn_dims *input_dims,
                                    const int8_t *input_data,
                                    const cmsis_nn_dims *filter_dims,
                                    const cmsis_nn_dims *output_dims,
                                    int8_t *output_data);

/**
 * @brief s16 max pooling function.
 *
 * @param[in, out] ctx          Function context (e.g. temporary buffer). Check the function
 *                              definition file to see if an additional buffer is required.
 *                              Optional function {API}_get_buffer_size() provides the buffer
 *                              size if an additional buffer is required.
 *                              The caller is expected to clear the buffer, if applicable, for security reasons.
 * @param[in]      pool_params  Pooling parameters
 * @param[in]      input_dims   Input (activation) tensor dimensions. Format: [H, W, C_IN]
 * @param[in]      src          Input (activation) data pointer. The input tensor must not
 *                              overlap with the output tensor. Data type: int16
 * @param[in]      filter_dims  Filter tensor dimensions. Format: [H, W]
 *                              Argument N and C are not used.
 * @param[in]      output_dims  Output tensor dimensions. Format: [H, W, C_OUT]
 *                              Argument N is not used.
 *                              C_OUT equals C_IN.
 * @param[in, out] dst          Output data pointer. Data type: int16
 *
 * @return     The function returns either
 *                  <code>ARM_CMSIS_NN_ARG_ERROR</code> if argument constraints fail. or,
 *                  <code>ARM_CMSIS_NN_SUCCESS</code> on successful completion.
 *
 * @details
 *    - Supported Framework: TensorFlow Lite
 *
 */
arm_cmsis_nn_status arm_max_pool_s16(const cmsis_nn_context *ctx,
                                     const cmsis_nn_pool_params *pool_params,
                                     const cmsis_nn_dims *input_dims,
                                     const int16_t *src,
                                     const cmsis_nn_dims *filter_dims,
                                     const cmsis_nn_dims *output_dims,
                                     int16_t *dst);

/**
 * @defgroup Softmax Softmax Functions
 *
 *
 */

/**
 * @brief S8 softmax function
 * @param[in]  input     Pointer to the input tensor
 * @param[in]  num_rows  Number of rows in the input tensor
 * @param[in]  row_size  Number of elements in each input row
 * @param[in]  mult      Input quantization multiplier
 * @param[in]  shift     Input quantization shift within the range [0, 31]
 * @param[in]  diff_min  Minimum difference with max in row. Used to check if
 *                       the quantized exponential operation can be performed
 * @param[out] output    Pointer to the output tensor
 *
 * @note Supported framework: TensorFlow Lite micro (bit-accurate)
 *
 */
void arm_softmax_s8(const int8_t *input,
                    const int32_t num_rows,
                    const int32_t row_size,
                    const int32_t mult,
                    const int32_t shift,
                    const int32_t diff_min,
                    int8_t *output);

/**
 * @brief S8 to s16 softmax function
 * @param[in]  input     Pointer to the input tensor
 * @param[in]  num_rows  Number of rows in the input tensor
 * @param[in]  row_size  Number of elements in each input row
 * @param[in]  mult      Input quantization multiplier
 * @param[in]  shift     Input quantization shift within the range [0, 31]
 * @param[in]  diff_min  Minimum difference with max in row. Used to check if
 *                       the quantized exponential operation can be performed
 * @param[out] output    Pointer to the output tensor
 *
 * @note Supported framework: TensorFlow Lite micro (bit-accurate)
 *
 */
void arm_softmax_s8_s16(const int8_t *input,
                        const int32_t num_rows,
                        const int32_t row_size,
                        const int32_t mult,
                        const int32_t shift,
                        const int32_t diff_min,
                        int16_t *output);

/**
 * @brief S16 softmax function
 * @param[in]  input           Pointer to the input tensor
 * @param[in]  num_rows        Number of rows in the input tensor
 * @param[in]  row_size        Number of elements in each input row
 * @param[in]  mult            Input quantization multiplier
 * @param[in]  shift           Input quantization shift within the range [0, 31]
 * @param[in]  softmax_params  Softmax s16 layer parameters with two pointers to LUTs speficied below.
 *                             For indexing the high 9 bits are used and 7 remaining for interpolation.
 *                             That means 512 entries for the 9-bit indexing and 1 extra for interpolation, i.e. 513
 *                             values for each LUT.
 *                             - Lookup table for exp(x), where x uniform distributed between [-10.0 , 0.0]
 *                             - Lookup table for 1 / (1 + x), where x uniform distributed between [0.0 , 1.0]
 * @param[out] output          Pointer to the output tensor
 * @return                        The function returns
 *                                    <code>ARM_CMSIS_NN_ARG_ERROR</code> Argument error check failed
 *                                    <code>ARM_CMSIS_NN_SUCCESS</code> - Successful operation
 *
 * @note Supported framework: TensorFlow Lite micro (bit-accurate)
 *
 */
arm_cmsis_nn_status arm_softmax_s16(const int16_t *input,
                                    const int32_t num_rows,
                                    const int32_t row_size,
                                    const int32_t mult,
                                    const int32_t shift,
                                    const cmsis_nn_softmax_lut_s16 *softmax_params,
                                    int16_t *output);

/**
 * @brief U8 softmax function
 * @param[in]  input     Pointer to the input tensor
 * @param[in]  num_rows  Number of rows in the input tensor
 * @param[in]  row_size  Number of elements in each input row
 * @param[in]  mult      Input quantization multiplier
 * @param[in]  shift     Input quantization shift within the range [0, 31]
 * @param[in]  diff_min  Minimum difference with max in row. Used to check if
 *                       the quantized exponential operation can be performed
 * @param[out] output    Pointer to the output tensor
 *
 * @note Supported framework: TensorFlow Lite micro (bit-accurate)
 *
 */

void arm_softmax_u8(const uint8_t *input,
                    const int32_t num_rows,
                    const int32_t row_size,
                    const int32_t mult,
                    const int32_t shift,
                    const int32_t diff_min,
                    uint8_t *output);

/**
 * @defgroup Reshape Reshape Functions
 *
 */

/**
 * @brief Reshape a s8 vector into another with different shape
 * @param[in]  input      points to the s8 input vector
 * @param[out] output     points to the s8 output vector
 * @param[in]  total_size total size of the input and output vectors in bytes
 *
 * @note The output is expected to be in a memory area that does not overlap with the input's
 *
 */
void arm_reshape_s8(const int8_t *input, int8_t *output, const uint32_t total_size);

/**
 * @defgroup Transpose Transpose Functions
 *
 */

/**
 * @brief Basic transpose function
 *
 * @param[in]       input_data            Input (activation) data pointer. Data type: int8
 * @param[out]      output_data           Output data pointer. Data type: int8
 * @param[in]       input_dims            Input (activation) tensor dimensions. Format: [N, H, W, C_IN]
 * @param[in]       output_dims           Output tensor dimensions. Format may be arbitrary relative to input format.
 *                                        The output dimension will depend on the permutation dimensions.
 *                                        In other words the out dimensions are the result of applying the permutation
 *                                        to the input dimensions.
 * @param[in]       transpose_params      Transpose parameters. Contains permutation dimensions.
 *
 * @return          The function returns either
 *                      <code>ARM_CMSIS_NN_ARG_ERROR</code> if argument constraints fail. or,
 *                      <code>ARM_CMSIS_NN_SUCCESS</code> on successful completion.
 *
 */
arm_cmsis_nn_status arm_transpose_s8(const int8_t *input_data,
                                     int8_t *const output_data,
                                     const cmsis_nn_dims *const input_dims,
                                     const cmsis_nn_dims *const output_dims,
                                     const cmsis_nn_transpose_params *const transpose_params);

/**
 * @defgroup Concatenation Concatenation Functions
 *
 */

/**
 * @brief int8/uint8 concatenation function to be used for concatenating N-tensors along the X axis
 *        This function should be called for each input tensor to concatenate. The argument offset_x
 *        will be used to store the input tensor in the correct position in the output tensor
 *
 *        i.e.    offset_x = 0
 *                for(i = 0 i < num_input_tensors; ++i)
 *                {
 *                    arm_concatenation_s8_x(&input[i], ..., &output, ..., ..., offset_x)
 *                    offset_x += input_x[i]
 *                }
 *
 *        This function assumes that the output tensor has:
 *        -# The same height of the input tensor
 *        -# The same number of channels of the input tensor
 *        -# The same batch size of the input tensor
 *
 *        Unless specified otherwise, arguments are mandatory.
 *
 * @note This function, data layout independent, can be used to concatenate either int8 or uint8 tensors because it
 *      does not involve any arithmetic operation
 *
 * @param[in]  input    Pointer to input tensor. Input tensor must not overlap with the output tensor.
 * @param[in]  input_x  Width of input tensor
 * @param[in]  input_y  Height of input tensor
 * @param[in]  input_z  Channels in input tensor
 * @param[in]  input_w  Batch size in input tensor
 * @param[out] output   Pointer to output tensor. Expected to be at least
 *                          (input_x * input_y * input_z * input_w) + offset_x
 *                      bytes.
 * @param[in]  output_x Width of output tensor
 * @param[in]  offset_x The offset (in number of elements) on the X axis to start concatenating the input tensor
 *                      It is user responsibility to provide the correct value
 *
 * <b> Input constraints</b>
 * offset_x is less than output_x
 *
 */
void arm_concatenation_s8_x(const int8_t *input,
                            const uint16_t input_x,
                            const uint16_t input_y,
                            const uint16_t input_z,
                            const uint16_t input_w,
                            int8_t *output,
                            const uint16_t output_x,
                            const uint32_t offset_x);

/**
 * @brief int8/uint8 concatenation function to be used for concatenating N-tensors along the Y axis
 *        This function should be called for each input tensor to concatenate. The argument offset_y
 *        will be used to store the input tensor in the correct position in the output tensor
 *
 *        i.e.    offset_y = 0
 *                for(i = 0 i < num_input_tensors; ++i)
 *                {
 *                    arm_concatenation_s8_y(&input[i], ..., &output, ..., ..., offset_y)
 *                    offset_y += input_y[i]
 *                }
 *
 *        This function assumes that the output tensor has:
 *        -# The same width of the input tensor
 *        -# The same number of channels of the input tensor
 *        -# The same batch size of the input tensor
 *
 *        Unless specified otherwise, arguments are mandatory.
 *
 * @note This function, data layout independent, can be used to concatenate either int8 or uint8 tensors because it
 *       does not involve any arithmetic operation
 *
 * @param[in]  input    Pointer to input tensor. Input tensor must not overlap with the output tensor.
 * @param[in]  input_x  Width of input tensor
 * @param[in]  input_y  Height of input tensor
 * @param[in]  input_z  Channels in input tensor
 * @param[in]  input_w  Batch size in input tensor
 * @param[out] output   Pointer to output tensor. Expected to be at least
 *                          (input_z * input_w * input_x * input_y) + offset_y
 *                      bytes.
 * @param[in]  output_y Height of output tensor
 * @param[in]  offset_y The offset on the Y axis to start concatenating the input tensor
 *                      It is user responsibility to provide the correct value
 *
 * <b> Input constraints</b>
 * offset_y is less than output_y
 *
 */
void arm_concatenation_s8_y(const int8_t *input,
                            const uint16_t input_x,
                            const uint16_t input_y,
                            const uint16_t input_z,
                            const uint16_t input_w,
                            int8_t *output,
                            const uint16_t output_y,
                            const uint32_t offset_y);

/**
 * @brief int8/uint8 concatenation function to be used for concatenating N-tensors along the Z axis
 *        This function should be called for each input tensor to concatenate. The argument offset_z
 *        will be used to store the input tensor in the correct position in the output tensor
 *
 *        i.e.    offset_z = 0
 *                for(i = 0 i < num_input_tensors; ++i)
 *                {
 *                    arm_concatenation_s8_z(&input[i], ..., &output, ..., ..., offset_z)
 *                    offset_z += input_z[i]
 *                }
 *
 *        This function assumes that the output tensor has:
 *        -# The same width of the input tensor
 *        -# The same height of the input tensor
 *        -# The same batch size of the input tensor
 *
 *        Unless specified otherwise, arguments are mandatory.
 *
 * @note This function, data layout independent, can be used to concatenate either int8 or uint8 tensors because it
 *       does not involve any arithmetic operation
 *
 * @param[in]  input    Pointer to input tensor. Input tensor must not overlap with output tensor.
 * @param[in]  input_x  Width of input tensor
 * @param[in]  input_y  Height of input tensor
 * @param[in]  input_z  Channels in input tensor
 * @param[in]  input_w  Batch size in input tensor
 * @param[out] output   Pointer to output tensor. Expected to be at least
 *                          (input_x * input_y * input_z * input_w) + offset_z
 *                      bytes.
 * @param[in]  output_z Channels in output tensor
 * @param[in]  offset_z The offset on the Z axis to start concatenating the input tensor
 *                      It is user responsibility to provide the correct value
 *
 * <b> Input constraints</b>
 * offset_z is less than output_z
 *
 */
void arm_concatenation_s8_z(const int8_t *input,
                            const uint16_t input_x,
                            const uint16_t input_y,
                            const uint16_t input_z,
                            const uint16_t input_w,
                            int8_t *output,
                            const uint16_t output_z,
                            const uint32_t offset_z);

/**
 * @brief int8/uint8 concatenation function to be used for concatenating N-tensors along the W axis (Batch size)
 *        This function should be called for each input tensor to concatenate. The argument offset_w
 *        will be used to store the input tensor in the correct position in the output tensor
 *
 *        i.e.    offset_w = 0
 *                for(i = 0 i < num_input_tensors; ++i)
 *                {
 *                    arm_concatenation_s8_w(&input[i], ..., &output, ..., ..., offset_w)
 *                    offset_w += input_w[i]
 *                }
 *
 *        This function assumes that the output tensor has:
 *        -# The same width of the input tensor
 *        -# The same height of the input tensor
 *        -# The same number o channels of the input tensor
 *
 *        Unless specified otherwise, arguments are mandatory.
 *
 * @note This function, data layout independent, can be used to concatenate either int8 or uint8 tensors because it
 *       does not involve any arithmetic operation
 *
 * @param[in]  input    Pointer to input tensor
 * @param[in]  input_x  Width of input tensor
 * @param[in]  input_y  Height of input tensor
 * @param[in]  input_z  Channels in input tensor
 * @param[in]  input_w  Batch size in input tensor
 * @param[out] output   Pointer to output tensor. Expected to be at least
 *                          input_x * input_y * input_z * input_w
 *                      bytes.
 * @param[in]  offset_w The offset on the W axis to start concatenating the input tensor
 *                      It is user responsibility to provide the correct value
 *
 */
void arm_concatenation_s8_w(const int8_t *input,
                            const uint16_t input_x,
                            const uint16_t input_y,
                            const uint16_t input_z,
                            const uint16_t input_w,
                            int8_t *output,
                            const uint32_t offset_w);
/**
 * @defgroup SVDF SVDF Functions
 *
 */

/**
 * @brief s8 SVDF function with 8 bit state tensor and 8 bit time weights
 *
 * @param[in, out] ctx                Function context (e.g. temporary buffer). Check the function
 *                                    definition file to see if an additional buffer is required.
 *                                    Optional function arm_fully_connected_s8_get_buffer_size() provides the buffer
 *                                    size if an additional buffer is required.
 *                                    The caller is expected to clear the buffer, if applicable, for security reasons.
 * @param[in]   input_ctx             Temporary scratch buffer
 *                                    The caller is expected to clear the buffer, if applicable, for security reasons.
 * @param[in]   output_ctx            Temporary output scratch buffer
 *                                    The caller is expected to clear the buffer, if applicable, for security reasons.
 * @param[in]   svdf_params           SVDF Parameters
 *                                    Range of svdf_params->input_offset  : [-128, 127]
 *                                    Range of svdf_params->output_offset  : [-128, 127]
 * @param[in]   input_quant_params    Input quantization parameters
 * @param[in]   output_quant_params   Output quantization parameters
 * @param[in]   input_dims            Input tensor dimensions
 * @param[in]   input_data            Pointer to input tensor
 * @param[in]   state_dims            State tensor dimensions
 * @param[in]   state_data            Pointer to state tensor
 * @param[in]   weights_feature_dims  Weights (feature) tensor dimensions
 * @param[in]   weights_feature_data  Pointer to the weights (feature) tensor
 * @param[in]   weights_time_dims     Weights (time) tensor dimensions
 * @param[in]   weights_time_data     Pointer to the weights (time) tensor
 * @param[in]   bias_dims             Bias tensor dimensions
 * @param[in]   bias_data             Pointer to bias tensor
 * @param[in]   output_dims           Output tensor dimensions
 * @param[out]  output_data           Pointer to the output tensor
 *
 * @return     The function returns either
 *                  <code>ARM_CMSIS_NN_ARG_ERROR</code> if argument constraints fail. or,
 *                  <code>ARM_CMSIS_NN_SUCCESS</code> on successful completion.
 *
 * @details
 *    1. Supported framework: TensorFlow Lite micro
 */
arm_cmsis_nn_status arm_svdf_s8(const cmsis_nn_context *ctx,
                                const cmsis_nn_context *input_ctx,
                                const cmsis_nn_context *output_ctx,
                                const cmsis_nn_svdf_params *svdf_params,
                                const cmsis_nn_per_tensor_quant_params *input_quant_params,
                                const cmsis_nn_per_tensor_quant_params *output_quant_params,
                                const cmsis_nn_dims *input_dims,
                                const int8_t *input_data,
                                const cmsis_nn_dims *state_dims,
                                int8_t *state_data,
                                const cmsis_nn_dims *weights_feature_dims,
                                const int8_t *weights_feature_data,
                                const cmsis_nn_dims *weights_time_dims,
                                const int8_t *weights_time_data,
                                const cmsis_nn_dims *bias_dims,
                                const int32_t *bias_data,
                                const cmsis_nn_dims *output_dims,
                                int8_t *output_data);

/**
 * @brief s8 SVDF function with 16 bit state tensor and 16 bit time weights
 *
 * @param[in]   input_ctx             Temporary scratch buffer
 *                                    The caller is expected to clear the buffer, if applicable, for security reasons.
 * @param[in]   output_ctx            Temporary output scratch buffer
 *                                    The caller is expected to clear the buffer, if applicable, for security reasons.
 * @param[in]   svdf_params           SVDF Parameters
 *                                    Range of svdf_params->input_offset  : [-128, 127]
 *                                    Range of svdf_params->output_offset  : [-128, 127]
 * @param[in]   input_quant_params    Input quantization parameters
 * @param[in]   output_quant_params   Output quantization parameters
 * @param[in]   input_dims            Input tensor dimensions
 * @param[in]   input_data            Pointer to input tensor
 * @param[in]   state_dims            State tensor dimensions
 * @param[in]   state_data            Pointer to state tensor
 * @param[in]   weights_feature_dims  Weights (feature) tensor dimensions
 * @param[in]   weights_feature_data  Pointer to the weights (feature) tensor
 * @param[in]   weights_time_dims     Weights (time) tensor dimensions
 * @param[in]   weights_time_data     Pointer to the weights (time) tensor
 * @param[in]   bias_dims             Bias tensor dimensions
 * @param[in]   bias_data             Pointer to bias tensor
 * @param[in]   output_dims           Output tensor dimensions
 * @param[out]  output_data           Pointer to the output tensor
 *
 * @return     The function returns <code>ARM_CMSIS_NN_SUCCESS</code>
 *
 * @details
 *    1. Supported framework: TensorFlow Lite micro
 */
arm_cmsis_nn_status arm_svdf_state_s16_s8(const cmsis_nn_context *input_ctx,
                                          const cmsis_nn_context *output_ctx,
                                          const cmsis_nn_svdf_params *svdf_params,
                                          const cmsis_nn_per_tensor_quant_params *input_quant_params,
                                          const cmsis_nn_per_tensor_quant_params *output_quant_params,
                                          const cmsis_nn_dims *input_dims,
                                          const int8_t *input_data,
                                          const cmsis_nn_dims *state_dims,
                                          int16_t *state_data,
                                          const cmsis_nn_dims *weights_feature_dims,
                                          const int8_t *weights_feature_data,
                                          const cmsis_nn_dims *weights_time_dims,
                                          const int16_t *weights_time_data,
                                          const cmsis_nn_dims *bias_dims,
                                          const int32_t *bias_data,
                                          const cmsis_nn_dims *output_dims,
                                          int8_t *output_data);

/**
 * @brief Get size of additional buffer required by arm_svdf_s8().
 * @param[in]      filter_dims             dimension of filter
 * @return         The function returns    required buffer size in bytes
 *
 */
int32_t arm_svdf_s8_get_buffer_size(const cmsis_nn_dims *filter_dims);

/**
 * @brief Get size of additional buffer required by arm_svdf_s8() for processors with DSP extension.
 *        Refer to arm_svdf_s8_get_buffer_size() for function argument details.
 *
 * @note       Intended for compilation on Host. If compiling for an Arm target, use
 *             arm_svdf_s8_get_buffer_size().
 *
 */
int32_t arm_svdf_s8_get_buffer_size_dsp(const cmsis_nn_dims *filter_dims);

/**
 * @brief Get size of additional buffer required by arm_svdf_s8() for Arm(R) Helium Architecture case.
 *        Refer to arm_svdf_s8_get_buffer_size() for function argument details.
 *
 * @note       Intended for compilation on Host. If compiling for an Arm target, use
 *             arm_svdf_s8_get_buffer_size().
 *
 */
int32_t arm_svdf_s8_get_buffer_size_mve(const cmsis_nn_dims *filter_dims);

/**
 * @defgroup LSTM LSTM Layer Functions
 *
 */

/**
 * @brief LSTM unidirectional function with 8 bit input and output and 16 bit gate output, 32 bit bias.
 *
 * @param[in]   input                      Pointer to input data
 * @param[out]  output                     Pointer to output data
 * @param[in]   params                     Struct containing all information about the lstm operator, see arm_nn_types.
 * @param[in]   buffers                    Struct containing pointers to all temporary scratch buffers needed for the
 * lstm operator, see arm_nn_types.
 *
 *
 * @return     The function returns <code>ARM_CMSIS_NN_SUCCESS</code>
 *
 * @details
 *    1. Supported framework: TensorFlow Lite Micro
 *
 */
arm_cmsis_nn_status arm_lstm_unidirectional_s8(const int8_t *input,
                                               int8_t *output,
                                               const cmsis_nn_lstm_params *params,
                                               cmsis_nn_lstm_context *buffers);

/**
 * @brief LSTM unidirectional function with 16 bit input and output and 16 bit gate output, 64 bit bias.
 *
 * @param[in]   input                      Pointer to input data
 * @param[out]  output                     Pointer to output data
 * @param[in]   params                     Struct containing all information about the lstm operator, see arm_nn_types.
 * @param[in]   buffers                    Struct containing pointers to all temporary scratch buffers needed for the
 * lstm operator, see arm_nn_types.
 *
 *
 * @return     The function returns <code>ARM_CMSIS_NN_SUCCESS</code>
 *
 * @details
 *    1. Supported framework: TensorFlow Lite Micro
 *
 */
arm_cmsis_nn_status arm_lstm_unidirectional_s16(const int16_t *input,
                                                int16_t *output,
                                                const cmsis_nn_lstm_params *params,
                                                cmsis_nn_lstm_context *buffers);

/**
 * @brief Batch matmul function with 8 bit input and output.
 *
 * @param[in]   ctx                   Temporary scratch buffer
 *                                    The caller is expected to clear the buffer, if applicable, for security reasons.
 *                                    Optional function arm_fully_connected_s8_get_buffer_size() provides the buffer
 *                                    size if an additional buffer is required.
 * @param[in]   bmm_params            Batch matmul Parameters
 *                                    Adjoint flags are currently unused.
 * @param[in]   quant_params          Quantization parameters
 * @param[in]   input_lhs_dims        Input lhs tensor dimensions.
 *                                    This should be NHWC where lhs C = rhs C
 * @param[in]   input_lhs             Pointer to input tensor
 * @param[in]   input_rhs_dims        Input lhs tensor dimensions.
 *                                    This is expected to be transposed so
 *                                    should be NHWC where lhs C = rhs C
 * @param[in]   input_rhs             Pointer to transposed input tensor
 * @param[in]   output_dims           Output tensor dimensions
 * @param[out]  output                Pointer to the output tensor
 *
 * @return     The function returns <code>ARM_CMSIS_NN_SUCCESS</code>
 *
 * @details
 *    1. Supported framework: TensorFlow Lite Micro
 *    2. Performs row * row matrix multiplication with the RHS transposed.
 *
 */
arm_cmsis_nn_status arm_batch_matmul_s8(const cmsis_nn_context *ctx,
                                        const cmsis_nn_bmm_params *bmm_params,
                                        const cmsis_nn_per_tensor_quant_params *quant_params,
                                        const cmsis_nn_dims *input_lhs_dims,
                                        const int8_t *input_lhs,
                                        const cmsis_nn_dims *input_rhs_dims,
                                        const int8_t *input_rhs,
                                        const cmsis_nn_dims *output_dims,
                                        int8_t *output);

/**
 * @brief Batch matmul function with 16 bit input and output.
 *
 * @param[in]   ctx                   Temporary scratch buffer
 *                                    The caller is expected to clear the buffer, if applicable, for security reasons.
 *                                    Optional function arm_fully_connected_s8_get_buffer_size() provides the buffer
 *                                    size if an additional buffer is required.
 * @param[in]   bmm_params            Batch matmul Parameters
 *                                    Adjoint flags are currently unused.
 * @param[in]   quant_params          Quantization parameters
 * @param[in]   input_lhs_dims        Input lhs tensor dimensions.
 *                                    This should be NHWC where LHS.C = RHS.C
 * @param[in]   input_lhs             Pointer to input tensor
 * @param[in]   input_rhs_dims        Input lhs tensor dimensions.
 *                                    This is expected to be transposed so
 *                                    should be NHWC where LHS.C = RHS.C
 * @param[in]   input_rhs             Pointer to transposed input tensor
 * @param[in]   output_dims           Output tensor dimensions
 * @param[out]  output                Pointer to the output tensor
 *
 * @return     The function returns <code>ARM_CMSIS_NN_SUCCESS</code>
 *
 * @details
 *    1. Supported framework: TensorFlow Lite Micro
 *    2. Performs row * row matrix multiplication with the RHS transposed.
 *
 */
arm_cmsis_nn_status arm_batch_matmul_s16(const cmsis_nn_context *ctx,
                                         const cmsis_nn_bmm_params *bmm_params,
                                         const cmsis_nn_per_tensor_quant_params *quant_params,
                                         const cmsis_nn_dims *input_lhs_dims,
                                         const int16_t *input_lhs,
                                         const cmsis_nn_dims *input_rhs_dims,
                                         const int16_t *input_rhs,
                                         const cmsis_nn_dims *output_dims,
                                         int16_t *output);

/**
 * @defgroup Pad Pad Layer Functions:
 *
 */

/**
 * @brief Expands the size of the input by adding constant values before and after the data, in all dimensions.
 *
 * @param[in]   input                      Pointer to input data
 * @param[out]  output                     Pointer to output data
 * @param[in]   pad_value                  Value to pad with
 * @param[in]   input_size                 Input tensor dimensions
 * @param[in]   pre_pad                           Padding to apply before data in each dimension
 * @param[in]        post_pad                   Padding to apply after data in each dimension
 *
 * @return     The function returns <code>ARM_CMSIS_NN_SUCCESS</code>
 *
 */
arm_cmsis_nn_status arm_pad_s8(const int8_t *input,
                               int8_t *output,
                               const int8_t pad_value,
                               const cmsis_nn_dims *input_size,
                               const cmsis_nn_dims *pre_pad,
                               const cmsis_nn_dims *post_pad);

/**
 * @brief Elementwise binary minimum with 8bit data.
 *
 * @param[in]   ctx                   Temporary scratch buffer
 *                                    The caller is expected to clear the buffer, if applicable, for security reasons.
 * @param[in]   input_1_data          Pointer to input1 tensor
 * @param[in]   input_1_dims          Input1 tensor dimensions
 * @param[in]   input_2_data          Pointer to input2 tensor
 * @param[in]   input_2_dims          Input2 tensor dimensions
 * @param[out]  output_data           Pointer to the output tensor
 * @param[in]   output_dims           Output tensor dimensions
 *
 * @return     The function returns <code>ARM_CMSIS_NN_SUCCESS</code>
 *
 * @details
 *    1. Supported framework: TensorFlow Lite Micro
 *
 */
arm_cmsis_nn_status arm_minimum_s8(const cmsis_nn_context *ctx,
                                   const int8_t *input_1_data,
                                   const cmsis_nn_dims *input_1_dims,
                                   const int8_t *input_2_data,
                                   const cmsis_nn_dims *input_2_dims,
                                   int8_t *output_data,
                                   const cmsis_nn_dims *output_dims);

/**
 * @brief Elementwise binary maximum with 8bit data.
 *
 * @param[in]   ctx                   Temporary scratch buffer
 *                                    The caller is expected to clear the buffer, if applicable, for security reasons.
 * @param[in]   input_1_data          Pointer to input1 tensor
 * @param[in]   input_1_dims          Input1 tensor dimensions
 * @param[in]   input_2_data          Pointer to input2 tensor
 * @param[in]   input_2_dims          Input2 tensor dimensions
 * @param[out]  output_data           Pointer to the output tensor
 * @param[in]   output_dims           Output tensor dimensions
 *
 * @return     The function returns <code>ARM_CMSIS_NN_SUCCESS</code>
 *
 * @details
 *    1. Supported framework: TensorFlow Lite Micro
 *
 */
arm_cmsis_nn_status arm_maximum_s8(const cmsis_nn_context *ctx,
                                   const int8_t *input_1_data,
                                   const cmsis_nn_dims *input_1_dims,
                                   const int8_t *input_2_data,
                                   const cmsis_nn_dims *input_2_dims,
                                   int8_t *output_data,
                                   const cmsis_nn_dims *output_dims);

#ifdef __cplusplus
}
#endif

#endif /* ARM_NNFUNCTIONS_H */
