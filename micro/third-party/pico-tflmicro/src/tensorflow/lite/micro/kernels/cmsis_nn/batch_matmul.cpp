/* Copyright 2024 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include "tensorflow/lite/micro/kernels/batch_matmul.h"

#include "third_party/cmsis_nn/Include/arm_nnfunctions.h"
#include "tensorflow/lite/c/builtin_op_data.h"
#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/kernels/internal/common.h"
#include "tensorflow/lite/kernels/internal/portable_tensor_utils.h"
#include "tensorflow/lite/kernels/internal/quantization_util.h"
#include "tensorflow/lite/kernels/internal/reference/batch_matmul.h"
#include "tensorflow/lite/kernels/internal/reference/transpose.h"
#include "tensorflow/lite/kernels/internal/tensor_ctypes.h"
#include "tensorflow/lite/kernels/kernel_util.h"
#include "tensorflow/lite/micro/kernels/kernel_util.h"
#include "tensorflow/lite/micro/micro_arena_constants.h"
#include "tensorflow/lite/micro/micro_log.h"

namespace tflite {
namespace {

struct OpData {
  OpDataBatchMatmul reference_op_data;

  cmsis_nn_dims output_shape;

  int buffer_idx;
};

cmsis_nn_dims FillVariableShape(int32_t rank, int32_t* tensor_dims) {
  if (rank == 4) {
    return {tensor_dims[0], tensor_dims[1], tensor_dims[2], tensor_dims[3]};
  } else if (rank == 3) {
    return {1, tensor_dims[0], tensor_dims[1], tensor_dims[2]};
  } else if (rank == 2) {
    return {1, 1, tensor_dims[0], tensor_dims[1]};
  } else {
    return {1, 1, 1, 1};
  }
}

inline TfLiteStatus PopulateEvalData(
    TfLiteContext* context, OpData* data, const TfLiteBatchMatMulParams* params,
    TfLiteNode* node, const TfLiteEvalTensor* original_lhs_input,
    RuntimeShape* lhs_shape, TfLiteEvalTensor** updated_lhs_input,
    const TfLiteEvalTensor* original_rhs_input, RuntimeShape* rhs_shape,
    TfLiteEvalTensor** updated_rhs_input, const TfLiteEvalTensor* output) {
  RuntimeShape orig_out_shape = tflite::micro::GetTensorShape(output);

  *updated_rhs_input = params->adj_y
                           ? const_cast<TfLiteEvalTensor*>(original_rhs_input)
                           : data->reference_op_data.rhs_transposed_tensor;
  *updated_lhs_input = params->adj_x
                           ? data->reference_op_data.lhs_transposed_tensor
                           : const_cast<TfLiteEvalTensor*>(original_lhs_input);

  TF_LITE_ENSURE(context, *updated_rhs_input != nullptr);
  TF_LITE_ENSURE(context, *updated_lhs_input != nullptr);
  if (!params->adj_y) {
    // TODO(b/154760341): Constant tensors should already be transposed, but
    // we transpose once if necessary for now.
    if (!(data->reference_op_data.rhs_is_constant_tensor &&
          data->reference_op_data.rhs_is_transposed)) {
      TransposeRowsColumns(*original_rhs_input, *updated_rhs_input);
      data->reference_op_data.rhs_is_transposed = true;
    }
  }
  if (params->adj_x) {
    TransposeRowsColumns(*original_lhs_input, *updated_lhs_input);
  }

  // Compress BatchMatMul when third from last RHS dimension is one.
  int32_t rhs_dims_count = rhs_shape->DimensionsCount();
  int32_t lhs_dims_count = lhs_shape->DimensionsCount();
  int32_t out_dims_count = orig_out_shape.DimensionsCount();
  // Compress ops where rhs shape is [..., 1, X, Y] and lhs shape is
  // [..., Q, R, S] which is equivalent to rhs: [..., X, Y] and
  // lhs: [..., Q * R, S].
  if (rhs_dims_count > 2 && lhs_dims_count > 2) {
    int rhs_one = rhs_shape->DimsData()[rhs_dims_count - 3];
    if (rhs_one == 1) {
      int32_t* lhs_dims = lhs_shape->DimsData();
      int32_t* rhs_dims = rhs_shape->DimsData();
      int32_t* out_dims = orig_out_shape.DimsData();
      RuntimeShape tmp_l(lhs_dims_count - 1, lhs_dims);
      tmp_l.SetDim(lhs_dims_count - 3,
                   lhs_dims[lhs_dims_count - 3] * lhs_dims[lhs_dims_count - 2]);
      tmp_l.SetDim(lhs_dims_count - 2, lhs_dims[lhs_dims_count - 1]);
      lhs_shape->ReplaceWith(tmp_l.DimensionsCount(), tmp_l.DimsData());
      RuntimeShape tmp_r(rhs_dims_count - 1, rhs_shape->DimsData());
      tmp_r.SetDim(rhs_dims_count - 3, rhs_dims[rhs_dims_count - 2]);
      tmp_r.SetDim(rhs_dims_count - 2, rhs_dims[rhs_dims_count - 1]);
      rhs_shape->ReplaceWith(tmp_r.DimensionsCount(), tmp_r.DimsData());
      rhs_dims_count = rhs_shape->DimensionsCount();
      lhs_dims_count = lhs_shape->DimensionsCount();

      RuntimeShape tmp_o(out_dims_count - 1, out_dims);
      tmp_o.SetDim(out_dims_count - 3, lhs_shape->Dims(lhs_dims_count - 2));
      tmp_o.SetDim(out_dims_count - 2, orig_out_shape.Dims(out_dims_count - 1));
      orig_out_shape.ReplaceWith(tmp_o.DimensionsCount(), tmp_o.DimsData());
      out_dims_count = orig_out_shape.DimensionsCount();
      data->output_shape =
          FillVariableShape(out_dims_count, orig_out_shape.DimsData());
    }
  }

  if (!params->adj_y) {
    RuntimeShape tmp_r = SwapRowColumnDims(*rhs_shape);
    rhs_shape->ReplaceWith(tmp_r.DimensionsCount(), tmp_r.DimsData());
  }
  // ReferenceOps and CMSIS-NN have different requirements for when the
  // lhs shape should be transposed, so we have to treat float differently.
  if (!params->adj_x && original_lhs_input->type == kTfLiteFloat32) {
    RuntimeShape tmp_l = SwapRowColumnDims(*lhs_shape);
    lhs_shape->ReplaceWith(tmp_l.DimensionsCount(), tmp_l.DimsData());
  } else if (params->adj_x && original_lhs_input->type != kTfLiteFloat32) {
    RuntimeShape tmp_l = SwapRowColumnDims(*lhs_shape);
    lhs_shape->ReplaceWith(tmp_l.DimensionsCount(), tmp_l.DimsData());
  }

  return kTfLiteOk;
}

TfLiteEvalTensor* AllocInitTransposeTensorFromTfLiteTensor(
    TfLiteContext* context, MicroContext* micro_context,
    const TfLiteTensor& tensor) {
  TfLiteEvalTensor* eval_tensor = static_cast<TfLiteEvalTensor*>(
      micro_context->AllocatePersistentBuffer(sizeof(TfLiteEvalTensor)));
  if (eval_tensor == nullptr) {
    return nullptr;
  }

  eval_tensor->type = tensor.type;

  const int tensor_rank = NumDimensions(&tensor);
  const size_t eval_dims_size = TfLiteIntArrayGetSizeInBytes(tensor_rank);
  eval_tensor->dims = static_cast<TfLiteIntArray*>(
      micro_context->AllocatePersistentBuffer(eval_dims_size));
  if (eval_tensor->dims == nullptr) {
    return nullptr;
  }
  eval_tensor->dims->size = tensor_rank;
  for (int i = 0; i < tensor_rank - 2; ++i) {
    eval_tensor->dims->data[i] = tensor.dims->data[i];
  }
  // Swap last two dimensions.
  eval_tensor->dims->data[tensor_rank - 2] = tensor.dims->data[tensor_rank - 1];
  eval_tensor->dims->data[tensor_rank - 1] = tensor.dims->data[tensor_rank - 2];

  const size_t eval_data_size = static_cast<size_t>(NumElements(&tensor)) *
                                TfLiteTypeGetSize(tensor.type);
  eval_tensor->data.data =
      micro_context->AllocatePersistentBuffer(eval_data_size);
  if (eval_tensor->data.data == nullptr) {
    return nullptr;
  }

  return eval_tensor;
}

void* Init(TfLiteContext* context, const char* buffer, size_t length) {
  TFLITE_DCHECK(context->AllocatePersistentBuffer != nullptr);
  return context->AllocatePersistentBuffer(context, sizeof(OpData));
}

TfLiteStatus Prepare(TfLiteContext* context, TfLiteNode* node) {
  TFLITE_DCHECK(node->user_data != nullptr);
  TFLITE_DCHECK(node->builtin_data != nullptr);

  OpData* data = static_cast<OpData*>(node->user_data);
  const auto params =
      static_cast<const TfLiteBatchMatMulParams*>(node->builtin_data);
  MicroContext* micro_context = GetMicroContext(context);
  TfLiteTensor* lhs_input =
      micro_context->AllocateTempInputTensor(node, kBatchMatmulInputLhsTensor);
  TF_LITE_ENSURE(context, lhs_input != nullptr);
  TfLiteTensor* rhs_input =
      micro_context->AllocateTempInputTensor(node, kBatchMatmulInputRhsTensor);
  TF_LITE_ENSURE(context, rhs_input != nullptr);
  TfLiteTensor* output =
      micro_context->AllocateTempOutputTensor(node, kBatchMatmulOutputTensor);
  TF_LITE_ENSURE(context, output != nullptr);

  TF_LITE_ENSURE_TYPES_EQ(context, lhs_input->type, rhs_input->type);
  TF_LITE_ENSURE_EQ(context, lhs_input->type, output->type);
  TF_LITE_ENSURE_MSG(context,
                     lhs_input->type == kTfLiteFloat32 ||
                         lhs_input->type == kTfLiteInt16 ||
                         lhs_input->type == kTfLiteInt8,
                     "Input data type not supported");

  const int lhs_rank = NumDimensions(lhs_input);
  const int rhs_rank = NumDimensions(rhs_input);

  TF_LITE_ENSURE(context, lhs_rank >= 2);
  TF_LITE_ENSURE(context, lhs_rank <= 4);
  TF_LITE_ENSURE(context, rhs_rank >= 2);
  TF_LITE_ENSURE(context, rhs_rank <= 4);

  data->reference_op_data.rhs_is_transposed = false;
  data->reference_op_data.lhs_is_constant_tensor = IsConstantTensor(lhs_input);
  data->reference_op_data.rhs_is_constant_tensor = IsConstantTensor(rhs_input);

  const int output_rank = std::max(lhs_rank, rhs_rank);
  TFLITE_DCHECK_GE(output_rank, 2);
  TFLITE_DCHECK_LE(output_rank, 4);

  const RuntimeShape extended_lhs_shape =
      RuntimeShape::ExtendedShape(output_rank, GetTensorShape(lhs_input));
  const RuntimeShape extended_rhs_shape =
      RuntimeShape::ExtendedShape(output_rank, GetTensorShape(rhs_input));

  // Ensure any batch dimensions obey broacasting rules.
  for (int i = 0; i < output_rank - 2; ++i) {
    const int lhs_dim = extended_lhs_shape.Dims(i);
    const int rhs_dim = extended_rhs_shape.Dims(i);
    if (lhs_dim != rhs_dim) {
      if (lhs_dim != 1) {
        TF_LITE_ENSURE_EQ(context, rhs_dim, 1);
      }
    }
  }

  bool adj_x = params->adj_x;
  bool adj_y = params->adj_y;
  // Ensure other dimensions work for matrix multiplication.
  int accum_dim_lhs = adj_x ? extended_lhs_shape.Dims(output_rank - 2)
                            : extended_lhs_shape.Dims(output_rank - 1);
  int accum_dim_rhs = adj_y ? extended_rhs_shape.Dims(output_rank - 1)
                            : extended_rhs_shape.Dims(output_rank - 2);

  TF_LITE_ENSURE_EQ(context, accum_dim_lhs, accum_dim_rhs);

  // Tensor for transposed LHS;
  if (adj_x) {
    data->reference_op_data.lhs_transposed_tensor =
        AllocInitTransposeTensorFromTfLiteTensor(context, micro_context,
                                                 *lhs_input);
    TF_LITE_ENSURE(context,
                   data->reference_op_data.lhs_transposed_tensor != nullptr);
  }

  // If RHS needs to be transposed, then it is actually in the correct shape
  // already.
  if (!adj_y) {
    data->reference_op_data.rhs_transposed_tensor =
        AllocInitTransposeTensorFromTfLiteTensor(context, micro_context,
                                                 *rhs_input);
    TF_LITE_ENSURE(context,
                   data->reference_op_data.rhs_transposed_tensor != nullptr);
  }

  TF_LITE_ENSURE_STATUS(ReshapeOutputTensor(context, node, extended_lhs_shape,
                                            extended_rhs_shape, adj_x, adj_y,
                                            output_rank, output));

  data->output_shape = FillVariableShape(
      output_rank, reinterpret_cast<int32_t*>(output->dims->data));

  int buf_size = 0;
  if (lhs_input->type != kTfLiteFloat32 && rhs_input->type != kTfLiteFloat32) {
    data->reference_op_data.quantization =
        static_cast<decltype(data->reference_op_data.quantization)>(
            micro_context->AllocatePersistentBuffer(
                sizeof(*data->reference_op_data.quantization)));
    TF_LITE_ENSURE(context, data->reference_op_data.quantization != nullptr);

    double real_multiplier = 0.0;
    TF_LITE_ENSURE_STATUS(GetQuantizedConvolutionMultipler(
        context, lhs_input, rhs_input, output, &real_multiplier));
    QuantizeMultiplier(real_multiplier,
                       &data->reference_op_data.quantization->output_multiplier,
                       &data->reference_op_data.quantization->output_shift);

    data->reference_op_data.quantization->lhs_zero_point =
        lhs_input->params.zero_point;
    data->reference_op_data.quantization->rhs_zero_point =
        rhs_input->params.zero_point;
    data->reference_op_data.quantization->output_zero_point =
        output->params.zero_point;

    if (lhs_input->type == kTfLiteInt8) {
      data->reference_op_data.quantization->output_activation_min =
          std::numeric_limits<int8_t>::min();
      data->reference_op_data.quantization->output_activation_max =
          std::numeric_limits<int8_t>::max();

      data->buffer_idx = -1;
      buf_size = arm_fully_connected_s8_get_buffer_size(&data->output_shape);
    } else {
      data->reference_op_data.quantization->output_activation_min =
          std::numeric_limits<int16_t>::min();
      data->reference_op_data.quantization->output_activation_max =
          std::numeric_limits<int16_t>::max();

      TF_LITE_ENSURE_EQ(context, lhs_input->params.zero_point, 0);
      TF_LITE_ENSURE_EQ(context, rhs_input->params.zero_point, 0);
      TF_LITE_ENSURE_EQ(context, output->params.zero_point, 0);
    }
  }

  if (buf_size > 0) {
    TF_LITE_ENSURE_STATUS(context->RequestScratchBufferInArena(
        context, buf_size, &data->buffer_idx));
  }

  micro_context->DeallocateTempTfLiteTensor(output);
  micro_context->DeallocateTempTfLiteTensor(lhs_input);
  micro_context->DeallocateTempTfLiteTensor(rhs_input);

  return kTfLiteOk;
}

TfLiteStatus EvalInt8(TfLiteContext* context, TfLiteNode* node) {
  TFLITE_DCHECK(node->user_data != nullptr);
  TFLITE_DCHECK(node->builtin_data != nullptr);

  const TfLiteEvalTensor* original_lhs_input =
      tflite::micro::GetEvalInput(context, node, kBatchMatmulInputLhsTensor);
  const TfLiteEvalTensor* original_rhs_input =
      tflite::micro::GetEvalInput(context, node, kBatchMatmulInputRhsTensor);
  TfLiteEvalTensor* output =
      tflite::micro::GetEvalOutput(context, node, kBatchMatmulOutputTensor);

  OpData& data = *(static_cast<OpData*>(node->user_data));
  const auto* params =
      static_cast<const TfLiteBatchMatMulParams*>(node->builtin_data);

  RuntimeShape rhs_shape = tflite::micro::GetTensorShape(original_rhs_input);
  RuntimeShape lhs_shape = tflite::micro::GetTensorShape(original_lhs_input);
  TfLiteEvalTensor* updated_lhs_input;
  TfLiteEvalTensor* updated_rhs_input;

  TF_LITE_ENSURE_STATUS(
      PopulateEvalData(context, &data, params, node, original_lhs_input,
                       &lhs_shape, &updated_lhs_input, original_rhs_input,
                       &rhs_shape, &updated_rhs_input, output));

  cmsis_nn_dims rhs_dims =
      FillVariableShape(rhs_shape.DimensionsCount(), rhs_shape.DimsData());
  cmsis_nn_dims lhs_dims =
      FillVariableShape(lhs_shape.DimensionsCount(), lhs_shape.DimsData());

  cmsis_nn_per_tensor_quant_params quant_params = {
      data.reference_op_data.quantization->output_multiplier,
      data.reference_op_data.quantization->output_shift};
  cmsis_nn_context ctx;
  ctx.buf = nullptr;
  ctx.size = 0;

  if (data.buffer_idx > -1) {
    ctx.buf = context->GetScratchBuffer(context, data.buffer_idx);
    // Note: ctx.size is currently not used in cmsis_nn.
    // The buffer should be allocated in the prepare function through
    // the corresponding arm_convolve_wrapper_[type]_get_buffer_size
  }

  cmsis_nn_fc_params fc_params;
  fc_params.input_offset = -data.reference_op_data.quantization->lhs_zero_point;
  fc_params.filter_offset =
      -data.reference_op_data.quantization->rhs_zero_point;
  fc_params.output_offset =
      data.reference_op_data.quantization->output_zero_point;

  cmsis_nn_activation activation;
  activation.min = data.reference_op_data.quantization->output_activation_min;
  activation.max = data.reference_op_data.quantization->output_activation_max;
  fc_params.activation = activation;

  cmsis_nn_bmm_params bmm_params = {
      params->adj_x,
      params->adj_y,
      fc_params,
  };

  TF_LITE_ENSURE_EQ(
      context,
      arm_batch_matmul_s8(
          &ctx, &bmm_params, &quant_params, &lhs_dims,
          tflite::micro::GetTensorData<int8_t>(updated_lhs_input), &rhs_dims,
          tflite::micro::GetTensorData<int8_t>(updated_rhs_input),
          &data.output_shape, tflite::micro::GetTensorData<int8_t>(output)),
      ARM_CMSIS_NN_SUCCESS);

  return kTfLiteOk;
}

TfLiteStatus EvalInt16(TfLiteContext* context, TfLiteNode* node) {
  TFLITE_DCHECK(node->user_data != nullptr);
  TFLITE_DCHECK(node->builtin_data != nullptr);

  const TfLiteEvalTensor* original_lhs_input =
      tflite::micro::GetEvalInput(context, node, kBatchMatmulInputLhsTensor);
  const TfLiteEvalTensor* original_rhs_input =
      tflite::micro::GetEvalInput(context, node, kBatchMatmulInputRhsTensor);
  TfLiteEvalTensor* output =
      tflite::micro::GetEvalOutput(context, node, kBatchMatmulOutputTensor);

  OpData& data = *(static_cast<OpData*>(node->user_data));
  const auto* params =
      static_cast<const TfLiteBatchMatMulParams*>(node->builtin_data);

  RuntimeShape rhs_shape = tflite::micro::GetTensorShape(original_rhs_input);
  RuntimeShape lhs_shape = tflite::micro::GetTensorShape(original_lhs_input);

  // These pointers will be updated to point at the actual tensor being used in
  // the batch matmul function
  TfLiteEvalTensor* updated_lhs_input;
  TfLiteEvalTensor* updated_rhs_input;

  TF_LITE_ENSURE_STATUS(
      PopulateEvalData(context, &data, params, node, original_lhs_input,
                       &lhs_shape, &updated_lhs_input, original_rhs_input,
                       &rhs_shape, &updated_rhs_input, output));

  cmsis_nn_dims rhs_dims =
      FillVariableShape(rhs_shape.DimensionsCount(), rhs_shape.DimsData());
  cmsis_nn_dims lhs_dims =
      FillVariableShape(lhs_shape.DimensionsCount(), lhs_shape.DimsData());

  cmsis_nn_per_tensor_quant_params quant_params = {
      data.reference_op_data.quantization->output_multiplier,
      data.reference_op_data.quantization->output_shift};
  cmsis_nn_context ctx;
  ctx.buf = nullptr;
  ctx.size = 0;

  cmsis_nn_fc_params fc_params;
  fc_params.input_offset = -data.reference_op_data.quantization->lhs_zero_point;
  fc_params.filter_offset =
      -data.reference_op_data.quantization->rhs_zero_point;
  fc_params.output_offset =
      data.reference_op_data.quantization->output_zero_point;

  cmsis_nn_activation activation;
  activation.min = data.reference_op_data.quantization->output_activation_min;
  activation.max = data.reference_op_data.quantization->output_activation_max;
  fc_params.activation = activation;

  cmsis_nn_bmm_params bmm_params = {
      params->adj_x,
      params->adj_y,
      fc_params,
  };

  TF_LITE_ENSURE_EQ(
      context,
      arm_batch_matmul_s16(
          &ctx, &bmm_params, &quant_params, &lhs_dims,
          tflite::micro::GetTensorData<int16_t>(updated_lhs_input), &rhs_dims,
          tflite::micro::GetTensorData<int16_t>(updated_rhs_input),
          &data.output_shape, tflite::micro::GetTensorData<int16_t>(output)),
      ARM_CMSIS_NN_SUCCESS);

  return kTfLiteOk;
}

TfLiteStatus Eval(TfLiteContext* context, TfLiteNode* node) {
  // Checks in Prepare ensure input, output and filter types are all the same.
  TFLITE_DCHECK(node->builtin_data != nullptr);
  const TfLiteEvalTensor* original_lhs_input =
      tflite::micro::GetEvalInput(context, node, kBatchMatmulInputLhsTensor);
  switch (original_lhs_input->type) {
    case kTfLiteFloat32: {
      const TfLiteEvalTensor* original_rhs_input = tflite::micro::GetEvalInput(
          context, node, kBatchMatmulInputRhsTensor);
      TfLiteEvalTensor* output =
          tflite::micro::GetEvalOutput(context, node, kBatchMatmulOutputTensor);

      TFLITE_DCHECK(node->user_data != nullptr);
      OpData& data = *(static_cast<OpData*>(node->user_data));
      const auto* params =
          static_cast<const TfLiteBatchMatMulParams*>(node->builtin_data);

      RuntimeShape rhs_shape =
          tflite::micro::GetTensorShape(original_rhs_input);
      RuntimeShape lhs_shape =
          tflite::micro::GetTensorShape(original_lhs_input);
      TfLiteEvalTensor* updated_lhs_input;
      TfLiteEvalTensor* updated_rhs_input;

      TF_LITE_ENSURE_STATUS(
          PopulateEvalData(context, &data, params, node, original_lhs_input,
                           &lhs_shape, &updated_lhs_input, original_rhs_input,
                           &rhs_shape, &updated_rhs_input, output));

      // Note we pass RHS args first, LHS args second.
      reference_ops::BatchMatMul(
          rhs_shape, tflite::micro::GetTensorData<float>(updated_rhs_input),
          lhs_shape, tflite::micro::GetTensorData<float>(updated_lhs_input),
          tflite::micro::GetTensorShape(output),
          tflite::micro::GetTensorData<float>(output));
    } break;
    case kTfLiteInt8:
      return EvalInt8(context, node);
    case kTfLiteInt16:
      return EvalInt16(context, node);
    default: {
      MicroPrintf("CMSIS-NN Batch Matmul: Type %s (%d) not supported.",
                  TfLiteTypeGetName(original_lhs_input->type),
                  original_lhs_input->type);
      return kTfLiteError;
    }
  }
  return kTfLiteOk;
}

}  // namespace

TFLMRegistration Register_BATCH_MATMUL() {
  return tflite::micro::RegisterOp(Init, Prepare, Eval);
}

TFLMRegistration Register_BATCH_MATMUL_INT8() {
  return tflite::micro::RegisterOp(Init, Prepare, EvalInt8);
}

TFLMRegistration Register_BATCH_MATMUL_INT16() {
  return tflite::micro::RegisterOp(Init, Prepare, EvalInt16);
}

}  // namespace tflite
