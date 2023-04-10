// // custom_op.cc
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/shape_inference.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/platform/env.h"

using namespace tensorflow;

extern "C" {
void process_strings_wrapper(const char** input, size_t len, int32_t** output, size_t* output_len);
}

REGISTER_OP("ProcessStrings")
    .Input("input: string")
    .Output("output: int32")
    .SetShapeFn([](shape_inference::InferenceContext* c) {
        c->set_output(0, c->MakeShape({c->UnknownDim()}));
        return Status::OK();
    });

class ProcessStringsOp : public OpKernel {
 public:
  explicit ProcessStringsOp(OpKernelConstruction* context) : OpKernel(context) {}

  void Compute(OpKernelContext* context) override {
    const Tensor& input_tensor = context->input(0);
    auto input_flat = input_tensor.flat<tstring>();

    std::vector<const char*> input_strings(input_flat.size());
    for (int i = 0; i < input_flat.size(); ++i) {
      input_strings[i] = input_flat(i).data();
    }

    int32_t* output_data = nullptr;
    size_t output_len = 0;
    process_strings_wrapper(input_strings.data(), input_strings.size(), &output_data, &output_len);    // custom_op.cc (continued)
    Tensor* output_tensor = nullptr;
    TensorShape output_shape({static_cast<int64>(output_len)});
    OP_REQUIRES_OK(context, context->allocate_output(0, output_shape, &output_tensor));

    auto output_flat = output_tensor->flat<int32>();
    std::memcpy(output_flat.data(), output_data, output_len * sizeof(int32));
    free(output_data);
  }
};

REGISTER_KERNEL_BUILDER(Name("ProcessStrings").Device(DEVICE_CPU), ProcessStringsOp);
