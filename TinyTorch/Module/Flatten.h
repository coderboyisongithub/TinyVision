#pragma once
#include "Module_base.h"

namespace TinyTorch::nn {
class Flatten : public Module {
 public:
  explicit Flatten(int32_t startDim = 0, int32_t endDim = -1)
      : startDim_(startDim), endDim_(endDim) {}

  Tensor forward(Tensor &input) override;

 private:
  int32_t startDim_;
  int32_t endDim_;
   std::string name_ = "Flatten";
};
}