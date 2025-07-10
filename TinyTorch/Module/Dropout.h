#pragma once
#include "Module_base.h"

namespace TinyTorch::nn {



class Dropout : public Module {
 public:
  explicit Dropout(float p = 0.5f) : p_(p) {}

  Tensor forward(Tensor &x) override;

 private:
  float p_;
   std::string name_ = "Dropout";
};
}