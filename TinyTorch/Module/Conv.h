#pragma once
#include "Module_base.h"
namespace TinyTorch::nn {
class Conv2D : public Module {
 public:
  Conv2D(int32_t inFeatures, int32_t outFeatures, Size2D kernelSize,
         Size2D stride = 1, Size2D padding = 0, bool bias = true);

  Tensor forward(Tensor &input) override;
  std::vector<Tensor *> parameters() override;
  std::vector<Tensor *> states() override;
  void resetParameters() override;
  void zeroGrad() override;

  Tensor &weights() { return weights_; }
  Tensor &bias() { return bias_; }
  Size2D &kernelsize() {return kernelSize_;}
  int32_t &outfeatures() {return outFeatures_;}
 private:
  int32_t inFeatures_;
  int32_t outFeatures_;
  Size2D kernelSize_;
  Size2D stride_;
  Size2D padding_;
  bool useBias_;
  Tensor weights_;
  Tensor bias_;
  std::string name_ = "Conv2D";
};


class Conv1D : public Module {
 public:
  Conv1D(int32_t inFeatures, int32_t outFeatures, Size1D kernelSize,
         Size1D stride = 1, Size1D padding = 0, bool bias = true);
  Tensor forward(Tensor &input) override;
  std::vector<Tensor *> parameters() override;
  std::vector<Tensor *> states() override;
  void resetParameters() override;
  void zeroGrad() override;
  Tensor &weights() { return weights_; }
  Tensor &bias() { return bias_; }
  Size1D &kernelsize() {return kernelSize_;}
  int32_t &outfeatures() {return outFeatures_;}
 private:
  int32_t inFeatures_;
  int32_t outFeatures_;
  Size1D kernelSize_;
  Size1D stride_;
  Size1D padding_;
  bool useBias_;
  Tensor weights_;
  Tensor bias_;
  std::string name_ = "Conv1D";
};

}