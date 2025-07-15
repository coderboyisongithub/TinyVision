/*
 * TinyTorch
 * @author 	: keith@robot9.me
 *
 */

#include "Module.h"

#include <cassert>

#include "Function.h"
#include "Init.h"
#include "Enums.h"
namespace TinyTorch::nn {

void Module::getTopologyTextHelper(std::stringstream& ss, int depth) const {
    ss << std::string(depth * 2, ' ') << "|-- " << name() << " (" << class_name() << ")" <<std::endl;
    for (const auto& submod : subModules_) {
        submod.get().getTopologyTextHelper(ss, depth + 1);
    }
}

std::vector<Tensor *> Module::parameters() {
  std::vector<Tensor *> ret;
  for (auto &module : subModules_) {
    for (auto p : module.get().parameters()) {
      ret.push_back(p);
    }
  }
  return ret;
}

std::vector<Tensor *> Module::states() {
  std::vector<Tensor *> ret;
  for (auto &module : subModules_) {
    for (auto p : module.get().states()) {
      ret.push_back(p);

    }
  }
  return ret;
}

void Module::resetParameters() {
  for (auto &module : subModules_) {
    module.get().resetParameters();
  }
}

void Module::zeroGrad() {
  for (auto &module : subModules_) {
    module.get().zeroGrad();
  }
}

void Module::load_state_dict(std::map<std::string, Tensor> param_dict, Device device) {
  for (auto &module : subModules_) {
    for (auto name : module.get().get_named_tensors_()) {
        std::string all_name = module.get().name().append("."+name.first);
        auto it = param_dict.find(all_name);
        if (it == param_dict.end()) {
          std::cerr << "Warning: " << all_name << " not found\n";
          continue;
        }
        Tensor* dest_tensor = name.second;
        Tensor src_tensor = param_dict[all_name];
        if (src_tensor.shape() != dest_tensor->shape()){
            std::cerr << "Warning: Tensor " << all_name << " shape=[";
            for (size_t dim : dest_tensor->shape()) {
                std::cerr << dim << " ";
            }
            std::cerr << "] " << " is not same as the param your provide.";
            std::cerr << " shape=[";
            for (size_t dim : src_tensor.shape()) {
                std::cerr << dim << " ";
            }
            std::cerr << "] \n";
        }
        else{
            *dest_tensor = Tensor(std::move(src_tensor.data()));
            }
        dest_tensor->to(device);
        }
    }
}


void Module::to(Device device) {
  for (auto &module : subModules_) {
    for (auto p : module.get().states()) {
      p->to(device);
    }
  }
}

void Module::to(Dtype type) {
  for (auto &module : subModules_) {
    for (auto p : module.get().states()) {
      p->to(type);
    }
  }
}

Tensor Sequential::forward(Tensor &input) {
  Tensor ret = {input};
  for (auto &module : modules_) {
    ret = (*module)(ret);
  }
  return ret;
}

std::vector<Tensor *> Sequential::parameters() {
  std::vector<Tensor *> ret;
  for (auto &module : modules_) {
    for (auto p : module->parameters()) {
      ret.push_back(p);
    }
  }
  return ret;
}

std::map<std::string, Tensor *> Sequential::get_named_tensors_() const {
  std::map<std::string,Tensor*> ret;
  for (auto &module : modules_) {
    for (auto m : module->get_named_tensors_()) {
        ret[module->name() + "." + m.first] = m.second;
    }
    }
  return ret;
}

std::vector<Tensor *> Sequential::states() {
  std::vector<Tensor *> ret;
  for (auto &module : modules_) {
    for (auto p : module->states()) {
      ret.push_back(p);
    }
  }
  return ret;
}

void Sequential::resetParameters() {
  for (auto &module : modules_) {
    module->resetParameters();
  }
}

void Sequential::zeroGrad() {
  for (auto &module : modules_) {
    module->zeroGrad();
  }
}

void Sequential::setTraining(bool mode) {
  Module::setTraining(mode);
  for (auto &module : modules_) {
    module->train(mode);
  }
}

MultiheadAttention::MultiheadAttention(int32_t inFeatures, int32_t head, int is_casual, bool bias, bool bias_proj)
    : head_(head), is_casual_(is_casual) , useBias_(bias), useprojBias_(bias_proj),
    qkv_proj_(inFeatures, inFeatures*3, bias), last_proj_(inFeatures, inFeatures, bias_proj) {
    registerModules({qkv_proj_, last_proj_});
    std::string  qkv_proj_name =  qkv_proj_.name() + ".qkv_proj";
    std::string  last_proj_name =  last_proj_.name() + ".last_proj";
    qkv_proj_.set_name(qkv_proj_name);
    last_proj_.set_name(last_proj_name);
    MultiheadAttention::resetParameters();
}

Tensor MultiheadAttention::forward(Tensor &input) {
  int B = input.shape()[0];
  int L = input.shape()[1];
  int C = input.shape()[2];
  input = Function::reshape(input, {B * L, C});
  input = qkv_proj_(input);
  input = Function::reshape(input, {B , L, 3 * C});
  input = Function::selfattention_qkv(input, head_, is_casual_);
  input = Function::reshape(input, {B * L, C});
  input = last_proj_(input);
  input = Function::reshape(input, {B, L, C});
  return input;
}

std::vector<Tensor *> MultiheadAttention::parameters() {
  if (useBias_ && useprojBias_) {
    return {&qkv_proj_.weights(), &qkv_proj_.bias(), &last_proj_.weights(), &last_proj_.bias()};
  }
  else if (useBias_ ) {
    return {&qkv_proj_.weights(), &qkv_proj_.bias(), &last_proj_.weights()};
  }
  else if (useprojBias_ ) {
    return {&qkv_proj_.weights(), &last_proj_.weights(), &last_proj_.bias()};
  }
  return {&qkv_proj_.weights(), &last_proj_.weights()};
}

std::vector<Tensor *> MultiheadAttention::states() { return parameters(); }

void MultiheadAttention::resetParameters() {
  Init::xavierUniform(qkv_proj_.weights(), 1.0f, FAN_AVG);
  Init::xavierUniform(last_proj_.weights(), 1.0f, FAN_AVG);
  if (useBias_) {
    Init::uniform(qkv_proj_.bias(), 0.0f, 0.0f);
  }
  if (useprojBias_) {
    Init::uniform(last_proj_.bias(), 0.0f, 0.0f);
  }
}

void MultiheadAttention::zeroGrad() {
  qkv_proj_.weights().zeroGrad();
  last_proj_.weights().zeroGrad();
  if (useBias_) {
    qkv_proj_.bias().zeroGrad();
  }
  if (useprojBias_) {
    last_proj_.bias().zeroGrad();
  }
}

Linear::Linear(int32_t inFeatures, int32_t outFeatures, bool bias)
    : inFeatures_(inFeatures), outFeatures_(outFeatures), useBias_(bias) {
  REGISTER_TENSOR(weight , Tensor::shape({outFeatures, inFeatures}, true));
  if (bias) {
    REGISTER_TENSOR(bias ,Tensor::shape({outFeatures}, true));
  }
  Linear::resetParameters();
}

Tensor Linear::forward(Tensor &input) {
  return Function::linear(input, weight_, bias_);
}

std::vector<Tensor *> Linear::parameters() {
  if (useBias_) {
    return {&weight_, &bias_};
  }
  return {&weight_};
}



std::vector<Tensor *> Linear::states() { return parameters(); }

void Linear::resetParameters() {
  Init::kaimingUniform(weight_, std::sqrt(5.f));
  if (useBias_) {
    auto fanIn = Init::calculateFan(weight_).first;
    const auto bound = fanIn > 0 ? 1.f / std::sqrt((float)fanIn) : 0;
    Init::uniform(bias_, -bound, bound);
  }
}

void Linear::zeroGrad() {
  weight_.zeroGrad();
  if (useBias_) {
    bias_.zeroGrad();
  }
}

Tensor Flatten::forward(Tensor &input) {
  return Function::flatten(input, startDim_, endDim_);
}

Tensor Relu::forward(Tensor &input) { return Function::relu(input); }

Tensor LeakyRelu::forward(Tensor &input) { return Function::leakyrelu(input, rate_); }

Tensor Dropout::forward(Tensor &input) {
  return Function::dropout(input, p_, training_);
}

Tensor Softmax::forward(Tensor &input) {
  return Function::softmax(input, dim_);
}

Tensor LogSoftmax::forward(Tensor &input) {
  return Function::logSoftmax(input, dim_);
}

Tensor MaxPool2D::forward(Tensor &input) {
  return Function::maxPool2d(input, kernelSize_, stride_, padding_);
}

Tensor AdaptiveAvgPool2D::forward(Tensor &input) {
  auto kernel_size = calculate_kernel_size(input.shape(), output_size_);
  return Function::avgPool2d(input, kernel_size, kernel_size, 0);
}

Tensor AvgPool2D::forward(Tensor &input) {
  return Function::avgPool2d(input, kernelSize_, stride_, padding_);
}


Conv1D::Conv1D(int32_t inFeatures, int32_t outFeatures, Size1D kernelSize,
               Size1D stride, Size1D padding, bool bias)
    : inFeatures_(inFeatures),
      outFeatures_(outFeatures),
      kernelSize_(kernelSize),
      stride_(stride),
      padding_(padding),
      useBias_(bias){
  if (bias) {
    REGISTER_TENSOR(bias , Tensor::shape({outFeatures}, true));
  }
  Conv1D::resetParameters();
}

Tensor Conv1D::forward(Tensor &input) {
  return Function::conv1d(input, weight_, bias_, stride_, padding_);
}

std::vector<Tensor *> Conv1D::parameters() {
  if (useBias_) {
    return {&weight_, &bias_};
  }
  return {&weight_};
}


std::vector<Tensor *> Conv1D::states() { return parameters(); }

void Conv1D::resetParameters() {
  Init::kaimingUniform(weight_, std::sqrt(5.f));
  if (useBias_) {
    auto fanIn = Init::calculateFan(weight_).first;
    if (fanIn != 0) {
      const auto bound = 1.f / std::sqrt((float)fanIn);
      Init::uniform(bias_, -bound, bound);
    }
  }
}

void Conv1D::zeroGrad() {
  weight_.zeroGrad();
  if (useBias_) {
    bias_.zeroGrad();
  }
}

Conv2D::Conv2D(int32_t inFeatures, int32_t outFeatures, Size2D kernelSize,
               Size2D stride, Size2D padding, bool bias)
    : inFeatures_(inFeatures),
      outFeatures_(outFeatures),
      kernelSize_(kernelSize),
      stride_(stride),
      padding_(padding),
      useBias_(bias){

  REGISTER_TENSOR(weight, Tensor::shape(
      {outFeatures, inFeatures, kernelSize_.h, kernelSize_.w}, true));
  if (bias) {
    REGISTER_TENSOR(bias , Tensor::shape({outFeatures}, true));
  }
  Conv2D::resetParameters();
}

Tensor Conv2D::forward(Tensor &input) {
  return Function::conv2d(input, weight_, bias_, stride_, padding_);
}

std::vector<Tensor *> Conv2D::parameters() {
  if (useBias_) {
    return {&weight_, &bias_};
  }
  return {&weight_};
}

std::vector<Tensor *> Conv2D::states() { return parameters(); }

void Conv2D::resetParameters() {
  Init::kaimingUniform(weight_, std::sqrt(5.f));
  if (useBias_) {
    auto fanIn = Init::calculateFan(weight_).first;
    if (fanIn != 0) {
      const auto bound = 1.f / std::sqrt((float)fanIn);
      Init::uniform(bias_, -bound, bound);
    }
  }
}

void Conv2D::zeroGrad() {
  weight_.zeroGrad();
  if (useBias_) {
    bias_.zeroGrad();
  }
}

GroupNorm::GroupNorm(int32_t numFeatures, float eps, int32_t num_groups, bool affine)
    : numFeatures_(numFeatures),
      eps_(eps),
      num_groups_(num_groups),
      affine_(affine){
  if (affine_) {
    REGISTER_TENSOR(weights , Tensor::shape({numFeatures_}, true));
    REGISTER_TENSOR(bias , Tensor::shape({numFeatures_}, true));
  }
  GroupNorm::resetParameters();
}

void GroupNorm::resetParameters() {
  if (affine_) {
    weights_.data().fill_(1.f);
    bias_.data().fill_(0.f);
  }
}

Tensor GroupNorm::forward(Tensor &input) {
  assert(input.dim() == 4);
  return Function::groupNorm(input, weights_, bias_, num_groups_, eps_);
}

std::vector<Tensor *> GroupNorm::parameters() {
  if (affine_) {
    return {&weights_, &bias_};
  }
  return {};
}

std::vector<Tensor *> GroupNorm::states() {
  std::vector<Tensor *> ret;
  if (affine_) {
    ret.push_back(&weights_);
    ret.push_back(&bias_);
  }
  return ret;
}

void GroupNorm::zeroGrad() {
  if (affine_) {
    weights_.zeroGrad();
    bias_.zeroGrad();
  }
}
BatchNorm2D::BatchNorm2D(int32_t numFeatures, float eps, float momentum,
                         bool affine, bool trackRunningStats)
    : numFeatures_(numFeatures),
      eps_(eps),
      momentum_(momentum),
      affine_(affine),
      trackRunningStats_(trackRunningStats),
      numBatchesTracked_(0) {
  if (affine_) {
    REGISTER_TENSOR(weights , Tensor::shape({numFeatures_}, true));
    REGISTER_TENSOR(bias , Tensor::shape({numFeatures_}, true));
  }
  if (trackRunningStats_) {
    REGISTER_TENSOR(runningMean , Tensor::shape({numFeatures_}, true));
    REGISTER_TENSOR(runningVar , Tensor::shape({numFeatures_}, true));
  }

  BatchNorm2D::resetParameters();
}

Tensor BatchNorm2D::forward(Tensor &input) {
  assert(input.dim() == 4);
  if (training_ && trackRunningStats_) {
    numBatchesTracked_++;
  }

  bool bnTrain = training_ || !trackRunningStats_;
  return Function::batchNorm(input, runningMean_, runningVar_, weights_, bias_,
                             bnTrain, momentum_, eps_);
}

std::vector<Tensor *> BatchNorm2D::parameters() {
  if (affine_) {
    return {&weights_, &bias_};
  }
  return {};
}

std::vector<Tensor *> BatchNorm2D::states() {
  std::vector<Tensor *> ret({&runningMean_, &runningVar_});
  if (affine_) {
    ret.push_back(&weights_);
    ret.push_back(&bias_);
  }
  return ret;
}

void BatchNorm2D::resetParameters() {
  if (affine_) {
    weights_.data().fill_(1.f);
    bias_.data().fill_(0.f);
  }

  if (trackRunningStats_) {
    runningMean_.data().fill_(0.f);
    runningVar_.data().fill_(1.f);
    numBatchesTracked_ = 0;
  }
}

void BatchNorm2D::zeroGrad() {
  if (affine_) {
    weights_.zeroGrad();
    bias_.zeroGrad();
  }
}

}  // namespace TinyTorch::nn
