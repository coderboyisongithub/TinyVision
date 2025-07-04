#pragma once
#include "../Tensor.h"

namespace TinyTorch::nn {
class Module {
 public:
  virtual ~Module() = default;
  virtual std::vector<Tensor *> parameters();
  virtual std::vector<Tensor *> states();
  virtual void resetParameters();
  virtual void zeroGrad();
  virtual std::string name() const { return "Module"; }
  virtual Tensor forward(Tensor &x) { return {}; }
  virtual Tensor forward(Tensor &x, Tensor &a) { return {}; }
  virtual Tensor forward(std::vector<Tensor> &x) { return {}; }
  virtual std::vector<Tensor> multi_return_forward(Tensor &x1) { return {}; }
  virtual std::vector<Tensor> multi_return_forward(std::vector<Tensor> &x) { return {}; }
  template <typename... Args>
  Tensor operator()(Args &&...args) {
    return forward(std::forward<Args>(args)...);
  }
  void registerModules(
      const std::vector<std::reference_wrapper<Module>> &modules) {
    subModules_.reserve(modules.size());
    for (auto module : modules) {
      subModules_.emplace_back(module.get());
    }
  }
  void registerModule(const std::reference_wrapper<Module> &module) {
    subModules_.push_back(module);
  }
  void to(Device device);
  void to(Dtype T);
  void eval() { train(false); }
  void train(bool mode = true) { setTraining(mode); }
  std::string getTopologyText() const {
            std::stringstream ss;
            getTopologyTextHelper(ss, 0);
            return ss.str();
        }
   virtual void getTopologyTextHelper(std::stringstream& ss, int depth) const;
 protected:

  virtual void setTraining(bool mode) { training_ = mode; }

  bool training_ = true;
  std::vector<std::reference_wrapper<Module>> subModules_;

};
}