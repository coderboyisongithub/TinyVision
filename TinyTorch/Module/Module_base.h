#pragma once
#include "../Tensor.h"

#define REGISTER_TENSOR(name, content)   \
    name##_ = content;                   \
    this->named_tensors_[#name] = &name##_;

namespace TinyTorch::nn {
class Module {
    // "name_" is no useful if you don't use set_name after register
    // because the "name_" is not a const variable and it's not implement virtual function,
    // when use Module.name() it returns "Module"
  #define REGISTER_MODULE_NAME(ClassName) \
    static constexpr auto _module_name = #ClassName; \
    std::string class_name() const override { return _module_name; } \
    std::string name_ = #ClassName;   \
    std::map<std::string, Tensor*> named_tensors_;\
    std::map<std::string, Tensor*> get_named_tensors_() const override{ \
        return named_tensors_;\
    }

  #define REGISTER_SEQMODULE_NAME(ClassName) \
      static constexpr auto _module_name = #ClassName; \
      std::string class_name() const override { return _module_name; } \
      std::string name_ = #ClassName;   \
      std::map<std::string, Tensor*> named_tensors_;\
      std::map<std::string, Tensor*> get_named_tensors_() const override;

 public:
  virtual ~Module() = default;
  virtual std::vector<Tensor *> parameters();
  virtual std::vector<Tensor *> states();
  virtual void resetParameters();
  virtual void zeroGrad();
  virtual std::string class_name() const = 0;
  std::string name() const { return name_; }
  void set_name(const std::string& name) { name_ = name; }
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
      std::string new_name = module.get().name();
      module.get().set_name(new_name);
      subModules_.emplace_back(module.get());
    }
  }
  void registerModule(const std::reference_wrapper<Module> &module) {
    std::string new_name =  module.get().name();
    module.get().set_name(new_name);
    subModules_.push_back(module);
  }
  void to(Device device);
  void to(Dtype T);
  void eval() { train(false); }
  void train(bool mode = true) { setTraining(mode); }

  // print Topology Graph
  std::string getTopologyText() const {
            std::stringstream ss;
            getTopologyTextHelper(ss, 0);
            return ss.str();
        }

  virtual void getTopologyTextHelper(std::stringstream& ss, int depth) const;

   // load param
  virtual Tensor* get_tensor(const std::string& full_name) {
        return named_tensors_[full_name];
    }
  void load(std::map<std::string, Tensor> param_dict, Device device = Device::CPU);
  virtual std::map<std::string, Tensor*> get_named_tensors_() const{
        return named_tensors_;
    }

  static constexpr auto _module_name = "Module";
  std::string name_ = "Module";
  std::map<std::string, Tensor*> named_tensors_;
 protected:

  virtual void setTraining(bool mode) { training_ = mode; }

  bool training_ = true;
  std::vector<std::reference_wrapper<Module>> subModules_;

};
}