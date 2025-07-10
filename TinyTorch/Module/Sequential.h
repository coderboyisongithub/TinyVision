#pragma once
#include "Module_base.h"

namespace TinyTorch::nn {

class Sequential : public Module {
 public:
  struct Slice {
    int start;
    int end;

    Slice(std::initializer_list<int> list) {
    if (list.size() != 2) {
      throw std::invalid_argument("Slice must have exactly two integers");
    }
    start = *list.begin();
    end = *(list.begin() + 1);
    }
  };

  template <typename... Modules,
            typename = std::enable_if_t<(std::is_same_v<std::decay_t<Modules>, std::shared_ptr<Module>> && ...)>>
  explicit Sequential(Modules &&...modules) {
    modules_.reserve(sizeof...(Modules));
    pushBack(std::forward<Modules>(modules)...);
  }
  explicit Sequential(std::vector<std::shared_ptr<Module>> modules) {
    modules_.reserve(modules.size());
    for (size_t i = 0; i < modules.size(); ++i) {
      auto& module = modules[i];
      std::string new_name = name() + "." + std::to_string(i) + "_" + module->name();
      module->set_name(new_name);
      modules_.emplace_back(std::move(module));
    }
  }
  Sequential(std::initializer_list<std::shared_ptr<Module>> modules) {
    modules_.reserve(modules.size());
    int index = modules_.size();
    for (const auto &module : modules) {
      std::string new_name = name() + "." + std::to_string(index++) + "_" + module->name();
      module->set_name(new_name);
      modules_.emplace_back(module);
    }
  }


  template <typename ModuleType>
  void pushBack(ModuleType &&module) {
    int index = modules_.size();
    std::string new_name = name() + "." + std::to_string(index++) + "_" + module->name();
    module->set_name(new_name);
    modules_.push_back(
        std::make_shared<ModuleType>(std::forward<ModuleType>(module)));
  }

  void pushBack(const std::shared_ptr<Module> &module) {
    int index = modules_.size();
    std::string new_name = name() + "." + std::to_string(index++) + "_" + module->name();
    module->set_name(new_name);
    modules_.emplace_back(module);
  }

  Tensor forward(Tensor &input) override;
  std::vector<Tensor *> parameters() override;
  std::vector<Tensor *> states() override;
  void resetParameters() override;
  void zeroGrad() override;
  size_t getsize() {return modules_.size();};
  Module &operator[](const int index) { return *modules_[index]; }
  Sequential operator[](const Slice& slice) const {
    if (slice.start < 0 || slice.start >= modules_.size()) {
      throw std::out_of_range("Start index out of range");
    }
    if (slice.end < 0 || slice.end > modules_.size()) {
      throw std::out_of_range("End index out of range");
    }
    if (slice.start > slice.end) {
      throw std::invalid_argument("Start index must be <= end index");
    }
    Sequential result;
    result.modules_.reserve(slice.end -  slice.start);
    for (int i = slice.start; i < slice.end; ++i) {
      result.pushBack(modules_[i]);
    }
    return result;
  }
  auto begin() { return modules_.begin(); }
    auto end() { return modules_.end(); }
    auto begin() const { return modules_.begin(); }
    auto end() const { return modules_.end(); }
 private:
  void setTraining(bool mode) override;
  template <typename First, typename Second, typename... Rest>
  void pushBack(First &&first, Second &&second, Rest &&...rest) {
    pushBack(std::forward<First>(first));
    pushBack(std::forward<Second>(second), std::forward<Rest>(rest)...);
  }
  void pushBack() {}
  std::vector<std::shared_ptr<Module>> modules_;
    void getTopologyTextHelper(std::stringstream& ss, int depth) const override {
        ss << std::string(depth * 2, ' ') << "|--Sequential" << std::endl;
        for (const auto& module : modules_) {
            module->getTopologyTextHelper(ss, depth + 1);
        }
    }
  std::string name_ = "Sequential";
};
}