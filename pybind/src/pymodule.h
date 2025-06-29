#include "Torch.h"
#include "pybind11/pybind11.h"
using namespace TinyTorch;

namespace TinyTorch::nn {
class PyModule : public nn::Module {
public:
    using Module::Module;
    Tensor forward(Tensor& x) override {
        PYBIND11_OVERRIDE_PURE(
            Tensor,
            Module,
            forward,
            x
        );
    }
    Tensor forward(std::vector<Tensor> &x) override {
        PYBIND11_OVERRIDE_PURE(
            Tensor, Module, forward, x);
    }
    std::vector<Tensor> multi_return_forward(Tensor &x) override {
        PYBIND11_OVERRIDE_PURE(
            std::vector<Tensor>, Module, multi_forward, x);
    }
    std::vector<Tensor> multi_return_forward(std::vector<Tensor> &x) override {
        PYBIND11_OVERRIDE_PURE(
            std::vector<Tensor>, Module, multi_forward, x);
    }

};
}