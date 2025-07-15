#include "pybind11/pybind11.h"
#include "pybind11/stl.h"
#include "pybind11/numpy.h"
#include <pybind11/operators.h>
#include "Torch.h"
#include "pymodule.h"
#include "Init.h"
#include "bind_init.cpp"
using namespace TinyTorch;
namespace py = pybind11;

std::string device_to_string(Device device) {
    switch (device) {
        case Device::CPU: return "cpu";
        case Device::CUDA: return "cuda";
        default: return "unknown";
    }
}

std::string type_to_string(Dtype type) {
    switch (type) {
        case Dtype::float32: return "float32";
        case Dtype::float16: return "float16";
        case Dtype::bfloat16: return "bfloat16";
        case Dtype::float8_e4m3: return "float8_e4m3";
        case Dtype::float8_e5m2: return "float8_e5m2";
        default: return "unknown";
    }
}
inline Device device_from_string(const std::string& str) {
    if (str == "cpu" || str == "CPU") return Device::CPU;
    if (str == "cuda" || str == "CUDA") return Device::CUDA;
    throw std::runtime_error("Unknown device: " + str);
}
inline Dtype dtype_from_string(const std::string& str) {
    if (str == "float32") return Dtype::float32;
    if (str == "float16") return Dtype::float16;
    if (str == "bfloat16") return Dtype::bfloat16;
    throw std::runtime_error("Unknown dtype: " + str);
}
namespace pybind11::detail {
    template<> struct type_caster<Dtype> {
        PYBIND11_TYPE_CASTER(Dtype, _("Dtype"));
        bool load(py::handle src, bool) {
            if (!src || !PyUnicode_Check(src.ptr())) return false;
            try {
                value = dtype_from_string(py::str(src));
                return true;
            } catch (...) { return false; }
        }
        static handle cast(Dtype src, return_value_policy, handle) {
            return py::str(src == Dtype::float32 ? "float32" :
            src == Dtype::float16 ? "float16" : "bfloat16").release();
        }
        };
    template<> struct type_caster<Device> {
        PYBIND11_TYPE_CASTER(Device, _("Device"));
        bool load(py::handle src, bool) {
            if (!src || !PyUnicode_Check(src.ptr())) return false;
            try {
                value = device_from_string(py::str(src));
                return true;
            } catch (...) { return false; }
        }
        static handle cast(Device src, return_value_policy, handle) {
            return py::str(src == Device::CPU ? "cpu" : "cuda").release();
        }
    };
}
py::array_t<float> numpy(const Tensor& tensor) {
    bool can_direct =
        tensor.device() == Device::CPU &&
        tensor.type() == Dtype::float32;
    if (can_direct) {
        const float* data_ptr = tensor.data().data<float>();
        const Shape& shape = tensor.shape();
        std::vector<size_t> dims;
        for (auto dim : shape) {
            dims.push_back(static_cast<size_t>(dim));
        }
        std::vector<size_t> strides;
        size_t stride = sizeof(float);
        for (int32_t i = shape.size() - 1; i >= 0; --i) {
            strides.insert(strides.begin(), stride);
            stride *= shape[i];
        }
        auto array = py::array_t<float>(dims);
        py::buffer_info buf = array.request();
        float* copy_data = static_cast<float*>(buf.ptr);

        size_t num_elements = tensor.numel();
        std::copy(data_ptr, data_ptr + num_elements, copy_data);
        return array;
    }
    else{
         std::ostringstream msg;
         msg << "If this tensor's device is GPU, Use .to('cpu') before .numpy(). \n"
                "If this tensor's type is not float32. Use .to('float32') before .numpy()";
         throw std::runtime_error(msg.str());
    }
  };

void bind_Tensor(py::module &m){
  py::class_<Tensor>(m, "Tensor")

      .def(py::init<>())

      .def(py::init([](py::array_t<float> array, bool requires_grad) {
            if (array.ndim() < 1)
                throw std::runtime_error("NumPy array must have at least 1 dimension");
            if (!array.dtype().is(py::dtype::of<float>())) {
                std::ostringstream msg;
                msg << "Expected float32 array, got " << py::cast<std::string>(py::str(array.dtype()));
                throw std::runtime_error(msg.str());
            }
            if (!array.flags()["C_CONTIGUOUS"])
                array = py::array_t<float>(array).attr("copy")();
            auto buffer = array.request();
            void* data_ptr = buffer.ptr;
            std::vector<int32_t> shape;
            for (int i = 0; i < buffer.ndim; ++i) {
                shape.push_back(buffer.shape[i]);
            }
            return Tensor(data_ptr, shape, requires_grad);
        }),
        py::arg("values"),
        py::arg("requires_grad") = false,
        "Create a Tensor from a NumPy array")

      .def(py::init<const Array1d&, bool>(),
           py::arg("values"), py::arg("requires_grad") = false)

      .def(py::init<const Array2d&, bool>(),
           py::arg("values"), py::arg("requires_grad") = false)

      .def(py::init<const Array3d&, bool>(),
           py::arg("values"), py::arg("requires_grad") = false)

      .def(py::init<const Array4d&, bool>(),
           py::arg("values"), py::arg("requires_grad") = false)

      .def(py::init<const Array5d&, bool>(),
           py::arg("values"), py::arg("requires_grad") = false)

      .def_property_readonly("shape",
                             [](const Tensor& t) {
                               const Shape& s = t.shape();
                               py::tuple shape_tuple(s.size());
                               for (size_t i = 0; i < s.size(); ++i) {
                                 shape_tuple[i] = s[i];
                               }
                               return shape_tuple;
                             },
                             "Get the tensor's shape as a tuple")

      .def_static("empty", [](const Shape &shape, bool requires_grad) {
        return Tensor::shape(shape, requires_grad);
      }, py::arg("shape"), py::arg("requires_grad") = false, "Create a tensor with the specified shape")

      .def_static("scalar", [](float value, bool requires_grad) {
        return Tensor::scalar(value, requires_grad);
      }, py::arg("value"), py::arg("requires_grad") = false)

      .def_static("scalar", [](float value, bool requires_grad, Device device, Dtype dtype) {
        return Tensor::scalar(value, requires_grad, device, dtype);
      }, py::arg("value"), py::arg("requires_grad"), py::arg("device"), py::arg("dtype"))

      .def_static("ones", &Tensor::ones,
                  py::arg("shape"), py::arg("requires_grad") = false)

      .def_static("ones_like", &Tensor::onesLike,
                  py::arg("t"), py::arg("requires_grad") = false)

      .def_static("zeros", &Tensor::zeros,
                  py::arg("shape"), py::arg("requires_grad") = false)

      .def_static("randn", &Tensor::randn,
                  py::arg("shape"), py::arg("requires_grad") = false)

      .def_static("arange", &Tensor::arange,
                  py::arg("start"), py::arg("stop"), py::arg("step"),
                  py::arg("requires_grad") = false)

      .def_static("linspace", &Tensor::linspace,
                  py::arg("start"), py::arg("end"), py::arg("steps"),
                  py::arg("requires_grad") = false)

      .def("dim", &Tensor::dim)
      .def("numel", &Tensor::numel)

      .def("device", [](const Tensor& self) {
            return device_to_string(self.device());
        }, "Get device as string")

      .def("type", [](const Tensor& self) {
            return type_to_string(self.type());
        }, "Get type as string")

      .def("item", &Tensor::item, "Get the element when item is 1")

      .def_static("reshape", [](const Tensor &t, const Shape& shape) {
            return Tensor::reshape(t, shape);
        }, py::arg("tensor"), py::arg("shape"))

      .def("view", [](Tensor& self, const Shape& shape) {
           return self.reshape(shape);
        }, py::arg("shape"))

      .def("flatten", [](Tensor& self, int32_t start, int32_t end){
           return self.flatten(start , end);
       }, py::arg("start"), py::arg("end"))

      .def("to", [](Tensor& self, const std::string& device_str) {
            if (device_str == "cpu") {
                return self.to(Device::CPU);
            } else if (device_str == "cuda") {
                return self.to(Device::CUDA);
            } else {
                throw std::invalid_argument("UnKnow Device: " + device_str);
            }
        }, py::arg("device"))

     .def("numpy", &numpy, "change Tensor to numpy")
     .def(py::self + py::self)              // __add__
     .def(py::self + float())                // __add__ (scalar)
     .def(float() + py::self)                // __radd__

     .def(py::self - py::self)               // __sub__
     .def(py::self - float())                // __sub__ (scalar)
     .def(float() - py::self)                // __rsub__

     .def(py::self * py::self)               // __mul__
     .def(py::self * float())                // __mul__ (scalar)
     .def(float() * py::self)                // __rmul__

     .def(py::self / py::self)               // __truediv__
     .def(py::self / float())                // __truediv__ (scalar)
     .def(float() / py::self)               // __rtruediv__

     .def(py::self += py::self)
     .def(py::self += float())
     .def(py::self -= py::self)
     .def(py::self -= float())
     .def(py::self *= py::self)
     .def(py::self *= float())
     .def(py::self /= py::self)
     .def(py::self /= float())
     /* ================ Trigonometric Functions ================ */
    .def("sin", &Tensor::sin, "Compute element-wise sine function")
    .def("cos", &Tensor::cos, "Compute element-wise cosine function")

    /* ================ Power Functions ================ */
    .def("pow", py::overload_cast<const float&>(&Tensor::pow, py::const_),
         py::arg("exponent"), "Raise tensor to scalar power")
    .def("pow", py::overload_cast<const Tensor&>(&Tensor::pow, py::const_),
         py::arg("exponent"), "Raise tensor to tensor power")
    .def("sqrt", [](const Tensor& self) {
        return self.pow(0.5f);
    }, "Compute element-wise square root")
    /* ================ Statistical Functions ================ */
    .def("sum", &Tensor::sum, "Compute sum of all elements")
    .def("mean", &Tensor::mean, "Compute mean of all elements")
    .def("var", &Tensor::var, "Compute variance of all elements")

    /* ================ Dimension Operations ================ */
    .def("squeeze", py::overload_cast<int32_t>(&Tensor::squeeze, py::const_),
         py::arg("dim") = -1, "Remove dimension(s) of size 1")

    .def("unsqueeze", &Tensor::unsqueeze, py::arg("dim"),
         "Insert singleton dimension at specified position")
    /* ================ Tensor Grad Function ================ */
    .def("isRequiresGrad", &Tensor::isRequiresGrad, "is require grad")
    .def_property("requires_grad",
            [](const Tensor& self) {
                return self.isRequiresGrad();
            },
            [](Tensor& self, bool require) {
                self.setRequiresGrad(require);
            },
            "Whether this tensor requires gradient computation")

    .def("backward", &Tensor::backward, py::arg("grad") = Tensor(),
             "Compute gradients of the tensor with respect to graph leaves");

}

void bindSize2D(py::module &m) {
    py::class_<Size2D>(m, "Size2D")
        .def(py::init<int32_t>()) // 单参数构造
        .def(py::init<int32_t, int32_t>()) // 双参数构造
        .def_readwrite("h", &Size2D::h)
        .def_readwrite("w", &Size2D::w)
        .def("__repr__", [](const Size2D &s) {
            return "Size2D(h=" + std::to_string(s.h) + ", w=" + std::to_string(s.w) + ")";
        });

    py::implicitly_convertible<py::int_, Size2D>();
    py::implicitly_convertible<py::tuple, Size2D>();
}

py::module_ create_nn_submodule(py::module_ &m) {
  auto nn = m.def_submodule("nn", "Neural Network Module");
  // 绑定基类 Module
  py::class_<nn::Module, nn::PyModule, std::shared_ptr<nn::Module>>(nn, "Module")
      .def(py::init<>())
      .def("__call__", py::overload_cast<Tensor&>(&nn::Module::forward), "Single input forward")
      .def("forward", py::overload_cast<Tensor&>(&nn::Module::forward), "Single input forward")
      .def("forward", py::overload_cast<std::vector<Tensor>&>(&nn::Module::forward), "Vector input forward")
      .def("multi_return_forward", py::overload_cast<Tensor&>(&nn::Module::multi_return_forward),
           "Single input with multiple outputs")
      .def("multi_return_forward", py::overload_cast<std::vector<Tensor>&>(&nn::Module::multi_return_forward),
           "Vector input with multiple outputs")

      .def("parameters", [](nn::Module& m) {
        return m.parameters();
      }, py::return_value_policy::reference_internal, "Get the parameters of the module")
      .def("zeroGrad", &nn::Module::zeroGrad)
      .def("name", &nn::Module::name)
      .def("to", py::overload_cast<Device>(&nn::Module::to))
      .def("to", py::overload_cast<Dtype>(&nn::Module::to))
      .def("to", py::overload_cast<Device>(&nn::Module::to), py::arg("device"))
      .def("to", py::overload_cast<Dtype>(&nn::Module::to), py::arg("dtype"))
      .def("registerModules", &nn::Module::registerModules)
      .def("eval", &nn::Module::eval)
      .def("train", &nn::Module::train, py::arg("mode") = true, "Set training mode")
      .def("load_state_dict_", &nn::Module::load_state_dict, py::arg("param_dict"), py::arg("device"))
      .def("set_name", &nn::Module::set_name, py::arg("name"))
      .def("summary", [](const nn::Module& self) {
         py::print(self.getTopologyText());
      });
  py::class_<nn::Conv2D, nn::Module, std::shared_ptr<nn::Conv2D>>(nn, "Conv2d")
      .def(py::init<int, int, Size2D, Size2D, Size2D, bool>(),
           py::arg("in_channels"), py::arg("out_channels"),
           py::arg("kernel_size") , py::arg("stride") = 1,
           py::arg("padding") = 0, py::arg("bias") = true,
           R"pbdoc(
             2D convolution layer
             Parameters:
                 in_channels: Number of input channels
                 out_channels: Number of output channels
                 kernel_size: Size of the convolving kernel
                 stride: Stride of the convolution (default=1)
             )pbdoc")
      .def("forward", &nn::Conv2D::forward)
      .def_property_readonly("weight", [](nn::Conv2D &self) {
        return self.weights();
      }, "Get convolution weights")
      .def_property_readonly("bias", [](nn::Conv2D &self) {
        return self.bias();
      }, "Get convolution bias");

  py::class_<nn::Dropout, nn::Module, std::shared_ptr<nn::Dropout>>(nn, "Dropout")
      .def(py::init<double>(), py::arg("p") = 0.5)
      .def("forward", &nn::Dropout::forward);

  py::class_<nn::BatchNorm2D, nn::Module, std::shared_ptr<nn::BatchNorm2D>>(nn, "BatchNorm2d")
      .def(py::init<int, float, float, bool, bool>(), py::arg("numFeatures"), py::arg("eps") = 1e-5,
           py::arg("momentum") = 0.1f, py::arg("affine") = true, py::arg("trackRunningStats") = true)
      .def("forward", &nn::BatchNorm2D::forward)
      .def_property_readonly("weight", [](nn::BatchNorm2D &self) {
           return self.weights();
       }, "Get convolution weights")
      .def_property_readonly("bias", [](nn::BatchNorm2D &self) {
           return self.bias();
      }, "Get convolution bias");

  py::class_<nn::GroupNorm, nn::Module, std::shared_ptr<nn::GroupNorm>>(nn, "GroupNorm")
      .def(py::init<int, float, int, bool>(), py::arg("numFeatures"), py::arg("eps") = 1e-5
            ,py::arg("group_nums") = 1, py::arg("affine") = true)
      .def("forward", &nn::GroupNorm::forward);

  py::class_<nn::Linear, nn::Module, std::shared_ptr<nn::Linear>>(nn, "Linear")
      .def(py::init<int, int>(), py::arg("in_channels"), py::arg("out_channels"))
      .def("forward", &nn::Linear::forward)
      .def_property_readonly("weight", [](nn::Linear &self) {
        return self.weights();
      }, "Get Linear weights")
      .def_property_readonly("bias", [](nn::Linear &self) {
        return self.bias();
      }, "Get Linear bias");

  py::class_<nn::Relu, nn::Module, std::shared_ptr<nn::Relu>>(nn, "ReLU")
      .def(py::init())
      .def("forward", &nn::Relu::forward);

  py::class_<nn::MaxPool2D, nn::Module, std::shared_ptr<nn::MaxPool2D>>(nn, "MaxPool2d")
      .def(py::init<Size2D, Size2D, Size2D>(),
           py::arg("kernel_size") , py::arg("stride"),
           py::arg("padding") = 0,
           R"pbdoc(
            Applies a 2D max pooling over an input signal composed of several input planes.

            Parameters:
                kernel_size (Size2D): Size of the pooling window.
                    Can be a single integer (for square kernel) or tuple (height, width).
                stride (Size2D, optional): Stride of the pooling operation.
                    Defaults to kernel_size if None.
                padding (Size2D, optional): Zero-padding added to both sides of the input.
                    Can be a single integer or tuple (height, width). Default: 0.

            Shape:
                - Input: (N, C, H_in, W_in)
                - Output: (N, C, H_out, W_out)
                Where:
                    H_out = floor((H_in + 2*padding[0] - kernel_size[0])/stride[0] + 1)
                    W_out = floor((W_in + 2*padding[1] - kernel_size[1])/stride[1] + 1)

            Examples:
                >>> pool = nn.MaxPool2d(kernel_size=3, stride=2)
                >>> input = tt.randn(1, 1, 32, 32)
                >>> output = pool(input)
             )pbdoc")
      .def("forward", &nn::MaxPool2D::forward);

  py::class_<nn::AvgPool2D, nn::Module, std::shared_ptr<nn::AvgPool2D>>(nn, "AvgPool2d")
      .def(py::init<Size2D, Size2D, Size2D>(),
           py::arg("kernel_size") , py::arg("stride"),
           py::arg("padding") = 0)
      .def("forward", &nn::AvgPool2D::forward);

  py::class_<nn::AdaptiveAvgPool2D, nn::Module, std::shared_ptr<nn::AdaptiveAvgPool2D>>(nn, "AdaptiveAvgPool2d")
        .def(py::init([](py::object arg) {
            if (py::isinstance<py::int_>(arg)) {
                int32_t size = arg.cast<int32_t>();
                return nn::AdaptiveAvgPool2D(Size2D(size, size));
            } else if (py::isinstance<py::tuple>(arg)) {
                py::tuple t = arg.cast<py::tuple>();
                if (t.size() != 2) {
                    throw std::runtime_error("Tuple must contain exactly two elements");
                }
                return nn::AdaptiveAvgPool2D(Size2D(
                    t[0].cast<int32_t>(),
                    t[1].cast<int32_t>()
                ));
            } else {
                throw std::runtime_error("Invalid argument type for AdaptiveAvgPool2d");
            }
        }), py::arg("output_size"))
        .def(py::init([](int32_t h, int32_t w) {
            return nn::AdaptiveAvgPool2D(Size2D(h, w));
        }), py::arg("h"), py::arg("w"))
        .def("forward", &nn::AdaptiveAvgPool2D::forward);

  py::class_<nn::Sequential, nn::Module, std::shared_ptr<nn::Sequential>>(nn, "Sequential")
        .def(py::init<>())
        .def(py::init<std::initializer_list<std::shared_ptr<nn::Module>>>())
        .def(py::init([](py::args args) {
            if (args.size() == 1 && py::isinstance<py::list>(args[0])) {
              std::vector<std::shared_ptr<nn::Module>> modules;
              for (auto handle : args[0]) {
                modules.push_back(handle.cast<std::shared_ptr<nn::Module>>());
              }
              return nn::Sequential(modules);
            }
            std::vector<std::shared_ptr<nn::Module>> modules;
              for (auto handle : args) {
                  modules.push_back(handle.cast<std::shared_ptr<nn::Module>>());
              }
            return nn::Sequential(modules);
        }))
        .def("forward", &nn::Sequential::forward, py::arg("input"))
        .def("size", &nn::Sequential::getsize)
         .def("push_back",
        static_cast<void (nn::Sequential::*)(const std::shared_ptr<nn::Module>&)>(
            &nn::Sequential::pushBack),
        py::arg("module"))
        .def("__len__", &nn::Sequential::getsize)
        .def("__getitem__", [](nn::Sequential& self, int index) -> nn::Module& {
            if (index < 0 || index >= self.getsize())
                throw py::index_error();
            return self[index];
        }, py::return_value_policy::reference_internal)
        .def("__getitem__", [](const nn::Sequential& self,
                              const nn::Sequential::Slice& slice) {
            return self[slice];
        })
        .def("__iter__", [](nn::Sequential& self) {
            return py::make_iterator(self.begin(), self.end());
        }, py::keep_alive<0, 1>())
        .def("__repr__", [](const nn::Sequential& self) {
            return self.getTopologyText();
        });
  return nn;
}

py::module_ create_optim_submodule(py::module_ &m) {
  auto optim = m.def_submodule("optim", "Optimization Module");
  py::class_<optim::Optimizer>(optim, "Optimizer")
      .def("step", &optim::Optimizer::step, "Perform a single optimization step")
      .def("zero_grad", &optim::Optimizer::zeroGrad, "Clear gradients")
      .def("get_lr", &optim::Optimizer::getLr, "Get learning rate")
      .def("set_lr", &optim::Optimizer::setLr, py::arg("lr"), "Set learning rate");

  py::class_<optim::SGD, optim::Optimizer>(optim, "SGD")
      .def(py::init<std::vector<Tensor*>&&, float, float, float, float, bool>(),
           py::arg("parameters"),
           py::arg("lr") = 0.001f,
           py::arg("momentum") = 0.0f,
           py::arg("dampening") = 0.0f,
           py::arg("weight_decay") = 0.0f,
           py::arg("nesterov") = false);

  py::class_<optim::Adam, optim::Optimizer>(optim, "Adam")
      .def(py::init<std::vector<Tensor*>&&, float, const std::pair<float, float>&, float, float, bool>(),
           py::arg("parameters"),
           py::arg("lr") = 0.001f,
           py::arg("betas") = std::make_pair(0.9f, 0.999f),
           py::arg("eps") = 1e-8f,
           py::arg("weight_decay") = 0.0f,
           py::arg("amsgrad") = false);

    py::class_<optim::AdaDelta, optim::Optimizer>(optim, "AdaDelta")
      .def(py::init<std::vector<Tensor*>&&, float, float, float, float>(),
           py::arg("parameters"),
           py::arg("lr") = 1.0f,
           py::arg("rho") = 0.9f,
           py::arg("eps") = 1e-6f,
           py::arg("weightDecay") = 0.f);

  py::class_<optim::AdamW, optim::Optimizer>(optim, "AdamW")
      .def(py::init<std::vector<Tensor*>&&, float, const std::pair<float, float>&, float, float, bool>(),
           py::arg("parameters"),
           py::arg("lr") = 0.001f,
           py::arg("betas") = std::make_pair(0.9f, 0.999f),
           py::arg("eps") = 1e-8f,
           py::arg("weight_decay") = 0.01f,
           py::arg("amsgrad") = false);

  return optim;
}


void create_lr_scheduler_submodule(py::module_ &optim){
    auto lr_scheduler = optim.def_submodule("lr_scheduler", "LR scheduler for optimization");
    py::class_<optim::lr_scheduler::StepLR>(lr_scheduler, "StepLR")
      .def(py::init<optim::Optimizer &, int32_t, float, int32_t>(),
           py::arg("Optimizer"), py::arg("stepSize"), py::arg("gamma") = 0.1, py::arg("lastEpoch") = -1)
      .def("getLr", &optim::lr_scheduler::StepLR::getLr);

}
void create_functional_submodule(py::module_ &nn) {
  auto functional = nn.def_submodule("functional", "Neural Network Functional Operations");
  functional.def("relu", &Function::relu);
  functional.def("mseloss", [](const Tensor& input,
                               const Tensor& target,
                               const std::string& reduction) {
                         if (reduction == "mean")
                           return Function::mseloss(input, target, LossReduction::MEAN);
                         else if (reduction == "none")
                           return Function::mseloss(input, target, LossReduction::NONE);
                         else if(reduction == "sum")
                           return Function::mseloss(input, target, LossReduction::SUM);
                          },py::arg("input"),
                            py::arg("target"),
                            py::arg("reduction") = 1
                          );

  functional.def("nllloss", [](const Tensor& input,
                               const Tensor& target,
                               const std::string& reduction = "mean") {
                         if (reduction == "mean")
                           return Function::nllloss(input, target, LossReduction::MEAN);
                         else if (reduction == "none")
                           return Function::nllloss(input, target, LossReduction::NONE);
                         else if(reduction == "sum")
                           return Function::nllloss(input, target, LossReduction::SUM);
                          },py::arg("input"),
                            py::arg("target"),
                            py::arg("reduction") = "mean"
                          );
  functional.def("log_softmax", &Function::logSoftmax, py::arg("input"), py::arg("dim"));
  functional.def("change_type", &Function::changetype, py::arg("input"), py::arg("dtype"));
  functional.def("max_pool2d",
                 [](const Tensor& input,
                    Size2D kernelSize,
                    py::object strideObj,
                    Size2D padding) {
                   std::optional<Size2D> stride;
                   if (!strideObj.is_none()) {
                     stride = strideObj.cast<Size2D>();
                   }
                   return Function::maxPool2d(input, kernelSize, stride, padding);
                 },
                 py::arg("input"),
                 py::arg("kernel_size"),
                 py::arg("stride") = py::none(),
                 py::arg("padding") = 0
  );
  functional.def("flatten", &Function::flatten,
                 py::arg("input"), py::arg("start_dim") = 0, py::arg("end_dim") = -1);
}

py::module_ create_data_submodule(py::module_ &m) {
  auto data = m.def_submodule("data", "Data Module");
  py::class_<data::DataLoader, std::shared_ptr<data::DataLoader>>(data, "DataLoader")
      .def(py::init<const std::shared_ptr<data::Dataset>&, size_t, bool, bool>(),
              py::arg("dataset"), py::arg("batchSize"), py::arg("shuffle") = true
      , py::arg("dropLast") = false)
      .def("__iter__", [](const data::DataLoader& loader) {
        return py::make_iterator(loader.begin(), loader.end());
      }, py::keep_alive<0, 1>())
      .def("__len__", &data::DataLoader::size, "Number of batches")
      .def("batch_size", &data::DataLoader::batchSize, "Get batch size")
      .def("dataset", &data::DataLoader::dataset, "Get underlying dataset");
  return data;
}

void create_dataset_submodule(py::module_ &m) {
    py::class_<data::Dataset, std::shared_ptr<data::Dataset>>(m, "DatasetBase")
      .def("size", &data::Dataset::size,
           "Get the size of the dataset")
      .def("__len__", &data::Dataset::size,
           "Get the size of the dataset")
      .def("get_item", &data::Dataset::getItem, py::arg("index"),
           "Get an item by index")
      .def("__getitem__", &data::Dataset::getItem, py::arg("index"),
           "Get an item by index");

    py::class_<data::PyDataset, data::Dataset, std::shared_ptr<data::PyDataset>>(m, "Dataset")
        .def(py::init<>())
        .def("size", &data::Dataset::size,
             "Get the size of the dataset")
        .def("__len__", &data::Dataset::size,
             "Get the size of the dataset")
        .def("get_item", &data::Dataset::getItem, py::arg("index"),
             "Get an item by index")
        .def("__getitem__", &data::Dataset::getItem, py::arg("index"),
             "Get an item by index");

   // MNIST for example
    py::enum_<data::DatasetMNIST::MnistDataType>(m, "MnistDataType")
        .value("TRAIN", data::DatasetMNIST::MnistDataType::TRAIN)
        .value("TEST", data::DatasetMNIST::MnistDataType::TEST)
        .export_values();

    py::class_<data::DatasetMNIST, data::Dataset, std::shared_ptr<data::DatasetMNIST>>(m, "DatasetMNIST")
        .def(py::init([](const std::string& dir,
                         const std::string& type_str,
                         const std::shared_ptr<data::transforms::Transform>& transform) {
               data::DatasetMNIST::MnistDataType type;
               std::string upper_type = type_str;
               std::transform(upper_type.begin(), upper_type.end(), upper_type.begin(), ::toupper);
               if (upper_type == "TRAIN") {
                 type = data::DatasetMNIST::MnistDataType::TRAIN;
               } else if (upper_type == "TEST") {
                 type = data::DatasetMNIST::MnistDataType::TEST;
               } else {
                 throw std::invalid_argument(
                     "Invalid type argument: '" + type_str +
                     "'. Must be 'train' or 'test' (case insensitive)."
                 );
               }
               return new data::DatasetMNIST(dir, type, transform);
             }),
             py::arg("dir"), py::arg("type"), py::arg("transform"),
             R"doc(
    MNIST Dataset Constructor
    Args:
        dir (str): Directory containing MNIST data files
        type (str): Dataset type - 'train' or 'test'
        transform (Transform): Transform to apply to images
    )doc")
        .def("size", &data::Dataset::size,
             "Get the size of the dataset")
        .def("__len__", &data::Dataset::size,
             "Get the size of the dataset")
        .def("get_item", &data::Dataset::getItem, py::arg("index"),
             "Get an item by index")
        .def("__getitem__", &data::Dataset::getItem, py::arg("index"),
             "Get an item by index");
}

void create_transforms_submodule(py::module_ &data){
    auto transforms = data.def_submodule("transforms", "Transforms for data");

    py::class_<data::transforms::Transform, std::shared_ptr<data::transforms::Transform>>(transforms, "Transform")
        .def(py::init<>())
        .def("process",
         (Tensor (data::transforms::Transform::*)(Tensor&) const)
             &data::transforms::Transform::process,
         py::arg("input"))
        .def("__call__",
         (Tensor (data::transforms::Transform::*)(Tensor&) const)
             &data::transforms::Transform::process,
         py::arg("input"));

    py::class_<data::transforms::Compose,
           data::transforms::Transform,
           std::shared_ptr<data::transforms::Compose>>(transforms, "Compose")
        .def(py::init<>())
        .def(py::init([](py::args args) {
            std::vector<std::shared_ptr<data::transforms::Transform>> transforms;
                for (py::handle handle : args) {
                    try {
                        transforms.emplace_back(
                            handle.cast<std::shared_ptr<data::transforms::Transform>>()
                        );
                    }
                    catch (const py::cast_error&) {
                        try {
                            data::transforms::Transform& transform =
                                handle.cast<data::transforms::Transform&>();
                            transforms.emplace_back(
                                std::make_shared<data::transforms::Transform>(transform)
                            );
                        }
                        catch (const py::cast_error& e) {
                            throw std::runtime_error(
                                "All arguments must be Transform instances or shared pointers to Transform. "
                                "Error: " + std::string(e.what())
                            );
                        }
                    }
                }
                auto compose = std::make_shared<data::transforms::Compose>();
                for (auto& transform : transforms) {
                    compose->pushBack(transform);
                }
                return compose;
            }),
            R"doc(
            Create a Compose from any number of transform arguments

            Args:
                *args: Transform instances or shared pointers to Transform

            Example:
                transform1 = Resize(224, 224)
                transform2 = ToTensor()
                transform3 = Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])

                compose = Compose(transform1, transform2, transform3)
            )doc")
        .def("push_back",
             [](data::transforms::Compose& self,
                const std::shared_ptr<data::transforms::Transform>& transform) {
                 self.pushBack(transform);
             }, py::arg("transform"),
             "Add a transform using shared pointer")
        .def("push_back",
             [](data::transforms::Compose& self,
                const data::transforms::Transform& transform) {
                 auto copy = std::make_shared<std::decay_t<decltype(transform)>>(transform);
                 self.pushBack(copy);
             }, py::arg("transform"),
             "Add a transform by value (creates a copy)")
        .def("__call__",
             (Tensor (data::transforms::Compose::*)(Tensor&) const)
                 &data::transforms::Compose::process,
             py::arg("input"),
             "Apply the transformation pipeline");

    py::class_<data::transforms::Normalize,
               data::transforms::Transform,  // 继承自 Transform
               std::shared_ptr<data::transforms::Normalize>>(transforms, "Normalize")
        .def(py::init<float, float>(),
             py::arg("mean"), py::arg("std"),
             "Normalize transform with given mean and standard deviation\n\n"
             "Args:\n"
             "    mean (float): Mean value\n"
             "    std (float): Standard deviation")
        .def("process", &data::transforms::Normalize::process,
             py::arg("input"),
             "Apply normalization to input tensor");

}

void bindDeviceDtype(py::module &m) {
    py::enum_<Device>(m, "Device")
        .value("CPU", Device::CPU)
        .value("CUDA", Device::CUDA)
        .value("cpu", Device::CPU)
        .value("cuda", Device::CUDA);

    py::enum_<Dtype>(m, "Dtype")
        .value("float32", Dtype::float32)
        .value("float16", Dtype::float16)
        .value("bfloat16", Dtype::bfloat16);
    py::implicitly_convertible<std::string, Device>();
    py::implicitly_convertible<std::string, Dtype>();
}

PYBIND11_MODULE(pytt, m) {
    m.doc() = "A deep learning framework for faster vision task";
    bindDeviceDtype(m);
    bind_Tensor(m);
    bindSize2D(m);
    auto nn = create_nn_submodule(m);
    create_functional_submodule(nn);
    auto optim = create_optim_submodule(m);
    create_lr_scheduler_submodule(optim);
    auto data = create_data_submodule(m);
    create_transforms_submodule(data);
    create_dataset_submodule(data);
    create_init_submodule(nn);
}