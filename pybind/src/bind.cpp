#include "pybind11/pybind11.h"
#include "pybind11/stl.h"
#include "pybind11/numpy.h"
#include <pybind11/operators.h>
#include "Torch.h"
#include "pymodule.h"

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
  py::class_<nn::Module, nn::PyModule>(nn, "Module")
      .def(py::init<>())
      .def("__call__", py::overload_cast<Tensor&>(&nn::Module::forward), "Single input forward")
      .def("forward", py::overload_cast<Tensor&>(&nn::Module::forward), "Single input forward")
      .def("forward", py::overload_cast<std::vector<Tensor>&>(&nn::Module::forward), "Vector input forward")
      .def("multi_return_forward", py::overload_cast<Tensor&>(&nn::Module::multi_return_forward),
           "Single input with multiple outputs")
      .def("multi_return_forward", py::overload_cast<std::vector<Tensor>&>(&nn::Module::multi_return_forward),
           "Vector input with multiple outputs")
      .def("to", py::overload_cast<Device>(&nn::Module::to))
      .def("to", py::overload_cast<Dtype>(&nn::Module::to))
      .def("to", [](nn::Module& self, const std::string& target) {
        if (target == "cpu" || target == "CPU") {
          return self.to(Device::CPU);
        } else if (target == "cuda" || target == "CUDA") {
          return self.to(Device::CUDA);
        }
        if (target == "float32") {
          return self.to(Dtype::float32);
        } else if (target == "float16") {
          return self.to(Dtype::float16);
        } else if (target == "bfloat16") {
          return self.to(Dtype::bfloat16);
        }
        throw std::runtime_error("Unknown target for to(): " + target);
      }, py::arg("target"),
           R"pbdoc(
        Move module to specified device or change data type

        Parameters:
            target: Can be either a device ("cpu", "cuda") or a data type ("float32", "float16", "bfloat16")

        Examples:
            model.to("cuda")      # Move to CUDA device
            model.to("float16")   # Convert to float16
        )pbdoc")

      .def("registerModules", &nn::Module::registerModules)
      .def("eval", &nn::Module::eval)
      .def("train", &nn::Module::train);

  // 绑定具体模块
  py::class_<nn::Conv2D, nn::Module>(nn, "Conv2D")
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
      .def_property_readonly("weights", [](nn::Conv2D &self) {
        return self.weights();
      }, "Get convolution weights")
      .def_property_readonly("bias", [](nn::Conv2D &self) {
        return self.bias();
      }, "Get convolution bias");

  py::class_<nn::Dropout, nn::Module>(nn, "Dropout")
      .def(py::init<double>())
      .def("forward", &nn::Dropout::forward);

  py::class_<nn::Linear, nn::Module>(nn, "Linear")
      .def(py::init<int, int>())
      .def("forward", &nn::Linear::forward);

  return nn;
}

void create_functional_submodule(py::module_ &nn) {
  auto functional = nn.def_submodule("functional", "Neural Network Functional Operations");
  functional.def("relu", &Function::relu);
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

PYBIND11_MODULE(pytt, m) {
    m.doc() = "A deep learning framework for faster vision task";

    py::enum_<Device>(m, "Device")
        .value("CPU", Device::CPU)
        .value("CUDA", Device::CUDA)
        .value("cpu", Device::CPU)
        .value("cuda", Device::CUDA);

    py::enum_<Dtype>(m, "Dtype")
        .value("float32", Dtype::float32)
        .value("float16", Dtype::float16)
        .value("bfloat16", Dtype::bfloat16);

    bind_Tensor(m);
    bindSize2D(m);
    auto nn = create_nn_submodule(m);
    create_functional_submodule(nn);
}