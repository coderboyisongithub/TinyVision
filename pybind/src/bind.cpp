#include "pybind11/pybind11.h"
#include "pybind11/stl.h"
#include "pybind11/numpy.h"
#include "Torch.h"
#include "pymodule.h"
using namespace TinyTorch;
namespace py = pybind11;

PYBIND11_MODULE(pytt, m) {
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
                  py::arg("requires_grad") = false);
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
}
void bindMaxPool2d(py::module &m) {
    m.def("max_pool2d",
        [](const Tensor& input, const Size2D& kernelSize,
           py::object strideObj, const Size2D& padding) {
            std::optional<Size2D> stride;
            if (!strideObj.is_none()) {
                stride = strideObj.cast<Size2D>();
            }
            return Function::maxPool2d(input, kernelSize, stride, padding);
        },
        py::arg("input"),
        py::arg("kernel_size"),
        py::arg("stride") = py::none(),
        py::arg("padding") = Size2D(0),
        R"pbdoc(
        Applies 2D max pooling over an input signal composed of several input planes.
        Args:
            input: Input tensor of shape (N, C, H_in, W_in)
            kernel_size: Size of the pooling window (height, width)
            stride: Stride of the pooling window. Default is kernel_size
            padding: Padding added to all four sides of the input. Default is 0
        Returns:
            Tensor of shape (N, C, H_out, W_out)
        )pbdoc"
    );
}

PYBIND11_MODULE(pytt, m) {
    m.doc() = "My custom deep learning framework";

    bindSize2D(m);
    py::enum_<Device>(m, "Device")
        .value("CPU", Device::CPU)
        .value("CUDA", Device::CUDA)
        .value("cpu", Device::CPU)
        .value("cuda", Device::CUDA);

    py::enum_<Dtype>(m, "Dtype")
        .value("float32", Dtype::float32)
        .value("float16", Dtype::float16)
        .value("bfloat16", Dtype::bfloat16);

    py::class_<nn::Module, nn::PyModule>(m, "Module")
        .def(py::init<>())
        .def("forward", &nn::Module::forward)
        .def("to", py::overload_cast<Device>(&nn::Module::to))
        .def("to", py::overload_cast<Dtype>(&nn::Module::to))
        .def("registerModules", &nn::Module::registerModules)
        .def("eval", &nn::Module::eval)
        .def("train", &nn::Module::train);

    py::class_<nn::Conv2D, nn::Module>(m, "Conv2D")
        .def(py::init<int, int, int, int>());

    py::class_<nn::Dropout, nn::Module>(m, "Dropout")
        .def(py::init<double>());

    py::class_<nn::Linear, nn::Module>(m, "Linear")
        .def(py::init<int, int>());

    m.def("relu", &Function::relu);
    m.def("logSoftmax", &Function::logSoftmax, py::arg("input"), py::arg("dim"));
    m.def("changetype", &Function::changetype, py::arg("input"), py::arg("dtype"));
    bindMaxPool2d(m);
    m.def("flatten", &Function::flatten, py::arg("input"), py::arg("start_dim"), py::arg("end_dim"));
}