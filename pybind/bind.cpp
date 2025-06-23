#include <pybind11/pybind11.h>
#include "Torch.h"
int add(int a, int b) { return a + b; }
using namespace TinyTorch;
namespace py = pybind11;

void bind_shape(py::module &m) {
  py::class_<Shape>(m, "Shape")
      .def(py::init<>())
      .def(py::init<const std::vector<int>&>())
      .def("__len__", &Shape::size)
      .def("__getitem__", [](const Shape& s, size_t index) {
        if (index >= s.size()) throw py::index_error();
        return s[index];
      })
      .def("__iter__", [](const Shape& s) {
        return py::make_iterator(s.begin(), s.end());
      }, py::keep_alive<0, 1>())
      .def("__repr__", [](const Shape& s) {
        std::ostringstream oss;
        oss << "Shape([";
        for (size_t i = 0; i < s.size(); ++i) {
          if (i > 0) oss << ", ";
          oss << s[i];
        }
        oss << "])";
        return oss.str();
      });
}

PYBIND11_MODULE(pytt, m) {
  bind_shape(m);
  py::class_<Tensor>(m, "Tensor")

      .def(py::init<>())

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

      .def_static("shape", [](const Shape &shape, bool requires_grad) {
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
