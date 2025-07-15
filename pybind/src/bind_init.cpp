#include "pybind11/pybind11.h"
#include "Torch.h"
#include "Init.h"
using namespace TinyTorch;
namespace py = pybind11;

nn::FanMode parse_fan_mode(const std::string& mode) {
    static const std::unordered_map<std::string, TinyTorch::nn::FanMode> mapping = {
        {"fan_avg", TinyTorch::nn::FAN_AVG},
        {"fan_in", TinyTorch::nn::FAN_IN},
        {"fan_out", TinyTorch::nn::FAN_OUT}
    };
    auto it = mapping.find(mode);
    if (it != mapping.end()) return it->second;
    throw std::invalid_argument("Unknown fan mode: " + mode);
}

nn::NonlinearityMode parse_nonlinearity(const std::string& nonlinearity) {
    static const std::unordered_map<std::string, TinyTorch::nn::NonlinearityMode> mapping = {
        {"linear", TinyTorch::nn::LINEAR},
        {"conv2d", TinyTorch::nn::CONV2D},
        {"sigmoid", TinyTorch::nn::SIGMOID},
        {"tanh", TinyTorch::nn::TANH},
        {"relu", TinyTorch::nn::RELU},
        {"leaky_relu", TinyTorch::nn::LEAKY_RELU},
        {"none", TinyTorch::nn::NONE}
    };
    auto it = mapping.find(nonlinearity);
    if (it != mapping.end()) return it->second;
    throw std::invalid_argument("Unknown nonlinearity: " + nonlinearity);
}
void kaiming_normal_wrapper(
    Tensor& tensor,
    const std::string& nonlinearity = "none",
    float gain = 0.0f,
    const std::string& mode = "fan_in"
) {
    nn::Init::kaimingNormal(
        tensor,
        gain,
        parse_nonlinearity(nonlinearity),
        parse_fan_mode(mode)
    );
}

void kaiming_uniform_wrapper(
    Tensor& tensor,
    float gain = 0.0f,
    const std::string& mode = "fan_in"
) {
    nn::Init::kaimingUniform(
        tensor,
        gain,
        parse_fan_mode(mode)
    );
}

void xavier_uniform_wrapper(
    Tensor& tensor,
    float gain = 1.0f,
    const std::string& mode = "fan_avg"
) {
    nn::Init::xavierUniform(
        tensor,
        gain,
        parse_fan_mode(mode)
    );
}

void xavier_normal_wrapper(
    Tensor& tensor,
    float gain = 1.0f,
    const std::string& mode = "fan_avg"
) {
    nn::Init::xavierNormal(
        tensor,
        gain,
        parse_fan_mode(mode)
    );
}

void create_init_submodule(py::module_ &nn) {
    auto init = nn.def_submodule("init", "Init Tensor");
    py::enum_<TinyTorch::nn::FanMode>(init, "FanMode")
            .value("fan_avg", TinyTorch::nn::FAN_AVG)
            .value("fan_in", TinyTorch::nn::FAN_IN)
            .value("fan_out", TinyTorch::nn::FAN_OUT);

    py::enum_<TinyTorch::nn::NonlinearityMode>(init, "Nonlinearity")
        .value("linear",TinyTorch::nn:: LINEAR)
        .value("conv2d", TinyTorch::nn::CONV2D)
        .value("sigmoid", TinyTorch::nn::SIGMOID)
        .value("tanh", TinyTorch::nn::TANH)
        .value("relu", TinyTorch::nn::RELU)
        .value("leaky_relu", TinyTorch::nn::LEAKY_RELU)
        .value("none", TinyTorch::nn::NONE);

    init.def("kaiming_normal_", &kaiming_normal_wrapper,
            py::arg("tensor"),
            py::arg("nonlinearity") = "none",
            py::arg("gain") = 1.0f,
            py::arg("mode") = "fan_in");

    init.def("xavier_uniform_", &xavier_uniform_wrapper,
            py::arg("tensor"),
            py::arg("gain") = 1.0f,
            py::arg("mode") = "fan_avg");

    init.def("xavier_normal_", &xavier_normal_wrapper,
            py::arg("tensor"),
            py::arg("gain") = 1.0f,
            py::arg("mode") = "fan_avg");

    init.def("uniform_", &nn::Init::uniform,
            py::arg("tensor"),
            py::arg("min"),
            py::arg("max"));

    init.def("constant_", &nn::Init::constant,
            py::arg("tensor"),
            py::arg("value"));
}
