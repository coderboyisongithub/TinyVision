/*
 * TinyTorch
 * @author 	: keith@robot9.me
 *
 */

#pragma once

#include "Tensor.h"

namespace TinyTorch::nn {

typedef enum {
  FAN_AVG = 0,
  FAN_IN = 1,
  FAN_OUT = 2,
} FanMode;

typedef enum {
    LINEAR,
    CONV2D,
    SIGMOID,
    TANH,
    RELU,
    LEAKY_RELU,
    NONE
} NonlinearityMode;

class Init {
 public:

  static void uniform(Tensor &tensor, float min, float max);
  static void kaimingUniform(Tensor &tensor, float gain = 0,
                             FanMode mode = FAN_IN);
  static void xavierUniform(Tensor &tensor, float gain = 1.0f,
                             FanMode mode = FAN_AVG);
  static void xavierNormal(Tensor &tensor, float gain = 1.0f,
                             FanMode = FAN_AVG);
  static void kaimingNormal(Tensor& tensor, float gain = 0.0f,
                            NonlinearityMode nonlinearity = NONE, FanMode mode = FAN_IN);
  static void constant(Tensor& tensor, float value);
  static std::pair<int32_t, int32_t> calculateFan(const Tensor &tensor);
  static int32_t calculateFan(const Tensor &tensor, FanMode mode);
  static float calculateGain(float param);
  static float calculateGain(float param, NonlinearityMode nonlinearity);

 private:
    static float calculateXavierBound(const Tensor &tensor, float gain, FanMode mode);
};

}  // namespace TinyTorch::nn
