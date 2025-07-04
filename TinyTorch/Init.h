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

class Init {
 public:
  static void uniform(Tensor &tensor, float min, float max);
  static void kaimingUniform(Tensor &tensor, float a = 0,
                             FanMode mode = FAN_IN);
  static void xavierUniform(Tensor &tensor, float gain = 1.0f,
                             FanMode mode = FAN_AVG);
  static void xavierNormal(Tensor &tensor, float gain = 1.0f,
                             FanMode = FAN_AVG);
  static std::pair<int32_t, int32_t> calculateFan(const Tensor &tensor);
  static int32_t calculateFan(const Tensor &tensor, FanMode mode);
  static float calculateGain(float param);
 private:
    static float calculateXavierBound(const Tensor &tensor, float gain, FanMode mode);
};

}  // namespace TinyTorch::nn
