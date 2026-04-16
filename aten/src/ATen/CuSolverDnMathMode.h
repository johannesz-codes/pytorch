#pragma once

#include <c10/util/Exception.h>

#include <ostream>
#include <string>

namespace at {

// Mirrors cusolverMathMode_t from cuSOLVER API
enum class CuSolverDnMathMode : int8_t {
  Default = 0,
  // Maps to CUSOLVER_FP32_EMULATED_BF16X9_MATH
  Fp32EmulatedBf16x9 = 1,
};

inline std::string CuSolverDnMathModeToString(
    at::CuSolverDnMathMode math_mode) {
  switch (math_mode) {
    case CuSolverDnMathMode::Default:
      return "at::CuSolverDnMathMode::Default";
    case CuSolverDnMathMode::Fp32EmulatedBf16x9:
      return "at::CuSolverDnMathMode::Fp32EmulatedBf16x9";
    default:
      TORCH_CHECK(false, "Unknown cuSolver math mode");
  }
}

inline std::ostream& operator<<(
    std::ostream& stream,
    at::CuSolverDnMathMode math_mode) {
  return stream << CuSolverDnMathModeToString(math_mode);
}

// Mirrors cusolverFpEmulationStrategy_t from cuSOLVER API
enum class CuSolverDnEmulationStrategy : int8_t {
  Default = 0,
  // Maps to CUSOLVER_FP_EMU_STRATEGY_EAGER
  Eager = 1,
};

inline std::string CuSolverDnEmulationStrategyToString(
    at::CuSolverDnEmulationStrategy strategy) {
  switch (strategy) {
    case CuSolverDnEmulationStrategy::Default:
      return "at::CuSolverDnEmulationStrategy::Default";
    case CuSolverDnEmulationStrategy::Eager:
      return "at::CuSolverDnEmulationStrategy::Eager";
    default:
      TORCH_CHECK(false, "Unknown cuSolver emulation strategy");
  }
}

inline std::ostream& operator<<(
    std::ostream& stream,
    at::CuSolverDnEmulationStrategy strategy) {
  return stream << CuSolverDnEmulationStrategyToString(strategy);
}

} // namespace at
