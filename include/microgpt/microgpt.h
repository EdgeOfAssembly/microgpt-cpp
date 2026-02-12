#pragma once

/**
 * microgpt-cpp: A faithful C++20 port of Andrej Karpathy's microGPT
 * 
 * This is a line-by-line translation of the Python implementation.
 * No optimizations, no CUDA, no SIMD - just correctness.
 * 
 * Original Python: https://gist.github.com/karpathy/8627fe009c40f57531cb18360106ce95
 */

#include "value.h"
#include "utils.h"
#include "model.h"
#include "optimizer.h"
