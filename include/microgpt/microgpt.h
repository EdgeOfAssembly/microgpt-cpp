#pragma once

/**
 * microgpt-cpp: A faithful C++20 port of Andrej Karpathy's microGPT
 * 
 * This is a line-by-line translation of the Python implementation
 * using a graph-based scalar autograd approach for educational clarity.
 * 
 * Original Python: https://gist.github.com/karpathy/8627fe009c40f57531cb18360106ce95
 */

#include "value.h"
#include "layers.h"
#include "utils.h"
#include "optimizer.h"
#include "model.h"
