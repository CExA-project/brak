#ifndef __UTILS_HPP__
#define __UTILS_HPP__

#include <cstdio>

void display(unsigned const iteration, double const residual,
             unsigned const interval) {
  if (iteration % interval != 0)
    return;

  std::printf("iteration=%u residual=%e\n", iteration, residual);
}

#endif // ifndef __UTILS_HPP__
