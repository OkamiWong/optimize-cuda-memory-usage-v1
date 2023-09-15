#include <z3++.h>

#include <iostream>
#include <sstream>
#include <vector>

void optExample() {
  z3::context c;
  z3::optimize opt(c);
  z3::params p(c);
  p.set("priority", c.str_symbol("pareto"));
  opt.set(p);
  z3::expr x = c.int_const("x");
  z3::expr y = c.int_const("y");
  opt.add(10 >= x && x >= 0);
  opt.add(10 >= y && y >= 0);
  opt.add(x + y <= 11);
  z3::optimize::handle h1 = opt.maximize(x);
  z3::optimize::handle h2 = opt.maximize(y);
  while (true) {
    if (z3::check_result::sat == opt.check()) {
      std::cout << x << ": " << opt.lower(h1) << " " << y << ": " << opt.lower(h2) << "\n";
    } else {
      break;
    }
  }
}

int main() {
  optExample();
  return 0;
}
