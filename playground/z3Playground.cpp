#include <z3++.h>

#include <iostream>
#include <sstream>
#include <vector>

using namespace z3;

void opt_example() {
  context c;
  optimize opt(c);
  params p(c);
  p.set("priority", c.str_symbol("pareto"));
  opt.set(p);
  expr x = c.int_const("x");
  expr y = c.int_const("y");
  opt.add(10 >= x && x >= 0);
  opt.add(10 >= y && y >= 0);
  opt.add(x + y <= 11);
  optimize::handle h1 = opt.maximize(x);
  optimize::handle h2 = opt.maximize(y);
  while (true) {
    if (sat == opt.check()) {
      std::cout << x << ": " << opt.lower(h1) << " " << y << ": " << opt.lower(h2) << "\n";
    } else {
      break;
    }
  }
}

int main() {
  opt_example();
  return 0;
}
