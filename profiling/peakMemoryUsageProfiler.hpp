#include <future>
#include <thread>

namespace memopt {

class PeakMemoryUsageProfiler {
 public:
  void start();
  size_t end();

 private:
  std::thread monitorThread;
  std::atomic<bool> stopFlag;
  std::promise<size_t> peakMemoryUsagePromise;

  void periodicallyCheckMemoryUsage();
};

}  // namespace memopt
