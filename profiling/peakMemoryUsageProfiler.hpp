#include <future>
#include <thread>

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
