#include <map>
#include <string>
#include <vector>

#include "../optimization/optimizationInput.hpp"

namespace memopt {

struct MemRedAnalysisParser {
  static constexpr std::string_view ANALYSIS_FILE = "./.memred.memory.analysis.out";

  struct KernelInfo {
    std::string funcName;
    std::string memoryEffect;
    std::vector<std::pair<int, std::string>> ptrArgInfos;
  };

  std::map<std::string, KernelInfo> funcNameToKernelInfoMap;

  MemRedAnalysisParser();

  OptimizationInput::TaskGroup::DataDependency getKernelDataDependency(cudaGraphNode_t kernelNode);
};

}  // namespace memopt