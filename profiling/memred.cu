#include <cstdio>
#include <fstream>
#include <iostream>
#include <map>
#include <memory>
#include <string>
#include <string_view>
#include <vector>

#include "../utilities/cudaGraphUtilities.hpp"
#include "../utilities/cudaUtilities.hpp"
#include "memred.hpp"

namespace memopt {

MemRedAnalysisParser::MemRedAnalysisParser() {
  std::ifstream in(ANALYSIS_FILE.data());
  std::string ignore;
  while (in) {
    KernelInfo kernelInfo;

    // Read the function name
    // Example: "Function void CalcMinDtOneBlock<1024>(double*, double*, double*, double*, int) (@_Z17CalcMinDtOneBlockILi1024EEvPdS0_S0_S0_i):"
    char ch;
    do {
      in.read(&ch, 1);
    } while (in && ch != '@');
    if (!in) break;

    std::string funcName;
    in >> funcName;
    funcName.erase(funcName.size() - 2);
    kernelInfo.funcName = funcName;

    // Read the function's memory effect
    // Example: "Memory Effect: ArgMemOnly"
    std::string memoryEffect;
    in >> ignore >> ignore >> memoryEffect;
    kernelInfo.memoryEffect = memoryEffect;

    // Read the argument information
    // Example: "Arg #0:	Effect: ReadOnly  Capture: No"
    while (1) {
      std::string argumentKeyword;
      in >> argumentKeyword;
      if (argumentKeyword == "Function" || !in)
        break;
      if (argumentKeyword != "Arg") {
        abort();
      }

      in.read(&ch, 1);  // ' '
      in.read(&ch, 1);  // '#'

      size_t argumentIndex;
      in >> argumentIndex;

      std::string ptrArgEffect;

      // : Effect: ReadOnly Capture: No
      in >> ignore >> ignore >> ptrArgEffect >> ignore >> ignore;

      kernelInfo.ptrArgInfos.push_back({argumentIndex, ptrArgEffect});
    }

    funcNameToKernelInfoMap[funcName] = kernelInfo;
  }
}

OptimizationInput::TaskGroup::DataDependency MemRedAnalysisParser::getKernelDataDependency(cudaGraphNode_t kernelNode) {
  CUDA_KERNEL_NODE_PARAMS nodeParams;
  getKernelNodeParams(kernelNode, nodeParams);

  const char *funcName;
  checkCudaErrors(cuFuncGetName(&funcName, nodeParams.func));

  std::string s(funcName);
  if (this->funcNameToKernelInfoMap.count(s) == 0) {
    std::cerr << "[MemRed]: Could not find kernel " << s << std::endl;
    abort();
  }

  auto kernelInfo = this->funcNameToKernelInfoMap[s];
  if (kernelInfo.memoryEffect == "AnyMem") {
    std::cerr << "[MemRed]: The memory effect of kernel " << s << "is AnyMem. ";
    std::cerr << "Please annotate the data dependency of the task containing the kernel explicitly." << std::endl;
    abort();
  }

  OptimizationInput::TaskGroup::DataDependency dataDependency;
  for (const auto &[index, effect] : kernelInfo.ptrArgInfos) {
    if (effect == "None") {
      continue;
    } else if (effect == "ReadOnly") {
      dataDependency.inputs.insert(*static_cast<void **>(nodeParams.kernelParams[index]));
    } else if (effect == "WriteOnly") {
      dataDependency.outputs.insert(*static_cast<void **>(nodeParams.kernelParams[index]));
    } else {
      dataDependency.inputs.insert(*static_cast<void **>(nodeParams.kernelParams[index]));
      dataDependency.outputs.insert(*static_cast<void **>(nodeParams.kernelParams[index]));
    }
  }

  return dataDependency;
}

}  // namespace memopt
