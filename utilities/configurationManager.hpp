#pragma once

#include <cassert>
#include <fstream>
#include <iostream>
#include <string>

#include "../include/json.hpp"

namespace memopt {

struct Configuration {
  struct Generic {
    bool optimize = false;
    bool verify = false;
    int repeat = 1;

    NLOHMANN_DEFINE_TYPE_INTRUSIVE(
      Generic,
      optimize,
      verify,
      repeat
    );
  } generic;

  struct Optimization {
    bool mergeConcurrentCudaGraphNodes = true;
    double prefetchingBandwidthInGB = 281.0;
    double acceptableRunningTimeFactor = 10.0;
    int minManagedArraySize = 0;
    std::string solver = "SCIP";
    double weightOfPeakMemoryUsage = 1.0;
    double weightOfTotalRunningTime = 0.0001;
    double weightOfNumberOfMigrations = 0.00001;

    NLOHMANN_DEFINE_TYPE_INTRUSIVE(
      Optimization,
      mergeConcurrentCudaGraphNodes,
      prefetchingBandwidthInGB,
      acceptableRunningTimeFactor,
      minManagedArraySize,
      solver,
      weightOfPeakMemoryUsage,
      weightOfTotalRunningTime,
      weightOfNumberOfMigrations
    );
  } optimization;

  struct Execution {
    bool useNvlink = false;
    bool measurePeakMemoryUsage = false;
    int mainDeviceId = 1;
    int storageDeviceId = 2;

    NLOHMANN_DEFINE_TYPE_INTRUSIVE(
      Execution,
      useNvlink,
      measurePeakMemoryUsage,
      mainDeviceId,
      storageDeviceId
    );
  } execution;

  // Tiled Cholesky
  struct TiledCholesky {
    int n = 256;
    int t = 4;

    NLOHMANN_DEFINE_TYPE_INTRUSIVE(
      TiledCholesky,
      n,
      t
    );
  } tiledCholesky;

  // LULESH

  struct Lulesh {
    int s = 45;
    bool constrainIterationCount = false;
    int targetIterationCount = 3000;
    int iterationBatchSize = 1;

    NLOHMANN_DEFINE_TYPE_INTRUSIVE(
      Lulesh,
      s,
      constrainIterationCount,
      targetIterationCount,
      iterationBatchSize
    );
  } lulesh;

  NLOHMANN_DEFINE_TYPE_INTRUSIVE(
    Configuration,
    generic,
    optimization,
    execution,
    tiledCholesky,
    lulesh
  );
};

class ConfigurationManager {
 public:
  static const Configuration& getConfig() {
    assert(initialized);
    return configuration;
  };

  static void exportDefaultConfiguration(std::string fileName = "defaultConfig.json") {
    Configuration defaultConfiguration;
    exportConfiguration(fileName, defaultConfiguration);
  }

  static void exportCurrentConfiguration(std::string fileName = "currentConfig.json") {
    exportConfiguration(fileName, configuration);
  }

  static void loadConfiguration(std::string fileName = "config.json") {
    try {
      std::ifstream f(fileName);
      auto j = nlohmann::json::parse(f);
      configuration = j.get<Configuration>();
      initialized = true;
    } catch (...) {
      std::cerr << "Failed to load configuration from " << fileName << std::endl;
      std::cerr << "See defaultConfig.json for sample configuration" << std::endl;

      exportDefaultConfiguration();

      exit(-1);
    }
  }

 private:
  inline static Configuration configuration;

  inline static bool initialized = false;

  static void exportConfiguration(const std::string& fileName, const Configuration& configuration) {
    nlohmann::json j = configuration;
    std::string s = j.dump(2);

    std::ofstream f(fileName);
    f << s << std::endl;
  }
};

};  // namespace memopt
