#pragma once

#include <thrust/host_vector.h>

#include <cassert>
#include <vector>

#include "../../profiling/memoryManager.hpp"
#include "../../utilities/cudaUtilities.hpp"

template <class T>
class Vector_h;

template <class T>
class Vector_d;

// host vector
template <class T>
class Vector_h : public thrust::host_vector<T> {
 public:
  // Constructors
  Vector_h() {}
  inline Vector_h(int N) : thrust::host_vector<T>(N) {}

  inline Vector_h<T>& operator=(const Vector_d<T>& a) {
    memopt::checkCudaErrors(cudaMemcpy(this->raw(), a.raw(), this->bytes(), cudaMemcpyDefault));
    return *this;
  }

  inline T* raw() {
    if (bytes() > 0)
      return thrust::raw_pointer_cast(this->data());
    else
      return 0;
  }

  inline const T* raw() const {
    if (bytes() > 0)
      return thrust::raw_pointer_cast(this->data());
    else
      return 0;
  }

  inline size_t bytes() const { return this->size() * sizeof(T); }
};

class VolatileVector_d {
 public:
  inline virtual void tryUpdateAddress(const std::map<void*, void*>& addressUpdate) {
    assert(false);
  };
};

struct VolatileVectorManager {
  inline static std::vector<VolatileVector_d*> volatileDeviceVectors;
};

// device vector
template <class T>
class Vector_d : VolatileVector_d {
 public:
  Vector_d() {
    this->_data = nullptr;
    this->_originalData = nullptr;
    this->_size = 0;
  }

  inline void allocate(size_t size, bool managed = false, bool input = false, bool output = false) {
    assert(this->_data == nullptr);
    this->_size = size;
    memopt::checkCudaErrors(cudaMalloc(&this->_data, this->bytes()));

    if (managed) {
      VolatileVectorManager::volatileDeviceVectors.push_back(this);
      memopt::registerManagedMemoryAddress(this->_data, this->bytes());
      if (input) {
        memopt::registerApplicationInput(this->_data);
      }
      if (output) {
        memopt::registerApplicationOutput(this->_data);
      }
    }
  }

  inline void allocate(size_t size, cudaStream_t stream) {
    assert(this->_data == nullptr);
    this->_size = size;
    memopt::checkCudaErrors(cudaMallocAsync(&this->_data, this->bytes(), stream));
  }

  inline void free() {
    assert(this->_data != nullptr);
    memopt::checkCudaErrors(cudaFree(this->_data));
    this->_data = nullptr;
    this->_size = 0;
  }

  inline void free(cudaStream_t stream) {
    assert(this->_data != nullptr);
    memopt::checkCudaErrors(cudaFreeAsync(this->_data, stream));
    this->_data = nullptr;
    this->_size = 0;
  }

  inline void tryUpdateAddress(const std::map<void*, void*>& addressUpdate) {
    if (this->_originalData == nullptr) {
      this->_originalData = this->_data;
    }

    if (addressUpdate.count(this->_originalData) > 0) {
      this->_data = static_cast<T*>(addressUpdate.at(this->_originalData));
    }
  }

  inline T* raw() {
    assert(this->_data != nullptr);
    return this->_data;
  }

  inline const T* raw() const {
    assert(this->_data != nullptr);
    return this->_data;
  }

  inline size_t size() {
    return this->_size;
  }

  inline size_t bytes() {
    return this->_size * sizeof(T);
  }

  inline Vector_d<T>& operator=(const Vector_h<T>& a) {
    memopt::checkCudaErrors(cudaMemcpy(this->raw(), a.raw(), this->bytes(), cudaMemcpyDefault));
    return *this;
  }

 private:
  T *_data, *_originalData;
  size_t _size;
};
