#pragma once

#include <thrust/host_vector.h>

#include <cassert>

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
    checkCudaErrors(cudaMemcpy(this->raw(), a.raw(), this->bytes(), cudaMemcpyDefault));
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

// device vector
template <class T>
class Vector_d {
 public:
  Vector_d() {
    this->_data = nullptr;
    this->_size = 0;
  }

  inline void allocate(size_t size, bool managed = false, bool input = false, bool output = false) {
    assert(this->_data == nullptr);
    this->_size = size;
    checkCudaErrors(cudaMalloc(&this->_data, this->bytes()));

    if (managed) {
      registerManagedMemoryAddress(this->_data, this->bytes());
      if (input) {
        registerApplicationInput(this->_data);
      }
      if (output) {
        registerApplicationOutput(this->_data);
      }
    }
  }

  inline void allocate(size_t size, cudaStream_t stream) {
    assert(this->_data == nullptr);
    this->_size = size;
    checkCudaErrors(cudaMallocAsync(&this->_data, this->bytes(), stream));
  }

  inline void free() {
    assert(this->_data != nullptr);
    checkCudaErrors(cudaFree(this->_data));
    this->_data = nullptr;
    this->_size = 0;
  }

  inline void free(cudaStream_t stream) {
    assert(this->_data != nullptr);
    checkCudaErrors(cudaFreeAsync(this->_data, stream));
    this->_data = nullptr;
    this->_size = 0;
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
    checkCudaErrors(cudaMemcpy(this->raw(), a.raw(), this->bytes(), cudaMemcpyDefault));
    return *this;
  }

 private:
  T* _data;
  size_t _size;
};
