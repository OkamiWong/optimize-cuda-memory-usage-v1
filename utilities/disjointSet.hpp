#pragma once

#include <map>
#include <utility>

template <typename KeyType>
class DisjointSet {
 public:
  DisjointSet() = default;
  KeyType findRoot(KeyType u) {
    auto it = this->parent.find(u);
    if (it == this->parent.end()) {
      const auto [insertedIt, success] = this->parent.insert({u, u});
      it = insertedIt;

      this->size[u] = 1;
    }

    auto p = it->second;
    if (p == u) {
      return u;
    } else {
      p = findRoot(p);
      it->second = p;
      return this->findRoot(p);
    }
  }

  void unionUnderlyingSets(KeyType u, KeyType v) {
    u = this->findRoot(u);
    v = this->findRoot(v);
    if (u == v) return;

    // Technical debt: The number of query can be reduced.
    if (this->size[u] < this->size[v]) std::swap(u, v);
    this->parent[v] = u;
    this->size[u] = this->size[u] + this->size[v];
  }

  size_t getSetSize(KeyType u) {
    return this->size[this->findRoot(u)];
  }

 private:
  std::map<KeyType, KeyType> parent;
  std::map<KeyType, size_t> size;
};
