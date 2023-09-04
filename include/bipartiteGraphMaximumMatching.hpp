/*
Augmenting Path Algorithm for finding the maximum matching on a bipartite graph.
Time complexity: O(|V||E|).
Reference: https://oi-wiki.org/graph/graph-matching/bigraph-match/#%E5%A2%9E%E5%B9%BF%E8%B7%AF%E7%AE%97%E6%B3%95-augmenting-path-algorithm
*/

#pragma once
#include <vector>

struct augment_path {
  // Edges.
  std::vector<std::vector<int>> g;

  // The match found for nodes in set A.
  std::vector<int> pa;

  // The match found for nodes in set B.
  std::vector<int> pb;

  // Whether a node in set A is visited.
  std::vector<int> vis;

  // n: The number of nodes in set A; m: The number of nodes in set B.
  int n, m;

  // Timestamp.
  int dfn;

  // The number of matches.
  int res;

  augment_path(int _n, int _m) : n(_n), m(_m) {
    assert(0 <= n && 0 <= m);
    pa = std::vector<int>(n, -1);
    pb = std::vector<int>(m, -1);
    vis = std::vector<int>(n);
    g.resize(n);
    res = 0;
    dfn = 0;
  }

  void add(int from, int to) {
    assert(0 <= from && from < n && 0 <= to && to < m);
    g[from].push_back(to);
  }

  bool dfs(int v) {
    vis[v] = dfn;
    for (int u : g[v]) {
      if (pb[u] == -1) {
        pb[u] = v;
        pa[v] = u;
        return true;
      }
    }
    for (int u : g[v]) {
      if (vis[pb[u]] != dfn && dfs(pb[u])) {
        pa[v] = u;
        pb[u] = v;
        return true;
      }
    }
    return false;
  }

  int solve() {
    while (true) {
      dfn++;
      int cnt = 0;
      for (int i = 0; i < n; i++) {
        if (pa[i] == -1 && dfs(i)) {
          cnt++;
        }
      }
      if (cnt == 0) {
        break;
      }
      res += cnt;
    }
    return res;
  }
};
