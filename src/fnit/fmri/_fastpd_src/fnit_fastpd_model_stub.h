#pragma once
#include <array>
#include <memory>
#include <vector>

namespace newmeshreg {

class DiscreteCostFunction {
public:
    std::vector<double> unary;
    std::vector<std::array<double, 4>> pair;
    double* getUnaryCosts() { return unary.data(); }
    double computePairwiseCost(int edge, int a, int b) {
        return pair[edge][2 * a + b];
    }
};

class DiscreteModel {
public:
    int count = 0;
    int labels = 2;
    std::vector<int> endpoints;
    std::shared_ptr<DiscreteCostFunction> cost = std::make_shared<DiscreteCostFunction>();
    int getNumNodes() const { return count; }
    int getNumLabels() const { return labels; }
    int getNumPairs() const { return static_cast<int>(endpoints.size() / 2); }
    const int* getPairs() const { return endpoints.data(); }
    std::shared_ptr<DiscreteCostFunction> getCostFunction() { return cost; }
    void applyLabeling(int*) {}
    void AddNode(int n) { count = n; cost->unary.resize(2 * n); }
    void AddUnaryTerm(int node, double zero, double one) {
        cost->unary[node] += zero;
        cost->unary[count + node] += one;
    }
    void AddPairwiseTerm(int a, int b, double c00, double c01, double c10, double c11) {
        endpoints.push_back(a);
        endpoints.push_back(b);
        cost->pair.push_back({c00, c01, c10, c11});
    }
};

}
