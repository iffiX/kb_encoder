#ifndef MATCHER_SRC_MATCHER_H
#define MATCHER_SRC_MATCHER_H
#include "cista.h"
#include "pybind11/stl.h"
#include <string>
#include <vector>
#include <unordered_map>
#include <unordered_set>
#include <iostream>

using Edge = std::tuple<long, long, long, float, std::string>;

class TrieNode {
public:
    int token = -1;
    bool isWordEnd = false;
    std::unordered_map<int, TrieNode*> children;
public:
    TrieNode() = default;
    TrieNode(int token, bool isWordEnd);
    ~TrieNode();
    TrieNode* addChild(int childToken, bool isChildWordEnd);
    bool removeChild(int childToken);
    void clear();
    size_t initializeFromString(const std::string &content, size_t start = 0);
    std::string serialize() const;
};

class Trie {
public:
    TrieNode root;

public:
    Trie() = default;
    explicit Trie(const std::vector<std::vector<int>> &words);
    void insert(const std::vector<int> &word);
    bool remove(const std::vector<int> &word);
    void clear();
    std::vector<int> matchForStart(const std::vector<int> &sentence, size_t start = 0) const;
    std::unordered_map<size_t, std::vector<int>> matchForAll(const std::vector<int> &sentence) const;
    void initializeFromString(const std::string &content);
    std::string serialize() const;
};

class ConceptNetReader {
public:
    // target id, edge ids point to the target.
    std::unordered_map<long, std::vector<size_t>> edgeToTarget;
    // source id, edge ids point from the source.
    std::unordered_map<long, std::vector<size_t>> edgeFromSource;
    // source id, relation id, target id, weight, annotated representation.
    std::vector<Edge> edges;
    std::vector<std::string> nodes;
    std::vector<std::string> relationships;
    std::vector<std::vector<int>> tokenizedNodes;
    std::vector<std::vector<int>> tokenizedRelationships;
    std::vector<std::vector<int>> tokenizedEdgeAnnotations;
    std::vector<std::string> rawRelationships;
public:
    ConceptNetReader() = default;
    explicit ConceptNetReader(const std::string &path);
    std::vector<Edge> getEdges(long source = -1, long target = -1) const;
    const std::vector<std::string> &getNodes() const;
    std::vector<std::string> getNodes(const std::vector<long> &nodeIndexes) const;
};


class ConceptNetMatcher {
public:
    const ConceptNetReader *reader;
public:
    explicit ConceptNetMatcher(const ConceptNetReader &tokenizedReader);
    explicit ConceptNetMatcher(const std::string &archivePath);
    ~ConceptNetMatcher();
    std::unordered_map<size_t, std::tuple<size_t, std::vector<std::vector<int>>>>
    match(const std::vector<int> &sentence,
          int maxTimes = 100, int maxDepth = 3, int maxEdges = 10, int seed = -1,
          const std::vector<std::vector<int>> &similarityExclude = {},
          const std::vector<std::vector<int>> &rankFocus = {},
          const std::vector<std::vector<int>> &rankExclude = {}) const;
    std::string getNodeTrie() const;
    std::vector<std::pair<std::vector<int>, long>> getNodeMap() const;
    void save(const std::string &archivePath) const;
    void load(const std::string &archivePath);

private:
    struct VectorHash {
        std::size_t operator()(std::vector<int> const& vec) const;
    };

    struct PairHash
    {
        template <class T1, class T2>
        std::size_t operator() (const std::pair<T1, T2> &pair) const;
    };

    struct SerializableEdge {
        long source;
        long relation;
        long target;
        float weight;
        cista::raw::string annotation;
    };

    struct VisitedPath {
        long root;
        int matchedFocusCount;
        std::vector<size_t> edges;
        std::unordered_map<size_t, float> similarities;
        std::unordered_set<long> visitedNodes;

        float uncoveredSimilarity;
        std::vector<size_t> uncoveredEdges;
    };

    struct VisitedSubGraph {
        std::vector<VisitedPath> visitedPaths;
        std::unordered_set<std::pair<long, long>, PairHash> coveredNodePairs;
        std::unordered_map<long, std::vector<size_t>> coveredSubGraph;
    };

    struct ConceptNetArchive {
        cista::raw::hash_map<long, cista::raw::vector<size_t>> edgeToTarget;
        cista::raw::hash_map<long, cista::raw::vector<size_t>> edgeFromSource;
        cista::raw::vector<SerializableEdge> edges;
        cista::raw::vector<cista::raw::string> nodes;
        cista::raw::vector<cista::raw::string> relationships;
        cista::raw::vector<cista::raw::vector<int>> tokenizedNodes;
        cista::raw::vector<cista::raw::vector<int>> tokenizedRelationships;
        cista::raw::vector<cista::raw::vector<int>> tokenizedEdgeAnnotations;
        cista::raw::vector<cista::raw::string> rawRelationships;
    };

private:
    bool isLoad;
    Trie nodeTrie;
    std::unordered_map<std::vector<int>, long, VectorHash> nodeMap;

private:
    std::vector<int> edgeToAnnotation(const Edge &edge) const;

    void updatePath(VisitedPath &path,
                    const std::unordered_set<std::pair<long, long>, PairHash> &coveredNodePairs,
                    int remainingEdges) const;

    static void joinVisitedSubGraph(VisitedSubGraph &vsgOut, const VisitedSubGraph &vsgIn);

    static size_t findPattern(const std::vector<int> &sentence, const std::vector<std::vector<int>> &patterns);

    static std::vector<int> filter(const std::vector<int> &sentence, const std::vector<std::vector<int>> &patterns);

    static float similarity(const std::vector<int> &source, const std::vector<int> &target);
    template <typename T>
    static cista::raw::vector<T> vector2ArchiveVector(const std::vector<T> &vec);
    template <typename T>
    static std::vector<T> archiveVector2Vector(const cista::raw::vector<T> &vec);
};
#endif //MATCHER_SRC_MATCHER_H
