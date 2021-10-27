#ifndef MATCHER_H
#define MATCHER_H
// Uncomment below macro to enable viewing the decision process
//#define DEBUG
#include "cista.h"
#include "highfive/H5File.hpp"
#include "highfive/H5DataSet.hpp"
#include "highfive/H5DataSpace.hpp"
#include "pybind11/stl.h"
#include <string>
#include <vector>
#include <memory>
#include <unordered_map>
#include <unordered_set>
#include <iostream>

using Edge = std::tuple<long, long, long, float, std::string>;
using MatchResult = std::unordered_map<size_t, std::tuple<size_t, std::vector<std::vector<int>>>>;

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

class KnowledgeBase {
public:
    struct SerializableEdge {
        long source;
        long relation;
        long target;
        float weight;
        cista::raw::string annotation;
    };

    struct KnowledgeArchive {
        cista::raw::hash_map<long, cista::raw::vector<size_t>> edgeToTarget;
        cista::raw::hash_map<long, cista::raw::vector<size_t>> edgeFromSource;
        cista::raw::vector<SerializableEdge> edges;
        cista::raw::vector<cista::raw::string> nodes;
        cista::raw::vector<cista::raw::string> relationships;
        cista::raw::vector<cista::raw::vector<int>> tokenizedNodes;
        cista::raw::vector<cista::raw::vector<int>> tokenizedRelationships;
        cista::raw::vector<cista::raw::vector<int>> tokenizedEdgeAnnotations;
        cista::raw::vector<cista::raw::string> rawRelationships;
        cista::raw::string nodeEmbeddingFileName;
    };

public:
    // target id, edge ids point to the target.
    std::unordered_map<long, std::vector<size_t>> edgeToTarget;
    // source id, edge ids point from the source.
    std::unordered_map<long, std::vector<size_t>> edgeFromSource;
    // source id, relation id, target id, weight, annotated representation.
    std::vector<Edge> edges;
    std::vector<std::string> nodes;
    std::vector<std::string> relationships;
    std::vector<std::string> rawRelationships;
    std::unordered_set<size_t> disabledEdges;
    std::shared_ptr<HighFive::File> nodeEmbeddingFile;
    std::shared_ptr<void> nodeEmbeddingMem;
    std::string nodeEmbeddingFileName;

    std::vector<std::vector<int>> tokenizedNodes;
    std::vector<std::vector<int>> tokenizedRelationships;
    std::vector<std::vector<int>> tokenizedEdgeAnnotations;

    // Each row is the distance to each landmark l0, l1, ..., ln
    std::vector<std::vector<int>> landmarkDistances;

public:
    KnowledgeBase() = default;
    void clearDisabledEdges();
    void disableEdgesOfRelationships(const std::vector<std::string> &relationships);
    void disableEdgesOfNodes(const std::vector<std::string> &nodes);
    std::vector<long> findNodes(const std::vector<std::string> &nodes) const;
    std::vector<Edge> getEdges(long source = -1, long target = -1) const;
    const std::vector<std::string> &getNodes() const;
    std::vector<std::string> getNodes(const std::vector<long> &nodeIndexes) const;
    void setNodeEmbeddingFileName(const std::string &path, bool loadEmbeddingToMem = true);
    bool isLandmarkInited() const;
    void initLandmarks(int seedNum = 100, int landmarkNum = 100, int seed = -1, const std::string &landmarkPath = "");
    int distance(long node1, long node2, bool fast = true) const;
    void save(const std::string &archivePath) const;
    void load(const std::string &archivePath, bool loadEmbeddingToMem = true);
    void refresh(bool loadEmbeddingToMem);

private:
    struct PriorityCmp {
        template <typename T1, typename T2>
        bool operator()(const std::pair<T1, T2> &pair1, const std::pair<T1, T2> &pair2);
    };

private:
    // For faster access
    std::unordered_map<long, std::unordered_set<long>> adjacency;

private:
    void loadEmbedding(bool loadEmbeddingToMem);

    void loadAdjacency();

    bool isNeighbor(long node1, long node2) const;

    std::vector<int> bfsDistances(long node) const;

    int bfsDistance(long node1, long node2, int maxDepth = 3) const;

    int landmarkDistance(long node1, long node2) const;

    int landmarkLowerBound(long node, long targetNode) const;

    int ALTDistance(long node1, long node2) const;

    template <typename T>
    static cista::raw::vector<T> vector2ArchiveVector(const std::vector<T> &vec);
    template <typename T>
    static std::vector<T> archiveVector2Vector(const cista::raw::vector<T> &vec);
};

class KnowledgeMatcher {
public:
    KnowledgeBase kb;
public:
    explicit KnowledgeMatcher(const KnowledgeBase &knowledgeBase);
    explicit KnowledgeMatcher(const std::string &archivePath);
    MatchResult
    matchByNode(const std::vector<int> &sourceSentence, const std::vector<int> &targetSentence = {},
                const std::vector<int> &sourceMask = {}, const std::vector<int> &targetMask = {},
                int maxTimes = 100, int maxDepth = 3, int maxEdges = 10, int seed = -1,
                float discardEdgesIfSimilarityBelow = 0,
                float discardEdgesIfRankBelow = 0) const;
    MatchResult
    matchByNodeEmbedding(const std::vector<int> &sourceSentence, const std::vector<int> &targetSentence = {},
                         const std::vector<int> &sourceMask = {}, const std::vector<int> &targetMask = {},
                         int maxTimes = 100, int maxDepth = 3, int maxEdges = 10, int seed = -1,
                         float discardEdgesIfSimilarityBelow = 0.5,
                         float discardEdgesIfRankBelow = 0) const;
    MatchResult
    matchByToken(const std::vector<int> &sourceSentence, const std::vector<int> &targetSentence = {},
                 const std::vector<int> &sourceMask = {}, const std::vector<int> &targetMask = {},
                 int maxTimes = 100, int maxDepth = 3, int maxEdges = 10, int seed = -1,
                 float discardEdgesIfSimilarityBelow = 0,
                 float discardEdgesIfRankBelow = 0,
                 const std::vector<std::vector<int>> &rankFocus = {},
                 const std::vector<std::vector<int>> &rankExclude = {}) const;
    std::string getNodeTrie() const;
    std::vector<std::pair<std::vector<int>, long>> getNodeMap() const;
    void save(const std::string &archivePath) const;
    void load(const std::string &archivePath, bool loadEmbeddingToMem = true);

private:
    struct VectorHash {
        std::size_t operator()(std::vector<int> const& vec) const;
    };

    struct PairHash
    {
        template <class T1, class T2>
        std::size_t operator() (const std::pair<T1, T2> &pair) const;
    };

    struct UndirectedPairHash
    {
        template <class T1, class T2>
        std::size_t operator() (const std::pair<T1, T2> &pair) const;
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

private:
    Trie nodeTrie;
    std::unordered_map<std::vector<int>, long, VectorHash> nodeMap;

private:
    std::vector<int> edgeToAnnotation(size_t edgeIndex) const;

    std::string edgeToStringAnnotation(size_t edgeIndex) const;

    void matchForSourceAndTarget(const std::vector<int> &sourceSentence,
                                 const std::vector<int> &targetSentence,
                                 const std::vector<int> &sourceMask,
                                 const std::vector<int> &targetMask,
                                 std::unordered_map<size_t, std::vector<int>> &sourceMatch,
                                 std::unordered_map<size_t, std::vector<int>> &targetMatch) const;

    MatchResult selectPaths(VisitedSubGraph &visitedSubGraph,
                            const std::unordered_map<long, std::pair<size_t, size_t>> &posRef,
                            int maxEdges, float discardEdgesIfRankBelow) const;

    void updatePath(VisitedPath &path,
                    const std::unordered_set<std::pair<long, long>, PairHash> &coveredNodePairs,
                    int remainingEdges) const;

    static void joinVisitedSubGraph(VisitedSubGraph &vsgOut, const VisitedSubGraph &vsgIn);

    static size_t findPattern(const std::vector<int> &sentence, const std::vector<std::vector<int>> &patterns);

    static std::vector<int> filter(const std::vector<int> &sentence, const std::vector<std::vector<int>> &patterns);

    static std::vector<int> mask(const std::vector<int> &sentence, const std::vector<int> &mask);

    static float similarity(const std::vector<int> &source, const std::vector<int> &target);

};
#endif //MATCHER_H
