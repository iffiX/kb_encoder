#define FMT_HEADER_ONLY

#include "matcher.h"
#include "tqdm.h"
#include "fmt/format.h"
#include "xtensor/xio.hpp"
#include "xtensor/xsort.hpp"
#include "xtensor/xview.hpp"
#include "xtensor/xadapt.hpp"
#include "xtensor/xtensor.hpp"
#include "highfive/H5File.hpp"
#include <queue>
#include <tuple>
#include <random>
#include <memory>
#include <vector>
#include <fstream>
#include <algorithm>
#include <string_view>

// A number large enough to represent unreachability, but not enough to overflow int
#define DISTANCE_MAX 1000000
using namespace std;

tuple<size_t, string> scan(const string &content, const vector<string> &delims, size_t start = 0) {
    vector<size_t> positions(delims.size(), string::npos);
    size_t i = 0;
    size_t minEnd = content.length();
    for (auto &delim : delims) {
        positions[i] = string_view(content.data() + start, minEnd).find(delim);
        minEnd = min(minEnd, positions[i]);
        i++;
    }
    size_t minDelim = min_element(positions.begin(), positions.end()) - positions.begin();
    if (positions[minDelim] == string::npos)
        return make_tuple(string::npos, "");
    else
        return make_tuple(positions[minDelim] + start, delims[minDelim]);
}

bool isAllMasked(size_t start, size_t end, const vector<int> &mask) {
    bool allMasked = not mask.empty();
    if (not mask.empty()) {
        for (size_t i = start; i < end; i++) {
            if (mask[i] == 1) {
                allMasked = false;
                break;
            }
        }
    }
    return allMasked;
}

template<typename T>
UnorderedPair<T>::UnorderedPair(T value1, T value2) : value1(value1), value2(value2) {}

template<typename T>
bool UnorderedPair<T>::operator==(const UnorderedPair<T> &other) {
    return (value1 == other.value1 && value2 == other.value2) ||
           (value2 == other.value1 && value1 == other.value2);
}

template<typename T>
std::size_t UnorderedPairHash<T>::operator()(const UnorderedPair<T> &pair) const {
    return hash<T>(pair.value1) ^ hash<T>(pair.value2);
}

TrieNode::TrieNode(int token, bool isWordEnd) : token(token), isWordEnd(isWordEnd) {}

TrieNode::~TrieNode() {
    clear();
}

TrieNode *TrieNode::addChild(int childToken, bool isChildWordEnd) {
    children[childToken] = new TrieNode(childToken, isChildWordEnd);
    return children[childToken];
}

bool TrieNode::removeChild(int childToken) {
    if (children.find(childToken) != children.end()) {
        delete children[childToken];
        children.erase(childToken);
        return true;
    } else
        return false;
}

void TrieNode::clear() {
    for (auto &child : children)
        delete child.second;
    children.clear();
}

size_t TrieNode::initializeFromString(const string &content, size_t start) {
    // About 30 seconds for a 500 KB file
    token = -1;
    isWordEnd = false;
    clear();
    auto nodeStart = scan(content, {"$,(", "$", ",("}, start);
    if (get<0>(nodeStart) == string::npos)
        throw invalid_argument("Cannot find a delimeter '$,(' or '$' or ',(' in input.");
    if (get<1>(nodeStart) == "$" or get<1>(nodeStart) == "$,(") {
        token = stoi(content.substr(start, get<0>(nodeStart)));
        isWordEnd = true;
        if (get<1>(nodeStart) == "$")
            return get<0>(nodeStart) + get<1>(nodeStart).length();
    }

    token = stoi(content.substr(start, get<0>(nodeStart)));
    size_t readPos = get<0>(nodeStart) + get<1>(nodeStart).length();
    while (readPos < content.length()) {
        if (content[readPos] == ')')
            break;
        auto childStart = scan(content, {"[",}, readPos);
        if (get<0>(childStart) == string::npos)
            throw invalid_argument("Cannot find a delimiter '[' for starting a child.");
        auto *child = new TrieNode();
        readPos = child->initializeFromString(content, get<0>(childStart) + get<1>(nodeStart).length());
        children[child->token] = child;
        auto childEnd = scan(content, {"],", "]"}, readPos);
        if (get<0>(childEnd) == string::npos)
            throw invalid_argument("Cannot find an enclosing delimiter ']' or '],' for child.");
        readPos = get<0>(childEnd) + get<1>(childEnd).length();
    }
    auto end = scan(content, {"),", ")"}, readPos);
    if (get<0>(end) == string::npos)
        throw invalid_argument("Cannot find a closing delimeter ')' or '),'.");
    return readPos + get<1>(end).length();
}

string TrieNode::serialize() const {
    string content = to_string(token);
    if (isWordEnd)
        content += "$";

    content += ",(";
    size_t count = 0;
    for (auto &child: children) {
        content += "[";
        content += child.second->serialize() + "]";
        if (count < children.size() - 1)
            content += ",";
        count++;
    }
    content += ")";

    return content;
}

Trie::Trie(const vector<vector<int>> &words) {
    for (auto &word : words)
        insert(word);
}

void Trie::insert(const vector<int> &word) {
    auto *currentNode = &root;
    for (int token : word) {
        if (token < 0)
            throw invalid_argument("Word must be a positive integer.");
        auto child = currentNode->children.find(token);
        if (child != currentNode->children.end())
            currentNode = child->second;
        else
            currentNode = currentNode->addChild(token, false);
    }
    currentNode->isWordEnd = true;
}

bool Trie::remove(const vector<int> &word) {
    TrieNode *parentNode = nullptr;
    auto *currentNode = &root;
    if (word.empty())
        return false;
    for (int token : word) {
        auto child = currentNode->children.find(token);
        if (child != currentNode->children.end()) {
            parentNode = currentNode;
            currentNode = child->second;
        } else
            return false;
    }
    if (currentNode->isWordEnd) {
        if (currentNode->children.empty())
            parentNode->removeChild(currentNode->token);
        else
            currentNode->isWordEnd = false;
        return true;
    } else
        return false;
}

void Trie::clear() {
    root.clear();
}

vector<vector<int>> Trie::matchForStart(const vector<int> &sentence, size_t start, bool allowSubMatch) const {
    vector<int> tmp;
    vector<vector<int>> result;
    auto *currentNode = &root;
    for (size_t i = start; i < sentence.size(); i++) {
        auto child = currentNode->children.find(sentence[i]);
        if (child != currentNode->children.end()) {
            currentNode = child->second;
            tmp.push_back(currentNode->token);
            if (currentNode->isWordEnd) {
                // move temporary memory to result
                result.push_back(tmp);
            }
        } else
            break;
    }
#ifdef DEBUG
    cout << "Matched results:" << endl;
    for (auto &r : result)
        cout << fmt::format("[{}]", fmt::join(r.begin(), r.end(), ",")) << endl;
#endif
    if (not allowSubMatch and not result.empty())
        result.erase(result.begin(), result.begin() + result.size() - 1);
    return move(result);
}

unordered_map<size_t, vector<vector<int>>> Trie::matchForAll(const vector<int> &sentence, bool allowSubMatch) const {
    unordered_map<size_t, vector<vector<int>>> result;
    for (size_t i = 0; i < sentence.size();) {
        vector<vector<int>> matches = move(matchForStart(sentence, i, allowSubMatch));
        if (not matches.empty()) {
            size_t match_size = matches.front().size();
            result.emplace(i, matches);
            i += match_size;
        } else
            i++;
    }
    return move(result);
}

void Trie::initializeFromString(const string &content) {
    root.initializeFromString(content);
}

string Trie::serialize() const {
    return move(root.serialize());
}

void KnowledgeBase::clearDisabledEdges() {
    isEdgeDisabled.clear();
    isEdgeDisabled.resize(edges.size(), false);
}

void KnowledgeBase::disableAllEdges() {
    isEdgeDisabled.clear();
    isEdgeDisabled.resize(edges.size(), true);
}

void KnowledgeBase::disableEdgesWithWeightBelow(float minWeight) {
    for (size_t edgeIndex = 0; edgeIndex < edges.size(); edgeIndex++) {
        if (get<3>(edges[edgeIndex]) < minWeight)
            isEdgeDisabled[edgeIndex] = true;
    }
}

void KnowledgeBase::disableEdgesOfRelationships(const vector<string> &rel) {
    unordered_set<string> disabledSet(rel.begin(), rel.end());
    unordered_set<long> disabledIds;
    for (long relationshipId = 0; relationshipId < relationships.size(); relationshipId++) {
        if (disabledSet.find(relationships[relationshipId]) != disabledSet.end())
            disabledIds.insert(relationshipId);
    }
    for (size_t edgeIndex = 0; edgeIndex < edges.size(); edgeIndex++) {
        if (disabledIds.find(get<1>(edges[edgeIndex])) != disabledIds.end())
            isEdgeDisabled[edgeIndex] = true;
    }
}

void KnowledgeBase::enableEdgesOfRelationships(const vector<string> &rel) {
    unordered_set<string> enabledSet(rel.begin(), rel.end());
    unordered_set<long> enabledIds;
    for (long relationshipId = 0; relationshipId < relationships.size(); relationshipId++) {
        if (enabledSet.find(relationships[relationshipId]) != enabledSet.end())
            enabledIds.insert(relationshipId);
    }
    for (size_t edgeIndex = 0; edgeIndex < edges.size(); edgeIndex++) {
        if (enabledIds.find(get<1>(edges[edgeIndex])) == enabledIds.end())
            isEdgeDisabled[edgeIndex] = true;
    }
}

void KnowledgeBase::disableEdgesOfNodes(const vector<string> &nod) {
    unordered_set<string> disabledSet(nod.begin(), nod.end());
    unordered_set<long> disabledIds;
    for (long nodeId = 0; nodeId < nodes.size(); nodeId++) {
        if (disabledSet.find(nodes[nodeId]) != disabledSet.end()) {
#ifdef DEBUG
            cout << fmt::format("Found node to be disabled [{}:{}]", nodes[nodeId], nodeId) << endl;
#endif
            disabledIds.insert(nodeId);
        }
    }
#ifdef DEBUG
    cout << "Begin disabling edges" << endl;
#endif
    for (size_t edgeIndex = 0; edgeIndex < edges.size(); edgeIndex++) {
        if (disabledIds.find(get<0>(edges[edgeIndex])) != disabledIds.end() ||
            disabledIds.find(get<2>(edges[edgeIndex])) != disabledIds.end()) {
#ifdef DEBUG
            cout << fmt::format("{}: [{}:{}] --[{}:{}]--> [{}:{}]",
                                edgeIndex,
                                nodes[get<0>(edges[edgeIndex])], get<0>(edges[edgeIndex]),
                                relationships[get<1>(edges[edgeIndex])], get<1>(edges[edgeIndex]),
                                nodes[get<2>(edges[edgeIndex])], get<2>(edges[edgeIndex])) << endl;
#endif
            isEdgeDisabled[edgeIndex] = true;
        }
    }
#ifdef DEBUG
    cout << "End disabling edges" << endl;
#endif
}

vector<long> KnowledgeBase::findNodes(const vector<string> &nod, bool quiet) const {
    vector<long> ids;
    for (auto &nName : nod) {
        bool found = false;
        for (long nodeId = 0; nodeId < nodes.size(); nodeId++) {
            if (nodes[nodeId] == nName) {
                ids.push_back(nodeId);
                found = true;
            }
        }
        if (not found) {
            if (not quiet)
                throw invalid_argument(fmt::format("Node {} not found", nName));
            else
                ids.push_back(-1);
        }

    }
    return move(ids);
}

vector<Edge> KnowledgeBase::getEdges(long source, long target) const {
    vector<Edge> result;
    if (source != -1 && target == -1) {
        if (edgeFromSource.find(source) != edgeFromSource.end())
            for (size_t edgeIndex: edgeFromSource.at(source))
                result.push_back(edges[edgeIndex]);
    } else if (source == -1 && target != -1) {
        if (edgeToTarget.find(target) != edgeToTarget.end())
            for (size_t edgeIndex: edgeToTarget.at(target))
                result.push_back(edges[edgeIndex]);
    } else if (edgeFromSource.find(source) != edgeFromSource.end())
        for (size_t edgeIndex : edgeFromSource.at(source))
            if (get<2>(edges[edgeIndex]) == target)
                result.push_back(edges[edgeIndex]);

    return move(result);
}

const vector<string> &KnowledgeBase::getNodes() const {
    return nodes;
}

vector<string> KnowledgeBase::getNodes(const vector<long> &nodeIndexes) const {
    vector<string> result;
    for (long nodeIdx : nodeIndexes) {
        result.push_back(nodes[nodeIdx]);
    }
    return result;
}

void KnowledgeBase::addCompositeNode(const string &compositeNode,
                                     const string &relationship,
                                     const vector<int> &tokenizedCompositeNode,
                                     const vector<int> &mask,
                                     const vector<int> &connectionMask,
                                     size_t split_node_minimum_edge_num,
                                     float split_node_minimum_similarity) {
    // Note: The similaity of the composite node to other nodes is computed by:
    // the maximum sub node similarity to other nodes

    long relationId = -1;
    for (long relId = 0; relId < long(relationships.size()); relId++) {
        if (relationships[relId] == relationship) {
            relationId = relId;
            break;
        }
    }
    if (relationId == -1)
        throw invalid_argument(fmt::format("Relationship [{}] not found", relationship));

    long newNodeId = long(nodes.size());
    nodes.emplace_back(compositeNode);
    tokenizedNodes.emplace_back(tokenizedCompositeNode);
    isNodeComposite.push_back(true);
    // Do not update nodeMap and nodeTrie because it is a composite node

    // Find all sub-nodes occuring in the composite node
    // Then add an edge from all sub nodes to the composite node with relationship=relationship

    if (not mask.empty() && mask.size() != tokenizedCompositeNode.size())
        throw invalid_argument(fmt::format(
                "Mask is provided for composite node but size does not match, composite node: {}, mask: {}",
                tokenizedCompositeNode.size(), mask.size()));

    if (not connectionMask.empty() && connectionMask.size() != tokenizedCompositeNode.size())
        throw invalid_argument(fmt::format(
                "Connection mask is provided for composite node but size does not match, composite node: {}, connection mask: {}",
                tokenizedCompositeNode.size(), connectionMask.size()));

    auto result = nodeTrie.matchForAll(tokenizedCompositeNode, false);
    unordered_map<long, float> components;
    unordered_set<long> connectedSource;
    for (auto &subNode : result) {
        size_t subNodeSize = subNode.second.back().size();
        if (isAllMasked(subNode.first, subNode.first + subNodeSize, mask))
            continue;

        long subNodeId = nodeMap.at(subNode.second.back());
        bool hasOut = edgeFromSource.find(subNodeId) != edgeFromSource.end();
        bool hasIn = edgeToTarget.find(subNodeId) != edgeToTarget.end();
        size_t outSize = hasOut ? edgeFromSource.at(subNodeId).size() : 0;
        size_t inSize = hasIn ? edgeToTarget.at(subNodeId).size() : 0;

        if (outSize + inSize < split_node_minimum_edge_num) {
#ifdef DEBUG
            cout << fmt::format("Splitting node [{}:{}]", nodes[subNodeId], subNodeId) << endl;
#endif
            unordered_map<size_t, vector<vector<int>>> subSubMatches = nodeTrie.matchForAll(
                    subNode.second.back(), true);

            for (auto &subSubMatch : subSubMatches) {
                // See normalize match
                for (auto &baseMatch : subSubMatch.second) {
                    if (not isAllMasked(subSubMatch.first + subNode.first,
                                        subSubMatch.first + subNode.first + baseMatch.size(), mask)) {
                        long baseNodeId = nodeMap.at(baseMatch);
                        if (cosineSimilarity(baseNodeId, subNodeId) > split_node_minimum_similarity) {
                            if (connectedSource.find(baseNodeId) == connectedSource.end()) {
                                connectedSource.insert(baseNodeId);
                                if (not isAllMasked(subSubMatch.first + subNode.first,
                                                    subSubMatch.first + subNode.first + baseMatch.size(), connectionMask)) {
                                    addCompositeEdge(baseNodeId, relationId, newNodeId);
                                }
                                components[baseNodeId] += 1;
                                compositeComponentCount[baseNodeId] += 1;
                            }

#ifdef DEBUG
                            cout << fmt::format("Adding component [{}:{}] to composite node [{}:{}]",
                                                nodes[baseNodeId], baseNodeId,
                                                nodes[newNodeId], newNodeId) << endl;
#endif
                        }
                    }
                }
            }
        } else {
            if (connectedSource.find(subNodeId) == connectedSource.end()) {
                connectedSource.insert(subNodeId);
                if (not isAllMasked(subNode.first, subNode.first + subNodeSize, connectionMask)) {
                    addCompositeEdge(subNodeId, relationId, newNodeId);
                }
                components[subNodeId] += 1;
                compositeComponentCount[subNodeId] += 1;
            }

#ifdef DEBUG
            cout << fmt::format("Adding component [{}:{}] to composite node [{}:{}]",
                                nodes[subNodeId], subNodeId,
                                nodes[newNodeId], newNodeId) << endl;
#endif
        }
    }
    compositeNodes.emplace(newNodeId, components);
}

void KnowledgeBase::addCompositeEdge(long sourceNodeId, long relationId, long compositeNodeId) {
    if (sourceNodeId < 0 || sourceNodeId >= nodes.size())
        throw std::invalid_argument(fmt::format("Invalid source node {}", sourceNodeId));
    if (compositeNodeId < 0 || compositeNodeId >= nodes.size() || not isNodeComposite[compositeNodeId])
        throw std::invalid_argument(fmt::format("Invalid target node {}", compositeNodeId));
    size_t edgeIndex = edges.size();
    edges.emplace_back(Edge{sourceNodeId, relationId, compositeNodeId, 1, ""});
    edgeToTarget[compositeNodeId].push_back(edgeIndex);
    edgeFromSource[sourceNodeId].push_back(edgeIndex);
    isEdgeDisabled.push_back(false);
    adjacency[compositeNodeId].insert(sourceNodeId);
    adjacency[sourceNodeId].insert(compositeNodeId);
    tokenizedEdgeAnnotations.emplace_back(vector<int>{});
#ifdef DEBUG
    cout << fmt::format("Connecting node [{}:{}] to composite node [{}:{}] with relation [{}:{}]",
                        nodes[sourceNodeId], sourceNodeId,
                        nodes[compositeNodeId], compositeNodeId,
                        relationships[relationId], relationId) << endl;
#endif
}

void KnowledgeBase::setNodeEmbeddingFileName(const string &path, bool loadEmbeddingToMem) {
    nodeEmbeddingFileName = path;
    loadEmbedding(loadEmbeddingToMem);
}

bool KnowledgeBase::isLandmarkInited() const {
    return not landmarkDistances.empty();
}

void KnowledgeBase::initLandmarks(int seedNum, int landmarkNum, int seed, const string &landmarkPath) {
    if (isLandmarkInited())
        return;
    if (seedNum > nodes.size())
        throw invalid_argument(fmt::format("Seed num {} exceeds node num {}", seedNum, nodes.size()));
    if (landmarkNum > nodes.size())
        throw invalid_argument(fmt::format("Landmark num {} exceeds node num {}", landmarkNum, nodes.size()));
    if (landmarkNum < 4)
        throw invalid_argument(fmt::format("Landmark num {} is too small, use at least 4", landmarkNum));

    string fileName;
    cout << "[KB] Initializing landmark distances" << endl;
    if (landmarkPath.empty()) {
        // create a digestion of the graph, later to be used as the file name
        size_t nodeHash = nodes.size() + edges.size();
        auto hasher = hash<string>();
        for (auto &n : nodes) {
            nodeHash ^= hasher(n) + 0x9e3779b9 + (nodeHash << 6) + (nodeHash >> 2);
        }
        fileName = fmt::format("/tmp/{}-s{}-l{}.cache", nodeHash, seedNum, landmarkNum);
        cout << fmt::format("[KB] Path not specified for landmark cache, use [{}] instead", fileName) << endl;
    } else
        fileName = landmarkPath;
    ifstream landmarkFile(fileName);
    landmarkDistances.clear();
    if (not landmarkFile.fail()) {
        cout << "[KB] Loading landmark distances from file [" << fileName << "]" << endl;
        landmarkFile.close();
        auto file = cista::file(fileName.c_str(), "r");
        auto content = file.content();
        auto ld = cista::deserialize<cista::raw::vector<cista::raw::vector<int>>>(content);
        for (size_t i = 0; i < nodes.size(); i++) {
            landmarkDistances.emplace_back(vector<int>(landmarkNum, 0));
            for (size_t j = 0; j < ld->size(); j++) {
                landmarkDistances[i][j] = (*ld)[j][i];
            }
        }
        cout << "[KB] Landmark loaded" << endl;
    } else {
        // randomly choose seed nodes and perform bfs to find best landmarkDistances
        cout << "[KB] Computing landmark distances with seedNum = "
             << seedNum << " landmarkNum = " << landmarkNum << endl;
        xt::xtensor<int, 2>::shape_type sh = {(size_t) seedNum, nodes.size()};
        xt::xtensor<int, 2> metricsRaw(sh);
        xt::xtensor<float, 1>::shape_type sh2 = {nodes.size()};
        xt::xtensor<float, 1> metrics(sh2);
        unordered_set<long> usedNodes;

        if (seed < 0) {
            random_device rd;
            seed = rd();
        }

        for (int s = 0; s < seedNum; s++) {
            mt19937 gen(seed);
            uniform_int_distribution<long> dist(0, nodes.size() - 1);
            long node = dist(gen);
            while (usedNodes.find(node) != usedNodes.end())
                node = dist(gen);
            usedNodes.insert(node);
        }

        size_t nodeNum = nodes.size();
        vector<long> seedNodes(usedNodes.begin(), usedNodes.end());

#pragma omp parallel for default(none) shared(seedNum, seedNodes, nodeNum, metricsRaw)
        for (int s = 0; s < seedNum; s++)
            xt::view(metricsRaw, s) = xt::adapt(bfsDistances(seedNodes[s]), vector<size_t>{nodeNum});

#pragma omp parallel for default(none) shared(seedNum, nodeNum, metrics, metricsRaw)
        for (int n = 0; n < nodeNum; n++) {
            auto column = xt::view(metricsRaw, xt::all(), n);
            float connectedNum = xt::sum<float>(xt::cast<float>(xt::not_equal(column, DISTANCE_MAX)))[0];
            float disconnectRatio = (seedNum - connectedNum) / seedNum;
            float connectedDist = xt::sum<float>(
                    xt::cast<float>(xt::where(xt::equal(column, DISTANCE_MAX), 0, column)))[0];
            // Use disconnect ratio since it might be possible that some seeds fall onto small disconnected components
            // eg: kukang, kukangs, kuchar are on the same small disconnected component
            metrics[n] = disconnectRatio > 0.1 ? DISTANCE_MAX : connectedDist / connectedNum;
        }
        auto topK = xt::argpartition(metrics, landmarkNum);
        vector<long> landMarks;
        vector<vector<int>> distances(landmarkNum);

#pragma omp parallel for default(none) shared(landmarkNum, distances, nodes, topK, nodeNum, cout)
        for (int i = 0; i < landmarkNum; i++) {
#ifdef DEBUG
            cout << fmt::format("[KB] Select landmark [{}:{}]", nodes[long(topK[i])], long(topK[i])) << endl;
#endif
            distances[i] = move(bfsDistances(long(topK[i])));
        }
        for (size_t i = 0; i < nodes.size(); i++) {
            landmarkDistances.emplace_back(vector<int>(landmarkNum, 0));
            for (size_t j = 0; j < distances.size(); j++) {
                landmarkDistances[i][j] = distances[j][i];
            }
        }
        cout << "[KB] Saving landmark distance to file [" << fileName << "]" << endl;
        auto file = cista::file(fileName.c_str(), "w");
        cista::raw::vector<cista::raw::vector<int>> ld;
        // save as transposed form to improve saving speed
        for (auto &ldd : distances)
            ld.push_back(vector2ArchiveVector(ldd));
        cista::serialize(file, ld);
        cout << "[KB] Landmark saved" << endl;
    }
}

int KnowledgeBase::distance(long node1, long node2, bool fast) const {
    if (isNodeComposite[node1]) {
        int min = 1000000;
        for (auto &subNode : compositeNodes.at(node1)) {
            int dist = distance(subNode.first, node2);
            min = dist > min ? min : dist;
        }
        return min;
    }
    if (isNodeComposite[node2]) {
        int min = 1000000;
        for (auto &subNode : compositeNodes.at(node1)) {
            int dist = distance(node1, subNode.first);
            min = dist > min ? min : dist;
        }
        return min;
    }
    if (not isLandmarkInited())
        throw runtime_error("Initialize landmark distances of the knowledge base first.");
    if (adjacency.empty())
        throw runtime_error("Refresh knowledge base first.");
    // Fast version: Fast shortest path distance estimation in large networks (Potamias, Michalis)
    // ALT distance: Computing the Shortest Path: A∗ Search Meets Graph Theory (Goldberg)
    if (node1 == node2)
        return 0;
    if (fast) {
        if (isNeighbor(node1, node2))
            return 1;
        else
            return (landmarkDistance(node1, node2) + landmarkLowerBound(node1, node2)) / 2 + 1;
    } else {
        if (isNeighbor(node1, node2))
            return 1;
        else {
            return ALTDistance(node1, node2);
        }
    }
}

inline bool KnowledgeBase::isNeighbor(long node1, long node2) const {
    return adjacency.find(node1) != adjacency.end() && adjacency.at(node1).find(node2) != adjacency.at(node1).end();
}

int KnowledgeBase::bfsDistance(long node1, long node2, int maxDepth) const {
    vector<long> current = {node1};
    vector<bool> added;
    int depth = 0;

    // Note: BFS distance here treats the graph as not directional
    while (not current.empty()) {
        vector<long> next;
        added.clear();
        added.resize(nodes.size(), false);
        for (auto currentNode : current) {
            if (currentNode == node2)
                return depth;
            // Iterate all neighbors
            if (adjacency.find(currentNode) != adjacency.end()) {
                for (long nextNode : adjacency.at(currentNode)) {
                    if (not added[nextNode]) {
                        next.push_back(nextNode);
                        added[nextNode] = true;
                    }
                }
            }
        }
        current.swap(next);
        depth++;
        if (depth >= maxDepth)
            break;
    }
    return DISTANCE_MAX;
}

float KnowledgeBase::cosineSimilarity(long node1, long node2) const {
    if (isNodeComposite[node1] or isNodeComposite[node2])
        throw invalid_argument("Composite nodes are not supported");
    if (nodeEmbeddingMem.get() != nullptr)
        return cosineSimilarityFromMem(nodeEmbeddingMem, node1, node2, nodeEmbeddingDim, nodeEmbeddingSimplifyWithInt8);
    else
        return cosineSimilarityFromDataset(nodeEmbeddingDataset, node1, node2, nodeEmbeddingDim,
                                           nodeEmbeddingSimplifyWithInt8);
}

void KnowledgeBase::save(const string &archivePath) const {
    KnowledgeArchive archive;
    if (not compositeNodes.empty())
        throw runtime_error("It's not safe to save after adding composite nodes");

    cout << "[KB] Begin saving" << endl;
    for (auto &ett : edgeToTarget)
        archive.edgeToTarget.emplace(ett.first, vector2ArchiveVector(ett.second));
    for (auto &efs : edgeFromSource)
        archive.edgeFromSource.emplace(efs.first, vector2ArchiveVector(efs.second));
    for (auto &e: edges)
        archive.edges.emplace_back(SerializableEdge{
                get<0>(e), get<1>(e), get<2>(e), get<3>(e), get<4>(e)});
    archive.nodes.set(nodes.begin(), nodes.end());
    archive.relationships.set(relationships.begin(), relationships.end());
    for (auto &tn : tokenizedNodes)
        archive.tokenizedNodes.emplace_back(vector2ArchiveVector(tn));
    for (auto &tr : tokenizedRelationships)
        archive.tokenizedRelationships.emplace_back(vector2ArchiveVector(tr));
    for (auto &tea : tokenizedEdgeAnnotations)
        archive.tokenizedEdgeAnnotations.emplace_back(vector2ArchiveVector(tea));
    archive.rawRelationships.set(rawRelationships.begin(), rawRelationships.end());
    archive.nodeEmbeddingFileName = nodeEmbeddingFileName;
    auto file = cista::file(archivePath.c_str(), "w");
    cista::serialize(file, archive);

    cout << "[KB] Saved node num: " << nodes.size() << endl;
    cout << "[KB] Saved edge num: " << edges.size() << endl;
    cout << "[KB] Saved relation num: " << relationships.size() << endl;
    cout << "[KB] Saved raw relation num: " << rawRelationships.size() << endl;
}

void KnowledgeBase::load(const string &archivePath, bool loadEmbeddingToMem) {
    edgeToTarget.clear();
    edgeFromSource.clear();
    edges.clear();
    isEdgeDisabled.clear();

    relationships.clear();
    rawRelationships.clear();

    nodes.clear();
    nodeTrie.clear();
    nodeMap.clear();
    isNodeComposite.clear();
    compositeNodes.clear();

    tokenizedNodes.clear();
    tokenizedRelationships.clear();
    tokenizedEdgeAnnotations.clear();

    nodeEmbeddingFile = nullptr;
    nodeEmbeddingDataset = nullptr;
    nodeEmbeddingMem = nullptr;

    cout << "[KB] Begin loading" << endl;
    auto file = cista::file(archivePath.c_str(), "r");
    auto content = file.content();
    auto *archive = cista::deserialize<KnowledgeArchive>(content);
    for (auto &ett : archive->edgeToTarget)
        edgeToTarget[ett.first] = archiveVector2Vector(ett.second);
    for (auto &efs : archive->edgeFromSource)
        edgeFromSource[efs.first] = archiveVector2Vector(efs.second);
    for (auto &e: archive->edges)
        edges.emplace_back(make_tuple(e.source, e.relation, e.target, e.weight, e.annotation));
    nodes.insert(nodes.end(), archive->nodes.begin(), archive->nodes.end());
    relationships.insert(relationships.end(), archive->relationships.begin(), archive->relationships.end());
    for (auto &tn : archive->tokenizedNodes)
        tokenizedNodes.emplace_back(archiveVector2Vector(tn));
    for (auto &tr : archive->tokenizedRelationships)
        tokenizedRelationships.emplace_back(archiveVector2Vector(tr));
    for (auto &tea : archive->tokenizedEdgeAnnotations)
        tokenizedEdgeAnnotations.emplace_back(archiveVector2Vector(tea));
    rawRelationships.insert(rawRelationships.end(), archive->rawRelationships.begin(), archive->rawRelationships.end());
    for (long index = 0; index < tokenizedNodes.size(); index++) {
        nodeTrie.insert(tokenizedNodes[index]);
        nodeMap[tokenizedNodes[index]] = index;
    }
    isNodeComposite.resize(nodes.size(), false);
    isEdgeDisabled.resize(edges.size(), false);
    nodeEmbeddingFileName = archive->nodeEmbeddingFileName;
    refresh(loadEmbeddingToMem);
    cout << "[KB] Loaded node num: " << nodes.size() << endl;
    cout << "[KB] Loaded edge num: " << edges.size() << endl;
    cout << "[KB] Loaded relation num: " << relationships.size() << endl;
    cout << "[KB] Loaded raw relation num: " << rawRelationships.size() << endl;
    cout << "[KB] Loading finished" << endl;
}

void KnowledgeBase::refresh(bool loadEmbeddingToMem) {
    loadEmbedding(loadEmbeddingToMem);
    loadAdjacency();
}

template<typename T1, typename T2>
bool KnowledgeBase::PriorityCmp::operator()(const pair<T1, T2> &pair1, const pair<T1, T2> &pair2) {
    return pair1.first > pair2.first;
}

size_t KnowledgeBase::VectorHash::operator()(const vector<int> &vec) const {
    // A simple hash function
    // https://stackoverflow.com/questions/20511347/a-good-hash-function-for-a-vector
    size_t seed = vec.size();
    for (auto &i : vec) {
        seed ^= i + 0x9e3779b9 + (seed << 6) + (seed >> 2);
    }
    return seed;
}

void KnowledgeBase::loadEmbedding(bool loadEmbeddingToMem) {
    nodeEmbeddingFile.reset();
    nodeEmbeddingDataset.reset();
    nodeEmbeddingMem.reset();
    if (not nodeEmbeddingFileName.empty()) {
        ifstream tmpFile(nodeEmbeddingFileName);
        cout << "[KB] Embedding configured, using embedding file [" << nodeEmbeddingFileName << "]" << endl;
        if (tmpFile.fail())
            cout << "[KB] Failed to load embedding file [" << nodeEmbeddingFileName << "], skipped" << endl;
        else {
            tmpFile.close();
            nodeEmbeddingFile = make_shared<HighFive::File>(nodeEmbeddingFileName, HighFive::File::ReadOnly);
            nodeEmbeddingDataset = make_shared<HighFive::DataSet>(nodeEmbeddingFile->getDataSet("embeddings"));
            auto shape = nodeEmbeddingDataset->getDimensions();
            auto dtype = nodeEmbeddingDataset->getDataType();
            if (shape.size() != 2)
                throw invalid_argument(
                        fmt::format("Knowledge base embedding should be 2-dimensional, but got shape [{}]",
                                    fmt::join(shape.begin(), shape.end(), ",")));
            nodeEmbeddingDim = shape[1];
            nodeEmbeddingSimplifyWithInt8 = dtype.getClass() == HighFive::DataTypeClass::Integer;

            if (loadEmbeddingToMem) {
                cout << "[KB] Loading embedding to memory" << endl;
                if (nodeEmbeddingSimplifyWithInt8) {
                    nodeEmbeddingMem = shared_ptr<int8_t[]>(new int8_t[shape[0] * shape[1]]);
                    nodeEmbeddingDataset->read(static_pointer_cast<int8_t[]>(nodeEmbeddingMem).get());
                } else {
                    nodeEmbeddingMem = shared_ptr<float[]>(new float[shape[0] * shape[1]]);
                    nodeEmbeddingDataset->read(static_pointer_cast<float[]>(nodeEmbeddingMem).get());
                }
                cout << "[KB] Closing embedding file" << endl;
                nodeEmbeddingDataset.reset();
                nodeEmbeddingFile.reset();
            }
        }
    } else {
        cout << "[KB] Embedding not configured, skipped" << endl;
    }
}

void KnowledgeBase::loadAdjacency() {
    for (auto &srcNodeAdj : edgeFromSource) {
        for (auto edgeIndex : srcNodeAdj.second) {
            adjacency[srcNodeAdj.first].insert(get<2>(edges[edgeIndex]));
        }
    }
    for (auto &tarNodeAdj : edgeToTarget) {
        for (auto edgeIndex : tarNodeAdj.second) {
            adjacency[tarNodeAdj.first].insert(get<0>(edges[edgeIndex]));
        }
    }
}

vector<int> KnowledgeBase::bfsDistances(long node) const {
    vector<int> distances(nodes.size(), DISTANCE_MAX);

    vector<long> current = {node};
    vector<bool> added;
    int depth = 0;

    // Note: BFS distance here treats the graph as not directional
    while (not current.empty()) {
        vector<long> next;
        added.clear();
        added.resize(nodes.size(), false);
        for (auto currentNode : current) {
            distances[currentNode] = depth;

            // Iterate all neighbors
            if (edgeFromSource.find(currentNode) != edgeFromSource.end()) {
                for (auto edgeIndex : edgeFromSource.at(currentNode)) {
                    long nextNode = get<2>(edges[edgeIndex]);
                    if (distances[nextNode] == DISTANCE_MAX and
                        not added[nextNode]) {
                        next.push_back(nextNode);
                        added[nextNode] = true;
                    }
                }
            }
            if (edgeToTarget.find(currentNode) != edgeToTarget.end()) {
                for (auto edgeIndex : edgeToTarget.at(currentNode)) {
                    long nextNode = get<0>(edges[edgeIndex]);
                    if (distances[nextNode] == DISTANCE_MAX and
                        not added[nextNode]) {
                        next.push_back(nextNode);
                        added[nextNode] = true;
                    }
                }
            }
        }
        current.swap(next);
        depth++;
    }
    return move(distances);
}

int KnowledgeBase::landmarkDistance(long node1, long node2) const {
    size_t landmarkNum = landmarkDistances[node1].size();
    vector<size_t> shape = {landmarkNum};
    auto landmarkDistance1 = xt::adapt(landmarkDistances[node1], shape);
    auto landmarkDistance2 = xt::adapt(landmarkDistances[node2], shape);
    return xt::amin(landmarkDistance1 + landmarkDistance2)[0];
}

int KnowledgeBase::landmarkLowerBound(long node, long targetNode) const {
    // Use a subset of landmarks, top 4 in this case, to speed up
    vector<size_t> shape = {4};
    auto landmarkDistance1 = xt::adapt(landmarkDistances[node], shape);
    auto landmarkDistance2 = xt::adapt(landmarkDistances[targetNode], shape);
    return xt::amax(xt::abs(landmarkDistance1 - landmarkDistance2))[0];
}

int KnowledgeBase::ALTDistance(long node1, long node2) const {
    vector<int> costSoFar(nodes.size(), DISTANCE_MAX);
    priority_queue<pair<int, long>, vector<pair<int, long>>, PriorityCmp> queue;

    queue.push(make_pair(0, node1));
    costSoFar[node1] = 0;
    size_t count = 0;

    while (not queue.empty()) {
        long currentNode = queue.top().second;
        queue.pop();
        if (currentNode == node2) {
            return costSoFar[currentNode];
        }
        // Iterate all neighbors
        if (adjacency.find(currentNode) != adjacency.end()) {
            // Note: 1 hop away, so added cost is 1
            int newCost = costSoFar[currentNode] + 1;
            for (long nextNode : adjacency.at(currentNode)) {
                if (newCost < costSoFar[nextNode]) {
                    costSoFar[nextNode] = newCost;
                    int priority = newCost + landmarkLowerBound(nextNode, node2);
                    // Even if nextNode is in queue, if we have a higher priority,
                    // this new insertion will com first, and the lower priority version will have no effect
                    queue.push(make_pair(priority, nextNode));
                }
            }
        }
    }
    throw runtime_error("Failed to perform A*, this shouldn't happen.");
}

inline float
KnowledgeBase::cosineSimilarityFromDataset(const shared_ptr<HighFive::DataSet> &embedding, long node1, long node2,
                                           size_t dim,
                                           bool simplifyWithInt8) {
    if (simplifyWithInt8) {
        xt::xtensor<int8_t, 1>::shape_type sh({dim});
        xt::xtensor<int8_t, 1> tmp1(sh), tmp2(sh);
        embedding->select({(size_t) node1, 0}, {1, dim}).read(tmp1.data());
        auto srcEmbed = xt::cast<int16_t>(tmp1);
        embedding->select({(size_t) node2, 0}, {1, dim}).read(tmp2.data());
        auto tarEmbed = xt::cast<int16_t>(tmp2);
        // cosine similarity
        float dot = xt::sum<int16_t>(srcEmbed * tarEmbed)[0];
        float norm1 = xt::sqrt(xt::cast<float>(xt::sum<int16_t>(srcEmbed * srcEmbed)))[0];
        float norm2 = xt::sqrt(xt::cast<float>(xt::sum<int16_t>(tarEmbed * tarEmbed)))[0];
        return dot / (norm1 * norm2);
    } else {
        xt::xtensor<float, 1>::shape_type sh({dim});
        xt::xtensor<float, 1> srcEmbed(sh), tarEmbed(sh);
        embedding->select({(size_t) node1, 0}, {1, dim}).read(srcEmbed.data());
        embedding->select({(size_t) node2, 0}, {1, dim}).read(tarEmbed.data());
        // cosine similarity
        float dot = xt::sum<float>(srcEmbed * tarEmbed)[0];
        float norm1 = xt::sqrt(xt::sum<float>(srcEmbed * srcEmbed))[0];
        float norm2 = xt::sqrt(xt::sum<float>(tarEmbed * tarEmbed))[0];
        return dot / (norm1 * norm2);
    }
}

inline float
KnowledgeBase::cosineSimilarityFromMem(const shared_ptr<void> &embedding, long node1, long node2, size_t dim,
                                       bool simplifyWithInt8) {
    if (simplifyWithInt8) {
        const int8_t *emb = static_pointer_cast<int8_t[]>(embedding).get();
        auto srcEmbed = xt::cast<int16_t>(xt::adapt(emb + node1 * dim, dim, xt::no_ownership(),
                                                    vector<size_t>{dim}));
        auto tarEmbed = xt::cast<int16_t>(xt::adapt(emb + node2 * dim, dim, xt::no_ownership(),
                                                    vector<size_t>{dim}));
        // cosine similarity
        float dot = xt::sum<int16_t>(srcEmbed * tarEmbed)[0];
        float norm1 = xt::sqrt(xt::cast<float>(xt::sum<int16_t>(srcEmbed * srcEmbed)))[0];
        float norm2 = xt::sqrt(xt::cast<float>(xt::sum<int16_t>(tarEmbed * tarEmbed)))[0];
        return dot / (norm1 * norm2);
    } else {
        float *emb = static_pointer_cast<float[]>(embedding).get();
        auto srcEmbed = xt::adapt(emb + node1 * dim, dim, xt::no_ownership(), vector<size_t>{dim});
        auto tarEmbed = xt::adapt(emb + node2 * dim, dim, xt::no_ownership(), vector<size_t>{dim});
        // cosine similarity
        float dot = xt::sum<float>(srcEmbed * tarEmbed)[0];
        float norm1 = xt::sqrt(xt::sum<float>(srcEmbed * srcEmbed))[0];
        float norm2 = xt::sqrt(xt::sum<float>(tarEmbed * tarEmbed))[0];
        return dot / (norm1 * norm2);
    }
}

template<typename T>
cista::raw::vector<T> KnowledgeBase::vector2ArchiveVector(const vector<T> &vec) {
    return move(cista::raw::vector<T>(vec.begin(), vec.end()));
}

template<typename T>
vector<T> KnowledgeBase::archiveVector2Vector(const cista::raw::vector<T> &vec) {
    return move(vector<T>(vec.begin(), vec.end()));
}

KnowledgeMatcher::KnowledgeMatcher(const KnowledgeBase &knowledgeBase) {
    cout << "[KM] Initializing matcher from knowledge base" << endl;
    kb = knowledgeBase;
    cout << "[KM] Matcher initialized" << endl;
}

KnowledgeMatcher::KnowledgeMatcher(const string &archivePath) {
    cout << "[KM] Initializing matcher from knowledge base archive" << endl;
    load(archivePath);
    cout << "[KM] Matcher initialized" << endl;
}

void KnowledgeMatcher::setCorpus(const std::vector<std::vector<int>> &corpus) {
    isCorpusSet = true;
    documentCountOfNodeInCorpus.clear();
    corpusSize = corpus.size();
    tqdm bar;
    bar.set_theme_basic();
    bar.disable_colors();
    cout << "Begin processing corpus." << endl;
    size_t processed = 0;
    for (auto &document : corpus) {
        auto result = kb.nodeTrie.matchForAll(document, false);
        unordered_set<long> insertedSubNodes;
        for (auto &subNode : result) {
            long subNodeId = kb.nodeMap.at(subNode.second.back());
            documentCountOfNodeInCorpus[subNodeId] += 1;
        }
        processed++;
        bar.progress(processed, corpus.size());
    }
    bar.finish();
}

std::string KnowledgeMatcher::findClosestConcept(string targetConcept, const vector<string> &concepts) {
    auto targetIdVec = kb.findNodes({targetConcept});
    long targetId = targetIdVec[0];
    vector<long> conceptIds = kb.findNodes(concepts, true);
    size_t bestConceptIdx = 0;
    float bestSimilarity = -1;
    for(size_t i = 0; i < conceptIds.size(); i++) {
        if (conceptIds[i] != -1) {
            float similarity = kb.cosineSimilarity(targetId, conceptIds[i]);
            if (similarity > bestSimilarity) {
                bestConceptIdx = i;
                bestSimilarity = similarity;
            }
        }
    }
    return concepts[bestConceptIdx];
}

KnowledgeMatcher::MatchResult
KnowledgeMatcher::matchByNodeEmbedding(const vector<int> &sourceSentence,
                                       const vector<int> &targetSentence,
                                       const vector<int> &sourceMask,
                                       const vector<int> &targetMask,
                                       const vector<long> &disableNodes,
                                       int maxTimes, int maxDepth, int seed,
                                       int edgeTopK, int sourceContextRange, bool trim,
                                       size_t split_node_minimum_edge_num,
                                       float split_node_minimum_similarity,
                                       float stopSearchingEdgeIfSimilarityBelow,
                                       float sourceContextWeight) const {
    if (kb.nodeEmbeddingFileName.empty())
        throw invalid_argument("Knowledge base doesn't have an embedding file.");

#ifdef DEBUG
    cout << "================================================================================" << endl;
    cout << "Method: match by node embedding" << endl;
    cout << fmt::format("sourceSentence: [{}]",
                        fmt::join(sourceSentence.begin(), sourceSentence.end(), ",")) << endl;
    cout << fmt::format("targetSentence: [{}]",
                        fmt::join(targetSentence.begin(), targetSentence.end(), ",")) << endl;
    cout << fmt::format("sourceMask: [{}]",
                        fmt::join(sourceMask.begin(), sourceMask.end(), ",")) << endl;
    cout << fmt::format("targetMask: [{}]",
                        fmt::join(targetMask.begin(), targetMask.end(), ",")) << endl;
    cout << fmt::format("disabledNodes: [{}]",
                        fmt::join(targetMask.begin(), targetMask.end(), ",")) << endl;
    cout << "maxTimes: " << maxTimes << endl;
    cout << "maxDepth: " << maxDepth << endl;
    cout << "seed: " << seed << endl;
    cout << "edgeTopK: " << edgeTopK << endl;
    cout << "sourceContextRange: " << sourceContextRange << endl;
    cout << "trim: " << trim << endl;
    cout << "stopSearchingEdgeIfSimilarityBelow: " << stopSearchingEdgeIfSimilarityBelow << endl;
    cout << "================================================================================" << endl;
#endif
    // start token position of the node, tokens made up of the node
    unordered_map<size_t, vector<int>> sourceMatch, targetMatch;
    matchForSourceAndTarget(sourceSentence,
                            targetSentence,
                            sourceMask,
                            targetMask,
                            sourceMatch,
                            targetMatch,
                            split_node_minimum_edge_num,
                            split_node_minimum_similarity);

    if (sourceMatch.empty()) {
#ifdef DEBUG
        cout << "Source match result is empty, return" << endl;
#endif
        return move(MatchResult());
    }
    if (targetMatch.empty()) {
#ifdef DEBUG
        cout << "Target match result is empty, return" << endl;
#endif
        return move(MatchResult());
    }

    unordered_set<long> disabledNodeSet(disableNodes.begin(), disableNodes.end());

    // node ids in targetSentence (or sourceSentence if targetSentence is empty), their occurence times
    unordered_map<long, float> targetNodes;

    // random walk, transition possibility determined by similarity metric.
    MatchResult result;

    // node id, start pos, end pos
    vector<tuple<long, size_t, size_t>> nodes;
    vector<vector<long>> nodeContextNeighbors;

    unordered_map<pair<long, long>, float, PairHash> similarityCache;
    unordered_set<long> visitedEndNodes;

    // Convert source match result to source nodes and find contextual neighbors for each source node
    for (auto &sm : sourceMatch) {
        nodes.emplace_back(kb.nodeMap.at(sm.second), sm.first, sm.first + sm.second.size());
    }

    sort(nodes.begin(), nodes.end(), [](const tuple<long, size_t, size_t> &node1,
                                        const tuple<long, size_t, size_t> &node2) {
        return get<1>(node1) < get<1>(node2);
    });
    for (int i = 0; i < int(nodes.size()); i++) {
        vector<long> neighbors;
        for (int j = i - sourceContextRange; j <= i + sourceContextRange && sourceContextRange > 0; j++) {
            if (j >= 0 && j != i && j < int(nodes.size())) {
                neighbors.push_back(get<0>(nodes[j]));
            }
        }
        nodeContextNeighbors.emplace_back(neighbors);
    }

    for (auto &tm : targetMatch)
        targetNodes[kb.nodeMap.at(tm.second)] += 1;

    result.targetNodeNum = targetNodes.size();

    if (seed < 0) {
        random_device rd;
        seed = rd();
    }

    if (maxTimes < 2 * int(nodes.size()))
        cout << "Parameter maxTimes " << maxTimes << " is smaller than 2 * node size " << nodes.size()
             << ", which may result in insufficient exploration, consider increase maxTimes." << endl;

    float compositeNodesCountSum = 0;
    float corpusNodesCountSum = 0;
    for (auto &count : documentCountOfNodeInCorpus)
        corpusNodesCountSum += count.second;
    for (auto &count : kb.compositeComponentCount)
        compositeNodesCountSum += count.second;

    // https://stackoverflow.com/questions/43168661/openmp-and-reduction-on-stdvector
#pragma omp declare reduction(vsg_join : VisitedSubGraph : joinVisitedSubGraph(omp_out, omp_in)) \
                        initializer(omp_priv = omp_orig)

#pragma omp parallel for reduction(vsg_join : visitedSubGraph) \
    default(none) \
    shared(disabledNodes, \
           nodes, sourceMatch, targetNodes, posRef, cout, edgeTopK, discardEdgesIfSimilarityBelow, \
           corpusNodesCountSum, compositeNodesCountSum) \
    firstprivate(seed, maxTimes, maxDepth, similarityCache)

    for (int i = 0; i < maxTimes; i++) {
        mt19937 gen(seed ^ i);

        // uniform sampling for efficient parallelization
        size_t nodeLocalIndex;
        long rootNode, currentNode;

        uniform_int_distribution<size_t> nodeDist(0, nodes.size() - 1);
        nodeLocalIndex = nodeDist(gen);

        rootNode = currentNode = get<0>(nodes[nodeLocalIndex]);
        auto &neighbors = nodeContextNeighbors[nodeLocalIndex];

        VisitedPath path;
        path.round = i;
        path.root = rootNode;
        path.rootStartPos = get<1>(nodes[nodeLocalIndex]);
        path.rootEndPos = get<2>(nodes[nodeLocalIndex]);
        path.matchedFocusCount = 0;
        path.visitedNodes.insert(currentNode);

        unordered_map<long, float> oldTargetNodes = targetNodes;
        // Only works for first node
        for (auto neighbor : neighbors)
            targetNodes[neighbor] += sourceContextWeight;

#ifdef DEBUG_DECISION
        cout << fmt::format("Round {}", i) << endl;
        cout << "Compare target:" << endl;
        for (auto &tNode : targetNodes)
            cout << fmt::format("[{}:{}] {}", kb.nodes[tNode.first], tNode.first, tNode.second) << endl;
        cout << "================================================================================" << endl;
#endif
        for (int d = 0; d < maxDepth; d++) {
            vector<float> sim;
            vector<long> simTarget;
            bool hasOut = kb.edgeFromSource.find(currentNode) != kb.edgeFromSource.end();
            bool hasIn = kb.edgeToTarget.find(currentNode) != kb.edgeToTarget.end();
            size_t outSize = hasOut ? kb.edgeFromSource.at(currentNode).size() : 0;
            size_t inSize = hasIn ? kb.edgeToTarget.at(currentNode).size() : 0;
            sim.resize(outSize + inSize, 0);
            simTarget.resize(outSize + inSize, -1);

#ifdef DEBUG_DECISION
            cout << fmt::format("Current node: [{}:{}]", kb.nodes[currentNode], currentNode) << endl;
#endif
            for (size_t j = 0; j < outSize + inSize; j++) {
                size_t edgeIndex = j < outSize ?
                                   kb.edgeFromSource.at(currentNode)[j] :
                                   kb.edgeToTarget.at(currentNode)[j - outSize];
                auto &edge = kb.edges[edgeIndex];
                // disable disabled edges, reflexive edges & path to visted nodes
                long otherNodeId = j < outSize ? get<2>(edge) : get<0>(edge);

                if (kb.isEdgeDisabled[edgeIndex] ||
                    (path.visitedNodes.find(otherNodeId) != path.visitedNodes.end()) ||
                    (disabledNodeSet.find(otherNodeId) != disabledNodeSet.end())) {
                    sim[j] = 0;
#ifdef DEBUG_DECISION
                    cout << fmt::format("Skipping edge because: edge disabled [{}], node visited [{}], node disabled [{}]",
                                        kb.isEdgeDisabled[edgeIndex],
                                        path.visitedNodes.find(otherNodeId) != path.visitedNodes.end(),
                                        disabledNodeSet.find(otherNodeId) != disabledNodeSet.end()) << endl;
#endif
                } else {
                    float precision = 1e-4, recall = 1e-4,
                            precision_idf_sum = 0, recall_idf_sum = 0;
                    bool isNodeComposite = kb.isNodeComposite[otherNodeId];

                    if (isNodeComposite) {
                        for (auto &tNode : targetNodes) {
                            float subNodeSim = -1, simToEachTarget = -1;
                            long subNodeBest = -1;

                            for (auto &subNode : kb.compositeNodes.at(otherNodeId)) {
                                auto simPair = make_pair(tNode.first, subNode.first);
                                if (similarityCache.find(simPair) == similarityCache.end()) {
                                    simToEachTarget = kb.cosineSimilarity(subNode.first, tNode.first);
                                    similarityCache[simPair] = simToEachTarget;
                                } else {
                                    simToEachTarget = similarityCache.at(simPair);
                                }
                                if (simToEachTarget > subNodeSim) {
                                    subNodeSim = simToEachTarget;
                                    subNodeBest = subNode.first;
                                }
                            }

                            float tfidf = computeTfidf(tNode.first, corpusNodesCountSum,
                                                       targetNodes, documentCountOfNodeInCorpus);

#ifdef DEBUG_DECISION
                            cout << fmt::format("Target node [{}:{}], most similar [{}:{}], sim {}, tfidf {}",
                                                kb.nodes[tNode.first], tNode.first,
                                                kb.nodes[subNodeBest], subNodeBest,
                                                subNodeSim, tfidf) << endl;
#endif
                            recall += subNodeSim * tfidf;
                            recall_idf_sum += tfidf;
                        }

                        for (auto &subNode : kb.compositeNodes.at(otherNodeId)) {
                            float tNodeSim = -1, simToEachTarget = -1;
                            long tNodeBest = -1;

                            for (auto &tNode : targetNodes) {
                                auto simPair = make_pair(tNode.first, subNode.first);
                                if (similarityCache.find(simPair) == similarityCache.end()) {
                                    simToEachTarget = kb.cosineSimilarity(subNode.first, tNode.first);
                                    similarityCache[simPair] = simToEachTarget;
                                } else {
                                    simToEachTarget = similarityCache.at(simPair);
                                }

                                if (simToEachTarget > tNodeSim) {
                                    tNodeSim = simToEachTarget;
                                    tNodeBest = tNode.first;
                                }
                            }

                            float tfidf = computeTfidf(subNode.first, compositeNodesCountSum,
                                                       kb.compositeNodes.at(otherNodeId),
                                                       kb.compositeComponentCount);
#ifdef DEBUG_DECISION
                            cout << fmt::format("Sub node [{}:{}], most similar [{}:{}], sim {}, tfidf {}",
                                                kb.nodes[subNode.first], subNode.first,
                                                kb.nodes[tNodeBest], tNodeBest,
                                                tNodeSim, tfidf) << endl;
#endif
                            precision += tNodeSim * tfidf;
                            precision_idf_sum += tfidf;
                        }
                        recall /= recall_idf_sum;
                        precision /= precision_idf_sum;
                    } else {
                        float tNodeSim = -1, simToEachTarget = -1;
                        long tNodeBest = -1;
                        for (auto &tNode : targetNodes) {
                            auto simPair = make_pair(tNode.first, otherNodeId);
                            if (similarityCache.find(simPair) == similarityCache.end()) {
                                simToEachTarget = kb.cosineSimilarity(otherNodeId, tNode.first);
                                similarityCache[simPair] = simToEachTarget;
                            } else {
                                simToEachTarget = similarityCache.at(simPair);
                            }

                            if (simToEachTarget > tNodeSim) {
                                tNodeSim = simToEachTarget;
                                tNodeBest = tNode.first;
                            }

                            float tfidf = computeTfidf(tNode.first, corpusNodesCountSum,
                                                       targetNodes, documentCountOfNodeInCorpus);

#ifdef DEBUG_DECISION
                            cout << fmt::format("Target node [{}:{}], node [{}:{}], sim {}, tfidf {}",
                                                kb.nodes[tNode.first], tNode.first,
                                                kb.nodes[otherNodeId], otherNodeId,
                                                simToEachTarget, tfidf) << endl;
#endif

                            recall += simToEachTarget * tfidf;
                            recall_idf_sum += tfidf;
                        }

#ifdef DEBUG_DECISION
                        cout << fmt::format("Node [{}:{}], most similar [{}:{}], sim {}, tfidf {}",
                                            kb.nodes[otherNodeId], otherNodeId,
                                            kb.nodes[tNodeBest], tNodeBest,
                                            tNodeSim, 1) << endl;
#endif
                        precision = tNodeSim;
                        recall = recall / recall_idf_sum;
                    }
                    recall = recall > 0 ? recall : 0;
                    precision = precision > 0 ? precision : 0;
#ifdef DEBUG_DECISION
                    cout << fmt::format("recall: {}, precision: {}", recall, precision) << endl;
#endif
                    float simTmp = (5 * recall * precision) / (recall + 4 * precision + 1e-6);
                    if (simTmp < stopSearchingEdgeIfSimilarityBelow)
                        simTmp = 0;
                    sim[j] = simTmp;
                }
#ifdef DEBUG_DECISION
                cout << fmt::format("{}: [{}:{}] --[{}:{}]--> [{}:{}], annotation [{}], "
                                    "similarity {} to target [{}:{}]",
                                    edgeIndex,
                                    kb.nodes[get<0>(edge)], get<0>(edge),
                                    kb.relationships[get<1>(edge)], get<1>(edge),
                                    kb.nodes[get<2>(edge)], get<2>(edge),
                                    edgeToStringAnnotation(edgeIndex),
                                    sim[j],
                                    simTarget[j] == -1 ? "" : kb.nodes[simTarget[j]], simTarget[j]) << endl;
#endif
            }

            if (d == 0)
                targetNodes.swap(oldTargetNodes);

            keepTopK(sim, edgeTopK);
            // break on meeting nodes with no match
            if (all_of(sim.begin(), sim.end(), [](float f) { return f <= 0; }))
                break;

            discrete_distribution<size_t> dist(sim.begin(), sim.end());
            size_t e = dist(gen);
            bool isEdgeOut = e < outSize;
            size_t selectedEdgeIndex = isEdgeOut ?
                                       kb.edgeFromSource.at(currentNode)[e] :
                                       kb.edgeToTarget.at(currentNode)[e - outSize];

            // move to next node
            currentNode = isEdgeOut ?
                          get<2>(kb.edges[selectedEdgeIndex]) :
                          get<0>(kb.edges[selectedEdgeIndex]);
            path.visitedNodes.insert(currentNode);
            path.edges.push_back(selectedEdgeIndex);
            path.similarities[selectedEdgeIndex] = sim[e];
            path.similarityTargets[selectedEdgeIndex] = simTarget[e];

#ifdef DEBUG_DECISION
            cout << endl;
            cout << "Choose edge " << selectedEdgeIndex << endl;
            cout << "Annotation: " << edgeToStringAnnotation(selectedEdgeIndex) << endl;
            cout << "--------------------------------------------------------------------------------" << endl;
#endif
        }
        if (trim)
            trimPath(path);
#ifdef DEBUG_DECISION
        cout << "================================================================================" << endl;
#endif
        result.visitedSubGraph.visitedPaths.push_back(path);
    }

    return move(result);
}

KnowledgeMatcher::MatchResult KnowledgeMatcher::joinMatchResults(const vector<MatchResult> &inMatchResults) const {
    // First assign new round number to each path in each match result,
    // to ensure that they are deterministically ordered, then join visited sub graphs
    // From:
    // [0, 1, 2, ..., 99], [0, 1, 2, ..., 99]
    // To:
    // [0, 1, 2, ..., 99], [100, 101, ..., 199]
    int offset = 0;
    vector<MatchResult> matchResults = inMatchResults;
    MatchResult joinedMatchResult;
    for (auto &result : matchResults) {
        for (auto &path : result.visitedSubGraph.visitedPaths)
            path.round += offset;
        offset += int(result.visitedSubGraph.visitedPaths.size());
        auto &vpIn = joinedMatchResult.visitedSubGraph.visitedPaths;
        auto &vpOut = result.visitedSubGraph.visitedPaths;
        vpIn.insert(vpIn.end(), vpOut.begin(), vpOut.end());
    }

    // Other properties of VisitedSubGraph are meant to be used by selectPaths and thus there is no need to join
    // Sort according round number to ensure deterministic ordering
    sort(joinedMatchResult.visitedSubGraph.visitedPaths.begin(),
         joinedMatchResult.visitedSubGraph.visitedPaths.end(),
         [](const VisitedPath &p1, const VisitedPath &p2) { return p1.round < p2.round; });

    return move(joinedMatchResult);
}

void KnowledgeMatcher::save(const string &archivePath) const {
    kb.save(archivePath);
}

void KnowledgeMatcher::load(const string &archivePath, bool loadEmbeddingToMem) {
    kb.load(archivePath, loadEmbeddingToMem);
}

template<typename T1, typename T2>
size_t KnowledgeMatcher::PairHash::operator()(const pair<T1, T2> &pair) const {
    return (hash<T1>()(pair.first) << 32) | hash<T2>()(pair.second);
}

vector<string> KnowledgeMatcher::matchResultPathsToStrings(const MatchResult &matchResult) const {
    vector<string> result;
    for (auto &path : matchResult.visitedSubGraph.visitedPaths) {
        vector<string> edgeStrings;
        for (size_t edgeIndex : path.edges) {
            edgeStrings.emplace_back(edgeToStringAnnotation(edgeIndex));
        }
        result.emplace_back(fmt::format("{}", fmt::join(edgeStrings.begin(), edgeStrings.end(), ",")));
    }
    return move(result);
}

vector<int> KnowledgeMatcher::edgeToAnnotation(size_t edgeIndex) const {
    const Edge &edge = kb.edges[edgeIndex];
    vector<int> edgeAnno(kb.tokenizedNodes[get<0>(edge)]);
    auto &rel = kb.tokenizedRelationships[get<1>(edge)];
    edgeAnno.insert(edgeAnno.end(), rel.begin(), rel.end());
    auto &tar = kb.tokenizedNodes[get<2>(edge)];
    edgeAnno.insert(edgeAnno.end(), tar.begin(), tar.end());
    return move(edgeAnno);
}

string KnowledgeMatcher::edgeToStringAnnotation(size_t edgeIndex) const {
    const Edge &edge = kb.edges[edgeIndex];
    string edgeAnno(kb.nodes[get<0>(edge)]);
    edgeAnno += " ";
    auto &rel = kb.relationships[get<1>(edge)];
    edgeAnno.insert(edgeAnno.end(), rel.begin(), rel.end());
    edgeAnno += " ";
    auto &tar = kb.nodes[get<2>(edge)];
    edgeAnno.insert(edgeAnno.end(), tar.begin(), tar.end());
    return move(edgeAnno);
}

float KnowledgeMatcher::computeTfidf(long node, float documentNodeCountSum,
                                     const unordered_map<long, float> &nodeCount,
                                     const unordered_map<long, float> &documentNodeCount) const {
    if (not isCorpusSet)
        return 1;
    float documentCount = 1;
    if (documentNodeCount.find(node) != documentNodeCount.end())
        documentCount += documentNodeCount.at(node);

    float idf = log(documentNodeCountSum / documentCount);

    float countSum = 0;
    for (auto &nc : nodeCount)
        countSum += nc.second;
    float tf = nodeCount.at(node) / countSum;
    return tf * idf;
}

void KnowledgeMatcher::matchForSourceAndTarget(const vector<int> &sourceSentence,
                                               const vector<int> &targetSentence,
                                               const vector<int> &sourceMask,
                                               const vector<int> &targetMask,
                                               unordered_map<size_t, vector<int>> &sourceMatch,
                                               unordered_map<size_t, vector<int>> &targetMatch,
                                               size_t split_node_minimum_edge_num,
                                               float split_node_minimum_similarity) const {
    if (not sourceMask.empty() && sourceMask.size() != sourceSentence.size())
        throw invalid_argument(fmt::format(
                "Mask is provided for source sentence but size does not match, sentence: {}, mask: {}",
                sourceSentence.size(), sourceMask.size()));

    if (not targetMask.empty() && targetMask.size() != targetSentence.size())
        throw invalid_argument(fmt::format(
                "Mask is provided for target sentence but size does not match, sentence: {}, mask: {}",
                targetSentence.size(), targetMask.size()));

#ifdef DEBUG
    cout << "Begin node matching for source and target sentence" << endl;
#endif
    unordered_map<size_t, vector<vector<int>>> sourceMatches = kb.nodeTrie.matchForAll(sourceSentence, false);
    if (not sourceMask.empty()) {
        for (auto &matches : sourceMatches) {
            // Check if there exists any not masked position
            bool allMasked = true;
            for (size_t i = 0; i < matches.second.back().size(); i++) {
                if (sourceMask[matches.first + i] == 1) {
                    allMasked = false;
                }
            }
            if (allMasked) {
#ifdef DEBUG
                cout << fmt::format("Removing match [{}]@{} in source sentence",
                                    fmt::join(matches.second.back().begin(),
                                              matches.second.back().end(), ","), matches.first) << endl;
#endif
            } else
                normalizeMatch(sourceMatch, sourceMask, matches.first, matches.second.back(),
                               split_node_minimum_edge_num, split_node_minimum_similarity); // Insert longest
        }
    } else {
        for (auto &matches : sourceMatches)
            normalizeMatch(sourceMatch, sourceMask, matches.first, matches.second.back(), split_node_minimum_edge_num,
                           split_node_minimum_similarity); // Insert longest
    }
    if (targetSentence.empty())
        targetMatch = sourceMatch;
    else {
        unordered_map<size_t, vector<vector<int>>> targetMatches = kb.nodeTrie.matchForAll(targetSentence, false);
        if (not targetMask.empty()) {
            for (auto &matches : targetMatches) {
                // Check if there exists any not masked position
                bool allMasked = true;
                for (size_t i = 0; i < matches.second.back().size(); i++) {
                    if (targetMask[matches.first + i] == 1) {
                        allMasked = false;
                    }
                }
                if (allMasked) {
#ifdef DEBUG
                    cout << fmt::format("Removing match [{}]@{} in target sentence",
                                        fmt::join(matches.second.back().begin(),
                                                  matches.second.back().end(), ","), matches.first) << endl;
#endif
                } else
                    normalizeMatch(targetMatch, targetMask, matches.first, matches.second.back(),
                                   split_node_minimum_edge_num, split_node_minimum_similarity); // Insert longest
            }
        } else {
            for (auto &matches : targetMatches)
                normalizeMatch(targetMatch, targetMask, matches.first, matches.second.back(),
                               split_node_minimum_edge_num, split_node_minimum_similarity); // Insert longest
        }
    }
#ifdef DEBUG
    cout << "Finish node matching for source and target sentence" << endl;
#endif

}

void KnowledgeMatcher::normalizeMatch(unordered_map<size_t, vector<int>> &match,
                                      const vector<int> &mask,
                                      size_t position,
                                      const vector<int> &node,
                                      size_t split_node_minimum_edge_num,
                                      float split_node_minimum_similarity) const {
    long nodeId = kb.nodeMap.at(node);
    bool hasOut = kb.edgeFromSource.find(nodeId) != kb.edgeFromSource.end();
    bool hasIn = kb.edgeToTarget.find(nodeId) != kb.edgeToTarget.end();
    size_t outSize = hasOut ? kb.edgeFromSource.at(nodeId).size() : 0;
    size_t inSize = hasIn ? kb.edgeToTarget.at(nodeId).size() : 0;
    if (outSize + inSize < split_node_minimum_edge_num) {
#ifdef DEBUG
        cout << fmt::format("Splitting node [{}:{}]", kb.nodes[nodeId], nodeId) << endl;
#endif
        unordered_map<size_t, vector<vector<int>>> subMatches = kb.nodeTrie.matchForAll(node, true);

        size_t currentOffset = 0;
        for (auto &subMatch : subMatches) {
            // When splitting node ABC
            // Might match [A, AB, ABC], [B, BC], [C] (each bracket is a subMatch)
            // For each subMatch X, if sim(X, ABC) > minimum similarity, insert it
            for (auto &subSubMatch : subMatch.second) {
                if (not isAllMasked(subMatch.first + position,
                                    subMatch.first + position + subSubMatch.size(),
                                    mask)) {
                    long subNodeId = kb.nodeMap.at(subSubMatch);
                    if (kb.cosineSimilarity(subNodeId, nodeId) > split_node_minimum_similarity) {
                        match.emplace(position + subMatch.first, subSubMatch);
#ifdef DEBUG
                        cout << fmt::format("Splitted node [{}:{}]", kb.nodes[subNodeId], subNodeId) << endl;
#endif
                    } else {
#ifdef DEBUG
                        cout << fmt::format("Ignore splitted node [{}:{}]", kb.nodes[subNodeId], subNodeId) << endl;
#endif
                    }
                }
            }
        }
    } else {
        match.emplace(position, node);
    }
}

KnowledgeMatcher::SelectResult
KnowledgeMatcher::selectPaths(const KnowledgeMatcher::MatchResult &inMatchResult,
                              int maxEdges,
                              float discardEdgesIfRankBelow,
                              bool filterShortAccuratePaths) const {
#ifdef DEBUG
    cout << "Begin selecting paths" << endl;
#endif

    int remainingEdges = maxEdges;
    // uncovered similarity, length
    vector<pair<float, size_t>> pathRank;

    MatchResult matchResult = inMatchResult;
    auto &visitedSubGraph = matchResult.visitedSubGraph;

    if (filterShortAccuratePaths) {
        for (auto path = visitedSubGraph.visitedPaths.begin(); path != visitedSubGraph.visitedPaths.end();) {
            if (path->edges.size() >= 1 && path->similarities[path->edges[0]] > 0.5)
                path = visitedSubGraph.visitedPaths.erase(path);
            else
                path++;
        }
    }

#pragma omp parallel for default(none) \
        shared(visitedSubGraph, pathRank) \
        firstprivate(remainingEdges)
    for (auto &path : visitedSubGraph.visitedPaths) {
        path.uncoveredEdges.insert(path.uncoveredEdges.end(), path.edges.begin(), path.edges.end());
        updatePath(path,
                   visitedSubGraph.coveredCompositeNodes,
                   visitedSubGraph.coveredNodePairs,
                   visitedSubGraph.sourceToTargetSimilarity,
                   remainingEdges);
        pathRank.emplace_back(make_pair(path.bestSimilarity, path.uncoveredEdges.size()));
    }

    while (remainingEdges > 0 && not pathRank.empty()) {
        size_t pathIndex = distance(pathRank.begin(),
                                    max_element(pathRank.begin(), pathRank.end(),
                                                [](const pair<float, size_t> &p1,
                                                   const pair<float, size_t> &p2) {
                                                    return p1.first < p2.first ||
                                                           p1.first == p2.first &&
                                                           p1.second > p2.second;
                                                }));
        if (pathRank[pathIndex].first <= discardEdgesIfRankBelow)
            break;
        auto &path = visitedSubGraph.visitedPaths[pathIndex];
        auto &addEdges = path.uncoveredEdges;
        auto &coveredEdges = visitedSubGraph.coveredSubGraph[make_pair(path.rootStartPos, path.rootEndPos)];
        coveredEdges.insert(coveredEdges.end(), addEdges.begin(), addEdges.end());
        for (size_t addEdgeIndex : addEdges) {
            long srcId = get<0>(kb.edges[addEdgeIndex]);
            long tarId = get<2>(kb.edges[addEdgeIndex]);
            bool isSrcComposite = kb.isNodeComposite[srcId];
            bool isTarComposite = kb.isNodeComposite[tarId];

            if (not isSrcComposite && not isTarComposite) {
                // Prevent inserting multiple edges connecting the same two nodes
                // If both of these two nodes are not composite
                visitedSubGraph.coveredNodePairs.insert(make_pair(srcId, tarId));
                visitedSubGraph.coveredNodePairs.insert(make_pair(tarId, srcId));
            } else {
                // Prevent inserting same composite nodes
                // (since composite nodes usually requires more space)
                if (isSrcComposite)
                    visitedSubGraph.coveredCompositeNodes.insert(srcId);
                if (isTarComposite)
                    visitedSubGraph.coveredCompositeNodes.insert(tarId);
            }
        }
        visitedSubGraph.sourceToTargetSimilarity[make_pair(path.root, path.bestSimilarityTarget)] = path.bestSimilarity;
        remainingEdges -= int(addEdges.size());

#ifdef DEBUG_DECISION
        cout << endl << "Rank result:" << endl;
        cout << "********************************************************************************" << endl;
        cout << "Root at position: " << path.rootStartPos << " Root: " << kb.nodes[path.root] << endl;
        cout << "Path rank: " << pathRank[pathIndex].first << " Length: " << pathRank[pathIndex].second << endl;
        cout << "Edges:" << endl;
        for (size_t addEdgeIndex : addEdges) {
            auto &addEdge = kb.edges[addEdgeIndex];
            if (path.similarityTargets[addEdgeIndex] != -1)
                cout << fmt::format("{}: [{}:{}] --[{}:{}]--> [{}:{}], annotation [{}], "
                                    "similarity {} to target [{}:{}]",
                                    addEdgeIndex,
                                    kb.nodes[get<0>(addEdge)], get<0>(addEdge),
                                    kb.relationships[get<1>(addEdge)], get<1>(addEdge),
                                    kb.nodes[get<2>(addEdge)], get<2>(addEdge),
                                    edgeToStringAnnotation(addEdgeIndex),
                                    path.similarities.at(addEdgeIndex),
                                    kb.nodes[path.similarityTargets[addEdgeIndex]],
                                    path.similarityTargets.at(addEdgeIndex)) << endl;
            else
                cout << fmt::format("{}: [{}:{}] --[{}:{}]--> [{}:{}], annotation [{}], "
                                    "similarity {}",
                                    addEdgeIndex,
                                    kb.nodes[get<0>(addEdge)], get<0>(addEdge),
                                    kb.relationships[get<1>(addEdge)], get<1>(addEdge),
                                    kb.nodes[get<2>(addEdge)], get<2>(addEdge),
                                    edgeToStringAnnotation(addEdgeIndex),
                                    path.similarities.at(addEdgeIndex)) << endl;
        }
        cout << "********************************************************************************" << endl;
#endif

#pragma omp parallel for default(none) \
        shared(visitedSubGraph, pathRank) \
        firstprivate(remainingEdges)
        for (size_t i = 0; i < visitedSubGraph.visitedPaths.size(); i++) {
            auto &vPath = visitedSubGraph.visitedPaths[i];
            updatePath(vPath,
                       visitedSubGraph.coveredCompositeNodes,
                       visitedSubGraph.coveredNodePairs,
                       visitedSubGraph.sourceToTargetSimilarity,
                       remainingEdges);
            pathRank[i] = make_pair(vPath.bestSimilarity, vPath.uncoveredEdges.size());
        }
    }
    SelectResult result;
    for (auto &nodeSubGraph : visitedSubGraph.coveredSubGraph) {
        for (size_t edgeIndex : nodeSubGraph.second) {
            size_t startPos = nodeSubGraph.first.first, endPos = nodeSubGraph.first.second;
            get<0>(result[endPos]) = startPos;
            get<1>(result[endPos]).emplace_back(edgeToAnnotation(edgeIndex));
            get<2>(result[endPos]).emplace_back(get<3>(kb.edges[edgeIndex]));
        }
    }
#ifdef DEBUG
    cout << "Finish selecting paths" << endl;
#endif
    return move(result);
}

void KnowledgeMatcher::trimPath(VisitedPath &path) const {
    path.bestSimilarity = 0;
    path.bestSimilarityTarget = -1;
    size_t trimPosition = 0;
    for (size_t i = 0; i < path.edges.size(); i++) {
        float similarity = path.similarities.at(path.edges[i]);
        if (similarity >= path.bestSimilarity and similarity > 0) {
            path.bestSimilarity = similarity;
            path.bestSimilarityTarget = path.similarityTargets.at(path.edges[i]);
            trimPosition = i + 1;
        }
    }
#ifdef DEBUG_DECISION
    cout << "Trimming path to length " << trimPosition << endl;
#endif
    vector<size_t> newEdges(path.edges.begin(), path.edges.begin() + trimPosition);
    path.edges.swap(newEdges);
}

void KnowledgeMatcher::updatePath(VisitedPath &path,
                                  const unordered_set<long> &coveredCompositeNodes,
                                  const unordered_set<pair<long, long>, PairHash> &coveredNodePairs,
                                  const unordered_map<pair<long, long>, float, PairHash> &sourceToTargetSimilarity,
                                  int remainingEdges) const {
    const float similarityFocusMultiplier = 3;
    // Only effective for match by token, and doesn't affect bestSimilarityTarget when match by node/embedding
    float focusMultiplier = pow(similarityFocusMultiplier, float(path.matchedFocusCount));

    float bestSimilarity = 0;
    long bestSimilarityTarget = -1;
    vector<size_t> uncoveredEdges;
    // Only search edges from start that can fit into the remaining Edges
    for (size_t uEdge : path.uncoveredEdges) {
        if (remainingEdges <= 0)
            break;
        long srcId = get<0>(kb.edges[uEdge]);
        long tarId = get<2>(kb.edges[uEdge]);
        // Only add edges to uncovered list, if:
        // 1. Both of its ends are not composite nodes and both ends are not covered by some edge
        // 2. Any of its end is a composite node, and if it is, the composite node must be not covered
        if ((not kb.isNodeComposite[srcId] &&
             not kb.isNodeComposite[tarId] &&
             coveredNodePairs.find(make_pair(srcId, tarId)) == coveredNodePairs.end() &&
             coveredNodePairs.find(make_pair(tarId, srcId)) == coveredNodePairs.end()) ||
            ((kb.isNodeComposite[srcId] || kb.isNodeComposite[tarId]) &&
             coveredCompositeNodes.find(srcId) == coveredCompositeNodes.end() &&
             coveredCompositeNodes.find(tarId) == coveredCompositeNodes.end())) {
            uncoveredEdges.emplace_back(uEdge);
            if (path.similarities.at(uEdge) * focusMultiplier > bestSimilarity) {
                auto pair = make_pair(path.root, path.similarityTargets.at(uEdge));
                // Consider a path from some root node R to some target node T
                // If its similarity is larger than another path from same root
                // R to target node Tin the covered set
                // Then make this path the best path from R -> T

                // In the mean time, since a path can be similar to multiple nodes
                // (or -1 indicating that do not compare to a single node but a group of nodes)
                // select the most similar target node T* amoung all target nodes {T}
                bool isBetterThanCurrent = (
                        sourceToTargetSimilarity.find(pair) == sourceToTargetSimilarity.end() ||
                        sourceToTargetSimilarity.at(pair) < path.similarities.at(uEdge) * focusMultiplier);
                if (isBetterThanCurrent or pair.second == -1) {
                    bestSimilarity = path.similarities.at(uEdge) * focusMultiplier;
                    bestSimilarityTarget = path.similarityTargets.at(uEdge);
                }
            }
            remainingEdges--;
        }
    }
    path.bestSimilarity = bestSimilarity;
    path.bestSimilarityTarget = bestSimilarityTarget;
    path.uncoveredEdges.swap(uncoveredEdges);
}

void KnowledgeMatcher::keepTopK(vector<float> &weights, int k) {
    if (k < 0)
        return;
    size_t size = weights.size();
    k = min(k, int(size));
    if (k == int(size))
        return;
    auto result = xt::argpartition(xt::adapt(weights, vector<size_t>{weights.size()}), size_t(size - k));
    for (size_t i = 0; i < size - k; i++) {
        weights[result[i]] = 0;
    }
}

void KnowledgeMatcher::joinVisitedSubGraph(VisitedSubGraph &vsgOut, const VisitedSubGraph &vsgIn) {
    vsgOut.visitedPaths.insert(vsgOut.visitedPaths.end(), vsgIn.visitedPaths.begin(), vsgIn.visitedPaths.end());
    // Other properties of VisitedSubGraph are meant to be used by selectPaths and thus there is no need to join

    // Sort according round number to ensure deterministic ordering
    sort(vsgOut.visitedPaths.begin(), vsgOut.visitedPaths.end(),
         [](const VisitedPath &p1, const VisitedPath &p2) { return p1.round < p2.round; });
}
