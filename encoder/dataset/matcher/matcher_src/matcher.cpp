#define FMT_HEADER_ONLY

#include "matcher.h"
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

int lcs(const vector<int> &x, const vector<int> &y) {
    unsigned int L[x.size() + 1][y.size() + 1];

    /* Following steps build L[m+1][n+1] in bottom up fashion. Note
      that L[i][j] contains length of LCS of X[0..i-1] and Y[0..j-1] */
    for (size_t i = 0; i <= x.size(); i++) {
        for (size_t j = 0; j <= y.size(); j++) {
            if (i == 0 || j == 0)
                L[i][j] = 0;

            else if (x[i - 1] == y[j - 1])
                L[i][j] = L[i - 1][j - 1] + 1;

            else
                L[i][j] = max(L[i - 1][j], L[i][j - 1]);
        }
    }

    /* L[m][n] contains length of LCS for X[0..n-1] and Y[0..m-1] */
    return L[x.size()][y.size()];
}

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

vector<long> KnowledgeBase::findNodes(const vector<string> &nod) const {
    vector<long> ids;
    for (auto &nName : nod) {
        for (long nodeId = 0; nodeId < nodes.size(); nodeId++) {
            if (nodes[nodeId] == nName)
                ids.push_back(nodeId);
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
                                     const vector<int> &tokenizedCompositeNode,
                                     const string &relationship) {
    long relationId = -1;
    for (long relId = 0; relId < long(relationships.size()); relId++) {
        if (relationships[relId] == relationship) {
            relationId = relId;
            break;
        }
    }
    if (relationId == -1)
        throw std::invalid_argument(fmt::format("Relationship [{}] not found", relationship));

    long newNodeId = long(nodes.size());
    nodes.emplace_back(compositeNode);
    tokenizedNodes.emplace_back(tokenizedCompositeNode);
    isNodeComposite.push_back(true);

    // Find all sub-nodes occuring in the composite node
    // Then add an edge from all sub nodes to the composite node with relationship=relationship

    // The similaity of the composite node to other nodes is computed by:
    // the maximum sub node similarity to other nodes
    auto result = nodeTrie.matchForAll(tokenizedCompositeNode, false);
    for(auto subNode : result) {
        long edgeId = long(edges.size());
        long sourceNodeId = nodeMap.at(subNode.second.back());
        edges.emplace_back(sourceNodeId, relationId, newNodeId, 1, "");
        edgeToTarget.at(newNodeId).push_back(relationId);
        edgeFromSource.at(sourceNodeId).push_back(relationId);
        adjacency[sourceNodeId].insert(newNodeId);
        adjacency[newNodeId].insert(sourceNodeId);
        tokenizedEdgeAnnotations.emplace_back(vector<int>{});
    }
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
        float min = -1;
        for (long subNode : compositeNodes.at(node1)) {
            float dist = distance(subNode, node2);
            min = dist > min ? min : dist;
        }
        return min;
    }
    if (isNodeComposite[node2]) {
        float min = -1;
        for (long subNode : compositeNodes.at(node1)) {
            float dist = distance(node1, subNode);
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
//            int dist = bfsDistance(node1, node2, 3);
//            if (dist != DISTANCE_MAX)
//                return dist;
//            else
//                return ALTDistance(node1, node2);
        }
    }
}

float KnowledgeBase::cosineSimilarity(long node1, long node2) const {
    if (isNodeComposite[node1]) {
        float max = -1;
        for (long subNode : compositeNodes.at(node1)) {
            float sim = cosineSimilarity(subNode, node2);
            max = sim > max ? sim : max;
        }
        return max;
    }
    if (isNodeComposite[node2]) {
        float max = -1;
        for (long subNode : compositeNodes.at(node2)) {
            float sim = cosineSimilarity(node1, subNode);
            max = sim > max ? sim : max;
        }
        return max;
    }
    if (nodeEmbeddingMem.get() != nullptr)
        return cosineSimilarityFromMem(nodeEmbeddingMem, node1, node2, nodeEmbeddingDim, nodeEmbeddingSimplifyWithInt8);
    else
        return cosineSimilarityFromDataset(nodeEmbeddingDataset, node1, node2, nodeEmbeddingDim,
                                           nodeEmbeddingSimplifyWithInt8);
}

void KnowledgeBase::save(const string &archivePath) const {
    KnowledgeArchive archive;
    if (not compositeNodes.empty())
        throw std::runtime_error("It's not safe to save after adding composite nodes");

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

inline bool KnowledgeBase::isNeighbor(long node1, long node2) const {
    return adjacency.find(node1) != adjacency.end() && adjacency.at(node1).find(node2) != adjacency.at(node1).end();
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

MatchResult
KnowledgeMatcher::matchByNode(const vector<int> &sourceSentence,
                              const vector<int> &targetSentence,
                              const vector<int> &sourceMask,
                              const vector<int> &targetMask,
                              int maxTimes, int maxDepth, int maxEdges, int seed,
                              int edgeBeamWidth, bool trim,
                              float discardEdgesIfSimilarityBelow,
                              float discardEdgesIfRankBelow) const {

#ifdef DEBUG
    cout << "================================================================================" << endl;
    cout << "Method: match by node" << endl;
    cout << fmt::format("sourceSentence: [{}]",
                        fmt::join(sourceSentence.begin(), sourceSentence.end(), ",")) << endl;
    cout << fmt::format("targetSentence: [{}]",
                        fmt::join(targetSentence.begin(), targetSentence.end(), ",")) << endl;
    cout << fmt::format("sourceMask: [{}]",
                        fmt::join(sourceMask.begin(), sourceMask.end(), ",")) << endl;
    cout << fmt::format("targetMask: [{}]",
                        fmt::join(targetMask.begin(), targetMask.end(), ",")) << endl;
    cout << "maxTimes: " << maxTimes << endl;
    cout << "maxDepth: " << maxDepth << endl;
    cout << "maxEdges: " << maxEdges << endl;
    cout << "seed: " << seed << endl;
    cout << "discardEdgesIfSimilarityBelow: " << discardEdgesIfSimilarityBelow << endl;
    cout << "discardEdgesIfRankBelow: " << discardEdgesIfRankBelow << endl;
    cout << "================================================================================" << endl;
#endif

    unordered_map<size_t, vector<int>> sourceMatch, targetMatch;
    matchForSourceAndTarget(sourceSentence,
                            targetSentence,
                            sourceMask,
                            targetMask,
                            sourceMatch,
                            targetMatch);

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

    // first: matched node id, match start position in sentence, match end position in sentence
    // also only store the first posision reference if there are multiple occurrences to prevent duplicate knowledge.
    unordered_map<long, pair<size_t, size_t>> posRef;

    // node ids in targetSentence (or sourceSentence if targetSentence is empty), their occurence times
    unordered_map<long, size_t> targetNodes;

    // random walk, transition possibility determined by similarity metric.
    VisitedSubGraph visitedSubGraph;

    // node id
    vector<long> nodes;

    unordered_map<pair<long, long>, float, UndirectedPairHash> similarityCache;

    for (auto &sm : sourceMatch) {
        if (posRef.find(kb.nodeMap.at(sm.second)) == posRef.end() || posRef[kb.nodeMap.at(sm.second)].first > sm.first)
            posRef[kb.nodeMap.at(sm.second)] = make_pair(sm.first, sm.first + sm.second.size());
    }

    for (auto &tm : targetMatch)
        targetNodes[kb.nodeMap.at(tm.second)] += 1;

    for (auto &item : posRef)
        nodes.push_back(item.first);

    if (seed < 0) {
        random_device rd;
        seed = rd();
    }

    if (maxTimes < 2 * int(nodes.size()))
        cout << "Parameter maxTimes " << maxTimes << " is smaller than 2 * node size " << nodes.size()
             << ", which may result in insufficient exploration, consider increase maxTimes." << endl;

    // https://stackoverflow.com/questions/43168661/openmp-and-reduction-on-stdvector
#pragma omp declare reduction(vsg_join : VisitedSubGraph : joinVisitedSubGraph(omp_out, omp_in)) \
                        initializer(omp_priv = omp_orig)

#pragma omp parallel for reduction(vsg_join : visitedSubGraph) \
    default(none) \
    shared(nodes, sourceMatch, targetNodes, posRef, cout, edgeBeamWidth, discardEdgesIfSimilarityBelow) \
    firstprivate(seed, maxTimes, maxDepth, similarityCache)

    for (int i = 0; i < maxTimes; i++) {
        mt19937 gen(seed ^ i);

        // uniform sampling for efficient parallelization
        size_t nodeLocalIndex;
        long rootNode, currentNode;

        uniform_int_distribution<size_t> nodeDist(0, nodes.size() - 1);
        nodeLocalIndex = nodeDist(gen);

        rootNode = currentNode = nodes[nodeLocalIndex];

        VisitedPath path;
        path.root = rootNode;
        path.matchedFocusCount = 0;
        path.visitedNodes.insert(currentNode);
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
                    (path.visitedNodes.find(otherNodeId) != path.visitedNodes.end()))
                    sim[j] = 0;
                else {
                    float simTmp = -1, simToEachTarget;
                    long simTmpTarget = -1;
                    for (auto &tNode : targetNodes) {
                        if (rootNode != tNode.first) {
                            auto simPair = make_pair(tNode.first, otherNodeId);
                            if (similarityCache.find(simPair) == similarityCache.end()) {
                                simToEachTarget = 1.0f / (kb.distance(tNode.first, otherNodeId) + 1e-1);
                                similarityCache[simPair] = simToEachTarget;
                            } else {
                                simToEachTarget = similarityCache.at(simPair);
                            }
                        }
                        else
                            simToEachTarget = 0;
                        if (simToEachTarget > simTmp) {
                            simTmpTarget = tNode.first;
                            simTmp = simToEachTarget;
                        }
                    }
                    if (simTmp < discardEdgesIfSimilarityBelow)
                        simTmp = 0;
                    sim[j] = simTmp;
                    simTarget[j] = simTmpTarget;
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
            keepTopK(sim, edgeBeamWidth);
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
        visitedSubGraph.visitedPaths.push_back(path);
    }

    return move(selectPaths(visitedSubGraph, posRef, maxEdges, discardEdgesIfRankBelow));
}


MatchResult
KnowledgeMatcher::matchByNodeEmbedding(const vector<int> &sourceSentence,
                                       const vector<int> &targetSentence,
                                       const vector<int> &sourceMask,
                                       const vector<int> &targetMask,
                                       int maxTimes, int maxDepth, int maxEdges, int seed,
                                       int edgeBeamWidth, bool trim,
                                       float discardEdgesIfSimilarityBelow,
                                       float discardEdgesIfRankBelow) const {
    if (kb.nodeEmbeddingFileName.empty() or kb.nodeEmbeddingFile == nullptr)
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
    cout << "maxTimes: " << maxTimes << endl;
    cout << "maxDepth: " << maxDepth << endl;
    cout << "maxEdges: " << maxEdges << endl;
    cout << "seed: " << seed << endl;
    cout << "discardEdgesIfSimilarityBelow: " << discardEdgesIfSimilarityBelow << endl;
    cout << "discardEdgesIfRankBelow: " << discardEdgesIfRankBelow << endl;
    cout << "================================================================================" << endl;
#endif
    unordered_map<size_t, vector<int>> sourceMatch, targetMatch;
    matchForSourceAndTarget(sourceSentence,
                            targetSentence,
                            sourceMask,
                            targetMask,
                            sourceMatch,
                            targetMatch);

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

    // first: matched node id, match start position in sentence, match end position in sentence
    // also only store the first posision reference if there are multiple occurrences to prevent duplicate knowledge.
    unordered_map<long, pair<size_t, size_t>> posRef;

    // node ids in targetSentence (or sourceSentence if targetSentence is empty), their occurence times
    unordered_map<long, size_t> targetNodes;

    // random walk, transition possibility determined by similarity metric.
    VisitedSubGraph visitedSubGraph;

    // node id
    vector<long> nodes;

    unordered_map<pair<long, long>, float, UndirectedPairHash> similarityCache;

    for (auto &sm : sourceMatch) {
        if (posRef.find(kb.nodeMap.at(sm.second)) == posRef.end() || posRef[kb.nodeMap.at(sm.second)].first > sm.first)
            posRef[kb.nodeMap.at(sm.second)] = make_pair(sm.first, sm.first + sm.second.size());
    }

    for (auto &tm : targetMatch)
        targetNodes[kb.nodeMap.at(tm.second)] += 1;

    for (auto &item : posRef)
        nodes.push_back(item.first);

    if (seed < 0) {
        random_device rd;
        seed = rd();
    }

    if (maxTimes < 2 * int(nodes.size()))
        cout << "Parameter maxTimes " << maxTimes << " is smaller than 2 * node size " << nodes.size()
             << ", which may result in insufficient exploration, consider increase maxTimes." << endl;

    // https://stackoverflow.com/questions/43168661/openmp-and-reduction-on-stdvector
#pragma omp declare reduction(vsg_join : VisitedSubGraph : joinVisitedSubGraph(omp_out, omp_in)) \
                        initializer(omp_priv = omp_orig)

#pragma omp parallel for reduction(vsg_join : visitedSubGraph) \
    default(none) \
    shared(nodes, sourceMatch, targetNodes, posRef, cout, edgeBeamWidth, discardEdgesIfSimilarityBelow) \
    firstprivate(seed, maxTimes, maxDepth, similarityCache)

    for (int i = 0; i < maxTimes; i++) {
        mt19937 gen(seed ^ i);

        // uniform sampling for efficient parallelization
        size_t nodeLocalIndex;
        long rootNode, currentNode;

        uniform_int_distribution<size_t> nodeDist(0, nodes.size() - 1);
        nodeLocalIndex = nodeDist(gen);

        rootNode = currentNode = nodes[nodeLocalIndex];

        VisitedPath path;
        path.root = rootNode;
        path.matchedFocusCount = 0;
        path.visitedNodes.insert(currentNode);
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
                    (path.visitedNodes.find(otherNodeId) != path.visitedNodes.end()))
                    sim[j] = 0;
                else {
                    float simTmp = -1, simToEachTarget;
                    long simTmpTarget = -1;
                    for (auto &tNode : targetNodes) {
                        if (rootNode != tNode.first) {
                            auto simPair = make_pair(tNode.first, otherNodeId);
                            if (similarityCache.find(simPair) == similarityCache.end()) {
                                simToEachTarget = kb.cosineSimilarity(otherNodeId, tNode.first);
                                similarityCache[simPair] = simToEachTarget;
                            } else {
                                simToEachTarget = similarityCache.at(simPair);
                            }
                        }
                        else
                            simToEachTarget = 0;
                        if (simToEachTarget > simTmp) {
                            simTmpTarget = tNode.first;
                            simTmp = simToEachTarget;
                        }
                    }
                    if (simTmp < discardEdgesIfSimilarityBelow)
                        simTmp = 0;
                    sim[j] = simTmp;
                    simTarget[j] = simTmpTarget;
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

            keepTopK(sim, edgeBeamWidth);
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
        visitedSubGraph.visitedPaths.push_back(path);
    }

    return move(selectPaths(visitedSubGraph, posRef, maxEdges, discardEdgesIfRankBelow));
}


MatchResult
KnowledgeMatcher::matchByToken(const vector<int> &sourceSentence,
                               const vector<int> &targetSentence,
                               const vector<int> &sourceMask,
                               const vector<int> &targetMask,
                               int maxTimes, int maxDepth, int maxEdges, int seed,
                               int edgeBeamWidth, bool trim,
                               float discardEdgesIfSimilarityBelow,
                               float discardEdgesIfRankBelow,
                               const vector<vector<int>> &rankFocus,
                               const vector<vector<int>> &rankExclude) const {
#ifdef DEBUG
    cout << "================================================================================" << endl;
    cout << "Method: match by token" << endl;
    cout << fmt::format("sourceSentence: [{}]",
                        fmt::join(sourceSentence.begin(), sourceSentence.end(), ",")) << endl;
    cout << fmt::format("targetSentence: [{}]",
                        fmt::join(targetSentence.begin(), targetSentence.end(), ",")) << endl;
    cout << fmt::format("sourceMask: [{}]",
                        fmt::join(sourceMask.begin(), sourceMask.end(), ",")) << endl;
    cout << fmt::format("targetMask: [{}]",
                        fmt::join(targetMask.begin(), targetMask.end(), ",")) << endl;
    cout << "maxTimes: " << maxTimes << endl;
    cout << "maxDepth: " << maxDepth << endl;
    cout << "maxEdges: " << maxEdges << endl;
    cout << "seed: " << seed << endl;
    cout << "discardEdgesIfSimilarityBelow: " << discardEdgesIfSimilarityBelow << endl;
    cout << "discardEdgesIfRankBelow: " << discardEdgesIfRankBelow << endl;
    cout << "rankFocus:" << endl;
    for (auto &item : rankFocus)
        cout << fmt::format("[{}]",
                            fmt::join(item.begin(), item.end(), ",")) << endl;
    cout << "rankExclude:" << endl;
    for (auto &item : rankExclude)
        cout << fmt::format("[{}]",
                            fmt::join(item.begin(), item.end(), ",")) << endl;
    cout << "================================================================================" << endl;
#endif

    // target match is not used here
    unordered_map<size_t, vector<int>> sourceMatch, targetMatch;

    matchForSourceAndTarget(sourceSentence,
                            sourceSentence,
                            sourceMask,
                            sourceMask,
                            sourceMatch,
                            targetMatch);

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


    // first: matched node id, match start position in sentence, match end position in sentence
    // also only store the first posision reference if there are multiple occurrences to prevent duplicate knowledge.
    unordered_map<long, pair<size_t, size_t>> posRef;

    // node ids in targetSentence (or sourceSentence if targetSentence is empty), their occurence times
    unordered_map<long, size_t> targetNodes;

    // random walk, transition possibility determined by similarity metric.
    VisitedSubGraph visitedSubGraph;

    // node id
    vector<long> nodes;

    for (auto &nm : sourceMatch) {
        if (posRef.find(kb.nodeMap.at(nm.second)) == posRef.end() || posRef[kb.nodeMap.at(nm.second)].first > nm.first)
            posRef[kb.nodeMap.at(nm.second)] = make_pair(nm.first, nm.first + nm.second.size());
    }

    for (auto &tm : targetMatch)
        targetNodes[kb.nodeMap.at(tm.second)] += 1;

    for (auto &item : posRef)
        nodes.push_back(item.first);

    if (seed < 0) {
        random_device rd;
        seed = rd();
    }

    if (maxTimes < 2 * int(nodes.size()))
        cout << "Parameter maxTimes " << maxTimes << " is smaller than 2 * node size " << nodes.size()
             << ", which may result in insufficient exploration, consider increase maxTimes." << endl;

    // https://stackoverflow.com/questions/43168661/openmp-and-reduction-on-stdvector
#pragma omp declare reduction(vsg_join : VisitedSubGraph : joinVisitedSubGraph(omp_out, omp_in)) \
                        initializer(omp_priv = omp_orig)

#pragma omp parallel for reduction(vsg_join : visitedSubGraph) \
    default(none) \
    shared(sourceMask, targetMask, rankFocus, rankExclude, nodes, targetNodes, sourceMatch, \
           sourceSentence, targetSentence, posRef, cout, \
           edgeBeamWidth, discardEdgesIfSimilarityBelow) \
    firstprivate(seed, maxTimes, maxDepth)

    for (int i = 0; i < maxTimes; i++) {
        mt19937 gen(seed ^ i);

        // uniform sampling for efficient parallelization
        size_t nodeLocalIndex;
        long rootNode, currentNode;

        uniform_int_distribution<size_t> nodeDist(0, nodes.size() - 1);
        nodeLocalIndex = nodeDist(gen);

        rootNode = currentNode = nodes[nodeLocalIndex];

        // remove matched node itself from the compare sentence
        // since some edges are like "forgets is derived form forget"
        // each time create a clean copy
        auto filterPattern = vector<vector<int>>{sourceMatch.at(posRef.at(currentNode).first)};

        vector<int> compareTarget;
        if (targetSentence.empty())
            compareTarget = move(filter(mask(sourceSentence, sourceMask), filterPattern));
        else
            compareTarget = move(filter(mask(targetSentence, targetMask), filterPattern));

        VisitedPath path;
        path.root = rootNode;
        path.matchedFocusCount = 0;
        path.visitedNodes.insert(currentNode);
#ifdef DEBUG_DECISION
        cout << fmt::format("Round {}", i) << endl;
        cout << fmt::format("Compare target: [{}]", fmt::join(compareTarget, ", ")) << endl;
        cout << "================================================================================" << endl;
#endif
        for (int d = 0; d < maxDepth; d++) {
            vector<float> sim;
            bool hasOut = kb.edgeFromSource.find(currentNode) != kb.edgeFromSource.end();
            bool hasIn = kb.edgeToTarget.find(currentNode) != kb.edgeToTarget.end();
            size_t outSize = hasOut ? kb.edgeFromSource.at(currentNode).size() : 0;
            size_t inSize = hasIn ? kb.edgeToTarget.at(currentNode).size() : 0;
            sim.resize(outSize + inSize, 0);

#ifdef DEBUG_DECISION
            cout << fmt::format("Current node: [{}:{}]", kb.nodes[currentNode], currentNode) << endl;
#endif
            for (size_t j = 0; j < outSize + inSize; j++) {
                size_t edgeIndex = j < outSize ?
                                   kb.edgeFromSource.at(currentNode)[j] :
                                   kb.edgeToTarget.at(currentNode)[j - outSize];
                auto &edge = kb.edges[edgeIndex];
                // disable disabled edges, reflexive edges & path to visted nodes

                if (kb.isEdgeDisabled[edgeIndex] ||
                    (j < outSize && path.visitedNodes.find(get<2>(edge)) != path.visitedNodes.end()) ||
                    (j >= outSize && path.visitedNodes.find(get<0>(edge)) != path.visitedNodes.end()))
                    sim[j] = 0;
                else {
                    float simTmp = similarity(edgeToAnnotation(edgeIndex), compareTarget);
                    if (simTmp < discardEdgesIfSimilarityBelow)
                        simTmp = 0;
                    sim[j] = simTmp;
                }
#ifdef DEBUG_DECISION
                cout << fmt::format("{}: [{}:{}] --[{}:{}]--> [{}:{}], annotation [{}], similarity {}, cmp_src [{}]",
                                    edgeIndex,
                                    kb.nodes[get<0>(edge)], get<0>(edge),
                                    kb.relationships[get<1>(edge)], get<1>(edge),
                                    kb.nodes[get<2>(edge)], get<2>(edge),
                                    edgeToStringAnnotation(edgeIndex),
                                    sim[j],
                                    fmt::join(edgeToAnnotation(edgeIndex), ", ")) << endl;
#endif
            }
            keepTopK(sim, edgeBeamWidth);
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
            // Since match by token doesn't compute similarity by comparing to a target node,
            // put a placeholder
            path.similarityTargets[selectedEdgeIndex] = -1;
            vector<int> compareSource;
            compareSource = edgeToAnnotation(selectedEdgeIndex);
            path.matchedFocusCount += findPattern(compareSource, rankFocus);
            path.matchedFocusCount -= findPattern(compareSource, rankExclude);

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
        visitedSubGraph.visitedPaths.push_back(path);
    }
    return move(selectPaths(visitedSubGraph, posRef, maxEdges, discardEdgesIfRankBelow));
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

template<typename T1, typename T2>
size_t KnowledgeMatcher::UndirectedPairHash::operator()(const pair<T1, T2> &pair) const {
    return hash<T1>()(pair.first) ^ hash<T2>()(pair.second);
}


vector<int> KnowledgeMatcher::edgeToAnnotation(size_t edgeIndex) const {
    const Edge &edge = kb.edges[edgeIndex];
//    if (kb.tokenizedEdgeAnnotations[edgeIndex].empty()) {
//        vector<int> edgeAnno(kb.tokenizedNodes[get<0>(edge)]);
//        auto &rel = kb.tokenizedRelationships[get<1>(edge)];
//        edgeAnno.insert(edgeAnno.end(), rel.begin(), rel.end());
//        auto &tar = kb.tokenizedNodes[get<2>(edge)];
//        edgeAnno.insert(edgeAnno.end(), tar.begin(), tar.end());
//        return move(edgeAnno);
//    }
//    else {
//        return kb.tokenizedEdgeAnnotations[edgeIndex];
//    }
    vector<int> edgeAnno(kb.tokenizedNodes[get<0>(edge)]);
    auto &rel = kb.tokenizedRelationships[get<1>(edge)];
    edgeAnno.insert(edgeAnno.end(), rel.begin(), rel.end());
    auto &tar = kb.tokenizedNodes[get<2>(edge)];
    edgeAnno.insert(edgeAnno.end(), tar.begin(), tar.end());
    return move(edgeAnno);
}

string KnowledgeMatcher::edgeToStringAnnotation(size_t edgeIndex) const {
    const Edge &edge = kb.edges[edgeIndex];
//    if (kb.tokenizedEdgeAnnotations[edgeIndex].empty()) {
//        string edgeAnno(kb.nodes[get<0>(edge)]);
//        edgeAnno += " ";
//        auto &rel = kb.relationships[get<1>(edge)];
//        edgeAnno.insert(edgeAnno.end(), rel.begin(), rel.end());
//        edgeAnno += " ";
//        auto &tar = kb.nodes[get<2>(edge)];
//        edgeAnno.insert(edgeAnno.end(), tar.begin(), tar.end());
//        return move(edgeAnno);
//    }
//    else {
//        return get<4>(edge);
//    }
    string edgeAnno(kb.nodes[get<0>(edge)]);
    edgeAnno += " ";
    auto &rel = kb.relationships[get<1>(edge)];
    edgeAnno.insert(edgeAnno.end(), rel.begin(), rel.end());
    edgeAnno += " ";
    auto &tar = kb.nodes[get<2>(edge)];
    edgeAnno.insert(edgeAnno.end(), tar.begin(), tar.end());
    return move(edgeAnno);
}

void KnowledgeMatcher::matchForSourceAndTarget(const vector<int> &sourceSentence,
                                               const vector<int> &targetSentence,
                                               const vector<int> &sourceMask,
                                               const vector<int> &targetMask,
                                               unordered_map<size_t, vector<int>> &sourceMatch,
                                               unordered_map<size_t, vector<int>> &targetMatch) const {
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
            for(size_t i = 0; i < matches.second.back().size(); i++) {
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
                sourceMatch.emplace(matches.first, matches.second.back()); // Insert longest
        }
    } else {
        for (auto &matches : sourceMatches)
            sourceMatch.emplace(matches.first, matches.second.back()); // Insert longest
    }
    if (targetSentence.empty())
        targetMatch = sourceMatch;
    else {
        unordered_map<size_t, vector<vector<int>>> targetMatches = kb.nodeTrie.matchForAll(targetSentence, false);
        if (not targetMask.empty()) {
            for (auto &matches : targetMatches) {
                // Check if there exists any not masked position
                bool allMasked = true;
                for(size_t i = 0; i < matches.second.back().size(); i++) {
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
                    targetMatch.emplace(matches.first, matches.second.back()); // Insert longest
            }
        } else {
            for (auto &matches : targetMatches)
                targetMatch.emplace(matches.first, matches.second.back()); // Insert longest
        }
    }
#ifdef DEBUG
    cout << "Finish node matching for source and target sentence" << endl;
#endif

}

MatchResult
KnowledgeMatcher::selectPaths(VisitedSubGraph &visitedSubGraph,
                              const unordered_map<long, pair<size_t, size_t>> &posRef,
                              int maxEdges,
                              float discardEdgesIfRankBelow) const {
#ifdef DEBUG
    cout << "Begin selecting paths" << endl;
#endif

    int remainingEdges = maxEdges;
    // uncovered similarity, length
    vector<pair<float, size_t>> pathRank;

#pragma omp parallel for default(none) \
        shared(visitedSubGraph, pathRank) \
        firstprivate(remainingEdges)
    for (auto &path : visitedSubGraph.visitedPaths) {
        path.uncoveredEdges.insert(path.uncoveredEdges.end(), path.edges.begin(), path.edges.end());
        updatePath(path, visitedSubGraph.coveredNodePairs, visitedSubGraph.sourceToTargetSimilarity, remainingEdges);
        pathRank.emplace_back(make_pair(path.bestSimilarity, path.uncoveredEdges.size()));
    }

    while (remainingEdges > 0) {
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
        auto &coveredEdges = visitedSubGraph.coveredSubGraph[path.root];
        coveredEdges.insert(coveredEdges.end(), addEdges.begin(), addEdges.end());
        for (size_t addEdgeIndex : addEdges) {
            long srcId = get<0>(kb.edges[addEdgeIndex]);
            long tarId = get<2>(kb.edges[addEdgeIndex]);
            visitedSubGraph.coveredNodePairs.insert(make_pair(srcId, tarId));
        }
        visitedSubGraph.sourceToTargetSimilarity[make_pair(path.root, path.bestSimilarityTarget)] = path.bestSimilarity;
        remainingEdges -= int(addEdges.size());

#ifdef DEBUG_DECISION
        cout << endl << "Rank result:" << endl;
        cout << "********************************************************************************" << endl;
        cout << "Root at position: " << posRef.at(path.root).first << " Root: " << kb.nodes[path.root] << endl;
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
                       visitedSubGraph.coveredNodePairs,
                       visitedSubGraph.sourceToTargetSimilarity,
                       remainingEdges);
            pathRank[i] = make_pair(vPath.bestSimilarity, vPath.uncoveredEdges.size());
        }
    }
    MatchResult result;
    for (auto &nodeSubGraph : visitedSubGraph.coveredSubGraph) {
        for (size_t edgeIndex : nodeSubGraph.second) {
            size_t startPos = posRef.at(nodeSubGraph.first).first, endPos = posRef.at(nodeSubGraph.first).second;
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
        if (similarity >= path.bestSimilarity) {
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
        if ((coveredNodePairs.find(make_pair(srcId, tarId)) == coveredNodePairs.end()) &&
            (coveredNodePairs.find(make_pair(tarId, srcId)) == coveredNodePairs.end())) {
            uncoveredEdges.emplace_back(uEdge);
            if (path.similarities.at(uEdge) * focusMultiplier > bestSimilarity) {
                auto pair = make_pair(path.root, path.similarityTargets.at(uEdge));
                bool isBetterThanCurrent = (
                        sourceToTargetSimilarity.find(pair) == sourceToTargetSimilarity.end() ||
                        sourceToTargetSimilarity.at(pair) < path.similarities.at(uEdge) * focusMultiplier);
                if (isBetterThanCurrent) {
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
}

size_t KnowledgeMatcher::findPattern(const vector<int> &sentence, const vector<vector<int>> &patterns) {
    size_t matchedPatterns = 0;
    for (auto &pattern : patterns) {
        size_t matched = 0;
        for (int word : sentence) {
            if (word != pattern[matched]) {
                matched = 0;
            } else {
                matched++;
                if (matched == pattern.size()) {
                    matchedPatterns++;
                    break;
                }
            }
        }
    }
    return matchedPatterns;
}

vector<int> KnowledgeMatcher::filter(const vector<int> &sentence, const vector<vector<int>> &patterns) {
    vector<int> lastResult = sentence, result;
    vector<int> tmp;
    for (auto &pattern : patterns) {
        tmp.clear();
        size_t matched = 0;
        for (int word : lastResult) {
            if (word != pattern[matched]) {
                matched = 0;
                if (not tmp.empty()) {
                    result.insert(result.end(), tmp.begin(), tmp.end());
                    tmp.clear();
                }
                result.push_back(word);
            } else {
                matched++;
                tmp.push_back(word);
                if (matched == pattern.size()) {
                    matched = 0;
                    tmp.clear();
                }
            }
        }
        if (not tmp.empty())
            result.insert(result.end(), tmp.begin(), tmp.end());
        lastResult.swap(result);
        result.clear();
    }
    return move(lastResult);
}

vector<int> KnowledgeMatcher::mask(const vector<int> &sentence, const vector<int> &mask) {
    vector<int> result;
    if (mask.size() != sentence.size())
        throw invalid_argument(fmt::format(
                "Mask is provided for input sentence but size does not match, sentence: {}, mask: {}",
                sentence.size(), mask.size()));
    for (size_t i = 0; i < sentence.size(); i++) {
        if (mask[i] != 0)
            result.push_back(sentence[i]);
    }
    return result;
}

float KnowledgeMatcher::similarity(const vector<int> &source, const vector<int> &target) {
    unordered_set<int> cmp_tar(target.begin(), target.end());
    unordered_set<int> cmp_src(source.begin(), source.end());
    size_t overlapCount = 0;
    for (int word : cmp_src) {
        if (cmp_tar.find(word) != cmp_tar.end())
            overlapCount++;
    }
    return overlapCount;
    //return float(overlapCount) / (float(source.size()) + 1);
    //return lcs(source, target);
}