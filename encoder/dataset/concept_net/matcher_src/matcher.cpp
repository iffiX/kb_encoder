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

inline float
cosineSimilarityDataset(const HighFive::DataSet &embedding, long node1, long node2, size_t dim, bool simplifyWithInt8) {
    if (simplifyWithInt8) {
        xt::xtensor<int8_t, 1>::shape_type sh({dim});
        xt::xtensor<int8_t, 1> tmp1(sh), tmp2(sh);
        embedding.select({(size_t) node1, 0}, {1, dim}).read(tmp1.data());
        auto srcEmbed = xt::cast<int16_t>(tmp1);
        embedding.select({(size_t) node2, 0}, {1, dim}).read(tmp2.data());
        auto tarEmbed = xt::cast<int16_t>(tmp2);
        // cosine similarity
        float dot = xt::sum<int16_t>(srcEmbed * tarEmbed)[0];
        float norm1 = xt::sqrt(xt::cast<float>(xt::sum<int16_t>(srcEmbed * srcEmbed)))[0];
        float norm2 = xt::sqrt(xt::cast<float>(xt::sum<int16_t>(tarEmbed * tarEmbed)))[0];
        return dot / (norm1 * norm2);
    } else {
        xt::xtensor<float, 1>::shape_type sh({dim});
        xt::xtensor<float, 1> srcEmbed(sh), tarEmbed(sh);
        embedding.select({(size_t) node1, 0}, {1, dim}).read(srcEmbed.data());
        embedding.select({(size_t) node2, 0}, {1, dim}).read(tarEmbed.data());
        // cosine similarity
        float dot = xt::sum<float>(srcEmbed * tarEmbed)[0];
        float norm1 = xt::sqrt(xt::sum<float>(srcEmbed * srcEmbed))[0];
        float norm2 = xt::sqrt(xt::sum<float>(tarEmbed * tarEmbed))[0];
        return dot / (norm1 * norm2);
    }
}

inline float
cosineSimilarityMem(const shared_ptr<void> &embedding, long node1, long node2, size_t dim, bool simplifyWithInt8) {
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

vector<int> Trie::matchForStart(const vector<int> &sentence, size_t start) const {
    vector<int> result, tmp;
    auto *currentNode = &root;
    for (size_t i = start; i < sentence.size(); i++) {
        auto child = currentNode->children.find(sentence[i]);
        if (child != currentNode->children.end()) {
            currentNode = child->second;
            tmp.push_back(currentNode->token);
            if (currentNode->isWordEnd) {
                // move temporary memory to result
                result.insert(result.end(), tmp.begin(), tmp.end());
                tmp.clear();
            }
        } else
            break;
    }
    return move(result);
}

unordered_map<size_t, vector<int>> Trie::matchForAll(const vector<int> &sentence) const {
    unordered_map<size_t, vector<int>> result;
    for (size_t i = 0; i < sentence.size();) {
        vector<int> match = move(matchForStart(sentence, i));
        if (not match.empty()) {
            size_t match_size = match.size();
            result[i] = move(match);
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
    disabledEdges.clear();
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
            disabledEdges.insert(edgeIndex);
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
            disabledEdges.insert(edgeIndex);
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
    if (landmarkPath.empty()) {
        // create a digestion of the graph, later to be used as the file name
        size_t nodeHash = nodes.size() + edges.size();
        auto hasher = hash<string>();
        for (auto &n : nodes) {
            nodeHash ^= hasher(n) + 0x9e3779b9 + (nodeHash << 6) + (nodeHash >> 2);
        }
        fileName = fmt::format("/tmp/{}-s{}-l{}.cache", nodeHash, seedNum, landmarkNum);
        cout << fmt::format("Path not specified for landmark cache, use [{}] instead", fileName) << endl;
    } else
        fileName = landmarkPath;
    ifstream landmarkFile(fileName);
    landmarkDistances.clear();
    if (not landmarkFile.fail()) {
        cout << "Loading landmark distances from file [" << fileName << "]" << endl;
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
    } else {
        // randomly choose seed nodes and perform bfs to find best landmarkDistances
        cout << "Computing landmark distances with seedNum = "
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
            cout << fmt::format("Select landmark [{}:{}]", nodes[long(topK[i])], long(topK[i])) << endl;
#endif
            distances[i] = move(bfsDistances(long(topK[i])));
        }
        for (size_t i = 0; i < nodes.size(); i++) {
            landmarkDistances.emplace_back(vector<int>(landmarkNum, 0));
            for (size_t j = 0; j < distances.size(); j++) {
                landmarkDistances[i][j] = distances[j][i];
            }
//            cout << fmt::format("[{}]", fmt::join(landmarkDistances[i].begin(), landmarkDistances[i].end(), ","))
//                 << endl;
        }
        cout << "Saving landmark distance to file [" << fileName << "]" << endl;
        auto file = cista::file(fileName.c_str(), "w");
        cista::raw::vector<cista::raw::vector<int>> ld;
        // save as transposed form to improve saving speed
        for (auto &ldd : distances)
            ld.push_back(vector2ArchiveVector(ldd));
        cista::serialize(file, ld);
    }
}

int KnowledgeBase::distance(long node1, long node2, bool fast) const {
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
    }
    else {
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

void KnowledgeBase::save(const string &archivePath) const {
    KnowledgeArchive archive;

    cout << "Begin saving" << endl;
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

    cout << "Saved node num: " << nodes.size() << endl;
    cout << "Saved edge num: " << edges.size() << endl;
    cout << "Saved relation num: " << relationships.size() << endl;
    cout << "Saved raw relation num: " << rawRelationships.size() << endl;
}

void KnowledgeBase::load(const string &archivePath, bool loadEmbeddingToMem) {
    edgeToTarget.clear();
    edgeFromSource.clear();
    edges.clear();
    nodes.clear();
    relationships.clear();
    tokenizedNodes.clear();
    tokenizedEdgeAnnotations.clear();
    rawRelationships.clear();
    disabledEdges.clear();
    nodeEmbeddingFile = nullptr;

    cout << "Begin loading" << endl;
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
    nodeEmbeddingFileName = archive->nodeEmbeddingFileName;
    refresh(loadEmbeddingToMem);
    cout << "Loaded node num: " << nodes.size() << endl;
    cout << "Loaded edge num: " << edges.size() << endl;
    cout << "Loaded relation num: " << relationships.size() << endl;
    cout << "Loaded raw relation num: " << rawRelationships.size() << endl;
}

void KnowledgeBase::refresh(bool loadEmbeddingToMem) {
    loadEmbedding(loadEmbeddingToMem);
    loadAdjacency();
}

template <typename T1, typename T2>
bool KnowledgeBase::PriorityCmp::operator()(const pair<T1, T2> &pair1, const pair<T1, T2> &pair2) {
    return pair1.first > pair2.first;
}

void KnowledgeBase::loadEmbedding(bool loadEmbeddingToMem) {
    nodeEmbeddingFile.reset();
    if (not nodeEmbeddingFileName.empty()) {
        ifstream tmpFile(nodeEmbeddingFileName);
        cout << "KB using embedding file [" << nodeEmbeddingFileName << "]" << endl;
        if (tmpFile.fail())
            cout << "Failed to load embedding file [" << nodeEmbeddingFileName << "], skipped" << endl;
        else {
            tmpFile.close();
            nodeEmbeddingFile = make_shared<HighFive::File>(nodeEmbeddingFileName, HighFive::File::ReadOnly);
            if (loadEmbeddingToMem) {
                cout << "Loading embedding to memory" << endl;
                auto dataset = nodeEmbeddingFile->getDataSet("embeddings");
                auto shape = dataset.getDimensions();
                auto dtype = dataset.getDataType();
                if (shape.size() != 2)
                    throw invalid_argument(
                            fmt::format("Knowledge base embedding should be 2-dimensional, but got shape [{}]",
                                        fmt::join(shape.begin(), shape.end(), ",")));
                size_t dim = shape[1];

                bool simplifyWithInt8 = dtype.getClass() == HighFive::DataTypeClass::Integer;
                if (simplifyWithInt8) {
                    nodeEmbeddingMem = shared_ptr<int8_t[]>(new int8_t[shape[0] * shape[1]]);
                    dataset.read(static_pointer_cast<int8_t[]>(nodeEmbeddingMem).get());
                } else {
                    nodeEmbeddingMem = shared_ptr<float[]>(new float[shape[0] * shape[1]]);
                    dataset.read(static_pointer_cast<float[]>(nodeEmbeddingMem).get());
                }
            }
        }
    } else {
        cout << "KB does not have an embedding file, skipped" << endl;
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

template<typename T>
cista::raw::vector<T> KnowledgeBase::vector2ArchiveVector(const vector<T> &vec) {
    return move(cista::raw::vector<T>(vec.begin(), vec.end()));
}

template<typename T>
vector<T> KnowledgeBase::archiveVector2Vector(const cista::raw::vector<T> &vec) {
    return move(vector<T>(vec.begin(), vec.end()));
}

KnowledgeMatcher::KnowledgeMatcher(const KnowledgeBase &knowledgeBase) {
    cout << "Initializing matcher from knowledge base" << endl;
    kb = knowledgeBase;
    for (long index = 0; index < knowledgeBase.tokenizedNodes.size(); index++) {
        nodeTrie.insert(knowledgeBase.tokenizedNodes[index]);
        nodeMap[knowledgeBase.tokenizedNodes[index]] = index;
    }
    cout << "Matcher initialized" << endl;
}

KnowledgeMatcher::KnowledgeMatcher(const string &archivePath) {
    cout << "Initializing matcher from knowledge base archive" << endl;
    load(archivePath);
    cout << "Matcher initialized" << endl;
}

MatchResult
KnowledgeMatcher::matchByNode(const vector<int> &sourceSentence,
                              const vector<int> &targetSentence,
                              const vector<int> &sourceMask,
                              const vector<int> &targetMask,
                              int maxTimes, int maxDepth, int maxEdges, int seed,
                              float discardEdgesIfSimilarityBelow,
                              float discardEdgesIfRankBelow) const {
    unordered_map<size_t, vector<int>> sourceMatch, targetMatch;
    matchForSourceAndTarget(sourceSentence,
                            targetSentence,
                            sourceMask,
                            targetMask,
                            sourceMatch,
                            targetMatch);

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
        if (posRef.find(nodeMap.at(sm.second)) == posRef.end() || posRef[nodeMap.at(sm.second)].first > sm.first)
            posRef[nodeMap.at(sm.second)] = make_pair(sm.first, sm.first + sm.second.size());
    }

    for (auto &tm : targetMatch)
        targetNodes[nodeMap.at(tm.second)] += 1;

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
    shared(nodes, sourceMatch, targetNodes, posRef, cout, discardEdgesIfSimilarityBelow) \
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
#ifdef DEBUG
        cout << fmt::format("Round {}", i) << endl;
        cout << "Compare target:" << endl;
        for (auto &tNode : targetNodes)
            cout << fmt::format("[{}:{}] {}", kb.nodes[tNode.first], tNode.first, tNode.second) << endl;
        cout << "================================================================================" << endl;
#endif
        for (int d = 0; d < maxDepth; d++) {
            vector<float> sim;
            bool hasOut = kb.edgeFromSource.find(currentNode) != kb.edgeFromSource.end();
            bool hasIn = kb.edgeToTarget.find(currentNode) != kb.edgeToTarget.end();
            size_t outSize = hasOut ? kb.edgeFromSource.at(currentNode).size() : 0;
            size_t inSize = hasIn ? kb.edgeToTarget.at(currentNode).size() : 0;
            sim.resize(outSize + inSize, 0);

#ifdef DEBUG
            cout << fmt::format("Current node: [{}:{}]", kb.nodes[currentNode], currentNode) << endl;
#endif
            for (size_t j = 0; j < outSize + inSize; j++) {
                size_t edgeIndex = j < outSize ?
                                   kb.edgeFromSource.at(currentNode)[j] :
                                   kb.edgeToTarget.at(currentNode)[j - outSize];
                auto &edge = kb.edges[edgeIndex];
                // disable disabled edges, reflexive edges & path to visted nodes
                long otherNodeId = j < outSize ? get<2>(edge) : get<0>(edge);
                if ((kb.disabledEdges.find(edgeIndex) != kb.disabledEdges.end()) ||
                    (path.visitedNodes.find(otherNodeId) != path.visitedNodes.end()))
                    sim[j] = 0;
                else {
                    float simTmp = -1, simToEachTarget;
                    for (auto &tNode : targetNodes) {
                        auto simPair = make_pair(tNode.first, otherNodeId);
                        if (similarityCache.find(simPair) == similarityCache.end()) {
                            simToEachTarget = 1.0f / (kb.distance(tNode.first, otherNodeId) + 1e-5);
                            similarityCache[simPair] = simToEachTarget;
                            simTmp = max(simTmp, simToEachTarget);
                        } else {
                            simTmp = max(simTmp, similarityCache.at(simPair));
                        }
                    }
                    if (simTmp < discardEdgesIfSimilarityBelow)
                        simTmp = 0;
                    sim[j] = simTmp;
                }
#ifdef DEBUG
                cout << fmt::format("{}: [{}:{}] --[{}:{}]--> [{}:{}], annotation [{}], similarity {}",
                                    edgeIndex,
                                    kb.nodes[get<0>(edge)], get<0>(edge),
                                    kb.relationships[get<1>(edge)], get<1>(edge),
                                    kb.nodes[get<2>(edge)], get<2>(edge),
                                    edgeToStringAnnotation(edgeIndex),
                                    sim[j]) << endl;
#endif
            }

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
            vector<int> compareSource;
            compareSource = edgeToAnnotation(selectedEdgeIndex);

#ifdef DEBUG
            cout << endl;
            cout << "Choose edge " << selectedEdgeIndex << endl;
            cout << "Annotation: " << edgeToStringAnnotation(selectedEdgeIndex) << endl;
            cout << "--------------------------------------------------------------------------------" << endl;
#endif
        }
#ifdef DEBUG
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
                                       float discardEdgesIfSimilarityBelow,
                                       float discardEdgesIfRankBelow) const {
    if (kb.nodeEmbeddingFileName.empty() or kb.nodeEmbeddingFile == nullptr)
        throw invalid_argument("Knowledge base doesn't have an embedding file.");

    auto tmpDataset = kb.nodeEmbeddingFile->getDataSet("embeddings");
    auto shape = tmpDataset.getDimensions();
    auto dtype = tmpDataset.getDataType();
    size_t dim = shape[1];
    bool simplifyWithInt8 = dtype.getClass() == HighFive::DataTypeClass::Integer;

    HighFive::DataSetAccessProps props;
    props.add(HighFive::Caching(8192, 1024 * 1024 * 1024));
    auto embedding = kb.nodeEmbeddingFile->getDataSet("embeddings", props);
    auto embeddingMem = kb.nodeEmbeddingMem;

    unordered_map<size_t, vector<int>> sourceMatch, targetMatch;
    matchForSourceAndTarget(sourceSentence,
                            targetSentence,
                            sourceMask,
                            targetMask,
                            sourceMatch,
                            targetMatch);

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
        if (posRef.find(nodeMap.at(sm.second)) == posRef.end() || posRef[nodeMap.at(sm.second)].first > sm.first)
            posRef[nodeMap.at(sm.second)] = make_pair(sm.first, sm.first + sm.second.size());
    }

    for (auto &tm : targetMatch)
        targetNodes[nodeMap.at(tm.second)] += 1;

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
    shared(nodes, sourceMatch, targetNodes, posRef, cout, discardEdgesIfSimilarityBelow, \
           simplifyWithInt8, embedding, embeddingMem) \
    firstprivate(seed, maxTimes, maxDepth, dim, similarityCache)

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
#ifdef DEBUG
        cout << fmt::format("Round {}", i) << endl;
        cout << "Compare target:" << endl;
        for (auto &tNode : targetNodes)
            cout << fmt::format("[{}:{}] {}", kb.nodes[tNode.first], tNode.first, tNode.second) << endl;
        cout << "================================================================================" << endl;
#endif
        for (int d = 0; d < maxDepth; d++) {
            vector<float> sim;
            bool hasOut = kb.edgeFromSource.find(currentNode) != kb.edgeFromSource.end();
            bool hasIn = kb.edgeToTarget.find(currentNode) != kb.edgeToTarget.end();
            size_t outSize = hasOut ? kb.edgeFromSource.at(currentNode).size() : 0;
            size_t inSize = hasIn ? kb.edgeToTarget.at(currentNode).size() : 0;
            sim.resize(outSize + inSize, 0);

#ifdef DEBUG
            cout << fmt::format("Current node: [{}:{}]", kb.nodes[currentNode], currentNode) << endl;
#endif
            for (size_t j = 0; j < outSize + inSize; j++) {
                size_t edgeIndex = j < outSize ?
                                   kb.edgeFromSource.at(currentNode)[j] :
                                   kb.edgeToTarget.at(currentNode)[j - outSize];
                auto &edge = kb.edges[edgeIndex];
                // disable disabled edges, reflexive edges & path to visted nodes
                long otherNodeId = j < outSize ? get<2>(edge) : get<0>(edge);

                if ((kb.disabledEdges.find(edgeIndex) != kb.disabledEdges.end()) ||
                    (path.visitedNodes.find(otherNodeId) != path.visitedNodes.end()))
                    sim[j] = 0;
                else {
                    float simTmp = -1, simToEachTarget;
                    for (auto &tNode : targetNodes) {
                        auto simPair = make_pair(tNode.first, otherNodeId);
                        if (similarityCache.find(simPair) == similarityCache.end()) {
                            if (embeddingMem != nullptr)
                                simToEachTarget = cosineSimilarityMem(embeddingMem, tNode.first, otherNodeId, dim,
                                                                      simplifyWithInt8);
                            else
                                simToEachTarget = cosineSimilarityDataset(embedding, tNode.first, otherNodeId, dim,
                                                                          simplifyWithInt8);
                            similarityCache[simPair] = simToEachTarget;
                            simTmp = max(simTmp, simToEachTarget);
                        } else {
                            simTmp = max(simTmp, similarityCache.at(simPair));
                        }
                    }
                    if (simTmp < discardEdgesIfSimilarityBelow)
                        simTmp = 0;
                    sim[j] = simTmp;
                }
#ifdef DEBUG
                cout << fmt::format("{}: [{}:{}] --[{}:{}]--> [{}:{}], annotation [{}], similarity {}",
                                    edgeIndex,
                                    kb.nodes[get<0>(edge)], get<0>(edge),
                                    kb.relationships[get<1>(edge)], get<1>(edge),
                                    kb.nodes[get<2>(edge)], get<2>(edge),
                                    edgeToStringAnnotation(edgeIndex),
                                    sim[j]) << endl;
#endif
            }

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
            vector<int> compareSource;
            compareSource = edgeToAnnotation(selectedEdgeIndex);

#ifdef DEBUG
            cout << endl;
            cout << "Choose edge " << selectedEdgeIndex << endl;
            cout << "Annotation: " << edgeToStringAnnotation(selectedEdgeIndex) << endl;
            cout << "--------------------------------------------------------------------------------" << endl;
#endif
        }
#ifdef DEBUG
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
                               float discardEdgesIfSimilarityBelow,
                               float discardEdgesIfRankBelow,
                               const vector<vector<int>> &rankFocus,
                               const vector<vector<int>> &rankExclude) const {

    unordered_map<size_t, vector<int>> sourceMatch, _targetMatch;

    // When matching by token, we don't find nodes for the target Sentence
    matchForSourceAndTarget(sourceSentence,
                            sourceSentence,
                            sourceMask,
                            sourceMask,
                            sourceMatch,
                            _targetMatch);

    // first: matched node id, match start position in sentence, match end position in sentence
    // also only store the first posision reference if there are multiple occurrences to prevent duplicate knowledge.
    unordered_map<long, pair<size_t, size_t>> posRef;

    // random walk, transition possibility determined by similarity metric.
    VisitedSubGraph visitedSubGraph;

    // node id
    vector<long> nodes;

    for (auto &nm : sourceMatch)
        if (posRef.find(nodeMap.at(nm.second)) == posRef.end() || posRef[nodeMap.at(nm.second)].first > nm.first)
            posRef[nodeMap.at(nm.second)] = make_pair(nm.first, nm.first + nm.second.size());

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
    shared(sourceMask, targetMask, rankFocus, rankExclude, nodes, sourceMatch, \
           sourceSentence, targetSentence, posRef, cout, \
           discardEdgesIfSimilarityBelow) \
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
#ifdef DEBUG
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

#ifdef DEBUG
            cout << fmt::format("Current node: [{}:{}]", kb.nodes[currentNode], currentNode) << endl;
#endif
            for (size_t j = 0; j < outSize + inSize; j++) {
                size_t edgeIndex = j < outSize ?
                                   kb.edgeFromSource.at(currentNode)[j] :
                                   kb.edgeToTarget.at(currentNode)[j - outSize];
                auto &edge = kb.edges[edgeIndex];
                // disable disabled edges, reflexive edges & path to visted nodes

                if ((kb.disabledEdges.find(edgeIndex) != kb.disabledEdges.end()) ||
                    (j < outSize && path.visitedNodes.find(get<2>(edge)) != path.visitedNodes.end()) ||
                    (j >= outSize && path.visitedNodes.find(get<0>(edge)) != path.visitedNodes.end()))
                    sim[j] = 0;
                else {
                    float simTmp = similarity(edgeToAnnotation(edgeIndex), compareTarget);
                    if (simTmp < discardEdgesIfSimilarityBelow)
                        simTmp = 0;
                    sim[j] = simTmp;
                }
#ifdef DEBUG
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
            vector<int> compareSource;
            compareSource = edgeToAnnotation(selectedEdgeIndex);
            path.matchedFocusCount += findPattern(compareSource, rankFocus);
            path.matchedFocusCount -= findPattern(compareSource, rankExclude);

#ifdef DEBUG
            cout << endl;
            cout << "Choose edge " << selectedEdgeIndex << endl;
            cout << "Annotation: " << edgeToStringAnnotation(selectedEdgeIndex) << endl;
            cout << "--------------------------------------------------------------------------------" << endl;
#endif
        }
#ifdef DEBUG
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
    nodeTrie.clear();
    nodeMap.clear();
    for (long index = 0; index < kb.tokenizedNodes.size(); index++) {
        nodeTrie.insert(kb.tokenizedNodes[index]);
        nodeMap[kb.tokenizedNodes[index]] = index;
    }
}

string KnowledgeMatcher::getNodeTrie() const {
    return move(nodeTrie.serialize());
}

vector<pair<vector<int>, long>> KnowledgeMatcher::getNodeMap() const {
    vector<pair<vector<int>, long>> result;
    result.insert(result.end(), nodeMap.begin(), nodeMap.end());
    return move(result);
}

size_t KnowledgeMatcher::VectorHash::operator()(const vector<int> &vec) const {
    // A simple hash function
    // https://stackoverflow.com/questions/20511347/a-good-hash-function-for-a-vector
    size_t seed = vec.size();
    for (auto &i : vec) {
        seed ^= i + 0x9e3779b9 + (seed << 6) + (seed >> 2);
    }
    return seed;
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
    sourceMatch = nodeTrie.matchForAll(sourceSentence);
    if (not sourceMask.empty()) {
        if (sourceMask.size() != sourceSentence.size())
            throw invalid_argument(fmt::format(
                    "Mask is provided for source sentence but size does not match, sentence: {}, mask: {}",
                    sourceSentence.size(), sourceMask.size()));
        for (auto iter = sourceMatch.begin(); iter != sourceMatch.end();) {
            // Check if the posision is masked
            if (sourceMask[iter->first] == 0) {
#ifdef DEBUG
                cout << fmt::format("Removing match [{}]@{} in source sentence",
                                    fmt::join(iter->second.begin(), iter->second.end(), ","), iter->first) << endl;
#endif
                iter = sourceMatch.erase(iter);
            } else
                iter++;
        }
    }
    if (targetSentence.empty())
        targetMatch = sourceMatch;
    else {
        targetMatch = move(nodeTrie.matchForAll(targetSentence));
        if (not targetMask.empty()) {
            if (targetMask.size() != targetSentence.size())
                throw invalid_argument(fmt::format(
                        "Mask is provided for target sentence but size does not match, sentence: {}, mask: {}",
                        targetSentence.size(), targetMask.size()));
            for (auto iter = targetMatch.begin(); iter != targetMatch.end();) {
                // Check if the posision is masked
                if (targetMask[iter->first] == 0) {
#ifdef DEBUG
                    cout << fmt::format("Removing match [{}]@{} in target sentence",
                                        fmt::join(iter->second.begin(), iter->second.end(), ","), iter->first) << endl;
#endif
                    iter = targetMatch.erase(iter);
                } else
                    iter++;
            }
        }
    }
}

MatchResult
KnowledgeMatcher::selectPaths(VisitedSubGraph &visitedSubGraph,
                              const unordered_map<long, pair<size_t, size_t>> &posRef,
                              int maxEdges, float discardEdgesIfRankBelow) const {
    const float similarityFocusMultiplier = 3;

    // set cover problem, use greedy algorithm to achieve 1-1/e approximation
    int remainingEdges = maxEdges;
    // uncovered similarity
    vector<float> pathRank;

    for (auto &path : visitedSubGraph.visitedPaths) {
        // trim end to make sure at least one path can fit into the result
        path.uncoveredEdges.insert(path.uncoveredEdges.end(), path.edges.begin(),
                                   path.edges.begin() + min(path.edges.size(), size_t(remainingEdges)));
        path.uncoveredSimilarity = 0;
        for (size_t edgeIndex : path.uncoveredEdges)
            path.uncoveredSimilarity += path.similarities[edgeIndex];
        path.uncoveredSimilarity *= pow(similarityFocusMultiplier, float(path.matchedFocusCount));
        pathRank.emplace_back(path.uncoveredSimilarity);
    }

    while (remainingEdges > 0) {
        size_t pathIndex = distance(pathRank.begin(), max_element(pathRank.begin(), pathRank.end()));
        if (pathRank[pathIndex] <= discardEdgesIfRankBelow)
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
        remainingEdges -= int(addEdges.size());

#ifdef DEBUG
        cout << endl << "Rank result:" << endl;
        cout << "********************************************************************************" << endl;
        cout << "Root at position: " << posRef.at(path.root).first << " Root: " << kb.nodes[path.root] << endl;
        cout << "Path rank: " << pathRank[pathIndex] << endl;
        cout << "Edges:" << endl;
        for (size_t addEdgeIndex : addEdges) {
            auto &addEdge = kb.edges[addEdgeIndex];
            cout << fmt::format("{}: [{}:{}] --[{}:{}]--> [{}:{}], annotation [{}], similarity {}",
                                addEdgeIndex,
                                kb.nodes[get<0>(addEdge)], get<0>(addEdge),
                                kb.relationships[get<1>(addEdge)], get<1>(addEdge),
                                kb.nodes[get<2>(addEdge)], get<2>(addEdge),
                                edgeToStringAnnotation(addEdgeIndex),
                                path.similarities[addEdgeIndex]) << endl;
        }
        cout << "********************************************************************************" << endl;
#endif

#pragma omp parallel for default(none) \
        shared(visitedSubGraph, pathRank) \
        firstprivate(similarityFocusMultiplier, remainingEdges)
        for (size_t i = 0; i < visitedSubGraph.visitedPaths.size(); i++) {
            auto &vPath = visitedSubGraph.visitedPaths[i];
            updatePath(vPath, visitedSubGraph.coveredNodePairs, remainingEdges);
            vPath.uncoveredSimilarity *= pow(similarityFocusMultiplier, float(vPath.matchedFocusCount));
            pathRank[i] = visitedSubGraph.visitedPaths[i].uncoveredSimilarity;
        }
    }
    MatchResult result;
    for (auto &nodeSubGraph : visitedSubGraph.coveredSubGraph) {
        for (size_t edgeIndex : nodeSubGraph.second) {
            size_t startPos = posRef.at(nodeSubGraph.first).first, endPos = posRef.at(nodeSubGraph.first).second;
            get<0>(result[endPos]) = startPos;
            get<1>(result[endPos]).emplace_back(edgeToAnnotation(edgeIndex));
        }
    }
    return move(result);
}

void KnowledgeMatcher::updatePath(VisitedPath &path,
                                  const unordered_set<pair<long, long>, PairHash> &coveredNodePairs,
                                  int remainingEdges) const {
    float uncoveredSimilarity = 0;
    vector<size_t> uncoveredEdges;
    for (size_t uEdge : path.uncoveredEdges) {
        if (remainingEdges <= 0)
            break;
        long srcId = get<0>(kb.edges[uEdge]);
        long tarId = get<2>(kb.edges[uEdge]);
        if ((coveredNodePairs.find(make_pair(srcId, tarId)) == coveredNodePairs.end()) &&
            (coveredNodePairs.find(make_pair(tarId, srcId)) == coveredNodePairs.end())) {
            uncoveredEdges.emplace_back(uEdge);
            uncoveredSimilarity += path.similarities[uEdge];
            remainingEdges--;
        }
    }
    path.uncoveredSimilarity = uncoveredSimilarity;
    path.uncoveredEdges.swap(uncoveredEdges);
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
    //return float(overlapCount) / float(source.size());
    //return lcs(source, target);
}