#include "matcher.h"
#include "tqdm.h"
#include <tuple>
#include <regex>
#include <random>
#include <vector>
#include <fstream>
#include <algorithm>
#include <string_view>

using namespace std;

int lcs(const vector<int>& x, const vector<int>& y)
{
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
    for (auto& delim : delims) {
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

string replaceAllChar(const string &input, char pattern, char replace) {
    string output(input.length(), ' ');
    for(size_t i = 0; i < input.length(); i++)
        output[i] = input[i] == pattern ? replace : input[i];
    return move(output);
}

TrieNode::TrieNode(int token, bool isWordEnd) : token(token), isWordEnd(isWordEnd) {}

TrieNode::~TrieNode() {
    clear();
}

TrieNode* TrieNode::addChild(int childToken, bool isChildWordEnd) {
    children[childToken] = new TrieNode(childToken, isChildWordEnd);
    return children[childToken];
}

bool TrieNode::removeChild(int childToken) {
    if (children.find(childToken) != children.end()) {
        delete children[childToken];
        children.erase(childToken);
        return true;
    }
    else
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
    while(readPos < content.length()) {
        if (content[readPos] == ')')
            break;
        auto childStart = scan(content, {"[",}, readPos);
        if (get<0>(childStart) == string::npos)
            throw invalid_argument("Cannot find a delimiter '[' for starting a child.");
        auto* child = new TrieNode();
        readPos = child->initializeFromString(content, get<0>(childStart) + get<1>(nodeStart).length());
        children[child->token] = child;
        auto childEnd = scan(content, {"],", "]"}, readPos);
        if (get<0>(childEnd) == string::npos)
            throw invalid_argument("Cannot find an enclosing delimiter ']' or '],' for child.");
        readPos = get<0>(childEnd) + get<1>(childEnd).length();
    }
    auto end = scan(content, {"),",")"}, readPos);
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

Trie::Trie(const std::vector<std::vector<int>> &words) {
    for(auto &word : words)
        insert(word);
}

void Trie::insert(const vector<int> &word) {
    auto* currentNode = &root;
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
    TrieNode* parentNode = nullptr;
    auto* currentNode = &root;
    if (word.empty())
        return false;
    for (int token : word) {
        auto child = currentNode->children.find(token);
        if (child != currentNode->children.end()) {
            parentNode = currentNode;
            currentNode = child->second;
        }
        else
            return false;
    }
    if (currentNode->isWordEnd) {
        if (currentNode->children.empty())
            parentNode->removeChild(currentNode->token);
        else
            currentNode->isWordEnd = false;
        return true;
    }
    else
        return false;
}

void Trie::clear() {
    root.clear();
}

vector<int> Trie::matchForStart(const vector<int> &sentence, size_t start) const {
    vector<int> result, tmp;
    auto* currentNode = &root;
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
        }
        else
            break;
    }
    return move(result);
}

unordered_map<size_t, vector<int>> Trie::matchForAll(const vector<int> &sentence) const {
    unordered_map<size_t, vector<int>> result;
    for(size_t i = 0; i < sentence.size();) {
        vector<int> match = move(matchForStart(sentence, i));
        if (not match.empty()) {
            size_t match_size = match.size();
            result[i] = move(match);
            i += match_size;
        }
        else
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

ConceptNetReader::ConceptNetReader(const string &path) {
    ifstream file(path);
    string line;
    unordered_map<string, long> relationshipsMap;
    unordered_set<string> rawRelationshipsSet;
    unordered_map<string, long> nodeMap;

    tqdm bar;
    bar.set_theme_basic();
    bar.disable_colors();
    size_t estimateNum = 100000;
    size_t processed = 0;
    while(getline(file, line)) {
        istringstream is(line);
        string uri, relation, srcEntity, tarEntity, info;
        getline(is, uri, '\t');
        getline(is, relation, '\t');
        getline(is, srcEntity, '\t');
        getline(is, tarEntity, '\t');
        getline(is, info, '\t');
        size_t prefixLen = sizeof("/c/en/") - 1;
        if (srcEntity.length() >= prefixLen && srcEntity.substr(0, prefixLen) == "/c/en/" &&
            tarEntity.length() >= prefixLen && tarEntity.substr(0, prefixLen) == "/c/en/") {

            string src = replaceAllChar(srcEntity.substr(prefixLen, srcEntity.find("/", prefixLen) - prefixLen), '_', ' ');
            string tar = replaceAllChar(tarEntity.substr(prefixLen, tarEntity.find("/", prefixLen) - prefixLen), '_', ' ');
            long srcId = -1, tarId = -1;
            if (nodeMap.find(src) != nodeMap.end())
                srcId = nodeMap[src];
            else {
                long id = long(nodeMap.size());
                srcId = nodeMap[src] = id;
                nodes.push_back(src);
            }
            if (nodeMap.find(tar) != nodeMap.end())
                tarId = nodeMap[tar];
            else {
                long id = long(nodeMap.size());
                tarId = nodeMap[tar] = id;
                nodes.push_back(tar);
            }

            string rawRel = relation.substr(sizeof("/r/") - 1);
            if (rawRelationshipsSet.find(rawRel) == rawRelationshipsSet.end()) {
                rawRelationshipsSet.insert(rawRel);
                rawRelationships.push_back(rawRel);
            }
            // only keep the last part of the relation
            // eg: for dbpedia/occupation it would be occupation
            auto pos = rawRel.rfind('/');
            string rel;
            long relId;
            if (pos == string::npos)
                rel = rawRel;
            else
                rel = rawRel.substr(pos + 1);
            if (relationshipsMap.find(rel) == relationshipsMap.end()) {
                long id = long(relationshipsMap.size());
                relId = relationshipsMap[rel] = id;
                relationships.push_back(rel);
            }
            else
                relId = relationshipsMap[rel];

            float weight;
            string annotation;
            smatch result;
            if (regex_search(info, result, regex(R"("weight":\s+([+-]?([0-9]*[.])?[0-9]+))")))
                weight = stof(result[1]);
            else
                weight = 1;
            smatch result2;
            if (regex_search(info, result2, regex("\"surfaceText\":\\s+\"(.+)\"[,}]"))) {
                annotation = replaceAllChar(result2[1], '[', ' ');
                annotation = replaceAllChar(annotation, ']', ' ');
            }
            edgeToTarget[tarId].push_back(edges.size());
            edgeFromSource[srcId].push_back(edges.size());
            edges.emplace_back(make_tuple(srcId, relId, tarId, weight, annotation));
            processed++;
            if (processed > estimateNum) {
                estimateNum *= 2;
                cout << endl << "Update estimation of total number." << endl;
            }
            bar.progress(processed, estimateNum);
        }
    }
    bar.finish();
    cout << "Processed lines: " << processed << endl;
    cout << "Node num: " << nodes.size() << endl;
    cout << "Edge num: " << edges.size() << endl;
    cout << "Relation num: " << relationships.size() << endl;
    cout << "Raw relation num: " << rawRelationships.size() << endl;
}

vector<Edge> ConceptNetReader::getEdges(long source, long target) const {
    vector<Edge> result;
    if (source != -1 && target == -1) {
        if (edgeFromSource.find(source) != edgeFromSource.end())
            for (size_t edgeIndex: edgeFromSource.at(source))
                result.push_back(edges[edgeIndex]);
    }
    else if (source == -1 && target != -1) {
        if (edgeToTarget.find(target) != edgeToTarget.end())
            for (size_t edgeIndex: edgeToTarget.at(target))
                result.push_back(edges[edgeIndex]);
    }
    else
        if(edgeFromSource.find(source) != edgeFromSource.end())
            for (size_t edgeIndex : edgeFromSource.at(source))
                if (get<2>(edges[edgeIndex]) == target)
                    result.push_back(edges[edgeIndex]);

    return move(result);
}

const vector<string> &ConceptNetReader::getNodes() const {
    return nodes;
}

vector<string> ConceptNetReader::getNodes(const vector<long> &nodeIndexes) const {
    vector<string> result;
    for (long nodeIdx : nodeIndexes) {
        result.push_back(nodes[nodeIdx]);
    }
    return result;
}

ConceptNetMatcher::ConceptNetMatcher(const ConceptNetReader &tokenizedReader)
: reader(&tokenizedReader), isLoad(false) {
    for(long index = 0; index < tokenizedReader.tokenizedNodes.size(); index++) {
        nodeTrie.insert(tokenizedReader.tokenizedNodes[index]);
        nodeMap[tokenizedReader.tokenizedNodes[index]] = index;
    }
}

ConceptNetMatcher::ConceptNetMatcher(const std::string &archivePath) : reader(nullptr), isLoad(false){
    load(archivePath);
}

ConceptNetMatcher::~ConceptNetMatcher() {
    if (isLoad)
        delete reader;
}

unordered_map<size_t, tuple<size_t, vector<vector<int>>>>
        ConceptNetMatcher::match(const vector<int> &sentence,
                                 int maxTimes, int maxDepth, int maxEdges, int seed,
                                 const vector<vector<int>> &similarityExclude,
                                 const vector<vector<int>> &rankFocus,
                                 const vector<vector<int>> &rankExclude) const {
    const float similarityFocusMultiplier = 3;
    unordered_map<size_t, vector<int>> nodeMatch = nodeTrie.matchForAll(sentence);

    // first: matched node id, match start position in sentence, match end position in sentence
    // also only store the first posision reference if there are multiple occurrences to prevent duplicate knowledge.
    unordered_map<long, pair<size_t, size_t>> posRef;

    // random walk, transition possibility determined by similarity metric.
    VisitedSubGraph visitedSubGraph;

    // node id
    vector<long> nodes;

    for(auto &nm : nodeMatch)
        if (posRef.find(nodeMap.at(nm.second)) == posRef.end() || posRef[nodeMap.at(nm.second)].first > nm.first)
            posRef[nodeMap.at(nm.second)] = make_pair(nm.first, nm.first + nm.second.size());

    for (auto &item : posRef)
        nodes.push_back(item.first);

    if (seed < 0) {
        random_device rd;
        seed = rd();
    }

    if (maxTimes < 2 * nodes.size())
        cout << "Parameter maxTimes " << maxTimes << " is smaller than 2 * node size " << nodes.size()
             << ", which may result in insufficient exploration, consider increase maxTimes." << endl;

    // https://stackoverflow.com/questions/43168661/openmp-and-reduction-on-stdvector
    #pragma omp declare reduction(vsg_join : VisitedSubGraph : joinVisitedSubGraph(omp_out, omp_in)) \
                        initializer(omp_priv = omp_orig)

    #pragma omp parallel for reduction(vsg_join : visitedSubGraph)
    for (int i=0; i<maxTimes; i++) {
        mt19937 gen (seed ^ i);

        // uniform sampling for efficient parallelization
        size_t nodeLocalIndex;
        long rootNode, currentNode;

        uniform_int_distribution<size_t> nodeDist(0, nodes.size() - 1);
        nodeLocalIndex = nodeDist(gen);

        rootNode = currentNode = nodes[nodeLocalIndex];

        // remove matched node itself from the compare sentence
        // since some edges are like "forgets is derived form forget"
        auto filterPatterns = similarityExclude;
        filterPatterns.push_back(nodeMatch.at(posRef.at(currentNode).first));
        auto compareTarget = filter(sentence, filterPatterns);

        VisitedPath path;
        path.root = rootNode;
        path.matchedFocusCount = 0;
        path.visitedNodes.insert(currentNode);

        for (int d = 0; d < maxDepth; d++) {
            vector<float> sim;
            bool hasOut = reader->edgeFromSource.find(currentNode) != reader->edgeFromSource.end();
            bool hasIn = reader->edgeToTarget.find(currentNode) != reader->edgeToTarget.end();
            size_t outSize = hasOut ? reader->edgeFromSource.at(currentNode).size() : 0;
            size_t inSize = hasIn ? reader->edgeToTarget.at(currentNode).size() : 0;
            sim.resize(outSize + inSize, 0);

            for (size_t j = 0; j < outSize + inSize; j++) {
                size_t edgeIndex = j < outSize ?
                                   reader->edgeFromSource.at(currentNode)[j] :
                                   reader->edgeToTarget.at(currentNode)[j - outSize];
                auto &edge = reader->edges[edgeIndex];
                // disable reflexive edge & path to visted nodes
                if ((j < outSize && path.visitedNodes.find(get<2>(edge)) != path.visitedNodes.end()) ||
                        (j >= outSize && path.visitedNodes.find(get<0>(edge)) != path.visitedNodes.end()))
                    sim[j] = 0;
                else {
                    vector<int> compareSource;
                    if (reader->tokenizedEdgeAnnotations[edgeIndex].empty())
                        sim[j] = similarity(edgeToAnnotation(edge), compareTarget);
                    else
                        sim[j] = similarity(reader->tokenizedEdgeAnnotations[edgeIndex], compareTarget);
                }
            }

            // break on meeting nodes with all edges being reflexive
            if (all_of(sim.begin(), sim.end(), [](float f) { return f == 0; }))
                break;

            discrete_distribution<size_t> dist(sim.begin(), sim.end());
            size_t e = dist(gen);
            bool isEdgeOut = e < outSize;
            size_t selectedEdgeIndex = isEdgeOut ?
                                       reader->edgeFromSource.at(currentNode)[e] :
                                       reader->edgeToTarget.at(currentNode)[e - outSize];

            // move to next node
            currentNode = isEdgeOut ?
                          get<2>(reader->edges[selectedEdgeIndex]) :
                          get<0>(reader->edges[selectedEdgeIndex]);
            path.visitedNodes.insert(currentNode);

            path.edges.push_back(selectedEdgeIndex);
            path.similarities[selectedEdgeIndex] = sim[e];
            vector<int> compareSource;
            if (reader->tokenizedEdgeAnnotations[selectedEdgeIndex].empty())
                compareSource = edgeToAnnotation(reader->edges[selectedEdgeIndex]);
            else
                compareSource =reader->tokenizedEdgeAnnotations[selectedEdgeIndex];
            path.matchedFocusCount += findPattern(compareSource, rankFocus);
            path.matchedFocusCount -= findPattern(compareSource, rankExclude);
        }

        visitedSubGraph.visitedPaths.push_back(path);
    }


    // set cover problem, use greedy algorithm to achieve 1-1/e approximation
    int remainingEdges = maxEdges;
    // uncovered similarity
    vector<float> pathRank;

    for(auto &path : visitedSubGraph.visitedPaths) {
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
        if (pathRank[pathIndex] == 0)
            break;
        auto &path = visitedSubGraph.visitedPaths[pathIndex];
        auto &addEdges = path.uncoveredEdges;
        auto &coveredEdges = visitedSubGraph.coveredSubGraph[path.root];
        coveredEdges.insert(coveredEdges.end(), addEdges.begin(), addEdges.end());
        for (size_t addEdge : addEdges) {
            long srcId = get<0>(reader->edges[addEdge]);
            long tarId = get<2>(reader->edges[addEdge]);
            visitedSubGraph.coveredNodePairs.insert(make_pair(srcId, tarId));
        }
        remainingEdges -= int(addEdges.size());

//        cout << "Root@ " << posRef[path.root].first << " Root: " << reader->tokenizedNodes[path.root] << endl;
//        for (size_t addEdge : addEdges) {
//            cout << "Edge " << reader->tokenizedNodes[get<0>(reader->edges[addEdge])] << "--"
//            << reader->tokenizedRelationships[get<1>(reader->edges[addEdge])] << "-->"
//            << reader->tokenizedNodes[get<2>(reader->edges[addEdge])] << endl;
//        }

        #pragma omp parallel for
        for(size_t i = 0; i < visitedSubGraph.visitedPaths.size(); i++) {
            auto &vPath = visitedSubGraph.visitedPaths[i];
            updatePath(vPath, visitedSubGraph.coveredNodePairs, remainingEdges);
            if (vPath.matchedFocusCount > 0)
                vPath.uncoveredSimilarity *= pow(similarityFocusMultiplier, float(path.matchedFocusCount));
            pathRank[i] = visitedSubGraph.visitedPaths[i].uncoveredSimilarity;
        }
    }
    unordered_map<size_t, tuple<size_t, vector<vector<int>>>> result;
    for(auto &nodeSubGraph : visitedSubGraph.coveredSubGraph) {
        for(size_t edgeIndex : nodeSubGraph.second) {
            size_t startPos = posRef[nodeSubGraph.first].first, endPos = posRef[nodeSubGraph.first].second;
            get<0>(result[endPos]) = startPos;
            if (reader->tokenizedEdgeAnnotations[edgeIndex].empty())
                get<1>(result[endPos]).emplace_back(edgeToAnnotation(reader->edges[edgeIndex]));
            else
                get<1>(result[endPos]).emplace_back(reader->tokenizedEdgeAnnotations[edgeIndex]);
        }
    }
    return move(result);
}

void ConceptNetMatcher::save(const std::string &archivePath) const {
    ConceptNetArchive archive;
    for (auto &ett : reader->edgeToTarget)
        archive.edgeToTarget.emplace(ett.first, vector2ArchiveVector(ett.second));
    for (auto &efs : reader->edgeFromSource)
        archive.edgeFromSource.emplace(efs.first, vector2ArchiveVector(efs.second));
    for (auto &e: reader->edges)
        archive.edges.emplace_back(SerializableEdge{
            get<0>(e), get<1>(e), get<2>(e), get<3>(e), get<4>(e)});
    archive.nodes.set(reader->nodes.begin(), reader->nodes.end());
    archive.relationships.set(reader->relationships.begin(), reader->relationships.end());
    for (auto &tn : reader->tokenizedNodes)
        archive.tokenizedNodes.emplace_back(vector2ArchiveVector(tn));
    for (auto &tr : reader->tokenizedRelationships)
        archive.tokenizedRelationships.emplace_back(vector2ArchiveVector(tr));
    for (auto &tea : reader->tokenizedEdgeAnnotations)
        archive.tokenizedEdgeAnnotations.emplace_back(vector2ArchiveVector(tea));
    archive.rawRelationships.set(reader->rawRelationships.begin(), reader->rawRelationships.end());
    auto file = cista::file(archivePath.c_str(), "w");
    cista::serialize(file, archive);
}

void ConceptNetMatcher::load(const std::string &archivePath) {
    if (isLoad)
        delete reader;
    auto *newReader = new ConceptNetReader();
    auto file = cista::file(archivePath.c_str(), "r");
    auto content = file.content();
    auto *archive = cista::deserialize<ConceptNetArchive>(content);
    for (auto &ett : archive->edgeToTarget)
        newReader->edgeToTarget[ett.first] = archiveVector2Vector(ett.second);
    for (auto &efs : archive->edgeFromSource)
        newReader->edgeFromSource[efs.first] = archiveVector2Vector(efs.second);
    for (auto &e: archive->edges)
        newReader->edges.emplace_back(make_tuple(e.source, e.relation, e.target, e.weight, e.annotation));
    newReader->nodes.insert(newReader->nodes.end(), archive->nodes.begin(), archive->nodes.end());
    newReader->relationships.insert(
            newReader->relationships.end(), archive->relationships.begin(), archive->relationships.end());
    for (auto &tn : archive->tokenizedNodes)
        newReader->tokenizedNodes.emplace_back(archiveVector2Vector(tn));
    for (auto &tr : archive->tokenizedRelationships)
        newReader->tokenizedRelationships.emplace_back(archiveVector2Vector(tr));
    for (auto &tea : archive->tokenizedEdgeAnnotations)
        newReader->tokenizedEdgeAnnotations.emplace_back(archiveVector2Vector(tea));
    newReader->rawRelationships.insert(
            newReader->rawRelationships.end(), archive->rawRelationships.begin(), archive->rawRelationships.end());
    reader = newReader;
    cout << "Reader node num: " << reader->nodes.size() << endl;
    cout << "Reader edge num: " << reader->edges.size() << endl;
    cout << "Reader relation num: " << reader->relationships.size() << endl;
    cout << "Reader raw relation num: " << reader->rawRelationships.size() << endl;

    nodeTrie.clear();
    nodeMap.clear();
    for(long index = 0; index <reader->tokenizedNodes.size(); index++) {
        nodeTrie.insert(reader->tokenizedNodes[index]);
        nodeMap[reader->tokenizedNodes[index]] = index;
    }
    isLoad = true;
}

std::string ConceptNetMatcher::getNodeTrie() const {
    return nodeTrie.serialize();
}

std::vector<std::pair<std::vector<int>, long>> ConceptNetMatcher::getNodeMap() const {
    vector<pair<vector<int>, long>> result;
    result.insert(result.end(), nodeMap.begin(), nodeMap.end());
    return move(result);
}

size_t ConceptNetMatcher::VectorHash::operator()(const vector<int> &vec) const {
    // A simple hash function
    // https://stackoverflow.com/questions/20511347/a-good-hash-function-for-a-vector
    size_t seed = vec.size();
    for(auto& i : vec) {
        seed ^= i + 0x9e3779b9 + (seed << 6) + (seed >> 2);
    }
    return seed;
}

template <typename T1, typename T2>
std::size_t ConceptNetMatcher::PairHash::operator()(const std::pair<T1, T2> &pair) const {
    return (hash<T1>()(pair.first) << 32) | hash<T2>()(pair.second);
}

vector<int> ConceptNetMatcher::edgeToAnnotation(const Edge &edge) const {
    vector<int> edgeAnno(reader->tokenizedNodes[get<0>(edge)]);
    auto &rel = reader->tokenizedRelationships[get<1>(edge)];
    edgeAnno.insert(edgeAnno.end(), rel.begin(), rel.end());
    auto &tar = reader->tokenizedNodes[get<2>(edge)];
    edgeAnno.insert(edgeAnno.end(), tar.begin(), tar.end());
    return move(edgeAnno);
}

void ConceptNetMatcher::updatePath(VisitedPath &path,
                                   const std::unordered_set<std::pair<long, long>, PairHash> &coveredNodePairs,
                                   int remainingEdges) const {
    float uncoveredSimilarity = 0;
    vector <size_t> uncoveredEdges;
    for (size_t uEdge : path.uncoveredEdges) {
        if (remainingEdges <= 0)
            break;
        long srcId = get<0>(reader->edges[uEdge]);
        long tarId = get<2>(reader->edges[uEdge]);
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

void ConceptNetMatcher::joinVisitedSubGraph(VisitedSubGraph &vsgOut, const VisitedSubGraph &vsgIn) {
    vsgOut.visitedPaths.insert(vsgOut.visitedPaths.end(), vsgIn.visitedPaths.begin(), vsgIn.visitedPaths.end());
}

size_t ConceptNetMatcher::findPattern(const vector<int> &sentence, const vector<vector<int>> &patterns) {
    size_t matchedPatterns = 0;
    for(auto &pattern : patterns) {
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

std::vector<int> ConceptNetMatcher::filter(const vector<int> &sentence, const vector<vector<int>> &patterns) {
    std::vector<int> lastResult = sentence, result;
    std::vector<int> tmp;
    for(auto &pattern : patterns) {
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

float ConceptNetMatcher::similarity(const vector<int> &source, const vector<int> &target) {
    std::unordered_set<int> cmp(target.begin(), target.end());
    size_t overlapCount = 0;
    for (int word : source) {
        if (cmp.find(word) != cmp.end())
            overlapCount++;
    }
    return overlapCount;
    //return float(overlapCount) / float(source.size());
    //return lcs(source, target);
}

template <typename T>
cista::raw::vector<T> ConceptNetMatcher::vector2ArchiveVector(const vector<T> &vec) {
    return move(cista::raw::vector<T>(vec.begin(), vec.end()));
}

template <typename T>
vector<T> ConceptNetMatcher::archiveVector2Vector(const cista::raw::vector<T> &vec) {
    return move(vector<T>(vec.begin(), vec.end()));
}