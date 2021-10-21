//
// Created by iffi on 2021/10/8.
//
#include <fstream>
#include <iostream>
#include "matcher.h"
using namespace std;

template <typename T>
std::ostream& operator<< (std::ostream& out, const std::vector<std::vector<T>>& vec) {
    if ( !vec.empty() ) {
        out << '[';
        for(auto &subVec : vec) {
            cout << '[';
            for (T v: subVec)
                out << v << ", ";
            out << "\b\b]";
        }
        out << "\b\b]"; // use two ANSI backspace characters '\b' to overwrite final ", "
    }
    else {
        out << "empty";
    }
    return out;
}

template <typename T1, typename T2>
std::ostream& operator<< (std::ostream& out, const std::tuple<T1, T2>& tuple) {
    out << '(' << std::get<0>(tuple) << ", "<< std::get<1>(tuple) << ')';
    return out;
}

template <typename T>
std::ostream& operator<< (std::ostream& out, const std::unordered_map<size_t, T>& map) {
    if ( !map.empty() ) {
        out << '{';
        for(auto &item : map)
            out << item.first << ": "<< item.second << ", ";
        out << "\b\b]"; // use two ANSI backspace characters '\b' to overwrite final ", "
    }
    else {
        out << "empty";
    }
    return out;
}

int main() {
    auto node = TrieNode();
    // pass
    //node.initializeFromString("1,([2$])");

    // err
    //node.initializeFromString("1,([2$],[3,([4$])]");

//    node.initializeFromString("1,([2$],[3$,([5$],[6$])])");
//    cout << node.serialize() << endl;
//
//    node.initializeFromString("1,([2$],[3,([4$],),],)");
//    cout << node.serialize() << endl;
//    node.initializeFromString("1,([2$],[3,([4$],[5,([6$],[7$])])])");
//    cout << node.serialize() << endl;
//
//    auto trie = Trie({{1, 2}, {1, 3, 4}, {1, 3}, {1, 3, 5, 6}, {1, 3, 5, 7}});
//    cout << trie.serialize() << endl;
//
//    trie.initializeFromString("-1,([1,([3,([5$,([7$],[6$])],[4$])],[2$])])");
//    cout << trie.serialize() << endl;
//    trie.insert({1,3,5,7,8});
//    cout << trie.serialize() << endl;
//    cout << trie.remove({1,3,5,7,8}) << endl;
//    cout << trie.serialize() << endl;
//    cout << trie.remove({1,3}) << endl;
//    cout << trie.serialize() << endl;
//    cout << trie.remove({1,3, 5}) << endl;
//    cout << trie.serialize() << endl;
//    trie.insert({1, 3, 5});
//    cout << trie.serialize() << endl;

//    auto trie = Trie({{2}, {3, 4}, {4, 1}, {1, 2}, {1, 3}, {1, 3, 4}, {1, 3, 5, 6}, {1, 3, 5, 7}});
//    cout << trie.serialize() << endl;
//    cout << trie.matchForStart({1, 2, 18, 32}) << endl;
//    cout << trie.matchForStart({3, 1, 2, 18, 32}) << endl;
//    cout << trie.matchForStart({123, 1, 2, 18, 32}) << endl;
//    cout << trie.matchForAll({3, 1, 2, 18, 32}) << endl;
    // 3, 4, 1 for checking overlap match
    // 1, 3, 5 for checking partial match (1, 3, and 1, 3, 5, 6)
//    cout << trie.matchForAll({9, 3, 4, 1, 3, 5, 1, 2, 8, 16, 1, 3, 4, 9}) << endl;

//    ConceptNetReader reader("/data/conceptnet-assertions-5.7.0.csv");
//    cout << "Node num: " << reader.nodes.size() << endl;
//    cout << "Relation num: " << reader.relationships.size() << endl;
//    cout << "Raw relation num: " << reader.rawRelationships.size() << endl;
//    cout << "Edge num: " << reader.edges.size() << endl;
//    cout << "Relations: " << reader.relationships << endl;

//    ConceptNetReader reader("/data/conceptnet-assertions-5.7.0_slice.csv");
//    ConceptNetMatcher matcher(reader);
//    matcher.save("/data/conceptnet-archive.data");
//      ConceptNetMatcher matcher2("/data/conceptnet-archive.data");
//      cout << matcher2.getNodeTrie() << endl;

//    Trie trie;
//    ifstream stream("/data/conceptnet-trie.txt");
//    string line;
//    getline(stream, line);
//    trie.initializeFromString(line);
//    cout << trie.serialize() << endl;

//    std::vector<int> sentence{1996, 6404, 2187, 24302, 2018, 5407, 3243, 2214, 1010, 2002, 14876, 8630, 2009, 3139, 1999, 18282, 1999, 1996, 2067, 1997, 2010, 2054, 1029, 10135, 1010, 18097, 1010, 7852, 8758, 1010, 16716, 1010, 2873};
//    ConceptNetMatcher matcher("/data/conceptnet-archive.data");
//    cout << matcher.match(sentence, 1000, 2, 5, 1920301, {{18097}, {2187, 24302}}, {{1996}, {1037}, {2019}, {2002}, {2000}, {2010}}) << endl;


    std::vector<int> sentence{71, 3, 60, 4571, 3745, 1365, 19, 4979, 21, 192, 2212, 1111, 6, 68, 34, 92, 4657, 38, 3, 9, 1034, 3613, 44, 3, 9, 125, 58, 4739, 6, 31866, 6, 221, 2274, 297, 1078, 6, 1982, 40, 6, 5534, 25453};
    ConceptNetMatcher matcher("/data/conceptnet-archive.data");
    cout << matcher.match(sentence, 1000, 2, 5, 1920301, {{18097}, {2187, 24302}}, {{1996}, {1037}, {2019}, {2002}, {2000}, {2010}}, {}) << endl;


    // cout << matcher.match(sentence, 1000, 2, 5, 1920301, {}, {{1996}, {1037}, {2019}, {2002}, {2000}, {2010}}) << endl;
    //
//    cout << ConceptNetMatcher::filter({1, 2, 3, 4, 5}, {{2, 3}}) << endl;
//    cout << ConceptNetMatcher::filter({1, 2, 3, 4, 5}, {{4, 5, 6}}) << endl;
//    cout << ConceptNetMatcher::filter({1, 2, 3, 4, 5}, {{1}}) << endl;
}
