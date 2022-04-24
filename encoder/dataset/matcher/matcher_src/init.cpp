// cppimport
#define FMT_HEADER_ONLY

#include "matcher.h"
#include "concept_net.h"
#include "pybind11/pybind11.h"
#include "fmt/format.h"
#include "backward-cpp/backward.hpp"
#include <omp.h>

///#define ENABLE_DEBUG
#ifdef ENABLE_DEBUG

#include <execinfo.h>
#include <csignal>

void handler(int sig) {
    void *array[20];
    size_t size;

    // get void*'s for all entries on the stack
    size = backtrace(array, 20);

    // print out all the frames to stderr
    fprintf(stderr, "Error: signal %d:\n", sig);
    backtrace_symbols_fd(array, size, STDERR_FILENO);
    exit(1);
}

#endif

#if defined(_OPENMP)
void set_omp_max_threads(int num) {
    if (num <= 0)
        throw std::invalid_argument("Thread num must be a number larger than 0!");
    int cores = omp_get_max_threads();
    int omp_max_threads = cores > num ? num : cores;
    omp_set_num_threads(omp_max_threads);
}
#endif

namespace py = pybind11;

PYBIND11_MODULE(matcher, m) {
#ifdef ENABLE_DEBUG
    signal(SIGSEGV, handler);
#endif
    backward::SignalHandling sh{};
#if defined(_OPENMP)
    m.def("set_omp_max_threads", &set_omp_max_threads);
#endif
    py::class_<KnowledgeBase>(m, "KnowledgeBase")
            .def(py::init<>())
            .def_readonly("edge_to_target", &KnowledgeBase::edgeToTarget)
            .def_readonly("edge_from_source", &KnowledgeBase::edgeFromSource)
            .def_readonly("edges", &KnowledgeBase::edges)
            .def_readonly("nodes", &KnowledgeBase::nodes)
            .def_readonly("relationships", &KnowledgeBase::relationships)
            .def_readonly("raw_relationships", &KnowledgeBase::rawRelationships)
            .def_readonly("node_embedding_file_name", &KnowledgeBase::nodeEmbeddingFileName)
            .def_readonly("landmark_distances", &KnowledgeBase::landmarkDistances)
            .def_readwrite("tokenized_nodes", &KnowledgeBase::tokenizedNodes)
            .def_readwrite("tokenized_relationships", &KnowledgeBase::tokenizedRelationships)
            .def_readwrite("tokenized_edge_annotations", &KnowledgeBase::tokenizedEdgeAnnotations)
            .def("clear_disabled_edges", &KnowledgeBase::clearDisabledEdges)
            .def("disable_all_edges", &KnowledgeBase::disableAllEdges)
            .def("disable_edges_with_weight_below", &KnowledgeBase::disableEdgesWithWeightBelow)
            .def("disable_edges_of_nodes", &KnowledgeBase::disableEdgesOfNodes)
            .def("disable_edges_of_relationships", &KnowledgeBase::disableEdgesOfRelationships)
            .def("enable_edges_of_relationships", &KnowledgeBase::enableEdgesOfRelationships)
            .def("find_nodes", &KnowledgeBase::findNodes,
                 py::arg("nodes"),
                 py::arg("quiet") = false)
            .def("get_edges", &KnowledgeBase::getEdges)
            .def("get_nodes", py::overload_cast<>(&KnowledgeBase::getNodes, py::const_))
            .def("get_nodes", py::overload_cast<const std::vector<long> &>(&KnowledgeBase::getNodes, py::const_))
            .def("add_composite_node", &KnowledgeBase::addCompositeNode,
                 py::arg("composite_node"),
                 py::arg("relationship"),
                 py::arg("tokenized_composite_node"),
                 py::arg("mask") = std::vector<int>{},
                 py::arg("connection_mask") = std::vector<int>{},
                 py::arg("split_node_minimum_edge_num") = 20,
                 py::arg("split_node_minimum_similarity") = 0.35)
            .def("add_composite_edge", &KnowledgeBase::addCompositeEdge,
                 py::arg("source_node_id"),
                 py::arg("relation_id"),
                 py::arg("composite_node_id"))
            .def("set_node_embedding_file_name", &KnowledgeBase::setNodeEmbeddingFileName,
                 py::arg("path"),
                 py::arg("load_embedding_to_mem") = true)
            .def("is_landmark_inited", &KnowledgeBase::isLandmarkInited)
            .def("init_landmarks", &KnowledgeBase::initLandmarks,
                 py::arg("seed_num") = 100,
                 py::arg("landmark_num") = 20,
                 py::arg("seed") = -1,
                 py::arg("landmark_path") = "")
            .def("distance", &KnowledgeBase::distance,
                 py::arg("node1"),
                 py::arg("node2"),
                 py::arg("fast") = true)
            .def("bfs_distance", &KnowledgeBase::bfsDistance,
                 py::arg("node1"),
                 py::arg("node2"),
                 py::arg("max_depth") = 3)
            .def("is_neighbor", &KnowledgeBase::isNeighbor,
                 py::arg("node1"),
                 py::arg("node2"))
            .def("cosine_similarity", &KnowledgeBase::cosineSimilarity,
                 py::arg("node1"),
                 py::arg("node2"))
            .def("save", &KnowledgeBase::save,
                 py::arg("archive_path"))
            .def("load", &KnowledgeBase::load,
                 py::arg("archive_path"),
                 py::arg("load_embedding_to_mem") = true)
            .def("__repr__",
                 [](const KnowledgeBase &cnr) {
                     return fmt::format("KnowledgeBase(node_num={}, relation_num={}, edge_num={})",
                                        cnr.nodes.size(), cnr.relationships.size(), cnr.edges.size());
                 });

    py::class_<KnowledgeMatcher::MatchResult>(m, "MatchResult")
            .def(pybind11::init<>())
            .def_readonly("target_node_num", &KnowledgeMatcher::MatchResult::targetNodeNum);

    py::class_<KnowledgeMatcher::TrainInfo>(m, "TrainInfo")
            .def(pybind11::init<>())
            .def_readonly("added_edges", &KnowledgeMatcher::TrainInfo::addedEdges)
            .def_readonly("train_connections", &KnowledgeMatcher::TrainInfo::trainConnections);

    py::class_<KnowledgeMatcher>(m, "KnowledgeMatcher")
            .def(py::init<const KnowledgeBase &>())
            .def(py::init<const std::string &>())
            .def_readwrite("kb", &KnowledgeMatcher::kb)
            .def_readonly("corpus_size", &KnowledgeMatcher::corpusSize)
            .def_readonly("document_count_of_node_in_corpus", &KnowledgeMatcher::documentCountOfNodeInCorpus)
            .def("set_corpus", &KnowledgeMatcher::setCorpus)
            .def("find_closest_concept", &KnowledgeMatcher::findClosestConcept)
            .def("match_by_node_embedding", &KnowledgeMatcher::matchByNodeEmbedding,
                 py::arg("source_sentence"),
                 py::arg("target_sentence") = std::vector<int>{},
                 py::arg("source_mask") = std::vector<int>{},
                 py::arg("target_mask") = std::vector<int>{},
                 py::arg("disabled_nodes") = std::vector<long>{},
                 py::arg("max_times") = 100, py::arg("max_depth") = 3, py::arg("seed") = -1,
                 py::arg("edge_top_k") = -1, py::arg("source_context_range") = 0,
                 py::arg("trim_path") = true,
                 py::arg("split_node_minimum_edge_num") = 20,
                 py::arg("split_node_minimum_similarity") = 0.35,
                 py::arg("stop_searching_edge_if_similarity_below") = 0,
                 py::arg("source_context_weight") = 0.2)
            .def("match_result_paths_to_strings", &KnowledgeMatcher::matchResultPathsToStrings)
            .def("join_match_results", &KnowledgeMatcher::joinMatchResults)
            .def("select_paths", &KnowledgeMatcher::selectPaths,
                 py::arg("match_result"),
                 py::arg("max_edges"),
                 py::arg("discard_edges_if_rank_below"),
                 py::arg("filter_short_accurate_paths") = false)
            .def("save", &KnowledgeMatcher::save,
                 py::arg("archive_path"))
            .def("load", &KnowledgeMatcher::load,
                 py::arg("archive_path"),
                 py::arg("load_embedding_to_mem") = true);

    py::class_<ConceptNetReader>(m, "ConceptNetReader")
            .def(py::init<>())
            .def("read", &ConceptNetReader::read,
                 py::arg("asserion_path"),
                 py::arg("weight_path") = "",
                 py::arg("weight_style") = "numberbatch",
                 py::arg("weight_hdf5_path") = "conceptnet_weights.hdf5",
                 py::arg("simplify_with_int8") = true);

#ifdef VERSION_INFO
    m.attr("__version__") = VERSION_INFO;
#else
    m.attr("__version__") = "dev";
#endif
}