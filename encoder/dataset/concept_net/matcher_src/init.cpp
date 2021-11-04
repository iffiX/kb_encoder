// cppimport
#define FMT_HEADER_ONLY

#include "matcher.h"
#include "concept_net.h"
#include "pybind11/pybind11.h"
#include "fmt/format.h"
#include <omp.h>

#define ENABLE_DEBUG
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

void set_omp_max_threads(int num) {
    if (num <= 0)
        throw std::invalid_argument("Thread num must be a number larger than 0!");
    int cores = omp_get_max_threads();
    int omp_max_threads = cores > num ? num : cores;
    omp_set_num_threads(omp_max_threads);
}

namespace py = pybind11;

PYBIND11_MODULE(matcher, m) {
#ifdef ENABLE_DEBUG
    signal(SIGSEGV, handler);
#endif
    m.def("set_omp_max_threads", &set_omp_max_threads);
    py::class_<KnowledgeBase>(m, "KnowledgeBase")
            .def(py::init<>())
            .def_readonly("edge_to_target", &KnowledgeBase::edgeToTarget)
            .def_readonly("edge_from_source", &KnowledgeBase::edgeFromSource)
            .def_readonly("edges", &KnowledgeBase::edges)
            .def_readonly("nodes", &KnowledgeBase::nodes)
            .def_readonly("relationships", &KnowledgeBase::relationships)
            .def_readonly("raw_relationships", &KnowledgeBase::rawRelationships)
            .def_readonly("disabled_edges", &KnowledgeBase::disabledEdges)
            .def_readonly("node_embedding_file_name", &KnowledgeBase::nodeEmbeddingFileName)
            .def_readonly("landmark_distances", &KnowledgeBase::landmarkDistances)
            .def_readwrite("tokenized_nodes", &KnowledgeBase::tokenizedNodes)
            .def_readwrite("tokenized_relationships", &KnowledgeBase::tokenizedRelationships)
            .def_readwrite("tokenized_edge_annotations", &KnowledgeBase::tokenizedEdgeAnnotations)
            .def("clear_disabled_edges", &KnowledgeBase::clearDisabledEdges)
            .def("disable_edges_of_nodes", &KnowledgeBase::disableEdgesOfNodes)
            .def("disable_edges_of_relationships", &KnowledgeBase::disableEdgesOfRelationships)
            .def("find_nodes", &KnowledgeBase::findNodes)
            .def("get_edges", &KnowledgeBase::getEdges)
            .def("get_nodes", py::overload_cast<>(&KnowledgeBase::getNodes, py::const_))
            .def("get_nodes", py::overload_cast<const std::vector<long> &>(&KnowledgeBase::getNodes, py::const_))
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

    py::class_<KnowledgeMatcher>(m, "KnowledgeMatcher")
            .def(py::init<const KnowledgeBase &>())
            .def(py::init<const std::string &>())
            .def_readwrite("kb", &KnowledgeMatcher::kb)
            .def("match_by_node", &KnowledgeMatcher::matchByNode,
                 py::arg("source_sentence"),
                 py::arg("target_sentence") = std::vector<int>{},
                 py::arg("source_mask") = std::vector<int>{},
                 py::arg("target_mask") = std::vector<int>{},
                 py::arg("max_times") = 100, py::arg("max_depth") = 3,
                 py::arg("max_edges") = 10, py::arg("seed") = -1,
                 py::arg("edge_beam_width") = -1,
                 py::arg("discard_edges_if_similarity_below") = 0,
                 py::arg("discard_edges_if_rank_below") = 0)
            .def("match_by_node_embedding", &KnowledgeMatcher::matchByNodeEmbedding,
                 py::arg("source_sentence"),
                 py::arg("target_sentence") = std::vector<int>{},
                 py::arg("source_mask") = std::vector<int>{},
                 py::arg("target_mask") = std::vector<int>{},
                 py::arg("max_times") = 100, py::arg("max_depth") = 3,
                 py::arg("max_edges") = 10, py::arg("seed") = -1,
                 py::arg("edge_beam_width") = -1,
                 py::arg("discard_edges_if_similarity_below") = 0.5,
                 py::arg("discard_edges_if_rank_below") = 0)
            .def("match_by_token", &KnowledgeMatcher::matchByToken,
                 py::arg("source_sentence"),
                 py::arg("target_sentence") = std::vector<int>{},
                 py::arg("source_mask") = std::vector<int>{},
                 py::arg("target_mask") = std::vector<int>{},
                 py::arg("max_times") = 100, py::arg("max_depth") = 3,
                 py::arg("max_edges") = 10, py::arg("seed") = -1,
                 py::arg("edge_beam_width") = -1,
                 py::arg("discard_edges_if_similarity_below") = 0,
                 py::arg("discard_edges_if_rank_below") = 0,
                 py::arg("rank_focus") = std::vector<std::vector<int>>{},
                 py::arg("rank_exclude") = std::vector<std::vector<int>>{})
            .def("save", &KnowledgeMatcher::save,
                 py::arg("archive_path"))
            .def("load", &KnowledgeMatcher::load,
                 py::arg("archive_path"),
                 py::arg("load_embedding_to_mem") = true)
            .def("get_node_trie", &KnowledgeMatcher::getNodeTrie)
            .def("get_node_map", &KnowledgeMatcher::getNodeMap);

    py::class_<ConceptNetReader>(m, "ConceptNetReader")
            .def(py::init<>())
            .def("read", &ConceptNetReader::read,
                 py::arg("asserion_path"),
                 py::arg("weight_path") = "",
                 py::arg("weight_hdf5_path") = "conceptnet_weights.hdf5",
                 py::arg("simplify_with_int8") = true);

#ifdef VERSION_INFO
    m.attr("__version__") = VERSION_INFO;
#else
    m.attr("__version__") = "dev";
#endif
}