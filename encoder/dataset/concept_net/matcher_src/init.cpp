// cppimport
#define FMT_HEADER_ONLY
#include "matcher.h"
#include "pybind11/pybind11.h"
#include "fmt/format.h"
#include <omp.h>
#define ENABLE_DEBUG
#ifdef ENABLE_DEBUG
#include <execinfo.h>
#include <signal.h>

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
    py::class_<ConceptNetReader>(m, "ConceptNetReader")
            .def(py::init<const std::string &>())
            .def_readonly("edge_to_target", &ConceptNetReader::edgeToTarget)
            .def_readonly("edge_from_source", &ConceptNetReader::edgeFromSource)
            .def_readonly("edges", &ConceptNetReader::edges)
            .def_readonly("nodes", &ConceptNetReader::nodes)
            .def_readonly("relationships", &ConceptNetReader::relationships)
            .def_readonly("raw_relationships", &ConceptNetReader::rawRelationships)
            .def_readwrite("tokenized_nodes", &ConceptNetReader::tokenizedNodes)
            .def_readwrite("tokenized_relationships", &ConceptNetReader::tokenizedRelationships)
            .def_readwrite("tokenized_edge_annotations", &ConceptNetReader::tokenizedEdgeAnnotations)
            .def("get_edges", &ConceptNetReader::getEdges)
            .def("get_nodes", py::overload_cast<>(&ConceptNetReader::getNodes, py::const_))
            .def("get_nodes", py::overload_cast<const std::vector<long> &>(&ConceptNetReader::getNodes, py::const_))
            .def("__repr__",
                 [](const ConceptNetReader &cnr) {
                     return fmt::format("ConceptNetReader(node_num={}, relation_num={}, edge_num={})",
                                        cnr.nodes.size(), cnr.relationships.size(), cnr.edges.size());
                 });

    py::class_<ConceptNetMatcher>(m, "ConceptNetMatcher")
            .def(py::init<const ConceptNetReader &>())
            .def(py::init<const std::string &>())
            .def("match", &ConceptNetMatcher::match,
                 py::arg("sentence"), py::arg("max_times") = 100, py::arg("max_depth") = 3,
                 py::arg("max_edges") = 10, py::arg("seed") = -1,
                 py::arg("similarity_exclude") = std::vector<std::vector<int>>{},
                 py::arg("rank_focus") = std::vector<std::vector<int>>{},
                 py::arg("rank_exclude") = std::vector<std::vector<int>>{})
            .def("save", &ConceptNetMatcher::save)
            .def("load", &ConceptNetMatcher::load)
            .def("get_node_trie", &ConceptNetMatcher::getNodeTrie)
            .def("get_node_map", &ConceptNetMatcher::getNodeMap);

#ifdef VERSION_INFO
    m.attr("__version__") = VERSION_INFO;
#else
    m.attr("__version__") = "dev";
#endif
}

/*
<%
cfg['sources'] = ['matcher.cpp']
cfg['dependencies'] = ['matcher.h']
cfg['compiler_args'] = ['-std=c++17', '-fopenmp', '-O3']
cfg['extra_link_args'] = ['-fopenmp']
setup_pybind11(cfg)
%>
*/