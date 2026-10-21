#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/stl_bind.h>
#include "tokenizer.hpp"

namespace py = pybind11;

PYBIND11_MODULE(_cpp_tokenizer, m) {
    m.doc() = "Fast C++ BPE tokenizer for curriculum-trained vocabularies";

    py::class_<curriculum::Tokenizer>(m, "Tokenizer")
        .def(py::init<>())
        .def("load", &curriculum::Tokenizer::load,
             py::arg("vocab_path"), py::arg("merges_path"),
             "Load tokenizer from vocab.json and merges.txt files")
        .def("encode", &curriculum::Tokenizer::encode,
             py::arg("text"),
             py::call_guard<py::gil_scoped_release>(),
             "Encode text to list of token IDs")
        .def("decode", &curriculum::Tokenizer::decode,
             py::arg("ids"),
             py::call_guard<py::gil_scoped_release>(),
             "Decode list of token IDs to text")
        .def("encode_batch", &curriculum::Tokenizer::encode_batch,
             py::arg("texts"), py::arg("num_threads") = 0,
             py::call_guard<py::gil_scoped_release>(),
             "Batch encode multiple texts")
        .def("set_cache_capacity", &curriculum::Tokenizer::set_cache_capacity,
             py::arg("capacity"), "Set the bounded piece-cache capacity; zero disables caching")
        .def("clear_cache", &curriculum::Tokenizer::clear_cache,
             "Clear cached pieces and cache counters")
        .def("cache_info", &curriculum::Tokenizer::cache_info,
             "Get cache capacity, entries, hits, misses, and payload bytes")
        .def("vocab_size", &curriculum::Tokenizer::vocab_size,
             "Get vocabulary size")
        .def("id_to_token", &curriculum::Tokenizer::id_to_token,
             py::arg("id"),
             "Get token string for an ID")
        .def("token_to_id", &curriculum::Tokenizer::token_to_id,
             py::arg("token"),
             "Get ID for a token string (-1 if not found)")
        .def("set_special_tokens", &curriculum::Tokenizer::set_special_tokens,
             py::arg("tokens"),
             "Register existing vocabulary tokens as atomic controls")
        .def("set_special_token_aliases", &curriculum::Tokenizer::set_special_token_aliases,
             py::arg("aliases"),
             "Register explicit legacy decode-only aliases without changing learned tokens")
        .def("configure_special_tokens", &curriculum::Tokenizer::configure_special_tokens,
             py::arg("token_ids"), py::arg("aliases"), py::arg("expected_vocab_size"),
             "Validate configured controls and append only explicitly declared legacy tail IDs");
}
