#pragma once

#include <nlohmann/json.hpp>
#include <xtensor/xjson.hpp>

#include "neur/brain.h"
#include "neur/layer.h"

namespace neur {
using json = nlohmann::json;

template <class T>
auto from_json( const json &j, layer<T> &l );
template <class T>
auto from_json( const json &j, brain<T> &b );

template <class T>
auto from_json( const json &j, layer<T> &l ) {
    auto num_neurs = j.at( "num_neurs" ).get<size_t>();
    auto num_links = j.at( "num_links" ).get<size_t>();
    j.at( "layer_data" ).get_to( l.container() );
    if ( num_neurs * num_links != l.container().size() ) {
        auto mes = "num_neurs*num_links != json_layer_data.size()";
        throw std::runtime_error( mes );
    }
}

template <class T>
auto from_json( const json &j, brain<T> &b ) {
    auto num_layers = j.at( "num_layers" ).get<size_t>();
    j.at( "layers_data" ).get_to( b.container() );
}

};  // namespace neur
