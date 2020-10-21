#pragma once
#include <nlohmann/json.hpp>
#include <vector>
#include "neur/layer.h"
#include "neur/brain.h"

namespace neur{

using json = nlohmann::json;

template<class T>
auto to_json(json& j, const layer<T> &l);

template<class T>
auto to_json(json& k, const brain<T> &b);

template<class T>
auto to_json(json& j, const layer<T> &l){
    j = json{{"num_neurs", l.neurons()},
        {"num_links", l.links()},
        {"layer_data", l.container()}
    };
}

template<class T>
auto to_json(json& j, const brain<T> &b){
    j = json{{"num_layers", b.size()},
        {"layers_data", b.container()}
    };
}

};