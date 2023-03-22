#pragma once

#include <functional>
#include <xtensor/xtensor.hpp>

namespace neur {
namespace acts {

// 1 / 1 + e^( -x )
static const auto sigmoid = []( auto& out ) {
    auto out_tmp = 1 / ( 1 + xt::exp( -out ) );
    out = std::move( out_tmp );
};

// nop
static const auto linear = []( auto& out ) {};

}  // namespace acts

}  // namespace neur
