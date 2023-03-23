#pragma once
#include <xtensor-blas/xlinalg.hpp>
#include <xtensor/xadapt.hpp>
#include <xtensor/xarray.hpp>
#include <xtensor/xview.hpp>

namespace neur {

template <class Arr1, class Arr2>
auto multiply( Arr1&& arr1, Arr2& arr2 ) {
    return xt::linalg::dot( std::forward<Arr1>( arr1 ),
                            std::forward<Arr2>( arr2 ) );
}

};  // namespace neur
