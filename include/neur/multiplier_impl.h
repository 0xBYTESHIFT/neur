#pragma once
#include <xtensor/xarray.hpp>
#include <xtensor/xadapt.hpp>
#include <xtensor/xview.hpp>

namespace neur{

#ifdef USE_BLAS
#include <xtensor-blas/xlinalg.hpp>

    template<class Arr1, class Arr2>
    auto multiply(Arr1 &&arr1, Arr2 &arr2){
        return xt::linalg::dot(std::forward<Arr1>(arr1), std::forward<Arr2>(arr2));
    }
#else
    template<class Arr1, class Arr2>
    auto multiply(Arr1 &&arr1, Arr2 &&arr2){
        using Arr1_dec = std::decay_t<Arr1>;
        using Arr2_dec = std::decay_t<Arr2>;
        using Arr1_val_t = typename Arr1_dec::value_type;
        using Arr2_val_t = typename Arr2_dec::value_type;
        using val_t = typename std::common_type<Arr1_val_t, Arr2_val_t>::type;
        xt::xarray<val_t> out = xt::empty<val_t>({(int)1, (int)arr2.shape(1)});
        for(auto i = 0; i<arr2.shape(1); i++){
            auto tmp = xt::view(arr2, xt::all(), i);
            xt::view(out, 0, i) = std::inner_product(tmp.begin(), tmp.end(), arr1.begin(), (val_t)0);
        }
        return out;
    }
#endif

};