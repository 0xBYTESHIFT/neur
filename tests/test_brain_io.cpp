#include <iostream>
#include <xtensor/xio.hpp>
#include <xtensor/xrandom.hpp>

#include "neur/brain.h"
#include "neur/io/loader.h"
#include "neur/io/saver.h"

using namespace neur;

int main( int argc, char *argv[] ) {
    using T = float;
    using layer = layer<T>;
    using brain = brain<T>;
    const int ins = 16;
    const int outs = 8;
    const int size = 32;

    xt::xarray<T> weights0 =
        xt::arange<T>( ins * size ).reshape( { ins, size } );
    xt::xarray<T> weights1 =
        xt::arange<T>( size * size ).reshape( { size, size } );
    xt::xarray<T> weights2 =
        xt::arange<T>( size * outs ).reshape( { size, outs } );
    xt::xarray<T> in0 = xt::random::rand<float>( { ins } ) * 1000;
    layer l0( std::move( weights0 ) );
    layer l1( std::move( weights1 ) );
    layer l2( std::move( weights2 ) );
    auto act = []( layer::output_t &out ) { out = xt::minimum( out, 1 ); };
    l0.set_activation( act );
    l1.set_activation( act );
    l2.set_activation( act );

    brain b;
    auto it = b.begin();
    it = b.insert( it, std::move( l0 ) );
    it = b.insert( it, std::move( l1 ) );
    it = b.insert( it, std::move( l2 ) );

    nlohmann::json json = b;
    // std::cout << "json:" << std::setw(4) << json << "\n";
    auto b_load = json.get<neur::brain<float>>();
    assert( b == b_load );
    std::cout << "io test successful\n";

    return 0;
}
