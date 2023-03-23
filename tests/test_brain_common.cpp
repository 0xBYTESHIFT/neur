#include <gtest/gtest.h>

#include <iostream>
#include <optional>
#include <xtensor/xio.hpp>
#include <xtensor/xrandom.hpp>

#include "neur/neur.h"

using namespace neur;

using T = float;
using TLayer = layer<T>;
using TBrain = brain<T>;
using TArr = xt::xarray<T>;

const double eps = 1e-6;

void in_out_test( int ins, int outs,
                  std::optional<TArr> result = std::nullopt ) {
    logger::logger_ptr lgr = logger::get_logger();
    lgr->debug( "starting test {}, ins:{}, outs:{}", __func__, ins, outs );
    TArr weights0 = xt::arange<T>( ins * outs ).reshape( { ins, outs } );
    weights0 = ( weights0 + 1 ) / ( ins * outs );
    TArr in0 = xt::ones<T>( { ins } );
    in0 = in0.reshape( { ins } );

    TLayer l0( std::move( weights0 ) );

    TBrain b;
    auto it = b.begin();
    it = b.insert( it, std::move( l0 ) );

    const auto t0 = std::chrono::steady_clock::now();
    const auto out = b.process( in0 );
    const auto t1 = std::chrono::steady_clock::now();
    const auto us =
        std::chrono::duration_cast<std::chrono::microseconds>( t1 - t0 );

    lgr->debug( "out: {}, shape: {}", out,
                logger::shape_to_str( out.shape() ) );
    lgr->debug( "ending test {}, time: {}us", __func__, us.count() );
    if ( result ) {
        auto eq = xt::less_equal( xt::abs( out - *result ), eps );
        lgr->debug( "res:{} out: {} eq: {}", *result, out, eq );
        bool res = xt::all( eq );
        assert( res );
    }
}

void in_mid_out_act_test( int ins, int midsize, int outs,
                          std::optional<TArr> result = std::nullopt,
                          neur::layer<T>::act_t act = neur::acts::sigmoid ) {
    const int size = midsize;
    logger::logger_ptr lgr = logger::get_logger();
    lgr->debug( "starting test {}, ins:{}, mid:{} outs:{}", __func__, ins, size,
                outs );
    TArr weights0 = xt::arange<T>( ins * ins ).reshape( { ins, ins } );
    TArr weights1 = xt::arange<T>( ins * size ).reshape( { ins, size } );
    TArr weights2 = xt::arange<T>( size * outs ).reshape( { size, outs } );
    // scale weights between 0 and 1
    weights0 = ( weights0 + 1 ) / ( ins * ins );
    weights1 = ( weights1 + 1 ) / ( ins * size );
    weights2 = ( weights2 + 1 ) / ( size * outs );
    TArr in0 = xt::ones<T>( { ins } );
    in0 = in0.reshape( { 1, -1 } );

    TBrain b;
    b.emplace_back( std::move( weights0 ) ).set_activation( act );
    b.emplace_back( std::move( weights1 ) ).set_activation( act );
    b.emplace_back( std::move( weights2 ) ).set_activation( act );

    const auto t0 = std::chrono::steady_clock::now();
    const auto out = b.process( in0 );
    const auto t1 = std::chrono::steady_clock::now();
    const auto us =
        std::chrono::duration_cast<std::chrono::microseconds>( t1 - t0 );

    lgr->debug( "out: {}, shape: {}", out,
                logger::shape_to_str( out.shape() ) );
    lgr->debug( "ending test {}, time: {}us", __func__, us.count() );
    if ( result ) {
        auto eq = xt::less_equal( xt::abs( out - *result ), eps );
        lgr->debug( "res:{} out: {} eq: {}", *result, out, eq );
        bool res = xt::all( eq );
        assert( res );
    }
}

void in_mid_out_test( int ins, int midsize, int outs,
                      std::optional<TArr> result = std::nullopt ) {
    const int size = midsize;
    logger::logger_ptr lgr = logger::get_logger();
    lgr->debug( "starting test {}, ins:{}, mid:{} outs:{}", __func__, ins, size,
                outs );
    TArr weights0 = xt::arange<T>( ins * ins ).reshape( { ins, ins } );
    TArr weights1 = xt::arange<T>( ins * size ).reshape( { ins, size } );
    TArr weights2 = xt::arange<T>( size * outs ).reshape( { size, outs } );
    // scale weights between 0 and 1
    weights0 = ( weights0 + 1 ) / ( ins * ins );
    weights1 = ( weights1 + 1 ) / ( ins * size );
    weights2 = ( weights2 + 1 ) / ( size * outs );
    TArr in0 = xt::ones<T>( { ins } );
    in0 = in0.reshape( { 1, -1 } );

    TLayer l0( std::move( weights0 ) );
    TLayer l1( std::move( weights1 ) );
    TLayer l2( std::move( weights2 ) );

    TBrain b;
    b.emplace_back( std::move( l0 ) );
    b.emplace_back( std::move( l1 ) );
    b.emplace_back( std::move( l2 ) );

    const auto t0 = std::chrono::steady_clock::now();
    const auto out = b.process( in0 );
    const auto t1 = std::chrono::steady_clock::now();
    const auto us =
        std::chrono::duration_cast<std::chrono::microseconds>( t1 - t0 );

    lgr->debug( "out: {}, shape: {}", out,
                logger::shape_to_str( out.shape() ) );
    lgr->debug( "ending test {}, time: {}us", __func__, us.count() );
    if ( result ) {
        auto eq = xt::less_equal( xt::abs( out - *result ), eps );
        lgr->debug( "res:{} out: {} eq: {}", *result, out, eq );
        bool res = xt::all( eq );
        assert( res );
    }
}

GTEST_TEST( BrainCommonTests, InOutTest ) {
    logger::logger_ptr lgr = logger::get_logger();
    lgr->set_level( logger::level::debug );
    in_out_test( 1, 1, TArr{ 1 } );
    in_out_test( 1, 2, TArr{ 0.5, 1 } );
    in_out_test( 2, 1, TArr{ 1.5 } );
    in_out_test( 2, 2, TArr{ 1, 1.5 } );
}

GTEST_TEST( BrainCommonTests, InMidOutTest ) {
    logger::logger_ptr lgr = logger::get_logger();
    lgr->set_level( logger::level::debug );
    in_mid_out_test( 1, 1, 1 );
    in_mid_out_test( 1, 1, 2 );
    in_mid_out_test( 1, 2, 1 );
    in_mid_out_test( 1, 2, 2 );
    in_mid_out_test( 2, 1, 1 );
    in_mid_out_test( 2, 1, 2 );
    in_mid_out_test( 2, 2, 1 );
    in_mid_out_test( 2, 2, 2 );
}

GTEST_TEST( BrainCommonTests, InMidOutActTest ) {
    logger::logger_ptr lgr = logger::get_logger();
    lgr->set_level( logger::level::debug );
    in_mid_out_act_test( 1, 1, 1, TArr{ 0.662630 } );
    in_mid_out_act_test( 1, 1, 2, TArr{ 0.583588, 0.66263 } );
    in_mid_out_act_test( 1, 2, 1, TArr{ 0.725165 } );
    in_mid_out_act_test( 1, 2, 2, TArr{ 0.657883, 0.725165 } );
    in_mid_out_act_test( 2, 1, 1, TArr{ 0.682548 } );
    in_mid_out_act_test( 2, 1, 2, TArr{ 0.594537, 0.682548 } );
    in_mid_out_act_test( 2, 2, 1, TArr{ 0.75214 } );
    in_mid_out_act_test( 2, 2, 2, TArr{ 0.678395, 0.75214 } );

    // just a huge test
    in_mid_out_act_test( 100, 1000, 500 );
}
