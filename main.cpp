#include <iostream>
#include <xtensor/xio.hpp>
#include <xtensor/xrandom.hpp>
#include "brain.h"

using namespace neur;

int main(int argc, char* argv[]){
    using T = float;
    using layer = layer<T>;
    using brain = brain<T>;
    const int ins  =  100;
    const int outs =  10;
    const int size = 5000;

    xt::xarray<T> weights0 = xt::arange<T>(size).reshape({ ins, size});
    xt::xarray<T> weights1 = xt::arange<T>(size*size).reshape({size, size});
    xt::xarray<T> weights2 = xt::arange<T>(size*outs).reshape({ size, outs});
    xt::xarray<T> in0 = xt::random::rand<float>({ins})*1000;
    layer l0(std::move(weights0));
    layer l1(std::move(weights1));
    layer l2(std::move(weights2));
    auto act = [](T &out, decltype(l0)::act_args_t &args){
        out = std::min(out, *args.begin());
    };
    l0.set_activation(act);
    l1.set_activation(act);
    l2.set_activation(act);
    l0.set_activation_args(xt::ones<T>({1, size}));
    l1.set_activation_args(xt::ones<T>({1, size}));
    l2.set_activation_args(xt::ones<T>({1, outs}));

    brain b;
    b.insert(b.begin()+0, std::move(l0));
    b.insert(b.begin()+1, std::move(l1));
    b.insert(b.begin()+2, std::move(l2));
    size_t i=0;
    for(auto &lr:b){
        std::cout << i << "/" << b.size()
            << " " << xt::adapt(lr.container().shape())
            << std::endl;
        i++;
    }
    auto t0 = std::chrono::steady_clock::now();
    auto out = b.process(in0);
    auto t1 = std::chrono::steady_clock::now();
    assert(xt::all(xt::equal(out, 1.f)));

    auto us = std::chrono::duration_cast<std::chrono::microseconds>(t1-t0);
    std::cout << us.count() << "us\n";
    std::cout << "in0" << in0 << std::endl;
    std::cout << "out" << out << std::endl;
    brain b2 = b; //test copy;
    brain b3 = std::move(b); //test move;
    assert(b2 == b3);

    return 0;
}