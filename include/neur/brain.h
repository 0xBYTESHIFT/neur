#pragma once
#include "neur/layer.h"
#include "neur/wrapper.h"

namespace neur{

template<class T>
class brain:public dynamic_wrapper<std::vector<layer<T>>>{
    using base_t = dynamic_wrapper<std::vector<layer<T>>>;
public:
    using iterator = typename base_t::iterator;
    using value_type = typename base_t::value_type;
    using input_t = typename layer<T>::input_t;
    using output_t = typename layer<T>::output_t;

    auto process(const input_t &in) -> output_t;
    auto process(input_t &&in) -> output_t;
protected:
    auto p_inference(output_t &&) -> output_t;
    using base_t::m_data;
};

template<class T>
bool operator ==(const brain<T> &lhs, const brain<T> &rhs);
template<class T>
bool operator !=(const brain<T> &lhs, const brain<T> &rhs);

template<class T>
auto brain<T>::process(const input_t &in)-> output_t{
    output_t out = in;
    out = p_inference(std::move(out));
    return out;
}
template<class T>
auto brain<T>::process(input_t &&in)-> output_t{
    output_t out = p_inference(std::move(in));
    return out;
}
template<class T>
auto brain<T>::p_inference(output_t &&out)-> output_t{
    out.reshape({1, -1});
    for(auto &l:m_data){
        out = l.process(std::move(out));
    }
    out.reshape({-1});
    return std::move(out);
}

template<class T>
bool operator ==(const brain<T> &lhs, const brain<T> &rhs){
    return lhs.container() == rhs.container();
}
template<class T>
bool operator !=(const brain<T> &lhs, const brain<T> &rhs){
    return lhs.container() != rhs.container();
}

};