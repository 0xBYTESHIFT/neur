#pragma once
#include <xtensor/xarray.hpp>
#include <xtensor/xadapt.hpp>
#include <xtensor/xview.hpp>
#include <xtensor-blas/xlinalg.hpp>
#include <algorithm>
#include "wrapper.h"

namespace neur{

template<class T>
class layer:public wrapper<xt::xarray<T, xt::layout_type::column_major>>{
    using base_t = wrapper<xt::xarray<T, xt::layout_type::column_major>>;
public:
    using container_t = typename base_t::container_t;
    using output_t = typename xt::xarray<T, xt::layout_type::row_major>;
    using input_t = output_t;
    using act_args_t = typename xt::xarray<T>;
    using act_t = std::function<void(T&, act_args_t&)>;

    layer();
    layer(const container_t &data);
    layer(container_t &&data);
    layer(const layer &rhs);
    layer(layer &&data);
    virtual ~layer(){}

    virtual auto neurons()const -> size_t;
    virtual auto links()const -> size_t;
    virtual auto link(const size_t &neur, const size_t &link)const -> const T&;
    virtual auto link(const size_t &neur, const size_t &link) -> T&;
    virtual auto set_weights(const container_t &data) -> void;
    virtual auto set_weights(container_t &&data) -> void;
    virtual auto set_activation_args(const act_args_t &data) -> void;
    virtual auto set_activation_args(act_args_t &&data) -> void;
    virtual auto activation_args() -> act_args_t&;
    virtual auto activation_args()const -> const act_args_t &;
    virtual auto set_activation(const act_t &activation) -> void;
    virtual auto process(const input_t &in) -> output_t;
    virtual auto process(input_t &&in) -> output_t;

    virtual auto operator =(const layer<T> &rhs) -> layer<T>&;
    virtual auto operator =(layer<T> &&rhs) -> layer<T>&;
protected:
    act_t m_activation;
    act_args_t m_act_args;
    using base_t::m_data;
};

template<class T>
bool operator ==(const layer<T> &lhs, const layer<T> &rhs);
template<class T>
bool operator !=(const layer<T> &lhs, const layer<T> &rhs);

template<class T>
layer<T>::layer() {}
template<class T>
layer<T>::layer(const layer &rhs){
    this->m_data = rhs.m_data;
    this->m_activation = rhs.m_activation;
    this->m_act_args = rhs.m_act_args;
}
template<class T>
layer<T>::layer(layer &&rhs){
    this->m_data = std::move(rhs.m_data);
    this->m_activation = std::move(rhs.m_activation);
    this->m_act_args = std::move(rhs.m_act_args);
}
template<class T>
layer<T>::layer(const container_t &data){
    set_weights(data);
}
template<class T>
layer<T>::layer(container_t &&data) {
    set_weights(std::move(data));
}
template<class T>
auto layer<T>::neurons()const -> size_t {
    return m_data.shape().at(1);
}
template<class T>
auto layer<T>::links()const -> size_t {
    return m_data.shape().at(0);
}
template<class T>
auto layer<T>::link(const size_t &neur, const size_t &link)const -> const T&{
    auto ptr = neur + link*neurons();
    return *(m_data.data()+ptr);
}
template<class T>
auto layer<T>::link(const size_t &neur, const size_t &link) -> T&{
    auto ptr = neur + link*neurons();
    return *(m_data.data()+ptr);
}
template<class T>
auto layer<T>::set_weights(const container_t &data) -> void{
    if(data.shape().size() != 2){
        auto mes = "weights shape should be {links, neurons}";
        throw std::runtime_error(std::move(mes));
    }
    this->m_data = data;
}
template<class T>
auto layer<T>::set_weights(container_t &&data) -> void{
    if(data.shape().size() != 2){
        auto mes = "weights shape should be {links, neurons}";
        throw std::runtime_error(std::move(mes));
    }
    this->m_data = std::move(data);
}
template<class T>
auto layer<T>::set_activation_args(const act_args_t &data) -> void{
    if(data.shape().size() != 2 || data.shape().at(1) != neurons()){
        auto mes = "activation_args shape should be {args, neurons}";
        throw std::runtime_error(std::move(mes));
    }
    this->m_act_args = data;
}
template<class T>
auto layer<T>::set_activation_args(act_args_t &&data) -> void{
    if(data.shape().size() != 2 || data.shape().at(1) != neurons()){
        auto mes = "activation_args shape should be {args, neurons}";
        throw std::runtime_error(std::move(mes));
    }
    this->m_act_args = std::move(data);
}
template<class T>
auto layer<T>::activation_args() -> act_args_t&{
    return m_act_args;
}
template<class T>
auto layer<T>::activation_args()const -> const act_args_t &{
    return m_act_args;
}
template<class T>
auto layer<T>::set_activation(const act_t &activation) -> void{
    this->m_activation = activation;
}
template<class T>
auto layer<T>::process(const input_t &in) -> layer<T>::output_t{
    output_t out = xt::empty<T>({(size_t)1, m_data.shape(1)});
    /*
    for(auto i = 0; i<m_data.shape(1); i++){
        const auto &tmp = xt::view(m_data, xt::all(), i);
        xt::view(out, 0, i) = std::inner_product(tmp.begin(), tmp.end(), in.begin(), 0);
    }
    */
    out = xt::linalg::dot(in, m_data);
    if(m_activation){
        for(size_t i=0; i<out.size(); i++){
            T &o = *(out.begin()+i);
            act_args_t args = xt::view(m_act_args, i, xt::all());
            m_activation(o, args);
            xt::view(m_act_args, i, xt::all()) = args;
        }
    }
    return out;
}
template<class T>
auto layer<T>::process(input_t &&in) -> layer<T>::output_t{
    const input_t &tmp = in;
    return process(tmp);
}
template<class T>
bool operator ==(const layer<T> &lhs, const layer<T> &rhs){
    return lhs.container() == rhs.container();
}
template<class T>
bool operator !=(const layer<T> &lhs, const layer<T> &rhs){
    return lhs.container() != rhs.container();
}
template<class T>
layer<T>& layer<T>::operator =(const layer<T> &rhs){
    this->m_data = rhs.m_data;
    return *this;
}
template<class T>
layer<T>& layer<T>::operator =(layer<T> &&rhs){
    this->m_data = std::move(rhs.m_data);
    return *this;
}

}