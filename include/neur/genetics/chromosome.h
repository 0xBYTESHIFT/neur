#pragma once
#include <vector>
#include "wrapper.h"
#include "layer.h"
#include "brain.h"

namespace neur{

template<class T>
class chromosome:public dynamic_wrapper<std::vector<T>>{
    using base_t = dynamic_wrapper<std::vector<T>>;
public:
    using base_t::at;
    using base_t::size;
    using base_t::begin;
    using base_t::end;
    using base_t::cbegin;
    using base_t::cend;
    using base_t::container;
    using base_t::data;
    using base_t::insert;
    using base_t::erase;
    using base_t::reserve;
    using base_t::resize;
protected:
    using base_t::m_data;
};

template<class T>
class layer_genes:public chromosome<T>{
    using base_t = chromosome<T>;
public:
    using base_t::at;
    using base_t::size;
    using base_t::begin;
    using base_t::end;
    using base_t::cbegin;
    using base_t::cend;
    using base_t::container;
    using base_t::data;
    using base_t::insert;
    using base_t::erase;
    using base_t::reserve;
    using base_t::resize;
    using layer_t = layer<T>;

    void from_layer(const layer_t &l);
    auto to_layer()const -> layer_t;
protected:
    using base_t::m_data;
};
template<class T>
class brain_genes:public chromosome<T>{
    using base_t = chromosome<T>;
public:
    using base_t::at;
    using base_t::size;
    using base_t::begin;
    using base_t::end;
    using base_t::cbegin;
    using base_t::cend;
    using base_t::container;
    using base_t::data;
    using base_t::insert;
    using base_t::erase;
    using base_t::reserve;
    using base_t::resize;

    void from_brain(const brain<T> &b);
    auto to_brain()const -> brain<T>;
};

template<class T>
void layer_genes<T>::from_layer(const layer_t &lr){
    m_data.clear();

    const auto neurs = lr.neurons();
    const auto links = lr.links();
    m_data.reserve(2 + neurs*links);
    m_data.emplace_back(neurs);
    m_data.emplace_back(links);
    for(size_t n=0; n<neurs; n++){
        for(size_t l=0; l<links; l++){
            T gene = lr.link(n, l);
            m_data.emplace_back(std::move(gene));
        }
    }
};

template<class T>
auto layer_genes<T>::to_layer()const -> layer_t{
    const auto neurs = m_data.at(0);
    const auto links = m_data.at(1);
    xt::xarray<T> weights = xt::zeros<T>({links, neurs});
    auto it = m_data.begin()+2;
    for(size_t n=0; n<neurs; n++){
        for(size_t l=0; l<links; l++){
            xt::view(weights, l, n) = *it;
            it++;
        }
    }
    return layer_t(std::move(weights));
}

template<class T>
void brain_genes<T>::from_brain(const brain<T> &b){
    size_t lrs = b.size();
    reserve(lrs);
    auto &cnt = container();
    cnt.emplace_back(std::move(lrs));

    for(const auto &lr:b){
        layer_genes<T> chr;
        chr.from_layer(lr);
        reserve(size()+chr.size());
        std::move(chr.begin(), chr.end(), std::back_inserter(cnt));
    }
}

template<class T>
auto brain_genes<T>::to_brain()const -> brain<T>{
    brain<T> b;
    auto it = begin();
    size_t lrs = *it;
    it++;
    while(it != end()){
        layer_genes<T> chr;
        auto neurs = *(it+0);
        auto links = *(it+1);
        auto size = neurs*links+2;
        chr.resize(size);
        std::copy(it, it+size, chr.begin());
        auto lr = chr.to_layer();
        b.insert(b.begin()+b.size(), std::move(lr));
        it += chr.size();
    }
    return b;
}

}