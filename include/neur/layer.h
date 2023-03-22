#pragma once
#include <algorithm>
#include <xtensor/xadapt.hpp>
#include <xtensor/xarray.hpp>
#include <xtensor/xio.hpp>
#include <xtensor/xview.hpp>

#include "neur/logger.h"
#include "neur/multiplier_impl.h"
#include "neur/wrapper.h"

namespace neur {

template <class T>
class layer : public wrapper<xt::xarray<T, xt::layout_type::column_major> > {
    using lgr = logger;
    using base_t = wrapper<xt::xarray<T, xt::layout_type::column_major> >;

   public:
    using container_t = typename base_t::container_t;
    using output_t = typename xt::xarray<T, xt::layout_type::row_major>;
    using input_t = output_t;
    using act_t = std::function<void( output_t& )>;

    layer();
    layer( const container_t& data );
    layer( container_t&& data );
    layer( const layer& rhs );
    layer( layer&& data );
    virtual ~layer();

    auto neurons() const -> size_t;
    auto links() const -> size_t;
    auto link( size_t neur, size_t link ) const -> const T&;
    auto link( size_t neur, size_t link ) -> T&;
    void set_weights( const container_t& data );
    void set_weights( container_t&& data );
    void set_activation( const act_t& activation );
    auto process( const input_t& in ) -> output_t;
    auto process( input_t&& in ) -> output_t;

    auto operator=( const layer<T>& rhs ) -> layer<T>&;
    auto operator=( layer<T>&& rhs ) -> layer<T>&;
    template <class T2>
    auto operator==( const layer<T2>& rhs ) const -> bool;
    template <class T2>
    auto operator!=( const layer<T2>& rhs ) const -> bool;

   protected:
    using base_t::m_data;

    lgr::logger_ptr m_lgr = logger::get_logger();
    act_t m_activation;
};

template <class T>
layer<T>::layer() {}

template <class T>
layer<T>::layer( const layer& rhs ) {
    m_lgr->trace( "layer created copy" );
    this->m_data = rhs.m_data;
    this->m_activation = rhs.m_activation;
}

template <class T>
layer<T>::layer( layer&& rhs ) {
    m_lgr->trace( "layer created move" );
    this->m_data = std::move( rhs.m_data );
    this->m_activation = std::move( rhs.m_activation );
}

template <class T>
layer<T>::layer( const container_t& data ) {
    set_weights( data );
}

template <class T>
layer<T>::layer( container_t&& data ) {
    set_weights( std::move( data ) );
}

template <class T>
layer<T>::~layer() {}

template <class T>
auto layer<T>::neurons() const -> size_t {
    return m_data.shape().at( 1 );
}
template <class T>
auto layer<T>::links() const -> size_t {
    return m_data.shape().at( 0 );
}
template <class T>
auto layer<T>::link( size_t neur, size_t link ) const -> const T& {
    m_lgr->trace( "layer linking with {} neurs, {} links", neur, link );
    auto ptr = neur + link * neurons();
    return *( m_data.data() + ptr );
}

template <class T>
auto layer<T>::link( size_t neur, size_t link ) -> T& {
    m_lgr->trace( "layer linking with {} neurs, {} links", neur, link );
    auto ptr = neur + link * neurons();
    return *( m_data.data() + ptr );
}

template <class T>
void layer<T>::set_weights( const container_t& data ) {
    std::cout << lgr::shape_to_str( data.shape() ) << std::endl;
    m_lgr->trace( "layer setting weights copy: {}, shape: {}", data,
                  lgr::shape_to_str( data.shape() ) );
    if ( data.shape().size() != 2 ) {
        auto mes = "weights shape should be {links, neurons}";
        throw std::runtime_error( std::move( mes ) );
    }
    this->m_data = data;
}

template <class T>
void layer<T>::set_weights( container_t&& data ) {
    m_lgr->trace( "layer setting weights move: {}, shape: {}", data,
                  lgr::shape_to_str( data.shape() ) );
    if ( data.shape().size() != 2 ) {
        auto mes = "weights shape should be {links, neurons}";
        throw std::runtime_error( std::move( mes ) );
    }
    this->m_data = std::move( data );
}

template <class T>
void layer<T>::set_activation( const act_t& activation ) {
    m_lgr->trace( "layer setting activation" );
    this->m_activation = activation;
}

template <class T>
auto layer<T>::process( const input_t& in ) -> output_t {
    m_lgr->trace( "layer processing in shape: {}, in: {}, lr data shape:{}",
                  lgr::shape_to_str( in.shape() ), lgr::to_str( in ),
                  lgr::shape_to_str( m_data.shape() ) );
    output_t out = neur::multiply( in, this->m_data );
    m_lgr->trace( "layer processing out shape: {}, out: {}",
                  lgr::shape_to_str( out.shape() ), lgr::to_str( out ) );
    if ( !m_activation ) {
        return out;
    }
    m_activation( out );
    m_lgr->trace( "layer processing out activation shape: {}, out: {}",
                  lgr::shape_to_str( out.shape() ), lgr::to_str( out ) );
    return out;
}

template <class T>
auto layer<T>::process( input_t&& in ) -> output_t {
    const input_t& tmp = in;
    return process( tmp );
}

template <class T>
auto layer<T>::operator=( const layer<T>& rhs ) -> layer<T>& {
    this->m_data = rhs.m_data;
    return *this;
}

template <class T>
auto layer<T>::operator=( layer<T>&& rhs ) -> layer<T>& {
    this->m_data = std::move( rhs.m_data );
    return *this;
}

template <class T>
template <class T2>
auto layer<T>::operator==( const layer<T2>& rhs ) const -> bool {
    return this->container() == rhs.container();
}

template <class T>
template <class T2>
auto layer<T>::operator!=( const layer<T2>& rhs ) const -> bool {
    return this->container() != rhs.container();
}

}  // namespace neur
