#pragma once
#include "neur/layer.h"
#include "neur/logger.h"
#include "neur/wrapper.h"

namespace neur {

template <class T>
class brain : public dyn_wrapper<std::vector<layer<T> > > {
    using lgr = logger;
    using base_t = dyn_wrapper<std::vector<layer<T> > >;

   public:
    using iterator = typename base_t::iterator;
    using value_type = typename base_t::value_type;
    using input_t = typename layer<T>::input_t;
    using output_t = typename layer<T>::output_t;

    auto process( const input_t& in ) -> output_t;
    auto process( input_t&& in ) -> output_t;

    template <class T2>
    auto operator==( const brain<T2>& rhs ) const -> bool;
    template <class T2>
    auto operator!=( const brain<T2>& rhs ) const -> bool;

   protected:
    using base_t::m_data;

    lgr::logger_ptr m_lgr = logger::get_logger();

    template <class Arg>
    auto p_inference( Arg&& ) -> output_t;
};

template <class T>
auto brain<T>::process( const input_t& in ) -> output_t {
    m_lgr->trace( "brain processing, copy data, shape: {}",
                  lgr::shape_to_str( in.shape() ) );
    output_t out = p_inference( in );
    return out;
}

template <class T>
auto brain<T>::process( input_t&& in ) -> output_t {
    m_lgr->trace( "brain processing, copy move, shape: {}",
                  lgr::shape_to_str( in.shape() ) );
    output_t out = p_inference( std::move( in ) );
    return out;
}

template <class T>
template <class Arg>
auto brain<T>::p_inference( Arg&& in ) -> output_t {
    m_lgr->trace( "brain inferencing, in shape: {}, in: {}",
                  lgr::shape_to_str( in.shape() ), lgr::to_str( in ) );
    output_t out = std::forward<Arg>( in );
    size_t i = 0;
    for ( decltype( auto ) l : m_data ) {
        output_t tmp = l.process( std::move( out ) );
        out = std::move( tmp );
        m_lgr->trace( "brain inferencing #{}, out shape: {}, out: {}", i,
                      lgr::shape_to_str( out.shape() ), lgr::to_str( out ) );
        ++i;
    }
    out.reshape( { -1 } );
    m_lgr->trace( "brain inferencing done, out shape: {}, out: {}",
                  lgr::shape_to_str( in.shape() ), lgr::to_str( out ) );
    return std::move( out );
}

template <class T>
template <class T2>
auto brain<T>::operator==( const brain<T2>& rhs ) const -> bool {
    return this->container() == rhs.container();
}

template <class T>
template <class T2>
auto brain<T>::operator!=( const brain<T2>& rhs ) const -> bool {
    return this->container() != rhs.container();
}

};  // namespace neur
