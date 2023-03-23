#pragma once
#include <cstddef>
#include <utility>

namespace neur {

template <class T>
class wrapper {
   public:
    using container_t = T;
    using value_type = typename T::value_type;
    using iterator = typename T::iterator;
    using const_iterator = typename T::const_iterator;

    wrapper();
    wrapper( const wrapper &rhs );
    wrapper( wrapper &&rhs );
    ~wrapper() = default;

    auto at( const size_t &i ) const -> const value_type &;
    auto at( const size_t &i ) -> value_type &;
    auto size() const -> size_t;
    auto begin() const -> const_iterator;
    auto end() const -> const_iterator;
    auto cbegin() const -> const_iterator;
    auto cend() const -> const_iterator;
    auto begin() -> iterator;
    auto end() -> iterator;
    auto container() const -> const container_t &;
    auto container() -> container_t &;
    auto data() const -> const value_type *;
    auto data() -> value_type *;

    auto operator=( const wrapper<T> &lhs ) -> wrapper<T>;
    auto operator=( wrapper<T> &&lhs ) -> wrapper<T>;

   protected:
    T m_data;
};

// dynamic wrapper
template <class T>
class dyn_wrapper : public wrapper<T> {
    using base_t = wrapper<T>;

   public:
    using container_t = typename base_t::container_t;
    using value_type = typename base_t::value_type;
    using iterator = typename base_t::iterator;

    using base_t::at;
    using base_t::begin;
    using base_t::cbegin;
    using base_t::cend;
    using base_t::container;
    using base_t::data;
    using base_t::end;
    using base_t::size;

    // inserts value before iterator. returns iterator to inserted value
    auto insert( const iterator it, const value_type &l ) -> iterator;
    auto insert( const iterator it, value_type &&l ) -> iterator;
    // emplaces value before iterator. returns iterator to inserted value
    template <class... Args>
    auto emplace( const iterator it, Args &&...args ) -> iterator;
    // emplaces value in the end
    template <class... Args>
    decltype( auto ) emplace_back( Args &&...args );

    auto erase( const iterator &it ) -> iterator;
    auto erase( iterator &&it ) -> iterator;
    void reserve( const size_t &s );
    void resize( const size_t &s );
    auto clear();

   protected:
    using base_t::m_data;
};

template <class T>
wrapper<T>::wrapper() {}

template <class T>
wrapper<T>::wrapper( const wrapper &rhs ) {
    this->m_data = rhs.m_data;
}

template <class T>
wrapper<T>::wrapper( wrapper &&rhs ) {
    this->m_data = std::move( rhs.m_data );
}

template <class T>
auto wrapper<T>::at( const size_t &i ) const -> const value_type & {
    return m_data.at( i );
}
template <class T>
auto wrapper<T>::at( const size_t &i ) -> value_type & {
    return m_data.at( i );
}
template <class T>
auto wrapper<T>::size() const -> size_t {
    return m_data.size();
}
template <class T>
auto wrapper<T>::begin() const -> const_iterator {
    return m_data.begin();
}
template <class T>
auto wrapper<T>::end() const -> const_iterator {
    return m_data.end();
}
template <class T>
auto wrapper<T>::cbegin() const -> const_iterator {
    return m_data.begin();
}
template <class T>
auto wrapper<T>::cend() const -> const_iterator {
    return m_data.end();
}
template <class T>
auto wrapper<T>::begin() -> iterator {
    return m_data.begin();
}
template <class T>
auto wrapper<T>::end() -> iterator {
    return m_data.end();
}
template <class T>
auto wrapper<T>::container() const -> const container_t & {
    return m_data;
}
template <class T>
auto wrapper<T>::container() -> container_t & {
    return m_data;
}
template <class T>
auto wrapper<T>::data() const -> const value_type * {
    return m_data.data();
}
template <class T>
auto wrapper<T>::data() -> value_type * {
    return m_data.data();
}
template <class T>
auto wrapper<T>::operator=( const wrapper<T> &lhs ) -> wrapper<T> {
    this->m_data = lhs.m_data;
    return *this;
}
template <class T>
auto wrapper<T>::operator=( wrapper<T> &&lhs ) -> wrapper<T> {
    this->m_data = std::move( lhs.m_data );
    return *this;
}

template <class T>
auto dyn_wrapper<T>::insert( const iterator it, const value_type &l )
    -> iterator {
    return m_data.insert( it, l );
}
template <class T>
auto dyn_wrapper<T>::insert( const iterator it, value_type &&l ) -> iterator {
    return m_data.insert( it, std::move( l ) );
}
template <class T>
template <class... Args>
auto dyn_wrapper<T>::emplace( const iterator it, Args &&...args ) -> iterator {
    return m_data.emplace( it, std::forward<Args &&>( args )... );
}
template <class T>
template <class... Args>
decltype( auto ) dyn_wrapper<T>::emplace_back( Args &&...args ) {
    m_data.emplace_back( std::forward<Args &&>( args )... );
    return *( m_data.end() - 1 );
}
template <class T>
auto dyn_wrapper<T>::erase( const iterator &it ) -> iterator {
    return m_data.erase( it );
}
template <class T>
auto dyn_wrapper<T>::erase( iterator &&it ) -> iterator {
    return m_data.erase( it );
}
template <class T>
void dyn_wrapper<T>::reserve( const size_t &s ) {
    m_data.reserve( s );
}
template <class T>
void dyn_wrapper<T>::resize( const size_t &s ) {
    m_data.resize( s );
}
template <class T>
auto dyn_wrapper<T>::clear() {
    m_data.clear();
}
};  // namespace neur
