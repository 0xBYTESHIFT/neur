#pragma once

#include <spdlog/async.h>
#include <spdlog/cfg/env.h>  // support for loading levels from the environment variable
#include <spdlog/fmt/ostr.h>  // support for user defined types
#include <spdlog/sinks/stdout_color_sinks.h>
#include <spdlog/spdlog.h>

#include <memory>
#include <sstream>
#include <vector>

#include "xtensor/xadapt.hpp"

namespace neur {

class logger {
   public:
    using level = spdlog::level::level_enum;
    using logger_ptr = std::shared_ptr<spdlog::async_logger>;

    static auto get_logger() -> logger_ptr;
    template <class T>
    static auto to_str( const T& data ) -> std::string;
    template <class T>
    static auto shape_to_str( const T& data ) -> std::string;

   private:
    static void p_init();

    static inline logger_ptr p_logger = nullptr;
};

inline void logger::p_init() {
    spdlog::init_thread_pool( 8192, 1 );
    auto stdout_sink = std::make_shared<spdlog::sinks::stdout_color_sink_mt>();
    std::vector<spdlog::sink_ptr> sinks{ stdout_sink };
    p_logger = std::make_shared<spdlog::async_logger>(
        "logger", sinks.begin(), sinks.end(), spdlog::thread_pool(),
        spdlog::async_overflow_policy::block );
    spdlog::register_logger( p_logger );
}

inline auto logger::get_logger() -> logger_ptr {
    static std::shared_ptr<logger> p_logger_instance = nullptr;
    if ( !logger::p_logger ) {
        p_init();
    }
    return p_logger;
}

template <class T>
auto logger::to_str( const T& data ) -> std::string {
    std::stringstream ss;
    ss << data;
    return ss.str();
}

template <class T>
auto logger::shape_to_str( const T& shape ) -> std::string {
    std::stringstream ss;
    ss << xt::adapt( shape );
    return ss.str();
}

}  // namespace neur
