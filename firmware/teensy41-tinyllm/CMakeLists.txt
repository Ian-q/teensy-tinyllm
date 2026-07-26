# Copyright (c) 2026 Ian Adelman
# SPDX-License-Identifier: MIT

cmake_minimum_required(VERSION 3.20.0)

find_package(Zephyr REQUIRED HINTS $ENV{ZEPHYR_BASE})
project(teensy41_tinyllm)

# The inference core is plain C99 shared verbatim with the host build. It is
# compiled straight into the app rather than vendored or wrapped, so there is
# exactly one copy of the numerics and the host tests cover the device.
file(GLOB TQ_CORE_SRC ${CMAKE_CURRENT_SOURCE_DIR}/../../core/src/*.c)

target_sources(app PRIVATE
    src/main.c
    src/psram_flexspi2.c
    src/model_store.c
    ${TQ_CORE_SRC}
)

target_include_directories(app PRIVATE
    src
    ${CMAKE_CURRENT_SOURCE_DIR}/../../core/include
    ${CMAKE_CURRENT_SOURCE_DIR}/../../core/src
)

# Warnings-as-errors on our own code only. Scoped to `app` rather than set
# globally so Zephyr's own subsystems (which emit deliberate #warnings, e.g.
# CDC-ACM forcing log levels) do not make the build brittle across SDK bumps.
target_compile_options(app PRIVATE -Werror)
