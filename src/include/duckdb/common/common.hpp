//===----------------------------------------------------------------------===//
//                         DuckDB
//
// duckdb/common/common.hpp
//
//
//===----------------------------------------------------------------------===//

#pragma once

#include "duckdb/common/constants.hpp"
#include "duckdb/common/helper.hpp"
#include "duckdb/common/vector.hpp"

// Pread profiling toggles. Define DUCKDB_PREAD_METRICS_ENABLED=0 at compile time to disable.
#ifndef DUCKDB_PREAD_METRICS_ENABLED
#define DUCKDB_PREAD_METRICS_ENABLED 1
#endif
