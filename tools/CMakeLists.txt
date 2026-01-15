if(NOT SUN AND BUILD_SHELL)
  add_subdirectory(sqlite3_api_wrapper)
  add_subdirectory(shell)
endif()

# DDS parquet preload utility (requires DDS POSIX frontend).
if(DUCKDB_USE_DDS_POSIX)
  add_executable(dds_parquet_loader utils/dds_parquet_loader.cpp)
  target_include_directories(dds_parquet_loader
                             PRIVATE
                             ${DUCKDB_DDS_POSIX_DIR}
                             ${DUCKDB_DDS_COMMON_INCLUDE_DIR}
                             ${DUCKDB_DDS_COMMON_INCLUDE_DIR}/Host)
  target_link_libraries(dds_parquet_loader PRIVATE ${DUCKDB_DDS_POSIX_LIBS} ${DUCKDB_SYSTEM_LIBS})
endif()
