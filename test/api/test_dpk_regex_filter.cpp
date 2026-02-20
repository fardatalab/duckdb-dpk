#include "catch.hpp"
#include "test_helpers.hpp"
#include "tpch_extension.hpp"
#include "parquet_reader.hpp"
#include "duckdb/common/multi_file/multi_file_data.hpp"
#include "duckdb/common/types/data_chunk.hpp"

#include <chrono>
#include <iostream>

using namespace duckdb;

// Helper: scan a parquet file using ParquetReader directly and return total row count
static idx_t ScanParquetWithFilter(ClientContext &context, const string &parquet_path,
                                   unique_ptr<DeleteFilter> dpk_filter) {
	ParquetOptions parquet_options;
	auto reader = make_uniq<ParquetReader>(context, OpenFileInfo(parquet_path), parquet_options);

	if (dpk_filter) {
		reader->dpk_regex_filter = std::move(dpk_filter);
	}

	// Set up column_ids and column_indexes for all columns
	for (idx_t i = 0; i < reader->GetColumns().size(); i++) {
		reader->column_ids.push_back(MultiFileLocalColumnId(i));
		reader->column_indexes.push_back(ColumnIndex(i));
	}

	// Prepare row groups to read: all of them
	duckdb::vector<idx_t> groups_to_read;
	for (idx_t i = 0; i < reader->NumRowGroups(); i++) {
		groups_to_read.push_back(i);
	}

	ParquetReaderScanState scan_state;
	reader->InitializeScan(context, scan_state, groups_to_read);

	duckdb::vector<LogicalType> types;
	for (auto &col : reader->GetColumns()) {
		types.push_back(col.type);
	}
	DataChunk result;
	result.Initialize(Allocator::DefaultAllocator(), types);

	idx_t total_rows = 0;
	while (true) {
		result.Reset();
		reader->Scan(context, scan_state, result);
		if (result.size() == 0) {
			break;
		}
		total_rows += result.size();
	}
	return total_rows;
}

TEST_CASE("Test DPKRegexFilter with TPCH Q13 predicate", "[dpk]") {
	DuckDB db(nullptr);
	Connection con(db);

	if (!db.ExtensionIsLoaded("tpch")) {
		return;
	}

	// Step 1: Generate TPCH data at SF=0.01
	auto res = con.Query("CALL dbgen(sf=0.01)");
	REQUIRE(!res->HasError());

	// Step 2: Export orders table to parquet
	string parquet_path = TestCreatePath("dpk_orders.parquet");
	res = con.Query("COPY orders TO '" + parquet_path + "' (FORMAT PARQUET)");
	REQUIRE(!res->HasError());

	auto &context = *con.context;

	// Step 3: Get total row count and expected filtered count
	auto total_result = con.Query("SELECT COUNT(*) FROM read_parquet('" + parquet_path + "')");
	REQUIRE(!total_result->HasError());
	auto total_rows = total_result->Fetch()->GetValue(0, 0).GetValue<int64_t>();

	auto expected_result = con.Query("SELECT COUNT(*) FROM read_parquet('" + parquet_path +
	                                 "') WHERE o_comment NOT LIKE '%special%requests%'");
	REQUIRE(!expected_result->HasError());
	auto expected_count = expected_result->Fetch()->GetValue(0, 0).GetValue<int64_t>();

	// The LIKE predicate must actually filter out some rows, otherwise the test is meaningless
	REQUIRE(expected_count < total_rows);

	// Step 4: Scan WITHOUT the DPKRegexFilter — should return all rows
	auto unfiltered_count = ScanParquetWithFilter(context, parquet_path, nullptr);
	REQUIRE(unfiltered_count == (idx_t)total_rows);

	// Step 5: Compute the global row positions of matching rows
	auto positions_result =
	    con.Query("SELECT pos FROM ("
	              "  SELECT row_number() OVER () - 1 AS pos, "
	              "         o_comment NOT LIKE '%special%requests%' AS matches "
	              "  FROM read_parquet('" +
	              parquet_path +
	              "')"
	              ") WHERE matches ORDER BY pos");
	REQUIRE(!positions_result->HasError());

	duckdb::vector<idx_t> matching_positions;
	auto chunk = positions_result->Fetch();
	while (chunk && chunk->size() > 0) {
		for (idx_t i = 0; i < chunk->size(); i++) {
			matching_positions.push_back(chunk->GetValue(0, i).GetValue<idx_t>());
		}
		chunk = positions_result->Fetch();
	}
	REQUIRE(matching_positions.size() == (idx_t)expected_count);

	// Step 6: Scan WITH the DPKRegexFilter — should return only matching rows
	auto filtered_count =
	    ScanParquetWithFilter(context, parquet_path, make_uniq<DPKRegexFilter>(matching_positions));

	// Step 7: Verify
	// The filter must have reduced the row count
	REQUIRE(filtered_count < unfiltered_count);
	// And it must match the expected count exactly
	REQUIRE(filtered_count == (idx_t)expected_count);
}

TEST_CASE("Test DPKRegexFilter through full TPCH Q13 pipeline", "[dpk]") {
	DuckDB db(nullptr);
	Connection con(db);

	if (!db.ExtensionIsLoaded("tpch")) {
		return;
	}

	// Step 1: Generate TPCH data at SF=0.01
	auto res = con.Query("CALL dbgen(sf=0.01)");
	REQUIRE(!res->HasError());

	// Step 2: Export tables to parquet
	string orders_path = TestCreatePath("dpk_q13_orders.parquet");
	string customer_path = TestCreatePath("dpk_q13_customer.parquet");
	res = con.Query("COPY orders TO '" + orders_path + "' (FORMAT PARQUET)");
	REQUIRE(!res->HasError());
	res = con.Query("COPY customer TO '" + customer_path + "' (FORMAT PARQUET)");
	REQUIRE(!res->HasError());

	// Step 3: Get the expected Q13 result using standard DuckDB (with LIKE evaluated normally)
	string q13_with_like =
	    "SELECT c_count, count(*) AS custdist FROM ("
	    "  SELECT c_custkey, count(o_orderkey) AS c_count"
	    "  FROM read_parquet('" + customer_path + "')"
	    "  LEFT OUTER JOIN read_parquet('" + orders_path + "')"
	    "    ON c_custkey = o_custkey"
	    "    AND o_comment NOT LIKE '%special%requests%'"
	    "  GROUP BY c_custkey"
	    ") AS c_orders"
	    " GROUP BY c_count ORDER BY custdist DESC, c_count DESC";

	auto expected = con.Query(q13_with_like);
	REQUIRE(!expected->HasError());

	// Collect expected results
	duckdb::vector<std::pair<int64_t, int64_t>> expected_rows;
	auto echunk = expected->Fetch();
	while (echunk && echunk->size() > 0) {
		for (idx_t i = 0; i < echunk->size(); i++) {
			auto c_count = echunk->GetValue(0, i).GetValue<int64_t>();
			auto custdist = echunk->GetValue(1, i).GetValue<int64_t>();
			expected_rows.push_back({c_count, custdist});
		}
		echunk = expected->Fetch();
	}
	REQUIRE(!expected_rows.empty());

	// Step 4: Pre-compute matching row positions for the LIKE predicate
	auto positions_result =
	    con.Query("SELECT pos FROM ("
	              "  SELECT row_number() OVER () - 1 AS pos, "
	              "         o_comment NOT LIKE '%special%requests%' AS matches "
	              "  FROM read_parquet('" + orders_path + "')"
	              ") WHERE matches ORDER BY pos");
	REQUIRE(!positions_result->HasError());

	auto matching_rows = make_shared_ptr<duckdb::vector<idx_t>>();
	auto chunk = positions_result->Fetch();
	while (chunk && chunk->size() > 0) {
		for (idx_t i = 0; i < chunk->size(); i++) {
			matching_rows->push_back(chunk->GetValue(0, i).GetValue<idx_t>());
		}
		chunk = positions_result->Fetch();
	}
	REQUIRE(!matching_rows->empty());

	// Step 5: Register the pre-computed filter in DPKFilterRegistry
	// When read_parquet() creates a ParquetReader, it will auto-detect this filter
	DPKFilterRegistry::Register(orders_path, matching_rows);

	// Step 6: Run the SAME Q13 query (with LIKE intact) — the query is NEVER modified.
	// At execution time, the DPKRegexFilter causes the scan to use pre-computed row positions
	// to skip non-matching rows, bypassing the LIKE evaluation in the pipeline.
	// Tuples are reconstructed exactly as vanilla DuckDB would produce.
	auto dpk_result = con.Query(q13_with_like);
	REQUIRE(!dpk_result->HasError());

	// Step 7: Unregister the filter
	DPKFilterRegistry::Unregister(orders_path);

	// Step 8: Compare results row by row
	duckdb::vector<std::pair<int64_t, int64_t>> dpk_rows;
	auto dchunk = dpk_result->Fetch();
	while (dchunk && dchunk->size() > 0) {
		for (idx_t i = 0; i < dchunk->size(); i++) {
			auto c_count = dchunk->GetValue(0, i).GetValue<int64_t>();
			auto custdist = dchunk->GetValue(1, i).GetValue<int64_t>();
			dpk_rows.push_back({c_count, custdist});
		}
		dchunk = dpk_result->Fetch();
	}

	REQUIRE(dpk_rows.size() == expected_rows.size());
	for (idx_t i = 0; i < expected_rows.size(); i++) {
		REQUIRE(dpk_rows[i].first == expected_rows[i].first);
		REQUIRE(dpk_rows[i].second == expected_rows[i].second);
	}
}

TEST_CASE("Benchmark DPKRegexFilter at varying selectivity", "[dpk][benchmark][.]") {
	DuckDB db(nullptr);
	Connection con(db);

	if (!db.ExtensionIsLoaded("tpch")) {
		return;
	}

	const double sf = 10.0;
	const int warmup_iters = 2;
	const int bench_iters = 5;

	auto res = con.Query("CALL dbgen(sf=" + to_string(sf) + ")");
	REQUIRE(!res->HasError());

	string parquet_path = TestCreatePath("dpk_bench_orders.parquet");
	res = con.Query("COPY orders TO '" + parquet_path + "' (FORMAT PARQUET)");
	REQUIRE(!res->HasError());

	auto total_result = con.Query("SELECT COUNT(*) FROM read_parquet('" + parquet_path + "')");
	auto total_rows = total_result->Fetch()->GetValue(0, 0).GetValue<idx_t>();

	// Baseline: full scan via SQL (multi-threaded)
	string full_scan_query = "SELECT COUNT(*) FROM read_parquet('" + parquet_path + "')";
	for (int i = 0; i < warmup_iters; i++) {
		con.Query(full_scan_query);
	}
	auto start = std::chrono::high_resolution_clock::now();
	for (int i = 0; i < bench_iters; i++) {
		con.Query(full_scan_query);
	}
	auto end = std::chrono::high_resolution_clock::now();
	double full_scan_ms =
	    std::chrono::duration_cast<std::chrono::microseconds>(end - start).count() / 1000.0 / bench_iters;

	std::cerr << "\n=== DPKRegexFilter Full Pipeline Benchmark (SF=" << sf << ", " << total_rows << " rows) ===" << std::endl;
	std::cerr << "Full scan COUNT(*) baseline: " << full_scan_ms << " ms" << std::endl;
	std::cerr << std::endl;

	struct PredicateTest {
		string predicate;
		string label;
	};
	duckdb::vector<PredicateTest> predicates = {
	    {"o_comment NOT LIKE '%special%requests%'", "NOT LIKE %special%requests% (Q13)"},
	    {"o_comment LIKE '%special%'",              "LIKE %special%"},
	    {"o_comment LIKE '%special%requests%'",     "LIKE %special%requests%"},
	    {"o_comment LIKE '%xyz%'",                  "LIKE %xyz%"},
	    {"o_comment LIKE 'a%'",                     "LIKE a%"},
	};

	std::cerr << "| Predicate                          | Selectivity | LIKE ms  | DPK ms  | Speedup |" << std::endl;
	std::cerr << "|------------------------------------|-------------|----------|---------|---------|" << std::endl;

	for (auto &test : predicates) {
		// Compute matching positions
		auto positions_result =
		    con.Query("SELECT pos FROM ("
		              "  SELECT row_number() OVER () - 1 AS pos, "
		              "         (" + test.predicate + ") AS matches "
		              "  FROM read_parquet('" + parquet_path + "')"
		              ") WHERE matches ORDER BY pos");
		REQUIRE(!positions_result->HasError());

		auto matching_positions = make_shared_ptr<duckdb::vector<idx_t>>();
		auto chunk = positions_result->Fetch();
		while (chunk && chunk->size() > 0) {
			for (idx_t i = 0; i < chunk->size(); i++) {
				matching_positions->push_back(chunk->GetValue(0, i).GetValue<idx_t>());
			}
			chunk = positions_result->Fetch();
		}

		double selectivity = 100.0 * matching_positions->size() / total_rows;

		// Benchmark regular LIKE query through full SQL pipeline (multi-threaded)
		string like_query = "SELECT COUNT(*) FROM read_parquet('" + parquet_path +
		                    "') WHERE " + test.predicate;
		for (int i = 0; i < warmup_iters; i++) {
			con.Query(like_query);
		}
		start = std::chrono::high_resolution_clock::now();
		for (int i = 0; i < bench_iters; i++) {
			con.Query(like_query);
		}
		end = std::chrono::high_resolution_clock::now();
		double like_ms =
		    std::chrono::duration_cast<std::chrono::microseconds>(end - start).count() / 1000.0 / bench_iters;

		// Benchmark DPK path through full SQL pipeline (multi-threaded) via DPKFilterRegistry
		// The query is IDENTICAL (LIKE stays); DPK filter bypasses LIKE evaluation at scan time
		string dpk_query = like_query;
		for (int i = 0; i < warmup_iters; i++) {
			DPKFilterRegistry::Register(parquet_path, matching_positions);
			con.Query(dpk_query);
			DPKFilterRegistry::Unregister(parquet_path);
		}
		start = std::chrono::high_resolution_clock::now();
		for (int i = 0; i < bench_iters; i++) {
			DPKFilterRegistry::Register(parquet_path, matching_positions);
			con.Query(dpk_query);
			DPKFilterRegistry::Unregister(parquet_path);
		}
		end = std::chrono::high_resolution_clock::now();
		double dpk_ms =
		    std::chrono::duration_cast<std::chrono::microseconds>(end - start).count() / 1000.0 / bench_iters;

		char buf[256];
		snprintf(buf, sizeof(buf), "| %-34s | %9.2f%% | %6.1f   | %6.1f   | %5.2fx  |",
		         test.label.c_str(), selectivity, like_ms, dpk_ms, like_ms / dpk_ms);
		std::cerr << buf << std::endl;
	}
}
