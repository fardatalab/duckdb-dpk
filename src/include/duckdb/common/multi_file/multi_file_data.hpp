//===----------------------------------------------------------------------===//
//                         DuckDB
//
// duckdb/common/multi_file/multi_file_data.hpp
//
//
//===----------------------------------------------------------------------===//

#pragma once

#include "duckdb/function/table_function.hpp"
#include "duckdb/function/copy_function.hpp"
#include "duckdb/common/exception/conversion_exception.hpp"
#include "duckdb/parser/expression/constant_expression.hpp"
#include "duckdb/common/open_file_info.hpp"
#include <numeric>
#include <mutex>

namespace duckdb {

enum class MultiFileFileState : uint8_t { UNOPENED, OPENING, OPEN, SKIPPED, CLOSED };

class DeleteFilter {
public:
	virtual ~DeleteFilter() = default;

public:
	virtual idx_t Filter(row_t start_row_index, idx_t count, SelectionVector &result_sel) = 0;
};

//! DPKRegexFilter: a DeleteFilter implementation that selects rows matching
//! pre-computed global row positions (e.g. from an external regex framework).
//! The matching_rows vector must be sorted in ascending order.
//! Each scan thread should have its own instance (cursor is not thread-safe).
class DPKRegexFilter : public DeleteFilter {
public:
	explicit DPKRegexFilter(vector<idx_t> matching_rows_p)
	    : matching_rows(std::move(matching_rows_p)), cursor(0) {
	}

	idx_t Filter(row_t start_row_index, idx_t count, SelectionVector &result_sel) override {
		idx_t result_count = 0;
		row_t end = start_row_index + UnsafeNumericCast<row_t>(count);

		// Advance cursor past rows before this chunk
		while (cursor < matching_rows.size() && UnsafeNumericCast<row_t>(matching_rows[cursor]) < start_row_index) {
			cursor++;
		}

		// Collect matches within this chunk
		while (cursor < matching_rows.size() && UnsafeNumericCast<row_t>(matching_rows[cursor]) < end) {
			idx_t local_idx = matching_rows[cursor] - UnsafeNumericCast<idx_t>(start_row_index);
			result_sel.set_index(result_count++, local_idx);
			cursor++;
		}
		return result_count;
	}

	//! Get the matching rows vector (for cloning into per-scan-state instances)
	const vector<idx_t> &GetMatchingRows() const {
		return matching_rows;
	}

private:
	vector<idx_t> matching_rows;
	idx_t cursor;
};

//! DPKFilterRegistry: global registry mapping file paths to pre-computed matching row positions.
//! Used to inject DPK hardware-accelerated regex results into the Parquet scan pipeline.
//! Usage:
//!   DPKFilterRegistry::Register("/path/to/file.parquet", matching_positions);
//!   // ... run query with LIKE predicate removed ...
//!   DPKFilterRegistry::Unregister("/path/to/file.parquet");
class DPKFilterRegistry {
public:
	static void Register(const string &file_path, shared_ptr<vector<idx_t>> matching_rows) {
		lock_guard<mutex> lock(GetMutex());
		GetRegistry()[file_path] = std::move(matching_rows);
	}

	static void Unregister(const string &file_path) {
		lock_guard<mutex> lock(GetMutex());
		GetRegistry().erase(file_path);
	}

	static shared_ptr<vector<idx_t>> Lookup(const string &file_path) {
		lock_guard<mutex> lock(GetMutex());
		auto &registry = GetRegistry();
		auto it = registry.find(file_path);
		if (it != registry.end()) {
			return it->second;
		}
		return nullptr;
	}

	static void Clear() {
		lock_guard<mutex> lock(GetMutex());
		GetRegistry().clear();
	}

private:
	static unordered_map<string, shared_ptr<vector<idx_t>>> &GetRegistry() {
		static unordered_map<string, shared_ptr<vector<idx_t>>> registry;
		return registry;
	}
	static mutex &GetMutex() {
		static mutex mtx;
		return mtx;
	}
};

struct HivePartitioningIndex {
	HivePartitioningIndex(string value, idx_t index);

	string value;
	idx_t index;

	DUCKDB_API void Serialize(Serializer &serializer) const;
	DUCKDB_API static HivePartitioningIndex Deserialize(Deserializer &deserializer);
};

struct MultiFileColumnDefinition {
public:
	MultiFileColumnDefinition(const string &name, const LogicalType &type) : name(name), type(type) {
	}

	MultiFileColumnDefinition(const MultiFileColumnDefinition &other)
	    : name(other.name), type(other.type), children(other.children),
	      default_expression(other.default_expression ? other.default_expression->Copy() : nullptr),
	      identifier(other.identifier) {
	}

	MultiFileColumnDefinition &operator=(const MultiFileColumnDefinition &other) {
		if (this != &other) {
			name = other.name;
			type = other.type;
			children = other.children;
			default_expression = other.default_expression ? other.default_expression->Copy() : nullptr;
			identifier = other.identifier;
		}
		return *this;
	}

public:
	static MultiFileColumnDefinition CreateFromNameAndType(const string &name, const LogicalType &type) {
		MultiFileColumnDefinition result(name, type);
		if (type.id() == LogicalTypeId::STRUCT) {
			// recursively create for children
			for (auto &child_entry : StructType::GetChildTypes(type)) {
				result.children.push_back(CreateFromNameAndType(child_entry.first, child_entry.second));
			}
		}
		return result;
	}

	static vector<MultiFileColumnDefinition> ColumnsFromNamesAndTypes(const vector<string> &names,
	                                                                  const vector<LogicalType> &types) {
		vector<MultiFileColumnDefinition> columns;
		D_ASSERT(names.size() == types.size());
		for (idx_t i = 0; i < names.size(); i++) {
			auto &name = names[i];
			auto &type = types[i];
			columns.push_back(CreateFromNameAndType(name, type));
		}
		return columns;
	}

	static void ExtractNamesAndTypes(const vector<MultiFileColumnDefinition> &columns, vector<string> &names,
	                                 vector<LogicalType> &types) {
		D_ASSERT(names.empty());
		D_ASSERT(types.empty());
		for (auto &column : columns) {
			names.push_back(column.name);
			types.push_back(column.type);
		}
	}

	int32_t GetIdentifierFieldId() const {
		D_ASSERT(!identifier.IsNull());
		D_ASSERT(identifier.type().id() == LogicalTypeId::INTEGER);
		return identifier.GetValue<int32_t>();
	}

	string GetIdentifierName() const {
		if (identifier.IsNull()) {
			// No identifier was provided, assume the name as the identifier
			return name;
		}
		D_ASSERT(identifier.type().id() == LogicalTypeId::VARCHAR);
		return identifier.GetValue<string>();
	}

	Value GetDefaultValue() const {
		D_ASSERT(default_expression);
		if (default_expression->type != ExpressionType::VALUE_CONSTANT) {
			throw NotImplementedException("Default expression that isn't constant is not supported yet");
		}
		auto &constant_expr = default_expression->Cast<ConstantExpression>();
		return constant_expr.value;
	}

public:
	string name;
	LogicalType type;
	vector<MultiFileColumnDefinition> children;
	unique_ptr<ParsedExpression> default_expression;

	//! Either the field_id or the name to map on
	Value identifier;
};

//! fwd declare
struct MultiFileCastMap;
struct MultiFileFilterEntry;

struct MultiFileLocalColumnId {
	friend struct MultiFileCastMap;

public:
	explicit MultiFileLocalColumnId(column_t column_id) : column_id(column_id) {
	}

public:
	operator idx_t() { // NOLINT: allow implicit conversion
		return column_id;
	}
	idx_t GetId() const {
		return column_id;
	}

private:
	column_t column_id;
};

//! fwd declare
template <class T>
struct MultiFileLocalColumnIds;
struct MultiFileGlobalColumnIds;
struct MultiFileColumnMapping;
struct MultiFileFilterMap;
struct MultiFileConstantMap;

struct MultiFileLocalIndex {
	//! these are allowed to access the index
	template <class T>
	friend struct MultiFileLocalColumnIds;
	friend struct MultiFileColumnMapping;
	friend struct MultiFileFilterEntry;

public:
	explicit MultiFileLocalIndex(idx_t index) : index(index) {
	}

public:
	operator idx_t() { // NOLINT: allow implicit conversion
		return index;
	}
	idx_t GetIndex() const {
		return index;
	}

private:
	idx_t index;
};

struct MultiFileGlobalIndex {
	friend struct MultiFileGlobalColumnIds;
	friend struct MultiFileFilterMap;

public:
	explicit MultiFileGlobalIndex(idx_t index) : index(index) {
	}

public:
	operator idx_t() { // NOLINT: allow implicit conversion
		return index;
	}
	idx_t GetIndex() const {
		return index;
	}

private:
	idx_t index;
};

struct MultiFileConstantEntry {
	MultiFileConstantEntry(MultiFileGlobalIndex column_idx, Value value_p)
	    : column_idx(column_idx), value(std::move(value_p)) {
	}
	//! The (global) column idx to apply the constant value to
	MultiFileGlobalIndex column_idx;
	//! The constant value
	Value value;
};

//! index used to access the constant map
struct MultiFileConstantMapIndex {
	friend struct MultiFileConstantMap;
	friend struct MultiFileFilterEntry;

public:
	explicit MultiFileConstantMapIndex(idx_t index) : index(index) {
	}

public:
private:
	idx_t index;
};

struct MultiFileFilterEntry {
public:
	MultiFileFilterEntry() : is_set(false), is_constant(false), index(DConstants::INVALID_INDEX) {
	}

public:
	void Set(MultiFileConstantMapIndex constant_index) {
		is_set = true;
		is_constant = true;
		index = constant_index.index;
	}
	void Set(MultiFileLocalIndex local_index) {
		is_set = true;
		is_constant = false;
		index = local_index.index;
	}

	bool IsSet() const {
		return is_set;
	}
	bool IsConstant() const {
		D_ASSERT(is_set);
		return is_constant;
	}

	MultiFileConstantMapIndex GetConstantIndex() const {
		D_ASSERT(is_set);
		D_ASSERT(is_constant);
		return MultiFileConstantMapIndex(index);
	}
	MultiFileLocalIndex GetLocalIndex() const {
		D_ASSERT(is_set);
		D_ASSERT(!is_constant);
		return MultiFileLocalIndex(index);
	}

private:
	bool is_set;
	bool is_constant;
	//! ConstantMapIndex if 'is_constant' else this is a LocalIndex
	idx_t index;
};

template <class T>
struct MultiFileLocalColumnIds {
public:
	void push_back(T column_id) { // NOLINT: matching name of std
		column_ids.push_back(column_id);
	}
	template <typename... Args>
	void emplace_back(Args &&...args) {
		column_ids.emplace_back(std::forward<Args>(args)...);
	}
	const T &operator[](MultiFileLocalIndex index) {
		return column_ids[index.index];
	}
	bool empty() const { // NOLINT: matching name of std
		return column_ids.empty();
	}
	idx_t size() const { // NOLINT: matching name of std
		return column_ids.size();
	}

private:
	vector<T> column_ids;
};

struct MultiFileFilterMap {
public:
	void push_back(const MultiFileFilterEntry &filter_entry) { // NOLINT: matching name of std
		filter_map.push_back(filter_entry);
	}
	MultiFileFilterEntry &operator[](MultiFileGlobalIndex global_index) {
		return filter_map[global_index.index];
	}
	void resize(idx_t size) { // NOLINT: matching name of std
		filter_map.resize(size);
	}

private:
	vector<MultiFileFilterEntry> filter_map;
};

struct MultiFileConstantMap {
public:
	using iterator = vector<MultiFileConstantEntry>::iterator;
	using const_iterator = vector<MultiFileConstantEntry>::const_iterator;

public:
	template <typename... Args>
	void Add(Args &&...args) {
		constant_map.emplace_back(std::forward<Args>(args)...);
	}
	const MultiFileConstantEntry &operator[](MultiFileConstantMapIndex constant_index) {
		return constant_map[constant_index.index];
	}
	idx_t size() const { // NOLINT: matching name of std
		return constant_map.size();
	}
	// Iterator support
	iterator begin() {
		return constant_map.begin();
	}
	iterator end() {
		return constant_map.end();
	}
	const_iterator begin() const {
		return constant_map.begin();
	}
	const_iterator end() const {
		return constant_map.end();
	}

private:
	vector<MultiFileConstantEntry> constant_map;
};

} // namespace duckdb
