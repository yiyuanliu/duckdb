#include "catch.hpp"

#include "arrow/arrow_test_helper.hpp"
#include "duckdb/common/types/arrow_aux_data.hpp"
#include "duckdb/function/table/arrow/arrow_duck_schema.hpp"
#include "duckdb/function/table/arrow/arrow_type_info.hpp"

#include <cstring>

using namespace duckdb;

// Regression test for https://github.com/duckdb/duckdb/issues/20606
//
// When Arrow string data is scanned via zero-copy, string_t values point
// directly into Arrow buffers. Buffer ownership is stored in
// vector.buffer->aux_data (as ArrowAuxiliaryData). COALESCE/CASE operators
// create temporary vectors, copy string_t via FillSwitch, and call
// StringVector::AddHeapReference — which previously only propagated
// vector.auxiliary, NOT vector.buffer->aux_data. When the temporary was
// destroyed, the Arrow buffer was freed, leaving dangling pointers.
//
// This test constructs a real Arrow string array (validity + offsets + data
// buffers), converts it to a DuckDB Vector using the production Arrow
// conversion code (ColumnArrowToDuckDB / SetVectorString), then verifies
// that the data survives ownership transfer through AddHeapReference.
// On destruction, the Arrow release callback poisons the buffer with 'X',
// making any use-after-free deterministically visible.

//===--------------------------------------------------------------------===//
// Helpers to build a real Arrow utf8 string array
//===--------------------------------------------------------------------===//

// Called when the ArrowArray is released. Overwrites the string data buffer
// with 'X' characters so that any dangling string_t reads "XXXXX..." instead
// of silently returning stale-but-correct data.
static void PoisonReleaseCallback(ArrowArray *array) {
	// buffers[2] is the string data buffer; buffers[1] is offsets
	auto *data_buf = const_cast<char *>(static_cast<const char *>(array->buffers[2]));
	auto *offsets = static_cast<const uint32_t *>(array->buffers[1]);
	auto total_len = offsets[array->length];
	memset(data_buf, 'X', total_len);

	// Free all buffers
	free(const_cast<void *>(array->buffers[1])); // offsets
	free(const_cast<void *>(array->buffers[2])); // string data
	free(const_cast<void *>(array->buffers[0])); // validity (may be null)
	delete[] array->buffers;
	array->release = nullptr;
}

// Build a real Arrow utf8 string array from a vector of C++ strings.
// Layout (Arrow format):
//   buffers[0] = validity bitmap (nullptr if no nulls)
//   buffers[1] = int32 offsets array (length + 1 entries)
//   buffers[2] = char data buffer (concatenated string bytes)
static ArrowArray MakeArrowStringArray(const vector<string> &strings) {
	auto n = strings.size();

	// Build offsets and concatenated data
	auto *offsets = static_cast<uint32_t *>(malloc((n + 1) * sizeof(uint32_t)));
	string all_data;
	offsets[0] = 0;
	for (idx_t i = 0; i < n; i++) {
		all_data += strings[i];
		offsets[i + 1] = static_cast<uint32_t>(all_data.size());
	}

	auto *data_buf = static_cast<char *>(malloc(all_data.size()));
	memcpy(data_buf, all_data.data(), all_data.size());

	// Build ArrowArray
	ArrowArray arr;
	memset(&arr, 0, sizeof(ArrowArray));
	arr.length = static_cast<int64_t>(n);
	arr.null_count = 0;
	arr.offset = 0;
	arr.n_buffers = 3;
	arr.buffers = new const void *[3];
	arr.buffers[0] = nullptr; // no nulls
	arr.buffers[1] = offsets;
	arr.buffers[2] = data_buf;
	arr.n_children = 0;
	arr.children = nullptr;
	arr.dictionary = nullptr;
	arr.release = PoisonReleaseCallback;
	arr.private_data = nullptr;

	return arr;
}

//===--------------------------------------------------------------------===//
// Test: real Arrow data → ColumnArrowToDuckDB → AddHeapReference
//===--------------------------------------------------------------------===//

TEST_CASE("Test arrow string ownership through COALESCE", "[arrow]") {
	DuckDB db;
	Connection con(db);

	// Generate test strings (>12 chars to avoid string_t inlining)
	const idx_t NUM_STRINGS = 100;
	vector<string> strings;
	for (idx_t i = 0; i < NUM_STRINGS; i++) {
		strings.push_back("long_arrow_string_" + to_string(i));
	}

	// Build a real Arrow utf8 string array
	ArrowArray arrow_array = MakeArrowStringArray(strings);

	// Wrap in ArrowArrayWrapper (shared_ptr) — this is exactly what arrow_scan does
	auto wrapper = make_shared_ptr<ArrowArrayWrapper>();
	wrapper->arrow_array = arrow_array;

	// Set up ArrowArrayScanState with owned_data pointing to our wrapper
	ArrowArrayScanState array_state(*con.context);
	array_state.owned_data = wrapper;

	// Describe the Arrow type as a normal (32-bit offset) utf8 string
	auto arrow_type = ArrowType(LogicalType::VARCHAR,
	                            make_uniq<ArrowStringInfo>(ArrowVariableSizeType::NORMAL));

	// Create the target vector and run the REAL Arrow-to-DuckDB conversion.
	// This calls SetVectorString which creates string_t pointing directly into
	// the Arrow data buffer (zero-copy), then stores ArrowAuxiliaryData on
	// the vector's buffer to keep the Arrow memory alive.
	Vector target(LogicalType::VARCHAR, NUM_STRINGS);

	{
		// Source vector: produced by the real Arrow conversion code
		// (simulates the temporary vector that COALESCE creates internally)
		Vector source(LogicalType::VARCHAR, NUM_STRINGS);
		ArrowToDuckDBConversion::ColumnArrowToDuckDB(
		    source, wrapper->arrow_array, 0, array_state, NUM_STRINGS, arrow_type);

		// Verify the conversion produced correct zero-copy strings
		auto source_data = FlatVector::GetData<string_t>(source);
		for (idx_t i = 0; i < NUM_STRINGS; i++) {
			REQUIRE(source_data[i].GetString() == strings[i]);
		}

		// Copy string_t values to target (this is what FillSwitch does)
		auto target_data = FlatVector::GetData<string_t>(target);
		auto src_data = FlatVector::GetData<string_t>(source);
		for (idx_t i = 0; i < NUM_STRINGS; i++) {
			target_data[i] = src_data[i];
		}

		// Transfer ownership — this is the critical call that the fix addresses
		StringVector::AddHeapReference(target, source);

		// Drop the wrapper and scan state references — now only the vectors
		// hold ownership of the Arrow data (matching the UDF scenario where
		// there is no scan state keeping extra references)
		wrapper.reset();
		array_state.owned_data.reset();

		// Source vector destroyed here (end of scope).
		// Without the fix: source's buffer->aux_data held the last reference
		// to ArrowArrayWrapper → PoisonReleaseCallback fires → data buffer
		// overwritten with 'X'
		// With the fix: target also holds a reference via AddHeapReference →
		// ArrowArrayWrapper stays alive → data intact
	}

	// Read strings from target — if the bug is present, we get "XXXXXXXXX..."
	auto target_data = FlatVector::GetData<string_t>(target);
	idx_t corrupted = 0;
	for (idx_t i = 0; i < NUM_STRINGS; i++) {
		if (target_data[i].GetString() != strings[i]) {
			corrupted++;
		}
	}
	if (corrupted > 0) {
		// Show first corruption for diagnostics
		WARN("First corrupted string: got '" + target_data[0].GetString() +
		     "' expected '" + strings[0] + "'");
	}
	REQUIRE(corrupted == 0);
}
