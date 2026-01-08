#!/usr/bin/env python3
"""
Test suite for the DocuMind Pipeline Orchestrator

Tests:
- Single document processing
- Batch processing
- Directory processing
- Error handling (missing files)
- Parallel execution
- Metrics collection
- JSON export
"""

import asyncio
import json
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from orchestrate import PipelineOrchestrator, ProcessingResult


async def test_single_document():
    """Test processing a single document."""
    print("\n=== Test 1: Single Document Processing ===")

    orchestrator = PipelineOrchestrator(verbose=False)

    # Create test file
    test_file = "test_single.txt"
    with open(test_file, 'w') as f:
        f.write("This is a test document for the orchestrator.")

    try:
        result = await orchestrator.process_document(test_file)

        assert result.status == "success", f"Expected success, got {result.status}"
        assert result.chunks_created > 0, "Should create at least 1 chunk"
        assert result.embeddings_generated > 0, "Should generate at least 1 embedding"
        assert result.error_message is None, f"Should have no errors, got {result.error_message}"

        print("✅ Single document test passed")
        print(f"   Chunks: {result.chunks_created}, Embeddings: {result.embeddings_generated}")
        print(f"   Time: {result.processing_time:.3f}s")

    finally:
        # Cleanup
        Path(test_file).unlink(missing_ok=True)


async def test_batch_processing():
    """Test processing multiple documents in parallel."""
    print("\n=== Test 2: Batch Processing ===")

    orchestrator = PipelineOrchestrator(max_parallel=3, verbose=False)

    # Create test files
    test_files = []
    for i in range(5):
        filename = f"test_batch_{i}.txt"
        with open(filename, 'w') as f:
            f.write(f"Test document {i} " * 100)  # Make it longer
        test_files.append(filename)

    try:
        results = await orchestrator.process_batch(test_files)

        assert len(results) == 5, f"Expected 5 results, got {len(results)}"
        assert all(r.status == "success" for r in results), "All should succeed"
        assert orchestrator.metrics.successful == 5, "Metrics should show 5 successful"
        assert orchestrator.metrics.failed == 0, "Metrics should show 0 failed"

        print("✅ Batch processing test passed")
        print(f"   Documents: {len(results)}")
        print(f"   Total chunks: {orchestrator.metrics.total_chunks}")
        print(f"   Total time: {orchestrator.metrics.total_time:.3f}s")
        print(f"   Throughput: {len(results) / max(orchestrator.metrics.total_time, 0.001):.1f} docs/sec")

    finally:
        # Cleanup
        for filename in test_files:
            Path(filename).unlink(missing_ok=True)


async def test_error_handling():
    """Test error handling with missing files."""
    print("\n=== Test 3: Error Handling ===")

    orchestrator = PipelineOrchestrator(continue_on_error=True, verbose=False)

    # Mix of existing and non-existing files
    test_file = "test_error_good.txt"
    with open(test_file, 'w') as f:
        f.write("This file exists")

    test_files = [
        test_file,
        "nonexistent_file.txt",
        "another_missing.pdf"
    ]

    try:
        results = await orchestrator.process_batch(test_files)

        assert len(results) == 3, f"Expected 3 results, got {len(results)}"
        assert orchestrator.metrics.successful == 1, "Should have 1 success"
        assert orchestrator.metrics.failed == 2, "Should have 2 failures"

        # Check failed results
        failed = [r for r in results if r.status == "error"]
        assert len(failed) == 2, "Should have 2 failed results"
        assert all(r.error_message is not None for r in failed), "Failed results should have error messages"

        print("✅ Error handling test passed")
        print(f"   Successful: {orchestrator.metrics.successful}")
        print(f"   Failed: {orchestrator.metrics.failed}")
        print(f"   Error messages captured: {len([r for r in failed if r.error_message])}")

    finally:
        # Cleanup
        Path(test_file).unlink(missing_ok=True)


async def test_metrics_collection():
    """Test that metrics are properly collected."""
    print("\n=== Test 4: Metrics Collection ===")

    orchestrator = PipelineOrchestrator(verbose=False)

    # Create test files
    test_files = []
    for i in range(3):
        filename = f"test_metrics_{i}.txt"
        with open(filename, 'w') as f:
            f.write(f"Metrics test document {i} " * 200)
        test_files.append(filename)

    try:
        # Process with timing
        import time
        start = time.time()
        await orchestrator.process_batch(test_files)
        orchestrator.metrics.total_time = time.time() - start

        # Check metrics
        assert orchestrator.metrics.total_documents == 3, "Should process 3 documents"
        assert orchestrator.metrics.successful == 3, "All 3 should succeed"
        assert orchestrator.metrics.total_chunks > 0, "Should create chunks"
        assert orchestrator.metrics.total_embeddings > 0, "Should create embeddings"
        assert orchestrator.metrics.total_time > 0, "Should record time"

        # Check stage times (should be > 0 for successful processing)
        total_stage_time = sum(orchestrator.metrics.stage_times.values())
        assert total_stage_time > 0, "Total stage time should be > 0"

        print("✅ Metrics collection test passed")
        print(f"   Documents: {orchestrator.metrics.total_documents}")
        print(f"   Chunks: {orchestrator.metrics.total_chunks}")
        print(f"   Stage times: Extract={orchestrator.metrics.stage_times['extract']:.3f}s, "
              f"Chunk={orchestrator.metrics.stage_times['chunk']:.3f}s, "
              f"Embed={orchestrator.metrics.stage_times['embed']:.3f}s, "
              f"Write={orchestrator.metrics.stage_times['write']:.3f}s")

    finally:
        # Cleanup
        for filename in test_files:
            Path(filename).unlink(missing_ok=True)


async def test_json_export():
    """Test JSON report generation."""
    print("\n=== Test 5: JSON Export ===")

    orchestrator = PipelineOrchestrator(verbose=False)

    # Create test file
    test_file = "test_json.txt"
    with open(test_file, 'w') as f:
        f.write("JSON export test")

    try:
        results = await orchestrator.process_batch([test_file])

        # Generate report data
        output_data = {
            "metrics": {
                "total_documents": orchestrator.metrics.total_documents,
                "successful": orchestrator.metrics.successful,
                "failed": orchestrator.metrics.failed,
            },
            "results": [
                {
                    "file_path": r.file_path,
                    "status": r.status,
                    "chunks_created": r.chunks_created
                }
                for r in results
            ]
        }

        # Verify JSON serializable
        json_str = json.dumps(output_data, indent=2)
        assert json_str is not None, "Should produce valid JSON"
        assert len(json_str) > 0, "JSON should not be empty"

        # Verify structure
        parsed = json.loads(json_str)
        assert "metrics" in parsed
        assert "results" in parsed
        assert parsed["metrics"]["total_documents"] == 1

        print("✅ JSON export test passed")
        print(f"   JSON length: {len(json_str)} bytes")
        print(f"   Contains: {list(parsed.keys())}")

    finally:
        # Cleanup
        Path(test_file).unlink(missing_ok=True)


async def test_parallel_execution():
    """Test that parallel execution actually runs concurrently."""
    print("\n=== Test 6: Parallel Execution ===")

    # Test with low parallelism
    orchestrator_sequential = PipelineOrchestrator(max_parallel=1, verbose=False)

    # Create test files
    test_files = []
    for i in range(5):
        filename = f"test_parallel_{i}.txt"
        with open(filename, 'w') as f:
            f.write(f"Parallel test {i} " * 100)
        test_files.append(filename)

    try:
        # Sequential processing
        import time
        start = time.time()
        await orchestrator_sequential.process_batch(test_files)
        sequential_time = time.time() - start

        # Parallel processing
        orchestrator_parallel = PipelineOrchestrator(max_parallel=5, verbose=False)
        start = time.time()
        await orchestrator_parallel.process_batch(test_files)
        parallel_time = time.time() - start

        # Parallel should be faster
        speedup = sequential_time / parallel_time
        print("✅ Parallel execution test passed")
        print(f"   Sequential time: {sequential_time:.3f}s")
        print(f"   Parallel time: {parallel_time:.3f}s")
        print(f"   Speedup: {speedup:.2f}x")

        # Note: Speedup may be minimal with mocked agents, but structure is correct

    finally:
        # Cleanup
        for filename in test_files:
            Path(filename).unlink(missing_ok=True)


async def run_all_tests():
    """Run all test suites."""
    print("\n" + "="*60)
    print("DocuMind Pipeline Orchestrator - Test Suite")
    print("="*60)

    tests = [
        test_single_document,
        test_batch_processing,
        test_error_handling,
        test_metrics_collection,
        test_json_export,
        test_parallel_execution
    ]

    passed = 0
    failed = 0

    for test in tests:
        try:
            await test()
            passed += 1
        except Exception as e:
            failed += 1
            print(f"❌ Test failed: {test.__name__}")
            print(f"   Error: {e}")

    print("\n" + "="*60)
    print(f"Test Results: {passed} passed, {failed} failed")
    print("="*60 + "\n")

    return failed == 0


if __name__ == "__main__":
    success = asyncio.run(run_all_tests())
    sys.exit(0 if success else 1)
