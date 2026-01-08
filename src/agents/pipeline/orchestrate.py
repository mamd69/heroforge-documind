#!/usr/bin/env python3
"""
DocuMind Pipeline Orchestrator

Coordinates the 4-stage document processing pipeline:
Extract → Chunk → Embed → Write

Features:
- Parallel processing using asyncio
- Error handling with continue-on-error
- Progress indicators
- Comprehensive metrics and reports
- CLI support for directory or file list input

Usage:
    python orchestrate.py demo-docs/
    python orchestrate.py file1.md file2.pdf
    python orchestrate.py -d demo-docs/ --max-parallel 20
"""

import asyncio
import argparse
import json
import sys
import time
import shutil
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any, Callable
import traceback


def get_terminal_width() -> int:
    """Get terminal width, defaulting to 80 if unavailable."""
    try:
        return shutil.get_terminal_size().columns
    except Exception:
        return 80


class ProgressTracker:
    """
    Real-time progress tracking with progress bar and ETA calculation.

    Features:
    - Visual progress bar with percentage
    - Estimated time remaining based on rolling average
    - Per-document status updates
    - Thread-safe for concurrent processing
    """

    def __init__(self, total: int, bar_width: int = 40):
        """
        Initialize progress tracker.

        Args:
            total: Total number of items to process
            bar_width: Width of the progress bar in characters
        """
        self.total = total
        self.bar_width = bar_width
        self.completed = 0
        self.successful = 0
        self.failed = 0
        self.start_time = time.time()
        self.processing_times: List[float] = []
        self._lock = asyncio.Lock()
        self._last_display_time = 0
        self._min_display_interval = 0.1  # Minimum time between display updates

    async def update(self, success: bool, processing_time: float, file_name: str = ""):
        """
        Update progress after processing a document.

        Args:
            success: Whether the document was processed successfully
            processing_time: Time taken to process the document
            file_name: Name of the processed file
        """
        async with self._lock:
            self.completed += 1
            if success:
                self.successful += 1
            else:
                self.failed += 1
            self.processing_times.append(processing_time)

    def get_eta_string(self) -> str:
        """Calculate and format estimated time remaining."""
        if not self.processing_times or self.completed == 0:
            return "calculating..."

        # Use rolling average of last 10 documents for better accuracy
        recent_times = self.processing_times[-10:]
        avg_time = sum(recent_times) / len(recent_times)
        remaining = self.total - self.completed
        eta_seconds = avg_time * remaining

        if eta_seconds < 60:
            return f"{eta_seconds:.0f}s"
        elif eta_seconds < 3600:
            minutes = int(eta_seconds // 60)
            seconds = int(eta_seconds % 60)
            return f"{minutes}m {seconds}s"
        else:
            hours = int(eta_seconds // 3600)
            minutes = int((eta_seconds % 3600) // 60)
            return f"{hours}h {minutes}m"

    def get_progress_bar(self) -> str:
        """Generate a visual progress bar string."""
        if self.total == 0:
            return "[" + "=" * self.bar_width + "] 100%"

        progress = self.completed / self.total
        filled = int(self.bar_width * progress)
        empty = self.bar_width - filled

        # Use different characters for better visualization
        bar = "█" * filled + "░" * empty
        percentage = progress * 100

        return f"[{bar}] {percentage:5.1f}%"

    def get_status_line(self) -> str:
        """Generate a complete status line with progress bar and stats."""
        progress_bar = self.get_progress_bar()
        eta = self.get_eta_string()
        elapsed = time.time() - self.start_time

        status = (
            f"\r{progress_bar} | "
            f"{self.completed}/{self.total} docs | "
            f"✅ {self.successful} ❌ {self.failed} | "
            f"ETA: {eta} | "
            f"Elapsed: {elapsed:.1f}s"
        )

        # Pad with spaces to clear any previous longer output
        terminal_width = get_terminal_width()
        return status.ljust(terminal_width - 1)

    def display(self, force: bool = False):
        """
        Display the current progress to stdout.

        Args:
            force: Force display even if within minimum interval
        """
        current_time = time.time()
        if not force and (current_time - self._last_display_time) < self._min_display_interval:
            return

        self._last_display_time = current_time
        print(self.get_status_line(), end="", flush=True)

    def finish(self):
        """Display final progress and move to new line."""
        self.display(force=True)
        print()  # New line after progress bar


@dataclass
class ProcessingResult:
    """Result of processing a single document through the pipeline."""
    file_path: str
    status: str  # "success" or "error"
    stage_completed: str  # "extract", "chunk", "embed", "write", or "none"
    chunks_created: int = 0
    embeddings_generated: int = 0
    processing_time: float = 0.0
    error_message: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class PipelineMetrics:
    """Aggregated metrics for the entire pipeline run."""
    total_documents: int = 0
    successful: int = 0
    failed: int = 0
    total_chunks: int = 0
    total_embeddings: int = 0
    total_time: float = 0.0
    stage_times: Dict[str, float] = field(default_factory=lambda: {
        "extract": 0.0,
        "chunk": 0.0,
        "embed": 0.0,
        "write": 0.0
    })
    stage_counts: Dict[str, int] = field(default_factory=lambda: {
        "extract": 0,
        "chunk": 0,
        "embed": 0,
        "write": 0
    })
    errors_by_stage: Dict[str, int] = field(default_factory=lambda: {
        "extract": 0,
        "chunk": 0,
        "embed": 0,
        "write": 0
    })


class PipelineOrchestrator:
    """
    Coordinates the 4-stage DocuMind pipeline.

    Pipeline stages:
    1. Extract: Read file and extract text content
    2. Chunk: Split text into semantic chunks (~500 words)
    3. Embed: Generate vector embeddings using OpenAI
    4. Write: Store chunks and embeddings in Supabase

    Attributes:
        max_parallel: Maximum number of documents to process concurrently
        continue_on_error: If True, keep processing even if some documents fail
        verbose: Enable detailed logging
        metrics: Aggregated processing metrics
    """

    def __init__(
        self,
        max_parallel: int = 10,
        continue_on_error: bool = True,
        verbose: bool = True,
        show_progress_bar: bool = True
    ):
        """
        Initialize the pipeline orchestrator.

        Args:
            max_parallel: Maximum concurrent document processing (default: 10)
            continue_on_error: Continue processing if one doc fails (default: True)
            verbose: Print progress indicators (default: True)
            show_progress_bar: Show real-time progress bar (default: True)
        """
        self.max_parallel = max_parallel
        self.continue_on_error = continue_on_error
        self.verbose = verbose
        self.show_progress_bar = show_progress_bar
        self.metrics = PipelineMetrics()
        self.semaphore = asyncio.Semaphore(max_parallel)
        self.progress_tracker: Optional[ProgressTracker] = None

        # Import agents (lazy loading for faster startup)
        self._extractor = None
        self._chunker = None
        self._embedder = None
        self._writer = None

    def _get_extractor(self):
        """Lazy load the extractor agent."""
        if self._extractor is None:
            try:
                # Try relative import first (when running as package)
                from .extractor import extract_document
                self._extractor = RealExtractorAgent(extract_document)
            except ImportError:
                try:
                    # Try absolute import (when running as script)
                    from extractor import extract_document
                    self._extractor = RealExtractorAgent(extract_document)
                except ImportError:
                    # Fallback to mock if agent not implemented yet
                    self._extractor = MockExtractorAgent()
        return self._extractor

    def _get_chunker(self):
        """Lazy load the chunker agent."""
        if self._chunker is None:
            try:
                from .chunker import TextChunker
                self._chunker = RealChunkerAgent(TextChunker())
            except ImportError:
                try:
                    from chunker import TextChunker
                    self._chunker = RealChunkerAgent(TextChunker())
                except ImportError:
                    self._chunker = MockChunkerAgent()
        return self._chunker

    def _get_embedder(self):
        """Lazy load the embedder agent."""
        if self._embedder is None:
            try:
                from .embedder import EmbedderAgent
                self._embedder = RealEmbedderAgent(EmbedderAgent())
            except ImportError:
                try:
                    from embedder import EmbedderAgent
                    self._embedder = RealEmbedderAgent(EmbedderAgent())
                except ImportError:
                    self._embedder = MockEmbedderAgent()
        return self._embedder

    def _get_writer(self):
        """Lazy load the writer agent."""
        if self._writer is None:
            try:
                from .writer import write_to_database
                self._writer = RealWriterAgent(write_to_database)
            except ImportError:
                try:
                    from writer import write_to_database
                    self._writer = RealWriterAgent(write_to_database)
                except ImportError:
                    self._writer = MockWriterAgent()
        return self._writer

    async def process_document(self, file_path: str) -> ProcessingResult:
        """
        Process a single document through the 4-stage pipeline.

        Args:
            file_path: Path to the document file

        Returns:
            ProcessingResult with status, metrics, and any error info

        Raises:
            Does not raise - all errors are captured in result
        """
        result = ProcessingResult(
            file_path=file_path,
            status="error",
            stage_completed="none"
        )

        start_time = time.time()
        file_name = Path(file_path).name

        try:
            # Use semaphore to limit concurrency
            async with self.semaphore:
                if self.verbose and not self.show_progress_bar:
                    print(f"📄 Processing: {file_name}")

                # Stage 1: Extract text from file
                stage_start = time.time()
                extracted_data = await self._stage_extract(file_path)
                extract_time = time.time() - stage_start
                self.metrics.stage_times["extract"] += extract_time
                self.metrics.stage_counts["extract"] += 1
                result.stage_completed = "extract"

                # Stage 2: Chunk the extracted text
                stage_start = time.time()
                chunks = await self._stage_chunk(extracted_data)
                chunk_time = time.time() - stage_start
                self.metrics.stage_times["chunk"] += chunk_time
                self.metrics.stage_counts["chunk"] += 1
                result.stage_completed = "chunk"
                result.chunks_created = len(chunks)

                # Stage 3: Generate embeddings for chunks
                stage_start = time.time()
                embedded_chunks = await self._stage_embed(chunks)
                embed_time = time.time() - stage_start
                self.metrics.stage_times["embed"] += embed_time
                self.metrics.stage_counts["embed"] += 1
                result.stage_completed = "embed"
                result.embeddings_generated = len(embedded_chunks)

                # Stage 4: Write to database
                stage_start = time.time()
                document_id = await self._stage_write(file_path, embedded_chunks, extracted_data)
                write_time = time.time() - stage_start
                self.metrics.stage_times["write"] += write_time
                self.metrics.stage_counts["write"] += 1
                result.stage_completed = "write"

                # Success!
                result.status = "success"
                result.metadata = {
                    "document_id": document_id,
                    "file_name": file_name,
                    "file_type": extracted_data.get("file_type", "unknown"),
                    "stage_times": {
                        "extract": extract_time,
                        "chunk": chunk_time,
                        "embed": embed_time,
                        "write": write_time
                    }
                }

                if self.verbose and not self.show_progress_bar:
                    print(f"  ✅ Success: {result.chunks_created} chunks, "
                          f"{result.embeddings_generated} embeddings")

        except Exception as e:
            result.error_message = str(e)

            # Only increment error count for valid stages
            if result.stage_completed in self.metrics.errors_by_stage:
                self.metrics.errors_by_stage[result.stage_completed] += 1

            if self.verbose and not self.show_progress_bar:
                print(f"  ❌ Failed at stage '{result.stage_completed}': {str(e)[:100]}")

            if not self.continue_on_error:
                raise

        finally:
            result.processing_time = time.time() - start_time

            # Update progress tracker if available
            if self.progress_tracker:
                await self.progress_tracker.update(
                    success=(result.status == "success"),
                    processing_time=result.processing_time,
                    file_name=file_name
                )
                if self.show_progress_bar:
                    self.progress_tracker.display()

        return result

    async def _stage_extract(self, file_path: str) -> Dict[str, Any]:
        """
        Stage 1: Extract text from document.

        Args:
            file_path: Path to the file

        Returns:
            Dictionary with extracted text and metadata
        """
        extractor = self._get_extractor()
        return await extractor.extract_text(file_path)

    async def _stage_chunk(self, extracted_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Stage 2: Split text into semantic chunks.

        Args:
            extracted_data: Output from extract stage

        Returns:
            List of chunk dictionaries
        """
        chunker = self._get_chunker()
        return await chunker.chunk_text(
            text=extracted_data["text"],
            metadata=extracted_data.get("metadata", {})
        )

    async def _stage_embed(self, chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Stage 3: Generate embeddings for chunks.

        Args:
            chunks: List of text chunks

        Returns:
            List of chunks with embeddings attached
        """
        embedder = self._get_embedder()
        chunk_texts = [chunk["text"] for chunk in chunks]
        embeddings = await embedder.generate_embeddings(chunk_texts)

        # Attach embeddings to chunks
        for i, chunk in enumerate(chunks):
            chunk["embedding"] = embeddings[i]

        return chunks

    async def _stage_write(
        self,
        file_path: str,
        embedded_chunks: List[Dict[str, Any]],
        extracted_data: Dict[str, Any]
    ) -> str:
        """
        Stage 4: Write chunks and embeddings to database.

        Args:
            file_path: Original file path
            embedded_chunks: Chunks with embeddings
            extracted_data: Original extraction metadata

        Returns:
            Document ID from database
        """
        writer = self._get_writer()
        return await writer.write_chunks(
            file_path=file_path,
            chunks=embedded_chunks,
            metadata=extracted_data.get("metadata", {})
        )

    async def process_batch(self, file_paths: List[str]) -> List[ProcessingResult]:
        """
        Process multiple documents in parallel.

        Args:
            file_paths: List of file paths to process

        Returns:
            List of ProcessingResult objects
        """
        self.metrics.total_documents = len(file_paths)

        if self.verbose:
            print(f"\n🚀 Starting pipeline for {len(file_paths)} documents")
            print(f"   Max parallel: {self.max_parallel}")
            print(f"   Continue on error: {self.continue_on_error}")

            if self.show_progress_bar:
                print(f"   Progress tracking: enabled\n")
            else:
                print()

        # Initialize progress tracker
        if self.show_progress_bar:
            self.progress_tracker = ProgressTracker(total=len(file_paths))
            self.progress_tracker.display(force=True)

        # Create tasks for all documents
        tasks = [self.process_document(path) for path in file_paths]

        # Process with asyncio.gather (handles exceptions if continue_on_error=True)
        if self.continue_on_error:
            results = await asyncio.gather(*tasks, return_exceptions=False)
        else:
            results = await asyncio.gather(*tasks)

        # Finish progress bar
        if self.progress_tracker and self.show_progress_bar:
            self.progress_tracker.finish()

        # Update metrics
        for result in results:
            if result.status == "success":
                self.metrics.successful += 1
                self.metrics.total_chunks += result.chunks_created
                self.metrics.total_embeddings += result.embeddings_generated
            else:
                self.metrics.failed += 1

        return results

    async def process_directory(self, directory: str) -> List[ProcessingResult]:
        """
        Process all supported files in a directory.

        Args:
            directory: Path to directory containing documents

        Returns:
            List of ProcessingResult objects
        """
        dir_path = Path(directory)
        if not dir_path.exists() or not dir_path.is_dir():
            raise ValueError(f"Directory not found: {directory}")

        # Find all supported files (recursive search)
        supported_extensions = {".txt", ".md", ".pdf", ".docx", ".xlsx"}
        file_paths = [
            str(file)
            for ext in supported_extensions
            for file in dir_path.rglob(f"*{ext}")
        ]

        if not file_paths:
            print(f"⚠️  No supported files found in {directory}")
            return []

        return await self.process_batch(file_paths)

    def generate_report(self, results: List[ProcessingResult]) -> str:
        """
        Generate a comprehensive processing report with bottleneck analysis.

        Args:
            results: List of processing results

        Returns:
            Formatted report string
        """
        # Calculate average times
        avg_time = (
            sum(r.processing_time for r in results) / len(results)
            if results else 0
        )

        # Calculate stage breakdown with proper counts
        stage_avg = {}
        for stage in ["extract", "chunk", "embed", "write"]:
            count = self.metrics.stage_counts.get(stage, 0)
            total_time = self.metrics.stage_times.get(stage, 0.0)
            stage_avg[stage] = total_time / max(count, 1)

        # Identify bottleneck (slowest stage)
        bottleneck_stage = max(stage_avg.keys(), key=lambda s: stage_avg[s])
        bottleneck_time = stage_avg[bottleneck_stage]
        total_stage_time = sum(stage_avg.values())
        bottleneck_pct = (bottleneck_time / total_stage_time * 100) if total_stage_time > 0 else 0

        # Calculate stage percentages for visualization
        stage_pcts = {
            stage: (time / total_stage_time * 100) if total_stage_time > 0 else 0
            for stage, time in stage_avg.items()
        }

        # Build stage bar chart
        def make_bar(pct: float, width: int = 20) -> str:
            filled = int(pct / 100 * width)
            return "█" * filled + "░" * (width - filled)

        # Build report
        report = f"""
╔══════════════════════════════════════════════════════════════════════════╗
║                    DOCUMIND PIPELINE PROCESSING REPORT                    ║
╚══════════════════════════════════════════════════════════════════════════╝

┌──────────────────────────────────────────────────────────────────────────┐
│ 📊 SUMMARY                                                               │
├──────────────────────────────────────────────────────────────────────────┤
│ Total Documents:        {self.metrics.total_documents:<10}                                        │
│ ✅ Successful:          {self.metrics.successful:<10} ({self._percentage(self.metrics.successful, self.metrics.total_documents):>6})                             │
│ ❌ Failed:              {self.metrics.failed:<10} ({self._percentage(self.metrics.failed, self.metrics.total_documents):>6})                             │
└──────────────────────────────────────────────────────────────────────────┘

┌──────────────────────────────────────────────────────────────────────────┐
│ 📦 OUTPUT                                                                │
├──────────────────────────────────────────────────────────────────────────┤
│ Total Chunks Created:   {self.metrics.total_chunks:<10}                                        │
│ Total Embeddings:       {self.metrics.total_embeddings:<10}                                        │
│ Avg Chunks/Document:    {self.metrics.total_chunks / max(self.metrics.successful, 1):<10.1f}                                        │
└──────────────────────────────────────────────────────────────────────────┘

┌──────────────────────────────────────────────────────────────────────────┐
│ ⏱️  PERFORMANCE                                                           │
├──────────────────────────────────────────────────────────────────────────┤
│ Total Time:             {self.metrics.total_time:<10.2f}s                                       │
│ Avg Time/Document:      {avg_time:<10.2f}s                                       │
│ Throughput:             {self.metrics.successful / max(self.metrics.total_time, 0.001):<10.1f} docs/second                          │
└──────────────────────────────────────────────────────────────────────────┘

┌──────────────────────────────────────────────────────────────────────────┐
│ ⚙️  STAGE BREAKDOWN (Average Time per Document)                           │
├──────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│ Extract  {make_bar(stage_pcts['extract'])} {stage_avg['extract']:>6.3f}s ({stage_pcts['extract']:>5.1f}%)        │
│ Chunk    {make_bar(stage_pcts['chunk'])} {stage_avg['chunk']:>6.3f}s ({stage_pcts['chunk']:>5.1f}%)        │
│ Embed    {make_bar(stage_pcts['embed'])} {stage_avg['embed']:>6.3f}s ({stage_pcts['embed']:>5.1f}%)        │
│ Write    {make_bar(stage_pcts['write'])} {stage_avg['write']:>6.3f}s ({stage_pcts['write']:>5.1f}%)        │
│                                                                          │
└──────────────────────────────────────────────────────────────────────────┘

┌──────────────────────────────────────────────────────────────────────────┐
│ 🔍 BOTTLENECK ANALYSIS                                                   │
├──────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│ ⚠️  Slowest Stage:       {bottleneck_stage.upper():<10} ({bottleneck_time:.3f}s avg, {bottleneck_pct:.1f}% of pipeline)     │
│                                                                          │
│ Recommendations:                                                         │"""

        # Add stage-specific recommendations
        if bottleneck_stage == "extract":
            report += """
│   • Consider using faster file parsers or caching                        │
│   • Pre-process large PDF/DOCX files                                     │"""
        elif bottleneck_stage == "chunk":
            report += """
│   • Adjust chunk size for optimal performance                            │
│   • Consider parallel chunking for large documents                       │"""
        elif bottleneck_stage == "embed":
            report += """
│   • Batch embeddings API calls for efficiency                            │
│   • Consider using a local embedding model                               │
│   • Increase max_parallel for I/O-bound embedding calls                  │"""
        elif bottleneck_stage == "write":
            report += """
│   • Use batch inserts for database writes                                │
│   • Consider connection pooling                                          │
│   • Enable async database operations                                     │"""

        report += """
│                                                                          │
└──────────────────────────────────────────────────────────────────────────┘

┌──────────────────────────────────────────────────────────────────────────┐
│ ❌ ERRORS BY STAGE                                                        │
├──────────────────────────────────────────────────────────────────────────┤
│ Extract failures:       {:<10}                                        │
│ Chunk failures:         {:<10}                                        │
│ Embed failures:         {:<10}                                        │
│ Write failures:         {:<10}                                        │
└──────────────────────────────────────────────────────────────────────────┘""".format(
            self.metrics.errors_by_stage['extract'],
            self.metrics.errors_by_stage['chunk'],
            self.metrics.errors_by_stage['embed'],
            self.metrics.errors_by_stage['write']
        )

        # Add failed documents section if any
        failed_docs = [r for r in results if r.status == "error"]
        if failed_docs:
            report += f"""

┌──────────────────────────────────────────────────────────────────────────┐
│ 🔴 FAILED DOCUMENTS ({len(failed_docs)} total)                                              │
├──────────────────────────────────────────────────────────────────────────┤"""
            for r in failed_docs[:5]:  # Show first 5
                filename = Path(r.file_path).name[:30]
                error_msg = (r.error_message or "Unknown error")[:35]
                report += f"\n│ • {filename:<30} │ {r.stage_completed:<7} │ {error_msg:<35}│"
            if len(failed_docs) > 5:
                report += f"\n│   ... and {len(failed_docs) - 5} more failed documents                                    │"
            report += "\n└──────────────────────────────────────────────────────────────────────────┘"

        report += f"""

╔══════════════════════════════════════════════════════════════════════════╗
║ Generated at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S'):<58} ║
╚══════════════════════════════════════════════════════════════════════════╝
"""

        return report

    @staticmethod
    def _percentage(part: int, total: int) -> str:
        """Calculate percentage string."""
        if total == 0:
            return "0.0%"
        return f"{100 * part / total:.1f}%"


# ============================================================================
# REAL AGENTS (Wrappers for actual implementations)
# ============================================================================

class RealExtractorAgent:
    """Real extractor agent wrapping extractor.py"""

    def __init__(self, extract_func):
        self.extract_func = extract_func

    async def extract_text(self, file_path: str) -> Dict[str, Any]:
        """Extract text using real extractor."""
        result = self.extract_func(file_path)
        if not result.get("success", False):
            raise Exception(result.get("error", "Extraction failed"))
        return {
            "text": result.get("content", ""),
            "file_type": result.get("file_type", "unknown"),
            "title": result.get("title", Path(file_path).stem),
            "metadata": {
                "file_name": Path(file_path).name,
                "file_size": result.get("size", 0),
                "word_count": len(result.get("content", "").split())
            }
        }


class RealChunkerAgent:
    """Real chunker agent wrapping chunker.py"""

    def __init__(self, chunker):
        self.chunker = chunker

    async def chunk_text(self, text: str, metadata: Dict = None) -> List[Dict[str, Any]]:
        """Chunk text using real chunker."""
        doc_id = metadata.get("file_name", "doc") if metadata else "doc"
        chunks = self.chunker.chunk_text(text, doc_id)
        return [
            {
                "text": chunk.content,
                "chunk_index": chunk.chunk_index,
                "word_count": chunk.word_count,
                "metadata": metadata or {}
            }
            for chunk in chunks
        ]


class RealEmbedderAgent:
    """Real embedder agent wrapping embedder.py"""

    def __init__(self, embedder):
        self.embedder = embedder

    async def generate_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings using real OpenAI embedder."""
        chunks = [{"chunk_id": str(i), "text": text} for i, text in enumerate(texts)]
        result = self.embedder.process(chunks)
        if not result.get("success", False):
            raise Exception(result.get("error", "Embedding failed"))
        # Return embeddings in same order as input
        embeddings_dict = {e["chunk_id"]: e["vector"] for e in result.get("embeddings", [])}
        return [embeddings_dict.get(str(i), []) for i in range(len(texts))]


class RealWriterAgent:
    """Real writer agent wrapping writer.py"""

    def __init__(self, write_func):
        self.write_func = write_func

    async def write_chunks(
        self,
        file_path: str,
        chunks: List[Dict[str, Any]],
        metadata: Dict = None
    ) -> str:
        """Write chunks to Supabase using real writer."""
        # Prepare data in writer's expected format
        data = {
            "document": {
                "title": metadata.get("file_name", Path(file_path).stem) if metadata else Path(file_path).stem,
                "content": " ".join(c.get("text", "") for c in chunks[:1]),  # First chunk as preview
                "file_type": metadata.get("file_type", Path(file_path).suffix.lstrip('.')),
                "metadata": metadata or {}
            },
            "chunks": [
                {
                    "chunk_text": chunk.get("text", ""),
                    "chunk_index": chunk.get("chunk_index", i),
                    "embedding": chunk.get("embedding", []),
                    "word_count": chunk.get("word_count", 0),
                    "metadata": chunk.get("metadata", {})
                }
                for i, chunk in enumerate(chunks)
            ]
        }

        result = self.write_func(data)
        if not result.get("success", False):
            raise Exception(result.get("error", "Write failed"))
        return result.get("document_id", "")


# ============================================================================
# MOCK AGENTS (Used when actual agents are not implemented)
# ============================================================================

class MockExtractorAgent:
    """Mock extractor for testing pipeline without full implementation."""

    async def extract_text(self, file_path: str) -> Dict[str, Any]:
        """Mock text extraction."""
        await asyncio.sleep(0.1)  # Simulate I/O

        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        # Read actual file content
        try:
            with open(path, 'r', encoding='utf-8') as f:
                content = f.read()
        except UnicodeDecodeError:
            content = "[Binary content - PDF parser required]"

        return {
            "text": content[:5000],  # First 5000 chars
            "file_type": path.suffix.lstrip('.'),
            "metadata": {
                "file_name": path.name,
                "file_size": path.stat().st_size,
                "word_count": len(content.split())
            }
        }


class MockChunkerAgent:
    """Mock chunker for testing pipeline."""

    async def chunk_text(self, text: str, metadata: Dict = None) -> List[Dict[str, Any]]:
        """Mock chunking - simple split by character count."""
        await asyncio.sleep(0.05)  # Simulate processing

        chunk_size = 500  # words
        words = text.split()
        chunks = []

        for i in range(0, len(words), chunk_size):
            chunk_words = words[i:i+chunk_size]
            chunks.append({
                "text": " ".join(chunk_words),
                "chunk_index": len(chunks),
                "word_count": len(chunk_words),
                "metadata": metadata or {}
            })

        return chunks if chunks else [{"text": text, "chunk_index": 0, "word_count": len(words), "metadata": {}}]


class MockEmbedderAgent:
    """Mock embedder for testing pipeline."""

    async def generate_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Mock embedding generation - returns random vectors."""
        await asyncio.sleep(0.2)  # Simulate API call

        # Generate mock 1536-dimensional embeddings
        import random
        embeddings = []
        for text in texts:
            # Use text hash for reproducible "embeddings"
            random.seed(hash(text) % (2**32))
            embedding = [random.random() for _ in range(1536)]
            embeddings.append(embedding)

        return embeddings


class MockWriterAgent:
    """Mock writer for testing pipeline."""

    async def write_chunks(
        self,
        file_path: str,
        chunks: List[Dict[str, Any]],
        metadata: Dict = None
    ) -> str:
        """Mock database write - just generates ID."""
        await asyncio.sleep(0.1)  # Simulate I/O

        # Generate deterministic document ID
        import hashlib
        doc_id = hashlib.md5(file_path.encode()).hexdigest()[:16]

        return doc_id


# ============================================================================
# CLI INTERFACE
# ============================================================================

async def main():
    """Command-line interface for the pipeline orchestrator."""
    parser = argparse.ArgumentParser(
        description="DocuMind Pipeline Orchestrator - Process documents through Extract → Chunk → Embed → Write",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Process a single file
  python orchestrate.py document.pdf

  # Process multiple files
  python orchestrate.py file1.md file2.pdf file3.docx

  # Process entire directory
  python orchestrate.py -d demo-docs/

  # Process with custom settings
  python orchestrate.py -d demo-docs/ --max-parallel 20 --no-continue-on-error

  # Save JSON report
  python orchestrate.py demo-docs/*.pdf --json-output report.json
        """
    )

    parser.add_argument(
        "files",
        nargs="*",
        help="File paths to process (supports glob patterns)"
    )
    parser.add_argument(
        "-d", "--directory",
        help="Process all supported files in directory (recursive)"
    )
    parser.add_argument(
        "--max-parallel",
        type=int,
        default=10,
        help="Maximum concurrent documents (default: 10)"
    )
    parser.add_argument(
        "--no-continue-on-error",
        action="store_true",
        help="Stop pipeline if any document fails"
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress progress indicators"
    )
    parser.add_argument(
        "--no-progress-bar",
        action="store_true",
        help="Disable real-time progress bar (use per-file logging instead)"
    )
    parser.add_argument(
        "--json-output",
        help="Save results to JSON file"
    )

    args = parser.parse_args()

    # Validate input
    if not args.files and not args.directory:
        parser.print_help()
        print("\n❌ Error: Must provide either files or --directory")
        sys.exit(1)

    # Initialize orchestrator
    orchestrator = PipelineOrchestrator(
        max_parallel=args.max_parallel,
        continue_on_error=not args.no_continue_on_error,
        verbose=not args.quiet,
        show_progress_bar=not args.no_progress_bar and not args.quiet
    )

    # Process documents
    start_time = time.time()

    try:
        if args.directory:
            results = await orchestrator.process_directory(args.directory)
        else:
            # Expand glob patterns in file list
            file_paths = []
            for pattern in args.files:
                if "*" in pattern or "?" in pattern:
                    from glob import glob
                    file_paths.extend(glob(pattern))
                else:
                    file_paths.append(pattern)

            results = await orchestrator.process_batch(file_paths)

        orchestrator.metrics.total_time = time.time() - start_time

        # Generate and print report
        report = orchestrator.generate_report(results)
        print(report)

        # Save JSON output if requested
        if args.json_output:
            output_data = {
                "metrics": {
                    "total_documents": orchestrator.metrics.total_documents,
                    "successful": orchestrator.metrics.successful,
                    "failed": orchestrator.metrics.failed,
                    "total_chunks": orchestrator.metrics.total_chunks,
                    "total_embeddings": orchestrator.metrics.total_embeddings,
                    "total_time": orchestrator.metrics.total_time,
                    "stage_times": orchestrator.metrics.stage_times,
                    "errors_by_stage": orchestrator.metrics.errors_by_stage
                },
                "results": [
                    {
                        "file_path": r.file_path,
                        "status": r.status,
                        "stage_completed": r.stage_completed,
                        "chunks_created": r.chunks_created,
                        "embeddings_generated": r.embeddings_generated,
                        "processing_time": r.processing_time,
                        "error_message": r.error_message,
                        "metadata": r.metadata
                    }
                    for r in results
                ]
            }

            with open(args.json_output, 'w') as f:
                json.dump(output_data, f, indent=2)

            print(f"📝 JSON report saved to: {args.json_output}")

        # Exit with error code if any failures
        sys.exit(0 if orchestrator.metrics.failed == 0 else 1)

    except Exception as e:
        print(f"\n❌ Fatal error: {e}")
        if not args.quiet:
            traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
