#!/usr/bin/env python3
"""
DocuMind Upload CLI - Fast batch document upload with embeddings.

Uploads documents to DocuMind with:
- Document processing (PDF, DOCX, CSV, XLSX, TXT, MD)
- Chunk generation with metadata
- OpenAI embedding generation (1536 dimensions)
- Storage in document_chunks table for RAG search

Usage:
    python -m src.documind.cli.upload_cli file1.pdf file2.docx file3.csv
    python -m src.documind.cli.upload_cli docs/workshops/S7-sample-docs/*.pdf
    python -m src.documind.cli.upload_cli --dir docs/workshops/S7-sample-docs/
    python -m src.documind.cli.upload_cli --no-embeddings file.pdf  # Skip embeddings
"""

import argparse
import sys
import time
import json
import os
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Dict, Any, Optional

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from dotenv import load_dotenv
load_dotenv()

from src.documind.processor import DocumentProcessor


# ANSI colors for terminal output
class Colors:
    GREEN = '\033[92m'
    RED = '\033[91m'
    YELLOW = '\033[93m'
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    BOLD = '\033[1m'
    END = '\033[0m'


# Lazy-loaded clients
_openai_client = None
_supabase_client = None


def get_openai_client():
    """Get or create OpenAI client."""
    global _openai_client
    if _openai_client is None:
        from openai import OpenAI
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY not set")
        _openai_client = OpenAI(api_key=api_key)
    return _openai_client


def get_supabase_client():
    """Get or create Supabase client."""
    global _supabase_client
    if _supabase_client is None:
        from supabase import create_client
        url = os.getenv("SUPABASE_URL")
        key = os.getenv("SUPABASE_SERVICE_KEY") or os.getenv("SUPABASE_ANON_KEY")
        if not url or not key:
            raise ValueError("SUPABASE_URL and SUPABASE_SERVICE_KEY not set")
        _supabase_client = create_client(url, key)
    return _supabase_client


def generate_embeddings(texts: List[str], batch_size: int = 50) -> List[List[float]]:
    """Generate embeddings for texts using OpenAI."""
    client = get_openai_client()
    all_embeddings = []

    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        response = client.embeddings.create(
            model="text-embedding-3-small",
            input=batch
        )
        all_embeddings.extend([item.embedding for item in response.data])

    return all_embeddings


def upload_to_documind(processed_doc, generate_emb: bool = True) -> dict:
    """Upload document with embeddings to DocuMind."""
    try:
        client = get_supabase_client()

        # Prepare document metadata
        metadata = {
            "fingerprint": processed_doc.metadata.fingerprint,
            "word_count": processed_doc.metadata.basic.word_count,
            "chunks": len(processed_doc.chunks),
            "source": "upload-cli",
            "processor": "documind-processor-v2",
            "has_embeddings": generate_emb
        }

        # Insert document into documents table
        doc_result = client.table("documents").insert({
            "title": processed_doc.file_name,
            "content": processed_doc.content,
            "file_type": processed_doc.extractor_used,
            "metadata": metadata
        }).execute()

        if not doc_result.data:
            return {"success": False, "error": "Document insert failed", "title": processed_doc.file_name}

        doc_id = doc_result.data[0].get("id")
        chunks_written = 0

        # Generate embeddings and store chunks
        if generate_emb and processed_doc.chunks:
            # Extract chunk texts
            chunk_texts = [chunk.content for chunk in processed_doc.chunks]

            # Generate embeddings
            embeddings = generate_embeddings(chunk_texts)

            # Prepare chunk records
            chunk_records = []
            for i, (chunk, embedding) in enumerate(zip(processed_doc.chunks, embeddings)):
                chunk_records.append({
                    "document_id": doc_id,
                    "content": chunk.content,
                    "chunk_index": chunk.chunk_index,
                    "embedding": embedding,
                    "word_count": chunk.word_count,
                    "metadata": {
                        "section_heading": getattr(chunk, 'section_heading', None),
                        "document_name": processed_doc.file_name
                    }
                })

            # Batch insert chunks
            if chunk_records:
                chunk_result = client.table("document_chunks").insert(chunk_records).execute()
                if chunk_result.data:
                    chunks_written = len(chunk_result.data)

        return {
            "success": True,
            "document_id": doc_id,
            "chunks_written": chunks_written,
            "title": processed_doc.file_name
        }

    except Exception as e:
        return {"success": False, "error": str(e), "title": getattr(processed_doc, 'file_name', 'unknown')}


def process_and_upload(file_path: str, processor: DocumentProcessor, generate_emb: bool = True) -> Dict[str, Any]:
    """Process a single document and upload to DocuMind with embeddings."""
    start = time.time()
    path = Path(file_path)

    try:
        # Process document
        result = processor.process_document(str(path))

        # Upload to DocuMind with embeddings
        upload_result = upload_to_documind(result, generate_emb=generate_emb)

        elapsed = time.time() - start

        return {
            "file": path.name,
            "success": upload_result.get("success", False),
            "document_id": upload_result.get("document_id"),
            "chunks_written": upload_result.get("chunks_written", 0),
            "format": result.extractor_used,
            "words": result.metadata.basic.word_count,
            "chunks": len(result.chunks),
            "time": elapsed,
            "error": upload_result.get("error")
        }
    except Exception as e:
        elapsed = time.time() - start
        return {
            "file": path.name,
            "success": False,
            "error": str(e),
            "time": elapsed
        }


def collect_files(paths: List[str], directory: str = None, recursive: bool = False) -> List[str]:
    """Collect all files to process."""
    files = []

    # Supported extensions
    supported = {'.pdf', '.docx', '.csv', '.xlsx', '.txt', '.md', '.markdown'}

    # Add files from directory
    if directory:
        dir_path = Path(directory)
        if dir_path.is_dir():
            pattern = '**/*' if recursive else '*'
            for ext in supported:
                files.extend([str(f) for f in dir_path.glob(f'{pattern}{ext}')])

    # Add individual files
    for path in paths:
        p = Path(path)
        if p.is_file() and p.suffix.lower() in supported:
            files.append(str(p))
        elif p.is_dir():
            for ext in supported:
                files.extend([str(f) for f in p.glob(f'*{ext}')])

    return list(set(files))  # Remove duplicates


def main():
    parser = argparse.ArgumentParser(
        description='Fast batch document upload to DocuMind',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s file1.pdf file2.docx
  %(prog)s --dir docs/workshops/S7-sample-docs/
  %(prog)s --dir docs/ --recursive
  %(prog)s *.pdf *.docx --workers 8
        """
    )

    parser.add_argument('files', nargs='*', help='Files to upload')
    parser.add_argument('--dir', '-d', help='Directory to scan for documents')
    parser.add_argument('--recursive', '-r', action='store_true', help='Scan directory recursively')
    parser.add_argument('--workers', '-w', type=int, default=4, help='Parallel workers (default: 4)')
    parser.add_argument('--dry-run', action='store_true', help='Process without uploading')
    parser.add_argument('--no-embeddings', action='store_true', help='Skip embedding generation (faster, but no RAG search)')
    parser.add_argument('--json', action='store_true', help='Output results as JSON')
    parser.add_argument('--quiet', '-q', action='store_true', help='Minimal output')

    args = parser.parse_args()

    # Collect files
    files = collect_files(args.files, args.dir, args.recursive)

    if not files:
        print(f"{Colors.RED}No supported files found.{Colors.END}")
        print(f"Supported formats: PDF, DOCX, CSV, XLSX, TXT, MD")
        sys.exit(1)

    generate_emb = not args.no_embeddings

    if not args.quiet:
        print(f"\n{Colors.BOLD}📤 DocuMind Upload CLI{Colors.END}")
        print(f"{'=' * 50}")
        print(f"Files to process: {len(files)}")
        print(f"Workers: {args.workers}")
        print(f"Embeddings: {Colors.GREEN}enabled{Colors.END}" if generate_emb else f"Embeddings: {Colors.YELLOW}disabled{Colors.END}")
        if args.dry_run:
            print(f"{Colors.YELLOW}DRY RUN - No uploads will be performed{Colors.END}")
        print()

    # Initialize processor
    processor = DocumentProcessor(auto_upload=False)

    # Process files in parallel
    results = []
    start_time = time.time()

    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        if args.dry_run:
            # Just process without upload
            futures = {
                executor.submit(processor.process_document, f): f
                for f in files
            }
        else:
            futures = {
                executor.submit(process_and_upload, f, processor, generate_emb): f
                for f in files
            }

        for i, future in enumerate(as_completed(futures), 1):
            file_path = futures[future]

            try:
                if args.dry_run:
                    doc = future.result()
                    result = {
                        "file": Path(file_path).name,
                        "success": True,
                        "format": doc.extractor_used,
                        "words": doc.metadata.basic.word_count,
                        "chunks": len(doc.chunks),
                        "fingerprint": doc.metadata.fingerprint[:16]
                    }
                else:
                    result = future.result()
            except Exception as e:
                result = {"file": Path(file_path).name, "success": False, "error": str(e)}

            results.append(result)

            if not args.quiet and not args.json:
                status = f"{Colors.GREEN}✓{Colors.END}" if result.get("success") else f"{Colors.RED}✗{Colors.END}"
                name = result["file"][:40]
                if result.get("success"):
                    words = result.get("words", 0)
                    fmt = result.get("format", "?")
                    emb_count = result.get("chunks_written", 0)
                    emb_info = f" [{emb_count} emb]" if emb_count > 0 else ""
                    print(f"  {status} [{i}/{len(files)}] {name:<40} {fmt:<5} {words:>5} words{emb_info}")
                else:
                    error = result.get("error", "Unknown error")[:50]
                    print(f"  {status} [{i}/{len(files)}] {name:<40} {Colors.RED}{error}{Colors.END}")

    total_time = time.time() - start_time

    # Summary
    success_count = sum(1 for r in results if r.get("success"))
    fail_count = len(results) - success_count

    if args.json:
        output = {
            "total": len(results),
            "success": success_count,
            "failed": fail_count,
            "time_seconds": round(total_time, 2),
            "results": results
        }
        print(json.dumps(output, indent=2))
    elif not args.quiet:
        print()
        print(f"{'=' * 50}")
        print(f"{Colors.BOLD}Summary:{Colors.END}")
        print(f"  {Colors.GREEN}✓ Success: {success_count}{Colors.END}")
        if fail_count:
            print(f"  {Colors.RED}✗ Failed:  {fail_count}{Colors.END}")
        print(f"  ⏱ Time:    {total_time:.2f}s ({len(files)/total_time:.1f} docs/sec)")
        print()

    sys.exit(0 if fail_count == 0 else 1)


if __name__ == "__main__":
    main()
