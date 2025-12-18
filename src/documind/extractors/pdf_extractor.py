"""
PDF Text Extraction using pdfplumber
Handles simple and complex PDF layouts
"""
import pdfplumber
from typing import Dict, List, Optional
from pathlib import Path

class PDFExtractor:
    """Extract text and metadata from PDF files"""

    def extract_text(self, pdf_path: str) -> Dict:
        """
        Extract all text from a PDF file.

        Args:
            pdf_path: Path to PDF file

        Returns:
            Dictionary with text, metadata, and page information
        """
        try:
            with pdfplumber.open(pdf_path) as pdf:
                # Extract metadata
                metadata = pdf.metadata or {}

                # Extract text from each page
                pages = []
                full_text = ""

                for i, page in enumerate(pdf.pages):
                    page_text = page.extract_text() or ""
                    pages.append({
                        "page_number": i + 1,
                        "text": page_text,
                        "width": page.width,
                        "height": page.height
                    })
                    full_text += page_text + "\n\n"

                return {
                    "success": True,
                    "text": full_text.strip(),
                    "pages": pages,
                    "page_count": len(pdf.pages),
                    "metadata": {
                        "author": metadata.get("Author", ""),
                        "title": metadata.get("Title", ""),
                        "subject": metadata.get("Subject", ""),
                        "creator": metadata.get("Creator", ""),
                        "producer": metadata.get("Producer", ""),
                        "creation_date": str(metadata.get("CreationDate", "")),
                    }
                }

        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "message": f"Failed to extract text from {pdf_path}"
            }

    def extract_text_simple(self, pdf_path: str) -> str:
        """
        Simple extraction - just return the text string.

        Args:
            pdf_path: Path to PDF file

        Returns:
            Extracted text as string
        """
        result = self.extract_text(pdf_path)
        return result.get("text", "") if result["success"] else ""

    def extract_tables(self, pdf_path: str) -> Dict:
        """
        Extract tables from PDF, preserving structure.

        Args:
            pdf_path: Path to PDF file

        Returns:
            Dictionary with tables and their metadata
        """
        try:
            with pdfplumber.open(pdf_path) as pdf:
                all_tables = []

                for page_num, page in enumerate(pdf.pages):
                    # Extract tables from this page
                    tables = page.extract_tables()

                    for table_num, table in enumerate(tables):
                        if table:
                            all_tables.append({
                                "page": page_num + 1,
                                "table_number": table_num + 1,
                                "rows": len(table),
                                "columns": len(table[0]) if table else 0,
                                "data": table,
                                "headers": table[0] if table else []
                            })

                return {
                    "success": True,
                    "table_count": len(all_tables),
                    "tables": all_tables
                }

        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "message": f"Failed to extract tables from {pdf_path}"
            }

    def tables_to_markdown(self, tables_result: Dict) -> str:
        """
        Convert extracted tables to Markdown format.

        Args:
            tables_result: Result from extract_tables()

        Returns:
            Markdown-formatted tables as string
        """
        if not tables_result["success"]:
            return ""

        markdown = ""

        for table in tables_result["tables"]:
            markdown += f"\n## Table {table['table_number']} (Page {table['page']})\n\n"

            data = table["data"]
            if not data:
                continue

            # Header row
            markdown += "| " + " | ".join(str(cell) for cell in data[0]) + " |\n"
            markdown += "|" + "|".join(["---"] * len(data[0])) + "|\n"

            # Data rows
            for row in data[1:]:
                markdown += "| " + " | ".join(str(cell) for cell in row) + " |\n"

            markdown += "\n"

        return markdown
    
# Test function
# Test function
if __name__ == "__main__":
    extractor = PDFExtractor()

    # Test table extraction with sample employee directory
    print("=" * 60)
    print("TESTING TABLE EXTRACTION")
    print("=" * 60)

    tables_result = extractor.extract_tables("docs/workshops/S7-sample-docs/employee_directory.pdf")

    if tables_result["success"]:
        print(f"Found {tables_result['table_count']} tables")
        print("\nMarkdown Format:")
        print(extractor.tables_to_markdown(tables_result))
    else:
        print(f"Error: {tables_result['error']}")