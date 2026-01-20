# HeroForge.AI Course: AI-Powered Software Development
## Lesson 7 Workshop: Advanced Data Extraction

**Estimated Time:** 45-60 minutes\
**Difficulty:** Intermediate-Advanced\
**Prerequisites:** Completed Sessions 1-6 (Document processing, RAG, Q&A system)

---

## Before You Begin: Plan Your Work!

> **📋 Reminder:** In Session 3, we learned about **Issue-Driven Development** - the practice of creating GitHub Issues *before* starting work. This ensures clear requirements, enables collaboration, and creates traceability between your code and original requirements.
>
> **Before diving into this workshop:**
> 1. Create a GitHub Issue for the features you'll build today
> 2. Reference that issue in your branch name (`issue-XX-feature-name`)
> 3. Include `Closes #XX` or `Relates to #XX` in your commit messages
>
> 👉 See [S3-Workshop: Planning Your Work with GitHub Issues](./S3-Workshop.md#planning-your-work-with-github-issues-5-minutes) for the full workflow.

---

## Workshop Objectives

By completing this workshop, you will:
- [x] Extract text from PDFs with varying complexity
- [x] Parse structured data from Word documents and spreadsheets
- [x] Preserve document structure and metadata during extraction
- [x] Handle tables, multi-column layouts, and complex formatting
- [x] Build production-ready multi-format document processors
- [x] Optimize extracted text for LLM consumption and RAG

---

## Module 1: PDF Text Extraction (15 minutes)

### 📥 Get Sample Documents
Instead of generating fake text files, we will use "real-world" corporate documents that contain messy tables and complex layouts.

**Sync with Upstream**

First, ensure you have the latest course materials by pulling from the upstream repository:

```bash
# If you haven't added the upstream remote yet:
git remote add upstream https://github.com/YOUR-INSTRUCTOR/heroforge-documind.git

# Fetch and merge the latest changes
git fetch upstream
git merge upstream/main
```

**Available Sample Documents:**

The sample documents are located at `docs/workshops/S7-sample-docs/` in your repository. You can use them directly from this path—no need to copy them elsewhere.

| File | Type | Description |
|------|------|-------------|
| `invoice_acme.pdf` | PDF | Complex invoice with tables and line items |
| `employee_directory.pdf` | PDF | Staff directory with multi-column layout |
| `simple_security_policy.pdf` | PDF | Basic policy document for testing |
| `employee_handbook.docx` | Word | HR document with headings and tables |
| `meeting_notes.docx` | Word | Meeting minutes with action items |
| `employee_data.csv` | CSV | Employee records with salary data |
| `quarterly_report.csv` | CSV | Financial data for analysis |
| `product_catalog.csv` | CSV | Product listing with pricing |

We will use these files for extraction testing throughout the workshop.

---

### Concept Review

**Why is PDF Extraction Challenging?**

PDFs are notoriously difficult to extract text from because:
- **Two types**: Text-based (digital) vs image-based (scanned)
- **Layout complexity**: Columns, tables, headers, footers, sidebars
- **No semantic structure**: PDF is "paint on canvas"—no paragraphs, just positioned text
- **Inconsistent quality**: Varying fonts, encodings, and formats

**PDF Extraction Tools Comparison:**

| Tool | Strengths | Limitations | Best For |
|------|-----------|-------------|----------|
| **PyPDF2** | Simple, fast, pure Python | No layout analysis, struggles with tables | Simple, single-column PDFs |
| **pdfplumber** | Table extraction, layout analysis | Slower, Python-only | Complex PDFs with tables |
| **PyMuPDF (fitz)** | Fast, powerful, good accuracy | C dependency, larger install | Production systems |
| **Unstructured** | AI-powered, handles all formats | Requires API or large model | Complex, multi-format documents |

For this workshop, we'll use **pdfplumber** for its excellent table handling and balance of simplicity vs capability.

---

### Exercise 1.1: Simple PDF Text Extraction

**Task:** Extract text from a simple, single-column PDF document using the sample `simple_security_policy.pdf`.

**Instructions:**

**Step 1: Install PDF Libraries (2 mins)**

```bash
# Install pdfplumber and dependencies
pip install pdfplumber Pillow

# Verify installation
python -c "import pdfplumber; print('pdfplumber installed successfully')"
```

**Step 2: Extract Text with pdfplumber (5 mins)**

Create `src/documind/extractors/pdf_extractor.py`:

```python
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

# Test function
if __name__ == "__main__":
    extractor = PDFExtractor()

    # Test with simple PDF (use the sample document)
    result = extractor.extract_text("docs/workshops/S7-sample-docs/simple_security_policy.pdf")

    if result["success"]:
        print("=" * 60)
        print("PDF EXTRACTION SUCCESSFUL")
        print("=" * 60)
        print(f"Pages: {result['page_count']}")
        print(f"Title: {result['metadata']['title']}")
        print("\nExtracted Text:")
        print("-" * 60)
        print(result["text"])
    else:
        print(f"Error: {result['error']}")
```

**Step 3: Test the Extractor (5 mins)**

```bash
# Create directory for extractors
mkdir -p src/documind/extractors
touch src/documind/extractors/__init__.py

# Run the extractor
python src/documind/extractors/pdf_extractor.py
```

**Expected Output:**
```
============================================================
PDF EXTRACTION SUCCESSFUL
============================================================
Pages: 1
Title: Simple Security Policy

Extracted Text:
------------------------------------------------------------
Company Security Policy

This document outlines our security requirements for all employees.

1. All employees must use strong passwords (minimum 12 characters)
2. Two-factor authentication is required for all systems
3. Sensitive data must be encrypted at rest and in transit
4. Report security incidents within 24 hours

For questions, contact security@company.com
```

> **Note:** Your output will reflect the actual content of the `simple_security_policy.pdf` sample document.

---

### Exercise 1.2: Complex PDF with Tables

**Task:** Extract tables from a PDF and preserve their structure using the sample `employee_directory.pdf`.

**Instructions:**

**Step 1: Add Table Extraction Methods (5 mins)**

Open `src/documind/extractors/pdf_extractor.py` in your editor and add these two methods **inside the `PDFExtractor` class**, right after the `extract_text_simple` method (before the `if __name__ == "__main__":` line):

```python
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
```

**Step 2: Update the Test Section (2 mins)**

In the same file, **replace** the existing `if __name__ == "__main__":` section at the bottom with this new test code:

```python
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
```

**Step 3: Run the Test (1 min)**

In your terminal, run:

```bash
python src/documind/extractors/pdf_extractor.py
```

**Expected Output:**
```
============================================================
TESTING TABLE EXTRACTION
============================================================
Found 1 tables

Markdown Format:

## Table 1 (Page 1)

| Name | Department | Email | Extension |
|---|---|---|---|
| Alice Johnson | Engineering | alice@company.com | 1001 |
| Bob Smith | Sales | bob@company.com | 1002 |
| Carol White | HR | carol@company.com | 1003 |
| David Brown | Engineering | david@company.com | 1004 |
```

> **Note:** Your output will reflect the actual content of the `employee_directory.pdf` sample document.

---

### Quiz 1:

**Question 1:** Why are PDFs challenging for text extraction?\
   a) PDFs are encrypted by default\
   b) PDFs can only contain images\
   c) PDFs store positioned text without semantic structure—no inherent paragraphs, sections, or reading order\
   d) PDFs are too small to extract text from

**Question 2:** What is the main advantage of pdfplumber over PyPDF2?\
   a) pdfplumber is faster\
   b) pdfplumber doesn't require Python\
   c) pdfplumber works on images only\
   d) pdfplumber can detect and extract tables while preserving structure, plus it handles complex layouts better

**Question 3:** When should you convert extracted tables to Markdown format?\
   a) Always, for every document type\
   b) When preparing text for LLM consumption, as Markdown preserves table structure in a text-friendly format\
   c) Never, JSON is always better\
   d) Only for images

**Answers:**
1. **c)** PDFs lack semantic structure—text is positioned, not organized into meaningful elements
2. **d)** pdfplumber excels at table detection and extraction with layout analysis capabilities
3. **b)** Markdown tables are LLM-friendly and preserve structure for RAG and Q&A systems

---

## Module 2: Multi-Format Document Support (15 minutes)

### Concept Review

**Document Formats and Their Challenges:**

| Format | Extraction Tool | Challenge | Solution |
|--------|----------------|-----------|----------|
| **Word (.docx)** | python-docx | Embedded objects, formatting | Extract paragraphs + tables separately |
| **Excel (.xlsx)** | pandas, openpyxl | Multiple sheets, formulas | Process each sheet, evaluate formulas |
| **PowerPoint (.pptx)** | python-pptx | Slides, speaker notes, layouts | Extract text + notes slide-by-slide |
| **HTML** | BeautifulSoup | Messy markup, scripts, styles | Parse with selector, clean text |
| **Markdown** | Plain text + regex | Metadata, code blocks | Parse frontmatter, preserve structure |

---

### Exercise 2.1: Word Document Extraction

**Task:** Extract text, tables, and metadata from Word documents.

**Instructions:**

**Step 1: Install Word Library (2 mins)**

```bash
pip install python-docx
```

**Step 2: Create Word Extractor (8 mins)**

Create `src/documind/extractors/docx_extractor.py`:

```python
"""
Word Document (.docx) Extraction
Handles paragraphs, tables, and metadata
"""
from docx import Document
from typing import Dict, List
from pathlib import Path

class DocxExtractor:
    """Extract content from Word documents"""

    def extract(self, docx_path: str) -> Dict:
        """
        Extract all content from a Word document.

        Args:
            docx_path: Path to .docx file

        Returns:
            Dictionary with text, tables, and metadata
        """
        try:
            doc = Document(docx_path)

            # Extract metadata
            core_properties = doc.core_properties
            metadata = {
                "title": core_properties.title or "",
                "author": core_properties.author or "",
                "subject": core_properties.subject or "",
                "created": str(core_properties.created) if core_properties.created else "",
                "modified": str(core_properties.modified) if core_properties.modified else "",
            }

            # Extract paragraphs
            paragraphs = []
            for para in doc.paragraphs:
                if para.text.strip():
                    paragraphs.append({
                        "text": para.text,
                        "style": para.style.name
                    })

            # Extract tables
            tables = []
            for table_num, table in enumerate(doc.tables):
                table_data = []
                for row in table.rows:
                    row_data = [cell.text for cell in row.cells]
                    table_data.append(row_data)

                tables.append({
                    "table_number": table_num + 1,
                    "rows": len(table_data),
                    "columns": len(table_data[0]) if table_data else 0,
                    "data": table_data
                })

            # Combine all text
            full_text = "\n\n".join(para["text"] for para in paragraphs)

            return {
                "success": True,
                "text": full_text,
                "paragraphs": paragraphs,
                "tables": tables,
                "metadata": metadata,
                "paragraph_count": len(paragraphs),
                "table_count": len(tables)
            }

        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "message": f"Failed to extract from {docx_path}"
            }

    def extract_text_only(self, docx_path: str) -> str:
        """Extract just the text content"""
        result = self.extract(docx_path)
        return result.get("text", "") if result["success"] else ""

    def format_for_llm(self, docx_path: str) -> str:
        """
        Format document content for LLM consumption.
        Combines text and tables in a readable format.
        """
        result = self.extract(docx_path)

        if not result["success"]:
            return ""

        output = []

        # Add metadata as header
        if result["metadata"]["title"]:
            output.append(f"# {result['metadata']['title']}\n")

        # Add paragraphs
        for para in result["paragraphs"]:
            # Check if it's a heading
            if para["style"].startswith("Heading"):
                level = para["style"][-1]
                output.append(f"{'#' * int(level)} {para['text']}\n")
            else:
                output.append(f"{para['text']}\n")

        # Add tables in markdown format
        for table in result["tables"]:
            output.append(f"\n## Table {table['table_number']}\n")

            data = table["data"]
            if not data:
                continue

            # Header
            output.append("| " + " | ".join(data[0]) + " |")
            output.append("|" + "|".join(["---"] * len(data[0])) + "|")

            # Rows
            for row in data[1:]:
                output.append("| " + " | ".join(row) + " |")

            output.append("")

        return "\n".join(output)

# Test
if __name__ == "__main__":
    # Test extraction using the sample document
    extractor = DocxExtractor()
    result = extractor.extract("docs/workshops/S7-sample-docs/meeting_notes.docx")

    print("=" * 60)
    print("DOCX EXTRACTION TEST")
    print("=" * 60)
    print(f"Title: {result['metadata']['title']}")
    print(f"Author: {result['metadata']['author']}")
    print(f"Paragraphs: {result['paragraph_count']}")
    print(f"Tables: {result['table_count']}")

    print("\n" + "-" * 60)
    print("LLM-Formatted Output:")
    print("-" * 60)
    print(extractor.format_for_llm("docs/workshops/S7-sample-docs/meeting_notes.docx"))
```

**Step 3: Test Word Extraction (5 mins)**

```bash
python src/documind/extractors/docx_extractor.py
```

**Expected Output:**
```
============================================================
DOCX EXTRACTION TEST
============================================================
Title: Team Meeting Notes
Author: Alice Johnson
Paragraphs: 6
Tables: 1

------------------------------------------------------------
LLM-Formatted Output:
------------------------------------------------------------
# Team Meeting Notes

# Team Meeting - November 24, 2025

Attendees: Alice, Bob, Carol, David

Topic: Q4 Planning and Budget Review

## Action Items

1. Alice: Submit budget proposal by Friday

2. Bob: Schedule follow-up meeting

3. Carol: Update project timeline

## Table 1

| Task | Owner | Due Date |
|---|---|---|
| Budget Proposal | Alice | Nov 29 |
| Follow-up Meeting | Bob | Dec 1 |
| Timeline Update | Carol | Nov 30 |
```

> **Note:** Your output will reflect the actual content of the `meeting_notes.docx` sample document.

---

### Exercise 2.2: Excel/CSV Extraction

**Task:** Extract data from spreadsheets.

**Instructions:**

**Step 1: Install Pandas (1 min)**

```bash
pip install pandas openpyxl
```

**Step 2: Create Spreadsheet Extractor (5 mins)**

Create `src/documind/extractors/spreadsheet_extractor.py`:

```python
"""
Spreadsheet (.xlsx, .csv) Extraction
"""
import pandas as pd
from typing import Dict, List
from pathlib import Path

class SpreadsheetExtractor:
    """Extract data from Excel and CSV files"""

    def extract_excel(self, file_path: str) -> Dict:
        """Extract all sheets from Excel file"""
        try:
            # Read all sheets
            sheets = pd.read_excel(file_path, sheet_name=None)

            result = {
                "success": True,
                "sheet_count": len(sheets),
                "sheets": {}
            }

            for sheet_name, df in sheets.items():
                result["sheets"][sheet_name] = {
                    "rows": len(df),
                    "columns": len(df.columns),
                    "column_names": df.columns.tolist(),
                    "data": df.to_dict(orient="records"),
                    "preview": df.head(5).to_string()
                }

            return result

        except Exception as e:
            return {
                "success": False,
                "error": str(e)
            }

    def extract_csv(self, file_path: str) -> Dict:
        """Extract data from CSV file"""
        try:
            df = pd.read_csv(file_path)

            return {
                "success": True,
                "rows": len(df),
                "columns": len(df.columns),
                "column_names": df.columns.tolist(),
                "data": df.to_dict(orient="records"),
                "preview": df.head(10).to_string()
            }

        except Exception as e:
            return {
                "success": False,
                "error": str(e)
            }

    def format_for_llm(self, file_path: str) -> str:
        """Format spreadsheet for LLM consumption"""
        ext = Path(file_path).suffix.lower()

        if ext == ".csv":
            result = self.extract_csv(file_path)
            if not result["success"]:
                return ""

            output = [f"# CSV Data: {Path(file_path).name}\n"]
            output.append(f"**Rows:** {result['rows']}")
            output.append(f"**Columns:** {', '.join(result['column_names'])}\n")
            output.append("## Data Preview\n")
            output.append(result["preview"])

            return "\n".join(output)

        elif ext in [".xlsx", ".xls"]:
            result = self.extract_excel(file_path)
            if not result["success"]:
                return ""

            output = [f"# Excel File: {Path(file_path).name}\n"]
            output.append(f"**Sheets:** {result['sheet_count']}\n")

            for sheet_name, sheet_data in result["sheets"].items():
                output.append(f"## Sheet: {sheet_name}")
                output.append(f"**Rows:** {sheet_data['rows']}")
                output.append(f"**Columns:** {', '.join(sheet_data['column_names'])}\n")
                output.append("### Data Preview\n")
                output.append(sheet_data["preview"])
                output.append("")

            return "\n".join(output)

        return ""

# Test
if __name__ == "__main__":
    # Test extraction using the sample CSV
    extractor = SpreadsheetExtractor()
    result = extractor.extract_csv("docs/workshops/S7-sample-docs/employee_data.csv")

    print("=" * 60)
    print("CSV EXTRACTION TEST")
    print("=" * 60)
    print(f"Rows: {result['rows']}")
    print(f"Columns: {result['columns']}")
    print(f"Column Names: {result['column_names']}")
    print("\nLLM-Formatted:\n")
    print(extractor.format_for_llm("docs/workshops/S7-sample-docs/employee_data.csv"))
```

**Step 3: Run the Test (1 min)**

In your terminal, run:

```bash
python src/documind/extractors/spreadsheet_extractor.py
```

**Expected Output:**
```
============================================================
CSV EXTRACTION TEST
============================================================
Rows: 4
Columns: 4
Column Names: ['Employee', 'Department', 'Salary', 'Start Date']

LLM-Formatted:

# CSV Data: employee_data.csv

**Rows:** 4
**Columns:** Employee, Department, Salary, Start Date

## Data Preview

  Employee   Department  Salary  Start Date
0    Alice  Engineering   95000  2020-01-15
1      Bob        Sales   75000  2021-03-20
2    Carol           HR   80000  2019-07-01
3    David  Engineering   90000  2022-05-10
```

> **Note:** Your output will reflect the actual content of the `employee_data.csv` sample document.

---

### Quiz 2:

**Question 1:** Why use python-docx instead of just reading .docx as text?\
   a) python-docx is faster\
   b) .docx files are actually ZIP archives with XML—python-docx parses the structure to extract text, tables, and metadata properly\
   c) Text extraction doesn't work on .docx files\
   d) python-docx is required by Microsoft

**Question 2:** What's the advantage of formatting extracted data as Markdown for LLMs?\
   a) Markdown files are smaller\
   b) Markdown is encrypted\
   c) Markdown preserves structure (headings, tables, lists) in a text format that LLMs understand well for Q&A and RAG\
   d) LLMs can only read Markdown

**Question 3:** When extracting spreadsheets, why might you want to process each sheet separately?\
   a) Sheets are stored in different files\
   b) It's required by pandas\
   c) You can only read one sheet at a time\
   d) Different sheets often contain different types of data that need separate context and processing for accurate LLM understanding

**Answers:**
1. **b)** .docx is a ZIP containing XML—python-docx properly parses the structure
2. **c)** Markdown preserves structure while being LLM-friendly for RAG pipelines
3. **d)** Separate sheets often have different semantics requiring separate processing and context

---

## Module 3: Metadata and Structure Preservation (15 minutes)

### Concept Review

**Why Preserve Structure?**

When extracting documents for RAG systems, preserving structure improves:
- **Retrieval accuracy**: Semantic search works better with structured text
- **Context quality**: LLMs understand hierarchical information better
- **Source attribution**: Users can trace answers back to specific sections
- **Chunk quality**: Intelligent chunking respects document structure

**Metadata to Extract:**
- **Document-level**: Title, author, creation date, file type
- **Content-level**: Headings, sections, reading order
- **Semantic**: Topics, entities, categories
- **Technical**: Page numbers, word count, language

---

### Exercise 3.1: Metadata Extraction Pipeline

**Task:** Build a unified metadata extractor for all document types.

**Instructions:**

**Step 1: Create Metadata Extractor (10 mins)**

Create `src/documind/extractors/metadata_extractor.py`:

```python
"""
Unified Metadata Extraction
Extracts rich metadata from all document types
"""
from pathlib import Path
from datetime import datetime
from typing import Dict, Optional
import re

class MetadataExtractor:
    """Extract and enrich document metadata"""

    def extract_basic_metadata(self, file_path: str, content: str) -> Dict:
        """
        Extract basic filesystem and content metadata.

        Args:
            file_path: Path to file
            content: Extracted text content

        Returns:
            Dictionary with metadata
        """
        path = Path(file_path)
        stats = path.stat()

        # Count content statistics
        words = content.split()
        lines = content.split('\n')

        return {
            "file_name": path.name,
            "file_path": str(path.absolute()),
            "file_size_bytes": stats.st_size,
            "file_type": path.suffix.lower(),
            "created_at": datetime.fromtimestamp(stats.st_ctime).isoformat(),
            "modified_at": datetime.fromtimestamp(stats.st_mtime).isoformat(),
            "word_count": len(words),
            "character_count": len(content),
            "line_count": len(lines),
            "estimated_read_time_minutes": max(1, len(words) // 200)
        }

    def extract_structure(self, content: str) -> Dict:
        """
        Extract document structure (headings, sections).

        Args:
            content: Document text

        Returns:
            Dictionary with structure information
        """
        # Detect headings (Markdown-style or all-caps lines)
        heading_pattern = r'^#+\s+(.+)$|^([A-Z][A-Z\s]{10,})$'
        headings = []

        for i, line in enumerate(content.split('\n')):
            match = re.match(heading_pattern, line.strip())
            if match:
                heading_text = match.group(1) or match.group(2)
                level = line.count('#') if '#' in line else 1
                headings.append({
                    "text": heading_text.strip(),
                    "level": level,
                    "line_number": i + 1
                })

        # Detect sections (text between headings)
        sections = len(headings) + 1

        # Detect lists
        list_items = len(re.findall(r'^\s*[-*•]\s+', content, re.MULTILINE))
        numbered_lists = len(re.findall(r'^\s*\d+\.\s+', content, re.MULTILINE))

        return {
            "heading_count": len(headings),
            "headings": headings,
            "section_count": sections,
            "list_items": list_items,
            "numbered_lists": numbered_lists,
            "has_tables": "---|---" in content or "|" in content[:1000]
        }

    def extract_entities(self, content: str) -> Dict:
        """
        Extract key entities (emails, dates, URLs).

        Args:
            content: Document text

        Returns:
            Dictionary with extracted entities
        """
        # Email addresses
        emails = re.findall(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', content)

        # URLs
        urls = re.findall(r'https?://[^\s]+', content)

        # Dates (various formats)
        date_patterns = [
            r'\b\d{4}-\d{2}-\d{2}\b',  # 2025-11-24
            r'\b\d{1,2}/\d{1,2}/\d{2,4}\b',  # 11/24/2025
            r'\b(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]* \d{1,2},? \d{4}\b'  # November 24, 2025
        ]
        dates = []
        for pattern in date_patterns:
            dates.extend(re.findall(pattern, content, re.IGNORECASE))

        # Phone numbers (simple US format)
        phones = re.findall(r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b', content)

        return {
            "emails": list(set(emails)),
            "urls": list(set(urls)),
            "dates": list(set(dates)),
            "phone_numbers": list(set(phones)),
            "entity_count": len(emails) + len(urls) + len(dates) + len(phones)
        }

    def extract_topics(self, content: str) -> Dict:
        """
        Extract probable topics/categories (simple keyword-based).

        Args:
            content: Document text

        Returns:
            Dictionary with suggested topics and tags
        """
        content_lower = content.lower()

        # Topic keywords
        topics = []

        topic_keywords = {
            "hr": ["employee", "benefit", "vacation", "policy", "hire", "salary", "payroll"],
            "security": ["password", "encryption", "firewall", "authentication", "security", "threat"],
            "engineering": ["code", "software", "api", "database", "development", "deploy"],
            "finance": ["budget", "expense", "revenue", "cost", "invoice", "payment"],
            "legal": ["contract", "agreement", "liability", "compliance", "terms"],
            "sales": ["customer", "revenue", "deal", "quota", "pipeline", "prospect"]
        }

        for topic, keywords in topic_keywords.items():
            if any(keyword in content_lower for keyword in keywords):
                topics.append(topic)

        # Extract frequent meaningful words (simple TF for tags)
        words = re.findall(r'\b[a-z]{4,}\b', content_lower)
        word_freq = {}
        for word in words:
            if word not in ["that", "this", "with", "from", "have", "will", "been", "were"]:
                word_freq[word] = word_freq.get(word, 0) + 1

        # Top 10 words as tags
        tags = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)[:10]
        tags = [word for word, count in tags if count > 2]

        return {
            "suggested_topics": topics,
            "suggested_tags": tags[:5]
        }

    def extract_all(self, file_path: str, content: str) -> Dict:
        """
        Extract all metadata from a document.

        Args:
            file_path: Path to document file
            content: Extracted text content

        Returns:
            Comprehensive metadata dictionary
        """
        return {
            "basic": self.extract_basic_metadata(file_path, content),
            "structure": self.extract_structure(content),
            "entities": self.extract_entities(content),
            "topics": self.extract_topics(content)
        }

# Test
if __name__ == "__main__":
    import pdfplumber

    # Read content from sample PDF for metadata testing
    sample_pdf = "docs/workshops/S7-sample-docs/simple_security_policy.pdf"

    with pdfplumber.open(sample_pdf) as pdf:
        test_content = "\n\n".join(page.extract_text() or "" for page in pdf.pages)

    # Extract metadata
    extractor = MetadataExtractor()
    metadata = extractor.extract_all(sample_pdf, test_content)

    print("=" * 60)
    print("METADATA EXTRACTION TEST")
    print("=" * 60)

    print("\n📄 Basic Metadata:")
    for key, value in metadata["basic"].items():
        print(f"  {key}: {value}")

    print("\n📋 Structure:")
    for key, value in metadata["structure"].items():
        if key != "headings":
            print(f"  {key}: {value}")

    print("\n  Headings:")
    for heading in metadata["structure"]["headings"]:
        print(f"    {'#' * heading['level']} {heading['text']} (line {heading['line_number']})")

    print("\n🏷️  Entities:")
    for key, value in metadata["entities"].items():
        if value:
            print(f"  {key}: {value}")

    print("\n🎯 Topics:")
    print(f"  Suggested Topics: {metadata['topics']['suggested_topics']}")
    print(f"  Suggested Tags: {metadata['topics']['suggested_tags']}")
```

**Step 2: Test Metadata Extraction (5 mins)**

```bash
python src/documind/extractors/metadata_extractor.py
```

**Expected Output:**
```
============================================================
METADATA EXTRACTION TEST
============================================================

📄 Basic Metadata:
  file_name: simple_security_policy.pdf
  file_path: docs/workshops/S7-sample-docs/simple_security_policy.pdf
  file_size_bytes: ...
  file_type: .pdf
  created_at: ...
  modified_at: ...
  word_count: ~100
  character_count: ~700
  line_count: ~30
  estimated_read_time_minutes: 1

📋 Structure:
  heading_count: 5
  section_count: 6
  list_items: 7
  numbered_lists: 4
  has_tables: False

  Headings:
    # Company Security Policy (line 2)
    ## Overview (line 6)
    ## Password Requirements (line 10)
    ## Two-Factor Authentication (line 19)
    ## Incident Reporting (line 27)

🏷️  Entities:
  emails: ['email@company.com', 'security@company.com']
  urls: ['https://vpn.company.com', 'https://security.company.com/report']
  dates: ['November 24, 2025', '2025-11-24']
  phone_numbers: ['555-123-4567']
  entity_count: 8

🎯 Topics:
  Suggested Topics: ['security', 'hr']
  Suggested Tags: ['security', 'password', 'policy', 'company', 'email']
```

> **Note:** Your output will reflect the actual content of the `simple_security_policy.pdf` sample document.

---

### Quiz 3:

**Question 1:** Why is preserving document structure important for RAG systems?\
   a) Structure makes files smaller\
   b) It's required by all databases\
   c) Structure helps LLMs understand context hierarchy and improves retrieval accuracy through semantic chunking and source attribution\
   d) Structure doesn't matter for RAG

**Question 2:** What types of entities should you extract from documents?\
   a) Only email addresses\
   b) Emails, URLs, dates, phone numbers, and other concrete identifiers that provide context and enable verification\
   c) Only numbers\
   d) Entities are not useful

**Question 3:** How can topic extraction improve document searchability?\
   a) Topics make documents longer\
   b) Topics are only for decoration\
   c) Topic extraction slows down search\
   d) Topics provide semantic categories that enable better filtering, faceted search, and relevance ranking in RAG retrieval

**Answers:**
1. **c)** Structure preservation enhances LLM understanding, chunking quality, and source attribution in RAG
2. **b)** Extract concrete identifiers (emails, URLs, dates, phones) for context and verification
3. **d)** Topics enable semantic categorization, filtering, and improved relevance in RAG retrieval

---

## Module 4: Challenge Project - Multi-Format Document Processor (15 minutes)

### Challenge Overview

Build a production-ready document processing system that handles PDF, DOCX, CSV, and TXT files with full metadata extraction, **embedding generation**, and LLM-optimized formatting.

**Your Mission:**
Create a unified document processor that:
1. Automatically detects file format
2. Extracts text and tables appropriately
3. Enriches with comprehensive metadata
4. Formats output for optimal LLM consumption
5. **Generates vector embeddings for RAG search**
6. Integrates with DocuMind database

---

### ⚠️ CRITICAL: Understanding the Database Architecture

> **Why Q&A might not find your documents:**
> DocuMind uses **TWO tables** for document storage. If you only upload to one, RAG search won't work!

**Database Schema:**

| Table | Purpose | Required for Q&A? |
|-------|---------|-------------------|
| `documents` | Stores full document content, title, metadata | ❌ Text search only |
| `document_chunks` | Stores chunks **WITH EMBEDDINGS** (1536-dim vectors) | ✅ **Required for RAG** |

**The RAG search pipeline:**
```
User Query → OpenAI Embedding → Vector Similarity Search (document_chunks) → Retrieved Context → LLM Answer
```

**If you only insert into `documents` table:**
- ❌ `production_qa.py` won't find your documents
- ❌ Semantic search returns zero results
- ❌ Q&A says "I don't have enough information"

**Correct upload flow:**
```
Document → Process → Chunk → Generate Embeddings → Insert documents + Insert document_chunks
```

---

### Challenge Requirements

**Feature:** Universal Document Processor with Embeddings

**What to Build:**

1. **Unified Processor Class**
   - Single entry point: `process_document(file_path)`
   - Auto-detects format from extension
   - Routes to appropriate extractor
   - Returns standardized output format with chunks

2. **Metadata Enrichment**
   - Extract all metadata types (basic, structure, entities, topics)
   - Generate document fingerprint (SHA-256 hash)
   - Create searchable tags automatically
   - Assign document category

3. **LLM Optimization**
   - Format text in Markdown with YAML frontmatter
   - Preserve structure (headings, lists, tables)
   - Clean and normalize text
   - Optimal chunk boundaries (500-1000 words, respect sections)

4. **🔑 Embedding Generation (REQUIRED FOR RAG)**
   - Use OpenAI `text-embedding-3-small` model (1536 dimensions)
   - Generate embedding for **each chunk**
   - Batch embeddings for efficiency (50-100 at a time)
   - Handle rate limits with retry logic

5. **Database Integration (TWO TABLES)**
   - Insert document record into `documents` table
   - Insert chunks with embeddings into `document_chunks` table
   - Store metadata in JSONB columns
   - Link chunks to document via `document_id` foreign key

6. **CLI Interface for Document Upload**
   - Command-line interface: `python -m src.documind.cli.upload_cli <files>`
   - Support single file upload: `file.pdf`
   - Support batch upload: `--dir path/to/folder/`
   - Support `--no-embeddings` flag for fast uploads (skips RAG)
   - Show progress with embedding count: `[N emb]`
   - Display upload summary (files processed, chunks created, errors)

---

### CLI Usage Examples

After implementation, your CLI should work like this:

```bash
# Upload a single PDF with embeddings
python -m src.documind.cli.upload_cli docs/workshops/S7-sample-docs/employee_directory.pdf

# Upload multiple files
python -m src.documind.cli.upload_cli file1.pdf file2.docx file3.csv

# Batch upload all documents in a folder
python -m src.documind.cli.upload_cli --dir docs/workshops/S7-sample-docs/

# Upload with more parallel workers (faster)
python -m src.documind.cli.upload_cli --dir docs/workshops/S7-sample-docs/ -w 8

# Fast upload WITHOUT embeddings (documents won't be searchable via RAG)
python -m src.documind.cli.upload_cli --no-embeddings --dir docs/

# Dry run - process without uploading (test extraction)
python -m src.documind.cli.upload_cli --dry-run --dir docs/workshops/S7-sample-docs/

# JSON output for scripting
python -m src.documind.cli.upload_cli --json --dir docs/workshops/S7-sample-docs/

# Show help
python -m src.documind.cli.upload_cli --help
```

**Expected Output (with embeddings):**
```
📤 DocuMind Upload CLI
==================================================
Files to process: 8
Workers: 4
Embeddings: enabled

  ✓ [1/8] employee_directory.pdf                   pdf      92 words [1 emb]
  ✓ [2/8] employee_data.csv                        csv     201 words [2 emb]
  ✓ [3/8] employee_handbook.docx                   docx    466 words [1 emb]
  ✓ [4/8] meeting_notes.docx                       docx    135 words [1 emb]
  ✓ [5/8] invoice_acme.pdf                         pdf     158 words [1 emb]
  ✓ [6/8] product_catalog.csv                      csv     241 words [1 emb]
  ✓ [7/8] quarterly_report.csv                     csv     143 words [1 emb]
  ✓ [8/8] simple_security_policy.pdf               pdf      93 words [1 emb]

==================================================
Summary:
  ✓ Success: 8
  ⏱ Time:    6.09s (1.3 docs/sec)
```

> **Note:** The `[N emb]` shows how many chunks with embeddings were stored in `document_chunks` table.

---

### Testing with Interactive Q&A

After uploading documents, test the RAG system built in Session 6:

```bash
python src/documind/rag/production_qa.py --interactive
```

**Test Questions:**
```
> Who works in engineering?
> How do I reach Alice Johnson?
> What is the vacation policy?
> What are the password requirements?
```

The Q&A system should find relevant chunks from your uploaded documents and provide accurate answers with source citations.

---

### Starter Code

Create `src/documind/processor.py`:

```python
"""
Unified Document Processor
Handles all document formats with metadata extraction
"""
from pathlib import Path
from typing import Dict, Optional, List
import hashlib
import json
import os

from extractors.pdf_extractor import PDFExtractor
from extractors.docx_extractor import DocxExtractor
from extractors.spreadsheet_extractor import SpreadsheetExtractor
from extractors.metadata_extractor import MetadataExtractor

class DocumentProcessor:
    """
    Unified processor for all document formats.
    Extracts text, tables, metadata, and formats for LLM consumption.
    """

    def __init__(self):
        self.pdf_extractor = PDFExtractor()
        self.docx_extractor = DocxExtractor()
        self.spreadsheet_extractor = SpreadsheetExtractor()
        self.metadata_extractor = MetadataExtractor()

    def process_document(self, file_path: str) -> Dict:
        """
        Process any supported document format.

        Args:
            file_path: Path to document

        Returns:
            Standardized dictionary with text, metadata, and formatting
        """
        path = Path(file_path)
        file_type = path.suffix.lower()

        # TODO: Route to appropriate extractor based on file type
        # TODO: Extract text and tables
        # TODO: Extract metadata
        # TODO: Format for LLM
        # TODO: Generate document fingerprint
        # TODO: Return standardized result with chunks

        pass

    def generate_fingerprint(self, content: str) -> str:
        """Generate SHA-256 hash of content for duplicate detection"""
        return hashlib.sha256(content.encode()).hexdigest()

    def chunk_content(self, content: str, chunk_size: int = 500) -> list:
        """
        Intelligently chunk content respecting structure.

        Args:
            content: Full document text
            chunk_size: Target words per chunk

        Returns:
            List of text chunks with metadata
        """
        # TODO: Split on section boundaries when possible
        # TODO: Create overlapping chunks (10% overlap)
        # TODO: Respect sentence boundaries
        # TODO: Add metadata to each chunk (section, page, etc.)

        pass
```

---

### Upload CLI with Embedding Generation

Create `src/documind/cli/upload_cli.py` - this handles the **critical embedding generation**:

```python
"""
DocuMind Upload CLI - Fast batch document upload with embeddings.
"""
import os
from typing import List, Dict
from openai import OpenAI
from supabase import create_client

# Lazy-loaded clients
_openai_client = None
_supabase_client = None

def get_openai_client():
    """Get or create OpenAI client."""
    global _openai_client
    if _openai_client is None:
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY not set")
        _openai_client = OpenAI(api_key=api_key)
    return _openai_client

def get_supabase_client():
    """Get or create Supabase client."""
    global _supabase_client
    if _supabase_client is None:
        url = os.getenv("SUPABASE_URL")
        key = os.getenv("SUPABASE_SERVICE_KEY") or os.getenv("SUPABASE_ANON_KEY")
        if not url or not key:
            raise ValueError("SUPABASE_URL and SUPABASE_SERVICE_KEY not set")
        _supabase_client = create_client(url, key)
    return _supabase_client

def generate_embeddings(texts: List[str], batch_size: int = 50) -> List[List[float]]:
    """
    Generate embeddings for texts using OpenAI.

    CRITICAL: These embeddings are REQUIRED for RAG search to work!
    Uses text-embedding-3-small model (1536 dimensions).
    """
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
    """
    Upload document WITH EMBEDDINGS to DocuMind.

    IMPORTANT: Inserts into BOTH tables:
    1. documents - for full document storage
    2. document_chunks - for RAG search (with embeddings)
    """
    client = get_supabase_client()

    # Insert document into documents table
    doc_result = client.table("documents").insert({
        "title": processed_doc.file_name,
        "content": processed_doc.content,
        "file_type": processed_doc.extractor_used,
        "metadata": {
            "fingerprint": processed_doc.metadata.fingerprint,
            "word_count": processed_doc.metadata.basic.word_count,
            "chunks": len(processed_doc.chunks),
        }
    }).execute()

    doc_id = doc_result.data[0].get("id")
    chunks_written = 0

    # CRITICAL: Generate embeddings and store chunks
    if generate_emb and processed_doc.chunks:
        # Extract chunk texts
        chunk_texts = [chunk.content for chunk in processed_doc.chunks]

        # Generate embeddings (1536 dimensions each)
        embeddings = generate_embeddings(chunk_texts)

        # Prepare chunk records with embeddings
        chunk_records = []
        for i, (chunk, embedding) in enumerate(zip(processed_doc.chunks, embeddings)):
            chunk_records.append({
                "document_id": doc_id,
                "content": chunk.content,
                "chunk_index": chunk.chunk_index,
                "embedding": embedding,  # <- THIS IS REQUIRED FOR RAG!
                "word_count": chunk.word_count,
                "metadata": {"document_name": processed_doc.file_name}
            })

        # Insert chunks into document_chunks table
        if chunk_records:
            chunk_result = client.table("document_chunks").insert(chunk_records).execute()
            chunks_written = len(chunk_result.data)

    return {
        "success": True,
        "document_id": doc_id,
        "chunks_written": chunks_written  # This shows as [N emb] in CLI output
    }
```

> **⚠️ Key Point:** The `generate_embeddings()` function and storing to `document_chunks` table are **required** for RAG search to work. Without embeddings, `production_qa.py` will return "I don't have enough information."

---

### Your Task

**Step 1: Implement the Document Processor (10 mins)**

Complete the implementation in `src/documind/processor.py`:

1. **`process_document` method**:
   - Detect file type from extension
   - Route to correct extractor (PDF/DOCX/CSV/TXT)
   - Extract text and tables
   - Call `metadata_extractor.extract_all()`
   - Format for LLM consumption (Markdown)
   - Generate fingerprint
   - Return standardized result

2. **`chunk_content` method**:
   - Split text into ~500-word chunks
   - Respect section boundaries (split on headings)
   - Add 10% overlap between chunks
   - Attach metadata (section name, chunk index)

3. **`upload_to_documind` method**:
   - Use documind MCP `upload_document` tool
   - Pass enriched metadata
   - Return document ID

**Step 2: Test with Multiple Formats (5 mins)**

Create `scripts/test_processor.py`:

```python
"""Test the unified document processor"""
from src.documind.processor import DocumentProcessor

processor = DocumentProcessor()

# Test files (use the sample documents)
test_files = [
    "docs/workshops/S7-sample-docs/simple_security_policy.pdf",
    "docs/workshops/S7-sample-docs/meeting_notes.docx",
    "docs/workshops/S7-sample-docs/employee_data.csv",
    "docs/workshops/S7-sample-docs/employee_handbook.docx"
]

for file_path in test_files:
    print(f"\n{'='*60}")
    print(f"Processing: {file_path}")
    print('='*60)

    result = processor.process_document(file_path)

    if result["success"]:
        print(f"✓ Extracted {result['metadata']['basic']['word_count']} words")
        print(f"✓ Found {result['metadata']['structure']['heading_count']} headings")
        print(f"✓ Detected topics: {result['metadata']['topics']['suggested_topics']}")
        print(f"✓ Document fingerprint: {result['fingerprint'][:16]}...")

        # Upload to DocuMind
        upload_result = processor.upload_to_documind(result)
        print(f"✓ Uploaded to DocuMind: {upload_result['document_id']}")
    else:
        print(f"✗ Error: {result['error']}")
```

---

### Success Criteria

Your implementation is complete when:

**Document Processing:**
- [ ] `process_document` correctly handles PDF, DOCX, CSV, and TXT files
- [ ] All metadata types are extracted (basic, structure, entities, topics)
- [ ] Text is formatted in clean Markdown for LLM consumption
- [ ] Tables are preserved in Markdown format
- [ ] Document fingerprint is generated for duplicate detection
- [ ] `chunk_content` creates sensible chunks respecting structure
- [ ] Chunks have 10% overlap for context continuity

**🔑 Embedding Generation (CRITICAL):**
- [ ] OpenAI API key is configured (`OPENAI_API_KEY` env var)
- [ ] `generate_embeddings()` uses `text-embedding-3-small` model
- [ ] Each chunk gets a 1536-dimension embedding vector
- [ ] CLI output shows `[N emb]` for each uploaded document
- [ ] Embeddings are batch-processed (50-100 per API call)

**Database Integration (TWO TABLES):**
- [ ] Documents inserted into `documents` table with metadata
- [ ] Chunks WITH embeddings inserted into `document_chunks` table
- [ ] `document_id` foreign key links chunks to parent document
- [ ] Verify data exists: `SELECT COUNT(*) FROM document_chunks WHERE embedding IS NOT NULL`

**CLI Interface:**
- [ ] CLI supports single file upload: `python -m src.documind.cli.upload_cli file.pdf`
- [ ] CLI supports batch upload: `--dir path/to/folder/`
- [ ] CLI shows progress with embedding count: `[N emb]`
- [ ] CLI has `--no-embeddings` flag (faster but no RAG)
- [ ] CLI shows summary (success count, time, docs/sec)

**End-to-End Q&A Testing (THE REAL TEST):**
- [ ] Upload sample documents via CLI with embeddings enabled
- [ ] Verify: CLI shows `[N emb]` counts > 0 for each file
- [ ] Run `python src/documind/rag/production_qa.py --interactive`
- [ ] Ask: "Who works in engineering?" → Get actual employee names
- [ ] Ask: "How do I reach Alice Johnson?" → Get contact info
- [ ] Ask: "What are the password requirements?" → Get security policy details

> **⚠️ If Q&A returns "I don't have enough information":**
> 1. Check that CLI showed `[N emb]` counts (not 0 or missing)
> 2. Verify `OPENAI_API_KEY` is set
> 3. Check `document_chunks` table has records with non-null embeddings
> 4. Re-upload with embeddings: `python -m src.documind.cli.upload_cli --dir docs/workshops/S7-sample-docs/`

**Bonus Challenges:**
- Add OCR support for scanned PDFs using Tesseract
- Implement duplicate detection (check fingerprint before upload)
- Add support for PowerPoint (.pptx) files
- Add progress bars for large file processing
- Add retry logic for OpenAI rate limits

---

## Answer Key

### Exercise 1.1 & 1.2 Solution

See the complete `pdf_extractor.py` code provided in Module 1.

**Key Functions:**
- `extract_text()`: Extracts all text from PDF pages
- `extract_tables()`: Detects and extracts tables with structure
- `tables_to_markdown()`: Converts tables to Markdown format

---

### Exercise 2.1 & 2.2 Solution

See complete extractor code in Module 2 for:
- `docx_extractor.py`: Word document extraction
- `spreadsheet_extractor.py`: Excel/CSV extraction

---

### Exercise 3.1 Solution

See complete `metadata_extractor.py` in Module 3.

**Key Methods:**
- `extract_basic_metadata()`: File stats and content statistics
- `extract_structure()`: Headings, sections, lists
- `extract_entities()`: Emails, URLs, dates, phones
- `extract_topics()`: Keyword-based topic detection

---

### Module 4 Challenge Solution

**Complete `processor.py` implementation:**

```python
def process_document(self, file_path: str) -> Dict:
    """Process any supported document format"""
    path = Path(file_path)
    file_type = path.suffix.lower()

    try:
        # Extract text based on file type
        if file_type == ".pdf":
            extract_result = self.pdf_extractor.extract_text(file_path)
            text = extract_result.get("text", "")
            tables_result = self.pdf_extractor.extract_tables(file_path)
            tables_md = self.pdf_extractor.tables_to_markdown(tables_result)
            full_text = text + "\n\n" + tables_md

        elif file_type in [".docx", ".doc"]:
            full_text = self.docx_extractor.format_for_llm(file_path)

        elif file_type in [".xlsx", ".xls", ".csv"]:
            full_text = self.spreadsheet_extractor.format_for_llm(file_path)

        elif file_type in [".txt", ".md"]:
            with open(file_path, 'r', encoding='utf-8') as f:
                full_text = f.read()

        else:
            return {
                "success": False,
                "error": f"Unsupported file type: {file_type}"
            }

        # Extract metadata
        metadata = self.metadata_extractor.extract_all(file_path, full_text)

        # Generate fingerprint
        fingerprint = self.generate_fingerprint(full_text)

        # Chunk content
        chunks = self.chunk_content(full_text)

        return {
            "success": True,
            "file_path": str(path.absolute()),
            "file_type": file_type,
            "text": full_text,
            "chunks": chunks,
            "metadata": metadata,
            "fingerprint": fingerprint
        }

    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "message": f"Failed to process {file_path}"
        }

def chunk_content(self, content: str, chunk_size: int = 500) -> list:
    """Intelligently chunk content"""
    import re

    # Split on double newlines (paragraph boundaries)
    paragraphs = content.split('\n\n')

    chunks = []
    current_chunk = []
    current_word_count = 0

    for para in paragraphs:
        para_words = para.split()
        para_word_count = len(para_words)

        if current_word_count + para_word_count > chunk_size and current_chunk:
            # Save current chunk
            chunk_text = '\n\n'.join(current_chunk)
            chunks.append({
                "chunk_index": len(chunks),
                "text": chunk_text,
                "word_count": current_word_count
            })

            # Start new chunk with 10% overlap (last paragraph)
            current_chunk = [current_chunk[-1], para]
            current_word_count = len(current_chunk[-2].split()) + para_word_count
        else:
            current_chunk.append(para)
            current_word_count += para_word_count

    # Add final chunk
    if current_chunk:
        chunk_text = '\n\n'.join(current_chunk)
        chunks.append({
            "chunk_index": len(chunks),
            "text": chunk_text,
            "word_count": current_word_count
        })

    return chunks

def upload_to_documind(self, processed_doc: Dict) -> Dict:
    """Upload via MCP"""
    # In Claude Code, use documind MCP:
    # mcp.documind.upload_document(
    #     title=processed_doc['metadata']['basic']['file_name'],
    #     content=processed_doc['text'],
    #     file_type=processed_doc['file_type'],
    #     metadata={
    #         **processed_doc['metadata'],
    #         'fingerprint': processed_doc['fingerprint'],
    #         'chunk_count': len(processed_doc['chunks'])
    #     }
    # )
    pass
```

---

## Key Takeaways

By completing this workshop, you've learned:

1. **PDF extraction is complex** - Use tools like pdfplumber for tables and layout
2. **Multi-format support is essential** - Different tools for different formats
3. **Metadata enriches RAG** - Structure, entities, topics improve retrieval
4. **LLM-optimized formatting matters** - Markdown preserves structure for AI consumption
5. **Intelligent chunking respects structure** - Split on sections, overlap for context
6. **🔑 Embedding generation is REQUIRED for RAG** - Without embeddings in `document_chunks`, Q&A won't find documents
7. **Two-table architecture** - `documents` for storage, `document_chunks` with embeddings for search

**The Complete Processing Pipeline:**
```
Document → Format Detection → Specialized Extractor → Metadata Enrichment → LLM Formatting → Chunking → Generate Embeddings → Insert documents + Insert document_chunks
```

**Why RAG Needs Embeddings:**
```
User Question → OpenAI Embedding → Vector Similarity Search (document_chunks) → Top-K Chunks → LLM Answer
```

> **Remember:** If your Q&A system returns "I don't have enough information," the problem is almost always missing embeddings. Use the upload CLI with embeddings enabled!

---

## Next Session Preview

In **Session 8: Vector Databases**, we'll:
- Enable pgvector extension in Supabase
- Generate embeddings for all document chunks
- Implement hybrid search (semantic + keyword)
- Optimize with HNSW indexes
- Benchmark vector search performance
- Integrate optimized search into DocuMind

**Preparation:**
1. Process 10-20 documents with your new processor
2. Store them in DocuMind database
3. Review vector embeddings concepts
4. Ensure OpenAI API key is configured

See you in Session 8!

---

**Workshop Complete! 🎉**

You've built a production-ready multi-format document processor with metadata extraction and LLM optimization. DocuMind can now ingest any document type!
