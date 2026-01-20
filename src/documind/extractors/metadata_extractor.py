"""
Unified Metadata Extraction
Extracts rich metadata from all document types
Includes document fingerprinting for duplicate detection
"""
from pathlib import Path
from datetime import datetime
from typing import Dict, Optional, List
import re
import hashlib

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

    def generate_fingerprint(self, content: str, normalize: bool = True) -> str:
        """
        Generate SHA-256 fingerprint of document content.

        Args:
            content: Document text content
            normalize: Whether to normalize whitespace before hashing

        Returns:
            Hexadecimal SHA-256 hash string
        """
        if normalize:
            # Normalize: lowercase, remove extra whitespace
            content = ' '.join(content.lower().split())

        return hashlib.sha256(content.encode('utf-8')).hexdigest()

    def extract_all(self, file_path: str, content: str,
                   format_metadata: Optional[Dict] = None) -> Dict:
        """
        Extract all metadata from a document.

        Args:
            file_path: Path to document file
            content: Extracted text content
            format_metadata: Optional format-specific metadata

        Returns:
            Comprehensive metadata dictionary with fingerprint
        """
        return {
            "basic": self.extract_basic_metadata(file_path, content),
            "structure": self.extract_structure(content),
            "entities": self.extract_entities(content),
            "topics": self.extract_topics(content),
            "fingerprint": self.generate_fingerprint(content),
            "format_metadata": format_metadata or {}
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