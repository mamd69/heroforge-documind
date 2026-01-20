"""
LLM-Optimized Content Formatting
Converts extracted content to clean Markdown for LLM consumption
"""
from typing import Dict, List, Any, Optional
import re
from datetime import datetime


class LLMFormatter:
    """Format content for optimal LLM consumption."""

    def __init__(self):
        """Initialize the formatter."""
        self.markdown_chars = ['*', '_', '`', '[', ']', '<', '>', '#', '|', '\\']

    def format_for_llm(self, content: str, tables: List[Dict],
                       metadata: Dict) -> str:
        """
        Convert content to LLM-optimized Markdown.

        Args:
            content: Raw extracted text
            tables: List of table data dictionaries
            metadata: EnrichedMetadata as dict

        Returns:
            Clean Markdown-formatted string
        """
        output = []

        # 1. Add YAML frontmatter
        output.append(self.create_frontmatter(metadata))
        output.append("")  # Blank line after frontmatter

        # 2. Process and clean content
        cleaned = self.clean_content(content)
        output.append(cleaned)

        # 3. Add tables section
        if tables:
            output.append("")  # Blank line before tables
            output.append(self.format_tables(tables))

        return "\n".join(output)

    def create_frontmatter(self, metadata: Dict) -> str:
        """
        Create YAML frontmatter with key metadata.

        Format:
        ---
        title: Document Name
        type: pdf
        topics: [security, hr]
        word_count: 1500
        fingerprint: abc123...
        ---

        Args:
            metadata: EnrichedMetadata as dict

        Returns:
            YAML frontmatter string
        """
        lines = ["---"]

        # Title
        if metadata.get('title'):
            title = self._escape_yaml_string(metadata['title'])
            lines.append(f"title: {title}")

        # Document type
        if metadata.get('doc_type'):
            lines.append(f"type: {metadata['doc_type']}")

        # Topics
        if metadata.get('topics'):
            topics = metadata['topics']
            if topics:
                topics_str = ', '.join(topics)
                lines.append(f"topics: [{topics_str}]")

        # Word count
        if metadata.get('word_count'):
            lines.append(f"word_count: {metadata['word_count']}")

        # Page count
        if metadata.get('page_count'):
            lines.append(f"page_count: {metadata['page_count']}")

        # Language
        if metadata.get('language'):
            lines.append(f"language: {metadata['language']}")

        # Fingerprint
        if metadata.get('fingerprint'):
            lines.append(f"fingerprint: {metadata['fingerprint']}")

        # Processing date
        if metadata.get('processed_at'):
            lines.append(f"processed_at: {metadata['processed_at']}")
        else:
            lines.append(f"processed_at: {datetime.utcnow().isoformat()}Z")

        # Table count
        if metadata.get('table_count'):
            lines.append(f"table_count: {metadata['table_count']}")

        # Entities (if present)
        if metadata.get('entities'):
            entities = metadata['entities']
            if isinstance(entities, dict):
                # Format entities as nested YAML
                lines.append("entities:")
                for entity_type, values in entities.items():
                    if values:
                        if isinstance(values, (list, tuple)):
                            values_str = ', '.join(str(v) for v in values[:5])  # Limit to 5
                            lines.append(f"  {entity_type}: [{values_str}]")
                        else:
                            # Handle scalar values (like entity_count)
                            lines.append(f"  {entity_type}: {values}")

        lines.append("---")
        return "\n".join(lines)

    def format_tables(self, tables: List[Dict]) -> str:
        """
        Convert tables to Markdown format.

        | Header 1 | Header 2 |
        |----------|----------|
        | Data 1   | Data 2   |

        Args:
            tables: List of table data dictionaries with 'headers' and 'rows'

        Returns:
            Markdown formatted tables
        """
        if not tables:
            return ""

        output = []
        output.append("## Tables")
        output.append("")

        for idx, table in enumerate(tables, 1):
            output.append(f"### Table {idx}")
            output.append("")

            headers = table.get('headers', [])
            rows = table.get('rows', [])

            if not headers and not rows:
                output.append("*Empty table*")
                output.append("")
                continue

            # If no headers, create generic ones
            if not headers and rows:
                num_cols = len(rows[0]) if rows else 0
                headers = [f"Column {i+1}" for i in range(num_cols)]

            # Format headers
            header_line = "| " + " | ".join(self._clean_cell(h) for h in headers) + " |"
            separator_line = "|" + "|".join(["-" * (len(h) + 2) for h in headers]) + "|"

            output.append(header_line)
            output.append(separator_line)

            # Format rows
            for row in rows:
                # Ensure row has same number of columns as headers
                padded_row = list(row) + [''] * (len(headers) - len(row))
                padded_row = padded_row[:len(headers)]

                row_line = "| " + " | ".join(self._clean_cell(str(cell)) for cell in padded_row) + " |"
                output.append(row_line)

            output.append("")  # Blank line after table

        return "\n".join(output)

    def clean_content(self, content: str) -> str:
        """
        Clean and normalize content.
        - Remove excessive whitespace
        - Normalize blank lines
        - Detect and format headings
        - Preserve intentional formatting

        Args:
            content: Raw text content

        Returns:
            Cleaned content string
        """
        if not content:
            return ""

        # 1. Normalize line endings
        text = content.replace('\r\n', '\n').replace('\r', '\n')

        # 2. Remove trailing whitespace from each line
        lines = [line.rstrip() for line in text.split('\n')]

        # 3. Detect and format headings
        lines = self._format_headings(lines)

        # 4. Join lines
        text = '\n'.join(lines)

        # 5. Normalize multiple blank lines to single blank line
        text = self.normalize_whitespace(text)

        # 6. Remove leading/trailing whitespace
        text = text.strip()

        return text

    def _format_headings(self, lines: List[str]) -> List[str]:
        """
        Detect and format potential headings.

        Heuristics:
        - Short lines (< 60 chars)
        - ALL CAPS or Title Case
        - Followed by blank line or content
        - Not part of a sentence (no period at end)

        Args:
            lines: List of text lines

        Returns:
            List of lines with formatted headings
        """
        formatted = []
        i = 0

        while i < len(lines):
            line = lines[i]

            # Skip empty lines
            if not line.strip():
                formatted.append(line)
                i += 1
                continue

            # Check if this looks like a heading
            if self._is_heading(line, i, lines):
                # Determine heading level based on position and context
                level = self._determine_heading_level(line, i, lines)
                heading = '#' * level + ' ' + line.strip()
                formatted.append(heading)
            else:
                formatted.append(line)

            i += 1

        return formatted

    def _is_heading(self, line: str, index: int, lines: List[str]) -> bool:
        """
        Determine if a line should be formatted as a heading.

        Args:
            line: The line to check
            index: Index in lines list
            lines: All lines

        Returns:
            True if line should be a heading
        """
        stripped = line.strip()

        # Already a heading
        if stripped.startswith('#'):
            return False

        # Too long for a heading
        if len(stripped) > 100:
            return False

        # Too short
        if len(stripped) < 3:
            return False

        # Ends with punctuation (likely a sentence)
        if stripped.endswith(('.', ',', ';', ':')):
            return False

        # Check if ALL CAPS (at least 50% letters and 80% uppercase)
        letters = [c for c in stripped if c.isalpha()]
        if len(letters) >= len(stripped) * 0.5:
            uppercase_ratio = sum(1 for c in letters if c.isupper()) / len(letters)
            if uppercase_ratio >= 0.8:
                return True

        # Check if Title Case
        words = stripped.split()
        if len(words) >= 2:
            title_case_words = sum(1 for w in words if w and w[0].isupper())
            if title_case_words >= len(words) * 0.7:
                return True

        return False

    def _determine_heading_level(self, line: str, index: int, lines: List[str]) -> int:
        """
        Determine heading level (1-6) based on context.

        Args:
            line: The heading line
            index: Index in lines list
            lines: All lines

        Returns:
            Heading level (1-6)
        """
        stripped = line.strip()

        # If at the beginning of document, likely h1
        if index < 5:
            return 1

        # Check if ALL CAPS (likely higher level)
        if stripped.isupper():
            return 2

        # Check length (shorter = higher level)
        if len(stripped) < 30:
            return 2
        elif len(stripped) < 50:
            return 3
        else:
            return 4

    def normalize_whitespace(self, text: str) -> str:
        """
        Replace multiple blank lines with single blank line.

        Args:
            text: Input text

        Returns:
            Text with normalized whitespace
        """
        # Replace 3 or more newlines with 2 newlines (1 blank line)
        return re.sub(r'\n{3,}', '\n\n', text)

    def escape_markdown(self, text: str, preserve_formatting: bool = True) -> str:
        """
        Escape Markdown special characters in plain text.

        Args:
            text: Text to escape
            preserve_formatting: If True, try to preserve intentional formatting

        Returns:
            Escaped text
        """
        if not text:
            return ""

        if preserve_formatting:
            # Only escape characters that are clearly not intentional formatting
            # This is conservative - only escape when clearly needed
            return text
        else:
            # Escape all Markdown special characters
            for char in self.markdown_chars:
                text = text.replace(char, '\\' + char)
            return text

    def _clean_cell(self, cell: str) -> str:
        """
        Clean a table cell for Markdown formatting.

        Args:
            cell: Cell content

        Returns:
            Cleaned cell content
        """
        if not cell:
            return ""

        # Remove newlines and excessive whitespace
        cleaned = ' '.join(cell.split())

        # Escape pipe characters in cells
        cleaned = cleaned.replace('|', '\\|')

        # Limit cell length
        max_length = 100
        if len(cleaned) > max_length:
            cleaned = cleaned[:max_length-3] + '...'

        return cleaned

    def _escape_yaml_string(self, text: str) -> str:
        """
        Escape a string for YAML frontmatter.

        Args:
            text: Text to escape

        Returns:
            YAML-safe string
        """
        if not text:
            return ""

        # Check if string needs quoting
        needs_quotes = any(c in text for c in [':', '#', '[', ']', '{', '}', '"', "'", '\n'])

        if needs_quotes:
            # Escape double quotes and wrap in quotes
            escaped = text.replace('"', '\\"')
            return f'"{escaped}"'

        return text

    def format_metadata_only(self, metadata: Dict) -> str:
        """
        Create just the YAML frontmatter without content.

        Args:
            metadata: EnrichedMetadata as dict

        Returns:
            YAML frontmatter string
        """
        return self.create_frontmatter(metadata)

    def add_section(self, content: str, section_title: str, section_content: str, level: int = 2) -> str:
        """
        Add a new section to existing Markdown content.

        Args:
            content: Existing Markdown content
            section_title: Title of new section
            section_content: Content of new section
            level: Heading level (default: 2)

        Returns:
            Updated content with new section
        """
        heading = '#' * level + ' ' + section_title
        new_section = f"\n\n{heading}\n\n{section_content}"
        return content + new_section
