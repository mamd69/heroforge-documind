"""
Plain Text and Markdown Extraction
Handles .txt and .md files with encoding detection and fallback.
"""
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import re
import logging

logger = logging.getLogger(__name__)


class TextExtractor:
    """Extract content from plain text and Markdown files."""

    # Encoding priority order
    ENCODINGS = ['utf-8', 'latin-1', 'cp1252', 'iso-8859-1']

    # Markdown patterns
    HEADING_PATTERN = re.compile(r'^(#{1,6})\s+(.+)$', re.MULTILINE)
    CODE_BLOCK_PATTERN = re.compile(r'```(\w+)?\n(.*?)```', re.DOTALL)
    INLINE_CODE_PATTERN = re.compile(r'`([^`]+)`')
    LINK_PATTERN = re.compile(r'\[([^\]]+)\]\(([^\)]+)\)')
    LIST_PATTERN = re.compile(r'^[\s]*[-*+]\s+(.+)$', re.MULTILINE)
    NUMBERED_LIST_PATTERN = re.compile(r'^[\s]*\d+\.\s+(.+)$', re.MULTILINE)

    def extract(self, file_path: str) -> Dict:
        """
        Extract text from a text/markdown file.

        Args:
            file_path: Path to .txt or .md file

        Returns:
            Dictionary with:
            - success: bool
            - text: str
            - metadata: dict (encoding used, line endings, etc.)
            - error: str (if failed)
        """
        path = Path(file_path)

        # Validate file exists
        if not path.exists():
            return {
                'success': False,
                'text': '',
                'metadata': {},
                'error': f'File not found: {file_path}'
            }

        # Validate file extension
        if path.suffix.lower() not in ['.txt', '.md', '.markdown']:
            return {
                'success': False,
                'text': '',
                'metadata': {},
                'error': f'Unsupported file type: {path.suffix}'
            }

        try:
            # Try to read with encoding detection
            text, encoding, line_ending = self._read_with_encoding_detection(str(path))

            if text is None:
                return {
                    'success': False,
                    'text': '',
                    'metadata': {},
                    'error': 'Failed to decode file with any supported encoding'
                }

            # Normalize line endings
            normalized_text = self._normalize_line_endings(text)

            # Build metadata
            metadata = {
                'file_name': path.name,
                'file_path': str(path.absolute()),
                'file_size': path.stat().st_size,
                'encoding': encoding,
                'line_ending': line_ending,
                'line_count': len(normalized_text.split('\n')),
                'char_count': len(normalized_text),
                'word_count': len(normalized_text.split()),
                'file_type': 'markdown' if path.suffix.lower() == '.md' else 'text'
            }

            return {
                'success': True,
                'text': normalized_text,
                'metadata': metadata,
                'error': None
            }

        except Exception as e:
            logger.error(f"Error extracting text from {file_path}: {str(e)}")
            return {
                'success': False,
                'text': '',
                'metadata': {},
                'error': f'Extraction failed: {str(e)}'
            }

    def extract_with_structure(self, file_path: str) -> Dict:
        """
        Extract text preserving structure information.
        For Markdown files, identify headings, code blocks, etc.

        Args:
            file_path: Path to text or markdown file

        Returns:
            Dictionary with basic extraction plus structure info
        """
        # Get basic extraction first
        result = self.extract(file_path)

        if not result['success']:
            return result

        text = result['text']
        file_type = result['metadata'].get('file_type', 'text')

        # Only parse structure for Markdown files
        if file_type == 'markdown':
            structure = self._parse_markdown_structure(text)
            result['structure'] = structure
        else:
            result['structure'] = {
                'type': 'plain_text',
                'paragraphs': self._extract_paragraphs(text)
            }

        return result

    def _read_with_encoding_detection(self, file_path: str) -> Tuple[Optional[str], Optional[str], Optional[str]]:
        """
        Try to read file with multiple encodings.

        Returns:
            Tuple of (text, encoding_used, line_ending_type)
        """
        path = Path(file_path)

        # Try each encoding in priority order
        for encoding in self.ENCODINGS:
            try:
                with open(path, 'r', encoding=encoding) as f:
                    text = f.read()

                # Detect line ending type
                line_ending = self._detect_line_ending(text)

                logger.debug(f"Successfully read {file_path} with {encoding} encoding")
                return text, encoding, line_ending

            except (UnicodeDecodeError, UnicodeError):
                logger.debug(f"Failed to read {file_path} with {encoding} encoding")
                continue
            except Exception as e:
                logger.error(f"Unexpected error reading {file_path} with {encoding}: {str(e)}")
                continue

        # If all encodings fail, try with 'errors=ignore'
        logger.warning(f"All encodings failed for {file_path}, trying with error handling")
        try:
            with open(path, 'r', encoding='utf-8', errors='ignore') as f:
                text = f.read()
            line_ending = self._detect_line_ending(text)
            return text, 'utf-8 (with errors ignored)', line_ending
        except Exception as e:
            logger.error(f"Failed to read {file_path} even with error handling: {str(e)}")
            return None, None, None

    def _detect_line_ending(self, text: str) -> str:
        """
        Detect the line ending type used in the text.

        Returns:
            'CRLF' (Windows), 'LF' (Unix), 'CR' (Old Mac), or 'Mixed'
        """
        has_crlf = '\r\n' in text
        has_lf = '\n' in text
        has_cr = '\r' in text

        if has_crlf and not (text.count('\r') > text.count('\r\n')):
            return 'CRLF'
        elif has_lf and not has_cr:
            return 'LF'
        elif has_cr and not has_lf:
            return 'CR'
        elif has_crlf or (has_lf and has_cr):
            return 'Mixed'
        else:
            return 'None'

    def _normalize_line_endings(self, text: str) -> str:
        """
        Normalize line endings to LF (Unix style).

        Args:
            text: Text with potentially mixed line endings

        Returns:
            Text with normalized line endings
        """
        # Replace CRLF first, then CR
        return text.replace('\r\n', '\n').replace('\r', '\n')

    def _parse_markdown_structure(self, text: str) -> Dict:
        """
        Parse Markdown structure to identify headings, code blocks, etc.

        Args:
            text: Markdown text content

        Returns:
            Dictionary with structure information
        """
        structure = {
            'type': 'markdown',
            'headings': self._extract_headings(text),
            'code_blocks': self._extract_code_blocks(text),
            'links': self._extract_links(text),
            'lists': self._extract_lists(text),
            'paragraphs': self._extract_paragraphs(text)
        }

        return structure

    def _extract_headings(self, text: str) -> List[Dict]:
        """Extract all headings with their levels and positions."""
        headings = []

        for match in self.HEADING_PATTERN.finditer(text):
            level = len(match.group(1))  # Count # characters
            title = match.group(2).strip()
            position = match.start()

            headings.append({
                'level': level,
                'title': title,
                'position': position,
                'line': text[:position].count('\n') + 1
            })

        return headings

    def _extract_code_blocks(self, text: str) -> List[Dict]:
        """Extract code blocks with language hints."""
        code_blocks = []

        for match in self.CODE_BLOCK_PATTERN.finditer(text):
            language = match.group(1) or 'plain'
            code = match.group(2).strip()
            position = match.start()

            code_blocks.append({
                'language': language,
                'code': code,
                'position': position,
                'line': text[:position].count('\n') + 1,
                'length': len(code)
            })

        return code_blocks

    def _extract_links(self, text: str) -> List[Dict]:
        """Extract all markdown links."""
        links = []

        for match in self.LINK_PATTERN.finditer(text):
            text_part = match.group(1)
            url = match.group(2)
            position = match.start()

            links.append({
                'text': text_part,
                'url': url,
                'position': position,
                'line': text[:position].count('\n') + 1
            })

        return links

    def _extract_lists(self, text: str) -> List[Dict]:
        """Extract both bulleted and numbered lists."""
        lists = []

        # Bulleted lists
        for match in self.LIST_PATTERN.finditer(text):
            item = match.group(1).strip()
            position = match.start()

            lists.append({
                'type': 'bullet',
                'item': item,
                'position': position,
                'line': text[:position].count('\n') + 1
            })

        # Numbered lists
        for match in self.NUMBERED_LIST_PATTERN.finditer(text):
            item = match.group(1).strip()
            position = match.start()

            lists.append({
                'type': 'numbered',
                'item': item,
                'position': position,
                'line': text[:position].count('\n') + 1
            })

        # Sort by position
        lists.sort(key=lambda x: x['position'])

        return lists

    def _extract_paragraphs(self, text: str) -> List[Dict]:
        """
        Extract paragraphs from text.
        A paragraph is defined as text separated by blank lines.
        """
        paragraphs = []

        # Split by double newlines (blank lines)
        parts = re.split(r'\n\s*\n', text)

        position = 0
        for part in parts:
            part = part.strip()
            if part:
                # Find actual position in original text
                actual_pos = text.find(part, position)
                if actual_pos != -1:
                    position = actual_pos

                paragraphs.append({
                    'text': part,
                    'position': position,
                    'line': text[:position].count('\n') + 1,
                    'word_count': len(part.split()),
                    'char_count': len(part)
                })

                position += len(part)

        return paragraphs


# Convenience function for simple extraction
def extract_text(file_path: str) -> Dict:
    """
    Quick extraction function.

    Args:
        file_path: Path to text or markdown file

    Returns:
        Extraction result dictionary
    """
    extractor = TextExtractor()
    return extractor.extract(file_path)


def extract_text_with_structure(file_path: str) -> Dict:
    """
    Extract text with structure parsing.

    Args:
        file_path: Path to text or markdown file

    Returns:
        Extraction result with structure information
    """
    extractor = TextExtractor()
    return extractor.extract_with_structure(file_path)
