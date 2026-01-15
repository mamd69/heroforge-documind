"""
TDD Tests for Document API Routes

Following Red-Green-Refactor cycle:
1. RED: Write failing tests first
2. GREEN: Implement minimal code to pass
3. REFACTOR: Clean up and integrate with real services

Run with: pytest tests/api/test_document_routes.py -v
"""

import pytest
from fastapi.testclient import TestClient
from io import BytesIO
import uuid


class TestUploadDocument:
    """Tests for POST /api/documents"""

    def test_upload_pdf_returns_201(self, client, sample_pdf):
        """Uploading a PDF returns 201 status"""
        files = {"file": ("test.pdf", sample_pdf, "application/pdf")}

        response = client.post("/api/documents", files=files)

        assert response.status_code == 201

    def test_upload_returns_document_id(self, client, sample_pdf):
        """Upload response includes document_id"""
        files = {"file": ("test.pdf", sample_pdf, "application/pdf")}

        response = client.post("/api/documents", files=files)
        data = response.json()

        assert "document_id" in data
        uuid.UUID(data["document_id"])

    def test_upload_returns_chunks_created(self, client, sample_pdf):
        """Upload response includes chunk count"""
        files = {"file": ("test.pdf", sample_pdf, "application/pdf")}

        response = client.post("/api/documents", files=files)
        data = response.json()

        assert "chunks_created" in data
        assert isinstance(data["chunks_created"], int)
        assert data["chunks_created"] >= 0

    def test_upload_returns_filename(self, client, sample_pdf):
        """Upload response includes original filename"""
        files = {"file": ("my_document.pdf", sample_pdf, "application/pdf")}

        response = client.post("/api/documents", files=files)
        data = response.json()

        assert data["filename"] == "my_document.pdf"

    def test_upload_docx(self, client, sample_docx):
        """DOCX files are accepted"""
        files = {"file": ("test.docx", sample_docx, "application/vnd.openxmlformats-officedocument.wordprocessingml.document")}

        response = client.post("/api/documents", files=files)

        assert response.status_code == 201

    def test_upload_txt(self, client):
        """Plain text files are accepted"""
        content = BytesIO(b"This is plain text content for testing.")
        files = {"file": ("test.txt", content, "text/plain")}

        response = client.post("/api/documents", files=files)

        assert response.status_code == 201

    def test_upload_csv(self, client):
        """CSV files are accepted"""
        csv_content = BytesIO(b"name,age,city\nAlice,30,NYC\nBob,25,LA")
        files = {"file": ("data.csv", csv_content, "text/csv")}

        response = client.post("/api/documents", files=files)

        assert response.status_code == 201

    def test_upload_xlsx(self, client, sample_xlsx):
        """Excel files are accepted"""
        files = {"file": ("data.xlsx", sample_xlsx, "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")}

        response = client.post("/api/documents", files=files)

        assert response.status_code == 201

    def test_upload_markdown(self, client):
        """Markdown files are accepted"""
        md_content = BytesIO(b"# Title\n\nThis is **markdown** content.")
        files = {"file": ("readme.md", md_content, "text/markdown")}

        response = client.post("/api/documents", files=files)

        assert response.status_code == 201

    def test_upload_unsupported_format_returns_400(self, client):
        """Unsupported file formats return 400"""
        exe_content = BytesIO(b"MZ\x90\x00")  # EXE header
        files = {"file": ("malware.exe", exe_content, "application/octet-stream")}

        response = client.post("/api/documents", files=files)

        assert response.status_code == 400
        assert "unsupported" in response.json()["detail"].lower()

    def test_upload_too_large_returns_413(self, client):
        """Files over 50MB return 413 Payload Too Large"""
        large_content = BytesIO(b"x" * (51 * 1024 * 1024))  # 51MB
        files = {"file": ("huge.pdf", large_content, "application/pdf")}

        response = client.post("/api/documents", files=files)

        assert response.status_code == 413

    def test_upload_no_file_returns_422(self, client):
        """Missing file returns 422"""
        response = client.post("/api/documents")

        assert response.status_code == 422

    def test_upload_duplicate_returns_status(self, client, sample_pdf):
        """Uploading duplicate returns duplicate status"""
        files = {"file": ("test.pdf", sample_pdf, "application/pdf")}

        # Upload first time
        client.post("/api/documents", files=files)

        # Upload same file again
        sample_pdf.seek(0)
        files = {"file": ("test.pdf", sample_pdf, "application/pdf")}
        response = client.post("/api/documents", files=files)

        data = response.json()
        assert data.get("status") in ["duplicate", "success"]
        if data["status"] == "duplicate":
            assert "existing_id" in data


class TestListDocuments:
    """Tests for GET /api/documents"""

    def test_list_documents_returns_200(self, client):
        """Listing documents returns 200"""
        response = client.get("/api/documents")

        assert response.status_code == 200

    def test_list_documents_returns_items_array(self, client):
        """Response includes items array"""
        response = client.get("/api/documents")
        data = response.json()

        assert "items" in data
        assert isinstance(data["items"], list)

    def test_list_documents_returns_total(self, client):
        """Response includes total count"""
        response = client.get("/api/documents")
        data = response.json()

        assert "total" in data
        assert isinstance(data["total"], int)

    def test_list_documents_returns_pagination_info(self, client):
        """Response includes page and limit"""
        response = client.get("/api/documents")
        data = response.json()

        assert "page" in data
        assert "limit" in data

    def test_list_documents_includes_uploaded(self, client, uploaded_document):
        """Listed documents include ones we uploaded"""
        response = client.get("/api/documents")
        data = response.json()

        doc_ids = [d["id"] for d in data["items"]]
        assert uploaded_document["document_id"] in doc_ids

    def test_list_documents_pagination(self, client):
        """Pagination parameters work correctly"""
        response = client.get("/api/documents?page=1&limit=5")
        data = response.json()

        assert data["page"] == 1
        assert len(data["items"]) <= 5

    def test_list_documents_filter_by_type(self, client):
        """Can filter documents by file type"""
        response = client.get("/api/documents?file_type=pdf")

        assert response.status_code == 200


class TestGetDocument:
    """Tests for GET /api/documents/{id}"""

    def test_get_document_returns_200(self, client, uploaded_document):
        """Getting a document returns 200"""
        doc_id = uploaded_document["document_id"]

        response = client.get(f"/api/documents/{doc_id}")

        assert response.status_code == 200

    def test_get_document_returns_details(self, client, uploaded_document):
        """Document details include required fields"""
        doc_id = uploaded_document["document_id"]

        response = client.get(f"/api/documents/{doc_id}")
        data = response.json()

        assert "id" in data
        assert "title" in data
        assert "file_type" in data
        assert "chunk_count" in data
        assert "created_at" in data

    def test_get_document_includes_metadata(self, client, uploaded_document):
        """Document includes metadata object"""
        doc_id = uploaded_document["document_id"]

        response = client.get(f"/api/documents/{doc_id}")
        data = response.json()

        assert "metadata" in data
        assert isinstance(data["metadata"], dict)

    def test_get_nonexistent_document_returns_404(self, client):
        """Getting non-existent document returns 404"""
        fake_id = str(uuid.uuid4())

        response = client.get(f"/api/documents/{fake_id}")

        assert response.status_code == 404


class TestDeleteDocument:
    """Tests for DELETE /api/documents/{id}"""

    def test_delete_document_returns_204(self, client, uploaded_document):
        """Deleting a document returns 204"""
        doc_id = uploaded_document["document_id"]

        response = client.delete(f"/api/documents/{doc_id}")

        assert response.status_code == 204

    def test_delete_removes_document(self, client, uploaded_document):
        """Deleted document is no longer accessible"""
        doc_id = uploaded_document["document_id"]

        # Delete
        client.delete(f"/api/documents/{doc_id}")

        # Verify gone
        response = client.get(f"/api/documents/{doc_id}")
        assert response.status_code == 404

    def test_delete_nonexistent_returns_404(self, client):
        """Deleting non-existent document returns 404"""
        fake_id = str(uuid.uuid4())

        response = client.delete(f"/api/documents/{fake_id}")

        assert response.status_code == 404


class TestDocumentChunks:
    """Tests for GET /api/documents/{id}/chunks"""

    def test_get_chunks_returns_200(self, client, uploaded_document):
        """Getting chunks returns 200"""
        doc_id = uploaded_document["document_id"]

        response = client.get(f"/api/documents/{doc_id}/chunks")

        assert response.status_code == 200

    def test_get_chunks_returns_array(self, client, uploaded_document):
        """Chunks response is an array"""
        doc_id = uploaded_document["document_id"]

        response = client.get(f"/api/documents/{doc_id}/chunks")
        data = response.json()

        assert isinstance(data, list)

    def test_chunks_have_required_fields(self, client, uploaded_document):
        """Each chunk has content, index, word_count"""
        doc_id = uploaded_document["document_id"]

        response = client.get(f"/api/documents/{doc_id}/chunks")
        chunks = response.json()

        if len(chunks) > 0:
            chunk = chunks[0]
            assert "chunk_id" in chunk
            assert "content" in chunk
            assert "chunk_index" in chunk
            assert "word_count" in chunk


# Fixtures

@pytest.fixture
def client():
    """Create test client for API"""
    from src.documind.api.main import app
    return TestClient(app)


@pytest.fixture
def sample_pdf():
    """Create a minimal PDF for testing"""
    # Minimal PDF structure
    pdf_content = b"""%PDF-1.4
1 0 obj << /Type /Catalog /Pages 2 0 R >> endobj
2 0 obj << /Type /Pages /Kids [3 0 R] /Count 1 >> endobj
3 0 obj << /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Contents 4 0 R >> endobj
4 0 obj << /Length 44 >> stream
BT /F1 12 Tf 100 700 Td (Test Document) Tj ET
endstream endobj
xref
0 5
0000000000 65535 f
0000000009 00000 n
0000000058 00000 n
0000000115 00000 n
0000000214 00000 n
trailer << /Size 5 /Root 1 0 R >>
startxref
306
%%EOF"""
    return BytesIO(pdf_content)


@pytest.fixture
def sample_docx():
    """Create a minimal DOCX for testing"""
    # DOCX is a ZIP file with specific structure
    # For testing, we'll use a pre-built minimal DOCX
    # In real tests, you'd use python-docx to create this
    from io import BytesIO
    import zipfile

    buffer = BytesIO()
    with zipfile.ZipFile(buffer, 'w') as zf:
        # Minimal DOCX structure
        zf.writestr('[Content_Types].xml', '''<?xml version="1.0" encoding="UTF-8"?>
<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">
<Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>
<Default Extension="xml" ContentType="application/xml"/>
<Override PartName="/word/document.xml" ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.document.main+xml"/>
</Types>''')
        zf.writestr('_rels/.rels', '''<?xml version="1.0" encoding="UTF-8"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
<Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" Target="word/document.xml"/>
</Relationships>''')
        zf.writestr('word/document.xml', '''<?xml version="1.0" encoding="UTF-8"?>
<w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main">
<w:body><w:p><w:r><w:t>Test Document Content</w:t></w:r></w:p></w:body>
</w:document>''')

    buffer.seek(0)
    return buffer


@pytest.fixture
def sample_xlsx():
    """Create a minimal XLSX for testing"""
    from io import BytesIO
    import zipfile

    buffer = BytesIO()
    with zipfile.ZipFile(buffer, 'w') as zf:
        # Minimal XLSX structure
        zf.writestr('[Content_Types].xml', '''<?xml version="1.0" encoding="UTF-8"?>
<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">
<Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>
<Default Extension="xml" ContentType="application/xml"/>
<Override PartName="/xl/workbook.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet.main+xml"/>
<Override PartName="/xl/worksheets/sheet1.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.worksheet+xml"/>
</Types>''')
        zf.writestr('_rels/.rels', '''<?xml version="1.0" encoding="UTF-8"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
<Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" Target="xl/workbook.xml"/>
</Relationships>''')
        zf.writestr('xl/workbook.xml', '''<?xml version="1.0" encoding="UTF-8"?>
<workbook xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main">
<sheets><sheet name="Sheet1" sheetId="1" r:id="rId1" xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships"/></sheets>
</workbook>''')
        zf.writestr('xl/worksheets/sheet1.xml', '''<?xml version="1.0" encoding="UTF-8"?>
<worksheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main">
<sheetData><row r="1"><c r="A1" t="inlineStr"><is><t>Test Data</t></is></c></row></sheetData>
</worksheet>''')
        zf.writestr('xl/_rels/workbook.xml.rels', '''<?xml version="1.0" encoding="UTF-8"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
<Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/worksheet" Target="worksheets/sheet1.xml"/>
</Relationships>''')

    buffer.seek(0)
    return buffer


@pytest.fixture
def uploaded_document(client, sample_pdf):
    """Upload a document and return its info"""
    files = {"file": ("test.pdf", sample_pdf, "application/pdf")}
    response = client.post("/api/documents", files=files)
    return response.json()
