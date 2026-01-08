"""
DocuMind Custom MCP Server
Provides document management tools for Claude Code

Compatible with MCP SDK v2.x (uses @server.list_tools / @server.call_tool)
"""

import os
import json
import asyncio
from datetime import datetime
from dotenv import load_dotenv
from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import Tool, TextContent

import httpx

# Load environment variables
load_dotenv()

# Initialize MCP server
server = Server("documind-mcp")

# Supabase configuration
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_SERVICE_KEY")


def get_supabase_client():
    """Get configured Supabase HTTP client"""
    return httpx.Client(
        base_url=f"{SUPABASE_URL}/rest/v1",
        headers={
            "apikey": SUPABASE_KEY,
            "Authorization": f"Bearer {SUPABASE_KEY}",
            "Content-Type": "application/json",
            "Prefer": "return=representation",
        },
    )


# ============================================================
# Tool Definitions - Tell Claude what tools are available
# ============================================================


@server.list_tools()
async def list_tools() -> list[Tool]:
    """Return list of available tools"""
    return [
        Tool(
            name="upload_document",
            description="Upload a document to the DocuMind knowledge base",
            inputSchema={
                "type": "object",
                "properties": {
                    "title": {"type": "string", "description": "Document title"},
                    "content": {
                        "type": "string",
                        "description": "Full document content",
                    },
                    "file_type": {
                        "type": "string",
                        "description": "Type of document (txt, pdf, docx, etc.)",
                        "default": "txt",
                    },
                    "metadata": {
                        "type": "object",
                        "description": "Optional metadata dictionary",
                    },
                },
                "required": ["title", "content"],
            },
        ),
        Tool(
            name="search_documents",
            description="Search documents by title or content",
            inputSchema={
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Search query string"},
                    "limit": {
                        "type": "integer",
                        "description": "Maximum number of results",
                        "default": 5,
                    },
                    "file_type": {
                        "type": "string",
                        "description": "Optional filter by file type",
                    },
                },
                "required": ["query"],
            },
        ),
        Tool(
            name="get_document",
            description="Retrieve a specific document by ID",
            inputSchema={
                "type": "object",
                "properties": {
                    "document_id": {
                        "type": "string",
                        "description": "UUID of the document",
                    }
                },
                "required": ["document_id"],
            },
        ),
        Tool(
            name="delete_document",
            description="Delete a document from the knowledge base",
            inputSchema={
                "type": "object",
                "properties": {
                    "document_id": {
                        "type": "string",
                        "description": "UUID of the document to delete",
                    }
                },
                "required": ["document_id"],
            },
        ),
        Tool(
            name="update_document",
            description="Update document metadata (enrichment, tags, summary, etc.)",
            inputSchema={
                "type": "object",
                "properties": {
                    "document_id": {
                        "type": "string",
                        "description": "UUID of document to update",
                    },
                    "metadata": {
                        "type": "object",
                        "description": "Dictionary of metadata to merge with existing",
                    },
                },
                "required": ["document_id", "metadata"],
            },
        ),
    ]


# ============================================================
# Tool Implementations - Handle tool calls from Claude
# ============================================================


@server.call_tool()
async def call_tool(name: str, arguments: dict) -> list[TextContent]:
    """Handle tool calls"""

    if name == "upload_document":
        result = upload_document(
            title=arguments["title"],
            content=arguments["content"],
            file_type=arguments.get("file_type", "txt"),
            metadata=arguments.get("metadata"),
        )
    elif name == "search_documents":
        result = search_documents(
            query=arguments["query"],
            limit=arguments.get("limit", 5),
            file_type=arguments.get("file_type"),
        )
    elif name == "get_document":
        result = get_document(document_id=arguments["document_id"])
    elif name == "delete_document":
        result = delete_document(document_id=arguments["document_id"])
    elif name == "update_document":
        result = update_document(
            document_id=arguments["document_id"], metadata=arguments["metadata"]
        )
    else:
        result = {"error": f"Unknown tool: {name}"}

    return [TextContent(type="text", text=json.dumps(result, indent=2))]


# ============================================================
# Tool Logic Functions
# ============================================================


def upload_document(
    title: str, content: str, file_type: str = "txt", metadata: dict = None
) -> dict:
    """Upload a document to the DocuMind knowledge base."""
    try:
        client = get_supabase_client()

        document = {
            "title": title,
            "content": content,
            "file_type": file_type,
            "metadata": metadata or {},
            "created_at": datetime.now().isoformat(),
            "updated_at": datetime.now().isoformat(),
        }

        response = client.post("/documents", json=document)
        response.raise_for_status()

        result = response.json()
        doc_id = result[0]["id"] if isinstance(result, list) else result["id"]

        return {
            "success": True,
            "document_id": doc_id,
            "title": title,
            "message": f"Document '{title}' uploaded successfully",
        }
    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "message": "Failed to upload document",
        }


def search_documents(query: str, limit: int = 5, file_type: str = None) -> dict:
    """Search documents by title or content."""
    try:
        client = get_supabase_client()

        search_filter = f"or=(title.ilike.*{query}*,content.ilike.*{query}*)"
        if file_type:
            search_filter += f",file_type.eq.{file_type}"

        response = client.get(
            f"/documents?{search_filter}&limit={limit}&order=created_at.desc"
        )
        response.raise_for_status()

        documents = response.json()

        return {
            "success": True,
            "count": len(documents),
            "documents": [
                {
                    "id": doc["id"],
                    "title": doc["title"],
                    "file_type": doc["file_type"],
                    "preview": (
                        doc["content"][:200] + "..."
                        if len(doc["content"]) > 200
                        else doc["content"]
                    ),
                    "created_at": doc["created_at"],
                }
                for doc in documents
            ],
        }
    except Exception as e:
        return {"success": False, "error": str(e), "message": "Search failed"}


def get_document(document_id: str) -> dict:
    """Retrieve a specific document by ID."""
    try:
        client = get_supabase_client()

        response = client.get(f"/documents?id=eq.{document_id}")
        response.raise_for_status()

        documents = response.json()

        if not documents:
            return {"success": False, "message": "Document not found"}

        document = documents[0]

        return {
            "success": True,
            "document": {
                "id": document["id"],
                "title": document["title"],
                "content": document["content"],
                "file_type": document["file_type"],
                "metadata": document["metadata"],
                "created_at": document["created_at"],
                "updated_at": document["updated_at"],
            },
        }
    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "message": "Failed to retrieve document",
        }


def delete_document(document_id: str) -> dict:
    """Delete a document from the knowledge base."""
    try:
        client = get_supabase_client()

        response = client.delete(f"/documents?id=eq.{document_id}")
        response.raise_for_status()

        return {
            "success": True,
            "message": f"Document {document_id} deleted successfully",
        }
    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "message": "Failed to delete document",
        }


def update_document(document_id: str, metadata: dict) -> dict:
    """Update document metadata (enrichment, tags, summary, etc.)"""
    try:
        client = get_supabase_client()

        # Get current document
        response = client.get(f"/documents?id=eq.{document_id}")
        response.raise_for_status()
        documents = response.json()

        if not documents:
            return {"success": False, "message": "Document not found"}

        # Merge metadata
        current_metadata = documents[0].get("metadata", {})
        updated_metadata = {**current_metadata, **metadata}

        # Update document
        update_response = client.patch(
            f"/documents?id=eq.{document_id}",
            json={
                "metadata": updated_metadata,
                "updated_at": datetime.now().isoformat(),
            },
        )
        update_response.raise_for_status()

        return {
            "success": True,
            "document_id": document_id,
            "message": "Metadata updated successfully",
        }
    except Exception as e:
        return {"success": False, "error": str(e), "message": "Update failed"}


# ============================================================
# Server Entry Point
# ============================================================


async def main():
    """Run the MCP server using stdio transport"""
    async with stdio_server() as (read_stream, write_stream):
        await server.run(
            read_stream, write_stream, server.create_initialization_options()
        )


if __name__ == "__main__":
    asyncio.run(main())
