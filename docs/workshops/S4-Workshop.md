# HeroForge.AI Course: AI-Powered Software Development
## Lesson 4 Workshop: MCP and A2A Communication

**Estimated Time:** 45-60 minutes\
**Difficulty:** Intermediate\
**Prerequisites:** Completed Sessions 1-3 (Claude Code fundamentals, Skills, Subagents, Hooks)

---

## Workshop Objectives

By completing this workshop, you will:
- [x] Understand Model Context Protocol (MCP) architecture and purpose
- [x] Install and configure Supabase MCP server for database connectivity
- [x] Build a custom MCP server with tools and resources
- [x] Implement Agent-to-Agent (A2A) communication patterns
- [x] Connect DocuMind to Supabase database with proper schema
- [x] Create multi-agent workflows using shared database state

---

## Prerequisites Check (Run Before Starting)

### Environment Verification

Before starting the workshop exercises, verify your environment is configured correctly:

```bash
# Check MCP SDK version (should be 2.x)
pip show mcp 2>/dev/null | grep Version || echo "MCP SDK not installed"

# If you see version 1.x, upgrade:
# pip install --upgrade mcp

# Note: MCP SDK v1.x uses different API (@server.tool)
# MCP SDK v2.x uses new API (@server.list_tools, @server.call_tool)
# Most current examples use v2.x
```

### Supabase Configuration Check

Run this Python snippet to verify your environment variables are set:

```bash
python3 -c "
from dotenv import load_dotenv
import os

load_dotenv()

checks = {
    'SUPABASE_URL': os.getenv('SUPABASE_URL'),
    'SUPABASE_SERVICE_KEY': os.getenv('SUPABASE_SERVICE_KEY'),
}

for var, val in checks.items():
    if val:
        print(f'✅ PASS: {var} configured ({val[:25]}...)')
    else:
        print(f'❌ FAIL: {var} missing - check your .env file')
"
```

**All checks must PASS before continuing.**

---

## Module 1: Understanding MCP (15 minutes)

### Concept Review

**What is Model Context Protocol (MCP)?**

MCP is a standardized protocol that allows AI models (like Claude) to connect to external systems, tools, and data sources in a consistent, secure way. Think of it as USB-C for AI—one universal standard instead of custom integrations for every tool.

**Why MCP Matters:**
- **Standardization**: One protocol for all tools (databases, APIs, file systems)
- **Security**: Built-in authentication and permission models
- **Composability**: Mix and match MCP servers like LEGO blocks
- **Context Sharing**: AI maintains awareness of external state

**MCP Components:**
1. **MCP Server**: A process that exposes tools, resources, and prompts
2. **Tools**: Functions the AI can call (e.g., `query_database`, `upload_file`)
3. **Resources**: Data sources the AI can read (e.g., files, database schemas)
4. **Prompts**: Reusable prompt templates with parameters

**MCP vs Traditional APIs:**
| Feature | Traditional API | MCP |
|---------|----------------|-----|
| **Integration** | Custom code for each API | Standard protocol |
| **Discovery** | Manual documentation | Auto-discovery of capabilities |
| **Context** | Stateless | Stateful with memory |
| **AI-Friendly** | Requires parsing/formatting | Native AI tool format |

---

### Exercise 1.1: Install Supabase MCP Server

**Task:** Set up Supabase MCP server to connect Claude Code to your Supabase database.

**Instructions:**

**Step 1: Create Supabase Project (5 mins)**

If you don't have a Supabase project yet:

1. Go to https://supabase.com
2. Sign in or create free account
3. Click "New Project"
4. Fill in:
   - **Name**: `documind-dev`
   - **Database Password**: Generate strong password (save it!)
   - **Region**: Choose closest to you
5. Wait 2-3 minutes for project creation
6. Note your **Project URL** and **API Keys** from Settings → API

**Step 2: Configure Environment Variables (3 mins)**

In your Codespace, copy `env.example` to `.env` and fill in your Supabase credentials:

```bash
# Copy the example environment file
cp env.example .env

# Edit .env with your credentials
# You'll need to get these values from Supabase:
```

**Required environment variables** (get from https://supabase.com/):

```bash
# ============================================
# REQUIRED: Database (Session 4+)
# ============================================
# Get from: https://supabase.com/ Settings -> API Keys --> Legacy anon, service_role API keys
SUPABASE_URL=https://xxxxx.supabase.co
SUPABASE_ANON_KEY=eyJxxxxx
SUPABASE_SERVICE_KEY=eyJxxxxx

# Personal Access Token (Required for MCP)
# Get from: https://supabase.com/dashboard/account/tokens
SUPABASE_ACCESS_TOKEN=sbp_xxxxx
```

⚠️ **IMPORTANT**: The `SUPABASE_ACCESS_TOKEN` is a **Personal Access Token (PAT)**, which is different from the project API keys. You must generate this from your Supabase account settings.

**Step 3: Install Supabase MCP Server (5 mins)**

We've created a skill that handles the Supabase MCP installation automatically.

**In Claude Code, simply type:**
```
Use the supabase mcp installer skill to install supabase mcp
```

Claude will use the skill to:
1. ✅ Read your credentials from `.env`
2. ✅ Install the correct MCP package (`@supabase/mcp-server-supabase`)
3. ✅ Configure authentication with your Personal Access Token
4. ✅ Verify the connection

**To verify installation manually:**
```bash
claude mcp list

# Or verify in Claude Code by typing: /mcp
```

**Expected output:**
```
supabase: npx @supabase/mcp-server-supabase - ✓ Connected
```

**Step 4: Test Supabase MCP (2 mins)**

In Claude Code (`dsp`):
```
Use the supabase MCP to list all tables in the database
```

**Expected outcome:** Claude should connect to your Supabase project and return a list of tables (probably empty for new project).

---

### Exercise 1.2: Create DocuMind Database Schema

**Task:** Use Supabase MCP to create the documents table for DocuMind.

**Instructions:**

**In Claude Code, type:**
```
Use the supabase MCP to create a table called 'documents' with these columns:
- id: UUID primary key with default gen_random_uuid()
- title: TEXT not null
- content: TEXT not null
- file_path: TEXT
- file_type: TEXT (e.g., 'pdf', 'docx', 'txt')
- metadata: JSONB (for flexible additional data)
- created_at: TIMESTAMP with default now()
- updated_at: TIMESTAMP with default now()

Also create an index on created_at for fast sorting.
```

**Claude should execute this via MCP and create the table.**

**Verify the table was created:**
```
Use supabase MCP to describe the structure of the 'documents' table
```

---

## Git: Pushing Your Work (5 minutes)

### Concept: Sharing Your Code

Now that you've made local commits, let's push them to GitHub so:
- Your work is backed up in the cloud
- Others can review your code
- You can collaborate with teammates

### Exercise 0.2: Push to Remote

**Step 1: Push Your Branch (2 mins)**

```bash
# Push your feature branch to GitHub
git push -u origin issue-12-document-processing

# The -u flag sets upstream tracking (only needed first time)
# Future pushes can just use: git push
```

**Expected output:**
```
Enumerating objects: 15, done.
Counting objects: 100% (15/15), done.
Delta compression using up to 8 threads
Compressing objects: 100% (8/8), done.
Writing objects: 100% (9/9), 1.23 KiB | 1.23 MiB/s, done.
Total 9 (delta 6), reused 0 (delta 0), pack-reused 0
remote: Resolving deltas: 100% (6/6), completed with 6 local objects.
To github.com:yourusername/course-ai-software-dev.git
 * [new branch]      issue-12-document-processing -> issue-12-document-processing
Branch 'issue-12-document-processing' set up to track remote branch 'issue-12-document-processing' from 'origin'.
```

**Step 2: Verify on GitHub (1 min)**

1. Go to your repository on GitHub
2. Click "Branches" (should show 2 branches now)
3. Find your `issue-12-document-processing` branch
4. Click it to view your commits

**Step 3: Understanding Push Frequency (2 mins)**

**When to push:**
- ✅ After completing a logical unit of work (e.g., one exercise)
- ✅ Before taking a break or ending your coding session
- ✅ After passing tests
- ✅ When you want feedback from teammates or instructor

**When NOT to push:**
- ❌ Code doesn't compile/run
- ❌ Tests are failing (unless explicitly working on fixing them)
- ❌ In the middle of incomplete changes
- ❌ Sensitive data (API keys, passwords) in code

**Pro Tip: Push Early, Push Often**
```bash
# Good rhythm:
git add <files>
git commit -m "feat: add embedding generator"
git push

# Continue working...
git add <files>
git commit -m "test: add embedding generator tests"
git push

# Each push is a savepoint you can return to
```

### Common Git Commands Reference

```bash
# Check status of files
git status

# View commit history
git log --oneline -10

# See what changed in files
git diff

# Undo uncommitted changes to a file
git checkout -- <filename>

# View remote repository URL
git remote -v

# Pull latest changes from main
git checkout main
git pull origin main

# Switch back to your feature branch
git checkout issue-12-document-processing

# Merge main into your feature branch (if needed)
git merge main
```

### Troubleshooting Common Push Issues

**Issue: "Permission denied (publickey)"**
```bash
# Solution: Add SSH key to GitHub
ssh-keygen -t ed25519 -C "your.email@example.com"
cat ~/.ssh/id_ed25519.pub
# Copy output and add to GitHub Settings → SSH Keys
```

**Issue: "Updates were rejected"**
```bash
# Solution: Pull first, then push
git pull origin your-branch-name
git push
```

**Issue: "fatal: The current branch has no upstream branch"**
```bash
# Solution: Set upstream with -u flag
git push -u origin your-branch-name
```

---

### Quiz 1:

**Question 1:** What is the primary advantage of MCP over traditional API integrations?\
   a) MCP provides a standardized protocol that works across all tools, enabling auto-discovery and consistent AI integration\
   b) MCP is faster than REST APIs\
   c) MCP doesn't require authentication\
   d) MCP only works with databases

**Question 2:** What are the three main components of an MCP server?\
   a) Tools (functions), Resources (data), and Prompts (templates)\
   b) Database, API, and File System\
   c) Client, Server, and Middleware\
   d) Read, Write, and Execute

**Question 3:** Why is Supabase a good choice for DocuMind's database?\
   a) It provides PostgreSQL with pgvector extension for vector search, plus a ready-to-use MCP server\
   b) It's the only database that works with Claude\
   c) It's completely free with no limits\
   d) It automatically generates embeddings

**Answers:**
1. **a)** MCP standardizes tool integration, enabling consistent AI workflows across different tools
2. **a)** Tools, Resources, and Prompts are the three core MCP components
3. **a)** Supabase offers PostgreSQL + pgvector + MCP integration, perfect for AI applications

---

## Module 2: Building a Custom MCP Server (15 minutes)

### Concept Review

**Why Build Custom MCP Servers?**

While pre-built MCP servers (like Supabase) are great, you'll often need custom tools specific to your application. Custom MCP servers let you:
- Wrap complex business logic as simple AI tools
- Integrate with proprietary systems
- Create domain-specific operations
- Enforce business rules and validation

**MCP Server Structure:**
```python
from mcp.server import Server
from mcp import Tool, Resource

server = Server("my-server")

@server.tool()
def my_tool(param: str) -> dict:
    """Tool description for AI"""
    # Implementation
    return {"result": "success"}

@server.resource("my-resource")
def my_resource() -> str:
    """Resource description"""
    return "Resource data"
```

---

### Exercise 2.1: Create DocuMind MCP Server

**Task:** Build a custom MCP server that provides document-specific operations.

**Instructions:**

**Step 1: Create MCP Server Directory (2 mins)**

```bash
mkdir -p src/documind-mcp
cd src/documind-mcp

# Create server file
touch server.py

# Create package file
touch __init__.py
```

**Step 2: Write the MCP Server (8 mins)**

Open `src/documind-mcp/server.py`:

```python
"""
DocuMind Custom MCP Server
Provides document management tools for Claude Code
"""
import os
import json
from datetime import datetime
from typing import Optional
from mcp.server import Server
from mcp.types import Tool, TextContent
import httpx

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
            "Prefer": "return=representation"
        }
    )

@server.tool()
def upload_document(
    title: str,
    content: str,
    file_type: str = "txt",
    metadata: Optional[dict] = None
) -> dict:
    """
    Upload a document to the DocuMind knowledge base.

    Args:
        title: Document title
        content: Full document content
        file_type: Type of document (txt, pdf, docx, etc.)
        metadata: Optional metadata dictionary

    Returns:
        Dictionary with document ID and upload status
    """
    try:
        client = get_supabase_client()

        # Prepare document data
        document = {
            "title": title,
            "content": content,
            "file_type": file_type,
            "metadata": metadata or {},
            "created_at": datetime.now().isoformat(),
            "updated_at": datetime.now().isoformat()
        }

        # Insert into database
        response = client.post("/documents", json=document)
        response.raise_for_status()

        result = response.json()
        doc_id = result[0]["id"] if isinstance(result, list) else result["id"]

        return {
            "success": True,
            "document_id": doc_id,
            "title": title,
            "message": f"Document '{title}' uploaded successfully"
        }

    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "message": "Failed to upload document"
        }

@server.tool()
def search_documents(
    query: str,
    limit: int = 5,
    file_type: Optional[str] = None
) -> dict:
    """
    Search documents by title or content.

    Args:
        query: Search query string
        limit: Maximum number of results
        file_type: Optional filter by file type

    Returns:
        Dictionary with matching documents
    """
    try:
        client = get_supabase_client()

        # Build search query
        search_filter = f"or=(title.ilike.*{query}*,content.ilike.*{query}*)"
        if file_type:
            search_filter += f",file_type.eq.{file_type}"

        # Query database
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
                    "preview": doc["content"][:200] + "..." if len(doc["content"]) > 200 else doc["content"],
                    "created_at": doc["created_at"]
                }
                for doc in documents
            ]
        }

    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "message": "Search failed"
        }

@server.tool()
def get_document(document_id: str) -> dict:
    """
    Retrieve a specific document by ID.

    Args:
        document_id: UUID of the document

    Returns:
        Dictionary with full document data
    """
    try:
        client = get_supabase_client()

        response = client.get(f"/documents?id=eq.{document_id}")
        response.raise_for_status()

        documents = response.json()

        if not documents:
            return {
                "success": False,
                "message": "Document not found"
            }

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
                "updated_at": document["updated_at"]
            }
        }

    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "message": "Failed to retrieve document"
        }

@server.tool()
def delete_document(document_id: str) -> dict:
    """
    Delete a document from the knowledge base.

    Args:
        document_id: UUID of the document to delete

    Returns:
        Dictionary with deletion status
    """
    try:
        client = get_supabase_client()

        response = client.delete(f"/documents?id=eq.{document_id}")
        response.raise_for_status()

        return {
            "success": True,
            "message": f"Document {document_id} deleted successfully"
        }

    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "message": "Failed to delete document"
        }

# Start server
if __name__ == "__main__":
    import asyncio
    asyncio.run(server.run())
```

**Step 3: Install MCP Server Dependencies (2 mins)**

```bash
# Update package.json to include MCP dependencies
cat >> package.json << 'EOF'
{
  "dependencies": {
    "@modelcontextprotocol/sdk": "^0.5.0",
    "httpx": "^1.0.0"
  }
}
EOF

npm install
```

**Step 4: Register the Custom MCP Server (3 mins)**

```bash
# Register the custom MCP server
claude mcp add --transport stdio documind -- python3 src/documind-mcp/server.py

# Verify registration
claude mcp list
# Or type /mcp in Claude Code
```

**Expected output:**
```
MCP servers:
  supabase: npx @supabase/mcp-server-supabase - ✓ Connected
  documind (stdio): python3 src/documind-mcp/server.py
```

---

### Exercise 2.2: Test Custom MCP Tools

**Task:** Use your custom MCP tools to upload and search documents.

**In Claude Code (`dsp`), type:**

```
Use the documind MCP to upload a document:
- Title: "Company Handbook"
- Content: "Welcome to our company! This handbook contains all policies and procedures. Section 1: Code of Conduct. Section 2: Benefits. Section 3: Time Off."
- File type: "txt"
```

**Then search for it:**
```
Use the documind MCP to search for documents containing "handbook"
```

**Then retrieve it by ID:**
```
Use the documind MCP to get the full document with ID [use the ID from search results]
```

---

### Quiz 2:

**Question 1:** What is the purpose of the `@server.tool()` decorator in a custom MCP server?\
   a) It registers a Python function as an AI-callable tool that Claude can discover and use\
   b) It makes the function run faster\
   c) It encrypts the function's output\
   d) It's optional and doesn't do anything

**Question 2:** Why do we include docstrings in MCP tool functions?\
   a) The docstrings become the tool descriptions that Claude reads to understand how to use each tool\
   b) Docstrings are required by Python\
   c) They make the code look professional\
   d) They have no impact on MCP functionality

**Question 3:** What's the advantage of using environment variables for Supabase credentials?\
   a) Security: credentials aren't hardcoded in code and can be managed separately per environment\
   b) Environment variables are faster than hardcoded values\
   c) It's required by Supabase\
   d) Environment variables make the code shorter

**Answers:**
1. **a)** `@server.tool()` exposes functions as discoverable, AI-callable tools in the MCP protocol
2. **a)** Docstrings provide Claude with descriptions of what each tool does and how to use it
3. **a)** Environment variables keep secrets secure and separate from code, preventing accidental exposure

---

## Module 3: Agent-to-Agent Communication (15 minutes)

### Concept Review

**What is Agent-to-Agent (A2A) Communication?**

A2A communication allows multiple AI agents to coordinate their work by sharing information, delegating tasks, and synchronizing state. Think of it as a team of specialists working together on a project.

**A2A Communication Patterns:**

1. **Shared State** (Database)
   - Agents read/write to common database
   - Indirect communication via data
   - Good for: async workflows, persistence

2. **Message Passing** (MCP)
   - Agents send explicit messages
   - Direct communication
   - Good for: real-time coordination, complex workflows

3. **Memory Coordination** (claude-flow)
   - Agents share context via memory layer
   - Hybrid approach
   - Good for: swarm orchestration

**A2A Use Cases:**
- Document processing pipeline (extractor → analyzer → embedder → storage)
- Code review workflow (analyzer → security checker → test generator)
- Research tasks (researcher → summarizer → fact-checker)

---

### Exercise 3.1: Multi-Agent Document Pipeline

**Task:** Create a two-agent workflow where one agent uploads documents and another verifies them.

**Instructions:**

**Step 1: Create Agent 1 - Document Uploader (5 mins)**

Create `.claude/subagents/doc-uploader.md`:

```markdown
---
name: Document Uploader
role: Document Upload Specialist
version: 1.0.0
---

# Document Uploader Agent

## Identity
You are a specialized agent responsible for uploading documents to the DocuMind knowledge base using the documind MCP server.

## Responsibilities
1. Accept documents from users or other systems
2. Validate document format and content
3. Generate appropriate metadata
4. Upload via documind MCP `upload_document` tool
5. Confirm successful upload with document ID

## Process
1. **Receive** document (title, content, type)
2. **Validate**:
   - Title is not empty
   - Content has substance (>50 characters)
   - File type is supported (txt, pdf, docx, md)
3. **Prepare metadata**:
   - Word count
   - Upload timestamp
   - Source information
4. **Upload** using documind MCP
5. **Report** success with document ID

## Output Format
Always respond with:
```json
{
  "status": "success" | "failure",
  "document_id": "uuid",
  "title": "string",
  "message": "descriptive message"
}
```

## Constraints
- Never upload empty documents
- Always include at least basic metadata
- Verify upload success before reporting
- Log any errors for troubleshooting
```

**Step 2: Create Agent 2 - Document Verifier (5 mins)**

Create `.claude/subagents/doc-verifier.md`:

```markdown
---
name: Document Verifier
role: Quality Assurance Specialist
version: 1.0.0
---

# Document Verifier Agent

## Identity
You are a quality assurance agent that verifies documents uploaded to DocuMind meet quality standards.

## Responsibilities
1. Retrieve documents by ID using documind MCP
2. Verify content quality and completeness
3. Check metadata accuracy
4. Flag issues or approve documents
5. Update document status

## Verification Checklist
- [ ] Document retrieved successfully
- [ ] Title is descriptive and meaningful
- [ ] Content length is adequate (>100 words for policies)
- [ ] No corrupted or garbled text
- [ ] Metadata is present and accurate
- [ ] File type matches content

## Process
1. **Retrieve** document using `get_document` tool
2. **Analyze** content against checklist
3. **Calculate** quality score (0-100)
4. **Report** findings

## Output Format
```json
{
  "document_id": "uuid",
  "status": "approved" | "rejected" | "needs_review",
  "quality_score": 85,
  "issues": ["list of any issues found"],
  "recommendations": ["suggested improvements"]
}
```

## Scoring Criteria
- Title clarity: 20 points
- Content completeness: 30 points
- Formatting quality: 20 points
- Metadata accuracy: 15 points
- Overall coherence: 15 points
```

**Step 3: Test A2A Workflow (5 mins)**

In Claude Code:

```
I need to test Agent-to-Agent communication.

Step 1: Invoke the doc-uploader subagent to upload this document:
- Title: "Remote Work Policy"
- Content: "Employees may work remotely up to 3 days per week. Remote work requests must be approved by direct manager. Equipment: Company provides laptop and monitor for home office setup. Communication: All remote workers must be available during core hours 10am-3pm. Security: Use VPN for all company system access."
- Type: "txt"

Step 2: Once uploaded, invoke the doc-verifier subagent to verify the document quality using the document ID from step 1.

Show me both agents' outputs.
```

**Expected outcome:**
- Agent 1 (uploader) uploads document and returns ID
- Agent 2 (verifier) retrieves that document by ID and provides quality assessment
- You can see the A2A coordination via shared database state

---

### Quiz 3:

**Question 1:** What is the primary benefit of Agent-to-Agent communication?\
   a) It enables specialized agents to collaborate on complex tasks that no single agent could complete alone\
   b) It makes agents run faster\
   c) It reduces the need for databases\
   d) It's required by MCP

**Question 2:** In the document pipeline example, how do the two agents communicate?\
   a) Through shared database state—agent 1 writes a document that agent 2 reads using the document ID\
   b) They send emails to each other\
   c) They don't actually communicate\
   d) Through a chat interface

**Question 3:** When should you use multi-agent A2A vs a single agent?\
   a) Use multi-agent when the task has distinct specialized phases that benefit from different expertise or when parallel processing improves performance\
   b) Always use multiple agents for every task\
   c) Only use multiple agents if you have more than 1000 documents\
   d) Single agents are always better

**Answers:**
1. **a)** A2A enables specialized agents to collaborate on complex, multi-step tasks through coordination
2. **a)** They communicate through shared Supabase database—agent 1 writes, agent 2 reads via document ID
3. **a)** Use multi-agent for specialized phases or parallel processing; use single agent for simple, linear tasks

---

## Module 4: Challenge Project - DocuMind Document Manager (15 minutes)

### Challenge Overview

Build a complete multi-agent document management system using custom MCP tools and A2A communication.

**Your Mission:**
Create a 3-agent workflow that:
1. Uploads documents with validation
2. Processes and enriches documents
3. Makes documents searchable

---

### Challenge Requirements

**Feature:** Intelligent Document Management Pipeline

**What to Build:**

1. **Agent 1: Document Processor**
   - Validates file format and content
   - Extracts metadata (word count, type, sections)
   - Uploads via documind MCP
   - Returns document ID

2. **Agent 2: Content Enricher**
   - Retrieves document by ID
   - Generates summary (3 sentences max)
   - Extracts key entities (names, dates, topics)
   - Tags document for searchability
   - Updates metadata with enriched information

3. **Agent 3: Search Interface**
   - Accepts natural language queries
   - Searches documents using documind MCP
   - Returns formatted results with relevance scores
   - Provides document previews

4. **Custom MCP Tool: Update Document**
   - Add `update_document` tool to your MCP server
   - Allows updating document metadata
   - Used by Agent 2 to save enriched data

---

### Starter Code

**Update `src/documind-mcp/server.py`** with new tool:

```python
@server.tool()
def update_document(
    document_id: str,
    metadata: dict
) -> dict:
    """
    Update document metadata (enrichment, tags, summary, etc.)

    Args:
        document_id: UUID of document to update
        metadata: Dictionary of metadata to merge with existing

    Returns:
        Dictionary with update status
    """
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
                "updated_at": datetime.now().isoformat()
            }
        )
        update_response.raise_for_status()

        return {
            "success": True,
            "document_id": document_id,
            "message": "Metadata updated successfully"
        }

    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "message": "Update failed"
        }
```

---

### Your Task

**Step 1: Create the Three Agents (10 mins)**

Create these subagent files:
- `.claude/subagents/doc-processor.md` (based on Exercise 3.1, add validation logic)
- `.claude/subagents/content-enricher.md` (new - summarize, extract entities, tag)
- `.claude/subagents/search-interface.md` (new - natural language search)

**Step 2: Build the Pipeline (5 mins)**

Test the complete workflow in Claude Code:

```
Execute this 3-agent pipeline for the following document:

Document:
Title: "Employee Benefits Guide 2025"
Content: "Our comprehensive benefits package includes health insurance (PPO and HMO plans), dental and vision coverage, 401(k) with 5% company match, flexible spending accounts (FSA and HSA), life insurance, disability coverage, employee assistance program (EAP), tuition reimbursement up to $5,000 annually, gym membership discounts, and commuter benefits. Enrollment period: November 1-30. New hires have 30 days from start date to enroll. Contact HR at benefits@company.com for questions."

Pipeline:
1. Invoke doc-processor to validate and upload
2. Invoke content-enricher to analyze and enhance (use document ID from step 1)
3. Invoke search-interface to test searchability with query "How do I enroll in benefits?"

Show the output from each agent.
```

---

### Success Criteria

Your implementation is complete when:

- [ ] `doc-processor` agent validates and uploads documents successfully
- [ ] `content-enricher` agent generates summaries and extracts entities
- [ ] `content-enricher` agent updates document metadata using `update_document` tool
- [ ] `search-interface` agent returns relevant documents for queries
- [ ] All three agents coordinate via MCP tools and shared database
- [ ] Pipeline processes a document end-to-end without errors
- [ ] Search returns the enriched document with improved metadata
- [ ] You can query the document using natural language

**Bonus Challenges:**
- Add a 4th agent that translates documents to other languages
- Implement error recovery (what if agent 2 fails?)
- Add duplicate detection (don't upload same document twice)
- Create a batch processing mode (process 10 documents in parallel)

---

## Answer Key

### Exercise 1.1 Solution

**Supabase MCP Installation:**

```bash
# Step 1: Copy env.example to .env and fill in your Supabase credentials
cp env.example .env

# Required values in .env:
# SUPABASE_URL=https://xxxxx.supabase.co        # From Settings -> API
# SUPABASE_ANON_KEY=eyJxxxxx                     # From Settings -> API Keys
# SUPABASE_SERVICE_KEY=eyJxxxxx                  # From Settings -> API Keys
# SUPABASE_ACCESS_TOKEN=sbp_xxxxx                # From Account Settings -> Access Tokens (PAT)

# Step 2: Install MCP server using the skill
# In Claude Code, type:
# "Use the supabase mcp installer skill to install supabase mcp"

# Step 3: Verify
claude mcp list
# Or type /mcp in Claude Code
```

**Testing:**
```
dsp> Use the supabase MCP to list all tables

[Claude uses Supabase MCP]
Tables in database:
- (empty) or list of existing tables
```

---

### Exercise 1.2 Solution

**Create Documents Table:**

```sql
-- Claude executes this via Supabase MCP:
CREATE TABLE documents (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    title TEXT NOT NULL,
    content TEXT NOT NULL,
    file_path TEXT,
    file_type TEXT,
    metadata JSONB DEFAULT '{}'::jsonb,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT now(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT now()
);

CREATE INDEX idx_documents_created_at ON documents(created_at DESC);
```

**Verification:**
```
dsp> Use supabase MCP to describe the 'documents' table

Columns:
- id: uuid (primary key)
- title: text (not null)
- content: text (not null)
- file_path: text
- file_type: text
- metadata: jsonb
- created_at: timestamp with time zone
- updated_at: timestamp with time zone

Indexes:
- documents_pkey (primary key on id)
- idx_documents_created_at (btree on created_at)
```

---

### Exercise 2.1 & 2.2 Solution

**Custom MCP Server:** See the complete `server.py` code provided in Exercise 2.1.

**Testing the Tools:**

```python
# Test 1: Upload document
dsp> Use documind MCP to upload a document titled "Company Handbook"...

Response:
{
  "success": true,
  "document_id": "a1b2c3d4-e5f6-7890-abcd-ef1234567890",
  "title": "Company Handbook",
  "message": "Document 'Company Handbook' uploaded successfully"
}

# Test 2: Search
dsp> Use documind MCP to search for "handbook"

Response:
{
  "success": true,
  "count": 1,
  "documents": [
    {
      "id": "a1b2c3d4-e5f6-7890-abcd-ef1234567890",
      "title": "Company Handbook",
      "file_type": "txt",
      "preview": "Welcome to our company! This handbook contains all policies...",
      "created_at": "2025-11-24T10:30:00Z"
    }
  ]
}

# Test 3: Get by ID
dsp> Use documind MCP to get document a1b2c3d4-e5f6-7890-abcd-ef1234567890

Response:
{
  "success": true,
  "document": {
    "id": "a1b2c3d4-e5f6-7890-abcd-ef1234567890",
    "title": "Company Handbook",
    "content": "Welcome to our company! This handbook contains all policies and procedures...",
    "file_type": "txt",
    "metadata": {},
    "created_at": "2025-11-24T10:30:00Z",
    "updated_at": "2025-11-24T10:30:00Z"
  }
}
```

---

### Exercise 3.1 Solution

**Agent 1 Output (doc-uploader):**
```json
{
  "status": "success",
  "document_id": "9f8e7d6c-5b4a-3210-9876-fedcba098765",
  "title": "Remote Work Policy",
  "message": "Document uploaded successfully with metadata: word_count=67, upload_time=2025-11-24T11:00:00Z"
}
```

**Agent 2 Output (doc-verifier):**
```json
{
  "document_id": "9f8e7d6c-5b4a-3210-9876-fedcba098765",
  "status": "approved",
  "quality_score": 88,
  "issues": [],
  "recommendations": [
    "Consider adding examples of remote work requests",
    "Specify VPN setup instructions"
  ],
  "checklist": {
    "title_clarity": "✓ Pass",
    "content_completeness": "✓ Pass",
    "formatting": "✓ Pass",
    "metadata": "✓ Pass",
    "coherence": "✓ Pass"
  }
}
```

---

### Module 4 Challenge Solution

**Agent 1: doc-processor.md**
```markdown
---
name: Document Processor
role: Document Validation and Upload Specialist
version: 1.0.0
---

# Document Processor Agent

## Identity
You process and validate documents before uploading to DocuMind.

## Process
1. **Validate**:
   - Title: 5-200 characters, meaningful
   - Content: Minimum 50 characters, maximum 1MB
   - File type: txt, pdf, docx, md only
2. **Extract Metadata**:
   - Word count: Split on whitespace
   - Character count: Length of content
   - Estimated read time: words / 200
   - Section count: Count headings
3. **Upload**: Use documind MCP `upload_document`
4. **Report**: Return document ID and metadata

## Output Format
```json
{
  "status": "success",
  "document_id": "uuid",
  "metadata": {
    "word_count": 150,
    "char_count": 890,
    "read_time_minutes": 1,
    "sections": 3
  }
}
```
```

**Agent 2: content-enricher.md**
```markdown
---
name: Content Enricher
role: Document Analysis and Enhancement Specialist
version: 1.0.0
---

# Content Enricher Agent

## Identity
You analyze documents and add semantic enrichment.

## Process
1. **Retrieve**: Use documind MCP `get_document(id)`
2. **Analyze**:
   - Generate 3-sentence summary
   - Extract key entities (people, dates, topics)
   - Identify document category (policy, guide, procedure)
   - Generate searchable tags
3. **Update**: Use documind MCP `update_document` with enriched metadata
4. **Report**: Confirm enrichment complete

## Enrichment Format
```json
{
  "summary": "3-sentence summary",
  "entities": {
    "people": ["names"],
    "dates": ["dates"],
    "topics": ["key topics"]
  },
  "category": "policy",
  "tags": ["tag1", "tag2", "tag3"],
  "enriched_at": "timestamp"
}
```
```

**Agent 3: search-interface.md**
```markdown
---
name: Search Interface
role: Natural Language Search Specialist
version: 1.0.0
---

# Search Interface Agent

## Identity
You help users find documents using natural language queries.

## Process
1. **Parse Query**: Extract intent and keywords
2. **Search**: Use documind MCP `search_documents`
3. **Rank**: Score results by relevance
4. **Format**: Present results with previews

## Output Format
```markdown
## Search Results for "[query]"

Found **N documents**

### 1. [Document Title]
**Type**: txt | **Date**: 2025-11-24
**Relevance**: ★★★★☆

[Preview text...]

**Tags**: tag1, tag2, tag3

### 2. [Next result...]
```
```

**Complete Pipeline Test:**

```
dsp> Execute 3-agent pipeline...

[Agent 1: doc-processor]
✓ Validated: Title length OK, Content >50 chars, Type valid
✓ Metadata: 89 words, 573 chars, ~1 min read, 0 sections
✓ Uploaded: document_id = "abc-123-def-456"

[Agent 2: content-enricher]
✓ Retrieved document abc-123-def-456
✓ Generated summary: "The employee benefits guide covers health insurance, retirement plans, and various wellness programs. Enrollment occurs annually in November, with new hires having 30 days to enroll. Employees should contact HR for questions about benefits."
✓ Extracted entities: benefits@company.com, November 1-30, $5,000
✓ Categories: HR Policy, Benefits
✓ Tags: benefits, health insurance, 401k, enrollment, HR
✓ Updated metadata successfully

[Agent 3: search-interface]
Query: "How do I enroll in benefits?"

## Search Results

Found **1 document**

### 1. Employee Benefits Guide 2025
**Type**: txt | **Date**: 2025-11-24
**Relevance**: ★★★★★

Our comprehensive benefits package includes health insurance... Enrollment period: November 1-30. New hires have 30 days from start date to enroll. Contact HR at benefits@company.com for questions.

**Tags**: benefits, health insurance, 401k, enrollment, HR
```

---

## Additional Challenges (Optional)

### Challenge 1: Translation Agent
Create a 4th agent that translates documents to Spanish, French, or German using an LLM and stores translations as separate documents linked by metadata.

### Challenge 2: Duplicate Detection
Add a `find_duplicates` MCP tool that uses content similarity to identify duplicate documents before upload.

### Challenge 3: Batch Processing
Modify the pipeline to process 10 documents in parallel using claude-flow swarm orchestration.

### Challenge 4: Error Recovery
Implement retry logic and fallback strategies when any agent in the pipeline fails.

---

## Troubleshooting

### Common Issue 1: Supabase MCP Not Connecting

**Symptom:**
```
Error: Failed to connect to Supabase
```

**Solution:**
1. Verify environment variables: `cat .env`
2. Check Supabase project status (must be active)
3. Verify API keys are correct (Settings → API in Supabase dashboard)
4. Restart Claude Code: exit and run `dsp` again

---

### Common Issue 2: Custom MCP Server Not Registering

**Symptom:**
```
Error: MCP server 'documind' not found
```

**Solution:**
1. Check server path is absolute: `realpath src/documind-mcp/server.py`
2. Verify server.py has no syntax errors: `python3 src/documind-mcp/server.py`
3. Re-register: `claude mcp add --transport stdio documind -- python3 src/documind-mcp/server.py`
4. Make sure to use `--` before the server command
5. Check MCP logs: `claude mcp logs documind`

---

### Common Issue 3: Agent Can't Find MCP Tools

**Symptom:** Agent responds with "I don't have access to that tool"

**Solution:**
1. List available tools: `claude mcp list`
2. Verify server is active (status should show "Active")
3. Restart MCP server: `claude mcp restart documind`
4. Be explicit in prompt: "Use the documind MCP server to call the upload_document tool..."

---

### Common Issue 4: Database Insert Fails

**Symptom:**
```
Error: duplicate key value violates unique constraint
```

**Solution:**
1. Check if document already exists: search before insert
2. Verify UUID generation is enabled: `CREATE EXTENSION IF NOT EXISTS "uuid-ossp";`
3. Don't manually specify ID (let database generate it)

---

### Common Issue 5: A2A Coordination Breaks

**Symptom:** Agent 2 can't find document uploaded by Agent 1

**Solution:**
1. Verify Agent 1 actually uploaded (check return value)
2. Check document ID format (must be valid UUID)
3. Add delay between agents if needed: `sleep 1`
4. Verify Supabase connection is shared (same project URL)

---

## Key Takeaways

By completing this workshop, you've learned:

1. **MCP standardizes AI tool integration** - One protocol for all external systems
2. **Custom MCP servers enable domain-specific operations** - Wrap your business logic as AI tools
3. **A2A communication enables agent collaboration** - Multiple specialized agents working together
4. **Shared state via database enables async coordination** - Agents don't need to run simultaneously
5. **DocuMind is now data-persistent** - Documents stored in Supabase, ready for RAG

**The Integration Pattern:**
```
Claude Code → MCP Protocol → Custom/Standard Servers → External Systems (DB, APIs, Files)
     ↓
Multiple Agents → Shared Database State → Coordinated Workflows
```

---

## Next Session Preview

In **Session 5: Multi-Agent Systems**, we'll:
- Initialize ClaudeFlow swarms with different topologies
- Spawn specialized agents with distinct capabilities
- Build a 4-agent document processing pipeline
- Process multiple documents in parallel
- Compare performance: sequential vs parallel processing
- Explore HeroForge AEF and Google Antigravity

**Preparation:**
1. Ensure DocuMind MCP server is working
2. Upload 5-10 test documents to your database
3. Install ClaudeFlow: `npm install -g claude-flow@alpha`
4. Get OpenAI API key (for embeddings in Session 5)

See you in Session 5!

---

**Workshop Complete! 🎉**

You've connected DocuMind to Supabase, built custom MCP tools, and created multi-agent workflows. You're ready to scale with swarm orchestration!
