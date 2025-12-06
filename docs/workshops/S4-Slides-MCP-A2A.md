# Session 4: MCP and A2A
## Model Context Protocol & Agent-to-Agent Communication

---

## What You'll Learn Today

### MCP Architecture
Understand the Model Context Protocol and why it matters for AI applications

### Supabase MCP Server
Connect Claude Code to Supabase database for persistent storage

### Custom MCP Server
Build a specialized document management MCP server for DocuMind

### Agent-to-Agent Communication
Implement multi-agent coordination patterns using shared state

### DocuMind Database
Store and retrieve documents with proper schema design

### Production Patterns
Learn security, error handling, and best practices for MCP servers

---

## 🚀 Pre-Built Demo Available

**Today's demos use pre-built code branches**

Instead of watching code being typed (which takes 3-6x longer), we'll:
- **Checkout** complete, working code
- **Walk through** the implementation together
- **Run** live demonstrations
- **Discuss** design decisions

This lets us focus on **understanding** rather than **waiting**

---

## Demo Workflow

```bash
# Quick checkout (30 seconds)
git checkout session-4-complete
```

| Phase | Time | Focus |
|-------|------|-------|
| Checkout branch | 30 sec | Get the code |
| Code walk-through | 10-15 min | Understand structure |
| Live demo | 10-15 min | See it work |
| Q&A | 5-10 min | Discussion |

**Total: 35 minutes** (fits our allocation!)

---

# Introduction & Database Connectivity

## Recap: Session 3 Foundations

What we built last session:

### DocuMind Vision
AI-powered chatbot for company knowledge bases

### Claude Skills
Created document-processor skill for file handling

### Subagents
Built summarizer subagent for content analysis

### Hooks Automation
Configured automatic formatting and validation

### Project Structure
Initialized foundation with Skills, Subagents, and Hooks

---

## Today's Goal: Data Persistence

Moving from file-based processing to database storage

**The Challenge:**
- Our documents currently live only in files
- No persistent storage or retrieval
- Can't query across multiple documents
- No structured metadata

**Today's Solution:**
- Connect DocuMind to Supabase database
- Store documents with full metadata
- Enable structured queries via MCP
- Support multi-agent coordination

**By the end:** DocuMind will store and retrieve documents from a real database

---

## Why Database Connectivity Matters

Real-world AI applications need persistent data

### Before Database
- Documents disappear when session ends
- No search across document corpus
- Limited metadata tracking
- Single-agent file operations

### With Database
- Persistent document storage
- Rich querying capabilities
- Structured metadata and relationships
- Multi-agent coordination via shared state

**Database connectivity transforms DocuMind from prototype to production-ready system**

---

# MCP Architecture

## What is Model Context Protocol?

**MCP = Standardized way to connect AI models to external systems**

Developed by Anthropic to solve a common problem:

### The Problem
Every integration required custom code:
- Different APIs for every database
- Unique patterns for each tool
- No standard way to expose context
- Difficult to share and reuse

### The Solution: MCP
Universal protocol for:
- Connecting Claude to external data
- Exposing tools and resources
- Standardizing authentication
- Enabling interoperability

**Think of MCP as USB for AI: one standard interface for everything**

---

## MCP Components

Four core primitives that make MCP powerful:

### 1. Servers
Host functionality and expose it to Claude
- Run as separate processes
- Maintain their own state
- Handle authentication
- Example: Supabase MCP server

### 2. Resources
Provide context and data to Claude
- Read-only information
- Example: database schema, documentation

### 3. Tools
Enable Claude to take actions
- Executable functions
- Example: insert_document, query_database

### 4. Prompts
Offer reusable prompt templates
- Pre-defined workflows
- Example: "Summarize all documents from last week"

---

## MCP Tools vs Claude Code Tools

### Understanding the Distinction

**MCP Tools** = Access to **EXTERNAL** systems
- `mcp__supabase__query` → Talks to Supabase servers
- `mcp__github__create_issue` → Talks to GitHub API
- Data lives somewhere else (cloud databases, remote APIs)

**Claude Code Tools** = **LOCAL** work on your machine
- `Write` tool → Creates files in your workspace
- `Bash` tool → Runs commands in your terminal
- `Task` tool → Spawns agents that work locally

### The Key Insight

When you call an MCP tool like `mcp__supabase__insert`:
- Your request goes **over the network** to Supabase
- Supabase **executes the operation** on their servers
- Result comes **back to Claude**

When you use Claude's `Write` tool:
- Claude **writes directly** to your local filesystem
- No network call to external service
- File appears **in your workspace**

**Think of it this way:** MCP = phone calls to external services. Claude Code tools = working at your own desk.

---

## MCP vs Traditional APIs

Key differences that make MCP special:

| Traditional API | Model Context Protocol |
|-----------------|------------------------|
| Direct HTTP calls | Standardized protocol |
| Custom integration per service | Universal client interface |
| No AI-aware design | Built for LLM consumption |
| Manual request formatting | Automatic parameter handling |
| Limited context sharing | Rich context resources |
| Complex error handling | Standardized error patterns |

**MCP abstracts complexity while providing AI-optimized interfaces**

---

## MCP Architecture Diagram

```
┌─────────────────────────────────────────────────────┐
│                   Claude Code                       │
│                  (MCP Client)                       │
└─────────────────┬───────────────────────────────────┘
                  │
                  │ MCP Protocol (JSON-RPC)
                  │
        ┌─────────┴─────────────────────────┐
        │                                    │
┌───────▼────────┐                 ┌────────▼─────────┐
│   Supabase     │                 │  documind-mcp    │
│   MCP Server   │                 │  (Custom Server) │
├────────────────┤                 ├──────────────────┤
│ Resources:     │                 │ Resources:       │
│ - Tables       │                 │ - Doc Schema     │
│ - Schema       │                 │ - Stats          │
├────────────────┤                 ├──────────────────┤
│ Tools:         │                 │ Tools:           │
│ - query        │                 │ - upload_doc     │
│ - insert       │                 │ - search_docs    │
│ - update       │                 │ - get_metadata   │
│ - delete       │                 │                  │
└────────┬───────┘                 └─────────┬────────┘
         │                                   │
         ▼                                   ▼
  ┌─────────────┐                    ┌──────────────┐
  │  Supabase   │                    │ Application  │
  │  PostgreSQL │                    │    Logic     │
  └─────────────┘                    └──────────────┘
```

---

## MCP Security Model

How MCP keeps your data safe:

### Authentication
- API keys stored in environment variables
- Never exposed in code or prompts
- Separate credentials per server
- Configurable access scopes

### Authorization
- Server-level permission checks
- Tool-level access control
- Read vs write capabilities
- Resource filtering by permission

### Network Security
- Servers run in isolated processes
- Communication via secure channels
- No direct network exposure
- Proxy through Claude Code

**Best Practice:** Always use environment variables for secrets

---

## Installing MCP Servers

Two ways to add MCP servers to Claude Code:

### Pre-built Servers (Using Skill)
```bash
# Step 1: Configure environment variables in .env
cp env.example .env
# Edit .env with your Supabase credentials:
# - SUPABASE_URL, SUPABASE_ANON_KEY, SUPABASE_SERVICE_KEY (from Settings → API)
# - SUPABASE_ACCESS_TOKEN (from Account Settings → Access Tokens)

# Step 2: In Claude Code, type:
# "Use the supabase mcp installer skill to install supabase mcp"

# Verify installation - type /mcp in Claude Code
```
*Note: The skill handles the correct package name and authentication automatically*

### Custom Servers
```bash
# Register your own MCP server
claude mcp add --transport stdio documind-mcp -- python3 src/mcp/server.py

# Or with Node.js
claude mcp add --transport stdio documind-mcp -- node src/mcp/server.js
```

**Configuration stored in:** `.mcp.json` in your project root (shared via git)

---

# Supabase MCP Server

## What is Supabase?

**Supabase = Open source Firebase alternative built on PostgreSQL**

Perfect for DocuMind because it provides:

### Database
PostgreSQL with full SQL capabilities

### Vector Storage
pgvector extension for embeddings (Session 8)

### Authentication
Built-in user management (optional for DocuMind)

### Real-time
Subscriptions for live updates

### Storage
File storage for documents

### Edge Functions
Serverless compute (optional)

**Why Supabase?** Everything we need in one platform, free tier for learning

---

## Setting Up Supabase

Create your Supabase project:

**Step 1:** Sign up at supabase.com (free tier)

**Step 2:** Create new project
- Choose project name: "documind-db"
- Set database password (save securely!)
- Select region (closest to you)
- Wait 2 minutes for provisioning

**Step 3:** Get credentials
- Navigate to Settings → API
- Copy "Project URL": `https://xxxxx.supabase.co`
- Copy "anon public" API key

**Step 4:** Create documents table (next slide)

---

## DocuMind Database Schema

Initial schema for Session 4:

```sql
-- Create documents table
CREATE TABLE documents (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  title TEXT NOT NULL,
  content TEXT NOT NULL,
  file_type VARCHAR(50),
  source_url TEXT,
  uploaded_by TEXT,
  created_at TIMESTAMPTZ DEFAULT NOW(),
  updated_at TIMESTAMPTZ DEFAULT NOW(),
  metadata JSONB
);

-- Create index for faster searches
CREATE INDEX documents_title_idx ON documents(title);
CREATE INDEX documents_created_at_idx ON documents(created_at DESC);

-- Add full-text search (we'll use this in Session 6)
ALTER TABLE documents ADD COLUMN search_vector tsvector
  GENERATED ALWAYS AS (to_tsvector('english', title || ' ' || content)) STORED;
CREATE INDEX documents_search_idx ON documents USING gin(search_vector);
```

**Note:** We'll evolve this schema in future sessions (chunks, embeddings)

---

## Installing Supabase MCP

Connect Claude Code to your Supabase database:

**Step 1:** Configure environment variables in `.env`
```bash
# Copy env.example to .env and fill in:
SUPABASE_URL=https://xxxxx.supabase.co           # Settings → API
SUPABASE_ANON_KEY=eyJxxxxx                        # Settings → API Keys
SUPABASE_SERVICE_KEY=eyJxxxxx                     # Settings → API Keys
SUPABASE_ACCESS_TOKEN=sbp_xxxxx                   # Account Settings → Access Tokens (PAT)
```
⚠️ The Personal Access Token (PAT) is **required** for MCP authentication!

**Step 2:** Install using the skill
In Claude Code, type:
```
Use the supabase mcp installer skill to install supabase mcp
```

**Step 3:** Verify installation
Type `/mcp` in Claude Code to check server status

**Step 4:** Test connection
Ask Claude Code: "List all tables in my Supabase database"

**Expected output:** Should see "documents" table

---

## Live Demo

### DEMO: Supabase MCP in Action

Let's explore database connectivity from Claude Code

**Part 1:** Install and configure Supabase MCP
```bash
# First, configure .env with your Supabase credentials
# Then in Claude Code, type:
# "Use the supabase mcp installer skill to install supabase mcp"
```

**Part 2:** Basic database operations
```
"List all tables in the database"
"Show me the schema for the documents table"
"Insert a test document"
"Query all documents"
```

**Part 3:** See results
- Claude Code uses Supabase MCP automatically
- Natural language → SQL queries
- Structured data returned
- No manual API calls needed

**This is the power of MCP: natural language database access**

---

## Supabase MCP Tools

What the Supabase MCP server provides:

### Available Tools

**query**
- Execute SELECT queries
- Filter, sort, paginate
- Join across tables
- Example: "Show me documents created today"

**insert**
- Add new records
- Batch inserts supported
- Returns created record
- Example: "Insert a document with title 'Policy Manual'"

**update**
- Modify existing records
- Filter by conditions
- Returns updated count
- Example: "Update document title where id = xxx"

**delete**
- Remove records
- Requires filters (safety)
- Returns deleted count
- Example: "Delete documents older than 1 year"

---

## Querying with Supabase MCP

Natural language queries → SQL:

### Simple Query
```
You: "Show me all documents"
Claude: Uses Supabase MCP query tool
SQL: SELECT * FROM documents ORDER BY created_at DESC
```

### Filtered Query
```
You: "Find documents with 'policy' in the title"
Claude: Uses Supabase MCP query tool with filter
SQL: SELECT * FROM documents WHERE title ILIKE '%policy%'
```

### Complex Query
```
You: "Show documents uploaded by Alice in the last week"
Claude: Uses Supabase MCP query tool with multiple filters
SQL: SELECT * FROM documents
     WHERE uploaded_by = 'Alice'
     AND created_at > NOW() - INTERVAL '7 days'
```

**Claude Code translates your intent into proper SQL**

---

# Custom MCP Server

## Why Build Custom MCP Servers?

When pre-built servers aren't enough:

### Domain-Specific Logic
- Business rules and validation
- Custom workflows
- Specialized data transformations

### Integration Complexity
- Combine multiple services
- Complex authentication flows
- Custom error handling

### Performance Optimization
- Caching strategies
- Batch operations
- Pre-processing

### Team Standards
- Enforce company patterns
- Standardize access patterns
- Audit and logging

**For DocuMind:** We'll build a custom MCP server for document-specific operations

---

## documind-mcp Server Design

Our custom MCP server will provide:

### Tools (Actions Claude can take)

**upload_document**
- Parameters: title, content, file_type, metadata
- Validates document format
- Extracts metadata
- Stores in Supabase
- Returns document ID

**search_documents**
- Parameters: query, limit, filters
- Searches title and content
- Returns ranked results
- Includes metadata

**get_document**
- Parameters: document_id
- Retrieves full document
- Returns content + metadata
- Handles missing documents

---

## Building documind-mcp: Setup

Project structure for our custom MCP server:

```
documind/
├── src/
│   └── mcp/
│       ├── server.py          # Main MCP server
│       ├── tools.py           # Tool implementations
│       ├── resources.py       # Resource providers
│       └── config.py          # Configuration
├── .env                       # Environment variables (not in git!)
└── requirements.txt           # Python dependencies
```

**Install MCP SDK:**
```bash
pip install mcp anthropic-mcp-server supabase-py
```

---

## Building documind-mcp: Server Implementation

Creating the MCP server structure:

### Demo Prompt: Create MCP Server Structure

**Natural Language Instruction to Claude:**
```
Create a custom MCP server for DocuMind at src/mcp/server.py with the following specifications:

1. Import the necessary MCP server components from the mcp.server module (Server, stdio_server)
2. Import asyncio for asynchronous operations
3. Import the tool functions we'll implement: upload_document, search_documents, get_document

4. Initialize an MCP Server instance named "documind-mcp"

5. Create a list_tools() handler decorated with @app.list_tools() that returns tool definitions:
   - Define an "upload_document" tool with:
     * Description: "Upload a document to DocuMind knowledge base"
     * Input schema with properties: title (string), content (string), file_type (string), metadata (object)
     * Required fields: title and content
   - Leave room for additional tool definitions (search_documents, get_document)

6. Create a call_tool() handler decorated with @app.call_tool() that:
   - Takes name (string) and arguments (dict) as parameters
   - Routes to upload_document function when name matches "upload_document"
   - Leaves placeholder comments for routing other tools

Follow MCP server best practices and use proper async/await patterns.
```

### Demo: Understanding the Structure

Here's how this was built. The structure includes:
- Set up the proper imports and server initialization
- Define the tool schemas with validation
- Implement the routing logic for tool calls

### Demo: Review

After Claude creates the file, review together:

**Key Architecture Points:**
1. **Server Initialization**: The Server("documind-mcp") creates a named MCP server instance
2. **Tool Registration**: @app.list_tools() decorator exposes available tools to Claude Code
3. **Input Schemas**: JSON Schema validates parameters before tool execution
4. **Tool Router**: @app.call_tool() dispatcher routes requests to correct handler functions
5. **Async Design**: All operations use async/await for non-blocking I/O

**Test the structure:**
```bash
# Check syntax
python -m py_compile src/mcp/server.py

# View the created file
cat src/mcp/server.py
```

---

## Building documind-mcp: Tool Implementation

Implementing the upload_document tool:

### Demo Prompt: Create Upload Document Tool

**Natural Language Instruction to Claude:**
```
Create the upload_document tool implementation at src/mcp/tools.py with the following specifications:

1. Import necessary modules:
   - supabase (create_client)
   - os (for environment variables)
   - datetime (for timestamps)

2. Initialize Supabase client using environment variables:
   - SUPABASE_URL
   - SUPABASE_KEY

3. Implement an async function upload_document with parameters:
   - title (str, required)
   - content (str, required)
   - file_type (str, optional, default None)
   - metadata (dict, optional, default None)

4. Add comprehensive docstring: "Upload document with validation and metadata extraction"

5. Implement validation logic:
   - Check if title and content are provided
   - Return error dict if either is missing: {"error": "Title and content are required"}

6. Prepare document dictionary with:
   - title, content fields
   - file_type (use "text/plain" if not provided)
   - created_at (current UTC timestamp in ISO format)
   - metadata (use empty dict if not provided)

7. Insert document into Supabase "documents" table:
   - Use supabase.table("documents").insert(doc).execute()
   - Store the result

8. Return success response if data exists:
   - success: True
   - document_id: extracted from result.data[0]["id"]
   - message: "Document '{title}' uploaded successfully"

9. Return error response if insertion failed:
   - error: "Failed to upload document"

Follow Python best practices and proper error handling patterns.
```

### Demo: Understanding the Upload Tool

Here's how this was built. The implementation includes:
- Set up Supabase connection with environment variables
- Add input validation
- Prepare document data structure
- Handle database insertion with error checking

### Demo: Review

After Claude creates the file, review together:

**Key Implementation Points:**
1. **Environment Security**: Credentials loaded from environment, never hardcoded
2. **Input Validation**: Early return with error message if required fields missing
3. **Default Values**: Sensible defaults for optional parameters (file_type, metadata)
4. **Timestamp Handling**: UTC timestamps in ISO format for consistency
5. **Error Responses**: Structured error objects for client handling
6. **Success Responses**: Include document_id for subsequent operations

**Test the tool:**
```bash
# Check syntax
python -m py_compile src/mcp/tools.py

# View the implementation
cat src/mcp/tools.py
```

---

## Building documind-mcp: Search Tool

Implementing semantic document search:

### Demo Prompt: Create Search Documents Tool

**Natural Language Instruction to Claude:**
```
Add the search_documents function to src/mcp/tools.py with the following specifications:

1. Create an async function search_documents with parameters:
   - query (str, required)
   - limit (int, optional, default 10)
   - filters (dict, optional, default None)

2. Add docstring: "Search documents by title and content"

3. Build Supabase query progressively:
   - Start with: supabase.table("documents").select("*")
   - Store in a query builder variable

4. Implement full-text search if query provided:
   - Use PostgreSQL tsvector search on "search_vector" column
   - Method: qb.text_search("search_vector", query)

5. Apply optional filters if provided:
   - Check if "file_type" exists in filters dict
     * If yes, add equality filter: qb.eq("file_type", filters["file_type"])
   - Check if "uploaded_by" exists in filters dict
     * If yes, add equality filter: qb.eq("uploaded_by", filters["uploaded_by"])

6. Add ordering and pagination:
   - Order by created_at descending (newest first)
   - Apply limit to control result count

7. Execute the query and store result

8. Return structured response:
   - success: True
   - count: number of documents found (len(result.data))
   - documents: the actual result data array

Use method chaining for clean query construction.
```

### Demo: Understanding the Search Tool

Here's how this was built. The implementation includes:
- Build the query progressively using method chaining
- Add full-text search capability
- Apply conditional filters
- Return structured results

### Demo: Review

After Claude adds the function, review together:

**Key Implementation Points:**
1. **Query Builder Pattern**: Progressive query construction with method chaining
2. **Full-Text Search**: Leverages PostgreSQL's tsvector for semantic search
3. **Conditional Filters**: Only applies filters if provided (optional parameters)
4. **Default Limit**: Prevents accidentally returning massive result sets
5. **Newest First**: DESC ordering shows most recent documents first
6. **Structured Response**: Includes metadata (count) along with results

**Test the search:**
```bash
# Verify the complete tools.py file
python -m py_compile src/mcp/tools.py

# View both functions
cat src/mcp/tools.py
```

---

## Building documind-mcp: Resources

Exposing document statistics as resources:

### Demo Prompt: Create MCP Resources

**Natural Language Instruction to Claude:**
```
Create MCP resource providers at src/mcp/resources.py with the following specifications:

1. Import necessary modules (app from server.py, supabase client, json, datetime)

2. Create a list_resources() function decorated with @app.list_resources():
   - Return an array of resource definitions
   - First resource:
     * uri: "documind://stats"
     * name: "Document Statistics"
     * description: "Overall statistics about the document corpus"
     * mimeType: "application/json"
   - Second resource:
     * uri: "documind://schema"
     * name: "Database Schema"
     * description: "Current database schema definition"
     * mimeType: "application/json"

3. Create a read_resource() function decorated with @app.read_resource():
   - Takes uri (string) parameter
   - For "documind://stats" uri:
     * Query Supabase documents table to get count
     * Return MCP resource response format with:
       - contents array containing single object
       - uri: the requested uri
       - mimeType: "application/json"
       - text: JSON string with:
         · total_documents: count from query result
         · last_updated: current UTC timestamp in ISO format

Note: This provides read-only context resources that Claude can access without tool calls.
```

### Demo: Understanding Resource Providers

Here's how this was built. The resource providers include:
- Define available resources with proper MCP metadata
- Implement read handlers for each resource URI
- Return statistics in standardized format

### Demo: Review

After Claude creates the file, review together:

**Key Concepts:**
1. **Resources vs Tools**: Resources provide read-only context, tools perform actions
2. **URI Scheme**: Custom documind:// scheme for resource identification
3. **Mime Types**: Proper content type declaration for clients
4. **Response Format**: MCP standard format with contents array
5. **JSON Serialization**: Data converted to JSON text for transmission

**Test the resources:**
```bash
# Check syntax
python -m py_compile src/mcp/resources.py

# View the implementation
cat src/mcp/resources.py
```

---

## Registering Custom MCP Server

Add documind-mcp to Claude Code:

**Step 1:** Register with Claude Code
```bash
claude mcp add --transport stdio documind-mcp -- python3 src/mcp/server.py
```

**Step 2:** Verify registration
Type `/mcp` in Claude Code to confirm both servers are active

Or in terminal:
```bash
claude mcp list
# Should show both 'supabase' and 'documind-mcp'
```

**Step 3:** Test
```
Ask Claude: "Upload a test document using documind-mcp"
```

---

## Live Demo

### DEMO: Custom MCP Server

Let's explore and use documind-mcp

**Part 1:** Let's walk through the server files
- Explore `src/mcp/server.py` with tool definitions
- Review `src/mcp/tools.py` with `upload_document` and `search_documents` implementations
- Understand `src/mcp/resources.py` for stats and schema resources

**Part 2:** Register with Claude Code
```bash
claude mcp add --transport stdio documind-mcp -- python3 src/mcp/server.py
```

**Part 3:** Test the tools with natural language
```
"Upload a document titled 'Company Policy' with documind-mcp"
"Search for documents containing 'policy'"
"Show me document statistics from documind resources"
```

**Demo: Review the Registration**

After registration, verify:
1. **MCP Server List**: Run `claude mcp list` to see documind-mcp registered
2. **Tool Discovery**: Ask Claude "What tools does documind-mcp provide?"
3. **Resource Access**: Ask Claude "Show me documind resource URIs"
4. **Integration Test**: Upload and search to verify end-to-end workflow

**See:** Custom tools working alongside Supabase MCP

---

## MCP Server Best Practices

Write production-ready MCP servers:

### Error Handling
- Validate all inputs
- Return structured errors
- Include helpful error messages
- Log errors for debugging

### Performance
- Use async/await for I/O operations
- Implement caching where appropriate
- Batch operations when possible
- Set reasonable timeouts

### Security
- Never log sensitive data
- Validate permissions
- Sanitize inputs (prevent injection)
- Use environment variables for secrets

### Documentation
- Clear tool descriptions
- Document all parameters
- Provide usage examples
- Include error scenarios

---

# Agent-to-Agent Communication

## Multi-Agent Coordination Patterns

How agents work together via shared state:

### Pattern 1: Message Passing
Agents communicate via database records
- Agent A writes message to "messages" table
- Agent B reads and processes message
- Agent B updates status in database

### Pattern 2: Shared Memory
Agents coordinate via shared state
- Agent A stores progress in "tasks" table
- Agent B checks progress before acting
- Both update same record

### Pattern 3: Event-Driven
Agents trigger each other via events
- Agent A completes task, writes event
- Agent B subscribes to events
- Agent B reacts when event appears

**For DocuMind:** We'll use Pattern 2 (Shared Memory via Supabase)

---

## Why Agent-to-Agent Communication?

Break complex tasks into specialized roles:

### Without A2A
**Single Agent Approach:**
- One agent does everything
- Context grows enormous
- Prone to confusion
- Difficult to debug
- No specialization

### With A2A
**Multi-Agent Approach:**
- Agent 1: Document uploader
- Agent 2: Metadata enricher
- Agent 3: Quality validator
- Agent 4: Index updater

**Benefits:** Focused expertise, parallel execution, easier debugging

---

## DocuMind Multi-Agent Workflow

Document upload via agent coordination:

```
┌──────────────────────────────────────────────────────┐
│                  Document Upload                     │
└──────────────┬───────────────────────────────────────┘
               │
               ▼
        ┌──────────────┐
        │   Agent 1    │ Upload document
        │   Uploader   │ via documind-mcp
        └──────┬───────┘
               │ Writes: status = "uploaded"
               ▼
        ┌──────────────┐
        │   Agent 2    │ Extracts metadata
        │   Enricher   │ (author, date, topics)
        └──────┬───────┘
               │ Updates: status = "enriched"
               ▼
        ┌──────────────┐
        │   Agent 3    │ Validates content
        │  Validator   │ (format, quality)
        └──────┬───────┘
               │ Updates: status = "validated"
               ▼
        ┌──────────────┐
        │   Agent 4    │ Updates search index
        │   Indexer    │ (full-text search)
        └──────┬───────┘
               │ Updates: status = "ready"
               ▼
           ┌────────┐
           │ READY  │ Document available for queries
           └────────┘
```

---

## Implementing A2A: Task Coordination Table

Add table for agent coordination:

```sql
-- Create tasks table for agent coordination
CREATE TABLE processing_tasks (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  document_id UUID REFERENCES documents(id) ON DELETE CASCADE,
  task_type VARCHAR(50) NOT NULL, -- 'upload', 'enrich', 'validate', 'index'
  status VARCHAR(50) DEFAULT 'pending', -- 'pending', 'in_progress', 'completed', 'failed'
  assigned_to VARCHAR(100), -- Agent name
  started_at TIMESTAMPTZ,
  completed_at TIMESTAMPTZ,
  error_message TEXT,
  metadata JSONB,
  created_at TIMESTAMPTZ DEFAULT NOW()
);

-- Index for fast status checks
CREATE INDEX tasks_status_idx ON processing_tasks(status);
CREATE INDEX tasks_document_idx ON processing_tasks(document_id);
```

**Agents check this table to coordinate their work**

---

## Implementing A2A: Uploader Agent

Agent 1 starts the workflow:

### Demo Prompt: Create Uploader Agent

**Natural Language Instruction to Claude:**
```
Create an uploader agent at src/agents/uploader.py with the following specifications:

1. Import necessary modules:
   - supabase (create_client)
   - os (for environment variables)

2. Initialize Supabase client using SUPABASE_URL and SUPABASE_KEY from environment

3. Create an async function upload_and_initiate with parameters:
   - title (str): Document title
   - content (str): Document content
   - file_type (str): File type/extension

4. Add comprehensive docstring: "Upload document and create processing tasks"

5. Implement two-step workflow:

   Step 1 - Upload Document:
   - Call the documind-mcp tool "upload_document" via MCP client
   - Pass title, content, and file_type as arguments
   - Extract document_id from the result

   Step 2 - Create Processing Tasks:
   - Create a list of three task dictionaries for other agents:
     * Task 1: {"document_id": document_id, "task_type": "enrich", "status": "pending"}
     * Task 2: {"document_id": document_id, "task_type": "validate", "status": "pending"}
     * Task 3: {"document_id": document_id, "task_type": "index", "status": "pending"}
   - Insert all tasks into Supabase "processing_tasks" table using batch insert

6. Print confirmation message: "Document {document_id} uploaded, tasks created for other agents"

This agent initiates the multi-agent workflow by uploading the document and creating coordination tasks.
```

### Demo: Understanding the Uploader Agent

Here's how this was built. The uploader agent includes:
- Set up Supabase connection
- Implement MCP tool call for document upload
- Create coordination tasks for downstream agents
- Add appropriate logging

### Demo: Review

After Claude creates the file, review together:

**Key Coordination Points:**
1. **MCP Integration**: Uses documind-mcp tool instead of direct database access
2. **Task Creation**: Initiates workflow by creating pending tasks for other agents
3. **Batch Operations**: Inserts all three tasks in single database operation
4. **Document ID**: Critical link between document and processing tasks
5. **Status Tracking**: "pending" status signals other agents to pick up work

**Test the uploader:**
```bash
# Check syntax
python -m py_compile src/agents/uploader.py

# View the implementation
cat src/agents/uploader.py
```

---

## Implementing A2A: Enricher Agent

Agent 2 waits for work and enriches metadata:

### Demo Prompt: Create Enricher Agent

**Natural Language Instruction to Claude:**
```
Create an enricher agent at src/agents/enricher.py with the following specifications:

1. Import necessary modules:
   - asyncio (for async operations and sleep)
   - supabase client (assume imported from shared config)

2. Create an async function watch_and_enrich():
   - Add docstring: "Watch for pending enrichment tasks"
   - Implement infinite while loop for continuous monitoring

3. Inside the loop, implement task discovery and processing:

   Step 1 - Find Pending Tasks:
   - Query processing_tasks table where task_type = "enrich" AND status = "pending"
   - Limit to 1 task
   - Store result

   Step 2 - Process if task found:
   If task exists (result.data is not empty):

   a) Mark task as in progress:
      - Update processing_tasks where id = task["id"]
      - Set: status = "in_progress", assigned_to = "enricher", started_at = "now()"

   b) Retrieve document:
      - Query documents table where id = task["document_id"]
      - Use .single() to get one result

   c) Enrich metadata:
      - Call helper function extract_metadata() with document content
      - Store enriched metadata result (assume this function extracts topics, entities, etc.)

   d) Update document with enriched data:
      - Update documents table where id = task["document_id"]
      - Set metadata field to enriched_metadata

   e) Mark task complete:
      - Update processing_tasks where id = task["id"]
      - Set: status = "completed", completed_at = "now()"

   f) Print confirmation: "Document {task['document_id']} enriched"

   Step 3 - Wait before next check:
   - Sleep for 5 seconds using asyncio.sleep(5)
   - Add comment: "Check every 5 seconds"

This agent continuously monitors for enrichment tasks and processes them independently.
```

### Demo: Understanding the Enricher Agent

Here's how this was built. The enricher agent includes:
- Implement polling loop for task discovery
- Add proper status transitions (pending → in_progress → completed)
- Handle document retrieval and metadata enrichment
- Include appropriate delays to avoid database hammering

### Demo: Review

After Claude creates the file, review together:

**Key Agent Pattern Points:**
1. **Polling Loop**: Continuous monitoring with sleep prevents CPU spinning
2. **Status Transitions**: Clear lifecycle (pending → in_progress → completed)
3. **Agent Assignment**: Records which agent is processing each task
4. **Timestamps**: Tracks when task started and completed
5. **Isolation**: Agent only processes tasks of its type ("enrich")
6. **Idempotency**: Can safely re-run if agent crashes

**Test the enricher:**
```bash
# Check syntax
python -m py_compile src/agents/enricher.py

# View the implementation
cat src/agents/enricher.py
```

**Discuss:** How would you add error handling and retry logic?

---

## Implementing A2A: Coordination Benefits

What we gain from agent coordination:

### Parallel Processing
- Multiple agents work simultaneously
- Each focuses on their specialty
- Overall throughput increases

### Fault Tolerance
- If one agent fails, others continue
- Failed tasks can be retried
- No single point of failure

### Clear Responsibilities
- Each agent has one job
- Easier to debug and maintain
- Can replace agents independently

### Scalability
- Add more agents as needed
- Scale specific bottlenecks
- Distribute load naturally

**Trade-off:** Additional coordination complexity vs capability gains

---

## Live Demo

### DEMO: Multi-Agent Coordination

Let's explore and test agent coordination

**Part 1:** Create processing_tasks table (keep SQL as reference)
```sql
-- Run SQL in Supabase dashboard - this creates the coordination table
CREATE TABLE processing_tasks ...
```

**Part 2:** Let's walk through the three agents
- Explore the uploader agent implementation
- Review the enricher agent logic
- Understand the validator agent pattern (adapted from enricher)

**Part 3:** Watch coordination in action

**Demo: Natural Language Testing**
```
"Start the uploader agent and upload a document titled 'Employee Handbook' with sample content"

"Show me the processing_tasks table - I should see three pending tasks"

"Start the enricher agent to process the enrich task"

"Check the document metadata - it should now be enriched"

"Show me the updated processing_tasks table - enrich task should be completed"
```

**Demo: Review the Coordination**

After running the agents, review:
1. **Task Lifecycle**: Watch status transitions (pending → in_progress → completed)
2. **Agent Assignment**: See which agent processed each task
3. **Timestamps**: Review started_at and completed_at for timing analysis
4. **Document Evolution**: Compare document before/after enrichment
5. **Error Handling**: Discuss what happens if an agent crashes mid-task

**Observe:** Agents coordinating via shared Supabase state with no direct communication

---

## Error Handling in A2A

Handling failures gracefully:

### Retry Logic
```python
MAX_RETRIES = 3

async def process_with_retry(task_id: str):
    for attempt in range(MAX_RETRIES):
        try:
            result = await process_task(task_id)
            return result
        except Exception as e:
            if attempt == MAX_RETRIES - 1:
                # Final failure
                mark_failed(task_id, str(e))
                raise
            else:
                # Retry with backoff
                await asyncio.sleep(2 ** attempt)
```

### Dead Letter Queue
```python
# Move failed tasks after max retries
supabase.table("failed_tasks").insert({
    "original_task": task,
    "error": error_message,
    "failed_at": datetime.utcnow()
}).execute()
```

---

## MCP as Coordination Layer

Using MCP for agent communication:

### Why MCP for A2A?
- Standardized protocol
- Built-in error handling
- Natural language interface
- Tool discoverability

### Pattern
```python
# Agent A calls tool via MCP
await mcp_client.call_tool(
    "documind-mcp",
    "create_task",
    {"document_id": doc_id, "task_type": "enrich"}
)

# Agent B queries via MCP
tasks = await mcp_client.call_tool(
    "documind-mcp",
    "get_pending_tasks",
    {"task_type": "enrich"}
)
```

**MCP provides clean abstraction over coordination logic**

---

# DocuMind Integration

## Integrating MCP with DocuMind

Connecting all the pieces:

### What We've Built
- Supabase database with documents table
- Supabase MCP server for direct database access
- Custom documind-mcp server with specialized tools
- Multi-agent coordination via processing_tasks

### Integration Points

**1. Document Upload Flow**
Skills → Subagents → MCP Tools → Database

**2. Document Retrieval**
Query → MCP Search → Results → Response

**3. Agent Coordination**
Task Creation → Pending Tasks → Agent Processing → Completion

---

## Updated DocuMind Architecture

How Session 4 components fit:

```
┌─────────────────────────────────────────────────────────┐
│                    DocuMind System                      │
├─────────────────────────────────────────────────────────┤
│                                                         │
│  ┌──────────────┐      ┌──────────────┐               │
│  │   Document   │──────│  Processing  │               │
│  │   Skills     │      │   Subagents  │               │
│  │   (S3)       │      │   (S3)       │               │
│  └──────┬───────┘      └──────┬───────┘               │
│         │                      │                        │
│         └──────────┬───────────┘                        │
│                    │ via MCP                            │
│                    ▼                                    │
│         ┌─────────────────────┐                        │
│         │   MCP Layer (S4)    │                        │
│         ├─────────────────────┤                        │
│         │ - Supabase MCP      │                        │
│         │ - documind-mcp      │                        │
│         └─────────┬───────────┘                        │
│                   │                                     │
│                   ▼                                     │
│  ┌──────────────────────────────────┐                 │
│  │   Supabase PostgreSQL (S4)       │                 │
│  │   - documents table              │                 │
│  │   - processing_tasks table       │                 │
│  └──────────────────────────────────┘                 │
│                                                         │
│  Future: S5 (Multi-Agent), S6 (RAG), S8 (pgvector)    │
└─────────────────────────────────────────────────────────┘
```

---

## Testing the Integration

End-to-end workflow validation:

**Test 1: Single Document Upload**
```
"Upload a document titled 'Employee Handbook' with content about vacation policies"
```
- Uses document-processor skill
- Calls documind-mcp upload_document
- Stores in Supabase via MCP
- Returns document ID

**Test 2: Search Documents**
```
"Find all documents about vacation policies"
```
- Uses documind-mcp search_documents
- Queries Supabase full-text search
- Returns ranked results

**Test 3: Multi-Agent Processing**
- Upload document (creates tasks)
- Watch enricher agent process
- Verify metadata added
- Check task status changes

---

## Environment Variables Setup

Secure configuration for DocuMind:

```bash
# .env file (never commit to git!)
SUPABASE_URL=https://xxxxx.supabase.co
SUPABASE_KEY=your-anon-key-here

# Optional: Separate keys for different agents
AGENT_1_SUPABASE_KEY=uploader-key
AGENT_2_SUPABASE_KEY=enricher-key

# MCP server configuration
MCP_SERVER_HOST=localhost
MCP_SERVER_PORT=3000
```

**Load in code:**
```python
from dotenv import load_dotenv
load_dotenv()

supabase_url = os.environ["SUPABASE_URL"]
```

**Add to .gitignore:**
```
.env
.env.local
```

---

# Wrap-up & Preview

## Key Takeaways

What we learned today:

### MCP Architecture
Standardized protocol for connecting AI to external systems

### Supabase MCP
Direct database access via natural language queries

### Custom MCP Servers
Build specialized tools and resources for domain needs

### Agent-to-Agent Communication
Coordinate multiple agents via shared database state

### DocuMind Database
Persistent storage for documents with full metadata

### Production Patterns
Security, error handling, and best practices

---

## MCP Benefits Recap

Why MCP matters for AI applications:

| Before MCP | With MCP |
|------------|----------|
| Custom integration per service | Universal protocol |
| Manual API calls | Natural language tools |
| Complex error handling | Standardized errors |
| Difficult to share | Reusable servers |
| Limited context | Rich resources |

**MCP = Infrastructure layer for AI applications**

---

## Agent Coordination Patterns

Quick reference:

| Pattern | Use Case | Complexity |
|---------|----------|------------|
| **Message Passing** | Event-driven workflows | Medium |
| **Shared Memory** | Status tracking | Low |
| **Event-Driven** | Real-time coordination | High |

**DocuMind uses:** Shared Memory (Supabase tables)

**Future:** Could add Event-Driven via Supabase Real-time

---

## DocuMind Progress Tracker

Where we are and where we're going:

✅ **Session 3:** Foundation - Skills, Subagents, Hooks

✅ **Session 4 (Today):** Database - Supabase + documind-mcp

🔜 **Session 5 (Next):** Multi-agent processing pipeline with ClaudeFlow

📋 **Session 6:** RAG implementation for intelligent Q&A

📋 **Session 7:** PDF/DOCX advanced parsing

📋 **Session 8:** pgvector semantic search

📋 **Session 9:** Conversation memory + learning

📋 **Session 10:** RAGAS/TruLens evaluation

**Each session builds on previous work**

---

## What's Next: Multi-Agent Systems

Preview of Session 5:

### ClaudeFlow Introduction
Orchestration framework for multi-agent systems

### Document Processing Pipeline
4-agent workflow:
- Agent 1: Extractor (reads files)
- Agent 2: Chunker (splits content)
- Agent 3: Embedder (generates vectors)
- Agent 4: Writer (stores in database)

### Parallel Execution
Process multiple documents simultaneously

### Performance Benefits
Compare parallel vs sequential processing

**We'll use today's MCP infrastructure for agent coordination**

---

## Practice Before Next Session

Reinforce today's learning:

**01.** Create your own custom MCP server for a different use case

**02.** Build a multi-agent workflow with at least 3 agents

**03.** Implement error handling and retry logic

**04.** Add more tools to documind-mcp (update_document, delete_document)

**05.** Experiment with Supabase Real-time subscriptions

**06.** Create database indexes for performance

**07.** Prepare questions for Session 5

**Bonus:** Add authentication to your MCP server

---

# Q&A and Additional Resources

## Questions?

Areas we can clarify:

### MCP Architecture
- Components and protocol
- Security model
- When to use MCP vs direct APIs

### Supabase
- Database setup
- Query patterns
- MCP server configuration

### Custom MCP Servers
- Server implementation
- Tool definitions
- Resource providers

### Agent Coordination
- Communication patterns
- Error handling
- Scaling strategies

### DocuMind
- Integration questions
- Database schema
- Future sessions

---

## Additional Resources

Links for further study:

### Model Context Protocol
- **Specification**: modelcontextprotocol.io/specification
- **MCP Servers**: github.com/modelcontextprotocol/servers
- **Python SDK**: pypi.org/project/mcp

### Supabase
- **Documentation**: supabase.com/docs
- **MCP Server**: github.com/supabase/mcp-server
- **PostgreSQL Guide**: supabase.com/docs/guides/database

### Custom MCP Development
- **Building Servers**: modelcontextprotocol.io/docs/building-servers
- **Best Practices**: modelcontextprotocol.io/docs/best-practices
- **Examples**: github.com/modelcontextprotocol/examples

### DocuMind Repository
- **Course Code**: github.com/your-repo/documind
- **Session 4 Branch**: git checkout session-4

---

## Session Complete

**Thank you for participating!**

### Remember:
- MCP = Universal protocol for AI integrations
- Supabase MCP = Direct database access
- Custom MCP = Domain-specific tools
- A2A = Multi-agent coordination patterns

### Next Session:
**Multi-Agent Systems: Coordinating Specialized AI Teams**

**Building:** 4-agent document processing pipeline with ClaudeFlow

**See you next time!**

---
