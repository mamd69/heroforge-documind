# HeroForge.AI Course: AI-Powered Software Development
## Lesson 5 Workshop: Multi-Agent Systems - Coordinating Specialized AI Teams

**Estimated Time:** 45-60 minutes\
**Difficulty:** Advanced\
**Prerequisites:** 
1. Completed Sessions 1-4 (Agentic Engineering fundamentals, Claude Code CLI, Skills/Subagents/Hooks, MCP & A2A)
2. Verify database state (see below)
3. If you have not already, sync course materials (see below)

---

### Verify Database State 
For this session, we need a stable database. If you suspect your database is messy from previous sessions, run this "Safe Reset" in your Supabase SQL Editor now. It ensures you have the documents table required for our agents.

```SQL

-- Run in Supabase SQL Editor
CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS documents (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  title TEXT NOT NULL,
  content TEXT NOT NULL,
  file_path TEXT,
  file_type TEXT,
  metadata JSONB DEFAULT '{}'::jsonb,
  created_at TIMESTAMPTZ DEFAULT now(),
  updated_at TIMESTAMPTZ DEFAULT now()
);
(This ensures no one gets "Table not found" errors when their agents try to write data.)
```
---

### Sync with Course Material
Before starting, ensure you have the latest workshop files.

Run these commands in your terminal:

```bash
# 1. Add the instructor repo as 'upstream' (only needs to be done once)
git remote add upstream [https://github.com/YOUR-USERNAME/heroforge-documind.git](https://github.com/YOUR-USERNAME/heroforge-documind.git) || true

# 2. Pull the latest changes from the instructor's main branch
git pull upstream main --no-rebase

# 3. If you run into merge conflicts that you can't resolve, you can reset 
#    your workshop file to the instructor's version (optional):
# git checkout upstream/main -- workshops/S5-Workshop.md
```
---

## Workshop Objectives

By completing this workshop, you will:
- [x] Design multi-agent architectures for complex workflows
- [x] Implement parallel agent execution with ClaudeFlow
- [x] Create specialized agents with distinct capabilities
- [x] Coordinate agent communication and synchronization
- [x] Build a document processing pipeline with multiple agents
- [x] Apply multi-agent patterns to DocuMind's ingestion system

---

## Before You Begin: Two Orchestration Layers (Critical Concept!)

This workshop involves TWO distinct orchestration systems. Understanding the difference is essential:

### Layer 1: Claude Flow (Development-Time Orchestration)

| Aspect | Description |
|--------|-------------|
| **What** | AI-powered development assistant with 54+ specialized agents |
| **When** | Runs while YOU are coding |
| **Purpose** | Helps you write code faster |
| **Example** | `npx claude-flow@alpha agent spawn --type coder` |

### Layer 2: DocuMind Pipeline (Runtime Orchestration)

| Aspect | Description |
|--------|-------------|
| **What** | Document processing workflow you're building |
| **When** | Runs when USERS upload documents |
| **Purpose** | Extract, chunk, embed, and store documents |
| **Example** | PDF → Text Extraction → Chunking → Embedding → Database |

### The Relationship
```
┌─────────────────────────────────────────────────────────┐
│  🏗️ Claude Flow (Development-Time)                      │
│  ─────────────────────────────────────                  │
│  YOU use Claude Flow to BUILD DocuMind's pipeline       │
│  Claude Flow makes development faster                   │
│  54+ agents help you code, test, review                 │
└─────────────────────────────────────────────────────────┘
                          │
                          │ builds
                          ▼
┌─────────────────────────────────────────────────────────┐
│  🏭 DocuMind Pipeline (Runtime)                         │
│  ─────────────────────────────────────                  │
│  The product USERS interact with                        │
│  Processes documents when users upload them             │
│  4 agents: Extractor → Chunker → Embedder → Writer      │
└─────────────────────────────────────────────────────────┘
```
### Analogy

- **Claude Flow** = Power tools (helps you build)
- **DocuMind Pipeline** = The house (what you're building)

You COULD build the house with hand tools (no Claude Flow), but power tools are faster.

---

## The DocuMind PRD: Your Specification

In this session, you'll use Claude Flow swarms to **generate** the DocuMind pipeline scripts using natural language commands. The specification for what you're building comes from the **DocuMind PRD** (Product Requirements Document).

### What You're Building

The DocuMind PRD specifies a **multi-agent document processing pipeline** with 5 components:

| Script | Role | PRD Reference |
|--------|------|---------------|
| `extractor.py` | Reads files, extracts raw text and metadata | Stage 1: Extractor Agent |
| `chunker.py` | Splits text into ~500-word semantic chunks | Stage 2: Chunker Agent |
| `embedder.py` | Generates vector embeddings via OpenAI API | Stage 3: Embedder Agent |
| `writer.py` | Stores documents and chunks in Supabase | Stage 4: Writer Agent |
| `orchestrate.py` | Coordinates the full pipeline with asyncio | Pipeline Coordinator |

### Where to Find the PRD

The PRD should be in your DocuMind project repository:

```bash
# In your DocuMind Codespace
cat docs/spec/documind-prd.md
```

If you don't have the PRD yet, you can find it here: https://github.com/mamd69/heroforge-documind/tree/main/docs/spec.

### Key PRD Sections for This Session

When generating pipeline scripts, reference these PRD sections:

1. **Multi-Agent Document Processing** (Key Features #8)
   - Parallel document ingestion
   - Specialized extraction for different formats
   - Automatic chunking and embedding generation

2. **Document Ingestion Flow** (Data Flows section)
   - Upload → Extract → Chunk → Embed → Store

3. **Technology Stack**
   - Python 3.10+
   - OpenAI `text-embedding-3-small` for embeddings
   - Supabase with pgvector for storage

### How You'll Use Claude Flow

Instead of copying pre-written code, you'll:

1. **Initialize a Claude Flow swarm** with specialized agents
2. **Describe what each script should do** in natural language
3. **Watch Claude generate the code** based on the PRD spec
4. **Test and verify** the generated scripts work

This mirrors real-world agentic development: you provide the spec, the AI generates the implementation.

---

## Module 1: Multi-Agent Architecture Design (15 minutes)

### Concept Review

**What is Multi-Agent Architecture?**

Multi-agent architecture is a pattern where multiple specialized AI agents work together to accomplish complex tasks. Instead of one "do-everything" agent, you create a team of experts that coordinate via communication and shared memory.

**When to Use Multiple Agents:**
- Task requires specialized expertise in different domains
- Parallel processing can significantly speed up execution
- Workflow has distinct, separable stages
- Failure isolation is important (one agent fails ≠ system fails)

**Agent Specialization Patterns:**

| Pattern | Description | Example |
|---------|-------------|---------|
| **Expert** | Deep knowledge in specific domain | Security auditor, API specialist |
| **Worker** | Executes specific type of task repeatedly | Document parser, file converter |
| **Coordinator** | Manages workflow and delegates to others | Pipeline orchestrator, task scheduler |
| **Validator** | Checks output quality and correctness | Test runner, data validator |

**Communication Topologies:**

```
MESH (Peer-to-Peer)          HIERARCHICAL (Tree)          STAR (Central Hub)
    A---B---C                      Coordinator                    D
    |\ /|\ /|                        /  |  \                    / | \
    | X | X |                       /   |   \                  A  B  C
    |/ \|/ \|                      A    B    C
    D---E---F                     /|   /|\   |\

Best for: Collaboration      Best for: Command chains    Best for: Central control
```

---

### Exercise 1.1: Understand the DocuMind Pipeline from the PRD

**Task:** Review the DocuMind PRD to understand the multi-agent architecture you'll generate.

**Context:** The PRD specifies a document processing pipeline. Your goal is to understand the requirements before using Claude Flow to generate the implementation.

**Instructions:**

**Step 1: Review the PRD Pipeline Specification (5 mins)**

Open your DocuMind PRD and find the "Multi-Agent Document Processing" section. The PRD specifies these requirements:
- Process documents in parallel for speed
- Handle multiple file formats (PDF, DOCX, XLSX, TXT, MD)
- Extract text, chunk, generate embeddings, and store
- Handle errors gracefully (one document failure doesn't stop others)

**Step 2: Identify the 5 Pipeline Components (5 mins)**

Based on the PRD, identify the distinct agents/scripts you'll generate:

1. **Extractor Agent** - Reads files and extracts raw text
   - Expertise: File format handling (PDF, DOCX, etc.)
   - Input: File path
   - Output: Raw text + metadata

2. **Chunker Agent** - Splits text into semantic chunks
   - Expertise: Text segmentation, semantic boundaries
   - Input: Raw text
   - Output: Array of text chunks (~500 words each)

3. **Embedder Agent** - Generates vector embeddings
   - Expertise: OpenAI API, embedding models
   - Input: Text chunks
   - Output: Embedding vectors (1536 dimensions)

4. **Writer Agent** - Stores chunks and embeddings in database
   - Expertise: Supabase operations, database transactions
   - Input: Chunks + embeddings
   - Output: Database records

5. **Coordinator Agent** - Orchestrates the entire pipeline
   - Expertise: Workflow management, error handling
   - Input: Document upload request
   - Output: Processing status report

**Step 2: Choose Communication Topology (3 mins)**

For document processing, we'll use **Hierarchical (Pipeline)** topology:
```
User Upload
     ↓
Coordinator (receives file)
     ↓
Extractor (parallel: multiple files at once)
     ↓
Chunker (parallel: multiple documents)
     ↓
Embedder (parallel: batch embeddings)
     ↓
Writer (parallel: database writes)
     ↓
Coordinator (reports completion)
```

**Why Hierarchical?**
- Clear data flow from one stage to the next
- Easy to add validation between stages
- Parallel processing at each stage
- Coordinator manages error recovery

**Step 3: Define Agent Interfaces (7 mins)**

Create `docs/spec/documind/agent-interfaces.md`:

```markdown
# DocuMind Agent Interfaces

## Coordinator Agent
**Role**: Orchestrates document processing pipeline

### Methods
- `process_document(file_path: str) -> dict`
  - Coordinates full pipeline
  - Returns: Processing report with status, chunks created, errors

- `process_batch(file_paths: list[str]) -> dict`
  - Processes multiple documents in parallel
  - Returns: Batch processing report

### Communication
- Delegates to: Extractor
- Receives from: Extractor, Chunker, Embedder, Writer
- Shared Memory Keys:
  - `pipeline/status/{document_id}` - Current stage
  - `pipeline/errors/{document_id}` - Error log

---

## Extractor Agent
**Role**: Extract text from various file formats

### Methods
- `extract_text(file_path: str) -> dict`
  - Returns: `{text: str, metadata: dict, format: str}`

- `supported_formats() -> list[str]`
  - Returns: [".pdf", ".docx", ".xlsx", ".txt", ".md"]

### Communication
- Receives from: Coordinator
- Delegates to: Chunker
- Shared Memory Keys:
  - `extraction/raw_text/{document_id}` - Extracted text
  - `extraction/metadata/{document_id}` - File metadata

---

## Chunker Agent
**Role**: Split text into semantic chunks

### Methods
- `chunk_text(text: str, strategy: str = "semantic") -> list[dict]`
  - Returns: Array of chunks with metadata
  - Strategies: "semantic", "fixed-size", "sentence"

- `optimize_chunk_size(text: str) -> int`
  - Returns: Recommended chunk size for this document

### Communication
- Receives from: Extractor
- Delegates to: Embedder
- Shared Memory Keys:
  - `chunking/chunks/{document_id}` - Generated chunks
  - `chunking/strategy/{document_id}` - Strategy used

---

## Embedder Agent
**Role**: Generate vector embeddings

### Methods
- `generate_embeddings(chunks: list[str]) -> list[list[float]]`
  - Batch generates embeddings
  - Returns: Array of 1536-dimensional vectors

- `batch_optimize(chunks: list[str]) -> int`
  - Returns: Optimal batch size for API efficiency

### Communication
- Receives from: Chunker
- Delegates to: Writer
- Shared Memory Keys:
  - `embeddings/vectors/{document_id}` - Generated embeddings
  - `embeddings/model/{document_id}` - Model used

---

## Writer Agent
**Role**: Store chunks and embeddings in Supabase

### Methods
- `write_chunks(document_id: str, chunks: list[dict]) -> bool`
  - Writes to document_chunks table
  - Returns: Success status

- `transaction_safe_write(data: dict) -> bool`
  - Writes with transaction rollback on error

### Communication
- Receives from: Embedder
- Reports to: Coordinator
- Shared Memory Keys:
  - `database/write_status/{document_id}` - Write confirmation
  - `database/record_ids/{document_id}` - Created record IDs
```

---

### Quiz 1:

**Question 1:** When should you use multiple agents instead of a single agent?\
   a) When the task requires specialized expertise, parallel processing, or distinct workflow stages\
   b) Only when you have more than 10,000 documents to process\
   c) Whenever you want to spend less money on API calls\
   d) Never - single agents are always better

**Question 2:** What is the primary advantage of hierarchical (pipeline) topology for document processing?\
   a) It uses less memory than other topologies\
   b) Agents can talk to each other randomly which is faster\
   c) Clear data flow, easy validation between stages, and parallel processing at each stage\
   d) It requires fewer agents

**Question 3:** In the DocuMind pipeline, why do we need separate Extractor and Chunker agents?\
   a) To make the system more complicated\
   b) They require different expertise (file parsing vs text segmentation) and can be optimized independently\
   c) Because Claude Code requires at least 5 agents\
   d) So we can charge more for the system

**Answers:**
1. **a)** Use multiple agents for specialized expertise, parallel processing, or distinct workflow stages
2. **c)** Hierarchical topology provides clear data flow, easy validation, and stage-level parallelism
3. **b)** Separation enables specialization and independent optimization of each concern

---

## Module 2: ClaudeFlow Setup and Agent Spawning (15 minutes)

### Concept Review

**What is ClaudeFlow?**

ClaudeFlow is a multi-agent orchestration framework that manages swarms of AI agents. It provides:
- **Swarm initialization**: Set up agent topologies (mesh, hierarchical, star, ring)
- **Agent spawning**: Create specialized agents with defined capabilities
- **Task orchestration**: Distribute work across agents
- **Memory coordination**: Shared state management
- **Performance monitoring**: Track agent activity and bottlenecks

**ClaudeFlow Commands:**
```bash
# Initialize swarm
npx claude-flow swarm init --topology <type> --max-agents <n>

# Spawn agent
npx claude-flow agent spawn --type <role> --name <identifier>

# List agents
npx claude-flow agent list

# Orchestrate task
npx claude-flow task orchestrate --task "<description>" --priority high

# Check status
npx claude-flow swarm status
```

---

### Exercise 2.1: Initialize ClaudeFlow Swarm

**Task:** Set up a ClaudeFlow swarm for DocuMind's document processing pipeline.

**Instructions:**

**Step 1: Install ClaudeFlow (2 mins)**

```bash
# Install and initialize ClaudeFlow as MCP server
npx claude-flow@alpha init --force

# Verify installation:
claude mcp list
```

# Should show: 
ruv-swarm: npx ruv-swarm mcp start - ✓ Connected
supabase: npx @supabase/mcp-server-supabase - ✓ Connected
documind: python3 src/documind-mcp/server.py - ✓ Connected
claude-flow: npx claude-flow@alpha mcp start - ✓ Connected
flow-nexus: npx flow-nexus@latest mcp start - ✓ Connected

NOTE: If any MCPs are not connected, use Claude Code to fix.  Then restart session.

# Reference: https://github.com/ruvnet/claude-flow


**Step 2: Initialize Swarm with Hierarchical Topology (3 mins)**

```bash
# Create hierarchical swarm for pipeline processing
npx claude-flow swarm init \
  --topology hierarchical \
  --max-agents 6 \
  --strategy adaptive

# Expected output:
# ✓ Swarm initialized successfully
# ✓ Topology: hierarchical
# ✓ Max agents: 6
# ✓ Strategy: adaptive
# ✓ Swarm ID: swarm-abc123
```

**What This Does:**
- Creates a swarm with hierarchical coordination
- Allows up to 6 agents (1 coordinator + 5 workers)
- Uses adaptive strategy (automatically adjusts based on load)
- Generates a unique swarm ID for reference

**Step 3: Verify Swarm Status (2 mins)**

```bash
npx claude-flow swarm status

# Expected output:
# Swarm Status:
# ============
# ID: swarm-abc123
# Topology: hierarchical
# Active Agents: 0 / 6
# Status: Ready
# Created: 2025-11-24 10:30:00
```

**Step 4: Configure Memory Coordination (3 mins)**

Create `.claude-flow/memory-config.json`:

```json
{
  "memory": {
    "type": "shared",
    "backend": "filesystem",
    "path": "./memory/swarm",
    "ttl": 3600,
    "namespaces": [
      "pipeline/status",
      "pipeline/errors",
      "extraction/raw_text",
      "extraction/metadata",
      "chunking/chunks",
      "chunking/strategy",
      "embeddings/vectors",
      "embeddings/model",
      "database/write_status",
      "database/record_ids"
    ]
  },
  "coordination": {
    "locking": true,
    "retryAttempts": 3,
    "timeoutMs": 30000
  },
  "monitoring": {
    "enabled": true,
    "metricsInterval": 10000,
    "logLevel": "info"
  }
}
```

**Step 5: Test Memory Access (5 mins)**

```bash
# Store test data in memory
npx claude-flow memory store \
  --key "pipeline/status/test-doc-1" \
  --value '{"stage": "extraction", "progress": 0.5}' \
  --namespace "pipeline/status"

# Retrieve test data
npx claude-flow memory retrieve \
  --key "pipeline/status/test-doc-1" \
  --namespace "pipeline/status"

# Expected output:
# {"stage": "extraction", "progress": 0.5}

# List all keys in namespace
npx claude-flow memory list --namespace "pipeline/status"

# Delete test data
npx claude-flow memory delete \
  --key "pipeline/status/test-doc-1" \
  --namespace "pipeline/status"
```

---

### Exercise 2.2: Prepare Your DocuMind Project for Code Generation

> **Important:** The DocuMind pipeline scripts you'll generate are **pure Python** - no Claude Flow dependency at runtime. Claude Flow helps you BUILD the scripts; they run independently in production.

**Task:** Set up your DocuMind project structure and prepare for swarm-driven code generation.

**Instructions:**

**Step 1: Ensure Your DocuMind Project Has the PRD (2 mins)**

In your **DocuMind Codespace** (not this course repo), verify the PRD exists:

```bash
# Check for PRD
ls docs/spec/documind-prd.md

# If missing, copy from course repo or create the docs directory
```

The PRD is your specification - Claude Flow will use it to understand what to generate.

**Step 2: Create the Pipeline Directory (2 mins)**

```bash
# Create the directory where generated scripts will live
mkdir -p src/agents/pipeline

# Create __init__.py for Python package
touch src/agents/pipeline/__init__.py

# Verify structure
ls -la src/agents/pipeline/
```

**Step 3: Create Demo Documents for Testing (3 mins)**

Create sample documents to test the generated pipeline:

```bash
# Create demo-docs directory
mkdir -p demo-docs

# Create a sample markdown document
cat > demo-docs/remote-work-policy.md << 'EOF'
# Remote Work Policy

## Overview
Employees may work remotely up to 3 days per week with manager approval.

## Requirements
- Stable internet connection (minimum 10 Mbps)
- Dedicated workspace
- Availability during core hours (10 AM - 3 PM local time)

## Equipment
Company provides laptop and $500 annual stipend for home office setup.

## Communication
- Daily standup via Slack or Teams
- Camera on for all team meetings
- Respond to messages within 2 hours during core hours
EOF

# Create another sample document
cat > demo-docs/expense-policy.md << 'EOF'
# Expense Reimbursement Policy

## Eligible Expenses
- Travel: flights, hotels, ground transportation
- Meals: up to $75/day for business travel
- Software: pre-approved tools only
- Training: requires manager approval

## Submission Process
1. Submit receipts within 30 days
2. Use the expense portal at expenses.company.com
3. Manager approval required for expenses over $500
EOF
```

**Step 4: Verify Environment Variables (3 mins)**

The generated scripts will use real APIs. Verify your keys from S4 are configured:

```bash
# Check for required environment variables
echo "OpenAI API Key: ${OPENAI_API_KEY:0:10}..."
echo "Supabase URL: ${SUPABASE_URL}"
echo "Supabase Key: ${SUPABASE_ANON_KEY:0:10}..."

# If any are missing, add them to your .env file
# These should be configured from Session 4
```

**What You'll Generate in Module 3:**

| Script | Purpose | Key Function |
|--------|---------|--------------|
| **extractor.py** | Reads files, extracts text | `extract_document(path) → dict` |
| **chunker.py** | Splits into ~500-word chunks | `chunk_content(text, size=500) → list` |
| **embedder.py** | Generates OpenAI embeddings | `generate_embeddings(chunks) → list` |
| **writer.py** | Stores in Supabase | `store_document(data) → bool` |
| **orchestrate.py** | Coordinates pipeline | `process_batch(files) → report` |

**Key Teaching Point:**
> "Claude Flow is about to WRITE these scripts for you using natural language prompts. You describe what you need, the swarm generates the implementation. This is agentic development in action."
```

---

### ⚠️ Verification Checkpoint: Agent Spawning

**Before continuing, verify your agents are properly configured:**

```bash
# 1. Check agents exist
npx claude-flow agent list | grep -c "Status: Active"
# Should output: 5

# 2. Check swarm topology
npx claude-flow swarm status | grep "Topology"
# Should show: hierarchical

# 3. Check memory is accessible
npx claude-flow memory get "swarm/status"
# Should return status data
```

**If any check fails:**
- Re-run the failed step with more verbose output (`--verbose` flag)
- Check error logs: `cat .claude-flow/logs/latest.log`
- Restart the swarm: `npx claude-flow swarm destroy && npx claude-flow swarm init ...`

**Important:** Agents report "spawned" immediately, but may not be fully ready. Wait 2-3 seconds after spawning before sending tasks.

---

### Quiz 2:

**Question 1:** What is the purpose of ClaudeFlow's swarm initialization?\
   a) To set up the topology, coordination strategy, and shared resources for multiple agents\
   b) To install Python packages\
   c) To create a new GitHub repository\
   d) To delete old files

**Question 2:** Why do we spawn 5 agents instead of just using 1 agent for everything?\
   a) Specialization enables expertise, parallel processing, and better error isolation\
   b) Because it looks cool to have many agents\
   c) To waste resources intentionally\
   d) Claude Code requires exactly 5 agents

**Question 3:** What is the role of the shared memory system in ClaudeFlow?\
   a) Enables agents to coordinate by storing and retrieving shared state across the pipeline\
   b) To save RAM on your computer\
   c) To slow down the agents intentionally\
   d) It has no real purpose

**Answers:**
1. **a)** Swarm initialization sets up topology, strategy, and shared resources for agent coordination
2. **a)** Multiple specialized agents enable expertise, parallelism, and error isolation
3. **a)** Shared memory enables coordination through shared state management

---

## Module 3: Swarm-Driven Code Generation (15 minutes)

### Concept Review

**Natural Language to Code:**

In this module, you'll use Claude Flow to generate the pipeline scripts using natural language. This is the core of agentic development:

1. **You describe** what each component should do
2. **The swarm generates** production-ready Python code
3. **You verify** the generated code meets the PRD spec

**The Power of Natural Language Prompts:**

A good prompt includes:
- **Purpose**: What the script should accomplish
- **Inputs/Outputs**: Expected data formats
- **Dependencies**: APIs, libraries to use
- **Constraints**: Error handling, performance requirements

**Example Prompt Structure:**
```
Generate a Python script called {name}.py that:
- Purpose: {what it does}
- Input: {expected input}
- Output: {expected output}
- Uses: {libraries/APIs}
- Must: {constraints/requirements}
```

---

### Exercise 3.1: Generate the Extractor Script

**Task:** Use natural language to have Claude generate `extractor.py`.

**Instructions:**

**Step 1: Craft Your Natural Language Prompt (3 mins)**

In Claude Code, type the following prompt (adapt based on your PRD):

```
Generate a Python script called src/agents/pipeline/extractor.py that implements the Document Extractor from the DocuMind PRD.

**Purpose:** Extract text content and metadata from documents.

**Requirements from PRD:**
- Support file formats: .md, .txt, .pdf, .docx
- Extract raw text content
- Capture metadata: title (from first heading or filename), file type, file size
- Return a dictionary with: success, file_path, title, content, file_type, size, error (if any)

**Implementation details:**
- Use pathlib for file handling
- For PDF: use PyPDF2 or pdfplumber (install if needed)
- For DOCX: use python-docx (install if needed)
- For MD/TXT: read directly with standard library
- Handle errors gracefully - return success=False with error message
- Make it executable standalone: python extractor.py <file_path>
- Output JSON to stdout

**Include:**
- Type hints
- Docstrings
- A main() function for CLI usage
- Proper shebang (#!/usr/bin/env python3)
```

**Step 2: Review the Generated Code (2 mins)**

After Claude generates the script, verify:
- [ ] File exists at `src/agents/pipeline/extractor.py`
- [ ] Has proper imports and error handling
- [ ] Supports the file formats specified
- [ ] Returns JSON output

**Step 3: Test the Generated Extractor (2 mins)**

```bash
# Make it executable
chmod +x src/agents/pipeline/extractor.py

# Test with a markdown file
python src/agents/pipeline/extractor.py demo-docs/remote-work-policy.md

# Expected output (JSON):
# {
#   "success": true,
#   "file_path": "demo-docs/remote-work-policy.md",
#   "title": "Remote Work Policy",
#   "content": "...",
#   "file_type": ".md",
#   "size": 523
# }
```

---

### Exercise 3.2: Generate the Remaining Pipeline Components

**Task:** Use natural language to generate chunker.py, embedder.py, and writer.py.

**Instructions:**

**Step 1: Generate chunker.py (3 mins)**

```
Generate src/agents/pipeline/chunker.py that implements the Text Chunker from the DocuMind PRD.

**Purpose:** Split text into semantic chunks for embedding.

**Requirements:**
- Default chunk size: 500 words with 50-word overlap
- Accept JSON input (from extractor) via stdin or file argument
- Output JSON array of chunks with metadata

**Each chunk should include:**
- chunk_id: sequential identifier
- content: the text content
- word_count: number of words
- start_position: character offset in original
- document_id: from input

**Implementation:**
- Split on sentence boundaries when possible
- Ensure overlap between consecutive chunks
- Handle edge cases (very short documents, single sentences)
- Pure Python - no external dependencies for chunking
```

**Step 2: Generate embedder.py (3 mins)**

```
Generate src/agents/pipeline/embedder.py that implements the Embedding Generator from the DocuMind PRD.

**Purpose:** Generate vector embeddings using OpenAI API.

**Requirements:**
- Use OpenAI text-embedding-3-small model
- Accept JSON input (array of chunks) via stdin or file argument
- Output JSON with embeddings array
- Handle API rate limits with retry logic

**Environment:**
- Read OPENAI_API_KEY from environment variable
- Batch chunks for efficiency (max 2000 tokens per batch)

**Output format:**
{
  "success": true,
  "embeddings": [
    {"chunk_id": "...", "vector": [0.1, 0.2, ...], "model": "text-embedding-3-small"}
  ]
}
```

**Step 3: Generate writer.py (3 mins)**

```
Generate src/agents/pipeline/writer.py that implements the Database Writer from the DocuMind PRD.

**Purpose:** Store documents, chunks, and embeddings in Supabase.

**Requirements:**
- Use Supabase Python client
- Accept JSON input with document data, chunks, and embeddings
- Write to tables: documents, document_chunks (with vectors)
- Use transactions for data integrity

**Environment:**
- Read SUPABASE_URL and SUPABASE_ANON_KEY from environment
- Handle connection errors gracefully

**Tables (from S4):**
- documents: id, title, content, file_type, created_at
- document_chunks: id, document_id, chunk_index, content, embedding (vector)
```

---

### Exercise 3.3: Generate the Pipeline Orchestrator

**Task:** Generate orchestrate.py to coordinate all pipeline stages.

**Instructions:**

**Prompt for orchestrate.py:**

```
Generate src/agents/pipeline/orchestrate.py that coordinates the DocuMind document processing pipeline.

**Purpose:** Orchestrate the 4-stage pipeline with parallel processing.

**Pipeline stages:**
1. extractor.py - Extract text from documents
2. chunker.py - Split into chunks
3. embedder.py - Generate embeddings
4. writer.py - Store in database

**Requirements:**
- Accept a directory path or list of file paths as input
- Process multiple documents in parallel using asyncio
- Chain stages: output of each stage feeds into the next
- Collect metrics: time per stage, success/failure counts
- Print a summary report at the end

**Features:**
- Continue on error: if one document fails, keep processing others
- Progress indicator: show which document is being processed
- Final report: total docs, successful, failed, time taken

**CLI usage:**
python orchestrate.py demo-docs/
python orchestrate.py file1.md file2.pdf file3.docx
```

**Verify the Complete Pipeline:**

```bash
# Run the full pipeline on demo documents
python src/agents/pipeline/orchestrate.py demo-docs/

# Expected output:
# Processing demo-docs/remote-work-policy.md...
#   ✓ Extracted (0.1s)
#   ✓ Chunked: 3 chunks (0.05s)
#   ✓ Embedded (1.2s)
#   ✓ Stored (0.3s)
# Processing demo-docs/expense-policy.md...
#   ✓ Extracted (0.1s)
#   ✓ Chunked: 2 chunks (0.04s)
#   ✓ Embedded (0.9s)
#   ✓ Stored (0.2s)
#
# ═══════════════════════════════════════
# Pipeline Complete
# Total: 2 documents
# Successful: 2
# Failed: 0
# Total time: 2.89s
# ═══════════════════════════════════════
```

---

### ⚠️ Verification Checkpoint: Generated Scripts

**After generating all scripts, verify they exist:**

```bash
# Check all pipeline scripts were created
ls -la src/agents/pipeline/

# Expected:
# extractor.py
# chunker.py
# embedder.py
# writer.py
# orchestrate.py
# __init__.py

# Verify each script is syntactically correct
python -m py_compile src/agents/pipeline/extractor.py
python -m py_compile src/agents/pipeline/chunker.py
python -m py_compile src/agents/pipeline/embedder.py
python -m py_compile src/agents/pipeline/writer.py
python -m py_compile src/agents/pipeline/orchestrate.py
```

**If any script is missing or has errors:**
- Re-run the natural language prompt with more specific instructions
- Ask Claude to fix the specific error
- Verify your environment variables are set correctly

---

### Quiz 3:

**Question 1:** What makes a good natural language prompt for code generation?\
   a) Including purpose, inputs/outputs, dependencies, and constraints\
   b) Using as few words as possible\
   c) Asking the AI to "just figure it out"\
   d) Only specifying the file name

**Question 2:** Why do we generate separate scripts for each pipeline stage instead of one monolithic script?\
   a) Separation enables independent testing, easier debugging, and parallel development\
   b) To use more disk space\
   c) Claude can only generate small files\
   d) It's required by Python

**Question 3:** What is the key benefit of using the PRD as input for code generation?\
   a) The PRD provides a clear specification so generated code matches requirements\
   b) The PRD makes files smaller\
   c) It's required by Claude Flow\
   d) PRDs are faster to read than code

**Answers:**
1. **a)** Good prompts include purpose, inputs/outputs, dependencies, and constraints for accurate code generation
2. **a)** Separation enables independent testing, easier debugging, and allows parallel development by multiple agents
3. **a)** The PRD provides a clear specification ensuring generated code matches intended requirements

---

## Module 4: Test and Enhance the Generated Pipeline (15 minutes)

### Overview

Now that you've generated all 5 pipeline scripts using natural language, it's time to:
1. **Test** the complete pipeline end-to-end
2. **Verify** data is stored correctly in Supabase
3. **Enhance** the pipeline using natural language prompts

---

### Exercise 4.1: Run the Complete Pipeline

**Task:** Test your generated pipeline with the demo documents.

**Instructions:**

**Step 1: Run the Full Pipeline (5 mins)**

```bash
# Navigate to your DocuMind project
cd ~/documind  # or wherever your project is

# Run the orchestrator on demo documents
python src/agents/pipeline/orchestrate.py demo-docs/

# Expected output:
# Processing demo-docs/remote-work-policy.md...
#   ✓ Extracted (0.12s)
#   ✓ Chunked: 3 chunks (0.05s)
#   ✓ Embedded (1.45s)
#   ✓ Stored (0.28s)
# Processing demo-docs/expense-policy.md...
#   ✓ Extracted (0.11s)
#   ✓ Chunked: 2 chunks (0.04s)
#   ✓ Embedded (0.92s)
#   ✓ Stored (0.21s)
#
# Pipeline Complete: 2 documents processed
```

**Step 2: Verify Data in Supabase (3 mins)**

Use Claude Code to verify the data was stored:

```
Use the Supabase MCP to:
1. Count documents in the 'documents' table
2. List the first 5 chunks from 'document_chunks' table
3. Verify embeddings exist (check if embedding column is not null)
```

**Expected verification:**
- Documents table should have 2 new records
- Chunks table should have 5 chunks (3 + 2)
- Each chunk should have a 1536-dimension embedding vector

---

### Exercise 4.2: Test Error Handling

**Task:** Verify the pipeline handles errors gracefully.

**Instructions:**

**Step 1: Test with Invalid File (2 mins)**

```bash
# Create a file that will cause an error
echo "not valid json" > demo-docs/invalid.xyz

# Run pipeline - should handle error gracefully
python src/agents/pipeline/orchestrate.py demo-docs/invalid.xyz

# Expected: Error message but no crash
# ✗ Failed to process invalid.xyz: Unsupported file format
```

**Step 2: Test with Missing API Key (2 mins)**

```bash
# Temporarily unset OpenAI key
OPENAI_API_KEY_BACKUP=$OPENAI_API_KEY
unset OPENAI_API_KEY

# Try to run embedder - should fail gracefully
echo '{"chunks": [{"content": "test"}]}' | python src/agents/pipeline/embedder.py

# Expected: Error about missing API key
# {"success": false, "error": "OPENAI_API_KEY not set"}

# Restore key
export OPENAI_API_KEY=$OPENAI_API_KEY_BACKUP
```

---

### Exercise 4.3: Enhance with Natural Language

**Challenge:** Use natural language to add features to your generated pipeline.

**Instructions:**

Choose ONE of the following enhancements and use natural language to implement it:

**Option A: Add Retry Logic to Embedder**

```
Enhance src/agents/pipeline/embedder.py to add retry logic:

- If the OpenAI API call fails, retry up to 3 times
- Use exponential backoff: wait 1s, then 2s, then 4s between retries
- Log each retry attempt
- If all retries fail, return success=False with error details
```

**Option B: Add Progress Reporting to Orchestrator**

```
Enhance src/agents/pipeline/orchestrate.py to show progress:

- Print a progress bar or percentage for each document
- Show estimated time remaining based on average processing time
- At the end, show a summary table with:
  - Total documents processed
  - Success/failure count
  - Average time per stage
  - Slowest stage (bottleneck)
```

**Option C: Add Metadata Extraction to Extractor**

```
Enhance src/agents/pipeline/extractor.py to extract more metadata:

- For markdown: extract all headings as a list
- For all files: count words, sentences, paragraphs
- Add a "summary" field with the first 200 characters
- Add "keywords" field with the 5 most common non-stopwords
```

**Verify Your Enhancement:**

```bash
# Test that the enhanced script still works
python src/agents/pipeline/orchestrate.py demo-docs/

# Check git diff to see what changed
git diff src/agents/pipeline/
```

---

### Challenge: Production-Ready Enhancements (Optional)

**For Advanced Students:**

Use this natural language prompt to have Claude generate an enhanced pipeline:

```
Generate src/agents/pipeline/enhanced_orchestrator.py with production-ready features:

**Purpose:** Enhanced pipeline coordinator with error recovery, monitoring, and dashboards.

**Features to implement:**

1. **Format Detection** - Detect document format from file extension (PDF, DOCX, TXT, MD)
2. **Retry with Exponential Backoff** - Retry failed operations up to 3 times (1s, 2s, 4s delays)
3. **Circuit Breaker Pattern** - Open circuit after 5 consecutive failures
4. **Processing Metrics** - Track time per stage, count retries and failures
5. **Status Dashboard** - Show success/failure rate, average time per stage, bottleneck analysis

**Include:** ProcessingMetrics dataclass, EnhancedPipelineCoordinator class, CLI interface
```

**Test Your Enhanced Pipeline:**

```bash
python src/agents/pipeline/enhanced_orchestrator.py demo-docs/
```

---

### Success Criteria

Your generated pipeline is complete when:

- [ ] All 5 base pipeline scripts exist and run without errors
- [ ] Pipeline successfully processes demo documents end-to-end
- [ ] Data is stored correctly in Supabase (verify with MCP)
- [ ] Error handling works (invalid files don't crash the pipeline)
- [ ] At least one enhancement was added using natural language

---

### Quiz 4:

**Question 1:** What's the main benefit of using natural language to enhance code vs. manual editing?\
   a) You describe WHAT you want, and the AI figures out HOW to implement it\
   b) It's faster to type natural language than code\
   c) Natural language is more precise than code\
   d) Manual editing doesn't work

**Question 2:** When testing generated code, what should you verify first?\
   a) The script runs without syntax errors and handles edge cases\
   b) The code is as short as possible\
   c) The variable names are creative\
   d) The comments are detailed

**Question 3:** What demonstrates the "Two Layers" concept in this session?\
   a) Claude Flow helped generate the pipeline scripts, which then run independently at runtime\
   b) We used two programming languages\
   c) We created two copies of each file\
   d) We ran the pipeline twice

**Answers:**
1. **a)** Natural language lets you focus on requirements; the AI handles implementation details
2. **a)** First verify the script runs and handles edge cases before checking other aspects
3. **a)** Claude Flow (development-time) generated scripts that run independently (runtime)

---

## Answer Key

### Exercise 1.1 Solution

The agent interfaces from the PRD specify 5 pipeline components:
1. **Extractor** - Reads files, extracts text and metadata
2. **Chunker** - Splits text into ~500-word semantic chunks
3. **Embedder** - Generates vector embeddings via OpenAI API
4. **Writer** - Stores in Supabase with pgvector
5. **Orchestrator** - Coordinates the pipeline with asyncio

Each component should be a standalone Python script that:
- Accepts JSON input
- Outputs JSON
- Handles errors gracefully
- Has no Claude Flow dependency at runtime

---

### Exercise 2.1 & 2.2 Solution

**Exercise 2.1:** ClaudeFlow swarm initialization:
```bash
npx claude-flow@alpha init --force
npx claude-flow swarm init --topology hierarchical --max-agents 6 --strategy adaptive
```

**Exercise 2.2:** Project setup verification:
```bash
# Directory structure should exist
ls src/agents/pipeline/
# __init__.py

# Demo docs should exist
ls demo-docs/
# remote-work-policy.md  expense-policy.md

# Environment variables should be set
echo $OPENAI_API_KEY
echo $SUPABASE_URL
```

**Key insight:** Claude Flow helps you DEVELOP faster (Layer 1). The Python scripts ARE the runtime pipeline (Layer 2). No Claude Flow dependency at runtime.

---

### Module 4 Challenge Solution

**Approach:** Use natural language to enhance your generated scripts with production features.

**Sample Enhancement Prompt:**
```
Enhance src/agents/pipeline/orchestrate.py with production-ready features:

1. Add exponential backoff retry logic (max 3 retries)
2. Add a circuit breaker that opens after 5 consecutive failures
3. Add timing metrics for each pipeline stage
4. Generate a status dashboard showing:
   - Documents processed (success/fail counts)
   - Average time per stage
   - Current bottleneck identification
   - Circuit breaker status
```

**Expected Result:** Claude Flow generates an enhanced orchestrate.py with:
- `process_with_retry()` function for fault tolerance
- `CircuitBreaker` class or state variables
- `ProcessingMetrics` dataclass for timing
- `generate_dashboard()` function for status reporting

**Example Enhanced Output:**
```
======================================================================
                    DocuMind Pipeline Status Dashboard
======================================================================

Overall Statistics:
   Total Documents:    5
   Successful:         5 (100.0%)
   Failed:             0
   Total Time:         8.45s
   Avg Time/Doc:       1.69s

Average Stage Times:
   Extraction:         0.12s
   Chunking:           0.08s
   Embedding:          1.25s (OpenAI API)
   Database:           0.24s (Supabase)

Performance Analysis:
   Bottleneck Stage:   embedding
   Circuit Breaker:    CLOSED
   Recommendation:     Use batch embedding API calls

======================================================================
```

**Key insight:** The swarm GENERATES the enhancement code based on your requirements. You describe what you want; Claude Flow implements it.

---

## Additional Challenges (Optional)

For students who finish early, use natural language prompts to extend your pipeline:

### Challenge 1: Batch Processing
```
Add a batch processing mode to orchestrate.py that:
- Accepts a directory path as input
- Processes all supported files in parallel (max 5 concurrent)
- Generates a summary report with success/failure counts
- Saves failed files to a retry queue
```

### Challenge 2: Document Deduplication
```
Create src/agents/pipeline/deduplicator.py that:
- Compares new document embeddings against existing Supabase vectors
- Uses cosine similarity with threshold 0.95 for duplicates
- Skips processing if duplicate found
- Returns: {is_duplicate, similar_doc_id, similarity_score}
```

### Challenge 3: Search Interface
```
Create src/agents/pipeline/search.py that:
- Accepts a natural language query
- Generates embedding using OpenAI API
- Queries Supabase for top 5 similar chunks
- Returns formatted results with source documents
```

### Challenge 4: Processing Analytics
```
Add analytics tracking to orchestrate.py:
- Log processing times to a CSV file
- Calculate rolling averages per stage
- Identify slow documents (>2x average time)
- Generate daily summary statistics
```

---

## Troubleshooting

### Common Issue 1: Agent Not Spawning
**Problem:** `npx claude-flow agent spawn` returns "Failed to spawn agent"

**Solution:**
1. Check swarm is initialized: `npx claude-flow swarm status`
2. Verify max agents not reached: `npx claude-flow agent list`
3. Check agent type is valid: coordinator, researcher, analyst, coder, tester
4. Restart swarm if needed: `npx claude-flow swarm init --topology hierarchical`

---

### Common Issue 2: Memory Coordination Failures
**Problem:** Agents can't access shared memory

**Solution:**
```bash
# Verify memory system is working
npx claude-flow memory store --key "test/key" --value "test-value"
npx claude-flow memory retrieve --key "test/key"

# Check memory configuration
cat .claude-flow/memory-config.json

# Reset memory if corrupted
rm -rf memory/swarm/*
npx claude-flow memory reset
```

---

### Common Issue 3: Parallel Processing Not Working
**Problem:** Documents process sequentially instead of in parallel

**Solution:**
```python
# Ensure you're using asyncio.gather for parallelism
tasks = [process_document(fp) for fp in file_paths]
results = await asyncio.gather(*tasks)  # ✓ Correct

# NOT this (sequential):
results = []
for fp in file_paths:
    result = await process_document(fp)  # ✗ Wrong - waits for each
    results.append(result)
```

---

### Common Issue 4: Circuit Breaker Won't Reset
**Problem:** Circuit breaker stays open even after fixing issues

**Solution:**
```python
# Manually reset circuit breaker in code
coordinator.circuit_breaker_failures = 0
coordinator.circuit_open = False

# Or restart the coordinator
coordinator = EnhancedPipelineCoordinator()
```

---

## Key Takeaways

By completing this workshop, you've learned:

1. **PRD-Driven Development**: Using a specification document to guide AI code generation
2. **Natural Language Prompting**: Describing requirements clearly for Claude Flow to implement
3. **Two Layers Architecture**: Development-time AI (Claude Flow) vs Runtime code (Python scripts)
4. **Swarm Orchestration**: Coordinating multiple specialized agents for code generation
5. **Iterative Enhancement**: Refining generated code through follow-up prompts

**The Two Layers Pattern:**
```
Layer 1 (Development): Claude Flow swarms generate and refine code
Layer 2 (Runtime):     Python scripts run independently, no AI needed
```

**Key Skill:** Effective natural language prompting produces better generated code.

---

## Next Session Preview

In **Session 6: RAG with Retrieval-Augmented Generation**, we'll:
- Understand the RAG pipeline (retrieve → augment → generate)
- Implement semantic search over document chunks
- Build a Q&A system with citation tracking
- Compare RAG vs Context-Augmented Generation (CAG)
- Use Claude Flow to generate DocuMind's chatbot interface

**Preparation:**
1. Ensure your generated pipeline scripts work end-to-end
2. Have documents ingested and stored in Supabase (from Module 4)
3. Keep your OpenAI API key configured for embeddings
4. Review vector similarity concepts (cosine, dot product)

See you in Session 6!

---

**Workshop Complete!**

You've used Claude Flow swarms to BUILD a document processing pipeline from a PRD specification. The key insight: describe what you want in natural language, and AI agents generate the implementation. This is the Two Layers pattern in action.
