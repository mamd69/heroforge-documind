# HeroForge.AI Course: AI-Powered Software Development
## Lesson 5 Workshop: Multi-Agent Systems - Coordinating Specialized AI Teams

**Estimated Time:** 40-50 minutes\
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

**Key insight:** The agent interfaces doc is the contract that Claude Flow agents use to generate consistent, interoperable code. Without it, the generated scripts might not work together properly.

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

## Module 2: Goal-Oriented Planning with Claude Flow (10 minutes)

### Concept Review

**What is Goal-Oriented Action Planning (GOAP)?**

GOAP is an AI planning technique borrowed from game development that dynamically creates plans to achieve complex objectives. Instead of following rigid scripts, GOAP:

1. **Analyzes the goal state** - What does "success" look like?
2. **Identifies preconditions** - What must be true before each action?
3. **Creates action sequences** - What order achieves the goal most efficiently?
4. **Defines success criteria** - How do we verify completion?

**The Goal-Planner Agent:**

Claude Flow includes a `goal-planner` agent that applies GOAP to software development. When you invoke it, the agent:
- Reads your specifications (PRD, requirements docs)
- Breaks complex goals into concrete milestones
- Identifies dependencies between tasks
- Creates a structured implementation plan

**Why Plan Before Implementing?**

| Without Planning | With Goal-Planner |
|------------------|-------------------|
| Jump into code, discover issues later | Identify blockers upfront |
| Miss edge cases | Consider all components |
| Rework due to missed dependencies | Clear execution order |
| No validation criteria | Built-in success metrics |

---

### Exercise 2.1: Generate an Implementation Plan

**Task:** Use the goal-planner agent to create a comprehensive plan for the 5 pipeline components.

**Instructions:**

**Step 1: Invoke the Goal-Planner (3 mins)**

In Claude Code, enter this prompt:

```
use @goal-planner to review the S5-Workshop Module 1 content and the
DocuMind PRD (docs/spec/documind-prd.md) to create an implementation
plan for the 5 pipeline components. Save the plan to
docs/plans/pipeline-components-plan.md
```

**What Happens:**
- The goal-planner agent reads Module 1 (agent interfaces)
- It analyzes the PRD requirements
- It creates a structured plan with milestones
- The plan is saved to `docs/plans/pipeline-components-plan.md`

**Step 2: Review the Generated Plan (5 mins)**

Open the generated plan and verify it includes:

- [ ] **All 5 components identified:**
  - Extractor (read files, extract text)
  - Chunker (split into ~500-word chunks)
  - Embedder (generate OpenAI embeddings)
  - Writer (store in Supabase)
  - Orchestrator (coordinate the pipeline)

- [ ] **Clear milestones with success criteria:**
  - Each component has a "done" definition
  - Validation steps are specified
  - Dependencies are explicit

- [ ] **Technology choices aligned with PRD:**
  - Python 3.10+
  - OpenAI text-embedding-3-small
  - Supabase with pgvector

- [ ] **Implementation order:**
  - Which component to build first?
  - What can be parallelized?

**Step 3: Understand the Plan Structure (2 mins)**

A well-formed GOAP plan includes:

```markdown
## Goal: Working Document Processing Pipeline

### Milestone 1: Extractor Component
**Preconditions:** Project structure exists
**Actions:** Create extractor.py with file reading logic
**Success Criteria:** Can extract text from .md, .txt, .pdf, .docx

### Milestone 2: Chunker Component
**Preconditions:** Extractor works
**Actions:** Create chunker.py with text splitting
**Success Criteria:** Produces ~500-word chunks with overlap

[...continues for all 5 components...]
```

---

### ⚠️ Verification Checkpoint: Planning Complete

Before proceeding to Module 3, confirm:

```bash
# 1. Plan file exists
cat docs/plans/pipeline-components-plan.md | head -20

# 2. Environment variables configured (from Session 4)
echo "OpenAI: ${OPENAI_API_KEY:0:10}..."
echo "Supabase URL: ${SUPABASE_URL}"
echo "Supabase Key: ${SUPABASE_ANON_KEY:0:10}..."
```

**If the plan doesn't exist:**
- Re-run the goal-planner prompt
- Ensure Claude Code has access to the PRD file
- Check for any error messages

**If environment variables are missing:**
- Review Session 4 setup
- Add keys to your `.env` file

---

### Quiz 2:

**Question 1:** What is Goal-Oriented Action Planning (GOAP)?\
   a) An AI technique that dynamically creates plans by analyzing goals, preconditions, and success criteria\
   b) A way to write Python code faster\
   c) A database query optimization technique\
   d) A method for compressing files

**Question 2:** Why use a goal-planner agent before implementing code?\
   a) To make the code run faster\
   b) To identify dependencies, create milestones, and define success criteria upfront\
   c) Because Claude Code requires it\
   d) To reduce the number of files

**Question 3:** What should a good implementation plan include?\
   a) Just the programming language to use\
   b) Only file names\
   c) Milestones, preconditions, actions, success criteria, and dependencies\
   d) Marketing materials

**Answers:**
1. **a)** GOAP dynamically creates plans by analyzing goals, preconditions, and success criteria
2. **b)** Planning upfront identifies dependencies, creates milestones, and defines how to verify success
3. **c)** Good plans include milestones, preconditions, actions, success criteria, and dependencies

---

## Module 3: Swarm Implementation from Plan (10 minutes)

### Concept Review

**Issue-Driven Development with AI:**

In professional development, work is tracked through issues (GitHub Issues, Jira tickets, etc.). This pattern works beautifully with AI:

1. **Plan becomes Issue** - Your implementation plan converts to a trackable issue
2. **Issue becomes Work** - The AI swarm implements the issue
3. **Work becomes Verified** - You validate the implementation

**The Swarm-Advanced Skill:**

Claude Flow's `swarm-advanced` skill orchestrates multiple specialized agents to implement complex tasks:

| Agent Type | Role |
|------------|------|
| **Coder** | Writes implementation code |
| **Tester** | Creates tests and validation |
| **Reviewer** | Checks code quality |
| **Coordinator** | Manages workflow between agents |

**Why Swarm Implementation?**

- **Parallel Execution**: Multiple agents work simultaneously
- **Specialization**: Each agent focuses on their expertise
- **Quality Built-In**: Review and testing happen automatically
- **Faster Results**: 2-4x speed improvement over sequential work

---

### Exercise 3.1: Create Implementation Issue

**Task:** Convert your plan into a GitHub issue.

**Instructions:**

**Step 1: Create the Issue (2 mins)**

In Claude Code, enter:

```
create a GitHub issue for docs/plans/pipeline-components-plan.md
```

**What Happens:**
- Claude reads the plan file
- Creates a structured GitHub issue
- Includes tasks from the plan as checkboxes
- Adds appropriate labels (enhancement, pipeline, etc.)

**Step 2: Review the Created Issue (2 mins)**

After the issue is created:
- Note the issue number (e.g., `#42`)
- Review the task breakdown
- See how milestones became checkboxes
- Observe the labels and description

**Key Teaching Point:**
> "The plan you created in Module 2 is now a trackable work item. This is how professional development teams operate - plans become issues, issues become code."

---

### Exercise 3.2: Implement with Swarm

**Task:** Use the swarm-advanced skill to implement all 5 pipeline components.

**Instructions:**

**Step 1: Launch Swarm Implementation (3 mins)**

In Claude Code, enter:

```
implement the issue using the swarm-advanced skill
```

**What You'll See:**

The swarm skill will:
1. **Initialize** - Set up the swarm topology
2. **Spawn Agents** - Create coder, tester, reviewer agents
3. **Distribute Work** - Assign components to agents
4. **Execute in Parallel** - Multiple scripts generated simultaneously
5. **Coordinate** - Ensure consistency across components
6. **Report** - Show progress and completion status

**Step 2: Observe the Swarm (2 mins)**

Watch the output as agents work:

```
🚀 Swarm initialized: hierarchical topology
   ├── Spawning coder agent...
   ├── Spawning tester agent...
   └── Spawning reviewer agent...

📝 Implementing pipeline components:
   ├── extractor.py (coder) ████████░░ 80%
   ├── chunker.py (coder) ██████████ 100% ✓
   ├── embedder.py (coder) ███████░░░ 70%
   ├── writer.py (coder) ██████████ 100% ✓
   └── orchestrate.py (coder) ████░░░░░░ 40%

🔍 Review in progress...
✅ Implementation complete: 5/5 components
```

**Step 3: Verify Implementation (3 mins)**

After the swarm completes, verify all scripts were created:

```bash
# List pipeline scripts
ls -la src/agents/pipeline/

# Expected output:
# __init__.py
# extractor.py
# chunker.py
# embedder.py
# writer.py
# orchestrate.py

# Verify syntax is correct
python -m py_compile src/agents/pipeline/extractor.py
python -m py_compile src/agents/pipeline/chunker.py
python -m py_compile src/agents/pipeline/embedder.py
python -m py_compile src/agents/pipeline/writer.py
python -m py_compile src/agents/pipeline/orchestrate.py

echo "✅ All scripts pass syntax validation"
```

**Step 4: Review Generated Code (2 mins)**

Open one of the generated scripts and verify quality:

```bash
# View the extractor script
cat src/agents/pipeline/extractor.py | head -50
```

Check for:
- [ ] Proper imports and error handling
- [ ] Matches PRD specifications (file formats, output structure)
- [ ] Has type hints and docstrings
- [ ] Follows the agent interface from Module 1

---

### ⚠️ Verification Checkpoint: Implementation Complete

Before proceeding to Module 4, confirm:

| Check | Command | Expected |
|-------|---------|----------|
| Plan exists | `ls docs/plans/pipeline-components-plan.md` | File present |
| Scripts created | `ls src/agents/pipeline/*.py \| wc -l` | 5 to 7 files, including the extractor, chunker, embedder, writer, and orchestrate agent (you may have some bonus agents too!) |
| Syntax valid | `python -m py_compile src/agents/pipeline/*.py` | No errors |
| Extractor works | `python src/agents/pipeline/extractor.py demo-docs/[choose_file_name].md` | JSON output |

**If any script is missing or has errors:**
- Ask Claude to fix the specific error
- Re-run the swarm implementation for the failed component
- Check the error messages for guidance

---

### The Two Layers in Action

Notice what just happened:

```
┌─────────────────────────────────────────────────────────────┐
│  Layer 1: Claude Flow (Development-Time)                    │
│  ──────────────────────────────────────                     │
│  - goal-planner created the plan                            │
│  - swarm-advanced implemented the code                      │
│  - Multiple agents worked in parallel                       │
│  - You observed and validated                               │
└─────────────────────────────────────────────────────────────┘
                          │
                          │ generated
                          ▼
┌─────────────────────────────────────────────────────────────┐
│  Layer 2: Python Scripts (Runtime)                          │
│  ──────────────────────────────────                         │
│  - extractor.py, chunker.py, embedder.py, writer.py         │
│  - orchestrate.py coordinates them all                      │
│  - NO Claude Flow dependency at runtime                     │
│  - Users can run: python orchestrate.py demo-docs/          │
└─────────────────────────────────────────────────────────────┘
```

**Key Insight:** Claude Flow helped you BUILD the pipeline. The pipeline RUNS independently.

---

### Quiz 3:

**Question 1:** What is the benefit of creating a GitHub issue from the implementation plan?\
   a) GitHub issues are required by Python\
   b) It makes the code run faster\
   c) It creates a trackable work item that can be assigned, monitored, and closed upon completion\
   d) To increase the file count

**Question 2:** What does the swarm-advanced skill do during implementation?\
   a) Deletes old files\
   b) Spawns specialized agents (coder, tester, reviewer) that work in parallel to implement the plan\
   c) Sends emails to the team\
   d) Compresses the codebase

**Question 3:** After swarm implementation, what should you verify?\
   a) All scripts exist, pass syntax validation, and produce expected output\
   b) Only that files were created\
   c) That the code has many comments\
   d) That the files are large

**Answers:**
1. **c)** GitHub issues create trackable work items for assignment, monitoring, and verification
2. **b)** Swarm-advanced spawns specialized agents that work in parallel
3. **a)** Verify scripts exist, pass syntax validation, and produce expected output

---

## Module 4: Test and Enhance the Generated Pipeline (15 minutes)

### Overview

Now that you've generated all 5 pipeline scripts using natural language, it's time to:
1. **Test** the complete pipeline end-to-end
2. **Verify** data is stored correctly in Supabase
3. **Enhance** the pipeline using natural language prompts

---

### Exercise 4.1: Run the Complete Pipeline

**Task:** Test your generated pipeline with the demo documents using Claude Code.

**Instructions:**

**Step 1: Run the Full Pipeline with Claude Code (5 mins)**

In Claude Code, simply ask:

```
run the full pipeline 'src/agents/pipeline' on 'demo-docs/'
```

Claude Code will execute the orchestrator and display the results:

```
🚀 Starting pipeline for 16 documents
   Max parallel: 10
   Continue on error: True

📄 Processing: sample3.md
📄 Processing: doc15.md
📄 Processing: doc11.md
📄 Processing: doc13.md
📄 Processing: doc8.md
📄 Processing: doc5.md
  ✅ Success: 1 chunks, 1 embeddings
  ✅ Success: 1 chunks, 1 embeddings
📄 Processing: doc4.md
📄 Processing: doc6.md
  ✅ Success: 1 chunks, 1 embeddings
  ...

╔══════════════════════════════════════════════════════════════╗
║           DOCUMIND PIPELINE PROCESSING REPORT                ║
╚══════════════════════════════════════════════════════════════╝

📊 SUMMARY
────────────────────────────────────────────────────────────
Total Documents:        16
✅ Successful:          16 (100.0%)
❌ Failed:              0 (0.0%)

📦 OUTPUT
────────────────────────────────────────────────────────────
Total Chunks Created:   16
Total Embeddings:       16
Avg Chunks/Document:    1.0

⏱️  PERFORMANCE
────────────────────────────────────────────────────────────
Total Time:             0.94s
Avg Time/Document:      0.65s
Throughput:             17.0 docs/second

⚙️  STAGE BREAKDOWN (Average Times)
────────────────────────────────────────────────────────────
Extract:                0.105s
Chunk:                  0.051s
Embed:                  0.211s
Write:                  0.102s

❌ ERRORS BY STAGE
────────────────────────────────────────────────────────────
Extract failures:       0
Chunk failures:         0
Embed failures:         0
Write failures:         0

════════════════════════════════════════════════════════════
Generated at: 2025-12-12 07:44:14

📝 JSON report saved to: pipeline-results.json
```

**Alternative: Run directly from terminal:**

```bash
# Run the orchestrator with JSON output
python src/agents/pipeline/orchestrate.py -d demo-docs/ --max-parallel 10 --json-output pipeline-results.json
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
- Documents table should have new records
- Chunks table should have chunks
- Each chunk should have a 1536-dimension embedding vector

---

### 🚨 Plot Twist: No Chunks? No Problem!

**Wait, what's this?** You ran the verification and got something like:

```
Documents table: 5 ✅
Chunks table: 0 ❌
Embeddings: 0 ❌

"document_chunks table doesn't exist"
```

**Don't panic!** 🎉 This is actually a *teachable moment* (fancy way of saying "oops, we forgot something").

Your plan likely missed adding the `document_chunks` table to Supabase. The pipeline ran successfully with mock agents, but there's nowhere to store the real chunks!

**The Fix: Let Claude Handle It**

Simply ask Claude Code:

```
Create a GitHub issue for adding a document_chunks table with vector embeddings support, then implement it.
```

**What Happens Next:**

Claude will:
1. 📝 Create a detailed GitHub issue with the table schema
2. 🔨 Apply a Supabase migration to create the table
3. ✅ Verify the table exists with all the right columns
4. 🎯 Close the issue with implementation notes

**Example Output:**
```
✓ Created issue #20: Create document_chunks table with vector embeddings support
✓ Applied migration: create_document_chunks_table
✓ Table created with columns: id, document_id, chunk_index, content, embedding, word_count, metadata, created_at
✓ Vector index created for similarity search
✓ Closed issue #20
```

> **💡 Pro Tip:** It may take Claude a few iterations to get everything right—configuring real agents, fixing import paths, handling environment variables. But hey, it's still WAY faster than most development teams... be patient!

**Now Re-run the Pipeline:**

After Claude creates the table, run the pipeline again:

```
run the full pipeline 'src/agents/pipeline' on 'demo-docs/'
```

This time you'll see REAL HTTP requests flying:
- 🤖 `POST https://api.openai.com/v1/embeddings` - Real embeddings!
- 💾 `POST https://yourproject.supabase.co/rest/v1/document_chunks` - Real storage!

**Verify Again:**

```
Use the Supabase MCP to:
1. Count documents in the 'documents' table
2. List the first 5 chunks from 'document_chunks' table
3. Verify embeddings exist (check if embedding column is not null)
```

**Expected (Happy) Results:**
```
Documents table: 21 ✅ (5 original + 16 new)
Chunks table: 16 ✅
Chunks with embeddings: 16 ✅ (100%!)
```

**Isn't that fun and easy?** 🎊 You just:
- Discovered a missing database table
- Had AI create a GitHub issue
- Had AI implement the fix
- Re-ran the pipeline with real data

Welcome to the future of software development!

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

**Pro-Tip:** Plan the job, create GitHub issues (and optionally branches), use claude flow (goalie to plan and swarm to implement). 

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
   a) Manual editing doesn't work\
   b) It's faster to type natural language than code\
   c) Natural language is more precise than code\
   d) You describe WHAT you want, and the AI figures out HOW to implement it

**Question 2:** When testing generated code, what should you verify first?\
   a) The comments are detailed\
   b) The code is as short as possible\
   c) The variable names are creative\
   d) The script runs without syntax errors and handles edge cases

**Question 3:** What demonstrates the "Two Layers" concept in this session?\
   a) Claude Flow helped generate the pipeline scripts, which then run independently at runtime\
   b) We used two programming languages\
   c) We created two copies of each file\
   d) We ran the pipeline twice

**Answers:**
1. **d)** Natural language lets you focus on requirements; the AI handles implementation details
2. **d)** First verify the script runs and handles edge cases before checking other aspects
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

### Exercise 2.1 Solution: Goal-Planner

**The prompt:**
```
use @goal-planner to review the S5-Workshop Module 1 content and the
DocuMind PRD (docs/spec/documind-prd.md) to create an implementation
plan for the 5 pipeline components. Save the plan to
docs/plans/pipeline-components-plan.md
```

**Expected plan structure:**
```markdown
# Pipeline Components Implementation Plan

## Goal: Working Document Processing Pipeline

### Milestone 1: Extractor Component
**Preconditions:** Project structure exists, demo docs available
**Actions:** Create extractor.py with multi-format file reading
**Success Criteria:** Extracts text from .md, .txt, .pdf, .docx files

### Milestone 2: Chunker Component
**Preconditions:** Extractor produces valid output
**Actions:** Create chunker.py with semantic text splitting
**Success Criteria:** Produces ~500-word chunks with 50-word overlap

### Milestone 3: Embedder Component
**Preconditions:** Chunker produces valid chunks
**Actions:** Create embedder.py with OpenAI API integration
**Success Criteria:** Generates 1536-dim vectors using text-embedding-3-small

### Milestone 4: Writer Component
**Preconditions:** Supabase tables exist from S4
**Actions:** Create writer.py with Supabase client
**Success Criteria:** Stores documents and chunks with embeddings

### Milestone 5: Orchestrator Component
**Preconditions:** All 4 components work independently
**Actions:** Create orchestrate.py with asyncio coordination
**Success Criteria:** Processes batch of files end-to-end
```

**Key insight:** The goal-planner uses GOAP (Goal-Oriented Action Planning) to break complex goals into milestones with preconditions and success criteria.

---

### Exercise 3.1 & 3.2 Solution: Swarm Implementation

**Create the issue:**
```
create a GitHub issue for docs/plans/pipeline-components-plan.md
```

**Implement with swarm:**
```
implement the issue using the swarm-advanced skill
```

**Verification commands:**
```bash
# All scripts should exist
ls src/agents/pipeline/
# extractor.py  chunker.py  embedder.py  writer.py  orchestrate.py  __init__.py

# All should pass syntax check
python -m py_compile src/agents/pipeline/*.py

# Extractor should work on demo file
python src/agents/pipeline/extractor.py demo-docs/remote-work-policy.md
# Should output JSON with: success, title, content, file_type
```

**Key insight:** The swarm-advanced skill spawns coder, tester, and reviewer agents that work in parallel. This is 2-4x faster than sequential implementation and includes automatic quality checks.

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
