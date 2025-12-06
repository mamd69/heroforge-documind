# Session 3: Claude Code Advanced Features
## Skills, Subagents, Hooks & Building DocuMind

---

## What You'll Learn Today

### DocuMind Product (Sample App))
Understand the AI chatbot system we'll build across sessions 3-10

### Claude Skills
Create reusable workflows with custom Skills for document processing

### Subagents
Implement specialized task delegation with focused Subagents

### Hooks Automation
Configure automatic pre/post-task actions for workflow optimization

### DocuMind Foundation
Build the foundation of our AI-powered knowledge management system

### Best Practices
Understand when to use Skills vs Subagents vs Hooks

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
git checkout session-3-complete
```

| Phase | Time | Focus |
|-------|------|-------|
| Checkout branch | 30 sec | Get the code |
| Code walk-through | 10-15 min | Understand structure |
| Live demo | 10-15 min | See it work |
| Q&A | 5-10 min | Discussion |

**Total: 35 minutes** (fits our allocation!)

---

# Introduction & DocuMind PRD

## Recap: Claude Code Basics

What we learned in Session 2:

### Installation & Authentication
Installing Claude Code and configuring authentication

### CLI Workflow
Navigating the command-line interface

### Core Features
Code generation, debugging, testing, and documentation

---

## Introducing DocuMind

**Our course project: An AI-Powered Knowledge Management System**

A chatbot interface for company knowledge bases that enables employees to:
- Ask questions in natural language
- Receive accurate, source-grounded answers
- Query multiple document formats
- Learn from user feedback over time

**Real-world business value:** Transform static documentation into intelligent conversation

---

## Why DocuMind?

Demonstrates all course concepts through progressive enhancement:

### Real-World Application
Solves actual business problems in knowledge management

### Progressive Complexity
Each session adds concrete features we'll implement together

### Modern Stack
Uses production-ready technologies (Supabase, OpenRouter, RAG)

### Complete Journey
From foundation to evaluation (Sessions 3-10)

**By Session 10, you'll have a fully functional AI chatbot system**

---

## DocuMind Product Vision

**What Problem Are We Solving?**

- Company documentation scattered across PDFs, Word docs, spreadsheets
- Employees struggle to find information quickly
- Knowledge buried in static files
- No way to ask questions conversationally

**Our Solution:**

- Natural language Q&A interface
- Source attribution for all answers
- Multi-format document support
- Continuous learning from feedback

---

## DocuMind Architecture Overview

```
┌─────────────────────────────────────────────────────────┐
│                    DocuMind System                      │
├─────────────────────────────────────────────────────────┤
│                                                         │
│  ┌──────────────┐      ┌──────────────┐               │
│  │   Document   │──────│  Processing  │               │
│  │   Ingestion  │      │   Pipeline   │               │
│  │  (S7: PDF,   │      │ (S5: Multi-  │               │
│  │   DOCX)      │      │   Agent)     │               │
│  └──────────────┘      └──────────────┘               │
│         │                      │                        │
│         ▼                      ▼                        │
│  ┌──────────────────────────────────┐                 │
│  │   Supabase PostgreSQL + pgvector │                 │
│  │   (S4: MCP, S8: Vector Search)   │                 │
│  └──────────────────────────────────┘                 │
│         │                      │                        │
│         ▼                      ▼                        │
│  ┌──────────────┐      ┌──────────────┐               │
│  │     RAG      │      │  Conversation│               │
│  │   Engine     │──────│    Memory    │               │
│  │  (S6: Q&A)   │      │  (S9: Learn) │               │
│  └──────────────┘      └──────────────┘               │
│         │                      │                        │
│         ▼                      ▼                        │
│  ┌──────────────────────────────────┐                 │
│  │   Evaluation & Monitoring        │                 │
│  │   (S10: RAGAS/TruLens)           │                 │
│  └──────────────────────────────────┘                 │
│                                                         │
│  Foundation: S3 (Skills, Subagents, Hooks)            │
└─────────────────────────────────────────────────────────┘
```

---

## DocuMind Feature Roadmap

Progressive enhancement across sessions:

| Session | Feature | What We Build |
|---------|---------|---------------|
| **S3** | Foundation | Skills, Subagents, Hooks automation |
| **S4** | Database | Supabase MCP + custom document server |
| **S5** | Processing | Multi-agent document pipeline |
| **S6** | Q&A | RAG implementation with retrieval |
| **S7** | Parsing | PDF/DOCX advanced extraction |
| **S8** | Search | pgvector semantic search |
| **S9** | Learning | Conversation memory + feedback |
| **S10** | Quality | RAGAS/TruLens evaluation |

**Each session builds on the previous**

---

## Live Demo

### DEMO: Final Product Preview

A glimpse of Session 10's completed DocuMind

We'll see:
- Natural language queries to company knowledge base
- Source-attributed answers with citations
- Multi-document search across PDFs and docs
- Learning from user feedback
- Quality metrics and evaluation

**This is where we're headed**

---

# Claude Skills Deep Dive

## What are Claude Skills?

Skills extend Claude Code with reusable, specialized capabilities

**Key Characteristics:**
- Folders containing instructions, scripts, and resources
- Loaded automatically when relevant to the task
- Model-invoked (Claude decides when to use them)
- Composable (multiple Skills work together)
- Shareable across teams via git

**Think of Skills as specialized tools in Claude's toolkit**

---

## Anatomy of a Skill

Structure of a Claude Skill:

```
my-skill/
├── SKILL.md          # Core instructions (required)
├── scripts/          # Executable code (optional)
├── templates/        # Reusable templates (optional)
└── resources/        # Supporting files (optional)
```

### SKILL.md Components:

```yaml
---
name: my-skill
description: Clear description for when Claude should use this skill
---

# Skill Instructions

Detailed markdown instructions for Claude to follow...
```

**SKILL.md is the brain of your Skill**

---

## Three Types of Skills

Where Skills live and how they're shared:

### 1. Personal Skills
**Location:** `~/.claude/skills/`
- Available across all your projects
- Not shared with team
- Your personal automation library

### 2. Project Skills
**Location:** `.claude/skills/`
- Checked into git repository
- Shared with entire team
- Project-specific workflows

### 3. Plugin Skills
**Location:** Installed via npm or marketplace
- Published by community or vendors
- Installed like packages
- Reusable across ecosystem

---

## When to Create a Skill

Build a Skill when you have:

### Repetitive Workflows
Tasks you perform frequently across projects

### Specialized Knowledge
Domain expertise Claude needs repeatedly

### Multi-Step Processes
Complex workflows with specific ordering

### Team Standards
Enforcing consistent patterns across team

### Tool Integrations
Custom scripts or external tool interactions

**Example:** document-processor for DocuMind file handling

---

## Workshop Setup

### Before We Build Skills

Environment setup. Required for all hands-on exercises.

**Reference:** Full instructions in `S3-Documind-setup-guide.md` (provided with workshop materials)

---

## Setup Step 1: Fork & Launch Codespace

### Get Your Development Environment

1. **Fork the repository:** `github.com/mamd69/heroforge-documind`
2. **Create Codespace:** Click green "Code" button → Codespaces → "Create codespace on main"
3. **Wait for initialization** (2-3 minutes)

You'll know it's ready when you see:
- VS Code interface in your browser
- File explorer showing project files
- Terminal panel at the bottom

---

## Setup Step 2: Install Claude Code

### Your AI Development Assistant

In your Codespace terminal, run:

```bash
npm install -g @anthropic-ai/claude-code
```

Verify installation:
```bash
claude --version
```

Create the `dsp` alias (recommended):
```bash
echo 'alias dsp="claude"' >> ~/.bashrc
source ~/.bashrc
```

Now you can type `dsp` instead of `claude` to start Claude Code.

---

## Setup Step 3: Install Dialogue Reporter

### Capture Your Claude Conversations

Dialogue Reporter automatically saves your Claude conversations as markdown files.

```bash
npx dialogue-reporter install
```

**Why install this?**
- **Training:** Share effective prompts with instructors
- **Learning:** Review what worked and what didn't
- **Documentation:** Keep a record of your development process

Conversations save to `docs/claude-conversations/` automatically.

---

## Setup Step 4: Initialize Claude Code

### Configure Claude Code for This Project

After installing Dialogue Reporter, initialize Claude Code:

1. Launch Claude Code:
```bash
dsp
```

2. Inside Claude Code, run:
```
/init
```

3. Follow the prompts to set up project context

4. Exit: Type `exit`

This creates a `CLAUDE.md` with project-specific instructions and helps Claude understand the codebase.

---

## API Key Security

### Protect Your Credentials

**NEVER do this:**
- ❌ Paste API keys in Claude Code chat
- ❌ Include keys in screenshots
- ❌ Commit `.env` files to Git

**ALWAYS do this:**
- ✅ Add keys directly to `.env` file using editor
- ✅ Verify `.env` is in `.gitignore` first
- ✅ Rotate keys immediately if exposed

**Why?** Chat logs get saved. Keys in chat = keys exposed.

---

## Setup Step 5: Configure API Keys

### Environment Variables

1. Copy the example file:
```bash
cp .env.example .env
```

2. Edit `.env` with your API keys:
```bash
# Required for Claude Code
ANTHROPIC_API_KEY=sk-ant-YOUR_KEY_HERE

# Required for embeddings (Session 5+)
OPENAI_API_KEY=sk-YOUR_KEY_HERE
```

3. Verify:
```bash
source .env && echo $ANTHROPIC_API_KEY
```

4. **Security:** Never commit `.env` files to git!  Claude may have updated your gitignore.  If not, tell it...
```
Update gitignore with .env
```

---

## Setup Step 6: Launch Claude Code

### Test Your Environment

```bash
dsp
```

Try this test prompt:
```
Hello! Can you verify my environment is set up correctly for the DocuMind workshop?
```
View your 
Exit when done: Type `exit` or press **Ctrl+C**

---

## Setup Checklist

### Ready to Build?

- [ ] Codespace running (VS Code in browser)
- [ ] Claude Code installed (`claude --version` works)
- [ ] Dialogue Reporter installed
- [ ] Ran `/init` inside Claude Code
- [ ] `.env` file with API keys
- [ ] `npm install` completed
- [ ] `dsp` command launches Claude Code

**All checked?** You're ready for the hands-on demos!

---

## Creating a Document Processor Skill

For DocuMind, we need to handle document uploads consistently

**What the Skill will do:**
- Accept file path as input
- Validate file format (PDF, DOCX, TXT, MD)
- Extract metadata (title, author, date)
- Return structured summary
- Log processing for audit trail

**Why as a Skill?**
- Reusable across document ingestion tasks
- Consistent validation logic
- Team can extend and improve
- Separates concerns from main application

---

## Skill-Builder: Our Meta-Skill

### We Have a Skill That Builds Skills!

Before we create our document-processor skill manually, let's look at something powerful.

In this project, we have a **skill-builder** skill located at:
```
.claude/skills/skill-builder/SKILL.md
```

This skill teaches Claude how to create production-ready skills with:
- Proper YAML frontmatter
- Progressive disclosure architecture
- Complete file/folder structure

**Let's use it to build our document-processor skill!**

---

## Live Demo

### DEMO: Creating document-processor Skill

Let's explore our first Project Skill **using Claude Code with natural language**

**Step 1:** Review the skill-builder skill (instructor shows the SKILL.md)

**Step 2:** Launch Claude Code in your Codespace:
```bash
dsp
```

**Step 3:** Give Claude this prompt:

---

## Demo Prompt: Create Document Processor Skill

### Natural Language Instruction to Claude

```
Using the skill-builder skill in this project, create a new skill
called "document-processor" for our DocuMind knowledge management system.

The skill should:
- Process document uploads with validation
- Support PDF, DOCX, TXT, and MD file formats
- Extract metadata (title, author, date, file size)
- Return a structured JSON summary
- Log processing events for audit trails

Create it as a project skill in .claude/skills/document-processor/
with proper YAML frontmatter and progressive disclosure structure.

Include example usage in the SKILL.md showing how to invoke it
for processing a sample document.
```

---

## Demo: Watch Claude Work

### What Claude Will Do

1. **Read** the skill-builder SKILL.md for guidance
2. **Create** the directory structure:
   ```
   .claude/skills/document-processor/
   └── SKILL.md
   ```
3. **Write** proper YAML frontmatter with name and description
4. **Add** progressive disclosure sections (Overview, Quick Start, Detailed Guide)
5. **Include** example usage and troubleshooting

**Key Insight:** We used natural language, not manual file creation!

---

## Demo: Review the Generated Skill

### What to Look For

After Claude creates the skill, review:

- **YAML frontmatter:** Does it have `name` and `description`?
- **Description quality:** Does it include "what" AND "when to use"?
- **Structure:** Does it follow progressive disclosure?
- **Examples:** Are usage examples clear and actionable?

```bash
# View the generated skill
cat .claude/skills/document-processor/SKILL.md
```

---

## Testing Your New Skill

### Invoke the Skill

Ask Claude to use your new skill:

```
Process the file docs/sample.md using the document-processor skill
and show me the extracted metadata.
```

Claude should:
1. Recognize the document-processor skill is relevant
2. Follow the skill's instructions
3. Return structured output

**This is model-invoked automation in action!**

---

## Skill Best Practices

Write effective Skills that Claude uses correctly:

**01.** Clear, specific descriptions (helps Claude know when to invoke)

**02.** Detailed instructions with examples

**03.** Define expected inputs and outputs

**04.** Include error handling guidance

**05.** Provide example usage in SKILL.md

**06.** Test with various scenarios

**07.** Document assumptions and limitations

**Remember:** You're teaching Claude a new capability

---

## Skills vs. Traditional Scripts

Key differences:

| Scripts | Skills |
|---------|--------|
| You invoke explicitly | Claude invokes automatically |
| Fixed execution path | Flexible based on context |
| Single purpose | Composable with other Skills |
| Code-only | Instructions + code + resources |
| No AI understanding | AI interprets and adapts |

**Skills are AI-aware automation, not just scripts**

---

# Subagents Explained

## What are Subagents?

Specialized instances of Claude with narrow focus and expertise

**Key Characteristics:**
- Defined in `.claude/subagents/` directory
- Each Subagent has specific role and capabilities
- Delegate tasks instead of doing everything yourself
- Maintain focused context for their domain
- Can be invoked explicitly or automatically

**Think of Subagents as team members with specific expertise**

---

## Delegation Pattern

Why delegate instead of doing everything?

### Problem: Context Overload
Main Claude instance has full project context - can be overwhelming

### Solution: Specialized Subagents
Delegate focused tasks to experts with limited scope

**Example:**
- Main Claude: Project coordination
- Summarizer Subagent: Extract key points from documents
- Validator Subagent: Check data quality
- Formatter Subagent: Apply formatting standards

**Benefit:** Better results through specialization

---

## Anatomy of a Subagent

Structure of a Subagent definition:

```
.claude/subagents/
└── summarizer.md
```

### Subagent Definition (summarizer.md):

```yaml
---
name: summarizer
role: Document summarization specialist
capabilities:
  - Extract key points from long documents
  - Identify main themes and topics
  - Generate concise summaries
  - Preserve critical details
constraints:
  - Focus only on summarization
  - Do not modify original content
  - Return structured output
---

# Summarizer Subagent

You are a specialized document summarizer...
```

---

## Creating a Document Summarizer Subagent

For DocuMind, we need consistent document summarization

**What the Subagent will do:**
- Receive document content
- Extract key information (purpose, topics, entities)
- Generate concise summary (3-5 sentences)
- Identify important sections
- Return structured JSON output

**Why as a Subagent?**
- Focused expertise in summarization
- Consistent output format
- No distraction from other project concerns
- Can be improved independently

---

## Live Demo

### DEMO: Building summarizer Subagent

Let's explore our specialized summarization expert **using Claude Code with natural language**

**Step 1:** Launch Claude Code (if not already running):
```bash
dsp
```

**Step 2:** Give Claude this prompt:

---

## Demo Prompt: Create Summarizer Subagent

### Natural Language Instruction to Claude

```
Create a subagent called "summarizer" for our DocuMind knowledge
management system.

The subagent should be a document summarization specialist with
these characteristics:

Role: Document summarization expert
Capabilities:
- Extract key points from long documents
- Identify main themes and topics
- Generate concise 3-5 sentence summaries
- Preserve critical details
- Return structured JSON output

Constraints:
- Focus only on summarization tasks
- Do not modify original content
- Always return structured output format

Output format should be JSON with:
- summary (string)
- topics (array of strings)
- entities (array of key people, dates, concepts)
- keyPoints (array of bullet points)

Create it at .claude/subagents/summarizer.md with proper YAML
frontmatter including name, role, capabilities, and constraints.
Include example interactions showing input and expected output.
```

---

## Demo: Watch Claude Create the Subagent

### What Claude Will Do

1. **Create** the subagents directory if needed
2. **Write** `.claude/subagents/summarizer.md` with:
   - YAML frontmatter (name, role, capabilities, constraints)
   - Detailed instructions for summarization
   - Output format specification
   - Example interactions

**Review the generated file:**
```bash
cat .claude/subagents/summarizer.md
```

---

## Demo: Test the Subagent

### Delegate a Task

Ask Claude to use your new subagent:

```
Delegate to the summarizer subagent to analyze the content
of docs/sample.md and provide a structured summary.
```

Watch Claude hand off the task to the specialist and return structured output

---

## Subagent Communication Patterns

How Subagents interact:

### 1. Explicit Delegation
```
"Delegate to summarizer to extract key points from quarterly_report.pdf"
```

### 2. Automatic Invocation
Claude recognizes when Subagent expertise needed

### 3. Chain Delegation
Subagent A → Subagent B → Subagent C

### 4. Parallel Execution
Multiple Subagents working simultaneously (future feature)

**Currently:** Sequential delegation pattern

---

## Managing Subagent Context

Controlling what Subagents know:

### Subagent Context Scope
- Limited to their defined role and capabilities
- Don't see full project history
- Focus only on delegated task
- Return results to main Claude

### Benefits of Limited Context
- Faster processing (less to consider)
- More focused results
- Lower token usage
- Clearer separation of concerns

**Trade-off:** Less context means Subagents can't make broader decisions

---

## Skills vs. Subagents

When to use which?

| Skills | Subagents |
|--------|-----------|
| Reusable workflows | Specialized expertise |
| Tools and scripts | Task delegation |
| Enhance capabilities | Divide and conquer |
| Can include code execution | Focus on AI reasoning |
| Best for: processes | Best for: analysis |

**Example:**
- **Skill:** document-processor (validates and processes files)
- **Subagent:** summarizer (analyzes content and extracts meaning)

**Often used together for powerful workflows**

---

## Subagent Best Practices

Design effective Subagents:

**01.** Single responsibility principle (do one thing well)

**02.** Clear role definition (what's in scope, what's not)

**03.** Explicit capabilities list

**04.** Define output format and structure

**05.** Set constraints and boundaries

**06.** Include example interactions

**07.** Test with various input types

**Remember:** Narrow focus = better results

---

# Hooks for Automation

## What are Hooks?

Automated scripts that run at specific points in Claude's workflow

**Lifecycle Events:**
- **Pre-task:** Before Claude starts a task
- **Post-task:** After Claude completes a task
- **Post-edit:** After any file is modified
- **Session-start:** When Claude session begins
- **Session-end:** When Claude session ends

**Think of Hooks as automated quality gates and helpers**

---

## Why Use Hooks?

Automate repetitive quality and setup tasks:

### Without Hooks
1. Claude generates code
2. You manually run formatter
3. You manually run linter
4. You manually update documentation
5. You manually commit changes

### With Hooks
1. Claude generates code
2. **post-edit hook:** Auto-formats code
3. **post-task hook:** Runs linter, updates docs
4. **session-end hook:** Creates summary report

**Benefit:** Consistency and time savings

---

## Pre-Task Hooks

Run before Claude starts working

**Common Use Cases:**
- Validate environment setup
- Load project context
- Check prerequisites
- Set up temporary resources
- Log task initiation

**Example for DocuMind:**
```bash
# .claude/hooks/pre-task.sh
# Validate document format before processing
if [[ "$TASK_TYPE" == "document-process" ]]; then
    validate_document_format
fi
```

---

## Post-Task Hooks

Run after Claude completes a task

**Common Use Cases:**
- Run tests to verify changes
- Format code automatically
- Update documentation
- Commit changes to git
- Send notifications
- Log completion metrics

**Example for DocuMind:**
```bash
# .claude/hooks/post-task.sh
# Run tests after implementation
npm test
# Update API documentation
npm run docs:generate
```

---

## Post-Edit Hooks

Run after any file modification

**Common Use Cases:**
- Auto-format on save (Prettier, Black)
- Run linters (ESLint, Pylint)
- Update table of contents
- Regenerate type definitions
- Sync related files

**Example for DocuMind:**
```bash
# .claude/hooks/post-edit.sh
# Auto-format markdown files
if [[ "$FILE_PATH" == *.md ]]; then
    npx prettier --write "$FILE_PATH"
fi
```

---

## Session Hooks

Run at session boundaries

### Session-Start Hook
```bash
# .claude/hooks/session-start.sh
# Restore context from previous session
load_session_memory
display_recent_tasks
check_for_updates
```

### Session-End Hook
```bash
# .claude/hooks/session-end.sh
# Generate session summary
create_work_summary
save_session_state
export_metrics
```

**Keep continuity across sessions**

---

## Hook Configuration

Hooks location and structure:

```
.claude/hooks/
├── pre-task.sh          # Before task starts
├── post-task.sh         # After task completes
├── post-edit.sh         # After file edit
├── session-start.sh     # Session beginning
└── session-end.sh       # Session ending
```

**Requirements:**
- Must be executable (`chmod +x`)
- Must exit with status code (0 = success)
- Can be any language (bash, Python, Node.js)
- Access to environment variables

---

## Live Demo

### DEMO: Creating post-edit Hook

Let's explore auto-formatting markdown files in DocuMind **using Claude Code with natural language**

**Step 1:** Launch Claude Code (if not already running):
```bash
dsp
```

**Step 2:** Give Claude this prompt:

---

## Demo Prompt: Create Post-Edit Hook

### Natural Language Instruction to Claude

```
Create a post-edit hook for our DocuMind project that automatically
formats files after Claude edits them.

The hook should:
1. Detect the file type by extension
2. For markdown files (.md): run prettier --write
3. For JavaScript/TypeScript (.js, .ts, .tsx): run prettier --write
4. For Python (.py): run black (if available)
5. Exit with status 0 on success

Create it at .claude/hooks/post-edit.sh and make it executable.

Include:
- Proper shebang (#!/bin/bash)
- Comments explaining what the hook does
- Error handling that doesn't break the workflow
- Logging of what was formatted

The hook receives the file path as $FILE_PATH environment variable.
```

---

## Demo: Watch Claude Create the Hook

### What Claude Will Do

1. **Create** the hooks directory if needed
2. **Write** `.claude/hooks/post-edit.sh` with:
   - Proper shebang and comments
   - File type detection logic
   - Formatter commands for each type
   - Error handling
3. **Make it executable** with chmod +x

**Review the generated hook:**
```bash
cat .claude/hooks/post-edit.sh
```

---

## Demo: Test the Hook

### Edit a File and Watch Auto-Format

Ask Claude to make a file change:

```
Add a new section called "Testing" to docs/sample.md
with some sample content about testing strategies.
```

**Watch:** After Claude edits the file, the post-edit hook runs automatically!

Check the terminal output for formatting confirmation

---

## Hook Environment Variables

Available to your hooks:

```bash
$TASK_DESCRIPTION    # What Claude is working on
$FILE_PATH           # Modified file (post-edit)
$HOOK_TYPE           # Which hook is running
$PROJECT_ROOT        # Project directory
$CLAUDE_SESSION_ID   # Current session ID
```

**Use these to make hooks context-aware**

**Example:**
```bash
if [[ "$FILE_PATH" == *test.js ]]; then
    # Only run tests if test file was edited
    npm test "$FILE_PATH"
fi
```

---

## Hook Best Practices

Write effective automation hooks:

**01.** Keep hooks fast (users wait for completion)

**02.** Handle errors gracefully (non-zero exit stops workflow)

**03.** Log actions for debugging

**04.** Make hooks idempotent (safe to run multiple times)

**05.** Test hooks independently before integration

**06.** Document what each hook does

**07.** Version control hooks with project

**Avoid:** Long-running or network-dependent operations in hooks

---

## Skills + Subagents + Hooks = Powerful Workflow

How they work together in DocuMind:

**Scenario:** User uploads a PDF document

1. **Pre-task Hook:** Validates file exists and format is supported
2. **Skill (document-processor):** Extracts text and metadata
3. **Subagent (summarizer):** Analyzes content and generates summary
4. **Post-task Hook:** Runs tests, formats output, commits to git
5. **Session-end Hook:** Logs processing metrics

**Result:** Fully automated, consistent document processing pipeline

---

# Review: What We Built Today

## DocuMind Structure After Demos

After our three demos, your heroforge-documind repository now has:

```
heroforge-documind/
├── .claude/
│   ├── skills/
│   │   ├── skill-builder/        # Pre-existing meta-skill
│   │   └── document-processor/   # Created in Demo 1
│   ├── subagents/
│   │   └── summarizer.md         # Created in Demo 2
│   └── hooks/
│       └── post-edit.sh          # Created in Demo 3
├── src/
│   └── documind/
├── tests/
├── docs/
└── package.json
```

**All created using natural language prompts!**

---

## Testing Everything Together

### Integration Test

Try this comprehensive prompt in Claude Code:

```
I have a markdown file at docs/sample.md. Please:
1. Use the document-processor skill to validate and extract metadata
2. Delegate to the summarizer subagent to analyze the content
3. Show me the combined results

The post-edit hook should auto-format any files you modify.
```

Watch how Skills, Subagents, and Hooks work together seamlessly!

---

## DocuMind PRD Document

### Product Requirements Document

Reference the PRD for the complete project vision:

**Location:** `docs/DocuMind-PRD.md`

**Contents:**
- Product vision and goals
- User stories and personas
- Feature requirements per session
- Architecture diagrams
- Technical specifications
- Success metrics

**Purpose:** Single source of truth for what we're building across Sessions 3-10

---

# Working Effectively with Claude Code

## 5 Ways to Get Better Results from Claude Code

### 1. Demand proof, not promises
Ask "show me the diff" or "run the tests" — don't accept "I've updated the file" without evidence.

### 2. Break tasks into atomic steps
Instead of "refactor auth," try: "list files touching auth → show current flow → update login function → run tests."

### 3. Use "then verify" as a suffix
"Add the env variable to .env, then cat the file to confirm." Build verification into every request.

### 4. Ask for state before and after
"Show test output, make the fix, show test output again." Clear before/after comparison reveals what changed.

### 5. Be skeptical of "Done!" without artifacts
No terminal output, diff, or file contents? Assume nothing happened. Real completions produce evidence.

**Core principle:** Treat Claude Code like a junior dev who's eager to please but needs to show their work.

---

# Wrap-up & Preview

## Key Takeaways

What we learned today:

### Natural Language Development
We built Skills, Subagents, and Hooks using **prompts, not manual coding**

### DocuMind Vision
AI-powered knowledge management system we'll build together

### Claude Skills
Reusable workflows created by describing what you need

### Subagents
Specialized delegation created through natural language definition

### Hooks
Automated actions configured by explaining the desired behavior

### Integration
Skills, Subagents, and Hooks work together seamlessly

---

## Skills vs. Subagents vs. Hooks

Quick reference for decision-making:

| Use Case | Use This |
|----------|----------|
| Reusable multi-step workflow | **Skill** |
| Tool integration or script execution | **Skill** |
| Specialized analysis or reasoning | **Subagent** |
| Task delegation with narrow focus | **Subagent** |
| Automatic formatting/linting | **Hook** |
| Pre/post task validation | **Hook** |
| Session continuity | **Hook** |

**Often used together:** Skill invokes Subagent, Hook runs tests

---

## DocuMind Progress Tracker

Where we are and where we're going:

✅ **Session 3 (Today):** Foundation - Skills, Subagents, Hooks

🔜 **Session 4 (Next):** MCP connectivity - Supabase + custom document server

📋 **Session 5:** Multi-agent processing pipeline

📋 **Session 6:** RAG implementation for Q&A

📋 **Session 7:** PDF/DOCX advanced parsing

📋 **Session 8:** pgvector semantic search

📋 **Session 9:** Conversation memory + learning

📋 **Session 10:** RAGAS/TruLens evaluation

**Progressive enhancement - each session builds on previous**

---

## What's Next: MCP and A2A

Preview of Session 4:

### Model Context Protocol (MCP)
Standardized way to connect Claude to external systems

### Supabase MCP Server
Connect DocuMind to PostgreSQL database

### Custom MCP Server
Build documind-mcp for specialized document operations

### Agent-to-Agent Communication
Multi-agent coordination patterns

**We'll store our first documents in the database**

---

## Practice Before Next Session

Reinforce today's learning:

**01.** Create a custom Skill for your own use case

**02.** Build a Subagent with specialized knowledge

**03.** Implement a hook that runs tests after code changes

**04.** Start your own mini DocuMind with basic file processing

**05.** Experiment with Skills + Subagents + Hooks together

**06.** Review the DocuMind PRD document

**07.** Prepare questions for Session 4

---

# Q&A and Additional Resources

## Questions?

Areas we can clarify:

### Skills
- Structure and organization
- When to create vs. use existing
- Testing and debugging

### Subagents
- Delegation patterns
- Context management
- Output formats

### Hooks
- Lifecycle events
- Error handling
- Performance considerations

### DocuMind
- Architecture questions
- Feature planning
- Technical choices

---

## Additional Resources

Links for further study:

### Claude Code Skills Documentation
docs.claude.com/en/docs/claude-code/skills

### Subagents Guide
docs.claude.com/en/docs/claude-code/subagents

### Hooks Reference
docs.claude.com/en/docs/claude-code/hooks

### Skills Marketplace
skillsmp.com (13,000+ Skills)

### DocuMind PRD
docs/DocuMind-PRD.md (in course repository)

### Community
Discord, Reddit r/ClaudeAI, Anthropic forums

---

## Session Complete

**Thank you for participating!**

### Remember:
- Skills = Reusable workflows
- Subagents = Specialized delegation
- Hooks = Automated actions
- Together = Powerful development system

### Next Session:
**MCP and A2A: Model Context Protocol & Agent-to-Agent Communication**

**See you next time!**

---
