# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

DocuMind is an AI-Powered Knowledge Management System - a teaching application for the "AI-Powered Software Development with Agentic Engineering" course. It's a production-ready intelligent Q&A chatbot that answers questions from company documents using Claude, RAG (Retrieval-Augmented Generation), and agentic engineering patterns.

## Technology Stack

- **Languages**: JavaScript (Node.js 20+) + Python 3.10+
- **AI/LLM**: Claude (Anthropic), OpenRouter (multi-model), OpenAI (embeddings)
- **Database**: Supabase (PostgreSQL + pgvector)
- **Development**: GitHub Codespaces (primary), VS Code
- **Testing**: Jest

## Common Commands

```bash
# Install dependencies
npm install

# Run tests
npm test

# Run tests in watch mode
npm test:watch

# Lint source code
npm run lint

# Quick setup (copies .env.example to .env and installs deps)
npm run setup

# Launch Claude Code CLI
dsp
```

## Architecture

DocuMind uses a multi-layered RAG architecture with agentic orchestration:

```
User Interface (CLI/Web)
        ↓
RAG Pipeline (Query Processing)
├─ Query Understanding & Embedding
├─ Semantic Search (pgvector)
├─ Hybrid Ranking
├─ Context Assembly
├─ Answer Generation (OpenRouter multi-model)
└─ Citation Extraction
        ↓
Document Ingestion (Multi-Agent System)
├─ Extractor Agent (reads PDF, DOCX, XLSX)
├─ Chunker Agent (semantic splitting)
├─ Embedder Agent (vector generation)
└─ Writer Agent (database storage)
        ↓
Supabase Backend
├─ documents table
├─ document_chunks table (with embeddings)
├─ conversations & messages (memory)
├─ query_feedback (learning)
└─ evaluation_runs (metrics)
```

## Key Directories

- `src/documind/` - Main application code (config, processing logic)
- `.claude/` - Claude Code configuration
  - `agents/core/` - AI agent definitions (coder, reviewer, tester)
  - `skills/` - Reusable skills (e.g., supabase-mcp-installer)
  - `hooks/` - Automation scripts (SessionStart, UserPromptSubmit, Stop)
- `tests/` - Unit, integration, and e2e tests
- `docs/` - Documentation
  - `guides/` - SETUP.md, FAQ.md, TROUBLESHOOTING.md
  - `spec/` - documind-prd.md (detailed product requirements)
  - `workshops/` - Session materials (S1-S4)
  - `claude-conversations/` - Dialogue capture logs

## Environment Configuration

Required environment variables (see `.env.example`):

```bash
ANTHROPIC_API_KEY=sk-ant-xxx          # Claude API
OPENAI_API_KEY=sk-xxx                 # Embeddings
OPENROUTER_API_KEY=sk-or-xxx          # Multi-model access

# Database (Session 4+)
SUPABASE_URL=https://xxxxx.supabase.co
SUPABASE_ANON_KEY=eyJxxxxx
SUPABASE_SERVICE_KEY=eyJxxxxx
SUPABASE_ACCESS_TOKEN=sbp_xxxxx       # Required for MCP integration
```

## Hooks Configuration

The project uses Claude Code hooks (`.claude/settings.json`) for automation:
- **SessionStart** - Creates conversation log file
- **UserPromptSubmit** - Captures user interactions
- **Stop** - Session finalization

Dialogue logs are saved to `docs/claude-conversations/`.

## Git Workflow

This is a teaching project where students work from forks:
- **Origin**: Student's fork
- **Upstream**: `https://github.com/mamd69/heroforge-documind.git`
- Workshops distributed via `upstream/prep-workshops` branch
- Students pull workshop content into their `main` branch

## Course Session Roadmap

The codebase evolves through 10 sessions:
- S1-S2: Agentic engineering fundamentals
- S3: Skills, Subagents, Hooks (document upload pipeline)
- S4: Database & MCP (Supabase integration)
- S5: Multi-Agent Systems (parallel processing)
- S6: RAG Implementation (Q&A interface)
- S7: Advanced Parsing (PDF/DOCX/XLSX)
- S8: Vector Databases (pgvector optimization)
- S9: Memory & Learning (conversation + feedback)
- S10: Evaluation (RAGAS + TruLens)

## Agent Guidelines

When working as the coder agent (see `.claude/agents/core/coder.md`):
- Write readable, well-structured code
- Use meaningful variable and function names
- Include proper error handling
- Write unit tests for new utilities
