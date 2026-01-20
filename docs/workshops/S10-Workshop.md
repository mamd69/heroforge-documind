# HeroForge.AI Course: AI-Powered Software Development
## Lesson 10 Workshop: Evaluation & Optimization Tools

**Workshop Duration:** 60 minutes\
**Structure:** 4 modules × 15 minutes each\
**Goal:** Build complete evaluation system for DocuMind

---

## Before You Begin: Plan Your Work!

> **📋 Reminder:** In Session 3, we learned about **Issue-Driven Development** - the practice of creating GitHub Issues *before* starting work. This ensures clear requirements, enables collaboration, and creates traceability between your code and original requirements.
>
> **Before diving into this workshop:**
> 1. Create a GitHub Issue for the features you'll build today
> 2. Reference that issue in your branch name (`issue-XX-feature-name`)
> 3. Include `Closes #XX` or `Relates to #XX` in your commit messages
>
> 👉 See [S3-Workshop: Planning Your Work with GitHub Issues](./S3-Workshop.md#planning-your-work-with-github-issues-5-minutes) for the full workflow.

---

## Workshop Overview

By the end of this workshop, you will have:
- ✅ Implemented RAGAS evaluation pipeline
- ✅ Integrated TruLens monitoring
- ✅ Created A/B testing framework
- ✅ Set up quality gates in CI/CD
- ✅ Built evaluation dashboard
- ✅ Deployed production monitoring

---

## Install Required Packages

**🖥️ Run in Terminal:**
```bash
pip install ragas trulens langchain-openai
```

**What are these packages?**
- **ragas** - RAG Assessment framework for evaluating retrieval-augmented generation pipelines with metrics like faithfulness, answer relevancy, and context precision
- **trulens** - Observability and evaluation toolkit for LLM applications with feedback functions and dashboards
- **langchain-openai** - LangChain integration for OpenAI models, required by RAGAS 0.4.x to configure the LLM and embeddings used during evaluation

---

## Prerequisites Check (Before Starting)

**🖥️ Run in Terminal:**
```bash
cd /workspaces/heroforge-documind
python -c "
import sys
import warnings
warnings.filterwarnings('ignore', category=DeprecationWarning)

print(f'Python {sys.version_info.major}.{sys.version_info.minor}', end=' ')
print('✅' if sys.version_info >= (3, 10) else '❌ Need 3.10+')

try:
    import ragas
    print(f'ragas {ragas.__version__} ✅')
except ImportError:
    print('ragas ❌ Not installed - run: pip install ragas')

try:
    from trulens.core import TruSession
    print('trulens ✅')
except ImportError:
    try:
        import trulens_eval
        print('trulens ✅ (legacy trulens_eval)')
    except ImportError:
        print('trulens ❌ Not installed - run: pip install trulens')

try:
    from langchain_openai import ChatOpenAI, OpenAIEmbeddings
    print('langchain-openai ✅')
except ImportError:
    print('langchain-openai ❌ Not installed - run: pip install langchain-openai')

try:
    from src.documind.hybrid_search import HybridSearcher
    print('DocuMind modules ✅')
except ImportError as e:
    print(f'DocuMind modules ❌ Error: {e}')
"
```

**Expected output:**
```
Python 3.10 ✅
ragas 0.4.3 ✅
trulens ✅
langchain-openai ✅
DocuMind modules ✅
```

---

### ⚠️ Important: Search Response Format

Before implementing evaluation, understand how DocuMind search returns data:

**Search Response Structure:**
```python
# DocuMind search returns a wrapper object
search_result = search_service.search(query)

# Response structure:
{
    "success": True,
    "documents": [
        {"id": "...", "content": "...", "score": 0.92},
        {"id": "...", "content": "...", "score": 0.88}
    ],
    "count": 2
}
```

**Common Error:** Passing the full response to evaluators expecting a list:
```python
# ❌ WRONG - RAGAS expects a list, not a dict
contexts = search_result  # This will fail!

# ✅ CORRECT - Unwrap the documents
contexts = search_result['documents']
```

**Helper Function Pattern:**
```python
def unwrap_search_response(response):
    """Extract documents list from search response wrapper."""
    if isinstance(response, dict) and 'documents' in response:
        return response['documents']
    # Fallback if already a list
    return response if isinstance(response, list) else []
```

Use this pattern whenever passing search results to RAGAS or other evaluators.

---

## Module 1: RAGAS Evaluation Implementation (15 minutes)

### Concept Review

**What is RAGAS?**

RAGAS (RAG Assessment) is an open-source framework specifically designed to evaluate RAG pipelines. Unlike traditional NLP metrics like BLEU or ROUGE that only measure text similarity, RAGAS measures the semantic quality of responses through four specialized metrics.

**The Four Core RAGAS Metrics (0.0 to 1.0 scale):**

| Metric | Target | What It Measures |
|--------|--------|------------------|
| **Faithfulness** | ≥0.70 | Is the answer grounded in retrieved context? (Hallucination detector) |
| **Answer Relevancy** | ≥0.80 | Does the answer address the question asked? |
| **Context Precision** | ≥0.80 | Are relevant chunks ranked higher in retrieval? |
| **Context Recall** | ≥0.85 | Did we retrieve all necessary information? |

**Why Each Metric Matters:**

1. **Faithfulness** - Detects hallucinations by checking if each claim in the answer is supported by the retrieved context. A confident-sounding but fabricated answer scores low.

2. **Answer Relevancy** - Catches off-topic responses. An answer can be factually correct but completely miss the question being asked.

3. **Context Precision** - Evaluates retrieval ranking. If relevant information is buried at position 10 but irrelevant content is at position 1, this metric suffers.

4. **Context Recall** - Measures completeness. High precision + low recall means good ranking but missing critical information.

**Key Insight:** Each metric catches different failure modes - you need all four for complete visibility into your RAG system's quality.

---

### Learning Objectives
- Understand RAGAS metrics
- Create evaluation dataset
- Implement RAGASEvaluator class
- Run first evaluation

### Exercise 1.1: Create Test Dataset (5 minutes)

**Task:** Create a comprehensive test dataset with diverse query types based on the HeroForge knowledge base.

**Query Types:**
- **Factual** - Direct fact-based questions (What, Who, How many)
- **Procedural** - How to do something step-by-step
- **Comparative** - Compare policies or options
- **Negative** - Questions that SHOULD return "I don't know" (not in knowledge base)

**🖥️ Run in Terminal:**
```bash
# Create directories for this workshop
mkdir -p data results scripts src/evaluation

# Create test dataset based on actual HeroForge knowledge base content
cat > data/evaluation_dataset.json << 'EOF'
[
  {
    "question": "How many vacation days do full-time employees receive per year?",
    "type": "factual",
    "ground_truth": "Full-time employees receive 15 days of paid vacation per year, accruing at 1.25 days per month."
  },
  {
    "question": "How do I request time off?",
    "type": "procedural",
    "ground_truth": "All time off requests must be submitted through the HR portal at least two weeks in advance."
  },
  {
    "question": "How many sick leave days do employees receive per year?",
    "type": "factual",
    "ground_truth": "Employees receive 10 days of paid sick leave per year."
  },
  {
    "question": "When is a doctor's note required for sick leave?",
    "type": "procedural",
    "ground_truth": "A doctor's note is required for absences over 3 consecutive days."
  },
  {
    "question": "How many personal days do employees get per year?",
    "type": "factual",
    "ground_truth": "Employees receive 3 personal days per year that can be used for any reason."
  },
  {
    "question": "How do I report a security incident?",
    "type": "procedural",
    "ground_truth": "Report security incidents to security@company.com or through the portal at https://security.company.com/report."
  },
  {
    "question": "What are the password requirements?",
    "type": "factual",
    "ground_truth": "Passwords must be at least 12 characters and include uppercase, lowercase, numbers, and symbols. Passwords must be changed every 90 days."
  },
  {
    "question": "How many vacation days can be carried over to the next year?",
    "type": "factual",
    "ground_truth": "Up to 5 days of unused vacation can be carried over to the next year."
  }
]
EOF

echo "✅ Test dataset created with 8 queries based on actual knowledge base content"
```

**Expected Output:**
```
✅ Test dataset created with 8 queries based on actual knowledge base content
```

> **Note:** The test dataset includes a `type` field to help analyze evaluation results by question category.

### Exercise 1.2: Implement RAGASEvaluator (5 minutes)

**Task:** Create a production-ready evaluation class that works with DocuMind's `ProductionQA`.

**📝 Create file `src/evaluation/ragas_evaluator.py`:**
```python
# src/evaluation/ragas_evaluator.py
import os
from typing import List, Dict
from statistics import mean
from datetime import datetime

from ragas import evaluate
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    answer_correctness
)
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from datasets import Dataset
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from dotenv import load_dotenv

load_dotenv()


class RAGASEvaluator:
    """Production RAGAS evaluator for DocuMind (compatible with RAGAS 0.4.x)"""

    def __init__(self, rag_pipeline):
        """
        Initialize evaluator with a RAG pipeline.

        Args:
            rag_pipeline: ProductionQA instance with query() method
        """
        self.rag_pipeline = rag_pipeline

        # Configure LLM for RAGAS evaluation (judges answer quality)
        llm = ChatOpenAI(
            model="gpt-4o-mini",
            temperature=0,
            api_key=os.getenv("OPENAI_API_KEY")
        )
        self.llm = LangchainLLMWrapper(llm)

        # Configure embeddings for RAGAS (measures semantic similarity)
        embeddings = OpenAIEmbeddings(
            model="text-embedding-3-small",
            api_key=os.getenv("OPENAI_API_KEY")
        )
        self.embeddings = LangchainEmbeddingsWrapper(embeddings)

        # Metrics to evaluate
        # - faithfulness: Is the answer grounded in the retrieved context?
        # - answer_relevancy: Is the answer relevant to the question?
        # - answer_correctness: Is the answer factually correct vs ground_truth?
        self.metrics = [
            faithfulness,
            answer_relevancy,
            answer_correctness
        ]

    def evaluate(self, test_dataset: List[Dict]) -> Dict:
        """Run RAGAS evaluation on test dataset"""
        print(f"\n{'='*60}")
        print(f"RAGAS EVALUATION STARTED: {datetime.now()}")
        print(f"{'='*60}")
        print(f"Test cases: {len(test_dataset)}")

        # Generate answers for each test case
        eval_data = []
        for i, test_case in enumerate(test_dataset, 1):
            print(f"[{i}/{len(test_dataset)}] {test_case['question'][:50]}...")

            # ProductionQA.query() returns dict with 'answer' and 'sources'
            result = self.rag_pipeline.query(test_case['question'])

            # Extract contexts from sources (ProductionQA uses 'preview' field)
            contexts = [
                source.get('preview', '')
                for source in result.get('sources', [])
                if source.get('preview')  # Filter out empty previews
            ]

            # Build evaluation record (RAGAS 0.4.x format)
            eval_data.append({
                'question': test_case['question'],
                'answer': result['answer'],
                'contexts': contexts,
                'ground_truth': test_case.get('ground_truth', '')
            })

        # Run RAGAS evaluation with configured LLM and embeddings
        print("\nRunning RAGAS metrics...")
        dataset = Dataset.from_list(eval_data)
        results = evaluate(
            dataset=dataset,
            metrics=self.metrics,
            llm=self.llm,
            embeddings=self.embeddings
        )

        return self.format_results(results)

    def format_results(self, results) -> Dict:
        """Format RAGAS results with pass/fail indicators"""
        # RAGAS 0.4.x returns a Dataset - convert to pandas and get means
        df = results.to_pandas()

        formatted = {}
        metric_names = ['faithfulness', 'answer_relevancy', 'answer_correctness']

        for metric in metric_names:
            if metric in df.columns:
                # Get mean, filtering out NaN values
                values = df[metric].dropna().tolist()
                formatted[metric] = mean(values) if values else 0.0
            else:
                formatted[metric] = 0.0

        # Calculate average of available metrics
        valid_scores = [v for v in formatted.values() if v > 0]
        formatted['average_score'] = mean(valid_scores) if valid_scores else 0.0

        # Add pass/fail based on thresholds
        thresholds = {
            'faithfulness': 0.70,
            'answer_relevancy': 0.80,
            'answer_correctness': 0.70
        }

        formatted['passed'] = all(
            formatted.get(metric, 0) >= threshold
            for metric, threshold in thresholds.items()
        )

        # Include per-question scores for detailed analysis
        formatted['per_question'] = df.to_dict('records')

        return formatted
```

> **Why the LangChain wrappers?** RAGAS 0.4.x requires explicit LLM and embedding configuration. The `LangchainLLMWrapper` wraps an LLM (gpt-4o-mini) that RAGAS uses to judge answer quality. The `LangchainEmbeddingsWrapper` wraps embeddings (text-embedding-3-small) for measuring semantic similarity between answers and contexts.

### Exercise 1.3: Run Evaluation (5 minutes)

**Task:** Run complete evaluation and analyze results.

**🖥️ Run in Terminal:**
```bash
# Create evaluation runner
cat > scripts/run_evaluation.py << 'EOF'
import json
import sys
sys.path.append('src')

from evaluation.ragas_evaluator import RAGASEvaluator
from documind.rag.production_qa import ProductionQA

# Load test dataset
with open('data/evaluation_dataset.json', 'r') as f:
    test_dataset = json.load(f)

# Initialize ProductionQA (the actual DocuMind RAG pipeline)
rag = ProductionQA(enable_logging=False)
evaluator = RAGASEvaluator(rag)

# Run evaluation
results = evaluator.evaluate(test_dataset)

# Display results
print("\n" + "="*60)
print("EVALUATION RESULTS")
print("="*60)

for metric in ['faithfulness', 'answer_relevancy', 'answer_correctness']:
    score = results.get(metric, 0.0)
    status = "✅" if score >= {'faithfulness': 0.70, 'answer_relevancy': 0.80,
                               'answer_correctness': 0.70}[metric] else "❌"
    print(f"{metric:.<25} {score:.3f} {status}")

print(f"\nAverage Score: {results['average_score']:.3f}")
print(f"Overall Status: {'✅ PASSED' if results['passed'] else '❌ FAILED'}")

# Save results
with open('results/evaluation_results.json', 'w') as f:
    json.dump(results, f, indent=2)
print("\n📁 Results saved to results/evaluation_results.json")
EOF

# Run it
python scripts/run_evaluation.py
```

**Expected Output:**
```
============================================================
RAGAS EVALUATION STARTED: 2025-01-15 10:30:45
============================================================
Test cases: 8
[1/8] How many vacation days do full-time employees...
[2/8] How do I request time off?...
...
[8/8] How many vacation days can be carried over...

Running RAGAS metrics...

============================================================
EVALUATION RESULTS
============================================================
faithfulness............. 0.938 ✅
answer_relevancy......... 0.811 ✅
answer_correctness....... 0.818 ✅

Average Score: 0.855
Overall Status: ✅ PASSED

📁 Results saved to results/evaluation_results.json
```

### Module 1 Quiz

**Question 1:** What does RAGAS faithfulness metric measure?
- A) How fast the RAG system responds
- B) Whether the answer is grounded in retrieved context
- C) The quality of the question
- D) The size of the knowledge base

**Question 2:** What is the target threshold for answer relevancy?
- A) ≥0.70
- B) ≥0.80
- C) ≥0.85
- D) ≥0.90

**Question 3:** Which metric measures if the answer is factually correct compared to the expected answer?
- A) Faithfulness
- B) Answer Relevancy
- C) Answer Correctness
- D) Context Precision

---

## Module 2: TruLens Monitoring Integration (15 minutes)

### Concept Review

**What is TruLens?**

TruLens is an open-source observability platform specifically designed for LLM applications. While RAGAS is excellent for batch evaluation (running after the fact), TruLens provides **real-time monitoring** of production queries as they happen.

**The Key Difference: Batch vs Real-Time**

| Approach | Tool | When It Runs | Best For |
|----------|------|--------------|----------|
| **Batch Evaluation** | RAGAS | Scheduled (daily, weekly) | Comprehensive testing, prompt optimization |
| **Real-Time Monitoring** | TruLens | Every production query | Catching issues immediately, user-level debugging |

**TruLens Three Core Feedback Functions:**

1. **Groundedness** - Is the answer supported by the retrieved context? Catches hallucinations in real-time.

2. **Answer Relevance** - Does the response actually address what the user asked? Detects off-topic answers.

3. **Context Relevance** - Did we retrieve content that's actually useful for answering? Identifies retrieval failures.

**Why Asynchronous Execution Matters:**

TruLens feedback functions run **after** the response is sent to the user, not during. This means:
- Zero latency impact on user experience
- Users get immediate responses
- Quality scores update in the background
- Dashboard refreshes to show latest metrics

**Key Insight:** Use RAGAS for systematic testing before deployment, TruLens for continuous monitoring after deployment. Together, they provide complete RAG observability.

---

### Learning Objectives
- Integrate TruLens with DocuMind
- Define custom feedback functions
- Launch monitoring dashboard
- Analyze real-time metrics

### Exercise 2.1: TruLens Wrapper (7 minutes)

**Task:** Create TruLens wrapper with feedback functions.

**📝 Create file `src/evaluation/trulens_monitor.py` using Claude Code or your editor:**
```python
# src/evaluation/trulens_monitor.py
import os
from trulens.core import TruSession, Feedback
from trulens.apps.custom import TruCustomApp, instrument
from trulens.providers.openai import OpenAI as OpenAIProvider
from dotenv import load_dotenv

load_dotenv()


class InstrumentedRAG:
    """Wrapper that instruments RAG pipeline methods for TruLens"""

    def __init__(self, rag_pipeline):
        self._rag = rag_pipeline

    @instrument
    def query(self, question: str) -> dict:
        """Instrumented query method for TruLens recording"""
        return self._rag.query(question)

    @instrument
    def retrieve(self, question: str) -> list:
        """Retrieve relevant documents"""
        result = self._rag.query(question)
        return result.get('sources', [])

    @instrument
    def generate(self, question: str, context: list) -> str:
        """Generate answer from context"""
        result = self._rag.query(question)
        return result.get('answer', '')


class DocuMindTruLens:
    """TruLens monitoring for DocuMind RAG (v2.x compatible)"""

    def __init__(self, rag_pipeline):
        # Wrap RAG pipeline with instrumented methods
        self.rag = InstrumentedRAG(rag_pipeline)

        # Initialize TruLens session
        self.session = TruSession()

        # Initialize OpenAI provider for feedback functions
        self.provider = OpenAIProvider(
            api_key=os.getenv("OPENAI_API_KEY"),
            model_engine="gpt-4o-mini"
        )

        # Define feedback functions
        self.feedbacks = [
            # Answer relevance - Is the answer relevant to the question?
            Feedback(self.provider.relevance_with_cot_reasons, name="answer_relevance")
            .on_input()
            .on_output(),

            # Groundedness - Is answer grounded in context?
            Feedback(self.provider.groundedness_measure_with_cot_reasons, name="groundedness")
            .on_input()
            .on_output()
        ]

        # Create TruCustomApp recorder
        self.tru_app = TruCustomApp(
            self.rag,
            app_name="DocuMind_RAG",
            app_version="production",
            feedbacks=self.feedbacks
        )

        print("✅ TruLens monitoring initialized")
        print("📊 Dashboard: run `trulens-eval leaderboard` or use launch_dashboard()")

    def query(self, question: str):
        """Run monitored query with TruLens recording"""
        with self.tru_app as recording:
            result = self.rag.query(question)
        return result

    def get_records(self):
        """Get all recorded sessions"""
        records, feedback = self.session.get_records_and_feedback()
        return records

    def launch_dashboard(self, port=8501):
        """Launch TruLens Streamlit dashboard"""
        from trulens.dashboard import run_dashboard
        run_dashboard(self.session, port=port)
```

### Exercise 2.2: Run Monitored Queries (5 minutes)

**Task:** Run queries with TruLens monitoring.

**🖥️ Run in Terminal:**
```bash
# Create monitoring demo
cat > scripts/monitor_queries.py << 'EOF'
# scripts/monitor_queries.py
import sys
sys.path.insert(0, 'src')

from evaluation.trulens_monitor import DocuMindTruLens
from documind.rag.production_qa import ProductionQA

# Initialize RAG pipeline and TruLens monitor
rag = ProductionQA()
monitor = DocuMindTruLens(rag)

# Sample queries based on actual knowledge base content
queries = [
    "How many vacation days do full-time employees receive?",
    "How do I request time off?",
    "What are the password requirements?",
    "How do I report a security incident?",
    "How many sick leave days do employees get?"
]

print("\n" + "="*60)
print("RUNNING MONITORED QUERIES")
print("="*60)

for i, query in enumerate(queries, 1):
    print(f"\n[{i}/{len(queries)}] {query}")
    result = monitor.query(query)
    print(f"✓ Answer length: {len(result['answer'])} chars")
    print(f"✓ Sources: {len(result.get('sources', []))} chunks")

print("\n" + "="*60)
print("✅ All queries monitored and logged")
print("📊 View dashboard: python -c \"from trulens_eval import Tru; Tru().run_dashboard()\"")
print("="*60)
EOF

python scripts/monitor_queries.py
```

**Expected Output:**
```
🦑 Initialized with db url sqlite:///default.sqlite .
🛑 Secret keys may be written to the database. See the `database_redact_keys` option of `TruSession` to prevent this.
✅ experimental otel_tracing enabled.
🔒 experimental otel_tracing is enabled and cannot be changed.
✅ TruLens monitoring initialized
📊 Dashboard: trulens-eval leaderboard

============================================================
RUNNING MONITORED QUERIES
============================================================

[1/5] How many vacation days do full-time employees receive?
✓ Answer length: 360 chars
✓ Sources: 4 chunks

[2/5] How do I request time off?
✓ Answer length: 1023 chars
✓ Sources: 3 chunks

[3/5] What are the password requirements?
✓ Answer length: 260 chars
✓ Sources: 1 chunks

[4/5] How do I report a security incident?
✓ Answer length: 267 chars
✓ Sources: 1 chunks

[5/5] How many sick leave days do employees get?
✓ Answer length: 193 chars
✓ Sources: 3 chunks

============================================================
✅ All queries monitored and logged
📊 View dashboard: python -c "from trulens_eval import Tru; Tru().run_dashboard()"
============================================================
```

### Exercise 2.3: Explore Dashboard (3 minutes)

**Task:** Launch TruLens dashboard and explore features.

**🖥️ Run in Terminal:**
```bash
# Launch dashboard (TruLens 2.x)
python -c "from trulens.dashboard import run_dashboard; from trulens.core import TruSession; run_dashboard(TruSession())"
```

> **Important:** When the dashboard opens, click the **"Refresh Data"** button in the top-right corner to load the recorded queries. Feedback functions run asynchronously, so you may need to refresh again after a few seconds to see updated scores.

**Dashboard Exploration Checklist:**
- [ ] Click "Refresh Data" to load records
- [ ] View app leaderboard (should show `DocuMind_RAG`)
- [ ] Check average groundedness score
- [ ] Explore individual query traces
- [ ] Review feedback function outputs
- [ ] Compare metrics across queries

### Module 2 Quiz

**Question 1:** What runs TruLens feedback functions?
- A) Synchronously during the query
- B) Asynchronously after the response
- C) Only when requested manually
- D) In a separate database

**Question 2:** What does TruLens groundedness measure?
- A) Whether the system is grounded to earth
- B) Whether the answer is supported by context
- C) The speed of response
- D) The database connection

**Question 3:** How do you launch the TruLens dashboard?
- A) `streamlit run trulens`
- B) `trulens dashboard`
- C) `run_dashboard(TruSession())`
- D) `python dashboard.py`

---

## Module 3: A/B Testing & Quality Gates (15 minutes)

### Concept Review

**What is A/B Testing for RAG?**

A/B testing lets you compare different RAG configurations using the same test dataset. This enables data-driven decisions about:
- **Model selection** - GPT-4o vs Claude vs local models
- **Prompt variations** - Which prompt template performs better?
- **Retrieval strategies** - Vector-only vs hybrid search
- **Chunk sizes** - 512 vs 1024 tokens

**A/B Testing Metrics:**

| Metric | What It Measures | Why It Matters |
|--------|------------------|----------------|
| **Quality Scores** | RAGAS metrics | Accuracy and relevance |
| **Latency** | Response time | User experience |
| **Cost** | API usage | Budget constraints |
| **Token Usage** | Input/output tokens | Efficiency |

**The Scientific Approach:**
1. Keep the test dataset **constant** across all variants
2. Run evaluations under **similar conditions** (same time, same load)
3. Compare **multiple metrics** (quality might increase but cost might too)
4. Make decisions based on **statistical significance**, not single runs

**What are Quality Gates?**

Quality gates are automated checkpoints that **block deployments** when quality metrics fall below defined thresholds. They're the safety net that prevents bad code from reaching production.

**Quality Gate Flow:**
```
Code Change → Run Evaluation → Check Thresholds → Pass? → Deploy
                                      ↓
                                   Fail? → Block Deployment + Alert Team
```

**Why Quality Gates Matter:**

Without quality gates, a developer could accidentally:
- Deploy a prompt change that increases hallucinations
- Update a model that's slower and less accurate
- Modify retrieval logic that breaks context quality

Quality gates catch these issues **before users are affected**.

**CI/CD Integration:**
- Quality gates return exit code 0 (pass) or 1 (fail)
- CI/CD systems interpret exit code 1 as build failure
- Failed builds block PR merges and deployments automatically

---

### Learning Objectives
- Implement model comparison framework
- Create quality gate checker
- Integrate with CI/CD
- Interpret A/B test results

### Exercise 3.1: A/B Testing Framework (7 minutes)

**Task:** Build model comparison system.

**📝 Create file `src/evaluation/ab_testing.py` using Claude Code or your editor:**
```python
# src/evaluation/ab_testing.py
from typing import List, Dict
from datetime import datetime
import time


class ABTester:
    """A/B testing framework for RAG model comparison"""

    def __init__(self):
        self.models = {}
        self.results = {}

    def add_model(self, name: str, rag_pipeline):
        """Add model to comparison"""
        self.models[name] = rag_pipeline
        print(f"➕ Added model: {name}")

    def run_comparison(self, test_dataset: List[Dict]) -> List[Dict]:
        """Compare all models on same dataset"""
        from evaluation.ragas_evaluator import RAGASEvaluator

        print(f"\n{'='*60}")
        print(f"A/B TESTING STARTED: {datetime.now()}")
        print(f"{'='*60}")
        print(f"Models: {len(self.models)}")
        print(f"Test queries: {len(test_dataset)}")

        results = []

        for model_name, rag_pipeline in self.models.items():
            print(f"\n{'─'*60}")
            print(f"Testing: {model_name}")
            print(f"{'─'*60}")

            # Track timing
            start_time = time.time()

            # Create evaluator and run
            evaluator = RAGASEvaluator(rag_pipeline)
            eval_results = evaluator.evaluate(test_dataset)

            elapsed_time = time.time() - start_time

            # Store results
            result = {
                "model": model_name,
                "faithfulness": eval_results.get("faithfulness", 0),
                "answer_relevancy": eval_results.get("answer_relevancy", 0),
                "answer_correctness": eval_results.get("answer_correctness", 0),
                "average_score": eval_results.get("average_score", 0),
                "passed": eval_results.get("passed", False),
                "time_seconds": round(elapsed_time, 2),
                "timestamp": datetime.now().isoformat()
            }
            results.append(result)

        # Sort by average score (descending)
        results.sort(key=lambda x: x["average_score"], reverse=True)

        return results

    def print_comparison(self, results: List[Dict]):
        """Print formatted comparison table"""
        print(f"\n{'='*80}")
        print("A/B TEST RESULTS - MODEL COMPARISON")
        print(f"{'='*80}")

        # Header
        print(f"{'Model':<25} {'Faith':>8} {'Relev':>8} {'Correct':>8} {'Avg':>8} {'Time':>8} {'Status':>8}")
        print(f"{'-'*80}")

        # Results rows
        for r in results:
            status = "✅ PASS" if r["passed"] else "❌ FAIL"
            print(
                f"{r['model']:<25} "
                f"{r['faithfulness']:>8.3f} "
                f"{r['answer_relevancy']:>8.3f} "
                f"{r['answer_correctness']:>8.3f} "
                f"{r['average_score']:>8.3f} "
                f"{r['time_seconds']:>7.1f}s "
                f"{status:>8}"
            )

        print(f"{'='*80}")

        # Winner announcement
        if results:
            winner = results[0]
            print(f"\n🏆 WINNER: {winner['model']} (avg score: {winner['average_score']:.3f})")
```

**🖥️ Run in Terminal:**
```bash
# Create A/B test script
cat > scripts/ab_test.py << 'EOF'
# scripts/ab_test.py
import json
import sys
sys.path.insert(0, 'src')

from evaluation.ab_testing import ABTester
from documind.rag.production_qa import ProductionQA

# Load dataset
with open('data/evaluation_dataset.json', 'r') as f:
    test_dataset = json.load(f)[:5]  # Use 5 queries for speed

# Setup comparison
tester = ABTester()

# Add current production model
rag = ProductionQA()
tester.add_model("GPT-4o-mini (Production)", rag)

# Note: To compare multiple models, you would create different
# ProductionQA instances with different configurations:
# rag_gpt4 = ProductionQA(model="gpt-4-turbo")
# tester.add_model("GPT-4 Turbo", rag_gpt4)

# Run comparison
results = tester.run_comparison(test_dataset)
tester.print_comparison(results)

# Save results
with open('results/ab_test_results.json', 'w') as f:
    json.dump(results, f, indent=2)

print(f"\n📁 Results saved to results/ab_test_results.json")
EOF

python scripts/ab_test.py
```

### Exercise 3.2: Quality Gates (5 minutes)

**Task:** Implement automated quality gates.

**🖥️ Run in Terminal:**
```bash
# Create quality gate script
cat > src/evaluation/quality_gate.py << 'EOF'
# src/evaluation/quality_gate.py
from typing import Dict
import sys


class QualityGate:
    """Quality gate checker for CI/CD"""

    THRESHOLDS = {
        'faithfulness': 0.70,
        'answer_relevancy': 0.80,
        'answer_correctness': 0.70
    }

    def check(self, results: Dict) -> bool:
        """Check if results pass quality gates"""
        print("\n" + "="*60)
        print("🚦 QUALITY GATE CHECK")
        print("="*60)

        all_passed = True

        for metric, threshold in self.THRESHOLDS.items():
            actual = results.get(metric, 0)
            passed = actual >= threshold
            status = "✅ PASS" if passed else "❌ FAIL"

            print(f"{metric:.<25} {actual:.3f} (≥{threshold:.2f}) {status}")

            if not passed:
                all_passed = False
                gap = threshold - actual
                print(f"  ⚠️ Below threshold by {gap:.3f}")

        print("="*60)

        if all_passed:
            print("✅ QUALITY GATE PASSED - Deployment allowed")
        else:
            print("❌ QUALITY GATE FAILED - Deployment blocked")

        return all_passed


# CLI usage
if __name__ == "__main__":
    import json

    with open('results/evaluation_results.json', 'r') as f:
        results = json.load(f)

    gate = QualityGate()
    passed = gate.check(results)

    sys.exit(0 if passed else 1)
EOF

echo "✅ Created quality_gate.py"

# Test with current results
python src/evaluation/quality_gate.py
echo "Exit code: $?"

# Test with failing results
cat > results/bad_results.json << 'EOF'
{
  "faithfulness": 0.56,
  "answer_relevancy": 0.71,
  "answer_correctness": 0.58
}
EOF

python -c "
import json
import sys
sys.path.append('src')
from evaluation.quality_gate import QualityGate

with open('results/bad_results.json', 'r') as f:
    results = json.load(f)

gate = QualityGate()
passed = gate.check(results)
sys.exit(0 if passed else 1)
"

echo "Exit code: $?"
```

**Expected Output:**
```
✅ Created quality_gate.py

============================================================
🚦 QUALITY GATE CHECK
============================================================
faithfulness............. 0.938 (≥0.70) ✅ PASS
answer_relevancy......... 0.811 (≥0.80) ✅ PASS
answer_correctness....... 0.818 (≥0.70) ✅ PASS
============================================================
✅ QUALITY GATE PASSED - Deployment allowed
Exit code: 0

============================================================
🚦 QUALITY GATE CHECK
============================================================
faithfulness............. 0.560 (≥0.70) ❌ FAIL
  ⚠️ Below threshold by 0.140
answer_relevancy......... 0.710 (≥0.80) ❌ FAIL
  ⚠️ Below threshold by 0.090
answer_correctness....... 0.580 (≥0.70) ❌ FAIL
  ⚠️ Below threshold by 0.120
============================================================
❌ QUALITY GATE FAILED - Deployment blocked
Exit code: 1
```

> **Note:** The first test uses your actual evaluation results (should pass). The second test uses intentionally bad scores to demonstrate how the quality gate blocks deployments when metrics fall below thresholds.

### Exercise 3.3: CI/CD Integration (3 minutes)

**Task:** Create GitHub Actions workflow for automated RAG evaluation.

> **💡 Workshop Note:** You may skip this exercise for the demo app, but make sure to include automated quality gates in your production RAG systems. CI/CD integration catches quality regressions before they reach users.

**🖥️ Run in Terminal:**
```bash
# Create GitHub Actions workflow directory and file
mkdir -p .github/workflows

cat > .github/workflows/evaluate.yml << 'EOF'
# .github/workflows/evaluate.yml
name: RAG Quality Evaluation

on:
  pull_request:
    branches: [main]
  push:
    branches: [main]
  schedule:
    - cron: '0 0 * * *'  # Daily at midnight UTC

jobs:
  evaluate:
    runs-on: ubuntu-latest

    steps:
      - uses: actions/checkout@v4

      - name: Set up Python
        uses: actions/setup-python@v5
        with:
          python-version: '3.11'

      - name: Install dependencies
        run: |
          pip install -r requirements.txt
          pip install ragas datasets langchain-openai

      - name: Run RAGAS evaluation
        env:
          OPENAI_API_KEY: ${{ secrets.OPENAI_API_KEY }}
          SUPABASE_URL: ${{ secrets.SUPABASE_URL }}
          SUPABASE_KEY: ${{ secrets.SUPABASE_KEY }}
        run: python scripts/run_evaluation.py

      - name: Check quality gates
        run: python src/evaluation/quality_gate.py

      - name: Upload results
        if: always()
        uses: actions/upload-artifact@v4
        with:
          name: evaluation-results
          path: results/
EOF

echo "✅ Created .github/workflows/evaluate.yml"
```

**What This Workflow Does:**

| Step | Description |
|------|-------------|
| **Triggers** | Runs on PRs to main, pushes to main, and daily at midnight |
| **Run evaluation** | Executes RAGAS evaluation against the test dataset |
| **Check quality gates** | Fails the build if metrics are below thresholds |
| **Upload results** | Saves evaluation results as downloadable artifacts |

**To Enable in Your Repository:**

1. **Add secrets** in GitHub → Settings → Secrets and variables → Actions:
   - `OPENAI_API_KEY` - Your OpenAI API key
   - `SUPABASE_URL` - Your Supabase project URL
   - `SUPABASE_KEY` - Your Supabase anon/service key

2. **Commit the workflow:**
   ```bash
   git add .github/workflows/evaluate.yml
   git commit -m "Add RAG evaluation CI/CD workflow"
   git push
   ```

3. **View results** in GitHub → Actions tab after pushing

> **Note:** The quality gate step will **block merging PRs** if evaluation scores fall below thresholds, ensuring only high-quality RAG updates reach production.

### Module 3 Quiz

**Question 1:** What is the purpose of quality gates?
- A) To make CI/CD slower
- B) To automatically block deployments that don't meet quality standards
- C) To generate reports
- D) To test network latency

**Question 2:** In A/B testing, what should you keep constant?
- A) The model being tested
- B) The test dataset
- C) The time of day
- D) The developer running the test

**Question 3:** What exit code should quality gate return on failure?
- A) 0
- B) 1
- C) -1
- D) 404

---

## Module 4: Production Deployment (15 minutes)

### Concept Review

**The Production Evaluation Architecture**

In production, RAGAS and TruLens work together in a complementary architecture:

```
┌─────────────────────────────────────────────────────────────┐
│                   RAG Production System                      │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  User Query → RAG Pipeline → Response                        │
│       │              │            │                          │
│       ▼              ▼            ▼                          │
│   TruLens       (Real-time)   TruLens                        │
│   Logging       Monitoring    Feedback                       │
│                                                              │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  Scheduled Jobs:                                             │
│  └─ RAGAS Batch Evaluation (nightly/weekly)                  │
│                                                              │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐   │
│  │  Evaluation  │ →  │  Dashboard   │ →  │   Alerts     │   │
│  │   Database   │    │  (Streamlit) │    │   (Email)    │   │
│  └──────────────┘    └──────────────┘    └──────────────┘   │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

**Three Pillars of Production Monitoring:**

1. **Evaluation Database** - Stores historical results for trend analysis
   - Track quality over time (is it improving or degrading?)
   - Compare model versions
   - Identify patterns (worse on certain question types?)

2. **Monitoring Dashboard** - Real-time visibility into system health
   - Current metric values with pass/fail indicators
   - Trend charts showing quality over time
   - Per-question breakdown for debugging

3. **Alerting System** - Proactive notification of issues
   - Alert when metrics drop below thresholds
   - Escalate based on severity (warning vs critical)
   - Include actionable context (what failed, what to check)

**Why Each Component Matters:**

| Component | Without It | With It |
|-----------|------------|---------|
| **Database** | No history, can't detect gradual degradation | Trend analysis, regression detection |
| **Dashboard** | Team can't see current state | Real-time visibility for everyone |
| **Alerts** | Problems discovered by angry users | Issues caught before user impact |

**Key Insight:** Production monitoring isn't optional - it's what separates a demo from a real system. Users will trust your RAG system when you can prove it's being monitored and maintained.

---

### Learning Objectives
- Set up evaluation database
- Create monitoring dashboard
- Implement alerting system
- Deploy complete evaluation system

### Exercise 4.1: Verify Results File (2 minutes)

**Task:** Confirm evaluation results are available for the dashboard.

> **Note:** For this workshop, we use JSON file storage for simplicity. In production, you'd use PostgreSQL with the schema shown in the "Production Database Setup" collapsible section below.

**🖥️ Run in Terminal:**
```bash
# Check that results file exists from our evaluation run
cat results/evaluation_results.json | python -m json.tool | head -20
```

**Expected Output:**
```json
{
    "faithfulness": 0.938,
    "answer_relevancy": 0.811,
    "answer_correctness": 0.818,
    "average_score": 0.855,
    "passed": true,
    ...
}
```

<details>
<summary>📦 Production Database Setup (Optional - Click to expand)</summary>

For production deployments, store evaluation results in PostgreSQL:

**🗄️ Run in SQL Editor (Supabase SQL Editor, pgAdmin, or psql):**
```sql
-- Create schema for evaluation tracking
CREATE TABLE IF NOT EXISTS evaluation_runs (
    id SERIAL PRIMARY KEY,
    run_name VARCHAR(255),
    model_name VARCHAR(100),
    faithfulness DECIMAL(5,4),
    answer_relevancy DECIMAL(5,4),
    answer_correctness DECIMAL(5,4),
    average_score DECIMAL(5,4),
    test_dataset_size INT,
    passed BOOLEAN,
    timestamp TIMESTAMP DEFAULT NOW()
);

CREATE INDEX idx_evaluation_timestamp
ON evaluation_runs(timestamp DESC);

CREATE INDEX idx_evaluation_passed
ON evaluation_runs(passed);
```

</details>

### Exercise 4.2: Monitoring Dashboard (5 minutes)

**Task:** Create Streamlit evaluation dashboard that visualizes results from our JSON file.

**🖥️ Run in Terminal to create the dashboard:**
```bash
cat > src/evaluation/evaluation_dashboard.py << 'EOF'
# src/evaluation/evaluation_dashboard.py
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import json
import os

st.set_page_config(
    page_title="DocuMind Evaluation Dashboard",
    page_icon="📊",
    layout="wide"
)

st.title("📊 DocuMind Evaluation Dashboard")

# Load results from JSON file
results_file = "results/evaluation_results.json"

if not os.path.exists(results_file):
    st.error(f"❌ No results found. Run `python scripts/run_evaluation.py` first.")
    st.stop()

with open(results_file, 'r') as f:
    results = json.load(f)

# Metrics overview
st.header("Latest Evaluation Results")

col1, col2, col3, col4 = st.columns(4)

faithfulness = results.get('faithfulness', 0)
answer_relevancy = results.get('answer_relevancy', 0)
answer_correctness = results.get('answer_correctness', 0)
average_score = results.get('average_score', 0)
passed = results.get('passed', False)

col1.metric(
    "Faithfulness",
    f"{faithfulness:.3f}",
    delta=f"{'✓' if faithfulness >= 0.70 else '✗'} Target: 0.70"
)
col2.metric(
    "Answer Relevancy",
    f"{answer_relevancy:.3f}",
    delta=f"{'✓' if answer_relevancy >= 0.80 else '✗'} Target: 0.80"
)
col3.metric(
    "Answer Correctness",
    f"{answer_correctness:.3f}",
    delta=f"{'✓' if answer_correctness >= 0.70 else '✗'} Target: 0.70"
)
col4.metric(
    "Average Score",
    f"{average_score:.3f}",
    delta="Overall"
)

# Status indicator
st.divider()
if passed:
    st.success("✅ Quality Gates: PASSING - Deployment allowed")
else:
    st.error("❌ Quality Gates: FAILING - Deployment blocked")

# Gauge chart for overall score
st.header("Overall Quality Score")

fig = go.Figure(go.Indicator(
    mode="gauge+number+delta",
    value=average_score * 100,
    domain={'x': [0, 1], 'y': [0, 1]},
    title={'text': "Average Score (%)"},
    delta={'reference': 75, 'increasing': {'color': "green"}},
    gauge={
        'axis': {'range': [0, 100]},
        'bar': {'color': "darkblue"},
        'steps': [
            {'range': [0, 50], 'color': "red"},
            {'range': [50, 70], 'color': "orange"},
            {'range': [70, 85], 'color': "yellow"},
            {'range': [85, 100], 'color': "green"}
        ],
        'threshold': {
            'line': {'color': "black", 'width': 4},
            'thickness': 0.75,
            'value': 75
        }
    }
))
fig.update_layout(height=300)
st.plotly_chart(fig, use_container_width=True)

# Bar chart comparing metrics
st.header("Metrics Comparison")

metrics_df = pd.DataFrame({
    'Metric': ['Faithfulness', 'Answer Relevancy', 'Answer Correctness'],
    'Score': [faithfulness, answer_relevancy, answer_correctness],
    'Threshold': [0.70, 0.80, 0.70]
})

fig2 = go.Figure()
fig2.add_trace(go.Bar(
    name='Score',
    x=metrics_df['Metric'],
    y=metrics_df['Score'],
    marker_color=['green' if s >= t else 'red' for s, t in zip(metrics_df['Score'], metrics_df['Threshold'])]
))
fig2.add_trace(go.Scatter(
    name='Threshold',
    x=metrics_df['Metric'],
    y=metrics_df['Threshold'],
    mode='markers',
    marker=dict(size=15, symbol='line-ew', line=dict(width=3, color='black'))
))
fig2.update_layout(
    yaxis_range=[0, 1],
    yaxis_title="Score",
    showlegend=True
)
st.plotly_chart(fig2, use_container_width=True)

# Per-question breakdown if available
if 'per_question' in results and results['per_question']:
    st.header("Per-Question Analysis")

    per_q_df = pd.DataFrame(results['per_question'])

    # Show table
    display_cols = ['question', 'faithfulness', 'answer_relevancy', 'answer_correctness']
    available_cols = [c for c in display_cols if c in per_q_df.columns]

    if available_cols:
        st.dataframe(per_q_df[available_cols], use_container_width=True)

# Footer
st.divider()
st.caption("📁 Results loaded from: " + results_file)
st.caption("Run `python scripts/run_evaluation.py` to generate new results")
EOF

echo "✅ Dashboard created at src/evaluation/evaluation_dashboard.py"
```

**Launch dashboard:**

**🖥️ Run in Terminal:**
```bash
streamlit run src/evaluation/evaluation_dashboard.py
```

**Expected Output:**
- Browser opens automatically to `http://localhost:8501`
- Dashboard shows metrics with gauge chart, bar chart, and per-question analysis
- Quality gate status displayed (PASSING/FAILING)

### Exercise 4.3: Alerting System (5 minutes)

**Task:** Implement quality alerting.

> **💡 Workshop Note:** For this demo application, you may skip implementing alerting. However, for production systems, quality alerts are essential for catching RAG degradation before it impacts users. Adapt the example below to your team's preferred notification channel (Slack, PagerDuty, Microsoft Teams, email, etc.).

**📝 Create file `src/evaluation/alerting.py` using Claude Code or your editor:**
```python
# src/evaluation/alerting.py
from typing import List, Dict
import smtplib
from email.mime.text import MIMEText
from datetime import datetime

class QualityAlerter:
    """Alert on quality issues"""
    
    def __init__(self, recipients: List[str], smtp_config: Dict = None):
        self.recipients = recipients
        self.smtp_config = smtp_config or {}
        self.alert_history = []
    
    def check_and_alert(self, results: Dict, run_name: str):
        """Check results and send alerts if needed"""
        issues = []
        
        # Check each metric
        thresholds = {
            'faithfulness': 0.70,
            'answer_relevancy': 0.80,
            'answer_correctness': 0.70
        }
        
        for metric, threshold in thresholds.items():
            actual = results.get(metric, 0)
            if actual < threshold:
                gap = threshold - actual
                issues.append(f"{metric}: {actual:.3f} (need {threshold:.2f}, gap: {gap:.3f})")
        
        if issues:
            self.send_alert(
                severity="HIGH",
                run_name=run_name,
                issues=issues,
                results=results
            )
    
    def send_alert(self, severity: str, run_name: str, 
                   issues: List[str], results: Dict):
        """Send alert email"""
        subject = f"[{severity}] DocuMind Quality Alert - {run_name}"
        
        body = f"""
DocuMind Evaluation Alert
========================

Run: {run_name}
Time: {datetime.now().isoformat()}
Severity: {severity}

Quality Issues Detected:
{chr(10).join('- ' + issue for issue in issues)}

Full Results:
- Faithfulness: {results.get('faithfulness', 0):.3f}
- Answer Relevancy: {results.get('answer_relevancy', 0):.3f}
- Answer Correctness: {results.get('answer_correctness', 0):.3f}

Action Required:
1. Review evaluation dashboard
2. Check recent code changes
3. Analyze failing test cases
4. Update prompts or configuration

Dashboard: http://localhost:8501
        """
        
        print(f"\n🚨 ALERT SENT")
        print(f"Subject: {subject}")
        print(f"Recipients: {', '.join(self.recipients)}")
        print(body)
        
        # In production, send actual email:
        # msg = MIMEText(body)
        # msg['Subject'] = subject
        # msg['From'] = self.smtp_config['from']
        # msg['To'] = ', '.join(self.recipients)
        # 
        # with smtplib.SMTP(self.smtp_config['server']) as server:
        #     server.send_message(msg)
```

**Test alerting:**

**🖥️ Run in Terminal:**
```bash
# Test with bad results
python -c "
import sys
sys.path.append('src')
from evaluation.alerting import QualityAlerter

bad_results = {
    'faithfulness': 0.56,
    'answer_relevancy': 0.71,
    'answer_correctness': 0.58
}

alerter = QualityAlerter(recipients=['team@company.com'])
alerter.check_and_alert(bad_results, 'test_run_20250115')
"
```

### Module 4 Quiz

**Question 1:** Why store evaluation results in a database?
- A) To waste storage space
- B) To track trends and regression over time
- C) To slow down evaluation
- D) To make it harder to access

**Question 2:** What should trigger a quality alert?
- A) Every evaluation run
- B) Only when all metrics are perfect
- C) When metrics fall below thresholds
- D) Never, alerts are annoying

**Question 3:** What's the benefit of a dashboard?
- A) Makes the project look professional
- B) Real-time visibility into system quality
- C) Required by management
- D) Replaces testing

---

---

## Module 5: The Unified Interface (15 minutes)

### Concept Review

**The Complete DocuMind System**

Throughout this course, we've built individual components that together form a production-ready RAG system:

| Session | Component | Purpose |
|---------|-----------|---------|
| **S5/S7** | Ingestion Agents | Process and chunk documents |
| **S8** | Vector Search | Semantic retrieval with hybrid search |
| **S9** | Memory & Feedback | User feedback and conversation memory |
| **S10** | Evaluation | RAGAS metrics and TruLens monitoring |

**The Problem: Separate Scripts**

Currently, each component exists as standalone code:
- `processor.py` - Document ingestion
- `search.py` - Vector search
- `production_qa.py` - RAG pipeline
- `ragas_evaluator.py` - Evaluation

Users need a **unified interface** to interact with the complete system.

**The Solution: Three Operational Modes**

A production RAG application typically needs three user-facing modes:

1. **Chat Mode** - The primary interface
   - Users ask questions in natural language
   - RAG pipeline retrieves context and generates answers
   - Sources are displayed for transparency
   - Feedback buttons enable continuous improvement

2. **Ingest Mode** - For administrators/content managers
   - Upload new documents (PDF, DOCX, TXT)
   - Process, chunk, and embed content
   - Add to the knowledge base
   - Track ingestion progress and status

3. **Explore Mode** - For debugging and discovery
   - Search the knowledge base semantically
   - View raw chunks and similarity scores
   - Understand what's in the vector store
   - Debug retrieval issues

**Why Streamlit?**

Streamlit provides the fastest path from Python code to web application:
- No frontend JavaScript required
- Automatic UI components for common patterns
- Built-in session state for conversations
- Easy deployment options

**Key Insight:** A unified interface transforms scattered scripts into a professional application. Users interact with one system, not five separate tools.

### Exercise 5.1: Create the App Entry Point

**Task:** Create `src/app.py` that imports your modules and provides a UI.

**Step 1: Create the App File**

**🖥️ Run in Terminal:**
```bash
cat > src/app.py << 'EOF'
import streamlit as st
import os
import time
from typing import List

# Import components from previous sessions
try:
    from documind.rag.production_qa import ProductionQA
    from documind.rag.search import search_documents
    from documind.processor import DocumentProcessor
except ImportError as e:
    st.error(f"Modules not found. Make sure you are running from the project root.\nError: {e}")
    st.stop()

# Page Config
st.set_page_config(page_title="DocuMind", page_icon="🧠", layout="wide")

st.title("🧠 DocuMind: Enterprise Knowledge Base")

# Sidebar: App Mode Selection
mode = st.sidebar.radio("Select Mode", ["Chat Assistant", "Document Ingestion", "Knowledge Explorer"])

# Initialize Session State for Chat
if "messages" not in st.session_state:
    st.session_state.messages = []
if "qa_system" not in st.session_state:
    # Initialize the ProductionQA system
    st.session_state.qa_system = ProductionQA(enable_logging=False)

# --- MODE 1: CHAT ASSISTANT ---
if mode == "Chat Assistant":
    st.header("Chat with your Documents")

    # Display chat history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Chat Input
    if prompt := st.chat_input("Ask a question about your documents..."):
        # 1. User Message
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # 2. Assistant Response
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                # Call ProductionQA
                response = st.session_state.qa_system.query(prompt)

                answer_text = response['answer']
                sources = response.get('sources', [])

                # Display Answer
                st.markdown(answer_text)

                # Display Sources
                if sources:
                    with st.expander("📚 View Sources"):
                        for s in sources:
                            st.markdown(f"- **{s.get('title', 'Doc')}**: {s.get('preview', '')}...")

                # Feedback buttons (visual only - extend with your feedback system)
                col1, col2 = st.columns(2)
                with col1:
                    if st.button("👍 Helpful", key=f"up_{len(st.session_state.messages)}"):
                        st.toast("Thanks for your feedback!")
                with col2:
                    if st.button("👎 Not Helpful", key=f"down_{len(st.session_state.messages)}"):
                        st.toast("Thanks for your feedback!")

        # Add to history
        st.session_state.messages.append({"role": "assistant", "content": answer_text})

# --- MODE 2: DOCUMENT INGESTION ---
elif mode == "Document Ingestion":
    st.header("📥 Ingest New Documents")
    st.info("Upload documents to add them to the Knowledge Base (supports PDF, DOCX, TXT).")

    uploaded_files = st.file_uploader("Choose files", accept_multiple_files=True)

    if st.button("Process Documents") and uploaded_files:
        processor = DocumentProcessor()

        progress_bar = st.progress(0)
        status_text = st.empty()

        for i, uploaded_file in enumerate(uploaded_files):
            status_text.text(f"Processing {uploaded_file.name}...")

            # Save temp file
            temp_path = f"temp_{uploaded_file.name}"
            with open(temp_path, "wb") as f:
                f.write(uploaded_file.getbuffer())

            try:
                # Extract & Chunk
                result = processor.process_document(temp_path)

                # Upload to DB
                upload_status = processor.upload_to_documind(result)

                st.success(f"✅ {uploaded_file.name}: {result['metadata']['basic']['page_count']} pages processed.")
            except Exception as e:
                st.error(f"❌ Failed {uploaded_file.name}: {str(e)}")
            finally:
                if os.path.exists(temp_path):
                    os.remove(temp_path)

            progress_bar.progress((i + 1) / len(uploaded_files))

        status_text.text("Ingestion Complete!")

# --- MODE 3: KNOWLEDGE EXPLORER ---
elif mode == "Knowledge Explorer":
    st.header("🔎 Explore Knowledge Base")

    search_term = st.text_input("Search documents by keyword or semantic meaning")

    if search_term:
        # Use semantic search
        results = search_documents(search_term, top_k=10)

        st.subheader(f"Found {len(results)} chunks")
        for r in results:
            with st.container(border=True):
                st.markdown(f"**Document:** {r.get('document_name', 'Unknown')}")
                st.caption(f"Score: {r.get('similarity', 0.0):.4f}")
                st.text(r.get('content', '')[:300] + "...")

# Footer
st.divider()
st.caption("DocuMind v1.0 | Built with HeroForge Agentic Engineering")
EOF

echo "✅ App created at src/app.py"
```

**Step 2: Run Your Final App**

**🖥️ Run in Terminal:**
```bash
streamlit run src/app.py
```

**What you will see:**
- A sidebar to switch between "Chat Assistant", "Document Ingestion", and "Knowledge Explorer"
- A Chat Interface that uses your ProductionQA RAG system
- A Document Uploader for adding new documents
- A Knowledge Explorer for semantic search

This is your final Capstone delivery! Congratulations! 🚀

Now, if you're an over-achiever, keep rolling with the Challenge Project!

---

## Challenge Project: Complete Evaluation System (Bonus)

**Objective:** Integrate all modules into production-ready system.

**Requirements:**
1. ✅ RAGAS evaluation on 20+ test queries
2. ✅ TruLens monitoring on production endpoint
3. ✅ Quality gates blocking bad deployments
4. ✅ A/B test comparing 2+ models
5. ✅ Database storing historical results
6. ✅ Streamlit dashboard with trends
7. ✅ Alerting on quality drops
8. ✅ CI/CD workflow with automated evaluation
9. ✅ Documentation of thresholds and decisions
10. ✅ Regression test detecting >5% quality drop

**Evaluation Criteria:**
- Code quality and organization
- Completeness of implementation
- Quality of test dataset
- Dashboard usability
- Alert configuration
- CI/CD integration
- Documentation clarity

**Submission:**
- GitHub repository with complete code
- Screenshot of TruLens dashboard
- Screenshot of Streamlit dashboard
- README with setup instructions
- Sample evaluation report

---

## Workshop Wrap-Up

### What You Built Today:
1. ✅ Production RAGAS evaluation pipeline
2. ✅ TruLens real-time monitoring
3. ✅ A/B testing framework
4. ✅ Automated quality gates
5. ✅ Evaluation database
6. ✅ Monitoring dashboard
7. ✅ Alerting system
8. ✅ CI/CD integration

### Key Takeaways:
- RAGAS provides quantifiable RAG quality metrics
- TruLens enables production observability
- Quality gates prevent regressions
- A/B testing drives data-driven decisions
- Continuous evaluation maintains quality
- Dashboards make metrics accessible
- Alerts catch issues before users do

### Next Steps:
- Run daily evaluation on your RAG system
- Build comprehensive test datasets
- Set up monitoring dashboards
- Integrate quality gates in CI/CD
- A/B test different models and prompts
- Track quality trends over time
- Optimize based on evaluation insights

**Congratulations on completing Session 10 and the entire course! You now have production-grade RAG evaluation skills!**

---

## Regression Testing for Prompts (10 minutes)

### Concept: Preventing Prompt Regressions

**The Problem:**

You optimize a prompt for one use case... and break three others:

```python
# Before (worked for 10 questions):
prompt = "Answer based on context: {context}\n\nQuestion: {question}"

# After "optimization" (works for 1 question, breaks 9):
prompt = "You are an expert. Use this context: {context}. Now answer: {question}"
# Oops! Now it hallucinates, ignores sources, and adds fluff.
```

**The Solution: Regression Test Suite**

Build a test dataset of known good examples. Run it after every prompt change.

### Exercise: Build Prompt Regression Suite

**Step 1: Create Golden Dataset (5 mins)**

**📝 Create file `tests/prompt_regression_tests.json` using Claude Code or your editor:**
```json
[
  {
    "question": "How many vacation days do employees get?",
    "expected_keywords": ["15", "20", "25", "tenure", "years"],
    "expected_sources": ["Company Vacation Policy"],
    "should_not_contain": ["I don't know", "I cannot", "unclear"]
  },
  {
    "question": "What is the remote work policy?",
    "expected_keywords": ["days", "manager", "approval"],
    "expected_sources": ["Remote Work Guidelines"],
    "should_not_contain": []
  },
  {
    "question": "What is the company's stance on quantum computing?",
    "expected_keywords": ["don't have", "no information", "not found"],
    "expected_sources": [],
    "should_not_contain": ["quantum", "computing"]
  }
]
```

**Step 2: Create Regression Test Runner (5 mins)**

**🤖 Ask Claude Code to create `tests/test_prompt_regression.py`:**
```
Create tests/test_prompt_regression.py with:

- Load tests/prompt_regression_tests.json
- For each test case:
  - Run RAG pipeline
  - Check expected_keywords are present
  - Check should_not_contain are absent
  - Verify correct sources cited (if specified)
  - Assert answer length is reasonable (50-500 chars)
- Use pytest with parametrize for each test case
- Print detailed failure messages

Make it production-ready with clear error messages.
```

**Step 3: Run Baseline (establish current behavior):**

**🖥️ Run in Terminal:**
```bash
pytest tests/test_prompt_regression.py -v

# Save output as baseline
pytest tests/test_prompt_regression.py -v > tests/baseline_results.txt
```

**All tests should pass (GREEN).** This is your baseline.

### Exercise: Catch a Prompt Regression

**Step 1: Break the prompt (intentionally):**

**📝 Edit `src/rag/qa_system.py` using Claude Code or your editor:**
```python
# In src/rag/qa_system.py, change the prompt:
def _build_rag_prompt(self, question: str, context: str) -> str:
    # OLD (good):
    # "Answer based ONLY on the context provided."

    # NEW (bad - encourages hallucination):
    prompt = f"""You are a helpful AI assistant. Use your knowledge and this context.

Context: {context}

Question: {question}

Provide a comprehensive answer using both your knowledge and the context."""
    return prompt
```

**Step 2: Run regression tests:**

**🖥️ Run in Terminal:**
```bash
pytest tests/test_prompt_regression.py -v
```

**Expected: Some tests FAIL (RED):**

```
tests/test_prompt_regression.py::test_question_3 FAILED

AssertionError: Regression detected!
Question: "What is the company's stance on quantum computing?"
Expected: ["don't have", "no information"]
Actual answer contained: "quantum computing is an emerging technology..."

❌ FAIL: AI hallucinated information not in context!
```

**Step 3: Revert the prompt:**

**🖥️ Run in Terminal:**
```bash
git checkout src/rag/qa_system.py
pytest tests/test_prompt_regression.py -v
# ✅ All tests pass again (GREEN)
```

### Workflow: Prompt Engineering with Safety

```
1. Establish baseline → Run regression tests (should pass)
2. Make prompt change → Modify _build_rag_prompt()
3. Run regression tests → pytest tests/test_prompt_regression.py
4. Check results:
   ✅ All pass → Accept change
   ❌ Some fail → Investigate:
      - Is failure expected? (Update test)
      - Is failure a regression? (Revert prompt)
      - Is tradeoff worth it? (Document decision)
5. Commit with test results:
   git commit -m "prompt: improve answer conciseness

   Changed prompt to encourage shorter answers.
   Regression tests: 15/15 passing ✅

   Closes #42"
```

### Adding to CI/CD

**📝 Update `.github/workflows/evaluate-rag.yml` using Claude Code or your editor:**
```yaml
- name: Run Prompt Regression Tests
  run: |
    echo "🧪 Running prompt regression tests..."
    pytest tests/test_prompt_regression.py -v
    # Fails workflow if any regression detected
```

**Now every PR checks for prompt regressions automatically!**

### Key Insights

1. **Prompts are code** - Version control them, test them, review them
2. **Regressions happen easily** - Small prompt changes have big impacts
3. **Golden datasets are valuable** - Invest time building good test cases
4. **CI/CD catches regressions** - Automate testing, don't rely on memory
5. **Document tradeoffs** - Sometimes regressions are acceptable

**This is how Anthropic and OpenAI manage prompts at scale.** 🎯

---

## Answer Key

### Module 1: RAGAS Evaluation

**Question 1:** B - Faithfulness measures whether the answer is supported by the retrieved context, detecting hallucinations.

**Question 2:** B - Answer relevancy should be ≥0.80 to ensure responses properly address the question.

**Question 3:** C - Answer Correctness compares the generated answer to the expected ground truth answer.

### Module 2: TruLens Monitoring

**Question 1:** B - Feedback functions run asynchronously so they don't add latency to user responses.

**Question 2:** B - Groundedness checks if the answer is supported by the retrieved context.

**Question 3:** C - Use `run_dashboard(TruSession())` to launch the Streamlit dashboard.

### Module 3: A/B Testing & Quality Gates

**Question 1:** B - Quality gates automatically prevent deploying code that doesn't meet minimum quality thresholds.

**Question 2:** B - Use the exact same test dataset for fair comparison between models.

**Question 3:** B - Exit code 1 indicates failure in CI/CD systems, blocking deployment.

### Module 4: Production Deployment

**Question 1:** B - Database storage enables tracking quality trends, detecting regressions, and historical analysis.

**Question 2:** C - Alerts should trigger when quality metrics drop below defined thresholds.

**Question 3:** B - Dashboards provide real-time visibility and make quality metrics accessible to the whole team.

---

## Resources

**Documentation:**
- RAGAS: https://docs.ragas.io/
- TruLens: https://www.trulens.org/
- Claude API: https://docs.anthropic.com/

**Code Examples:**
- https://github.com/your-repo/documind-evaluation
- https://github.com/explodinggradients/ragas/tree/main/examples

**Community:**
- Discord: [Course Discord]
- Office Hours: [Schedule]

**Certification:**
Submit your challenge project to earn your course completion certificate!
