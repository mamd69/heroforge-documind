# HeroForge.AI Course: AI-Powered Software Development
## Lesson 9 Workshop: Memory Architectures - Persistence, Context & Learning in AI Systems

**Estimated Time:** 45-60 minutes\
**Difficulty:** Intermediate-Advanced\
**Prerequisites:** Completed Sessions 1-8 (RAG system, Supabase integration, semantic search)

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

## Workshop Objectives

By completing this workshop, you will:
- [x] Understand different memory types for AI systems (short-term, long-term, working)
- [x] Implement conversation memory with context management
- [x] Build feedback collection and learning systems
- [x] Create personalization based on user history
- [x] Integrate memory with RAG for improved responses
- [x] Implement cross-session persistence and context restoration

---

### ⚠️ Supabase Client Type Warning

This session uses Supabase extensively for memory persistence. Ensure your client is set up correctly:

**Standard Pattern (used in this workshop):**
```python
from supabase import create_client, Client
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")
supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)
```

**Common Errors and Fixes:**

| Error | Cause | Fix |
|-------|-------|-----|
| `AttributeError: 'Client' object has no attribute 'from_'` | Old SDK version | Run `pip install --upgrade supabase` |
| `TypeError: create_client() missing 1 required positional argument` | Missing env vars | Verify `.env` file has both `SUPABASE_URL` and `SUPABASE_KEY` |
| `postgrest.exceptions.APIError: permission denied` | RLS policies blocking | Check table RLS policies in Supabase Dashboard |

**Verify Your Setup Before Starting:**
```bash
# Run in Terminal (VS Code terminal or command line)
# Quick connection test
python -c "
from supabase import create_client
import os
c = create_client(os.getenv('SUPABASE_URL'), os.getenv('SUPABASE_KEY'))
result = c.from_('conversations').select('id').limit(1).execute()
print('✅ Supabase connected and conversations table accessible!')
"
```
NOTE: If using free version, your Supabase project may be paused.  If so, restore in Supabase to continue.
---

## Module 1: Memory Types and Conversation Context (15 minutes)

### Concept Review

**What is Memory in AI Systems?**

Memory allows AI systems to maintain state across interactions, learn from experience, and provide personalized responses. Unlike humans with unified memory, AI systems use different memory types for different purposes.

**Memory Types:**

1. **Short-Term Memory (Conversation Context)**
   - Current conversation history
   - Limited to recent messages (context window)
   - Cleared when conversation ends
   - Used for: Follow-up questions, pronoun resolution

2. **Long-Term Memory (Persistent Storage)**
   - User preferences and history
   - Knowledge learned from feedback
   - Persists across sessions
   - Used for: Personalization, continuous improvement

3. **Working Memory (Task State)**
   - Intermediate results during task execution
   - Retrieved documents, partial answers
   - Temporary, task-specific
   - Used for: Multi-step reasoning, context assembly

4. **Episodic Memory (Experience Replay)**
   - Past successful/failed queries
   - User feedback patterns
   - Query-response pairs
   - Used for: Learning, optimization

**Why Memory Matters:**
- **Context**: "What was the vacation policy again?" (refers to previous question)
- **Learning**: User rates answer low → System adjusts retrieval
- **Personalization**: User asks mostly HR questions → Prioritize HR docs
- **Efficiency**: Remember user preferences, avoid repeating questions

---

### Exercise 1.1: Implement Conversation Memory

**Task:** Build a conversation memory system that tracks multi-turn conversations.

**Instructions:**

**Step 1: Create Database Schema (3 mins)**

```sql
-- Run in Supabase SQL Editor

-- Conversations table
CREATE TABLE conversations (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id VARCHAR(255),
    title TEXT,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

-- Messages table
CREATE TABLE messages (
    id BIGSERIAL PRIMARY KEY,
    conversation_id UUID REFERENCES conversations(id) ON DELETE CASCADE,
    role VARCHAR(50) NOT NULL, -- 'user' or 'assistant'
    content TEXT NOT NULL,
    sources JSONB, -- Retrieved documents for this message
    metadata JSONB, -- Model, response time, etc.
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- Indexes for performance
CREATE INDEX idx_conversations_user_id ON conversations(user_id);
CREATE INDEX idx_conversations_updated_at ON conversations(updated_at DESC);
CREATE INDEX idx_messages_conversation_id ON messages(conversation_id);
CREATE INDEX idx_messages_created_at ON messages(created_at);
```

**Step 2: Create Memory Module (7 mins)**

Create `src/documind/memory/__init__.py` **(in VS Code or use Claude Code to create the file)**:

```python
"""
Memory system for DocuMind
Handles conversation context and persistence
"""
```

Create `src/documind/memory/conversation.py` **(in VS Code or use Claude Code to create the file)**:

```python
"""
Conversation Memory Implementation
Tracks multi-turn conversations with context management
"""
import os
from typing import List, Dict, Any, Optional
from datetime import datetime
import json
from supabase import create_client, Client

# Configuration
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")

supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

class ConversationMemory:
    """Manages conversation history with context window management."""

    def __init__(self, conversation_id: Optional[str] = None, user_id: Optional[str] = None):
        """
        Initialize conversation memory.

        Args:
            conversation_id: Existing conversation ID, or None to create new
            user_id: User identifier for conversation tracking
        """
        self.conversation_id = conversation_id
        self.user_id = user_id or "anonymous"
        self.messages = []
        self.max_context_messages = 10  # Keep last 10 messages in context

        if conversation_id:
            self.load_history()
        else:
            self.create_conversation()

    def create_conversation(self, title: str = "New Conversation") -> str:
        """
        Create a new conversation in the database.

        Args:
            title: Conversation title

        Returns:
            Conversation ID
        """
        result = supabase.from_('conversations').insert({
            'user_id': self.user_id,
            'title': title
        }).execute()

        self.conversation_id = result.data[0]['id']
        return self.conversation_id

    def load_history(self) -> List[Dict[str, Any]]:
        """
        Load conversation history from database.

        Returns:
            List of message dictionaries
        """
        result = supabase.from_('messages') \
            .select('*') \
            .eq('conversation_id', self.conversation_id) \
            .order('created_at', desc=False) \
            .execute()

        self.messages = result.data
        return self.messages

    def add_message(
        self,
        role: str,
        content: str,
        sources: Optional[List[Dict]] = None,
        metadata: Optional[Dict] = None
    ) -> Dict[str, Any]:
        """
        Add a message to the conversation.

        Args:
            role: 'user' or 'assistant'
            content: Message content
            sources: Retrieved documents (for assistant messages)
            metadata: Additional metadata (model, response time, etc.)

        Returns:
            Created message dictionary
        """
        message_data = {
            'conversation_id': self.conversation_id,
            'role': role,
            'content': content,
            'sources': json.dumps(sources) if sources else None,
            'metadata': json.dumps(metadata) if metadata else None
        }

        result = supabase.from_('messages').insert(message_data).execute()

        message = result.data[0]
        self.messages.append(message)

        # Update conversation timestamp
        supabase.from_('conversations') \
            .update({'updated_at': datetime.now().isoformat()}) \
            .eq('id', self.conversation_id) \
            .execute()

        return message

    def get_context(self, max_messages: Optional[int] = None) -> List[Dict[str, str]]:
        """
        Get recent conversation context for the LLM.

        Args:
            max_messages: Maximum number of messages to include

        Returns:
            List of message dicts with 'role' and 'content'
        """
        max_msgs = max_messages or self.max_context_messages
        recent_messages = self.messages[-max_msgs:] if len(self.messages) > max_msgs else self.messages

        context = []
        for msg in recent_messages:
            context.append({
                'role': msg['role'],
                'content': msg['content']
            })

        return context

    def get_context_summary(self) -> str:
        """
        Get a text summary of the conversation context.

        Returns:
            Formatted string with conversation history
        """
        context = self.get_context()

        summary_parts = []
        for msg in context:
            role = msg['role'].upper()
            content = msg['content']
            summary_parts.append(f"{role}: {content}")

        return "\n\n".join(summary_parts)

    def clear_context(self) -> None:
        """Clear in-memory context (keeps database intact)."""
        self.messages = []

    def get_conversation_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the conversation.

        Returns:
            Dictionary with message counts, duration, etc.
        """
        if not self.messages:
            return {'message_count': 0}

        user_messages = [m for m in self.messages if m['role'] == 'user']
        assistant_messages = [m for m in self.messages if m['role'] == 'assistant']

        first_message = self.messages[0]['created_at']
        last_message = self.messages[-1]['created_at']

        return {
            'message_count': len(self.messages),
            'user_messages': len(user_messages),
            'assistant_messages': len(assistant_messages),
            'started_at': first_message,
            'last_activity': last_message,
            'conversation_id': self.conversation_id
        }

# Utility functions
def list_user_conversations(user_id: str, limit: int = 10) -> List[Dict[str, Any]]:
    """
    List all conversations for a user.

    Args:
        user_id: User identifier
        limit: Maximum number of conversations to return

    Returns:
        List of conversation dictionaries
    """
    result = supabase.from_('conversations') \
        .select('*') \
        .eq('user_id', user_id) \
        .order('updated_at', desc=True) \
        .limit(limit) \
        .execute()

    return result.data

def delete_conversation(conversation_id: str) -> None:
    """
    Delete a conversation and all its messages.

    Args:
        conversation_id: Conversation to delete
    """
    supabase.from_('conversations').delete().eq('id', conversation_id).execute()

# Test the conversation memory
if __name__ == "__main__":
    import sys

    # Create a test conversation
    print("Creating test conversation...")
    conv = ConversationMemory(user_id="test_user")

    # Add some messages
    conv.add_message("user", "What is our vacation policy?")
    conv.add_message(
        "assistant",
        "All full-time employees receive 15 days of paid vacation annually.",
        sources=[{'document': 'employee_handbook.pdf', 'similarity': 0.89}],
        metadata={'model': 'claude-3.5-sonnet', 'response_time': 1.2}
    )

    conv.add_message("user", "How do I request time off?")
    conv.add_message(
        "assistant",
        "Submit requests through the HR portal at least 2 weeks in advance.",
        sources=[{'document': 'hr_policies.md', 'similarity': 0.85}]
    )

    # Display context
    print("\n" + "="*60)
    print("Conversation Context:")
    print("="*60)
    print(conv.get_context_summary())

    print("\n" + "="*60)
    print("Conversation Stats:")
    print("="*60)
    stats = conv.get_conversation_stats()
    for key, value in stats.items():
        print(f"{key}: {value}")

    # List user's conversations
    print("\n" + "="*60)
    print("User's Conversations:")
    print("="*60)
    conversations = list_user_conversations("test_user")
    for c in conversations:
        print(f"- {c['id']}: {c['title']} (updated: {c['updated_at']})")
```

**Step 3: Test Conversation Memory (5 mins)**

```bash
# Run in Terminal (VS Code terminal or command line)
python src/documind/memory/conversation.py

# Expected output:
# Creating test conversation...
# ============================================================
# Conversation Context:
# ============================================================
# USER: What is our vacation policy?
#
# ASSISTANT: All full-time employees receive 15 days of paid vacation annually.
#
# USER: How do I request time off?
#
# ASSISTANT: Submit requests through the HR portal at least 2 weeks in advance.
# ============================================================
# Conversation Stats:
# ============================================================
# message_count: 4
# user_messages: 2
# assistant_messages: 2
# ...
```

---

### Quiz 1:

**Question 1:** What is the difference between short-term and long-term memory in AI systems?\
   a) Short-term memory is faster than long-term memory\
   b) Short-term memory uses more storage than long-term memory\
   c) There is no difference\
   d) Short-term memory holds current conversation context; long-term memory persists user preferences and history across sessions

**Question 2:** Why do we limit context to the last N messages (context window management)?\
   a) To save money on database storage\
   b) To make queries faster\
   c) To stay within the LLM's token limit and maintain relevant, recent context without overwhelming the model\
   d) Because older messages are automatically deleted

**Question 3:** What information should be stored with each assistant message?\
   a) Only the message text\
   b) The message content, retrieved sources, model used, and response metadata for analytics and debugging\
   c) Just the timestamp\
   d) The user's IP address

**Answers:**
1. **d)** Short-term = current conversation; long-term = persistent cross-session data
2. **c)** Context window management prevents token limit overflow and maintains relevance
3. **b)** Store content, sources, model, and metadata for full traceability

---

## Module 2: Feedback Systems and Data Collection (15 minutes)

### Concept Review

**Why Collect Feedback?**

User feedback is the foundation of learning systems. Without feedback, your AI system cannot improve beyond its initial design.

**Types of Feedback:**

1. **Explicit Feedback**
   - Star ratings (1-5)
   - Thumbs up/down
   - Text comments
   - Quality reports

2. **Implicit Feedback**
   - Click-through rate on sources
   - Time spent reading answer
   - Follow-up question patterns
   - Query reformulations (indicates poor answer)

**Feedback Metrics:**
- **Answer Quality**: Was the answer helpful? (1-5 stars)
- **Source Relevance**: Were the retrieved documents relevant?
- **Completeness**: Did the answer fully address the question?
- **Accuracy**: Was the information correct?

**Using Feedback:**
- Identify failing queries (low ratings)
- Improve retrieval (adjust similarity thresholds)
- Optimize prompts (better answer generation)
- Personalize results (learn user preferences)

---

### Exercise 2.1: Implement Feedback Collection

**Task:** Build a feedback system to collect and analyze user ratings.

**Instructions:**

**Step 1: Create Feedback Schema (3 mins)**

```sql
-- Run in Supabase SQL Editor

-- Feedback table
CREATE TABLE feedback (
    id BIGSERIAL PRIMARY KEY,
    conversation_id UUID REFERENCES conversations(id) ON DELETE CASCADE,
    message_id BIGINT REFERENCES messages(id) ON DELETE CASCADE,
    user_id VARCHAR(255),
    rating INTEGER CHECK (rating >= 1 AND rating <= 5),
    comment TEXT,
    feedback_type VARCHAR(50), -- 'answer_quality', 'source_relevance', etc.
    metadata JSONB, -- Additional context (query, retrieved sources, etc.)
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- Indexes
CREATE INDEX idx_feedback_conversation_id ON feedback(conversation_id);
CREATE INDEX idx_feedback_message_id ON feedback(message_id);
CREATE INDEX idx_feedback_rating ON feedback(rating);
CREATE INDEX idx_feedback_created_at ON feedback(created_at);
```

**Step 2: Create Feedback Module (7 mins)**

Create `src/documind/memory/feedback.py` **(in VS Code or use Claude Code to create the file)**:

```python
"""
Feedback Collection and Analysis
Learn from user ratings to improve the system
"""
import os
from typing import List, Dict, Any, Optional
from datetime import datetime, timedelta
import json
from supabase import create_client, Client

# Configuration
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")

supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

class FeedbackCollector:
    """Collects and analyzes user feedback."""

    def __init__(self, user_id: Optional[str] = None):
        self.user_id = user_id or "anonymous"

    def submit_feedback(
        self,
        conversation_id: str,
        message_id: int,
        rating: int,
        comment: Optional[str] = None,
        feedback_type: str = "answer_quality",
        metadata: Optional[Dict] = None
    ) -> Dict[str, Any]:
        """
        Submit user feedback for a message.

        Args:
            conversation_id: Conversation ID
            message_id: Message being rated
            rating: 1-5 star rating
            comment: Optional text feedback
            feedback_type: Type of feedback (answer_quality, source_relevance, etc.)
            metadata: Additional context (query, sources, etc.)

        Returns:
            Created feedback record
        """
        if not (1 <= rating <= 5):
            raise ValueError("Rating must be between 1 and 5")

        feedback_data = {
            'conversation_id': conversation_id,
            'message_id': message_id,
            'user_id': self.user_id,
            'rating': rating,
            'comment': comment,
            'feedback_type': feedback_type,
            'metadata': json.dumps(metadata) if metadata else None
        }

        result = supabase.from_('feedback').insert(feedback_data).execute()
        return result.data[0]

    def get_message_feedback(self, message_id: int) -> List[Dict[str, Any]]:
        """
        Get all feedback for a specific message.

        Args:
            message_id: Message ID

        Returns:
            List of feedback records
        """
        result = supabase.from_('feedback') \
            .select('*') \
            .eq('message_id', message_id) \
            .execute()

        return result.data

    def get_conversation_feedback(self, conversation_id: str) -> List[Dict[str, Any]]:
        """
        Get all feedback for a conversation.

        Args:
            conversation_id: Conversation ID

        Returns:
            List of feedback records
        """
        result = supabase.from_('feedback') \
            .select('*') \
            .eq('conversation_id', conversation_id) \
            .execute()

        return result.data

    def analyze_feedback(self, days: int = 7) -> Dict[str, Any]:
        """
        Analyze recent feedback to identify patterns.

        Args:
            days: Number of days to analyze

        Returns:
            Analysis results with statistics and insights
        """
        cutoff = datetime.now() - timedelta(days=days)

        # Fetch recent feedback
        result = supabase.from_('feedback') \
            .select('*') \
            .gte('created_at', cutoff.isoformat()) \
            .execute()

        feedback_items = result.data

        if not feedback_items:
            return {
                'message': 'No feedback in time period',
                'days': days
            }

        # Calculate statistics
        total_feedback = len(feedback_items)
        ratings = [f['rating'] for f in feedback_items]
        avg_rating = sum(ratings) / len(ratings)

        # Rating distribution
        rating_dist = {i: ratings.count(i) for i in range(1, 6)}

        # Low-rated items (need improvement)
        low_rated = [f for f in feedback_items if f['rating'] <= 2]

        # High-rated items (successful patterns)
        high_rated = [f for f in feedback_items if f['rating'] >= 4]

        # Comments analysis
        comments = [f['comment'] for f in feedback_items if f.get('comment')]

        return {
            'period_days': days,
            'total_feedback': total_feedback,
            'average_rating': round(avg_rating, 2),
            'rating_distribution': rating_dist,
            'low_rated_count': len(low_rated),
            'high_rated_count': len(high_rated),
            'comments_count': len(comments),
            'needs_improvement': len(low_rated) / total_feedback if total_feedback > 0 else 0,
            'satisfaction_rate': len(high_rated) / total_feedback if total_feedback > 0 else 0
        }

    def identify_failing_queries(self, min_rating: int = 2) -> List[Dict[str, Any]]:
        """
        Identify queries that received low ratings.

        Args:
            min_rating: Maximum rating to consider as "failing"

        Returns:
            List of low-rated queries with details
        """
        # Fetch low-rated feedback with message details
        result = supabase.from_('feedback') \
            .select('*, messages(content, sources)') \
            .lte('rating', min_rating) \
            .execute()

        failing_queries = []
        for feedback in result.data:
            if feedback.get('messages'):
                # Get the user query (previous message)
                message_id = feedback['message_id']
                msg_result = supabase.from_('messages') \
                    .select('*') \
                    .eq('id', message_id - 1) \
                    .single() \
                    .execute()

                if msg_result.data:
                    failing_queries.append({
                        'query': msg_result.data['content'],
                        'answer': feedback['messages']['content'],
                        'rating': feedback['rating'],
                        'comment': feedback.get('comment'),
                        'sources': feedback['messages'].get('sources'),
                        'feedback_id': feedback['id']
                    })

        return failing_queries

    def get_improvement_suggestions(self) -> Dict[str, List[str]]:
        """
        Generate improvement suggestions based on feedback patterns.

        Returns:
            Dictionary with categorized suggestions
        """
        analysis = self.analyze_feedback(days=30)
        failing = self.identify_failing_queries()

        suggestions = {
            'retrieval': [],
            'generation': [],
            'system': []
        }

        # Check satisfaction rate
        if analysis.get('satisfaction_rate', 0) < 0.7:
            suggestions['system'].append(
                f"Overall satisfaction is {analysis.get('satisfaction_rate', 0):.1%}. "
                "Consider reviewing retrieval and generation quality."
            )

        # Check for retrieval issues
        if len(failing) > 0:
            low_similarity_count = sum(
                1 for f in failing
                if f.get('sources') and any(
                    s.get('similarity', 0) < 0.75 for s in json.loads(f['sources'])
                )
            )

            if low_similarity_count > len(failing) * 0.5:
                suggestions['retrieval'].append(
                    f"{low_similarity_count} failing queries had low similarity scores. "
                    "Consider adjusting retrieval parameters or improving document embeddings."
                )

        # Check rating distribution
        rating_dist = analysis.get('rating_distribution', {})
        if rating_dist.get(1, 0) + rating_dist.get(2, 0) > analysis.get('total_feedback', 0) * 0.3:
            suggestions['generation'].append(
                "High number of low ratings. Review answer generation prompts and ensure "
                "answers are comprehensive and well-sourced."
            )

        return suggestions

# Test feedback system
if __name__ == "__main__":
    from conversation import ConversationMemory

    print("Testing Feedback System")
    print("="*60)

    # Create test conversation
    conv = ConversationMemory(user_id="test_user")
    msg1 = conv.add_message("user", "What is our vacation policy?")
    msg2 = conv.add_message(
        "assistant",
        "Full-time employees get 15 days of paid vacation annually.",
        sources=[{'document': 'handbook.pdf', 'similarity': 0.89}]
    )

    # Submit feedback
    collector = FeedbackCollector(user_id="test_user")

    feedback = collector.submit_feedback(
        conversation_id=conv.conversation_id,
        message_id=msg2['id'],
        rating=5,
        comment="Very helpful, exactly what I needed!",
        metadata={'query': msg1['content'], 'helpful': True}
    )

    print(f"\nFeedback submitted: {feedback['id']}")
    print(f"Rating: {feedback['rating']} stars")
    print(f"Comment: {feedback['comment']}")

    # Analyze feedback
    print("\n" + "="*60)
    print("Feedback Analysis (Last 7 Days)")
    print("="*60)

    analysis = collector.analyze_feedback(days=7)
    for key, value in analysis.items():
        print(f"{key}: {value}")

    # Get improvement suggestions
    print("\n" + "="*60)
    print("Improvement Suggestions")
    print("="*60)

    suggestions = collector.get_improvement_suggestions()
    for category, items in suggestions.items():
        if items:
            print(f"\n{category.upper()}:")
            for suggestion in items:
                print(f"  - {suggestion}")
```

**Step 3: Test Feedback System (5 mins)**

```bash
# Run in Terminal (VS Code terminal or command line)
python src/documind/memory/feedback.py
```

---

### Quiz 2:

**Question 1:** What is the difference between explicit and implicit feedback?\
   a) Explicit feedback is automated; implicit feedback is manual\
   b) Explicit feedback is always positive; implicit feedback is always negative\
   c) Explicit feedback is direct user input (ratings, comments); implicit feedback is inferred from user behavior (clicks, time spent)\
   d) There is no difference

**Question 2:** Why is it important to track feedback with message metadata (query, sources, model)?\
   a) To increase database size\
   b) To diagnose issues and understand why certain queries failed or succeeded, enabling targeted improvements\
   c) To make the system slower\
   d) To comply with regulations

**Question 3:** What should you do when multiple queries receive low ratings with similar patterns?\
   a) Delete the feedback\
   b) Ignore it\
   c) Disable the rating system\
   d) Investigate the common failure mode and adjust retrieval parameters, prompts, or document quality accordingly

**Answers:**
1. **c)** Explicit = direct input; implicit = inferred behavior
2. **b)** Metadata enables root cause analysis and targeted improvements
3. **d)** Patterns indicate systematic issues requiring investigation and adjustment

---

## Module 3: Learning Loops and System Improvement (15 minutes)

### Concept Review

**What is a Learning Loop?**

A learning loop is a continuous cycle where the system:
1. Collects feedback
2. Analyzes patterns
3. Adjusts parameters
4. Measures improvement
5. Repeats

**Learning Strategies:**

1. **Retrieval Optimization**
   - Adjust similarity thresholds based on feedback
   - Re-weight document importance
   - Improve query reformulation

2. **Prompt Engineering**
   - Refine generation prompts based on low-rated answers
   - Add examples from high-rated responses
   - Adjust tone and style preferences

3. **Personalization**
   - Track user query topics
   - Boost documents matching user interests
   - Adapt response style to user preferences

4. **Document Quality**
   - Identify frequently retrieved but low-rated docs
   - Improve document chunking strategies
   - Update outdated content

**Metrics to Track:**
- Average rating over time (trending up?)
- Response time (getting faster?)
- Retrieval accuracy (better sources?)
- User engagement (more queries?)

---

### Exercise 3.1: Implement Learning Algorithm

**Task:** Build a learning system that improves retrieval based on feedback.

**Instructions:**

**Step 1: Create Learning Module (10 mins)**

Create `src/documind/memory/learning.py` **(in VS Code or use Claude Code to create the file)**:

```python
"""
Learning System - Improve from Feedback
Continuously optimizes retrieval and generation based on user feedback
"""
import os
from typing import List, Dict, Any, Optional
from datetime import datetime, timedelta
import json
import statistics
from supabase import create_client, Client

from .feedback import FeedbackCollector

# Configuration
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")

supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

class LearningSystem:
    """Learns from feedback to improve system performance."""

    def __init__(self):
        self.feedback_collector = FeedbackCollector()
        self.learning_history = []

    def analyze_retrieval_quality(self, days: int = 7) -> Dict[str, Any]:
        """
        Analyze retrieval quality based on feedback.

        Args:
            days: Number of days to analyze

        Returns:
            Analysis of retrieval performance
        """
        cutoff = datetime.now() - timedelta(days=days)

        # Fetch feedback with source information
        result = supabase.from_('feedback') \
            .select('*, messages(sources, metadata)') \
            .gte('created_at', cutoff.isoformat()) \
            .execute()

        feedback_items = result.data

        if not feedback_items:
            return {'message': 'No data to analyze'}

        # Analyze similarity scores vs ratings
        similarity_ratings = []

        for item in feedback_items:
            if item.get('messages') and item['messages'].get('sources'):
                sources = json.loads(item['messages']['sources'])
                avg_similarity = statistics.mean([s.get('similarity', 0) for s in sources])
                similarity_ratings.append({
                    'similarity': avg_similarity,
                    'rating': item['rating']
                })

        if not similarity_ratings:
            return {'message': 'No similarity data available'}

        # Calculate correlation between similarity and rating
        high_sim_high_rating = sum(
            1 for sr in similarity_ratings
            if sr['similarity'] >= 0.8 and sr['rating'] >= 4
        )

        low_sim_low_rating = sum(
            1 for sr in similarity_ratings
            if sr['similarity'] < 0.7 and sr['rating'] <= 2
        )

        # Calculate optimal similarity threshold
        # Find threshold that maximizes high ratings
        thresholds = [0.6, 0.65, 0.7, 0.75, 0.8, 0.85]
        threshold_performance = {}

        for threshold in thresholds:
            filtered = [sr for sr in similarity_ratings if sr['similarity'] >= threshold]
            if filtered:
                avg_rating = statistics.mean([sr['rating'] for sr in filtered])
                threshold_performance[threshold] = avg_rating

        optimal_threshold = max(threshold_performance.items(), key=lambda x: x[1])[0]

        return {
            'total_analyzed': len(similarity_ratings),
            'high_similarity_high_rating': high_sim_high_rating,
            'low_similarity_low_rating': low_sim_low_rating,
            'optimal_threshold': optimal_threshold,
            'threshold_performance': threshold_performance,
            'recommendation': (
                f"Consider using similarity threshold of {optimal_threshold} "
                f"for best average rating of {threshold_performance[optimal_threshold]:.2f}"
            )
        }

    def learn_user_preferences(self, user_id: str, days: int = 30) -> Dict[str, Any]:
        """
        Learn user preferences from query and feedback history.

        Args:
            user_id: User to analyze
            days: Number of days to analyze

        Returns:
            User preference profile
        """
        cutoff = datetime.now() - timedelta(days=days)

        # Fetch user's conversations
        conv_result = supabase.from_('conversations') \
            .select('id') \
            .eq('user_id', user_id) \
            .gte('updated_at', cutoff.isoformat()) \
            .execute()

        conversation_ids = [c['id'] for c in conv_result.data]

        if not conversation_ids:
            return {'message': 'No user activity in time period'}

        # Fetch messages from these conversations
        msg_result = supabase.from_('messages') \
            .select('*') \
            .in_('conversation_id', conversation_ids) \
            .eq('role', 'user') \
            .execute()

        queries = [m['content'] for m in msg_result.data]

        # Fetch feedback for highly-rated responses
        feedback_result = supabase.from_('feedback') \
            .select('*, messages(sources)') \
            .in_('conversation_id', conversation_ids) \
            .gte('rating', 4) \
            .execute()

        # Extract document preferences
        preferred_documents = {}

        for feedback in feedback_result.data:
            if feedback.get('messages') and feedback['messages'].get('sources'):
                sources = json.loads(feedback['messages']['sources'])
                for source in sources:
                    doc = source.get('document', 'unknown')
                    preferred_documents[doc] = preferred_documents.get(doc, 0) + 1

        # Sort by frequency
        top_documents = sorted(
            preferred_documents.items(),
            key=lambda x: x[1],
            reverse=True
        )[:5]

        # Analyze query topics (simple keyword extraction)
        from collections import Counter
        import re

        # Extract important words from queries
        stop_words = {'what', 'how', 'when', 'where', 'who', 'is', 'are', 'the', 'a', 'an', 'our', 'my'}
        words = []
        for query in queries:
            query_words = re.findall(r'\b\w+\b', query.lower())
            words.extend([w for w in query_words if w not in stop_words and len(w) > 3])

        top_topics = Counter(words).most_common(10)

        return {
            'user_id': user_id,
            'query_count': len(queries),
            'feedback_count': len(feedback_result.data),
            'top_documents': [{'document': doc, 'count': count} for doc, count in top_documents],
            'top_topics': [{'topic': topic, 'count': count} for topic, count in top_topics],
            'preference_score': len(feedback_result.data) / max(len(queries), 1)
        }

    def apply_learning(self) -> Dict[str, Any]:
        """
        Apply learned optimizations to the system.

        Returns:
            Summary of applied changes
        """
        # Analyze retrieval quality
        retrieval_analysis = self.analyze_retrieval_quality(days=7)

        changes_applied = []

        # Adjust similarity threshold if needed
        if 'optimal_threshold' in retrieval_analysis:
            optimal = retrieval_analysis['optimal_threshold']

            # Store configuration (in production, update config file or database)
            config = {
                'similarity_threshold': optimal,
                'updated_at': datetime.now().isoformat(),
                'reason': 'Learning from feedback'
            }

            changes_applied.append({
                'change': 'similarity_threshold',
                'old_value': 0.7,  # default
                'new_value': optimal,
                'rationale': retrieval_analysis['recommendation']
            })

        # Get improvement suggestions
        suggestions = self.feedback_collector.get_improvement_suggestions()

        # Log learning event
        learning_event = {
            'timestamp': datetime.now().isoformat(),
            'changes': changes_applied,
            'suggestions': suggestions,
            'metrics': {
                'retrieval_quality': retrieval_analysis
            }
        }

        self.learning_history.append(learning_event)

        return {
            'changes_applied': changes_applied,
            'suggestions': suggestions,
            'learning_event': learning_event
        }

    def get_learning_report(self) -> Dict[str, Any]:
        """
        Generate comprehensive learning report.

        Returns:
            Report with all learning insights
        """
        # Get feedback analysis
        feedback_analysis = self.feedback_collector.analyze_feedback(days=30)

        # Get retrieval analysis
        retrieval_analysis = self.analyze_retrieval_quality(days=30)

        # Get failing queries
        failing = self.feedback_collector.identify_failing_queries()

        # Get improvement suggestions
        suggestions = self.feedback_collector.get_improvement_suggestions()

        return {
            'generated_at': datetime.now().isoformat(),
            'feedback_summary': feedback_analysis,
            'retrieval_analysis': retrieval_analysis,
            'failing_queries_count': len(failing),
            'improvement_suggestions': suggestions,
            'learning_history_count': len(self.learning_history)
        }

# Test learning system
if __name__ == "__main__":
    print("="*60)
    print("Learning System Test")
    print("="*60)

    learning = LearningSystem()

    # Analyze retrieval quality
    print("\nAnalyzing Retrieval Quality...")
    retrieval = learning.analyze_retrieval_quality(days=30)

    print(f"\nTotal Analyzed: {retrieval.get('total_analyzed', 0)}")
    print(f"Optimal Threshold: {retrieval.get('optimal_threshold', 'N/A')}")
    if 'recommendation' in retrieval:
        print(f"Recommendation: {retrieval['recommendation']}")

    # Apply learning
    print("\n" + "="*60)
    print("Applying Learning...")
    print("="*60)

    results = learning.apply_learning()

    print(f"\nChanges Applied: {len(results['changes_applied'])}")
    for change in results['changes_applied']:
        print(f"  - {change['change']}: {change['old_value']} → {change['new_value']}")
        print(f"    Rationale: {change['rationale']}")

    print(f"\nSuggestions:")
    for category, items in results['suggestions'].items():
        if items:
            print(f"\n  {category.upper()}:")
            for suggestion in items:
                print(f"    - {suggestion}")

    # Generate report
    print("\n" + "="*60)
    print("Learning Report")
    print("="*60)

    report = learning.get_learning_report()
    print(f"\nGenerated: {report['generated_at']}")
    print(f"Feedback Items: {report['feedback_summary'].get('total_feedback', 0)}")
    print(f"Average Rating: {report['feedback_summary'].get('average_rating', 'N/A')}")
    print(f"Satisfaction Rate: {report['feedback_summary'].get('satisfaction_rate', 0):.1%}")
    print(f"Failing Queries: {report['failing_queries_count']}")
```

**Step 2: Test the Learning System (5 mins)**

```bash
# Run in Terminal (VS Code terminal or command line)
python src/documind/memory/learning.py
```

---

### Quiz 3:

**Question 1:** What is a learning loop in AI systems?\
   a) A programming loop that never ends\
   b) A continuous cycle of collecting feedback, analyzing patterns, adjusting parameters, and measuring improvement\
   c) A bug in the code\
   d) A way to save memory

**Question 2:** How does the learning system determine the optimal similarity threshold?\
   a) It picks a random number\
   b) It always uses 0.7\
   c) It asks the user to choose\
   d) It tests multiple thresholds and selects the one that results in the highest average user ratings

**Question 3:** Why is user preference learning important for personalization?\
   a) It makes the database larger\
   b) It slows down queries\
   c) It allows the system to prioritize documents and topics that align with individual user interests, improving relevance\
   d) It's required by law

**Answers:**
1. **b)** Learning loop = feedback → analysis → adjustment → measurement → repeat
2. **d)** Optimal threshold is found by maximizing average ratings
3. **c)** Preference learning enables personalized, relevant results

---

## Module 4: Challenge Project - Intelligent DocuMind with Memory (15 minutes)

### Challenge Overview

Build a complete learning system for DocuMind that remembers conversations, learns from feedback, and personalizes responses.

**Your Mission:**
Create an intelligent DocuMind that:
1. Maintains multi-turn conversation context
2. Collects and analyzes user feedback
3. Learns optimal retrieval parameters
4. Personalizes results based on user history
5. Provides cross-session context restoration

---

### Challenge Requirements

**Feature:** Intelligent Learning DocuMind

**What to Build:**

1. **Conversation-Aware Q&A**
   - Integrate ConversationMemory with Q&A pipeline
   - Support follow-up questions with context
   - Resolve pronouns (it, that, them) using history

2. **Feedback Integration**
   - Add feedback prompts after each answer
   - Store ratings with full context
   - Display feedback statistics

3. **Learning Engine**
   - Automatically adjust retrieval parameters
   - Learn user document preferences
   - Apply optimizations weekly

4. **Personalization**
   - Boost documents user prefers
   - Adapt response style
   - Track and use query topics

5. **Session Management**
   - Save conversation state
   - Restore previous conversations
   - Show conversation history

---

### Starter Code

Create `src/documind/intelligent_qa.py` **(in VS Code or use Claude Code to create the file)**:

```python
"""
Intelligent DocuMind Q&A System
With memory, learning, and personalization
"""
import os
from typing import Optional, Dict, Any, List
from datetime import datetime
from openai import OpenAI

from documind.memory.conversation import ConversationMemory, list_user_conversations
from documind.memory.feedback import FeedbackCollector
from documind.memory.learning import LearningSystem
from documind.rag.search import search_documents

# Configuration
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")

client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=OPENROUTER_API_KEY
)

class IntelligentQA:
    """
    Intelligent Q&A system with memory and learning.
    """

    def __init__(self, user_id: str = "anonymous"):
        self.user_id = user_id
        self.conversation: Optional[ConversationMemory] = None
        self.feedback_collector = FeedbackCollector(user_id=user_id)
        self.learning_system = LearningSystem()
        self.user_preferences = None

    def start_conversation(self, conversation_id: Optional[str] = None) -> str:
        """
        TODO: Start a new conversation or restore existing one.

        Args:
            conversation_id: Optional ID to restore existing conversation

        Returns:
            Conversation ID
        """
        pass

    def ask(
        self,
        question: str,
        model: str = 'anthropic/claude-3.5-sonnet'
    ) -> Dict[str, Any]:
        """
        TODO: Ask a question with conversation context.

        Steps:
        1. Add user message to conversation
        2. Get conversation context
        3. Retrieve documents (with personalization)
        4. Generate answer with context
        5. Add assistant message
        6. Return response
        """
        pass

    def submit_feedback(
        self,
        message_id: int,
        rating: int,
        comment: Optional[str] = None
    ) -> None:
        """
        TODO: Submit feedback for an answer.

        Args:
            message_id: Message being rated
            rating: 1-5 stars
            comment: Optional comment
        """
        pass

    def personalize_search(
        self,
        query: str,
        top_k: int = 5
    ) -> List[Dict[str, Any]]:
        """
        TODO: Personalized document retrieval.

        Steps:
        1. Get user preferences
        2. Retrieve documents
        3. Boost preferred documents
        4. Re-rank results
        """
        pass

    def apply_learning(self) -> Dict[str, Any]:
        """
        TODO: Apply learning from feedback.

        Returns:
            Learning results
        """
        pass

    def get_conversation_list(self) -> List[Dict[str, Any]]:
        """
        TODO: List user's conversations.

        Returns:
            List of conversations with metadata
        """
        pass

    def get_insights(self) -> Dict[str, Any]:
        """
        TODO: Get user insights and analytics.

        Returns:
            User statistics and preferences
        """
        pass

def main():
    """
    TODO: Implement interactive CLI with all features:

    Commands:
    - /new - Start new conversation
    - /list - List conversations
    - /load <id> - Load conversation
    - /feedback - Rate last answer
    - /insights - Show user insights
    - /learn - Apply learning
    - /quit - Exit

    Features:
    - Multi-turn conversation
    - Context awareness
    - Feedback collection
    - Personalization
    - Learning
    """
    pass

if __name__ == "__main__":
    main()
```

---

### Your Task

**Step 1: Implement IntelligentQA Class (10 mins)**

Complete all TODO methods in `intelligent_qa.py`:
1. `start_conversation()` - Create or restore conversation
2. `ask()` - Context-aware Q&A
3. `submit_feedback()` - Collect ratings
4. `personalize_search()` - User-specific retrieval
5. `apply_learning()` - Apply optimizations
6. `get_conversation_list()` - List conversations
7. `get_insights()` - User analytics

**Step 2: Build Interactive CLI (3 mins)**

Implement `main()` with all commands and features.

**Step 3: Test Complete System (2 mins)**

```bash
# Run in Terminal (VS Code terminal or command line)
python src/documind/intelligent_qa.py

# Test scenario:
# 1. Start conversation
# 2. Ask "What is our vacation policy?"
# 3. Rate the answer
# 4. Ask follow-up "How do I request it?"
# 5. Check insights
# 6. Apply learning
```

---

### Success Criteria

Your implementation is complete when:

- [ ] Conversations maintain multi-turn context
- [ ] Follow-up questions work correctly
- [ ] Feedback is collected and stored
- [ ] Learning system adjusts parameters
- [ ] Personalization boosts preferred documents
- [ ] Cross-session restoration works
- [ ] CLI provides all commands
- [ ] User insights are meaningful

---

## Answer Key

Complete solution provided in workshop materials. Key implementations:
- Conversation context management
- Feedback integration with Q&A pipeline
- Learning algorithm with parameter optimization
- Personalization based on user history
- CLI with all features

---

## Key Takeaways

By completing this workshop, you've learned:

1. **Memory Types**: Short-term (conversation), long-term (preferences), working (task state), episodic (history)
2. **Feedback Collection**: Explicit (ratings) and implicit (behavior) feedback
3. **Learning Loops**: Continuous improvement through feedback analysis
4. **Personalization**: User-specific document ranking and response adaptation
5. **Integration**: Memory enhances RAG with context and learning

**The Memory Formula:**
```
Intelligence = Base Capability + Memory + Learning from Experience
```

---

## Next Session Preview

In **Session 10: Evaluation Tools**, we'll:
- Implement RAGAS evaluation framework
- Measure answer quality automatically
- Use TruLens for observability
- Compare models quantitatively
- Build comprehensive evaluation suites

**Preparation:**
1. Install: `pip install ragas trulens-eval`
2. Create test Q&A dataset (20+ queries with expected answers)
3. Review Session 6 RAG implementation

See you in Session 10!

---

**Workshop Complete! 🎉**

You've built an intelligent DocuMind that remembers, learns, and improves over time. Your AI system now has memory and can personalize responses!
