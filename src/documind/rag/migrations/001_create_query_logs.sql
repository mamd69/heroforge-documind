-- Migration: Create query_logs table for Production Q&A System
-- Run this in Supabase SQL Editor
-- Version: 001
-- Date: 2025-12-18

-- =============================================================================
-- QUERY LOGS TABLE
-- =============================================================================

CREATE TABLE IF NOT EXISTS query_logs (
    id BIGSERIAL PRIMARY KEY,

    -- Query information
    question TEXT NOT NULL,
    answer TEXT NOT NULL,

    -- Model information
    model VARCHAR(100) NOT NULL,
    fallback_used BOOLEAN DEFAULT false,

    -- Sources (JSONB for flexibility)
    sources JSONB,

    -- Performance metrics
    response_time FLOAT NOT NULL DEFAULT 0,

    -- Metadata
    complexity VARCHAR(20) DEFAULT 'medium',

    -- User feedback (optional)
    feedback_rating INTEGER CHECK (feedback_rating BETWEEN 1 AND 5),
    feedback_comment TEXT,

    -- Timestamps
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

-- =============================================================================
-- INDEXES
-- =============================================================================

-- Primary index for time-based queries
CREATE INDEX IF NOT EXISTS idx_query_logs_created_at
    ON query_logs(created_at DESC);

-- Index for model usage analysis
CREATE INDEX IF NOT EXISTS idx_query_logs_model
    ON query_logs(model);

-- Index for response time analysis
CREATE INDEX IF NOT EXISTS idx_query_logs_response_time
    ON query_logs(response_time);

-- GIN index for JSONB sources (allows querying inside sources)
CREATE INDEX IF NOT EXISTS idx_query_logs_sources
    ON query_logs USING GIN (sources);

-- =============================================================================
-- TRIGGER FOR UPDATED_AT
-- =============================================================================

CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

-- Create trigger only if it doesn't exist
DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM pg_trigger WHERE tgname = 'update_query_logs_updated_at'
    ) THEN
        CREATE TRIGGER update_query_logs_updated_at
            BEFORE UPDATE ON query_logs
            FOR EACH ROW
            EXECUTE FUNCTION update_updated_at_column();
    END IF;
END
$$;

-- =============================================================================
-- ROW LEVEL SECURITY (Optional - Enable if using Supabase Auth)
-- =============================================================================

-- Uncomment below to enable RLS
-- ALTER TABLE query_logs ENABLE ROW LEVEL SECURITY;

-- Allow all operations for service role
-- CREATE POLICY "Service role can do everything" ON query_logs
--     FOR ALL
--     TO service_role
--     USING (true)
--     WITH CHECK (true);

-- Allow anon to insert (for logging)
-- CREATE POLICY "Allow anon insert" ON query_logs
--     FOR INSERT
--     TO anon
--     WITH CHECK (true);

-- Allow anon to select (for analytics)
-- CREATE POLICY "Allow anon select" ON query_logs
--     FOR SELECT
--     TO anon
--     USING (true);

-- =============================================================================
-- SAMPLE DATA (Optional - for testing)
-- =============================================================================

-- INSERT INTO query_logs (question, answer, model, sources, response_time, complexity)
-- VALUES
--     ('What is the vacation policy?',
--      'According to [Source 1], employees receive 15 days of vacation per year.',
--      'google/gemini-2.5-flash-lite',
--      '[{"document": "hr_policies.txt", "chunk": 0, "similarity": 0.89, "cited": true}]',
--      1.234,
--      'simple'),
--     ('How many sick days do I get?',
--      'Based on [Source 1], you are entitled to 10 sick days annually.',
--      'google/gemini-2.5-flash-lite',
--      '[{"document": "benefits_guide.txt", "chunk": 2, "similarity": 0.92, "cited": true}]',
--      0.987,
--      'simple');

-- =============================================================================
-- VERIFICATION
-- =============================================================================

-- Check table exists
SELECT EXISTS (
    SELECT FROM information_schema.tables
    WHERE table_name = 'query_logs'
) AS table_exists;

-- Check indexes
SELECT indexname FROM pg_indexes WHERE tablename = 'query_logs';

-- Success message
DO $$
BEGIN
    RAISE NOTICE '✓ Migration 001_create_query_logs completed successfully!';
END
$$;
