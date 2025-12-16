# Workflow Patterns

This reference covers patterns for structuring multi-step processes in skills.

## Sequential Workflows

Use sequential workflows when tasks have a clear linear progression with ordered steps.

### Pattern Structure

```markdown
## Workflow Overview

This process involves 5 steps:
1. [Step 1 name]
2. [Step 2 name]
3. [Step 3 name]
4. [Step 4 name]
5. [Step 5 name]

## Step 1: [Name]

[Instructions for step 1]

## Step 2: [Name]

[Instructions for step 2]
...
```

### Example: PDF Form Filling

```markdown
## Workflow Overview

PDF form filling involves 5 steps:
1. Analyze the form structure
2. Create field mapping
3. Validate the mapping
4. Populate the form
5. Verify output

## Step 1: Analyze Form

Run the analysis script to extract form field information:
\`\`\`bash
python scripts/analyze_form.py input.pdf
\`\`\`

## Step 2: Create Field Mapping

Based on the analysis, create a mapping of field names to values...
```

## Conditional Workflows

Use conditional workflows when the next steps depend on an initial determination or when different paths apply to different scenarios.

### Pattern Structure

```markdown
## Workflow Decision Tree

First, determine the task type:

### Creating New Content?

If creating new content from scratch:
1. [Creation step 1]
2. [Creation step 2]
3. [Creation step 3]

### Editing Existing Content?

If modifying existing content:
1. [Editing step 1]
2. [Editing step 2]
3. [Editing step 3]
```

### Example: Document Processing

```markdown
## Workflow Decision Tree

First, determine the document operation:

### Creating a New Document?

1. Initialize document structure
2. Add content sections
3. Apply formatting
4. Save to output path

### Editing an Existing Document?

1. Load the existing document
2. Locate the target section
3. Apply modifications
4. Preserve existing formatting
5. Save changes
```

## Best Practices

1. **Provide upfront visibility** - Start with an overview so Claude understands the full scope
2. **Number steps clearly** - Use explicit numbering for sequential steps
3. **Include decision points** - Make branching conditions explicit
4. **Reference scripts when needed** - Point to executable code for complex operations
5. **Keep steps atomic** - Each step should be a single, completable action
