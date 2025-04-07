# Large GraphQL Schema Agent

An intelligent agent for querying and analyzing large GraphQL schemas using local TensorFlow embeddings and Claude LLMs.

## Overview

This TypeScript-based LangChain agent provides natural language understanding of complex GraphQL schemas:

- **Local Embedding Generation**: Uses TensorFlow.js with Universal Sentence Encoder to create semantic embeddings of your GraphQL schema on your local machine, avoiding repeated cloud embedding costs
- **Smart Chunking**: Intelligently splits and manages schema types to maintain semantic coherence
- **Relationship Analysis**: Understands and can explain relationships between types in your schema
- **High-Performance**: Optimized for handling large schemas with efficient batching and memory management
- **Claude Integration**: Leverages Anthropic's Claude models for deep reasoning about schema design and structure as well as Claude's friendly persona and easy-going nature
- **Contextual Retrieval**: Automatically retrieves and includes related types for comprehensive analysis

## Installation

```bash
# Install dependencies
npm install

# Copy your GraphQL schema
cp path/to/your/schema.graphql ./large-schema.graphql

# If you want to use GPU acceleration for embeddings (optional)
# Uncomment the GPU import in the code and install CUDA dependencies
```

**By default, the Anthropic Opus 3 model is selected, which is excellent
but expensive. You can change this in index.ts. Sonnet is a good middle ground.**

## Prerequisites

- Node.js 18+
- An Anthropic API key (Claude 3 Opus or Sonnet 3.5+ recommended for best results)
- TensorFlow.js dependencies

## Configuration

The agent can be configured with several options:

- `maxTokensPerChunk`: Controls how schema types are split (default: 4000)
- `batchSize`: Adjusts embedding batch size for performance tuning (default: 16)
- `temperature`: Controls LLM response determinism (default: 0)

## Usage

1. Set your Anthropic API key:

   ```bash
   export ANTHROPIC_API_KEY=your_key_here
   ```

2. Start the agent:

   ```bash
   npm start
   ```

3. Wait for indexing to complete (typically 20-60 seconds depending on schema size)

4. Enter natural language queries about your schema

## Query Examples

The agent can answer questions like:

- "What are the main query entry points in this schema?"
- "How are JobResponse and WorkMaster types related?"
- "Explain the authentication system in this schema"
- "What fields are required when creating a new Equipment?"
- "What's the relationship between WorkDirective and WorkMaster?"
- "Show me all types that reference the WorkflowSpecificationi type"
- "What mutations are available for managing personnel?"

## Example Interaction

**Prompt**

```
Would it be correct to think of a WorkDirective as a dispatched WorkMaster?
```

**Response**

```
Yes, based on the provided GraphQL schema, it would be correct to think of a WorkDirective as a dispatched WorkMaster.

A few key points from the schema:

1. The WorkMaster represents template information not associated with any specific job order. It defines the detailed steps needed to accomplish an operation.

2. WorkDirectives start as copies of WorkMasters and are augmented with information for a specific job order.

3. The WorkDirective type has a workMaster field that references the WorkMaster it was created from.

4. The WorkMaster type does not have any direct association to a job order, while the WorkDirective has jobOrders and jobResponses fields linking it to specific job execution.

So in summary, the WorkMaster acts as a reusable template that gets copied into a WorkDirective when it needs to be executed for a specific job. The WorkDirective then represents the dispatched work with links to the job details. This matches the concept of a WorkDirective being a dispatched version of a WorkMaster.

Query Metrics:
- Tokens In: 89,321
- Tokens Out: 256
- Processing Time: 23.06s
- Chunks Processed: 709
- Total Chunks: 2656
- Chunks Used: 26.69%
```

**Prompt**

```
If I want to update an Equipment with an id of `EQUIP-1` and set its effectiveEnd to today's date, how can I do  that
```

**Response**

````
Based on the GraphQL schema chunks provided, I don't see the complete Equipment type definition or its update mutation in the provided schema chunks. However, I can infer how to update an Equipment based on the patterns shown for other entity updates in the schema.

To update an Equipment with ID "EQUIP-1" and set its effectiveEnd to today's date, you would likely use a mutation similar to:

```graphql
mutation UpdateEquipment {
  updateEquipment(
    input: {
      filter: {
        id: { eq: "EQUIP-1" }
      },
      set: {
        effectiveEnd: "2023-07-12T00:00:00Z"  # Use today's date in ISO format
      }
    }
  ) {
    equipment {
      id
      label
      effectiveStart
      effectiveEnd
    }
    numUids
  }
}

This follows the pattern seen in other update mutations in the schema, such as `UpdateOperationsRequest` and `UpdateEquipmentCapability`. The mutation would:

1. Target the equipment with ID "EQUIP-1" using the filter
2. Set the effectiveEnd field to today's date
3. Return the updated equipment with its key fields

Note that you should replace the date value with the current date in ISO 8601 format (YYYY-MM-DDThh:mm:ssZ).

While the specific Equipment update mutation isn't shown in the provided schema chunks, this approach follows the consistent pattern used throughout the schema for entity updates.

Query Metrics:
- Tokens In: 9,468
- Tokens Out: 336
- Processing Time: 6.92s
- Chunks Processed: 43
- Total Chunks: 2656
- Chunks Used: 1.62%


````

## Performance Tuning

### Memory Usage

For very large schemas, adjust the batch size in `TensorflowEmbeddings` constructor:

```typescript
const embeddings = new TensorflowEmbeddings({
  batchSize: 8, // Lower for less memory usage
});
```

### Processing Speed

For faster processing on capable hardware [I couldn't get this working on WSL2 even with CUDA support ON, if you can, let me know how]:

```typescript
// Import GPU backend instead of CPU
import * as tf from '@tensorflow/tfjs-node-gpu';
```

## Advanced Usage

The `GraphQLSchemaAgent` class provides several methods beyond basic querying:

- `getAllTypeNames()`: List all type definitions in the schema
- `getTypeInfo(typeName)`: Get details about a specific type
- `findConnectionsBetweenTypes(typeA, typeB)`: Analyze relationships between types
- `getMetrics()`: Get detailed usage and performance metrics
- `getMetricsReport()`: Generate a formatted report of agent activity

## Extending the Agent

The agent is built on LangChain, making it easy to extend with:

- Additional vector stores (Pinecone, Milvus, etc.) for persistence so you don't need to wait on every start
- Different embedding models
- Alternative LLMs (OpenAI, etc.)
- Custom schema processing logic

## Contributing

Please help me fix this! Please feel free to submit a Pull Request.
