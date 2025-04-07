import { ChatAnthropic } from '@langchain/anthropic';
import { GraphQLSchemaAgent } from './graph-schema-agent.js';
import { readFileSync } from 'fs';
import { MemoryVectorStore } from 'langchain/vectorstores/memory';
import { EmbeddingsInterface } from '@langchain/core/embeddings';

import * as tf from '@tensorflow/tfjs-node'; // cpu based model for compatibility
// import * as tf from '@tensorflow/tfjs-node-gpu'; // gpu based model for performance

import * as readline from 'readline';
import { isCompositeType } from 'graphql';

// We'll dynamically import USE since it's an ESM module
let use: any = null;
tf.setBackend('tensorflow');

/**
 * This example uses TensorFlow.js with Universal Sentence Encoder
 * for local embeddings generation
 *
 * Prerequisites:
 * 1. Install necessary packages:
 *    npm install @tensorflow/tfjs-node @tensorflow-models/universal-sentence-encoder
 */

// Custom embeddings class using TensorFlow.js Universal Sentence Encoder
class TensorflowEmbeddings implements EmbeddingsInterface {
  private model: any;
  private initialized: boolean = false;
  private initializationPromise: Promise<void> | null = null;
  private batchSize: number;

  constructor(options: { batchSize?: number } = {}) {
    this.batchSize = options.batchSize || 32;
  }

  private async initialize() {
    if (this.initialized) return;

    if (!this.initializationPromise) {
      this.initializationPromise = (async () => {
        try {
          // Dynamically import the Universal Sentence Encoder
          if (!use) {
            console.log('Importing Universal Sentence Encoder...');
            use = await import('@tensorflow-models/universal-sentence-encoder');
          }

          console.log('Loading Universal Sentence Encoder model...');
          this.model = await use.load();
          console.log('Universal Sentence Encoder model loaded successfully');

          this.initialized = true;
        } catch (error) {
          console.error('Failed to initialize TensorFlow embeddings:', error);
          throw error;
        }
      })();
    }

    return this.initializationPromise;
  }

  async embedDocuments(texts: string[]): Promise<number[][]> {
    await this.initialize();

    const batches = [];
    for (let i = 0; i < texts.length; i += this.batchSize) {
      batches.push(texts.slice(i, i + this.batchSize));
    }

    const embeddings: number[][] = [];

    for (const batch of batches) {
      try {
        // Process texts to handle potential model limitations
        const processedBatch = batch.map((text) => {
          // Truncate long texts to avoid potential issues
          // USE handles about 512 tokens max
          if (text.length > 8000) {
            return text.substring(0, 8000);
          }
          return text;
        });

        //console.log(`Embedding batch of ${processedBatch.length} documents...`);
        const embeddingsTensor = await this.model.embed(processedBatch);

        // Convert to TypedArray, then to standard arrays
        const embeddingsArray = await embeddingsTensor.array();
        embeddings.push(...embeddingsArray);
      } catch (error) {
        console.error('Error embedding batch:', error);

        // Fall back to one-by-one processing if batch fails
        for (const text of batch) {
          try {
            const singleEmbedding = await this.embedQuery(text);
            embeddings.push(singleEmbedding);
          } catch (innerError) {
            console.error(`Failed to embed text "${text.substring(0, 100)}...":`, innerError);
            // Use a zero vector as fallback (USE produces 512-dimension vectors)
            embeddings.push(new Array(512).fill(0));
          }
        }
      }
    }

    return embeddings;
  }

  async embedQuery(text: string): Promise<number[]> {
    await this.initialize();

    try {
      // Truncate long queries
      const processedText = text.length > 8000 ? text.substring(0, 8000) : text;

      const embeddingsTensor = await this.model.embed([processedText]);
      const embeddingsArray = await embeddingsTensor.array();

      return embeddingsArray[0];
    } catch (error) {
      console.error('Error embedding query:', error);
      throw error;
    }
  }
}

async function main() {
  // Load your large GraphQL schema
  const schemaString = readFileSync('./large-schema.graphql', 'utf8');

  console.log(`Loaded schema: ${schemaString.length} characters`);

  // Initialize Claude with large context window
  const llm = new ChatAnthropic({
    //modelName: 'claude-3-opus-20240229',
    modelName: 'claude-3-7-sonnet-20250219',
    //modelName: 'claude-3-5-haiku-20241022',
    temperature: 0,
    maxTokens: 4000,
    anthropicApiKey: process.env.ANTHROPIC_API_KEY,
  });

  // Initialize TensorFlow.js embeddings with a conservative batch size
  const embeddings = new TensorflowEmbeddings({
    batchSize: 16, // Adjust batch size for performance
  });

  const vectorStore = new MemoryVectorStore(embeddings);

  console.log('Initializing GraphQL Schema Agent with TensorFlow.js embeddings...');
  const agent = new GraphQLSchemaAgent({
    llm,
    schemaString,
    vectorStore,
    maxTokensPerChunk: 4000, // More conservative chunk size for USE
    systemPrompt: `You are a helpful assistant that analyzes GraphQL schemas.
    You will be provided with chunks of a GraphQL schema.
    Answer questions based on the schema information provided to you.
    Always provide concise, accurate answers based solely on the schema definition.
    If asked about implementation details not visible in the schema, indicate that
    you can only answer based on the schema structure.`,
  });

  console.log('Agent initialized. Type definitions found:', agent.getAllTypeNames().length);

  agent.onReady = () => {
    const rl = readline.createInterface({
      input: process.stdin,
      output: process.stdout,
    });

    console.log('\nGraphQL Schema Query Interface');
    console.log('Type "exit" or "quit" to end the session');
    console.log('----------------------------------------');

    // Get metrics after the agent is fully ready (including indexing)
    const agentMetrics = agent.getMetrics();

    // Function to ask for input and process query
    function askQuestion() {
      rl.question('\nEnter your query about the schema: ', async (query) => {
        if (query.toLowerCase() === 'exit' || query.toLowerCase() === 'quit') {
          console.log('Exiting session...');
          rl.close();
          return;
        }

        try {
          console.log('Processing query...');
          const response = await agent.query(query);
          console.log('\nResponse:');
          console.log('----------------------------------------');
          console.log(response);

          // Display metrics for the query
          const metrics = agent.getMetrics();
          const latestQuery = metrics.queries.queryDetails[metrics.queries.queryDetails.length - 1];

          const chunksPercentageUsed = ((latestQuery.chunks / metrics.totalChunks) * 100).toFixed(
            2
          );

          console.log('\nQuery Metrics:');
          console.log(`- Tokens In: ${latestQuery.tokensIn.toLocaleString()}`);
          console.log(`- Tokens Out: ${latestQuery.tokensOut.toLocaleString()}`);
          console.log(`- Processing Time: ${(latestQuery.processingTimeMs / 1000).toFixed(2)}s`);
          console.log(`- Chunks Processed: ${latestQuery.chunks}`);
          console.log(`- Total Chunks: ${metrics.totalChunks}`);
          console.log(`- Chunks Used: ${chunksPercentageUsed}%`);
        } catch (error) {
          console.error('Error processing query:', error);
        }

        // Continue with next question
        askQuestion();
      });
    }
    askQuestion();
  };
}

// Run the example
main().catch(console.error);
