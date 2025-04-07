import { BaseMessage, HumanMessage, AIMessage, SystemMessage } from '@langchain/core/messages';

import { BaseChatModel } from '@langchain/core/language_models/chat_models';

import {
  GraphQLSchema,
  buildSchema,
  parse,
  DefinitionNode,
  Kind,
  TypeDefinitionNode,
  ObjectTypeDefinitionNode,
  InterfaceTypeDefinitionNode,
  InputObjectTypeDefinitionNode,
} from 'graphql';

import { VectorStore } from '@langchain/core/vectorstores';
import { OpenAIEmbeddings } from '@langchain/openai';
import { Document } from 'langchain/document';
import { RecursiveCharacterTextSplitter } from 'langchain/text_splitter';
import { MemoryVectorStore } from 'langchain/vectorstores/memory';

// Type definitions for our agent
export interface SchemaAgentOptions {
  llm: BaseChatModel;
  schemaString: string;
  vectorStore?: VectorStore;
  maxTokensPerChunk?: number;
  systemPrompt?: string;
}

export class GraphQLSchemaAgent {
  private llm: BaseChatModel;
  private schema: GraphQLSchema;
  private schemaAST: readonly DefinitionNode[];
  private vectorStore: VectorStore;
  private chunkMap: Map<string, SchemaChunk> = new Map();
  private systemPrompt: string;
  private maxTokensPerChunk: number;
  private typeDefinitionMap: Map<string, TypeDefinitionNode> = new Map();
  private contextHistory: BaseMessage[] = [];
  onReady?: () => void;

  // Metrics tracking
  private metrics: {
    totalChunks: number;
    totalSplitChunks: number;
    totalOriginalChunks: number;
    estimatedTokensIndexed: number;
    queriesPerformed: number;
    chunks: Record<string, number>;
    splitChunks: Record<string, number>;
    tokensIn: Record<string, number>;
    tokensOut: Record<string, number>;
    processingTimes: Record<string, number>;
  } = {
    chunks: {},
    splitChunks: {},
    totalChunks: 0,
    totalSplitChunks: 0,
    totalOriginalChunks: 0,
    estimatedTokensIndexed: 0,
    queriesPerformed: 0,
    tokensIn: {},
    tokensOut: {},
    processingTimes: {},
  };

  constructor(options: SchemaAgentOptions) {
    this.llm = options.llm;
    this.maxTokensPerChunk = options.maxTokensPerChunk || 4000;
    this.systemPrompt =
      options.systemPrompt ||
      `You are a helpful assistant that analyzes GraphQL schemas. 
      You will be provided with chunks of a GraphQL schema. 
      Answer questions based on the schema information provided to you.`;

    try {
      // Parse the schema string to get both AST and executable schema
      this.schema = buildSchema(options.schemaString);
      this.schemaAST = parse(options.schemaString).definitions;

      // Process and chunk the schema
      this.processSchema();

      // Set up vector store for semantic search
      this.vectorStore = options.vectorStore || new MemoryVectorStore(new OpenAIEmbeddings());

      // Index the chunks for retrieval
      const p = this.indexChunks();
      p.then(() => {
        this.onReady && this.onReady();
      });
    } catch (error) {
      throw new Error(`Failed to initialize GraphQL Schema Agent: ${error}`);
    }
  }

  /**
   * Process the schema into manageable chunks for analysis
   */
  private processSchema(): void {
    // First pass: Build a map of all type definitions by name for easy lookup
    for (const def of this.schemaAST) {
      if (this.isTypeDefinition(def)) {
        this.typeDefinitionMap.set(def.name.value, def);
      }
    }

    // Second pass: Create chunks for each type definition
    for (const def of this.schemaAST) {
      if (this.isTypeDefinition(def)) {
        const chunk = this.createChunkFromTypeDefinition(def);
        this.chunkMap.set(chunk.id, chunk);
      }
    }

    // Create special chunks for the schema directives, queries, mutations, and subscriptions
    this.createSchemaLevelChunks();
  }

  /**
   * Check if a node is a type definition
   */
  private isTypeDefinition(node: DefinitionNode): node is TypeDefinitionNode {
    return [
      Kind.OBJECT_TYPE_DEFINITION,
      Kind.INTERFACE_TYPE_DEFINITION,
      Kind.INPUT_OBJECT_TYPE_DEFINITION,
      Kind.ENUM_TYPE_DEFINITION,
      Kind.UNION_TYPE_DEFINITION,
      Kind.SCALAR_TYPE_DEFINITION,
    ].includes(node.kind);
  }

  /**
   * Create a chunk from a type definition
   */
  private createChunkFromTypeDefinition(def: TypeDefinitionNode): SchemaChunk {
    const typeName = def.name.value;
    let chunkType: ChunkType;

    switch (def.kind) {
      case Kind.OBJECT_TYPE_DEFINITION:
        chunkType = ChunkType.OBJECT;
        break;
      case Kind.INTERFACE_TYPE_DEFINITION:
        chunkType = ChunkType.INTERFACE;
        break;
      case Kind.INPUT_OBJECT_TYPE_DEFINITION:
        chunkType = ChunkType.INPUT;
        break;
      case Kind.ENUM_TYPE_DEFINITION:
        chunkType = ChunkType.ENUM;
        break;
      case Kind.UNION_TYPE_DEFINITION:
        chunkType = ChunkType.UNION;
        break;
      case Kind.SCALAR_TYPE_DEFINITION:
        chunkType = ChunkType.SCALAR;
        break;
      default:
        chunkType = ChunkType.OTHER;
    }

    // Get related types for context enrichment
    const relatedTypes: string[] = this.getRelatedTypes(def);

    // Format the definition for storage and embedding
    const formattedDef = this.formatTypeDefinition(def);

    return {
      id: typeName,
      type: chunkType,
      content: formattedDef,
      relatedTypes,
    };
  }

  /**
   * Get related types for a given type definition
   */
  private getRelatedTypes(def: TypeDefinitionNode): string[] {
    const relatedTypes = new Set<string>();

    // Logic to extract related types based on definition kind
    if (def.kind === Kind.OBJECT_TYPE_DEFINITION) {
      const objectDef = def as ObjectTypeDefinitionNode;

      // Add implemented interfaces
      if (objectDef.interfaces) {
        for (const iface of objectDef.interfaces) {
          relatedTypes.add(iface.name.value);
        }
      }

      // Add field types
      if (objectDef.fields) {
        for (const field of objectDef.fields) {
          // Process field type (could be list, non-null, or named type)
          this.extractTypesFromField(field.type, relatedTypes);

          // Process argument types
          if (field.arguments) {
            for (const arg of field.arguments) {
              this.extractTypesFromField(arg.type, relatedTypes);
            }
          }
        }
      }
    } else if (def.kind === Kind.INTERFACE_TYPE_DEFINITION) {
      const interfaceDef = def as InterfaceTypeDefinitionNode;

      // Add field types
      if (interfaceDef.fields) {
        for (const field of interfaceDef.fields) {
          this.extractTypesFromField(field.type, relatedTypes);

          // Process argument types
          if (field.arguments) {
            for (const arg of field.arguments) {
              this.extractTypesFromField(arg.type, relatedTypes);
            }
          }
        }
      }
    } else if (def.kind === Kind.INPUT_OBJECT_TYPE_DEFINITION) {
      const inputDef = def as InputObjectTypeDefinitionNode;

      // Add field types
      if (inputDef.fields) {
        for (const field of inputDef.fields) {
          this.extractTypesFromField(field.type, relatedTypes);
        }
      }
    } else if (def.kind === Kind.ENUM_TYPE_DEFINITION) {
      // Enums don't have related types
    }

    // Remove self-reference
    relatedTypes.delete(def.name.value);

    return Array.from(relatedTypes);
  }

  /**
   * Extract type names from a field's type definition
   */
  private extractTypesFromField(typeNode: any, relatedTypes: Set<string>): void {
    if (typeNode.kind === Kind.NAMED_TYPE) {
      relatedTypes.add(typeNode.name.value);
    } else if (typeNode.kind === Kind.LIST_TYPE || typeNode.kind === Kind.NON_NULL_TYPE) {
      this.extractTypesFromField(typeNode.type, relatedTypes);
    }
  }

  /**
   * Format type definition as a string
   */
  private formatTypeDefinition(def: TypeDefinitionNode): string {
    // For simplicity, use the original schema text for this type
    // In a real implementation, you might want to pretty-print the AST
    return def.loc?.source.body.substring(def.loc.start, def.loc.end) || '';
  }

  /**
   * Create special chunks for schema-level definitions
   */
  private createSchemaLevelChunks(): void {
    // Find Query, Mutation, and Subscription types
    const queryType = this.schema.getQueryType();
    const mutationType = this.schema.getMutationType();
    const subscriptionType = this.schema.getSubscriptionType();

    if (queryType) {
      const chunk = this.chunkMap.get(queryType.name);
      if (chunk) {
        chunk.type = ChunkType.QUERY;
      }
    }

    if (mutationType) {
      const chunk = this.chunkMap.get(mutationType.name);
      if (chunk) {
        chunk.type = ChunkType.MUTATION;
      }
    }

    if (subscriptionType) {
      const chunk = this.chunkMap.get(subscriptionType.name);
      if (chunk) {
        chunk.type = ChunkType.SUBSCRIPTION;
      }
    }

    // Create a chunk for directives
    const directiveDefs = this.schemaAST.filter((def) => def.kind === Kind.DIRECTIVE_DEFINITION);
    if (directiveDefs.length > 0) {
      const directivesContent = directiveDefs
        .map((def) => def.loc?.source.body.substring(def.loc.start, def.loc.end) || '')
        .join('\n\n');

      this.chunkMap.set('_directives', {
        id: '_directives',
        type: ChunkType.DIRECTIVE,
        content: directivesContent,
        relatedTypes: [],
      });
    }
  }

  /**
   * Index all chunks in the vector store for semantic search
   * with safeguards for embedding model context limits
   */
  private async indexChunks(): Promise<void> {
    console.log(`Preparing to index ${this.chunkMap.size} chunks...`);

    // Maximum safe size for embedding models like text-embedding-ada-002
    const MAX_EMBEDDING_TOKENS = 7000; // Conservative limit (actual is 8192)
    const TOKENS_PER_CHAR = 0.25; // Rough estimate: 4 chars ≈ 1 token

    const documents: Document[] = [];
    const splitter = new RecursiveCharacterTextSplitter({
      chunkSize: Math.floor(MAX_EMBEDDING_TOKENS / TOKENS_PER_CHAR),
      chunkOverlap: 200,
    });

    // Metrics tracking
    this.metrics.totalOriginalChunks = this.chunkMap.size;
    let totalTokensEstimate = 0;
    let totalSplitChunks = 0;

    for (const chunk of this.chunkMap.values()) {
      // Check if the chunk might be too large for embeddings
      const estimatedTokens = chunk.content.length * TOKENS_PER_CHAR;
      totalTokensEstimate += estimatedTokens;

      if (estimatedTokens > MAX_EMBEDDING_TOKENS) {
        console.log(
          `Splitting large chunk: ${chunk.id} (est. ${Math.floor(estimatedTokens)} tokens)`
        );

        // Split the large chunk into smaller pieces
        const subDocs = await splitter.createDocuments([chunk.content]);
        totalSplitChunks += subDocs.length;

        // Create a document for each sub-chunk with appropriate metadata
        for (let i = 0; i < subDocs.length; i++) {
          documents.push(
            new Document({
              pageContent: subDocs[i].pageContent,
              metadata: {
                id: `${chunk.id}_part${i + 1}`,
                parentId: chunk.id,
                type: chunk.type,
                relatedTypes: chunk.relatedTypes.join(','),
                isPart: true,
                partIndex: i,
                totalParts: subDocs.length,
              },
            })
          );
        }
      } else {
        // Normal-sized chunk can be indexed directly
        documents.push(
          new Document({
            pageContent: chunk.content,
            metadata: {
              id: chunk.id,
              type: chunk.type,
              relatedTypes: chunk.relatedTypes.join(','),
            },
          })
        );
      }
    }

    // Update metrics
    this.metrics.totalChunks = documents.length;
    this.metrics.totalSplitChunks = totalSplitChunks;
    this.metrics.estimatedTokensIndexed = totalTokensEstimate;

    console.log(
      `Indexing ${documents.length} documents (including split chunks) in batches to keep your CPU from exploding...`
    );

    // Process documents in batches to avoid overwhelming the API
    const BATCH_SIZE = 100;
    for (let i = 0; i < documents.length; i += BATCH_SIZE) {
      const batch = documents.slice(i, i + BATCH_SIZE);
      await this.vectorStore.addDocuments(batch);
      console.log(
        `Indexed batch ${Math.floor(i / BATCH_SIZE) + 1}/${Math.ceil(documents.length / BATCH_SIZE)}`
      );
    }

    console.log(`Indexing complete. ${this.metrics.totalChunks} chunks indexed.`);
  }

  /**
   * Retrieve relevant chunks for a query
   */
  private async retrieveRelevantChunks(query: string, k: number = 5): Promise<SchemaChunk[]> {
    // Perform similarity search
    const results = await this.vectorStore.similaritySearch(query, k * 2); // Get more initially to account for split chunks

    // Process the results to get actual chunks
    const relevantChunks = new Map<string, SchemaChunk>();
    const processedParentIds = new Set<string>();

    for (const doc of results) {
      const metadata = doc.metadata;

      // For split chunks, get the original complete chunk
      if (metadata.isPart && metadata.parentId) {
        const parentId = metadata.parentId;

        // Only process each parent chunk once
        if (!processedParentIds.has(parentId)) {
          processedParentIds.add(parentId);

          const parentChunk = this.chunkMap.get(parentId);
          if (parentChunk && relevantChunks.size < k) {
            relevantChunks.set(parentId, parentChunk);
          }
        }
      } else {
        // Regular non-split chunk
        const chunkId = metadata.id;
        const chunk = this.chunkMap.get(chunkId);

        if (chunk && relevantChunks.size < k) {
          relevantChunks.set(chunkId, chunk);
        }
      }

      // Stop if we have enough chunks
      if (relevantChunks.size >= k) break;
    }

    // Get the final list of chunks
    const chunks = Array.from(relevantChunks.values());

    // Expand context by including directly related chunks
    const expandedChunks = new Map<string, SchemaChunk>();

    for (const chunk of chunks) {
      expandedChunks.set(chunk.id, chunk);

      // Add one level of related chunks for context
      for (const relatedType of chunk.relatedTypes) {
        const relatedChunk = this.chunkMap.get(relatedType);
        if (relatedChunk && !expandedChunks.has(relatedType)) {
          expandedChunks.set(relatedType, relatedChunk);
        }
      }
    }

    return Array.from(expandedChunks.values());
  }

  /**
   * Format chunks into a context string for the LLM, with adaptive chunking
   * to respect token limits
   */
  private formatChunksAsContext(chunks: SchemaChunk[], maxContextTokens: number = 100000): string {
    // First, prioritize chunks by relevance (assumed already done by retrieval)
    // Sort by type to group similar chunks together
    chunks.sort((a, b) => a.type.localeCompare(b.type));

    let context = 'GraphQL Schema Chunks:\n\n';
    let estimatedTokens = 0;
    const tokensPerChar = 0.25; // Rough estimate: 4 chars ≈ 1 token

    // Add chunks until we approach the token limit
    const includedChunks: SchemaChunk[] = [];

    for (const chunk of chunks) {
      // Estimate tokens in this chunk
      const chunkTokens = Math.ceil((chunk.content.length + 50) * tokensPerChar); // +50 for the header

      // If adding this chunk would exceed our limit, stop
      if (estimatedTokens + chunkTokens > maxContextTokens * 0.9) {
        // Leave 10% buffer
        break;
      }

      includedChunks.push(chunk);
      estimatedTokens += chunkTokens;
    }

    // Build the context string from included chunks
    for (const chunk of includedChunks) {
      context += `--- ${chunk.type}: ${chunk.id} ---\n`;
      context += `${chunk.content}\n\n`;
    }

    // Add a note about how many chunks were included
    context += `Note: Included ${includedChunks.length} of ${chunks.length} relevant schema chunks based on your question.`;

    return context;
  }

  /**
   * Query the agent about the GraphQL schema
   */
  public async query(question: string, contextSize: number = 100000): Promise<string> {
    const queryId = `query_${this.metrics.queriesPerformed + 1}`;
    const startTime = Date.now();

    try {
      // Retrieve relevant chunks
      const relevantChunks = await this.retrieveRelevantChunks(question, 20); // Increase retrieved chunks

      // Format chunks as context, respecting the context window
      const context = this.formatChunksAsContext(relevantChunks, contextSize);

      // Prepare messages for the LLM
      const messages: BaseMessage[] = [
        new SystemMessage(this.systemPrompt),
        ...this.contextHistory.slice(-5), // Reduce history to save tokens
        new HumanMessage(`
          ${context}
          
          Question: ${question}
          
          Please answer based on the GraphQL schema chunks provided above.
        `),
      ];

      // Estimate tokens in the prompt
      const promptText = messages.map((msg) => msg.content).join('\n');
      const promptTokens = Math.ceil(promptText.length * 0.25); // Rough estimate

      // Query the LLM
      const response = await this.llm.call(messages);

      // Estimate tokens in the response
      const responseTokens = Math.ceil(response.content.length * 0.25); // Rough estimate

      // Convert response content to string
      const responseString =
        typeof response.content === 'string' ? response.content : JSON.stringify(response.content);

      // Update history
      this.contextHistory.push(new HumanMessage(question));
      this.contextHistory.push(new AIMessage(responseString));

      console.log(
        `Query ID: ${queryId}, chunks: ${relevantChunks.length}, tokens in: ${promptTokens}, tokens out: ${responseTokens}`
      );

      // Track metrics
      this.metrics.queriesPerformed++;
      this.metrics.tokensIn[queryId] = promptTokens;
      this.metrics.tokensOut[queryId] = responseTokens;
      this.metrics.processingTimes[queryId] = Date.now() - startTime;
      this.metrics.chunks[queryId] = relevantChunks.length;
      this.metrics.splitChunks[queryId] = relevantChunks.filter(
        (chunk) => chunk.content.length > this.maxTokensPerChunk
      ).length;

      return responseString;
    } catch (error: any) {
      // If we get a context length error, retry with a smaller context
      if (error.message && error.message.includes('context length')) {
        console.warn('Context length exceeded, retrying with smaller context');
        return this.query(question, Math.floor(contextSize * 0.7)); // Reduce by 30%
      }

      // Track failed query metrics
      this.metrics.processingTimes[queryId] = Date.now() - startTime;

      throw error;
    }
  }

  /**
   * Get information about a specific type
   */
  public async getTypeInfo(typeName: string): Promise<SchemaChunk | null> {
    const chunk = this.chunkMap.get(typeName);
    return chunk || null;
  }

  /**
   * Get all type names in the schema
   */
  public getAllTypeNames(): string[] {
    return Array.from(this.chunkMap.keys()).filter((id) => !id.startsWith('_'));
  }

  /**
   * Get embedding and token usage metrics
   */
  public getMetrics(): any {
    const totalQueries = this.metrics.queriesPerformed;
    const totalTokensIn = Object.values(this.metrics.tokensIn).reduce((sum, val) => sum + val, 0);
    const totalTokensOut = Object.values(this.metrics.tokensOut).reduce((sum, val) => sum + val, 0);

    return {
      schema: {
        originalTypeCount: this.metrics.totalOriginalChunks,
        totalChunks: this.metrics.totalChunks,
        splitChunks: this.metrics.totalSplitChunks,
        estimatedTokensIndexed: Math.floor(this.metrics.estimatedTokensIndexed),
      },
      queries: {
        count: totalQueries,
        totalTokensIn,
        totalTokensOut,
        averageTokensIn: totalQueries > 0 ? Math.floor(totalTokensIn / totalQueries) : 0,
        averageTokensOut: totalQueries > 0 ? Math.floor(totalTokensOut / totalQueries) : 0,
        queryDetails: Object.entries(this.metrics.tokensIn).map(([queryId, tokensIn]) => ({
          queryId,
          tokensIn,
          tokensOut: this.metrics.tokensOut[queryId] || 0,
          processingTimeMs: this.metrics.processingTimes[queryId] || 0,
          chunks: this.metrics.chunks[queryId] || 0,
        })),
      },
      totalChunks: this.metrics.totalChunks,
    };
  }

  /**
   * Get a formatted string report of metrics
   */
  public getMetricsReport(): string {
    const metrics = this.getMetrics();

    let report = `## GraphQL Schema Agent Metrics\n\n`;

    report += `### Schema Metrics\n`;
    report += `- Original type definitions: ${metrics.schema.originalTypeCount}\n`;
    report += `- Total chunks after processing: ${metrics.schema.totalChunks}\n`;
    report += `- Split chunks (for large types): ${metrics.schema.splitChunks}\n`;
    report += `- Estimated tokens indexed: ${metrics.schema.estimatedTokensIndexed.toLocaleString()}\n\n`;

    report += `### Query Metrics\n`;
    report += `- Total queries performed: ${metrics.queries.count}\n`;
    report += `- Total tokens in: ${metrics.queries.totalTokensIn.toLocaleString()}\n`;
    report += `- Total tokens out: ${metrics.queries.totalTokensOut.toLocaleString()}\n`;
    report += `- Average tokens in per query: ${metrics.queries.averageTokensIn.toLocaleString()}\n`;
    report += `- Average tokens out per query: ${metrics.queries.averageTokensOut.toLocaleString()}\n\n`;

    report += `### Detailed Query Metrics\n`;
    report += `| Query | Tokens In | Tokens Out | Processing Time |\n`;
    report += `| ----- | --------- | ---------- | --------------- |\n`;

    metrics.queries.queryDetails.forEach((detail: any) => {
      report += `| ${detail.queryId} | ${detail.tokensIn.toLocaleString()} | ${detail.tokensOut.toLocaleString()} | ${(detail.processingTimeMs / 1000).toFixed(2)}s |\n`;
    });

    return report;
  }

  /**
   * Find paths between types
   */
  public async findConnectionsBetweenTypes(typeA: string, typeB: string): Promise<any> {
    // Build query for LLM to analyze connections
    const typeAChunk = this.chunkMap.get(typeA);
    const typeBChunk = this.chunkMap.get(typeB);

    if (!typeAChunk || !typeBChunk) {
      return { error: 'One or both types not found in schema' };
    }

    // Get related types to expand context
    const relatedToA = new Set<string>([typeA, ...typeAChunk.relatedTypes]);
    const relatedToB = new Set<string>([typeB, ...typeBChunk.relatedTypes]);

    // Find common related types
    const commonRelatedTypes = [...relatedToA].filter((type) => relatedToB.has(type));

    // Get chunks for all involved types
    const relevantChunks: SchemaChunk[] = [];

    for (const type of new Set([...relatedToA, ...relatedToB])) {
      const chunk = this.chunkMap.get(type);
      if (chunk) {
        relevantChunks.push(chunk);
      }
    }

    // Format context for LLM
    const context = this.formatChunksAsContext(relevantChunks);

    // Query LLM about the relationship
    const response = await this.llm.call([
      new SystemMessage(this.systemPrompt),
      new HumanMessage(`
        ${context}
        
        Question: Analyze the relationship between type "${typeA}" and type "${typeB}" in this GraphQL schema.
        How are they connected? Identify direct connections, connections through common fields, or connections through intermediary types.
        If there are multiple paths, identify the shortest or most direct path.
        
        Please answer based on the GraphQL schema chunks provided above.
      `),
    ]);

    return {
      typeA,
      typeB,
      commonRelatedTypes,
      analysis: response.content,
    };
  }
}

// types.ts file content
export enum ChunkType {
  QUERY = 'QUERY',
  MUTATION = 'MUTATION',
  SUBSCRIPTION = 'SUBSCRIPTION',
  OBJECT = 'OBJECT',
  INTERFACE = 'INTERFACE',
  INPUT = 'INPUT',
  ENUM = 'ENUM',
  UNION = 'UNION',
  SCALAR = 'SCALAR',
  DIRECTIVE = 'DIRECTIVE',
  OTHER = 'OTHER',
}

export interface SchemaChunk {
  id: string;
  type: ChunkType;
  content: string;
  relatedTypes: string[];
}
