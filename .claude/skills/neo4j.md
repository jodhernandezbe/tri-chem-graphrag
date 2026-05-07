# Neo4j Skill

## Graph Database Best Practices
- Use parameterized queries to prevent injection attacks
- Batch operations for bulk inserts (UNWIND with 500-1000 records)
- Connection pooling to manage database connections efficiently
- Index creation for frequently queried properties

## Cypher Query Optimization
- Use EXPLAIN and PROFILE to analyze query performance
- Leverage indexes for node lookups
- Minimize cartesian products
- Use appropriate relationship directions

## Graph Modeling
- Design schema based on query patterns
- Normalize vs denormalize based on access patterns
- Use relationship properties for weighted connections
- Consider graph algorithms for complex analysis

## Performance Patterns
```cypher
// Batch insert pattern
UNWIND $batch_data AS row
MERGE (n:NodeType {id: row.id})
SET n += row.properties
```

## Transaction Management
- Use explicit transactions for multi-step operations
- Keep transactions short to avoid locks
- Handle transaction failures gracefully
- Use read/write transaction separation
