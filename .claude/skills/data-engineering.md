# Data Engineering Skill

## ETL Pipeline Patterns
- Extract large datasets with streaming/chunking (pandas.read_csv(chunksize=10000))
- Transform with validation and error handling at each stage
- Load with batch operations to optimize database performance

## Large Dataset Processing
- Use pandas chunking for files >1GB
- Implement memory-efficient processing patterns
- Monitor memory usage during transformations
- Use generators for iterative processing

## Data Validation
- Validate data types and ranges early in pipeline
- Handle missing values explicitly
- Check for duplicates and data quality issues
- Log validation results for debugging

## Performance Optimization
- Vectorized operations over loops
- Efficient joins and merges
- Index usage for large DataFrames
- Parallel processing where appropriate

## Error Handling
- Graceful degradation for partial failures
- Detailed logging for data pipeline monitoring
- Retry logic for transient failures
- Data lineage tracking for debugging
