# OpenSearch Embedding Speed Test with Large Document Analysis

A comprehensive Python benchmarking tool that tests OpenSearch neural sparse encoding performance across documents of varying sizes, providing detailed statistical analysis and system performance insights.

## Features

- ⚡ **Individual & Batch Processing**: Compare single vs batch embedding performance
- 📊 **Advanced Statistics**: Mean, median, std dev, variance, coefficient of variation, outlier detection
- 📏 **Document Length Analysis**: Automatic categorization (Small, Medium, Large, X-Large) with performance correlation
- 💻 **System Information**: CPU, memory, platform details for reproducible benchmarking  
- 📈 **Throughput Analysis**: Items/second and characters/second processing rates
- 🔗 **Correlation Analysis**: Document length vs processing time relationships
- 🎯 **Similarity Scoring**: Query-document relevance with embedding quality metrics
- ⚡ **Performance Insights**: Variance analysis with HIGH/MODERATE/LOW classifications
- 🚀 **Batch Efficiency**: Quantified performance gains from batch processing

## Setup

This project uses `uv` for fast Python package management.

### Prerequisites

- Python 3.8 or higher
- [uv](https://github.com/astral-sh/uv) package manager

### Installation

1. Clone or navigate to the project directory:
```bash
cd opensearch-embed
```

2. Install dependencies with uv:
```bash
uv pip install -e .
```

**Note**: The project uses CPU-only PyTorch to avoid compatibility issues. If you encounter bus errors during model initialization, install CPU-only PyTorch explicitly:
```bash
uv pip install torch --index-url https://download.pytorch.org/whl/cpu --force-reinstall
```

## Usage

### Basic Speed Test

Run the speed test with the provided examples:

```bash
uv run python speed_test.py
```

### Adding More Examples

Edit `examples.json` to add more query-document pairs:

```json
[
  {
    "query": "Your search query here",
    "document": "The document content to match against"
  }
]
```

## Output

The script provides comprehensive performance analysis across multiple dimensions:

### System Information
- Platform, processor, and architecture details
- CPU core count (physical and logical)
- Total system memory
- Python version for reproducible benchmarking

### Automatic Test Generation
- Creates documents of varying sizes: Small (<500 chars), Medium (500-1500), Large (1500-4000), X-Large (4000+)
- Generates test cases across all document size categories
- Shows document size distribution before testing

### Individual Operation Analysis
- Per-example timing: Query embedding, document embedding, similarity calculation
- Document categorization with length and word counts
- Real-time processing rates (characters/second)

### Comprehensive Statistics
- **Central Tendency**: Mean, median for all operations
- **Variability**: Standard deviation, variance, coefficient of variation, min/max ranges  
- **Efficiency**: Batch vs individual processing comparison with percentage gains
- **Throughput**: Items/second and characters/second processing rates

### Document Length Impact Analysis
- Performance breakdown by document size category
- Average processing times per size class
- Processing rate analysis (chars/sec by category)
- Correlation coefficients between document length and processing time

### Performance Variance Insights  
- Coefficient of variation analysis (HIGH/MODERATE/LOW classifications)
- Outlier detection for performance anomalies
- Variance explanations and recommendations

### Embedding Quality Metrics
- Token count analysis (query vs document)
- Token overlap percentages
- Similarity score distributions

## Example Output

```
🔬 OpenSearch Embedding Speed Test with Large Document Analysis
====================================================================================================

💻 System Information
====================================================================================================
Platform: Linux-6.8.0-78-generic-x86_64-with-glibc2.39
Processor: x86_64
Architecture: 64bit
CPU Cores: 64 logical, 32 physical
Memory: 314.55 GB
Python: 3.12.3

📁 Loaded 10 original examples from examples.json
📈 Created 25 total examples (including 15 large document tests)

Document size distribution: {'X-Large': 7, 'Small': 13, 'Large': 4, 'Medium': 1}

🔧 Initializing OpenSearch neural sparse encoder...
✅ Model initialized in 3.517 seconds (using CPU for compatibility)

🚀 Running embedding tests on 25 examples...
  Processing example 1/25: 'What's the weather in ny now?' (Doc: Small, 28 chars, 5 words)
    Query: 0.0125s | Document: 0.0599s | Similarity: 0.0006s | Total: 0.0730s | Sim Score: 12.5747

📊 Detailed Performance Statistics (25 examples)
====================================================================================================

DOCUMENT EMBEDDING ANALYSIS:
  • Individual Times - Mean: 0.2066s | Median: 0.0760s
  • Variability - Min: 0.0544s | Max: 1.1326s | Range: 1.0782s
  • Spread - Std Dev: 0.2759s | Variance: 0.076101 | Coeff of Variation: 133.5%
  • Batch Performance - Total: 5.0703s | Per item: 0.2028s
  • Throughput - Individual: 121.02 items/sec | Batch: 4.93 items/sec
  • Batch Efficiency: 1.8% faster than individual processing

📏 DOCUMENT LENGTH IMPACT ANALYSIS
====================================================================================================

SMALL Documents (13 samples):
  • Average length: 155 characters (1.6 hundred chars)
  • Document embedding: 0.0648s ± 0.0059s
  • Total processing: 0.0683s
  • Average similarity: 15.0633
  • Processing rate: 2397 chars/sec

X-LARGE Documents (7 samples):
  • Average length: 7093 characters (70.9 hundred chars)
  • Document embedding: 0.5226s ± 0.3737s
  • Total processing: 0.5253s
  • Average similarity: 9.9524
  • Processing rate: 13573 chars/sec

🔍 CORRELATION ANALYSIS
====================================================================================================
Document Length vs Processing Time Correlations:
  • Length ↔ Document Embedding Time: 0.993 (Strong)
  • Length ↔ Total Processing Time: 0.993 (Strong)
  • Length ↔ Similarity Score: -0.418 (Moderate)

⚡ PERFORMANCE VARIANCE INSIGHTS
====================================================================================================
Document Embedding Variance Analysis:
  • Overall coefficient of variation: 133.5%
  • HIGH variance detected - document length significantly impacts processing time
  • Performance outliers detected: 2 examples beyond 2 standard deviations
    - Example 24: 1.1326s (X-Large, 13973 chars)
    - Example 25: 1.0007s (X-Large, 13621 chars)
```

## Model Details

This benchmark uses the OpenSearch neural sparse encoding model:
- **Model**: `opensearch-project/opensearch-neural-sparse-encoding-doc-v3-gte`
- **Type**: Sparse encoder for neural search
- **Revision**: `40ced75c3017eb27626c9d4ea981bde21a2662f4`

## Performance Notes

- **Document Size Impact**: Very strong correlation (>0.99) between document length and processing time
- **Batch Efficiency**: Batch processing shows marginal improvements over individual processing
- **Variance Analysis**: Coefficient of variation >100% indicates very high performance variance based on document characteristics
- **Processing Rates**: Larger documents achieve higher chars/sec throughput due to model efficiency at scale
- **CPU-Only**: Uses CPU-only PyTorch to avoid memory alignment issues and ensure compatibility across systems
- **Model Size**: First-time download may take several minutes (550MB model)
- **System Info**: All benchmarks include system specifications for reproducible results
- **Bus Error Fix**: If encountering bus errors, install CPU-only PyTorch with the provided installation command

## Document Size Categories

The test automatically categorizes documents and generates comprehensive test data:

- **Small**: <500 characters (typical queries, short descriptions)  
- **Medium**: 500-1,500 characters (paragraphs, article excerpts)
- **Large**: 1,500-4,000 characters (full articles, documentation)
- **X-Large**: 4,000+ characters (long-form content, research papers)

## Understanding the Statistics

### Variance Classifications
- **LOW (<15%)**: Consistent performance across document sizes
- **MODERATE (15-30%)**: Some variation, manageable performance differences  
- **HIGH (>30%)**: Significant variation, document length strongly affects performance

### Correlation Strength
- **Strong (>0.7)**: Document length is a primary performance factor
- **Moderate (0.3-0.7)**: Document length has noticeable impact
- **Weak (<0.3)**: Document length has minimal performance impact

## Customization

### Adding New Test Cases

Extend `examples.json` with domain-specific query-document pairs. The script will automatically:
- Generate additional large document test cases
- Categorize all documents by size
- Provide size-specific performance analysis

### Advanced Configuration

Edit `speed_test.py` to:
- Modify document size thresholds for categorization
- Adjust correlation analysis parameters  
- Add custom performance metrics
- Change statistical analysis depth
- Customize output formatting

## Dependencies

- `sentence-transformers`: OpenSearch sparse encoder model
- `torch`: PyTorch backend for model inference (CPU-only version recommended)
- `numpy`: Numerical operations and statistics
- `psutil`: System information collection (CPU, memory, platform details)
- `scipy`: Scientific computing library for advanced mathematical functions

## License

This project is provided as-is for benchmarking purposes.
