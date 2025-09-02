#!/usr/bin/env python3
"""
OpenSearch Embedding Speed Test

This script tests the performance of the OpenSearch neural sparse encoding model
by running embedding operations on examples from a JSON file and measuring timing.
"""

import json
import time
import statistics
import platform
import psutil
import random
from typing import List, Dict, Tuple, Optional
from pathlib import Path

from sentence_transformers.sparse_encoder import SparseEncoder


def get_system_info() -> Dict[str, str]:
    """Get system and CPU information."""
    cpu_info = {
        'platform': platform.platform(),
        'processor': platform.processor(),
        'architecture': platform.architecture()[0],
        'cpu_count': psutil.cpu_count(logical=True),
        'cpu_count_physical': psutil.cpu_count(logical=False),
        'memory_total_gb': round(psutil.virtual_memory().total / (1024**3), 2),
        'python_version': platform.python_version()
    }
    return cpu_info


def print_system_info():
    """Print system and CPU information."""
    info = get_system_info()
    print("\n💻 System Information")
    print("=" * 80)
    print(f"Platform: {info['platform']}")
    print(f"Processor: {info['processor']}")
    print(f"Architecture: {info['architecture']}")
    print(f"CPU Cores: {info['cpu_count']} logical, {info['cpu_count_physical']} physical")
    print(f"Memory: {info['memory_total_gb']} GB")
    print(f"Python: {info['python_version']}")


def generate_large_documents() -> List[str]:
    """Generate test documents of varying lengths."""
    base_texts = [
        "Machine learning is a subset of artificial intelligence that enables computers to learn and improve from experience without being explicitly programmed. It involves algorithms that can identify patterns in data and make predictions or decisions based on that information.",
        "Climate change refers to long-term shifts in global temperatures and weather patterns. While climate variations are natural, human activities have been the main driver of climate change since the 1800s, primarily through the burning of fossil fuels.",
        "Quantum computing harnesses the phenomena of quantum mechanics to process information in fundamentally different ways than classical computers. Instead of using bits that represent either 0 or 1, quantum computers use quantum bits or qubits.",
        "Renewable energy comes from natural sources that replenish themselves over time, such as solar, wind, hydroelectric, and geothermal power. These sources are considered sustainable alternatives to fossil fuels.",
        "Blockchain technology is a distributed ledger system that maintains a continuously growing list of records, called blocks, which are linked and secured using cryptography. Each block contains transaction data and timestamps."
    ]
    
    documents = []
    
    # Small documents (50-200 words)
    for base_text in base_texts[:3]:
        documents.append(base_text)
    
    # Medium documents (200-500 words)
    for base_text in base_texts:
        extended = base_text + " " + " ".join([base_text.split('.')[0] + f" This technology has applications in {field} and continues to evolve rapidly with new research and development efforts." 
                                              for field in ["healthcare", "finance", "education", "transportation", "communication"]])
        documents.append(extended)
    
    # Large documents (500-1500 words)
    for i, base_text in enumerate(base_texts):
        large_doc = base_text
        # Add multiple paragraphs with variations
        for j in range(8):
            variation = base_text.replace("technology", "innovation").replace("systems", "frameworks").replace("process", "methodology")
            large_doc += f" \n\nParagraph {j+2}: {variation} Furthermore, this field has seen significant advancements in recent years, with researchers and practitioners developing new approaches and methodologies. The implications of these developments are far-reaching and continue to shape our understanding of the domain."
        documents.append(large_doc)
    
    # Extra large documents (1500+ words)
    for base_text in base_texts[:2]:
        xl_doc = base_text
        for section in range(15):
            section_content = f"\n\nSection {section+2}: {base_text} This section explores additional aspects and considerations that are crucial for understanding the broader context and implications. "
            section_content += "The evolution of this field has been marked by significant milestones and breakthrough discoveries that have fundamentally changed how we approach problems and develop solutions. "
            section_content += "Research in this area continues to uncover new possibilities and applications that were previously thought impossible or impractical. "
            section_content += "The interdisciplinary nature of this work brings together experts from various fields, creating a rich collaborative environment that fosters innovation and creative problem-solving."
            xl_doc += section_content
        documents.append(xl_doc)
    
    return documents


def load_examples(file_path: str) -> List[Dict[str, str]]:
    """Load query-document examples from JSON file."""
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def create_extended_examples(original_examples: List[Dict[str, str]]) -> List[Dict[str, str]]:
    """Create extended examples with various document lengths."""
    extended = original_examples.copy()
    large_docs = generate_large_documents()
    
    # Create test cases with large documents
    large_doc_queries = [
        "What is machine learning and how does it work?",
        "Explain the causes and effects of climate change",
        "How does quantum computing differ from classical computing?",
        "What are the benefits of renewable energy sources?",
        "Describe blockchain technology and its applications",
        "What are the latest developments in artificial intelligence?",
        "How do sustainable technologies impact the environment?",
        "What are the challenges in implementing new technologies?"
    ]
    
    # Pair queries with large documents
    for i, doc in enumerate(large_docs):
        query = large_doc_queries[i % len(large_doc_queries)]
        extended.append({"query": query, "document": doc})
    
    return extended


def initialize_model() -> Tuple[SparseEncoder, float]:
    """Initialize the OpenSearch sparse encoder model and time the operation."""
    print("🔧 Initializing OpenSearch neural sparse encoder...")
    start_time = time.perf_counter()
    
    model = SparseEncoder(
        "opensearch-project/opensearch-neural-sparse-encoding-doc-v3-gte", 
        trust_remote_code=True, 
        model_kwargs={"code_revision": "40ced75c3017eb27626c9d4ea981bde21a2662f4"},
        device="cpu"  # Force CPU usage for compatibility
    )
    
    init_time = time.perf_counter() - start_time
    print(f"✅ Model initialized in {init_time:.3f} seconds (using CPU for compatibility)")
    return model, init_time


def categorize_document_length(doc: str) -> str:
    """Categorize document by character length."""
    length = len(doc)
    if length < 500:
        return "Small"
    elif length < 1500:
        return "Medium"
    elif length < 4000:
        return "Large"
    else:
        return "X-Large"


def run_embedding_tests(model: SparseEncoder, examples: List[Dict[str, str]]) -> Tuple[Dict[str, List[float]], List[Dict]]:
    """Run embedding tests and collect detailed timing and metadata."""
    print(f"\n🚀 Running embedding tests on {len(examples)} examples...")
    
    query_times = []
    document_times = []
    similarity_times = []
    total_times = []
    detailed_results = []
    
    for i, example in enumerate(examples, 1):
        query = example['query']
        document = example['document']
        
        doc_length = len(document)
        doc_words = len(document.split())
        doc_category = categorize_document_length(document)
        
        print(f"  Processing example {i}/{len(examples)}: '{query[:50]}{'...' if len(query) > 50 else ''}' "
              f"(Doc: {doc_category}, {doc_length} chars, {doc_words} words)")
        
        # Time the complete operation
        total_start = time.perf_counter()
        
        # Time query embedding
        query_start = time.perf_counter()
        query_embed = model.encode_query(query)
        query_time = time.perf_counter() - query_start
        query_times.append(query_time)
        
        # Time document embedding
        doc_start = time.perf_counter()
        document_embed = model.encode_document(document)
        doc_time = time.perf_counter() - doc_start
        document_times.append(doc_time)
        
        # Time similarity calculation
        sim_start = time.perf_counter()
        similarity = model.similarity(query_embed, document_embed)  # type: ignore
        sim_time = time.perf_counter() - sim_start
        similarity_times.append(sim_time)
        
        total_time = time.perf_counter() - total_start
        total_times.append(total_time)
        
        # Store detailed result
        detailed_results.append({
            'query': query,
            'document_length': doc_length,
            'document_words': doc_words,
            'document_category': doc_category,
            'query_time': query_time,
            'document_time': doc_time,
            'similarity_time': sim_time,
            'total_time': total_time,
            'similarity_score': similarity.item()
        })
        
        print(f"    Query: {query_time:.4f}s | Document: {doc_time:.4f}s | "
              f"Similarity: {sim_time:.4f}s | Total: {total_time:.4f}s | "
              f"Sim Score: {similarity.item():.4f}")
    
    return {
        'query': query_times,
        'document': document_times,
        'similarity': similarity_times,
        'total': total_times
    }, detailed_results


def run_batch_test(model: SparseEncoder, examples: List[Dict[str, str]]) -> Tuple[Dict[str, float], Dict[str, Dict]]:
    """Run batch embedding test with length-based analysis."""
    print(f"\n🔄 Running batch embedding test...")
    
    queries = [ex['query'] for ex in examples]
    documents = [ex['document'] for ex in examples]
    
    # Categorize documents for batch analysis
    doc_categories = [categorize_document_length(doc) for doc in documents]
    category_counts = {cat: doc_categories.count(cat) for cat in set(doc_categories)}
    
    print(f"  Document distribution: {category_counts}")
    
    # Batch query embeddings
    batch_query_start = time.perf_counter()
    query_embeds = [model.encode_query(q) for q in queries]
    batch_query_time = time.perf_counter() - batch_query_start
    
    # Batch document embeddings with category timing
    category_times = {cat: 0.0 for cat in set(doc_categories)}
    batch_doc_start = time.perf_counter()
    doc_embeds = []
    
    for i, (doc, cat) in enumerate(zip(documents, doc_categories)):
        cat_start = time.perf_counter()
        embed = model.encode_document(doc)
        cat_time = time.perf_counter() - cat_start
        category_times[cat] += cat_time
        doc_embeds.append(embed)
    
    batch_doc_time = time.perf_counter() - batch_doc_start
    
    # Batch similarities
    batch_sim_start = time.perf_counter()
    similarities = [model.similarity(q_emb, d_emb) for q_emb, d_emb in zip(query_embeds, doc_embeds)]  # type: ignore
    batch_sim_time = time.perf_counter() - batch_sim_start
    
    batch_total_time = batch_query_time + batch_doc_time + batch_sim_time
    
    print(f"  Batch query embedding: {batch_query_time:.4f}s")
    print(f"  Batch document embedding: {batch_doc_time:.4f}s") 
    print(f"  Batch similarity calculation: {batch_sim_time:.4f}s")
    print(f"  Batch total time: {batch_total_time:.4f}s")
    
    # Calculate per-category averages
    category_stats = {}
    for cat, count in category_counts.items():
        if count > 0:
            category_stats[cat] = {
                'count': count,
                'total_time': category_times[cat],
                'avg_time': category_times[cat] / count
            }
            print(f"  {cat} documents ({count}): {category_times[cat]:.4f}s total, {category_stats[cat]['avg_time']:.4f}s avg")
    
    return {
        'query': batch_query_time,
        'document': batch_doc_time,
        'similarity': batch_sim_time,
        'total': batch_total_time
    }, category_stats


def calculate_correlation(x_values: List[float], y_values: List[float]) -> float:
    """Calculate Pearson correlation coefficient."""
    if len(x_values) != len(y_values) or len(x_values) < 2:
        return 0.0
    
    n = len(x_values)
    sum_x = sum(x_values)
    sum_y = sum(y_values)
    sum_xy = sum(x * y for x, y in zip(x_values, y_values))
    sum_x2 = sum(x * x for x in x_values)
    sum_y2 = sum(y * y for y in y_values)
    
    numerator = n * sum_xy - sum_x * sum_y
    denominator = ((n * sum_x2 - sum_x * sum_x) * (n * sum_y2 - sum_y * sum_y)) ** 0.5
    
    return numerator / denominator if denominator != 0 else 0.0


def print_detailed_statistics(timing_data: Dict[str, List[float]], batch_data: Dict[str, float], 
                            detailed_results: List[Dict], category_stats: Dict[str, Dict], num_examples: int):
    """Print comprehensive timing statistics with variance analysis."""
    print(f"\n📊 Detailed Performance Statistics ({num_examples} examples)")
    print("=" * 100)
    
    operations = ['query', 'document', 'similarity', 'total']
    
    # Overall statistics
    for operation in operations:
        times = timing_data[operation]
        mean_time = statistics.mean(times)
        median_time = statistics.median(times)
        std_dev = statistics.stdev(times) if len(times) > 1 else 0
        min_time = min(times)
        max_time = max(times)
        variance = statistics.variance(times) if len(times) > 1 else 0
        coeff_var = (std_dev / mean_time * 100) if mean_time > 0 else 0
        
        print(f"\n{operation.upper()} EMBEDDING ANALYSIS:")
        print(f"  • Individual Times - Mean: {mean_time:.4f}s | Median: {median_time:.4f}s")
        print(f"  • Variability - Min: {min_time:.4f}s | Max: {max_time:.4f}s | Range: {max_time-min_time:.4f}s")
        print(f"  • Spread - Std Dev: {std_dev:.4f}s | Variance: {variance:.6f} | Coeff of Variation: {coeff_var:.1f}%")
        print(f"  • Batch Performance - Total: {batch_data[operation]:.4f}s | Per item: {batch_data[operation]/num_examples:.4f}s")
        print(f"  • Throughput - Individual: {num_examples/mean_time:.2f} items/sec | Batch: {num_examples/batch_data[operation]:.2f} items/sec")
        
        # Efficiency comparison
        efficiency_gain = ((mean_time * num_examples) - batch_data[operation]) / (mean_time * num_examples) * 100
        print(f"  • Batch Efficiency: {efficiency_gain:.1f}% faster than individual processing")
    
    # Document length analysis
    print(f"\n\n📜 DOCUMENT LENGTH IMPACT ANALYSIS")
    print("=" * 100)
    
    # Group by document category
    categories = {}
    for result in detailed_results:
        cat = result['document_category']
        if cat not in categories:
            categories[cat] = {'lengths': [], 'doc_times': [], 'total_times': [], 'similarities': []}
        
        categories[cat]['lengths'].append(result['document_length'])
        categories[cat]['doc_times'].append(result['document_time'])
        categories[cat]['total_times'].append(result['total_time'])
        categories[cat]['similarities'].append(result['similarity_score'])
    
    for category, data in categories.items():
        count = len(data['lengths'])
        avg_length = statistics.mean(data['lengths'])
        avg_doc_time = statistics.mean(data['doc_times'])
        avg_total_time = statistics.mean(data['total_times'])
        std_doc_time = statistics.stdev(data['doc_times']) if count > 1 else 0
        avg_similarity = statistics.mean(data['similarities'])
        
        print(f"\n{category.upper()} Documents ({count} samples):")
        print(f"  • Average length: {avg_length:.0f} characters ({avg_length/100:.1f} hundred chars)")
        print(f"  • Document embedding: {avg_doc_time:.4f}s ± {std_doc_time:.4f}s")
        print(f"  • Total processing: {avg_total_time:.4f}s")
        print(f"  • Average similarity: {avg_similarity:.4f}")
        print(f"  • Processing rate: {avg_length/avg_doc_time:.0f} chars/sec")
    
    # Correlation analysis
    print(f"\n\n🔍 CORRELATION ANALYSIS")
    print("=" * 100)
    
    lengths = [r['document_length'] for r in detailed_results]
    doc_times = [r['document_time'] for r in detailed_results]
    total_times = [r['total_time'] for r in detailed_results]
    similarities = [r['similarity_score'] for r in detailed_results]
    
    length_doc_corr = calculate_correlation(lengths, doc_times)
    length_total_corr = calculate_correlation(lengths, total_times)
    length_sim_corr = calculate_correlation(lengths, similarities)
    
    print(f"Document Length vs Processing Time Correlations:")
    print(f"  • Length ↔ Document Embedding Time: {length_doc_corr:.3f} {'(Strong)' if abs(length_doc_corr) > 0.7 else '(Moderate)' if abs(length_doc_corr) > 0.3 else '(Weak)'}")
    print(f"  • Length ↔ Total Processing Time: {length_total_corr:.3f} {'(Strong)' if abs(length_total_corr) > 0.7 else '(Moderate)' if abs(length_total_corr) > 0.3 else '(Weak)'}")
    print(f"  • Length ↔ Similarity Score: {length_sim_corr:.3f} {'(Strong)' if abs(length_sim_corr) > 0.7 else '(Moderate)' if abs(length_sim_corr) > 0.3 else '(Weak)'}")
    
    # Performance variance insights
    print(f"\n\n⚡ PERFORMANCE VARIANCE INSIGHTS")
    print("=" * 100)
    
    doc_times_overall = timing_data['document']
    cv_doc = (statistics.stdev(doc_times_overall) / statistics.mean(doc_times_overall)) * 100
    
    print(f"Document Embedding Variance Analysis:")
    print(f"  • Overall coefficient of variation: {cv_doc:.1f}%")
    
    if cv_doc > 30:
        print(f"  • HIGH variance detected - document length significantly impacts processing time")
    elif cv_doc > 15:
        print(f"  • MODERATE variance - some performance variation based on document characteristics")
    else:
        print(f"  • LOW variance - consistent processing times across different document sizes")
    
    # Find outliers
    mean_doc_time = statistics.mean(doc_times_overall)
    std_doc_time = statistics.stdev(doc_times_overall) if len(doc_times_overall) > 1 else 0
    outliers = [(i, t) for i, t in enumerate(doc_times_overall) if abs(t - mean_doc_time) > 2 * std_doc_time]
    
    if outliers and std_doc_time > 0:
        print(f"  • Performance outliers detected: {len(outliers)} examples beyond 2 standard deviations")
        for idx, time_val in outliers[:3]:  # Show first 3 outliers
            example = detailed_results[idx]
            print(f"    - Example {idx+1}: {time_val:.4f}s ({example['document_category']}, {example['document_length']} chars)")


def demonstrate_embeddings(model: SparseEncoder, examples: List[Dict[str, str]]):
    """Demonstrate embedding decoding with detailed analysis."""
    if not examples:
        return
        
    example = examples[0]
    query = example['query']
    document = example['document']
    doc_category = categorize_document_length(document)
    doc_length = len(document)
    
    print(f"Query: {query}")
    print(f"Document ({doc_category}, {doc_length} chars): {document[:200]}{'...' if len(document) > 200 else ''}")
    
    # Time the embedding operations for this demonstration
    start_time = time.perf_counter()
    query_embed = model.encode_query(query)
    query_time = time.perf_counter() - start_time
    
    start_time = time.perf_counter()
    document_embed = model.encode_document(document)
    doc_time = time.perf_counter() - start_time
    
    start_time = time.perf_counter()
    similarity = model.similarity(query_embed, document_embed)  # type: ignore
    sim_time = time.perf_counter() - start_time
    
    print(f"\nProcessing Times:")
    print(f"  • Query embedding: {query_time:.4f}s")
    print(f"  • Document embedding: {doc_time:.4f}s ({doc_length/doc_time:.0f} chars/sec)")
    print(f"  • Similarity calculation: {sim_time:.4f}s")
    print(f"  • Similarity score: {similarity.item():.4f}")
    
    decoded_query = model.decode(query_embed)  # type: ignore
    decoded_document = model.decode(document_embed)  # type: ignore
    

    # Show embedding statistics
    query_tokens = len(decoded_query)
    doc_tokens = len(decoded_document)
    matching_tokens = sum(1 for q_token, _ in decoded_query if any(q_token == d_token for d_token, _ in decoded_document))
    
    print(f"\nEmbedding Statistics:")
    print(f"  • Query tokens: {query_tokens}")
    print(f"  • Document tokens: {doc_tokens}")
    print(f"  • Matching tokens: {matching_tokens}/{query_tokens} ({matching_tokens/query_tokens*100:.1f}% overlap)")


def main():
    """Main execution function."""
    print("🔬 OpenSearch Embedding Speed Test with Large Document Analysis")
    print("=" * 100)
    
    # Show system information
    print_system_info()
    
    # Load original examples
    examples_file = Path("examples.json")
    if not examples_file.exists():
        print(f"❌ Error: {examples_file} not found!")
        return
    
    original_examples = load_examples(str(examples_file))
    print(f"\n📁 Loaded {len(original_examples)} original examples from {examples_file}")
    
    # Create extended examples with large documents
    extended_examples = create_extended_examples(original_examples)
    print(f"📈 Created {len(extended_examples)} total examples (including {len(extended_examples) - len(original_examples)} large document tests)")
    
    # Show document distribution
    doc_categories = [categorize_document_length(ex['document']) for ex in extended_examples]
    category_counts = {cat: doc_categories.count(cat) for cat in set(doc_categories)}
    print(f"\nDocument size distribution: {category_counts}")
    
    # Initialize model
    model, init_time = initialize_model()
    
    # Run individual tests
    timing_data, detailed_results = run_embedding_tests(model, extended_examples)
    
    # Run batch test
    batch_data, category_stats = run_batch_test(model, extended_examples)
    
    # Print comprehensive statistics
    print_detailed_statistics(timing_data, batch_data, detailed_results, category_stats, len(extended_examples))
    
    # Demonstrate embeddings on a large document example
    large_doc_examples = [ex for ex in extended_examples if categorize_document_length(ex['document']) in ['Large', 'X-Large']]
    if large_doc_examples:
        print(f"\n\n🔍 LARGE DOCUMENT EMBEDDING ANALYSIS")
        print("=" * 100)
        demonstrate_embeddings(model, [large_doc_examples[0]])
    
    print(f"\n🎉 Comprehensive speed test completed!")
    total_individual_time = sum(timing_data['total'])
    total_batch_time = batch_data['total']
    print(f"\n⏱️  TIMING SUMMARY:")
    print(f"  • Model initialization: {init_time:.3f}s")
    print(f"  • Individual processing: {total_individual_time:.3f}s")
    print(f"  • Batch processing: {total_batch_time:.3f}s")
    print(f"  • Total runtime: {total_individual_time + total_batch_time + init_time:.3f}s")
    print(f"  • Batch efficiency gain: {((total_individual_time - total_batch_time) / total_individual_time * 100):.1f}%")


if __name__ == "__main__":
    # Add psutil requirement check
    try:
        import psutil
    except ImportError:
        print("❌ Error: psutil package is required for system information.")
        print("Install it with: pip install psutil")
        exit(1)
    
    main()
