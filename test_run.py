import time
from citationsim import CitationValidator

# Initialize the validator once with Vertex AI disabled
validator = CitationValidator(task_type="scientific")
validator.initialize_model()

# Sample summaries and citation batches
summaries = [
    "Transformers introduced self-attention for context-aware NLP models.",
    "ReLU activation helps prevent vanishing gradients in deep neural networks."
]

citation_batches = [
    [
        "Transformer models rely on self-attention to handle long-range dependencies.",
        "RNNs process sequences step by step without explicit attention mechanisms.",
        "CNNs are mainly used for image data."
    ],
    [
        "ReLU functions allow gradients to pass unimpeded for positive inputs.",
        "Sigmoid functions squash input into a small range, which can slow learning.",
        "Dropout prevents overfitting in deep neural networks."
    ]
]

# Initialize model with the first summary
t0 = time.time()
validator.initialize_model()
print(f"Model initialization time: {time.time() - t0:.4f} seconds\n")

# Batch processing
t1 = time.time()
for i, (summary, citations) in enumerate(zip(summaries, citation_batches)):
    print(f"--- Summary {i+1} ---")
    results = validator.validate(summary, citations)
    for r in results:
        status = "✅ Supported" if r["supported"] else "❌ Flagged"
        print(f"{status} | {r['score']:.3f} | {r['citation']}")
    print()
print(f"Total batch processing time: {time.time() - t1:.4f} seconds")
