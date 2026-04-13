MONLCD Algorithm Project 

1. Environment Requirements The implementation is based on the following configurations:

Python: 3.12 
PyTorch: 2.3.0 
CUDA: 12.1 
Core Libraries:
networkx==3.3 
numpy==1.26.4 
openai==1.55.3 
scikit-learn==1.5.2 
transformers==4.44.2
peft==0.12.0
datasets==2.21.0
pandas==2.2.2
langchain==0.2.14

2. File Descriptions

MONLCD.py: Implements the two-stage collaborative framework. The first stage uses PageRank to filter the top-k candidate nodes for LLM-based local community identification. The second stage executes global merging using a similarity-constrained modularity function.

Evaluation.py: Provides implementation for community detection assessment. It calculates Normalized Mutual Information (NMI) to evaluate overall partition quality and Pairwise F1 score to assess the precision of local community boundaries.

LLM_lora.py: A local inference wrapper for LLaMA 3.1 8B. It enables node-level community identification through natural language reasoning based on local topology.

LLM_train.py: Implements the self-supervised adaptation strategy. It allows the model to learn structural decision logic from graphs without manual labels.
