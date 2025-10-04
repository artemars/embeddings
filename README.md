# Engineering Embedding Spaces for Financial Sentiment Regression  

This project introduces a contrastive learning framework for financial sentiment regression that re-maps sentence embeddings into a sentiment-aware space adjusted for labelling bias. The method is simple, interpretable, and generalises to analogous tasks across domains.  

## Key Highlights  
- Developed a novel approach involving contrastive embedding re-mapping using cosine similarity loss on a similarity proxy, reducing embedding anisotropy. For sentiment labels $s_n, s_m \in [-1, 1]$, the proxy similarity is defined as $1 - |s_n - s_m|$.  
- Achieved SOTA **MSE = 0.0483, R² = 0.7073** on the FiQA 2018 Task 1 dataset under hold-out validation, outperforming prior methods.  
- Applied a bespoke two-stage training pipeline: trained on MSE over cosine similarity and tuned using Pearson cosine score, then tuned MLP regressors on a separate held-out validation set. This reduced average embedding similarity by 41%, yielding a more structured sentiment space while preserving test performance.  

## Abstract
Financial sentiment analysis (FSA) is a common upstream machine learning task, formulated as both a regression and classification problem. Current state-of-the-art (SOTA) models on the former leverage contextual embeddings from pre-trained large language models (LLMs), while integrating domain knowledge from financial lexicons. These methods fail to address the non-uniform (anisotropic) nature of LLM embedding spaces, ignoring powerful partial regularisation techniques through contrastive learning. We reframe sentence embedding learning as a contrastive learning problem, using cosine similarity loss to map an embedding space that reflects sentiment similarity—a process we call "fanning." These engineered embeddings are used for regression through dense neural layers. To achieve this, we construct a dataset of randomly sampled sentence pairs, define a proxy for cosine similarity, and perform two-stage hold-out validation: first for embedding training, then for dense layer regression. Our method achieves SOTA performance on the FiQA 2018 Task 1 dataset, with an average MSE of 0.0483 and an average R² of 0.7073, demonstrating the effectiveness of embedding space fanning as a feature engineering technique for FSA.

## Results Snapshot  
| Model         | MSE (↓)   | R² (↑)   |
|---------------|-----------|----------|
| RoBERTa       | 0.0558    | 0.6693   |
| k-RoBERTa     | 0.0490    | **0.7110**   |
| **Our model** | **0.0483** | 0.7073   |

**Note:** The slightly higher R² of k-RoBERTa is due to the test labels being closer to the test mean on average. This means a greater reduction in MSE is required to achieve the same proportional gain in variance explained.  
