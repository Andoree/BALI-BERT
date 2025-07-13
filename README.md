# BALI-BERT


# BALI: Enhancing Biomedical Language Representations

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.11%2B-orange)](https://pytorch.org)

This is the repository for the [BALI: Enhancing Biomedical Language Representations through Knowledge Graph and Language Model Alignment](https://doi.org/10.1145/3726302.3729901) accepted to [SIGIR 2025](https://sigir2025.dei.unipd.it/accepted-papers.html). The paper proposes a novel joint pre-training method that enhances biomedical language models with the information UMLS large biomedical Knowledge Graph (KG) through text-KG representation alignment.

> 📄 **Paper**: [BALI: Enhancing Biomedical Language Representations through Knowledge Graph and Language Model Alignment](https://doi.org/10.1145/3726302.3729901)  
> 🗓️ **SIGIR 2025**


## Installation
```bash
git clone https://github.com/Andoree/BALI-BERT.git
cd BALI-BERT

# Create conda environment
conda create -n bali python=3.8
conda activate bali

# Install dependencies
pip install -r requirements.txt

## Available HuggingFace Checkpoiints

| Model | Description |
|-------|-------------|
| [`andorei/BALI-BERT-BioLinkBERT-large-lingraph`](https://huggingface.co/andorei/BALI-BERT-BioLinkBERT-large-lingraph) | BioLinkBERT-large model pre-trained with BALI using linear graph encoder |
| [`andorei/BALI-BERT-BioLinkBERT-base-GNN`](https://huggingface.co/andorei/BALI-BERT-BioLinkBERT-base-GNN) | BioLinkBERT-base model pre-trained with BALI using GNN-based alignment |
| [`andorei/BALI-BERT-PubMedBERT-GNN`](https://huggingface.co/andorei/BALI-BERT-PubMedBERT-GNN) | PubMedBERT model pre-trained with BALI using GNN-based alignment |

## 📚 Citation

If you use our model in your research, please cite our SIGIR 2025 paper:

```bibtex
@inproceedings{Sakhovskiy2025BALI,
  author = {Sakhovskiy, Andrey and Tutubalina, Elena},
  title = {BALI: Enhancing Biomedical Language Representations through Knowledge Graph and Language Model Alignment},
  booktitle = {Proceedings of the 48th International ACM SIGIR Conference on Research and Development in Information Retrieval (SIGIR '25)},
  year = {2025}
}
```
