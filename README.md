# Bug Report Classification for Deep Learning Frameworks

This project implements and evaluates multiple classification models to detect performance-related bug reports in deep learning frameworks such as TensorFlow, PyTorch, Keras, and Caffe.

## Quick Start

1. Check `requirements.pdf` for installation instructions
2. Check `manual.pdf` for usage instructions and commands
3. Check `replication.pdf` to see how to replicate experiments and interpret results

## What's Included

- **Three Classification Models**:
  - Baseline Model (Naive Bayes + TF-IDF)
  - Intermediate Model (SVM + Word2Vec)
  - Hybrid Model (Domain-specific ensemble classifier) (My main tool)

- **Documentation**:
  - `requirements.pdf`: Installation and project structure
  - `manual.pdf`: User manual with commands and features
  - `replication.pdf`: Step-by-step instructions for reproducing experiments

- **Key Files** (in `lab1/` directory):
  - Model implementations (baseline_model.py, intermediate_model.py, hybrid_model.py)
  - Evaluation scripts (test_*_model.py)

## Project Structure

The core implementation is in the `lab1/` directory, with key outputs saved to:
- `lab1/results/`: Raw evaluation results and summary report
- `lab1/plots/`: Visualisations comparing model performance
- `lab1/comprehensive_results/`: Detailed evaluation output by framework

## For More Information

- For installation details, see `requirements.pdf`
- For usage instructions and commands, see `manual.pdf`
- For replicating experiments and interpreting results, see `replication.pdf`