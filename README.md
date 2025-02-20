# LoRA-Adapted RoBERTa for Portuguese News Classification

## Overview

This project fine-tunes a pre-trained English RoBERTa model to classify news articles in Portuguese using Low-Rank Adaptation (LoRA). LoRA enables efficient adaptation of large language models to new languages and tasks with minimal computational resources.

## Background

This implementation is based on the tutorial [Lightweight RoBERTa Sequence Classification Fine-tuning with LoRA](https://achimoraites.medium.com/lightweight-roberta-sequence-classification-fine-tuning-with-lora-using-the-hugging-face-peft-8dd9edf99d19), which demonstrates LoRA-based fine-tuning for sequence classification in English. I've adapted this approach to cross-lingual transfer learning from English to Portuguese.

## Dataset

I used the [LIACC/Emakhuwa-Portuguese-News-MT](https://huggingface.co/datasets/LIACC/Emakhuwa-Portuguese-News-MT) dataset from Hugging Face, which contains Portuguese news articles with their corresponding categories.

### **Dataset Preprocessing Steps**
The original dataset consisted of three splits: **train (17.4k samples), validation (964 samples), and test (993 samples)**. However, a detailed analysis revealed several issues:

1. **Label Mismatch Across Splits**: Some categories in the validation and test sets were missing in the training set, making evaluation unreliable.
2. **Overlapping Categories**: Certain labels were highly ambiguous, even for native speakers. For example, the sentence: 
   - *"Castro voltou a exigir que os Estados Unidos eliminem o embargo contra Cuba em vigor há 53 anos."* ("Castro once again demanded that the United States eliminate the embargo against Cuba that has been in place for 53 years.")
   - This could be easily categorized as **"política" (politics)** or **"economia" (economy)** by a human, but was actually labeled as **"mundo" (world)**.
3. **Uneven Split Proportions**: The original dataset's test and validation sizes were very small relative to the training set.

### **Solution & Final Dataset**
Since the main focus of this project is to explore LoRA's capabilities for cross-lingual transfer learning from English to Portuguese rather than evaluating RoBERTa's classification performance, I also aimed to create a more balanced and meaningful dataset by performing the following preprocessing steps:

- **Filtered the dataset to retain only five core categories**: `'cultura', 'desporto', 'economia', 'mundo', 'saude'`.
- **Merged all original splits** into a single dataset (19.3K samples → 14.3K after filtering).
- **Created a new shuffled split**: 
  - **Train**: 11,478 samples (80%)
  - **Validation**: 1,435 samples (10%)
  - **Test**: 1,435 samples (10%)
- **Applied tokenization** using a RoBERTa tokenizer to preprocess the text data before model training.

This restructuring allows for more reliable model evaluation and ensures that all labels are present in every split.

## Training Setup & Experiments

- **Base Model**: RoBERTa (English pre-trained)
- **Adaptation Method**: Low-Rank Adaptation (LoRA)

### Using local machine
- **LoRA Rank**: 8 (very lightweight adaptation)
- **Trainable Parameters**: 889,349 (0.71% of total 125,538,826)
- **Batch Size**: 4
- **Training Epochs**: 1 (first experiment), 3 (second experiment)
- **Hardware**: NVIDIA GeForce 940MX
- **Training Time**: ~3 hours (n_epochs = 1), ~10 hours (n_epochs = 3)

### Using Google Colab Pro
- **LoRA Rank**: 64
- **Trainable Parameters**: 2,953,733 (2.31% of total 127,603,210)
- **Batch Size**: 16
- **Training Epochs**: 20
- **Hardware**: NVIDIA A100
- **Training Time**: ~1 hour

## Results  
The LoRA-adapted model with a rank of **8** achieved an accuracy of **0.65** on the test set after a single epoch and surpassed **0.70** after three epochs, significantly outperforming the original model’s **0.19** accuracy.  

A larger LoRA adaptation with a rank of **64** further improved performance, reaching **0.74** accuracy after 20 epochs.

## Analysis & Conclusion  

These results are quite impressive given the constraints of this experiment. The LoRA rank used was very low (**r=8**, meaning only **0.71% of the original model size**), and the training dataset was relatively small (**under 12K samples**). In contrast, the dataset used in the original tutorial (`ag_news`) was in English (the original language used for training RoBERTa) and had a training split of **120K samples** — more than **10 times larger** than the one used here.  

Even when increasing the LoRA rank to **64** and training for **20 epochs**, the improvement was marginal (**0.74** accuracy vs. **0.70** with **r=8** and just 3 epochs). This suggests that the limited size of the training dataset, rather than model capacity, is likely the main bottleneck.  

These findings highlight the potential for further improvement by building a larger, high-quality training dataset in Portuguese, which could better leverage LoRA’s efficiency while further enhancing model performance.  

## Acknowledgements

- Hugging Face for the PEFT library and dataset hosting
- The original tutorial author for the LoRA implementation guidance
- LIACC for providing the Portuguese news dataset

## License

This project is licensed under the MIT License - see the LICENSE file for details.

