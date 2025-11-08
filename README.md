# CRISPRight: Off-Target Prediction in CRISPR-Cas9 Using Deep Learning

## 🧬 Overview

**CRISPRight** is a machine learning and deep learning-based project aimed at predicting **off-target effects** in the CRISPR-Cas9 gene editing system. The project evaluates multiple deep learning architectures (CNN and LSTM) and encoding techniques (One-Hot and K-mer) to determine which combination yields the most accurate prediction results.

<img width="1024" height="608" alt="image" src="https://github.com/user-attachments/assets/03c179de-7647-44a0-8654-4d287ae32e22" />

The dataset used in this study was sourced from the [CRISPRoffT website]([https://rth.dk/resources/crispr/](https://ccsm.uth.edu/CRISPRoffT/)), containing guide RNA sequences, their respective DNA targets, and PAM motifs labeled as either **ON-target** (successful cut) or **OFF-target** (unintended cut).

---

## 🎯 Objective

To compare the performance of different deep learning models — CNN and LSTM — using both One-Hot and K-mer encodings for CRISPR-Cas9 off-target prediction, and determine the most effective architecture and encoding strategy.

---

## 📂 Dataset Description

* **Source:** [CRISPRoffT Dataset]([https://rth.dk/resources/crispr/](https://ccsm.uth.edu/CRISPRoffT/))
* **Attributes:**

  * `Guide_sequence` – The gRNA sequence.
  * `Target_sequence` – The DNA target site.
  * `PAM` – Protospacer Adjacent Motif required for Cas9 recognition.
  * `Identity` – Label specifying ON-target or OFF-target.

During preprocessing, sequences were validated for correct nucleotide composition (A, T, G, C) and standardized in length. The dataset was balanced using resampling techniques to mitigate class imbalance between ON and OFF labels.

---

## ⚙️ Pipeline Workflow

The entire CRISPRight system is structured into a modular pipeline as implemented in **`CRISPRight.py`**.

![2f1c97230398300e56142ba825a5b8b91e26087f](https://github.com/user-attachments/assets/564ea7e3-7956-4401-abea-a6e57bb4c5b6)

### 1. **Dataset Loading**

The dataset is read from `.txt` format and saved as `.csv` for easier manipulation.

```python
load_dataset(input_file, output_file)
```

### 2. **Preprocessing**

* Selects relevant columns and removes missing data.
* Converts sequences to uppercase.
* Validates nucleotides.
* Converts ON/OFF labels into binary (1/0).
* Creates a combined 49-character sequence: `Guide + Target + PAM`.
* Performs class balancing.

```python
preprocess_data(df)
```

### 3. **Data Encoding**

Two types of encodings were used:

* **One-Hot Encoding:** Converts each base (A, T, G, C) into a binary vector of length 4.
* **K-mer Encoding:** Converts overlapping sub-sequences (of length *k*) into vectors capturing sequence patterns.

```python
prepare_train_test(df, encoding="onehot", test_size=0.2)
prepare_train_test(df, encoding="kmer", test_size=0.2, k=3)
```

### 4. **Model Architectures**

The project implements four deep learning models:

#### 🧩 CNN Models

1. **CNN-OneHot** – A convolutional neural network trained on One-Hot encoded sequences.
2. **CNN-Kmer** – A convolutional neural network trained on K-mer encoded sequences.

#### 🔁 RNN Models

1. **LSTM-OneHot** – A bidirectional LSTM trained on One-Hot encoded data.
2. **LSTM-Kmer** – A bidirectional LSTM trained on K-mer encoded data.

All models use **Adam optimizer**, **binary cross-entropy loss**, and are monitored using **accuracy**, **AUC**, **precision**, and **recall** metrics. Early stopping and learning rate reduction were applied to prevent overfitting.

### 5. **Training & Evaluation**

Each model is trained using the `train_model()` function with validation monitoring. Post-training, models are evaluated using metrics such as:

* Accuracy
* Precision
* Recall
* F1-Score
* AUC-ROC & AUC-PR
* Sensitivity & Specificity

```python
evaluate_model(model, X_test, y_test)
```

Visualization functions (`plot_evaluation_results`) generate comparative performance plots including ROC curves, precision-recall curves, and confusion matrices.

### 6. **Model Comparison**

The final comparison across all four models is summarized both numerically and visually using `compare_models()`.

---

## 📊 Results Summary

**Performance Metrics Summary:**

| Model         | Accuracy   | Precision  | Recall     | F1-Score   | AUC-ROC    | AUC-PR     | Sensitivity | Specificity |
| ------------- | ---------- | ---------- | ---------- | ---------- | ---------- | ---------- | ----------- | ----------- |
| CNN-OneHot    | 0.9467     | 0.7196     | 0.8529     | 0.7806     | 0.9628     | 0.8877     | 0.8529      | 0.9584      |
| LSTM-OneHot   | 0.9046     | 0.5393     | 0.9676     | 0.6926     | 0.9827     | 0.8615     | 0.9676      | 0.8967      |
| CNN-Kmer      | 0.9366     | 0.6871     | 0.7882     | 0.7342     | 0.9205     | 0.8424     | 0.7882      | 0.9551      |
| **LSTM-Kmer** | **0.9654** | **0.7910** | **0.9353** | **0.8571** | **0.9903** | **0.9532** | **0.9353**  | **0.9691**  |

### 🏆 Best Performers by Metric

* **Overall Best Model:** LSTM-Kmer
* **Accuracy:** LSTM-Kmer (0.9654)
* **Precision:** LSTM-Kmer (0.7910)
* **Recall:** LSTM-OneHot (0.9676)
* **AUC-ROC:** LSTM-Kmer (0.9903)
* **F1-Score:** LSTM-Kmer (0.8571)

### 📈 Statistical Summary

| Metric    | Mean   | Std   | Min   | Max   |
| --------- | ------ | ----- | ----- | ----- |
| Accuracy  | 0.9383 | 0.025 | 0.904 | 0.965 |
| Precision | 0.6843 | 0.106 | 0.539 | 0.791 |
| Recall    | 0.8860 | 0.081 | 0.788 | 0.967 |
| F1-Score  | 0.7661 | 0.070 | 0.692 | 0.857 |
| AUC-ROC   | 0.9641 | 0.031 | 0.920 | 0.990 |
| AUC-PR    | 0.8862 | 0.048 | 0.842 | 0.953 |

---

## 🧠 Key Insights

* **LSTM-Kmer** achieved the highest overall performance across all key metrics.
* **K-mer encoding** captures contextual nucleotide relationships better than One-Hot encoding.
* CNNs perform strongly in feature extraction, while LSTMs excel in sequential dependency modeling.

---

## 🧩 Technologies Used

* **Language:** Python 3.12
* **Frameworks:** TensorFlow, Keras
* **Libraries:** NumPy, Pandas, Matplotlib, Seaborn, Scikit-learn
* **Environment:** Google Colab

---

## 🚀 How to Run

```bash
# Clone the repository
git clone https://github.com/HemantBatra873/CRISPRight.git
cd CRISPRight

# Run the main pipeline
python CRISPRight.py
```

You can modify the input paths and model parameters at the bottom of the script under the `__main__` section.

---

## 📘 Citation

If you use this work in your research, please cite the CRISPRoffT dataset and this repository.

---

## 📄 License

This project is licensed under the **Apache License** — see the [LICENSE](LICENSE) file for details.

---

## 🧩 Contributors

* **Project Author:** Hemant Batra
* **Data Source:** CRISPRoffT

---

## 🏁 Summary

CRISPRight demonstrates that **sequence encoding techniques** significantly impact deep learning performance in biological datasets. The **LSTM-Kmer** model emerged as the optimal solution for CRISPR off-target prediction, achieving an overall accuracy of **96.5%** with strong precision-recall tradeoffs, indicating its robustness in genomic sequence modeling.
