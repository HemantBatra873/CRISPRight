# 📊 CRISPRoffT Dataset Description

## 🧬 Dataset Overview

The **CRISPRoffT** dataset is a comprehensive and experimentally validated database designed to study **off-target effects** in the **CRISPR-Cas9** genome-editing system. It serves as one of the most extensive and well-curated resources for understanding how guide RNAs (gRNAs) interact with unintended DNA targets across different organisms, enzymes, and experimental contexts.


<img width="2400" height="2560" alt="image" src="https://github.com/user-attachments/assets/08100a0c-2f48-4a6c-9647-a1eee7306c8c" />




Dataset Source: [CRISPRoffT Database](https://ccsm.uth.edu/CRISPRoffT/)

Publication Reference: [Nucleic Acids Research, 2025, Volume 53, D914–D921](https://academic.oup.com/nar/article/53/D1/D914/7889256)

---

## 📚 Key Features of the Dataset

* **Total entries:** ~226,000 potential off-target sites
* **Guide RNAs covered:** 371 unique gRNAs
* **Validated off-targets:** 8,940 sites with experimental evidence
* **Cas enzyme coverage:** Data includes 85 different Cas/gRNA combinations
* **Organisms & cell types:** 34 different human and mouse cell lines/tissues
* **Validation technologies:** Includes multiple experimental techniques like GUIDE-seq, Digenome-seq, CIRCLE-seq, etc.

---

## 🧠 Dataset Structure

Each record in the dataset represents a single **guide–target pair**, with detailed annotations describing their sequence similarity, genomic position, and validation status.

### 🔹 Main Columns

| Column Name           | Description                                                                                        |
| --------------------- | -------------------------------------------------------------------------------------------------- |
| **Guide_sequence**    | The 20-nucleotide RNA sequence used in CRISPR targeting.                                           |
| **Target_sequence**   | DNA sequence potentially recognized by the guide RNA.                                              |
| **PAM**               | Protospacer Adjacent Motif — a short motif required for Cas enzyme binding (e.g., NGG for SpCas9). |
| **Chromosome**        | Chromosomal location of the target site.                                                           |
| **Start / End**       | Genomic coordinates of the target region.                                                          |
| **Strand**            | Indicates whether the target site is on the + or – strand.                                         |
| **Mismatch_count**    | Number of mismatched bases between gRNA and target DNA.                                            |
| **Indel_info**        | Information about insertions or deletions, if any.                                                 |
| **Cas_system**        | Type of CRISPR-Cas enzyme used (e.g., SpCas9, SaCas9, Cpf1).                                       |
| **Cell_line**         | Name of the cell line or tissue where the experiment was performed.                                |
| **Validation_status** | Whether the off-target site has been experimentally validated (TRUE/FALSE).                        |
| **Technology_used**   | Method used for validation (e.g., GUIDE-seq, Digenome-seq).                                        |
| **Gene_name**         | The gene overlapping or nearest to the target site.                                                |

---

## ⚙️ Data Usage in This Project

In this project (CRISPRight – Off-Target Prediction), a subset of the CRISPRoffT dataset is used to:

* Train ML/DL models to predict off-target sites based on gRNA–target sequence pairs.
* Encode sequence data using one-hot or k-mer embeddings.
* Label pairs as **1 (off-target)** or **0 (on-target)** using the `Validation_status` field.

---

## 🧩 Data Format

The dataset is typically available in **TSV (tab-separated values)** or **CSV** format, with each line representing one gRNA–target pair.

Example snippet:

```
Guide_sequence   Target_sequence   PAM   Chromosome   Start   End   Strand   Mismatch_count   Cas_system   Validation_status
GAGTCCGAGCAGAAGAAGA   GAGTCCGAGCAGAAGAAGG   NGG   chr1   1456790   1456812   +   1   SpCas9   TRUE
```

---

## 📜 Licensing and Citation

The CRISPRoffT dataset is publicly available for academic and research purposes. When using or referencing this dataset, please cite:

> "CRISPRoffT: A comprehensive database of CRISPR/Cas off-targets across enzymes and organisms," *Nucleic Acids Research*, 2025.

Dataset URL: [https://ccsm.uth.edu/CRISPRoffT/](https://ccsm.uth.edu/CRISPRoffT/)

---

## 🧑‍🔬 Notes

* The dataset contains **both predicted and experimentally validated** off-targets — users can choose subsets based on validation confidence.
* Imbalance exists between positive (validated) and negative (non-validated) samples.
* Data can be filtered by enzyme type, organism, or technology depending on model objectives.

---

**Maintainer:** CRISPRight Team
**Data Source:** [CRISPRoffT Database](https://ccsm.uth.edu/CRISPRoffT/)
