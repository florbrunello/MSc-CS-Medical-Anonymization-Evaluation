# MSc Computer Sciences
## Evaluation of Different Approaches to the Anonymization of Spanish Medical Texts

**Summary**

This project evaluates and compares different approaches for Named Entity Recognition (NER) in Spanish medical documents, using four specialized datasets and three types of models. This repository contains the datasets, code and experiments for the research project "Evaluation of different approaches to anonymization of Spanish clinical texts". The project compares three families of solutions for de-identification of clinical text: rule-based regular expressions (REGEX), BiLSTM-CRF sequence models, and large language models (Llama 3.1). Experiments were executed on four datasets (two public and two synthetic). Everything required to reproduce the experiments is provided.

> **Note:** The Spanish version of this README is available in `README_es.md`.

## Repository structure

```
README.md                    <- this file (English)
README_es.md                 <- Spanish version of the README
Datasets/                    <- .txt, .ann, .xml (BRAT/XML)
  ├─ MEDDOCAN/
  ├─ SPG/
  ├─ SPGExtended/
  └─ CARMEN-I/   (restricted by license)
Models/
  ├─ REGEX/
  ├─ BiLSTM-CRF/
  └─ Llama3.1/
Results/
  ├─ results_dev.xlsx
  ├─ results_test.xlsx
  └─ plots_dev/
  └─ plots_dev/
Models/Scripts/
  └─ confusion_matrix_generator_global.py
LICENSE
```

## Datasets (`Datasets/`)
Contains four annotated medical datasets in BRAT format:

- **MEDDOCAN**: public corpus of 1,000 medical documents (500 train, 250 dev, 250 test). Used in IberLEF, MEDDOCAN. 
- **SPG**: 1,000 synthetic clinical records generated with the Synthetic Patient Generator module (500 train, 250 dev, 250 test).  
- **SPGExtended**: 448 urated synthetic records produced by extending SPG outputs and performing automatic/manual curation (358 train, 45 dev, 45 test).
- **CARMEN-I**: subset of 1,697 clinical case documents in Spanish (847 train, 425 dev, 425 test). This dataset is license-restricted and therefore not included in this repository. The dataset can be accessed through PhysioNet: https://physionet.org/content/carmen-i/1.0.1/


## Models (`Models/`)
Implementation and evaluation of three different approaches:

##### 1. **REGEX** - Symbolic rules based on domain knowledge using regular expressions
- Rule-based patterns developed and collected from the ARPHAI community.
- Outputs are post-processed and converted to BRAT format for evaluation.
- Strengths: interpretability and low computational cost. Weaknesses: limited recall and brittle behaviour on variable text.

##### 2. **BiLSTM-CRF** - Recurrent Neural Network
- Implementation follows a Bi-directional LSTM + CRF architecture (inspired by Ma & Hovy and Saluja et al.).
- Requires word/subword embeddings, tokenization and supervised training.
- Offers a good trade-off between performance and computational efficiency.

##### 3. **Llama 3.1** - Large Language Model
- Uses Llama 3.1 model for NER. Used via prompting (one-/few-shot) to generate inline annotations (XML) which are converted to BRAT for scoring.
- LLMs perform strongly on text similar to their training distribution but may show decreased robustness on out-of-distribution real clinical records. Risks include hallucinations and inconsistent offset handling.

## Results
Results are located in the `Results/` folder:

### Quantitative Metrics
- **`results_dev.xlsx`**: Evaluation metrics on development set
- **`results_test.xlsx`**: Evaluation metrics on test set
- Contains detailed metrics (Precision, Recall, F1-Score) by model and dataset

### Visualizations
- **`plots_dev/`**: Confusion matrices and plots for development set
- **`plots_test/`**: Confusion matrices and plots for test set
- Visual comparison between models (REGEX, BiLSTM-CRF, Llama3.1)

## Execution

To run specific evaluations, refer to the `ejecucion.md` files in each model:
- `Models/REGEX/[dataset]/ejecucion.md`
- `Models/BiLSTM-CRF/ejecucion.md` 
- `Models/Llama3.1/[dataset]/ejecucion.md`

To generate global confusion matrices:
```bash
cd Models/Scripts
python3 confusion_matrix_generator_global.py [MODEL] --partition [dev/test/both]
```

## License

This project is licensed under **CC0 1.0 Universal** (Creative Commons Zero) - see the [LICENSE](LICENSE) file for more details.

The CC0 license allows free use, modification and distribution of the code without restrictions, effectively dedicating the work to the public domain.

## Acknowledgements

This work used computational resources from UNC Supercómputo (CCAD), which is part of SNCAD, Argentina and resources from the ARPHAI community. 
