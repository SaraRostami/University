# University Projects: ML, Deep Learning & Data Analysis

Coursework, projects, teaching and workshop material from my MSc in Artificial Intelligence & Robotics at the **University of Tehran** (2021–2024). Every featured project below has its own README covering the problem, the approach, the results taken from the notebooks, and known limitations.

**Stack across projects:** Python · PyTorch · TensorFlow/Keras · scikit-learn · Hugging Face · SHAP / LIME · pandas · librosa · OpenCV · R · MATLAB

---

## Featured Projects

| Project | What it covers | Key result |
|---|---|---|
| [**Explainable AI: SHAP, D-RISE & LIME**](Trustworthy%20AI%20-%20Spring%202023/HW2_Model%20Interpretability) | Deep SHAP and Kernel SHAP on a life-expectancy MLP; D-RISE saliency maps for a Faster R-CNN detector; LIME superpixel explanations for MobileNetV2 | Test R² 0.956; Deep and Kernel SHAP both rank **Schooling** as the top predictor |
| [**GANs: DCGAN, AC-GAN & WGAN**](Neural%20Networks%20-%20Fall%202022/Assignments/NNDL-HW6) | Three GAN variants trained on a 1K-image hand-gesture dataset; label smoothing and noisy labels for training stability | Label smoothing and noisy labels stopped the WGAN from collapsing; it then trained stably for all 50 epochs |
| [**BERT Encoder from Scratch & BEiT**](Neural%20Networks%20-%20Fall%202022/Assignments/NNDL-HW5) | Multi-head attention, GELU, Add & Norm and the pooler implemented by hand in Keras; BEiT / SegFormer segmentation on ADE20K | **73.8%** sentiment accuracy on 84K Rotten Tomatoes reviews |
| [**Fraud Detection, Liveness & OCR**](Neural%20Networks%20-%20Fall%202022/Assignments/Extra%20Homework) | SMOTE + denoising autoencoder for credit-card fraud (0.17% positives); CNN face-liveness classifier and blink-detection demo; Persian handwritten digit recognition | **87%** fraud recall (vs. 2% baseline); **99.48%** HODA digit accuracy |
| [**Customer Retention Cohort Analysis**](Data%20Analysis%20-%20Fall%202022/Assignments/HW5) | Monthly acquisition cohorts and a retention heatmap; channel and product-size breakdowns; process mining on a hospital event log | 64% of new customers are lost in their first month; the weakest cohorts are identified |
| [**Musical Instrument Classification from Audio**](Machine%20Learning%20-%20Fall%202021/Final%20Project) | End-to-end: data collection, MFCC and zero-crossing features with librosa, 5 classifiers, 5 clustering algorithms, 6 instruments (4 Persian) | MLP **84%** accuracy across 6 classes (chance 17%) |

## More Projects

**Deep learning** ([Neural Networks & Deep Learning, Fall 2022](Neural%20Networks%20-%20Fall%202022/Assignments))
- [HW1](Neural%20Networks%20-%20Fall%202022/Assignments/NNDL-HW1): McCulloch-Pitts neurons, Adaline/Madaline, an RBM recommender and an MLP regressor
- [HW2](Neural%20Networks%20-%20Fall%202022/Assignments/NNDL-HW2): how input resolution affects CNN accuracy on CIFAR-10
- [HW3](Neural%20Networks%20-%20Fall%202022/Assignments/NNDL-HW3): transfer learning, face recognition under occlusion, and YOLOv6 for chess-piece detection
- [HW4](Neural%20Networks%20-%20Fall%202022/Assignments/NNDL-HW4): LSTM air-pollution forecasting and CNN-LSTM fake-news detection

**Trustworthy AI** (Spring 2023)
- [Generalization & robustness](Trustworthy%20AI%20-%20Spring%202023/HW1_Investigating%20the%20Generalization%20and%20Robustness%20of%20a%20Deep%20Learning%20Model): training on limited CIFAR-10 data and testing robustness to noise
- [Fairness & security](Trustworthy%20AI%20-%20Spring%202023/HW3_Fairness%20and%20Security): adversarial debiasing on Adult, a backdoor attack on a cats-vs-dogs classifier, and out-of-distribution detection
- [Final project](Trustworthy%20AI%20-%20Spring%202023/FinalProject): presentation on evaluating ML fairness under uncertain and incomplete information

**Data analysis** (Fall 2022)
- [Cryptocurrency price-direction prediction](Data%20Analysis%20-%20Fall%202022/Final%20Project): web scraping, technical indicators, and classical vs. deep models for BTC
- [Assignments](Data%20Analysis%20-%20Fall%202022/Assignments): descriptive analysis of hate-crime data, geospatial analysis with QGIS, and more

**Other courses**
- [Machine Learning, Fall 2021](Machine%20Learning%20-%20Fall%202021/Assignments): problem sets and implementations
- [Statistical Inference, Fall 2021](Statistical%20Inference%20-%20Fall%202021): assignments in R
- [Introduction to Cognitive Neuroscience, Spring 2022](Introduction%20to%20Cognitive%20Neuroscience%20-%20Spring%202022/Assignments): EEG and fMRI analysis, and LIF / Morris-Lecar neuron models
- [Bio-Inspired Computing, Spring 2023](Bio-inspired%20Computing%20-%20Spring%202023): combinatorial search (independent set, zero-one equations) and cellular automata (Game of Life and 3-D variants)

## Teaching & Workshops
- **Teaching Assistant, Statistical Inference** (Spring 2023, Dr. Abdol-hossein Vahabie): co-designed and graded [Homework 1](Files/Statistical%20Inference_HW1.pdf). It covers sampling strategies, conditional probability, experimental vs. observational studies, confounders, misinformation analysis, and an R programming question.
- **Teaching Assistant, Introduction to Cognitive Neuroscience** (Spring 2023, Dr. Mohammadreza Abolghasemi Dehaqani). Course materials: [EEG homework](Files/Cognitive%20Neuroscience_EEG_HW.pdf) · [final project](Files/Cognitive%20Neuroscience_Final%20Project.pdf).
- **[MNE-Python Workshop, CuttingGardens 2023](CuttingEEG%20Conference%20-%20MNE-Python%20Workshop)**: co-presented the MNE-Python EEG-analysis tutorial (preprocessing, ERPs, time-frequency analysis) at the [Tehran Garden](https://cuttinggardens2023.org/gardens/tehran/) of [CuttingGardens](https://cuttinggardens2023.org/), part of the [CuttingEEG](https://cuttingeeg.org/) conference series.

## Courses

| Course | Term | Instructor(s) |
|---|---|---|
| Machine Learning | Fall 2021 | [Dr. M. Abolghasemi Dehaqani](https://ece.ut.ac.ir/en/~dehaqani), [Dr. B. Nadjar Araabi](https://ece.ut.ac.ir/en/~araabi) |
| Statistical Inference | Fall 2021 | [Dr. B. Bahrak](https://scholar.google.com/citations?user=1IdcoLMAAAAJ&hl=en) |
| Introduction to Cognitive Neuroscience | Spring 2022 | [Dr. M. Abolghasemi Dehaqani](https://scholar.google.com/citations?user=HuMGDxIAAAAJ&hl=en) |
| Data Analysis | Fall 2022 | [Dr. M. A. Sadeghi](https://scholar.google.com/citations?hl=en&user=Viogmi8AAAAJ&view_op=list_works&sortby=pubdate), [Dr. M. Abolghasemi Dehaqani](https://scholar.google.com/citations?user=HuMGDxIAAAAJ&hl=en) |
| Neural Networks & Deep Learning | Fall 2022 | [Dr. A. Kalhor](https://scholar.google.com/citations?user=m7xdmMgAAAAJ&hl=en) |
| Trustworthy AI | Spring 2023 | [Dr. M. A. Sadeghi](https://scholar.google.com/citations?hl=en&user=Viogmi8AAAAJ&view_op=list_works&sortby=pubdate), [Dr. M. Tavassolipour](https://scholar.google.com/citations?user=oVAT1lYAAAAJ&hl=en) |
| Bio-Inspired Computing | Spring 2023 | [Dr. M. Asadpour](https://scholar.google.com/citations?hl=en&user=MKwwcvIAAAAJ&view_op=list_works&sortby=pubdate) |

Each course folder follows the same layout: one folder per assignment, containing the problem set, the data, the solution code (notebooks or scripts) and the written report. Most reports are in Persian; every featured README is in English. Most projects were done in teams of two to four, and each featured README lists the team.
