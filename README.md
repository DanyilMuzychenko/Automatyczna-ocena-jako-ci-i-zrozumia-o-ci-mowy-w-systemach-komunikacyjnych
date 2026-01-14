# EN
# 🎧 Non-Intrusive Speech Quality Assessment (MOS Estimation)

> ⚠️ **Work in progress**  
> This repository contains an ongoing MSc thesis project.  
> The codebase, experiments, and documentation are still under active development.

---

## 📌 Project Overview

The goal of this project is to design and evaluate **artificial intelligence models for automatic, non-intrusive estimation of speech quality**, expressed as **Mean Opinion Score (MOS)**, without the involvement of human listeners.

The work investigates whether modern deep learning techniques can reliably predict **subjective speech quality ratings** based solely on acoustic features or raw audio signals, achieving performance comparable to human assessment.

The project is developed as part of a **master’s thesis** and is still in progress.

---

## 🧠 Model Architectures

Several neural network architectures are evaluated, depending on the type of acoustic representation used.

### 1️⃣ CNN-based models (Mel-spectrograms & MFCC)

For **Mel-spectrogram** and **MFCC** features, the following architectures are implemented:

- **CNN**  
  Pure convolutional neural network operating on 2D time–frequency representations.

- **CNN + GRU**  
  Convolutional feature extractor followed by a GRU layer to capture temporal dependencies.

- **CNN + LSTM**  
  Similar to CNN+GRU, using LSTM units for sequence modeling.

Each architecture is trained **independently** for Mel-spectrograms and MFCC features.

---

### 2️⃣ wav2vec2.0-based models

For raw audio processing, pretrained self-supervised models are used:

- **wav2vec2.0 + GRU**
- **wav2vec2.0 + LSTM**

In this setup, wav2vec2.0 acts as a feature extractor, producing high-level speech embeddings, which are then processed by recurrent layers.

> CNN-based architectures are **not applied** to wav2vec2.0 embeddings.

---

## 🎼 Acoustic Feature Extraction

The following acoustic representations are used in the project:

| Feature | Supported models |
|------|------------------|
| Mel-spectrogram | CNN / CNN+GRU / CNN+LSTM |
| MFCC | CNN / CNN+GRU / CNN+LSTM |
| wav2vec2.0 embeddings | wav2vec2 + GRU / LSTM |

No explicit statistical normalization (e.g., mean–variance normalization) of acoustic features is applied. Feature scaling is handled implicitly by neural network layers.

---

## 🎯 MOS Scaling

MOS values are linearly normalized during training to the range:

\[
\text{MOS}_{norm} = \frac{\text{MOS} - 1}{4}
\]

For evaluation and visualization, predictions are rescaled back to the standard **MOS ∈ [1, 5]** range.

---

## 📊 Evaluation Metrics

Model performance is evaluated using the following metrics:

- **MSE (Mean Squared Error)**
- **RMSE (Root Mean Squared Error)**
- **Pearson Correlation Coefficient**
- **Spearman Rank Correlation Coefficient**

Metrics are computed on the validation set.

---

## 📂 Dataset

Experiments are conducted using the **NISQA Corpus**, which contains diverse speech samples with various degradations, annotated with subjective MOS ratings.

---

## 🚧 Project Status

- ✔ Feature extraction pipelines implemented  
- ✔ Multiple neural network architectures implemented  
- ✔ Training and validation framework completed  
- 🔄 Hyperparameter tuning in progress  
- 🔄 Extended evaluation and result analysis in progress  
- 🔄 Documentation and thesis writing in progress  

---

## 📌 Notes

This repository reflects an **experimental research setup**.  
The structure, models, and evaluation procedures may evolve as the thesis work progresses.

---

## 📄 License

This project is developed for academic research purposes.


# PL
# 🎧 Non-Intrusive Speech Quality Assessment (Estymacja MOS)

> ⚠️ **Projekt w trakcie realizacji**  
> Repozytorium zawiera kod oraz eksperymenty realizowane w ramach pracy magisterskiej.  
> Implementacja, eksperymenty oraz dokumentacja są nadal rozwijane.

---

## 📌 Opis projektu

Celem niniejszego projektu jest opracowanie oraz ocena modeli sztucznej inteligencji umożliwiających **automatyczną, nieinwazyjną estymację jakości mowy**, wyrażonej za pomocą wskaźnika **MOS (Mean Opinion Score)**, bez udziału człowieka.

Projekt bada, czy nowoczesne techniki uczenia maszynowego, w szczególności modele oparte na **głębokich sieciach neuronowych**, są w stanie skutecznie przewidywać subiektywne oceny jakości mowy na podstawie cech akustycznych lub sygnału audio, w sposób porównywalny z oceną ludzką.

Projekt realizowany jest jako część **pracy magisterskiej** i pozostaje w fazie rozwoju.

---

## 🧠 Architektury modeli

W projekcie zaimplementowano i przetestowano kilka architektur sieci neuronowych, w zależności od rodzaju zastosowanej reprezentacji sygnału mowy.

### 1️⃣ Modele CNN (Mel-spektrogramy i MFCC)

Dla reprezentacji opartych na **mel-spektrogramach** oraz **współczynnikach MFCC** zastosowano następujące architektury:

- **CNN**  
  Konwolucyjna sieć neuronowa przetwarzająca dwuwymiarowe reprezentacje czasowo-częstotliwościowe.

- **CNN + GRU**  
  Ekstraktor cech oparty na CNN połączony z warstwą GRU w celu modelowania zależności czasowych.

- **CNN + LSTM**  
  Analogiczna architektura z wykorzystaniem warstw LSTM.

Każda architektura trenowana jest **oddzielnie** dla mel-spektrogramów oraz MFCC.

---

### 2️⃣ Modele oparte na wav2vec2.0

Dla pracy na surowym sygnale audio wykorzystano modele samouczące się:

- **wav2vec2.0 + GRU**
- **wav2vec2.0 + LSTM**

W tym podejściu model wav2vec2.0 pełni rolę ekstraktora embeddingów mowy, które następnie przetwarzane są przez sieci rekurencyjne.

> Architektury konwolucyjne nie są stosowane bezpośrednio do embeddingów wav2vec2.0.

---

## 🎼 Ekstrakcja cech akustycznych

W projekcie wykorzystano następujące reprezentacje danych:

| Reprezentacja | Obsługiwane modele |
|---------------|-------------------|
| Mel-spektrogramy | CNN / CNN + GRU / CNN + LSTM |
| MFCC | CNN / CNN + GRU / CNN + LSTM |
| Embeddingi wav2vec2.0 | wav2vec2 + GRU / LSTM |

W ramach niniejszej pracy **nie zastosowano explicite statystycznej normalizacji cech akustycznych** (np. normalizacji średniej i wariancji). Skalowanie cech realizowane jest pośrednio przez warstwy sieci neuronowych.

---

## 🎯 Skala MOS

Podczas uczenia modeli wartości MOS są normalizowane do zakresu:

\[
\text{MOS}_{\text{norm}} = \frac{\text{MOS} - 1}{4}
\]

Na etapie ewaluacji oraz wizualizacji wyniki są przeskalowywane z powrotem do standardowego zakresu **MOS ∈ [1, 5]**.

---

## 📊 Metryki ewaluacji

Jakość modeli oceniana jest przy użyciu następujących miar:

- **MSE (Mean Squared Error)** – błąd średniokwadratowy  
- **RMSE (Root Mean Squared Error)** – pierwiastek błędu średniokwadratowego  
- **Pearson Correlation Coefficient** – korelacja liniowa  
- **Spearman Rank Correlation Coefficient** – korelacja rangowa  

Metryki obliczane są na zbiorze walidacyjnym.

---

## 📂 Zbiory danych

Eksperymenty przeprowadzono z wykorzystaniem **NISQA Corpus**, zawierającego nagrania mowy o zróżnicowanych degradacjach jakościowych wraz z subiektywnymi ocenami MOS.

---

## 🚧 Status projektu

- ✔ Implementacja ekstrakcji cech  
- ✔ Implementacja architektur modeli  
- ✔ Pipeline treningu i walidacji  
- 🔄 Strojenie hiperparametrów  
- 🔄 Analiza wyników eksperymentów  
- 🔄 Opracowanie końcowej dokumentacji  

---

## 📌 Uwagi

Repozytorium stanowi środowisko badawcze.  
Struktura projektu, architektury oraz procedury ewaluacyjne mogą ulec zmianie w trakcie dalszych prac.

---

## 📄 Licencja

Projekt realizowany w celach naukowych i dydaktycznych.
