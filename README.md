
# Effectiveness of Large Language Models in Predicting Stock Movements

## Overview

This repository contains a research study that investigates the effectiveness of Large Language Models (LLMs), specifically Mistral and LLaMA2, in predicting stock price movements. The study uses sentiment analysis on financial news headlines to evaluate the predictive accuracy of these models. The research compares the predictions of these open-source LLM models with human-labeled financial sentiment and traditional methods for financial forecasting.

## Research Objectives

The primary goals of this research are:
- To evaluate the sentiment prediction capabilities of Mistral and LLaMA2 for stock market data.
- To compare the LLM-based predictions against human-labeled financial sentiment.
- To analyze which model provides more reliable and accurate outputs for use in financial forecasting pipelines.

## Methodology

1. **Data Collection**: A curated dataset of labeled financial news headlines is used. These headlines are annotated with ground truth sentiment (Positive, Neutral, Negative).
2. **Models Used**:
    - **Mistral**: An open-source LLM model used for sentiment analysis.
    - **LLaMA2**: Another open-source LLM model, used for comparison against Mistral.
3. **Metrics for Evaluation**:
    - Accuracy
    - Precision
    - Recall
    - F1-Score
4. **Testing Procedure**: The dataset of headlines is passed through both models. The output predictions are then compared with ground truth labels.

## Dataset

The dataset used for the sentiment analysis consists of:
- Financial news headlines.
- Sentiment labels (Positive, Neutral, Negative).
- S&P 500 index data (Close and Change values).
- AAII Sentiment Survey (high-frequency investor sentiment data).

## Installation

To run this code locally, clone this repository:

```bash
git clone https://github.com/colabre2020/LLM-StockPrediction.git
cd LLM-StockPrediction

## Install the necessary Python dependencies:

pip install -r requirements.txt


## Model Setup

from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

model = AutoModelForSeq2SeqLM.from_pretrained('path-to-model')
tokenizer = AutoTokenizer.from_pretrained('path-to-model')


## Predict Sentiment

input_text = "Stock market surges on positive earnings report"
inputs = tokenizer(input_text, return_tensors="pt")
outputs = model.generate(inputs['input_ids'])
print(tokenizer.decode(outputs[0], skip_special_tokens=True))


## Analyze Performance

import matplotlib.pyplot as plt
import seaborn as sns

sns.heatmap(confusion_matrix, annot=True, fmt="d")
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()


## Results

Headline | True Label | Mistral | LLaMA2
Stock market surges on positive earnings report | Positive | Positive | Positive
Investors worry about rising inflation | Neutral | Neutral | Negative
Markets remain steady amid global tensions | Neutral | Neutral | Positive

## Contributing

Contributions are welcome! Please open an issue or submit a pull request if you have any suggestions, improvements, or bug fixes.


## Reference

Meta AI (2023). LLaMA2: Open Foundation and Chat Models.
Mistral AI (2023). Mistral 7B Model Card.
The website of the American Association of Individual Investors (http://www.aaii.com/)
https://doi.org/10.1016/j.iref.2021.11.018
