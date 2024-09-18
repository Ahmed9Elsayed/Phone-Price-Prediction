# **Phone Price Classification**

  

## Overview

This project aims to classify phones based on several specs of the phone. The model is trained on a dataset that contains phone specs associated with the price category that the phone belongs to, allowing the system to classify phones about the most likely price category based on the input specs.

## Table of Contents

- [Overview](#overview)
- [Project Description](#project-description)
- [Dataset](#dataset)
- [Installation](#installation)
- [Usage](#usage)
- [Evaluation](#evaluation)
- [Results](#results)
- [Conclusion](#Conclusion)
- [Future Work](#future-work)
- [Contact Information](#Contact-Information)


## Project Description

In this notebook, we explore a supervised learning approach for phone price category prediction using specs. The goal is to create a machine learning model that, given a set of phone specs, predicts the most likely price category. This system can assist in categorizing new released phones into a specific price range.

## Dataset

The dataset contains rows with phone specs as input features and a corresponding price category as the target label.

### Features:

- **Specs**:
	-  id - ID
	- battery_power - Total energy a battery can store in one time measured in mAh
	- blue - Has Bluetooth or not
	- clock_speed - The speed at which the microprocessor executes instructions
	- dual_sim - Has dual sim support or not
	- fc - Front Camera megapixels
	- four_g - Has 4G or not
	- int_memory - Internal Memory in Gigabytes
	- m_dep - Mobile Depth in cm
	- mobile_wt - Weight of mobile phone
	- n_cores - Number of cores of the processor
	- pc - Primary Camera megapixels
	- px_height - Pixel Resolution Height
	- px_width - Pixel Resolution Width
	- ram - Random Access Memory in Megabytes
	- sc_h - Screen Height of mobile in cm
	- sc_w - Screen Width of mobile in cm
	- talk_time - longest time that a single battery charge will last when you are
	- three_g - Has 3G or not
	- touch_screen - Has touch screen or not
	- wifi - Has wifi or not

- **Target**: The target label is the price category.
	- 0 (low cost)
	- 1 (medium cost)
	- 2 (high cost)
	- 3 (very high cost)

## **Installation**

  

To run this notebook, you need a Python environment with the following libraries installed:

  

- `pandas`

- `numpy`

- `scikit-learn`

- `matplotlib`

- `seaborn`

  

You can install the dependencies using:

  

`pip install pandas numpy scikit-learn matplotlib seaborn`

  

## **Usage**

  

1. **Load the Notebook:**

    - Open the notebook in Kaggle, Colab or your local Jupyter environment.

2. **Data Preprocessing and EDA:**

    - The notebook includes code to preprocess all issues in the dataset.
    - also some EDA code about the dataset to understand it further.

3. **Feature Engineering:**

    - used Mutual Information (MI) to capture any type of relationship between features and target
    - did Feature selection on the specs to subset only important predictors
    - created 2 new features to make it easier for the model to capture relations and patterns
	    - resolution = px_height * px_width
	    - mobile_size = sqrt( (sc_h ** 2) + (sc_w ** 2) )
	


4. **Model Training:**

    Several machine learning models are used in this project, including:
    - **Random Forest Classifier**
	- **XGBoost Classifier**
	- **LightGBM Classifier**
 The models are trained using the specs input features, and the price prediction is based on the likelihood derived from the specs.

  

5. **Evaluation:**

    - Performance metrics such as accuracy, precision, recall, and F1- score are used to evaluate the models.

    - Feature importance is analyzed to identify the key factors contributing to the predictions.

  

## **Results**
The model is evaluated using several metrics, including:

- **Accuracy**: Measures the percentage of correct predictions.
- **Precision**: Measures the number of true positives over the sum of true positives and false positives.
- **Recall**: Measures the number of true positives over the sum of true positives and false negatives.
- **F1-Score**: The harmonic mean of precision and recall.

These metrics are calculated using a test dataset that was not part of the training data.

#### Results before feature selection
- **Random Forest / LightGBM** with an accuracy of **89.4%** and **XGBoost** with an accuracy of **90.9%**

#### Results after feature selection
- All of the three models got a test accuracy of **100%** when training the model on the selected features only
## **Conclusion**

  
The notebook demonstrates that traditional machine learning models can effectively predict the price category of a phone based on its published specs.
  

## **Future Work**

  

- Experiment with other advanced models, such as Neural Networks.

- Gather more diverse Test data to further investigate model's predictive performance.

  
## **Contact Information**

  

If you have any questions or suggestions to improve, please contact me at ahmedelsayedtaha467@gmail.com.