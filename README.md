# Intelligent Healthcare Diagnostic Assistant

## Overview

This project is an end-to-end AI system that integrates multiple artificial intelligence sub-fields into a unified healthcare diagnostic and recommendation platform. The system mimics a medical board of specialists working together to diagnose patients and generate treatment plans.

## System Architecture

The application relies on a central Intelligent Agent that coordinates six distinct diagnostic modules:

* **Module 1: Intelligent Agent** - The central triage coordinator that perceives patient data, orchestrates the other modules, and aggregates the results.


* **Module 2: Logic & Knowledge Base** - Uses First-Order Logic and forward/backward chaining for rule-based diagnostics.


* **Module 3: Bayesian Network** - Handles probabilistic reasoning under uncertainty using Naïve Bayes.


* **Module 4: ML Classifier** - Evaluates patient data using Decision Trees, Random Forests, and Gradient Boosting to find learned patterns.


* **Module 5: Deep Neural Network** - A Multi-Layer Perceptron (MLP) built with TensorFlow/Keras for complex pattern recognition.


* **Module 6: Fuzzy Severity Assessor** - Calculates a patient's overall severity score (0-100) and urgency using Fuzzy Logic.


* **Module 7: AI Planner** - Uses Breadth-First Search (BFS) and STRIPS planning to generate a logical, step-by-step medical treatment plan.



## Prerequisites

Ensure you have the following installed on your machine:

* Python 3.9+


* Git



## Setup & Installation

1. **Clone the repository:**
```bash
git clone https://github.com/sirrom/Capstone-Project-Intelligent-Healthcare-Diagnostic-Assistant.git
cd ai-capstone-healthcare
```


2. **Create and activate a virtual environment:**
* **Windows:**
```bash
python -m venv venv
venv\Scripts\activate
```


* **Mac/Linux:**
```bash
python -m venv venv
source venv/bin/activate
```




3. **Install the required dependencies:**
```bash
pip install -r requirements.txt
```



## Usage

To run the complete end-to-end system and process the built-in test patients, simply execute the main application file:

```bash
python app.py
```

*Note: Upon startup, the system will take a few moments to generate synthetic data and train the Machine Learning and Deep Learning models. If graphical evaluation plots appear, close the windows to allow the terminal to finish processing the patients.*
