# ============================================================
# MODULE 4: ML Classifier — Supervised Diagnosis
# Covers: Week 9 (Supervised Learning & Decision Trees)
# ============================================================

import numpy as np
import pandas as pd
from typing import List, Dict
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

class MLDiagnosticClassifier:
    """
    It combines multiple supervised learning models to predict
    likely diseases from a binary symptom vector.
    """

    # The features used as input for the models.
    SYMPTOM_FEATURES = [
        'fever', 'cough', 'fatigue', 'headache',
        'body_aches', 'loss_of_smell', 'chest_pain',
        'rash', 'joint_pain', 'shortness_of_breath',
        'sweating', 'frequent_urination', 'excessive_thirst',
        'blurred_vision', 'night_sweats', 'weight_loss',
        'stiff_neck', 'light_sensitivity'
    ]

    # The labels that the classifier can predict.
    DISEASE_LABELS = [
        'flu', 'covid19', 'dengue', 'cardiac_event',
        'diabetes', 'common_cold', 'tuberculosis', 'meningitis'
    ]

    def __init__(self):
        # Create the three supervised learning models used for comparison.
        self.models = {
            'Decision Tree': DecisionTreeClassifier(
                max_depth=8, criterion='entropy', random_state=42),
            'Random Forest': RandomForestClassifier(
                n_estimators=100, max_depth=10, random_state=42),
            'Gradient Boosting': GradientBoostingClassifier(
                n_estimators=100, learning_rate=0.1, random_state=42),
        }
        # Store the best-performing model after training.
        self.best_model = None
        self.best_model_name = None
        # Convert class labels into numeric values for the models.
        self.label_encoder = LabelEncoder()
        # Tracks whether the model has already been trained.
        self.is_trained = False

    def _generate_synthetic_data(self, n_samples: int = 2000) -> pd.DataFrame:
        """Generate a synthetic medical dataset with realistic symptom patterns."""
        # Fix the random seed so the data generation is reproducible.
        np.random.seed(42)
        records = []

        # How likely each symptom is for a given disease to generate binary symptom values.
        profiles = {
            'flu': {'fever': 0.90, 'cough': 0.85, 'fatigue': 0.88,
                    'headache': 0.70, 'body_aches': 0.80, 'loss_of_smell': 0.20},
            'covid19': {'fever': 0.88, 'cough': 0.80, 'fatigue': 0.90,
                        'loss_of_smell': 0.85, 'headache': 0.65, 'body_aches': 0.60},
            'dengue': {'fever': 0.98, 'rash': 0.75, 'joint_pain': 0.85,
                       'headache': 0.90, 'fatigue': 0.80, 'body_aches': 0.88},
            'cardiac_event': {'chest_pain': 0.92, 'shortness_of_breath': 0.88,
                              'fatigue': 0.70, 'sweating': 0.75, 'headache': 0.30},
            'diabetes': {'fatigue': 0.82, 'frequent_urination': 0.95,
                         'excessive_thirst': 0.92, 'blurred_vision': 0.70,
                         'weight_loss': 0.50},
            'common_cold': {'cough': 0.90, 'fever': 0.50, 'headache': 0.60,
                            'fatigue': 0.55, 'body_aches': 0.50},
            'tuberculosis': {'cough': 0.95, 'weight_loss': 0.85, 'night_sweats': 0.80,
                             'fatigue': 0.88, 'fever': 0.70},
            'meningitis': {'headache': 0.95, 'stiff_neck': 0.90, 'fever': 0.92,
                           'light_sensitivity': 0.85, 'fatigue': 0.80},
        }

        # Create roughly equal numbers of samples for each disease.
        n_per_class = n_samples // len(profiles)
        for disease, symptom_probs in profiles.items():
            for _ in range(n_per_class):
                # Start with a zero vector and assign symptoms based on probabilities.
                record = {f: 0 for f in self.SYMPTOM_FEATURES}
                for symptom, prob in symptom_probs.items():
                    if symptom in record:
                        record[symptom] = int(np.random.random() < prob)

                # Add a small amount of random noise to make the data less idealized.
                for feat in self.SYMPTOM_FEATURES:
                    if record[feat] == 0 and np.random.random() < 0.05:
                        record[feat] = 1

                # Attach the disease label to the record.
                record['disease'] = disease
                records.append(record)

        # Shuffle rows so the dataset is not ordered by disease class.
        df = pd.DataFrame(records).sample(frac=1, random_state=42)
        return df

    def train(self, verbose: bool = True) -> Dict:
        """Train all models and select the best-performing one."""
        # Create the dataset and split it into features (X) and labels (y).
        df = self._generate_synthetic_data(2000)
        X = df[self.SYMPTOM_FEATURES].values
        y = self.label_encoder.fit_transform(df['disease'])

        # Split into training and testing sets while preserving class balance.
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y)

        results = {}
        best_acc = 0.0

        if verbose:
            print("=" * 55)
            print("  ML Diagnostic Classifier — Training")
            print("=" * 55)

        # Train each classifier and evaluate it using cross-validation and test accuracy.
        for name, model in self.models.items():
            model.fit(X_train, y_train)
            cv_scores = cross_val_score(model, X, y, cv=5, scoring='accuracy')
            test_acc = model.score(X_test, y_test)
            results[name] = {
                'cv_mean': cv_scores.mean(),
                'cv_std': cv_scores.std(),
                'test_acc': test_acc
            }
            if verbose:
                print(f"\n  🌲 {name}")
                print(f"     CV Accuracy : {cv_scores.mean():.4f} "
                      f"± {cv_scores.std():.4f}")
                print(f"     Test Accuracy: {test_acc:.4f}")

            # Select the model with the highest test accuracy.
            if test_acc > best_acc:
                best_acc = test_acc
                self.best_model = model
                self.best_model_name = name

        # Mark the classifier as trained and save the test set for later evaluation.
        self.is_trained = True
        self._X_test = X_test
        self._y_test = y_test

        if verbose:
            print(f"\n  🏆 Best Model: {self.best_model_name} "
                  f"({best_acc:.4f})")
        return results

    def predict(self, symptoms: List[str]) -> Dict:
        """Predict the disease from a list of symptoms."""
        if not self.is_trained:
            self.train(verbose=False)

        # Convert the symptom list into the same binary vector format used during training.
        features = np.array([
            [1 if s in symptoms else 0 for s in self.SYMPTOM_FEATURES]
        ])

        # Make the prediction and get probabilities for each disease class.
        pred_encoded = self.best_model.predict(features)[0]
        pred_proba = self.best_model.predict_proba(features)[0]

        # Convert numeric predictions back to human-readable disease names.
        disease = self.label_encoder.inverse_transform([pred_encoded])[0]
        classes = self.label_encoder.inverse_transform(range(len(pred_proba)))
        prob_map = dict(zip(classes, pred_proba))
        top5 = sorted(prob_map.items(), key=lambda x: x[1], reverse=True)[:5]

        return {
            'diagnosis': disease,
            'confidence': round(float(pred_proba[pred_encoded]), 4),
            'top5': top5,
            'model_used': self.best_model_name,
            'symptom_vector': features[0].tolist()
        }

    def analyze(self, percept) -> Dict:
        """Module interface for the agent."""
        # The agent passes a percept object that contains the user's symptoms.
        result = self.predict(percept.symptoms)
        result['summary'] = (
            f"{result['model_used']}: {result['diagnosis']} "
            f"({result['confidence']:.2%})"
        )
        return result

    def plot_evaluation(self):
        """Visualize model performance using a confusion matrix and feature importances."""
        if not self.is_trained:
            self.train(verbose=False)

        # Use the best model to predict labels for the held-out test set.
        y_pred = self.best_model.predict(self._X_test)
        cm = confusion_matrix(self._y_test, y_pred)
        labels = self.label_encoder.classes_

        fig, axes = plt.subplots(1, 2, figsize=(16, 6))

        # Plot the confusion matrix to show correct vs incorrect predictions.
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=labels, yticklabels=labels, ax=axes[0])
        axes[0].set_title(f"Confusion Matrix\n({self.best_model_name})",
                          fontweight='bold')
        axes[0].set_xlabel("Predicted")
        axes[0].set_ylabel("True")
        plt.setp(axes[0].xaxis.get_majorticklabels(), rotation=45, ha='right')

        # Plot the most important symptoms for the best model.
        if hasattr(self.best_model, 'feature_importances_'):
            importances = self.best_model.feature_importances_
            sorted_idx = np.argsort(importances)[::-1][:12]
            top_features = [self.SYMPTOM_FEATURES[i] for i in sorted_idx]
            top_values = importances[sorted_idx]
            colors = plt.cm.RdYlGn(top_values / top_values.max())
            axes[1].barh(range(len(top_features)), top_values[::-1],
                         color=colors[::-1])
            axes[1].set_yticks(range(len(top_features)))
            axes[1].set_yticklabels(top_features[::-1])
            axes[1].set_title("Feature Importances (Top 12)",
                              fontweight='bold')
            axes[1].set_xlabel("Importance Score")

        # Add a title and save the figure to disk.
        plt.suptitle(f"ML Diagnostic Model Evaluation — {self.best_model_name}",
                     fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig("ml_evaluation.png", dpi=150, bbox_inches='tight')
        plt.show()
        print("✅ Saved: ml_evaluation.png")