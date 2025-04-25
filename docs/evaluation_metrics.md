# Model Evaluation Metrics Explanation

This document explains the evaluation metrics reported by the `python src/cli.py evaluate --model all` command for the different models used in the retail analytics project.

## 1. Forecasting Model (XGBoost)

These metrics evaluate how accurately the model predicts future sales values compared to the actual sales values in the test dataset.

*   **RMSE (Root Mean Squared Error): 74.97**
    *   **What it is:** Measures the average magnitude of the errors (the difference between predicted and actual sales). It squares the errors before averaging, giving higher weight to larger errors, and then takes the square root to return the metric to the original units (sales currency/amount).
    *   **Interpretation:** On average, the model's sales predictions are off by about 74.97 units. Lower values are better.
*   **MAE (Mean Absolute Error): 31.39**
    *   **What it is:** Measures the average absolute difference between predicted and actual sales. It treats all errors equally, regardless of their magnitude.
    *   **Interpretation:** On average, the model's sales predictions are off by about 31.39 units. Lower values are better. MAE is often easier to interpret directly than RMSE.
*   **R² (R-squared): 0.97**
    *   **What it is:** Represents the proportion of the variance in the actual sales data that is explained by the model's predictions. It ranges from 0 to 1 (or can be negative for very poor models).
    *   **Interpretation:** An R² of 0.97 means the model explains 97% of the variability in the sales data. Values closer to 1 indicate a better fit. This is a very good score.
*   **MAPE (Mean Absolute Percentage Error): 2.53%**
    *   **What it is:** Measures the average percentage difference between predicted and actual sales. It's useful for understanding the error relative to the actual value.
    *   **Interpretation:** On average, the model's predictions are off by about 2.53% of the actual sales value. Lower percentages are better.

## 2. Segmentation Model (KMeans)

These metrics evaluate the quality of the customer/store clusters formed by the model. Good clustering means data points within a cluster are similar to each other, and data points in different clusters are dissimilar.

*   **Silhouette Score: 0.2756**
    *   **What it is:** Measures how similar an object is to its own cluster compared to other clusters. It ranges from -1 to 1.
    *   **Interpretation:**
        *   Values near +1 indicate that the object is well matched to its own cluster and poorly matched to neighboring clusters.
        *   Values near 0 indicate that the object is on or very close to the decision boundary between two neighboring clusters.
        *   Values near -1 indicate that the object is probably assigned to the wrong cluster.
    *   A score of 0.2756 suggests the clusters are somewhat distinct but might have some overlap or points close to boundaries. Higher is generally better, but the interpretation depends on the data.
*   **Calinski-Harabasz Score: 9232.6770**
    *   **What it is:** Also known as the Variance Ratio Criterion. It measures the ratio of the variance between clusters to the variance within clusters. Higher scores indicate better-defined clusters (more separation between clusters and less variance within clusters).
    *   **Interpretation:** A higher score generally indicates better clustering. There's no absolute scale, so it's often used to compare different clustering results on the *same* data. 9232 seems relatively high, suggesting good separation.
*   **Number of Clusters: 3**
    *   **What it is:** The number of distinct groups the model identified in the data.
    *   **Interpretation:** The model grouped the data points (likely stores or customers based on sales patterns) into 3 segments.
*   **Cluster Sizes (min, avg, max): min=5627, avg=6083.3, max=6498**
    *   **What it is:** Statistics about the number of data points assigned to each cluster.
    *   **Interpretation:** The smallest cluster has 5627 members, the largest has 6498, and the average size is around 6083. This indicates relatively balanced cluster sizes.

## 3. Sentiment Model (Logistic Regression)

These metrics evaluate how well the model classifies product reviews into 'positive', 'neutral', or 'negative' categories.

*   **Accuracy: 1.0000**
    *   **What it is:** The overall proportion of reviews that were correctly classified.
    *   **Interpretation:** An accuracy of 1.0 means 100% of the reviews in the test set were classified correctly. *Note: An accuracy of 1.0 is often suspicious and might indicate overfitting or issues with the test data (e.g., data leakage). It's worth investigating further.*
*   **Classification Report:** Provides more detailed metrics for each class (negative, neutral, positive):
    *   **Precision:** Of all the reviews the model *predicted* as belonging to a certain class, what proportion actually belonged to that class? (e.g., Precision for 'positive' = TP / (TP + FP)). High precision means fewer false positives.
    *   **Recall (Sensitivity):** Of all the reviews that *actually* belonged to a certain class, what proportion did the model correctly identify? (e.g., Recall for 'positive' = TP / (TP + FN)). High recall means fewer false negatives.
    *   **F1-score:** The harmonic mean of Precision and Recall (2 * (Precision * Recall) / (Precision + Recall)). It provides a single score balancing both concerns. Useful when class distribution is uneven.
    *   **Support:** The number of actual occurrences of the class in the test dataset.
*   **Interpretation (based on the 1.0 scores):** The model achieved perfect precision, recall, and F1-score for all three sentiment classes on this test set. As mentioned, this perfect score warrants a closer look at the data splitting and training process to ensure it's not an artifact of the setup.

### Classification report:
              precision    recall  f1-score   support

    negative       1.00      1.00      1.00        33
     neutral       1.00      1.00      1.00        50
    positive       1.00      1.00      1.00       117

    accuracy                           1.00       200
    macro avg      1.00      1.00      1.00       200
    weighted avg   1.00      1.00      1.00       200
