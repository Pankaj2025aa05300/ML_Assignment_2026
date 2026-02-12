**Problem Statement**

This assignment aims to implement and comparatively analyze six supervised machine learning classification algorithms using the Dry Bean Dataset from the UCI Machine Learning Repository. The dataset is a publicly available benchmark commonly used for multi-class classification research in agricultural analytics and food quality assessment. The dataset consists of morphological and geometric features extracted from dry bean samples belonging to seven distinct classes. The primary objective is to develop predictive models capable of accurately classifying bean varieties based on quantitative shape-related attributes.Multi-class classification in agricultural datasets presents several challenges, including High inter-feature correlation,Potential class overlap,Variability in feature distributions,Non-linear relationships between features. Therefore, this assignment emphasizes both model implementation and systematic benchmarking across diverse algorithmic paradigms. The selected models represent different learning approaches: 

Linear Model: Logistic Regression

Tree-Based Model: Decision Tree

Distance-Based Method: K-Nearest Neighbors

Probabilistic Model: Naive Bayes

Ensemble – Bagging: Random Forest

Ensemble – Boosting: XGBoost

Performance is evaluated using multiple metrics to ensure comprehensive assessment and robust comparison.

**Dataset Description**

•	The dataset used in this assignment is the Dry Bean Dataset obtained from the UCI Machine Learning Repository. It is publicly available for academic and research purposes.
Dataset URL: https://archive.ics.uci.edu/ml/datasets/Dry+Bean+Dataset

•	The dataset contains 13,611 instances, where each instance represents a single dry bean sample. Each sample is described using 16 numerical morphological and geometric features extracted through image processing techniques.

•	The dataset includes 7 distinct bean classes, making it a multi-class classification problem. The objective is to correctly classify each bean sample based on its morphological characteristics.

•	The dataset was divided into 90% training (12,249 samples) and 10% testing (1,362 samples). Feature scaling was applied to normalize numerical attributes and improve model performance.

**Models Implemented**

•	Logistic Regression

•	Decision Tree

•	K-Nearest Neighbors (KNN)

•	Naive Bayes (GaussianNB)

•	Random Forest (Ensemble – Bagging)

•	XGBoost (Ensemble - Boosting)


**Evaluation Metrics Used**

•	Accuracy

•	AUC Score (One-vs-Rest for multi-class)

•	Precision (Weighted)

•	Recall (Weighted)

•	F1 Score (Weighted)

•	Matthews Correlation Coefficient (MCC)


**Model Comparison Table**

| Model               | Accuracy   | AUC Score | Precision | Recall | F1 Score | MCC Score  |
| ------------------- | ---------- | --------- | --------- | ------ | -------- | ---------- |
| XGBoost             | 0.9324     | 0.9964    | 0.9324    | 0.9324 | 0.9323   | 0.9183     |
| Logistic Regression | 0.9310     | 0.9960    | 0.9312    | 0.9310 | 0.9310   | 0.9165     |
| KNN                 | 0.9266     | 0.9917    | 0.9270    | 0.9266 | 0.9266   | 0.9112     |
| Random Forest       | 0.9244     | 0.9953    | 0.9249    | 0.9244 | 0.9245   | 0.9086     |
| Naive Bayes         | 0.9001     | 0.9938    | 0.9014    | 0.9001 | 0.9002   | 0.8796     |
| Decision Tree       | 0.8979     | 0.9679    | 0.8980    | 0.8979 | 0.8979   | 0.8765     |


**Observations**

| ML Model            | Performance Observation |
|----------           |--------------------------|
| Logistic Regression | Demonstrated strong and stable performance (93.10% accuracy) with high AUC. The results suggest that the dataset exhibits relatively well-separated and discriminative feature patterns, enabling effective linear classification. |
| Decision Tree       | Achieved moderate accuracy (89.79%). While interpretable, it showed signs of overfitting compared to ensemble methods. |
| K-Nearest Neighbors (KNN) | Performed competitively (92.66%) after feature scaling. As a distance-based method, its performance was influenced by normalized feature space representation. |
| Naive Bayes | Computationally efficient but achieved lower accuracy (90.01%). The independence assumption among features may limit its performance on correlated morphological attributes. |
| Random Forest (Ensemble – Bagging) | Improved generalization compared to a single Decision Tree by reducing variance. Delivered stable and strong predictive performance (92.44%). |
| XGBoost (Ensemble – Boosting) | Achieved the highest overall performance across most evaluation metrics, including Accuracy and MCC. Boosting effectively captured complex feature interactions and minimized classification errors. |


