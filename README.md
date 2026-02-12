**Problem Statement**

This assignment task aims to implement and comparatively analyze six supervised machine learning classification algorithms using the Dry Bean Dataset, obtained from the UCI Machine Learning Repository. It is a publicly available benchmark dataset commonly used for multi-class classification research in agricultural and food quality analysis. The dataset consists of morphological and geometric features extracted from dry bean samples belonging to seven distinct classes. The primary objective is to develop predictive models capable of accurately classifying bean varieties based on these quantitative attributes. Multi-class classification problems in agricultural datasets present several challenges, including high inter-feature correlation, potential class overlap, and variability in feature distributions. Selecting an appropriate model requires careful evaluation of both predictive performance and generalization capability. Therefore, this assignment not only focuses on model implementation but also emphasizes systematic performance benchmarking across diverse algorithmic approaches. The selected models represent different learning paradigms, including linear models (Logistic Regression), tree-based models (Decision Tree), distance-based methods (K-Nearest Neighbors), probabilistic models (Naive Bayes), and ensemble techniques such as bagging (Random Forest) and boosting (XGBoost). By comparing these models using multiple evaluation metrics—Accuracy, AUC, Precision, Recall, F1 Score, and Matthews Correlation Coefficient (MCC)—the assignment aims to identify the most robust and reliable classification approach for the given dataset. Finally, this comparative analysis provides insights into the effectiveness of ensemble learning methods relative to individual classifiers in solving complex multi-class classification problems.

**Dataset Description**

•	The dataset used in this assignment is the Dry Bean Dataset obtained from the UCI Machine Learning Repository. It is publicly available for academic and research purposes.
Dataset URL: https://archive.ics.uci.edu/ml/datasets/Dry+Bean+Dataset

•	The dataset contains 13,611 instances, where each instance represents a single dry bean sample. Each sample is described using 16 numerical morphological and geometric features extracted through image processing techniques.

•	The dataset includes 7 distinct bean classes, making it a multi-class classification problem. The objective is to correctly classify each bean sample based on its morphological characteristics.

•	The dataset was divided into 80% training (10,888 samples) and 20% testing (2,723 samples). Feature scaling was applied to normalize numerical attributes and improve model performance.

**Models Implemented**

•	Logistic Regression

•	Decision Tree

•	K-Nearest Neighbors (KNN)

•	Naive Bayes (GaussianNB)

•	Random Forest (Ensemble)

•	XGBoost (Ensemble Boosting)


**Evaluation Metrics Used**
•	Accuracy

•	AUC Score

•	Precision (Weighted)

•	Recall (Weighted)

•	F1 Score (Weighted)

•	Matthews Correlation Coefficient (MCC)


**Model Comparison Table**

| Model Name            | Accuracy | AUC     | Precision| Recall  | F1 Score | MCC     |
|-----------------------|----------|---------|----------|---------|----------|---------|
| XGBoost               | 0.9258   | 0.9954  | 0.9258   | 0.9258  | 0.9258   | 0.9103  |
| Logistic Regression   | 0.9207   | 0.9948  | 0.9215   | 0.9207  | 0.9209   | 0.9042  |
| Random Forest         | 0.9192   | 0.9939  | 0.9195   | 0.9192  | 0.9192   | 0.9023  |
| KNN                   | 0.9166   | 0.9833  | 0.9174   | 0.9166  | 0.9168   | 0.8992  |
| Decision Tree         | 0.9023   | 0.9595  | 0.9026   | 0.9023  | 0.9024   | 0.8819  |
| Naive Bayes           | 0.8979   | 0.9916  | 0.9007   | 0.8979  | 0.8981   | 0.8773  |

**Observations**

| ML Model            | Performance Observation |
|----------           |--------------------------|
| Logistic Regression | Achieved high accuracy (92.07%) and strong AUC, indicating good linear separability among bean classes. Demonstrated stable and consistent performance across categories. |
| Decision Tree       | Produced moderate accuracy (90.23%). Although interpretable, it showed signs of slight overfitting compared to ensemble methods. |
| K-Nearest Neighbors (KNN) | Performed well (91.66%) after feature scaling. Sensitive to distance metrics but delivered competitive results. |
| Naive Bayes | Computationally efficient but achieved the lowest accuracy (89.79%). Performance may be affected due to the assumption of feature independence. |
| Random Forest (Ensemble – Bagging) | Improved generalization over Decision Tree by reducing variance. Delivered strong and stable performance (91.92%). |
| XGBoost (Ensemble – Boosting) | Achieved the best overall performance (92.58% accuracy, highest MCC). Boosting effectively minimized classification errors and enhanced predictive power. |


