# Machine-Learning

### **HW1: Basics of Machine Learning**
1. Basics of Machine Learning
2. The Learning Problems
3. Feasibility of Learning
4. Experiments with Perceptron Learning Algorithm

### **HW2: Generalization & Decision Stumps**
1. Theory of Generalization & Decision Stumps
2. Linear Models
3. Beyond Gradient Descent
4. Experiments with Decision Stumps

### **HW3: Linear Models and Regularization**
1. Linear Models and More
2. Playing with Regularization
3. Virtual Examples and Regularization
4. Experiments with Linear and Nonlinear Models

### **HW4: Support Vector Machines**
1. More about Regularization
2. Validation
3. Support Vector Machine
4. Experiments with Regularized Logistic Regression

### **HW5: Bagging and Boosting**
1. Support Vector Machines
2. Bagging and Boosting
3. Experiments with Soft-Margin SVM and AdaBoost

---

## Final Project: Predicting Danceability with Machine Learning

### **Introduction**
The project explores different machine learning approaches for predicting the danceability of music tracks. The focus is on evaluating the performance of multiple models by preprocessing the data and experimenting with various techniques. The final goal is to achieve a balance between model complexity and accuracy.

### **Data Preprocessing**
- **Dealing with Missing Values**: 
  - Fill missing values with the median to avoid outliers’ influence.
- **Text Features (Artist, Composer, etc.)**: 
  - Calculated mean danceability for each artist/composer. Sparse data were treated as missing values.
- **Standardization**: 
  - Numerical features were standardized to balance their influence and enhance model performance.

### **Feature Selection**
- **Correlation Analysis**: Examined correlation between features like energy, liveness, and danceability. 
- **Experimentation**: Selected features based on performance, testing combinations of numerical values, artist, composer, and other attributes.

### **Modeling Approaches**
1. **Linear Regression**: 
   - Simple and efficient, performed well after feature selection.
2. **Ridge Regression**: 
   - Applied regularization, but improvement was minimal.
3. **Support Vector Regression (SVR)**: 
   - Significant improvement with the RBF kernel and fine-tuned regularization parameters.
4. **Random Forest**: 
   - Tended to overfit; required careful tuning of depth and sample leaf size.
5. **Neural Networks**: 
   - Best performance overall, capturing complex relationships in the data.

### **Dealing with Overfitting**
- Cross-validation and AdaBoost were applied, but with limited success in reducing the gap between training and test errors.

### **Best Model**
- The **Neural Network Model** outperformed others with an `Eout = 1.84`, proving the ability to model nonlinear relationships more effectively.

### **Other Adjustments**
- **Blending**: Combined predictions from multiple models, leading to the optimal result of `Eout = 1.83`.

### **Conclusion**
Preprocessing text features and blending models were critical for improving performance. While simpler models like Linear Regression provided reasonable results, the Neural Network was the best at capturing complex patterns.
