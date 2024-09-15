# Decision Tree Visualization

## Description
This project demonstrates how to train a **Decision Tree Classifier** on the Iris dataset and visualize the tree structure, showing how decisions are made at each node. The project also visualizes the importance of each feature in the classification process. The visualization includes a decision tree diagram that outlines the split points and final classifications.

## Features
- **Decision Tree Classifier**: Implements a decision tree classifier using the **Gini** criterion.
- **Tree Visualization**: Visualizes the decision tree structure using `graphviz`.
- **Feature Importance Plot**: Displays the importance of each feature using a bar chart.
- **Dataset**: Uses the **Iris dataset**, a popular dataset in machine learning for classification tasks.

## Libraries
- `scikit-learn`: For the decision tree classifier and dataset.
- `graphviz`: To render the decision tree structure.
- `matplotlib`: To plot the feature importance.

### Installation
You can install the required libraries by running the following command:

```bash
pip install numpy matplotlib scikit-learn graphviz
```

To install **Graphviz**, which is required for visualizing the tree:
- On **Linux** or **Ubuntu**:
  ```bash
  sudo apt-get install graphviz
  ```
- On **MacOS**:
  ```bash
  brew install graphviz
  ```
- On **Windows**, download and install Graphviz from the [official website](https://graphviz.gitlab.io/download/).

## Dataset
The project uses the **Iris dataset** from `scikit-learn`. The Iris dataset contains 150 samples of iris flowers, categorized into 3 species:
- Setosa
- Versicolor
- Virginica

Each sample includes the following features:
- Sepal length
- Sepal width
- Petal length
- Petal width

The goal is to classify the iris species based on these features.

## How It Works

1. **Train a Decision Tree Classifier**:
   - The Decision Tree model is trained using the **Gini criterion**, which measures the quality of splits.
   - The depth of the tree is limited to **3 levels** to prevent overfitting and ensure clarity in visualization.

2. **Tree Visualization**:
   - The tree structure is visualized using `graphviz`. The tree diagram shows the following:
     - **Splits**: The decision rules at each node.
     - **Classes**: The predicted class at the leaf nodes.
     - **Samples**: The number of samples at each node.
     - **Gini Index**: The measure of impurity at each node.
   
3. **Feature Importance Plot**:
   - After training the model, a bar chart is generated to display the importance of each feature in the classification process.

## Code Explanation

### 1. Data Loading and Preprocessing
We load the **Iris dataset** from `scikit-learn` and split it into training and testing sets using `train_test_split`.

```python
from sklearn import datasets
from sklearn.model_selection import train_test_split

# Load dataset
iris = datasets.load_iris()
X = iris.data  # Features
y = iris.target  # Target labels

# Split dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
```

- **`X`**: The feature matrix, which contains 4 features per sample.
- **`y`**: The target labels, representing the species of the iris flowers.
- **`train_test_split`**: Splits the dataset into 80% training and 20% testing.

### 2. Train the Decision Tree Classifier
We initialize the decision tree classifier with the **Gini** criterion and limit the depth to 3.

```python
from sklearn.tree import DecisionTreeClassifier

# Initialize the Decision Tree Classifier
dtree = DecisionTreeClassifier(criterion='gini', max_depth=3, random_state=42)

# Train the model
dtree.fit(X_train, y_train)
```

- **`criterion='gini'`**: The Gini criterion measures the impurity at each node.
- **`max_depth=3`**: The maximum depth of the tree to prevent overfitting and make the tree more interpretable.

### 3. Visualize the Decision Tree
We use `graphviz` to visualize the tree structure. The `tree.export_graphviz()` function generates the DOT-format data, which `graphviz.Source()` uses to render the visualization.

```python
from sklearn import tree
import graphviz

# Visualize the decision tree
def visualize_tree(model, feature_names, class_names):
    dot_data = tree.export_graphviz(model, out_file=None, 
                                    feature_names=feature_names,  
                                    class_names=class_names,  
                                    filled=True, rounded=True,  
                                    special_characters=True)  
    graph = graphviz.Source(dot_data)  
    return graph

# Render the tree and save it as a PDF
graph = visualize_tree(dtree, feature_names=iris.feature_names, class_names=iris.target_names)
graph.render("iris_decision_tree")
```

- **`tree.export_graphviz()`**: Exports the decision tree in a DOT format for visualization.
- **`graphviz.Source()`**: Renders the DOT data as a tree diagram and saves it as `iris_decision_tree.pdf`.

### 4. Plot Feature Importance
We also visualize the feature importance using a horizontal bar chart.

```python
import matplotlib.pyplot as plt

# Plot the feature importance
def plot_feature_importance(model, feature_names):
    plt.figure(figsize=(8, 6))
    plt.barh(feature_names, model.feature_importances_)
    plt.xlabel("Feature Importance")
    plt.ylabel("Features")
    plt.title("Feature Importance of Decision Tree Classifier")
    plt.show()

# Plot feature importance
plot_feature_importance(dtree, iris.feature_names)
```

- **`model.feature_importances_`**: Provides the importance of each feature based on the decision tree splits.
- **`plt.barh()`**: Creates a horizontal bar chart to show feature importance.

## Output
1. **Decision Tree Visualization**: The decision tree is rendered as a PDF file named `iris_decision_tree.pdf`. This file contains the complete structure of the decision tree, with node splits, Gini index, and predicted classes.
2. **Feature Importance Plot**: A horizontal bar chart shows the contribution of each feature (sepal length, sepal width, petal length, petal width) in the classification process.

## How to Run

1. **Clone the Repository or Copy the Code**:
   Download or copy the code to your local machine.

2. **Install the Required Libraries**:
   Run the following command to install the necessary dependencies:
   ```bash
   pip install numpy matplotlib scikit-learn graphviz
   ```

3. **Run the Python Script**:
   Execute the script in your Python environment:
   ```bash
   python decision_tree_visualization.py
   ```

4. **Check the Output**:
   - The decision tree visualization will be saved as `iris_decision_tree.pdf`.
   - A bar chart will be displayed, showing the feature importance.

   ![Output1.png](https://github.com/AartiDashore/DecisionTreeVisualization/blob/main/Output1.png)

## Customization
- You can adjust the `max_depth` parameter in the Decision Tree to allow deeper splits.
- You can change the dataset by loading different data or using other splitting criteria like `entropy`.
- Modify the script to handle multiclass classification tasks by using other datasets with more than two classes.

## Concepts Explained

- **Decision Tree**: A tree-like model used for classification and regression. It splits the dataset based on features, recursively forming a structure of decision nodes and leaf nodes.
- **Splitting Criteria**: The Gini impurity or entropy is used to evaluate the quality of each split.
- **Feature Importance**: Shows the relevance of each feature for decision-making within the tree.

## Conclusion
This project provides a detailed visualization of how a Decision Tree Classifier works. It helps you understand the decision-making process within the tree and how important features are in determining the target labels.
