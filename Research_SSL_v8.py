#!/usr/bin/env python
# coding: utf-8

# In[1]:


# !pip install pandas numpy==1.24.3 SciPy==1.10.0 scikit-learn==1.3.0 matplotlib==3.7.2 seaborn==0.12.2 openpyxl==3.0.10 networkx==3.1


# In[2]:


# # =======================
# # Mount Google Drive
# # =======================
# from google.colab import drive
# drive.mount('/content/drive')


# In[3]:


# import the necessary libraries
import pandas as pd
import numpy as np
import scipy
import sklearn
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import openpyxl
import networkx as nx
import random
from sklearn.model_selection import GridSearchCV, train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.semi_supervised import LabelSpreading
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


# In[4]:


# let's check the version of the libraies

# Check the versions of the libraries
print("Pandas version:", pd.__version__)
print("NumPy version:", np.__version__)
print("SciPy version:", scipy.__version__)
print("Scikit-learn version:", sklearn.__version__)
print("Matplotlib version:", matplotlib.__version__)
print("Seaborn version:", sns.__version__)
print("OpenPyXL version:", openpyxl.__version__)
print("NetworkX version:", nx.__version__)


# In[5]:


# file path
input_path = "data/maternal_pregnancy_risk_dataset.csv"


# load the dataset
df = pd.read_csv(input_path)


# In[6]:


df.head()


# In[7]:


# Rename the column "Gestational " to "Gestational" to remove the extra space
df.rename(columns={'Gestational ': 'Gestational'}, inplace=True)


# In[8]:


df.columns


# In[9]:


df.shape


# In[10]:


df.info()


# In[11]:


df.describe()


# In[12]:


df.isnull().sum()


# In[13]:


# Since the value of both columns 'Fetal Movement' and 'Jaundice' is zero (0) we will drop these columns.
print(df['Fetal Movement'].unique())
print(df['Jaundice'].unique())


# In[14]:


# Drop the "Urine test albumin" column since the najority of the values of this column are NaN.
df = df.drop(columns=["Urine test albumin", "Fetal Movement", "Jaundice"])

# Fill the anemia column with foraward fill method
df.ffill(inplace=True)


# In[15]:


df.isnull().sum()


# In[16]:


df.shape


# In[17]:


df.columns


# ## **Patient Similarity Network Based on BMI**

# In[18]:


# Create a unique ID for each patient
df['ID'] = df.index

# Calculate BMI (Weight in kg / (Height in meters)^2)
# Assuming 'Height in meter' is in meters and 'Weight' is in kg
df['BMI'] = df['Weight'] / (df['Height in meter'] ** 2)

# Create a graph
G = nx.Graph()

# Add nodes for each patient
for index, row in df.iterrows():
    G.add_node(row['ID'], age=row['Age'], bmi=row['BMI'])

# Add edges based on some criteria (e.g., similar BMI)
for i in range(len(df)):
    for j in range(i + 1, len(df)):
        if abs(df.loc[i, 'BMI'] - df.loc[j, 'BMI']) < 2.0:  # example condition for similarity
            G.add_edge(df.loc[i, 'ID'], df.loc[j, 'ID'])

# Display the graph info
print("Nodes in the graph:", G.nodes(data=True))
print("Edges in the graph:", G.edges())


# ## **Visualization of Patient Similarity Network Based on BMI**

# In[19]:


# Set figure size
plt.figure(figsize=(12, 8))

# Choose a layout for the graph
pos = nx.spring_layout(G, k=0.3, iterations=50)  # Adjusted for better spacing

# Draw nodes with a specific color and size
nx.draw_networkx_nodes(G, pos, node_size=300, node_color='skyblue', alpha=0.7)

# Draw edges
nx.draw_networkx_edges(G, pos, width=1.0, alpha=0.5, edge_color='gray')

# # Optional: Label nodes with 'ID' or attributes
# labels = {node: f"ID: {node}\nAge: {data['age']}\nBMI: {data['bmi']:.1f}"
#           for node, data in G.nodes(data=True)}
# nx.draw_networkx_labels(G, pos, labels, font_size=8, font_family="sans-serif")

# Set title and remove axes
plt.title("Patient Similarity Network (Connected by BMI Δ < 2.0)", size=16)
plt.axis("off")  # Remove axes
plt.tight_layout()
plt.show()


# # **1. Distribution Plots**

# ## a. Enhanced Box Plot (Age vs Risk Status)

# In[20]:


plt.figure(figsize=(10, 6))
sns.boxplot(x='High-risk pregnancy', y='Age', data=df, palette=['#66c2a5','#fc8d62'])
sns.stripplot(x='High-risk pregnancy', y='Age', data=df, color='black', alpha=0.3, jitter=0.2)
plt.title('Age Distribution by Pregnancy Risk Status', fontsize=14)
plt.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.show()


# ## b. Violin Plot (BMI Distribution)

# In[21]:


plt.figure(figsize=(10, 6))
sns.violinplot(x='High-risk pregnancy', y='BMI', data=df, palette=['#9ecae1','#fdae6b'])
plt.title('BMI Distribution by Pregnancy Risk Status', fontsize=14)
plt.grid(axis='y', alpha=0.3)
plt.show()


# # **2. Relationship Plots**

# ## a. Scatter Plot Matrix

# In[22]:


sns.pairplot(df[['Age', 'BMI', 'Systolic', 'Diastolic', 'High-risk pregnancy']],
             hue='High-risk pregnancy', palette='Set2', corner=True)
plt.suptitle('Pairwise Relationship Plot', y=1.02)
plt.show()


# ## b. Correlation Heatmap

# In[23]:


plt.figure(figsize=(12, 8))
sns.heatmap(df.corr(numeric_only=True), annot=True, fmt=".2f", cmap='coolwarm', center=0)
plt.title('Feature Correlation Matrix', fontsize=14)
plt.xticks(rotation=45, ha="right")
plt.tight_layout()
plt.show()


# ## **3. Categorical Analysis**

# ## a. Stacked Bar Chart (Risk by Age Group)

# In[24]:


df['Age Group'] = pd.cut(df['Age'], bins=[15,25,30,35,40,45])
risk_by_age = df.groupby(['Age Group', 'High-risk pregnancy']).size().unstack()
risk_by_age.plot(kind='bar', stacked=True, color=['#66c2a5','#fc8d62'])
plt.title('Pregnancy Risk Distribution by Age Group')
plt.ylabel('Count')
plt.xticks(rotation=45)
plt.show()


# ## **4. Advanced Visualizations**

# ## a. PCA Projection (2D)

# In[25]:


X = df.select_dtypes(include=[np.number]).drop('High-risk pregnancy', axis=1)
X = StandardScaler().fit_transform(X)
pca = PCA(n_components=2)
principal_components = pca.fit_transform(X)

plt.figure(figsize=(10,6))
sns.scatterplot(x=principal_components[:,0], y=principal_components[:,1],
                hue=df['High-risk pregnancy'], palette='Set2', alpha=0.7)
plt.title('2D PCA Projection of Health Indicators', fontsize=14)
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.legend(title='High-risk Pregnancy')
plt.grid(alpha=0.3)
plt.show()


# ## b. Time-Series Trend

# In[26]:


# Plot of Blood Pressure by Gestational time
plt.figure(figsize=(12, 6))
sns.lineplot(x='Gestational', y='Systolic', hue='High-risk pregnancy',
             data=df, errorbar=('ci', 95), palette='Set2')
plt.title('Blood Pressure Trends by Gestational Time', fontsize=14)
plt.grid(alpha=0.3)
plt.show()


# # **Random Forest with Gridsearch CV**

# ## Separate features (X) and target (y)

# In[27]:


# Separate features (X) and target (y)
X = df.drop(['High-risk pregnancy', 'Age Group'], axis=1)  # Drop target, and Age Group columns
y = df['High-risk pregnancy']  # Target column


# ## Check again for missing values

# In[28]:


# Check for missing values
print("Missing values in features:\n", X.isnull().sum())


# ## Randomly mask some labels to simulate unlabeled data

# In[29]:


# Set the random seed for reproducibility
random.seed(42)  # You can choose any integer value as the seed

# Randomly mask some labels to simulate unlabeled data
mask_percentage = 0.3  # 30% of the labels will be masked
n_samples = len(y)
n_masked = int(n_samples * mask_percentage)

# Randomly select indices to mask
masked_indices = random.sample(range(n_samples), n_masked)

# Create a copy of y to mask labels
y_masked = y.copy()
for idx in masked_indices:
    y_masked.iloc[idx] = -1  # Set to -1 to indicate unlabeled


# ## Store the masked indices for later use

# In[30]:


# Store the masked indices for later use
masked_indices_list = masked_indices
# Print the masked indices and corresponding y values
print("Masked Indices:", masked_indices_list)
print("\n Masked y values:", y.iloc[masked_indices_list].values)


# ## Split into train (70%) and test (30%) sets and filterout the unlabeled data

# In[31]:


# Combine labeled and unlabeled data
X_combined = X.copy()
y_combined = y_masked.copy()

# Split into train (70%) and test (30%) sets, including unlabeled data
X_train, X_test, y_train, y_test = train_test_split(
    X_combined,
    y_combined,
    test_size=0.3,    # 30% test data
    random_state=42,  # For reproducibility
    stratify=None      # We do not stratify since we have unlabeled data
)

# Verify shapes
print("Training shapes:", X_train.shape, y_train.shape)
print("Testing shapes:", X_test.shape, y_test.shape)


# In[32]:


X_train.columns


# ## Initialize the StandardScaler and fit the scaler on the training and test sets

# In[33]:


# Initialize the StandardScaler
scaler = StandardScaler()

# Fit the scaler on the training data and transform both training and test sets
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


# ## Define the parameter grid for hyperparameter tuning

# In[34]:


# Define the parameter grid for hyperparameter tuning
param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [None, 5, 10, 20],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'max_features': ['sqrt', 'log2']
}


# ## Initialize the Random Forest model

# In[35]:


# Initialize the Random Forest model
rf = RandomForestClassifier(class_weight='balanced', random_state=42)


# # Initialize GridSearchCV

# In[36]:


# Initialize GridSearchCV
grid_search = GridSearchCV(
    estimator=rf,
    param_grid=param_grid,
    cv=2,  # 2-fold cross validation
    n_jobs=-1,  # Use all available cores
    scoring='accuracy',
    verbose=2
)


# ## Fit the Gridsearch on the training data

# In[37]:


# Fit the Gridsearch on the training data
grid_search.fit(X_train_scaled[y_train != -1], y_train[y_train != -1])


# ## Make prediction using the best parameters

# In[38]:


# Print the best parameters and best score
print("Best Parameters:", grid_search.best_params_)
print("Best Accuracy Score:", round(grid_search.best_score_,2))

# Evaluate on test set using the best model
best_rf = grid_search.best_estimator_

# Make prediction
y_pred = best_rf.predict(X_test_scaled)

print(f"\n {y_pred}")


# ## Perform cross-validation on the best model

# In[39]:


# Perform cross-validation on the best model
cv_scores = cross_val_score(best_rf, X_train_scaled[y_train != -1], y_train[y_train != -1], cv=3)  # 3-fold cross-validation

# Print cross-validation results
print("Cross-Validation Scores:", list(cv_scores))
print("Mean Cross-Validation Score:", round(cv_scores.mean(),2))


# # **Prediction of Materinal Risk**

# In[56]:


# Make predictions
predictions = best_rf.predict(X_test_scaled)

# Calculate accuracy
accuracy = accuracy_score(y_test[y_test != -1], predictions[y_test != -1])

# Generate classification report
class_report = classification_report(y_test[y_test != -1], predictions[y_test != -1])

# Print the results
print(f"Accuracy: {accuracy * 100:.2f}%")

print("\nClassification Report:")
print(class_report)


# In[57]:


# Create confusion matrix
cm = confusion_matrix(y_test[y_test != -1], predictions[y_test != -1])
print("Confusion Matrix:\n", cm)
# Visualize confusion matrix
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Class 0', 'Class 1'],
            yticklabels=['Class 0', 'Class 1'])
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.title('Confusion Matrix')
plt.show()

