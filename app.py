# pip install streamlit

import streamlit as st 
import pandas as pd
import matplotlib.pyplot as plt

# st.title("My first Streamlit App")
st.header("Logistic Regression Model")
st.subheader("Pumpkin Dataset")

st.sidebar.title("My first Streamlit App")
pumpkins = pd.read_csv('US-pumpkins.csv')
st.markdown("## The real Data :coffee:")
st.dataframe(pumpkins)
new_columns = ['Color','Origin','Item Size','Variety','City Name','Package']
new_pumpkins = pumpkins.drop([c for c in pumpkins.columns if c not in new_columns], axis=1)
new_pumpkins.dropna(inplace=True)
st.markdown("## The manipulated Data :recycle:")
st.dataframe(new_pumpkins)
from sklearn.preprocessing import LabelEncoder
# # https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.LabelEncoder.html

new_pumpkins = new_pumpkins.apply(LabelEncoder().fit_transform)
st.subheader("Label encoded data :rocket:")
st.dataframe(new_pumpkins)

import seaborn as sns
# g = sns.PairGrid(new_pumpkins)
# g.map(sns.scatterplot)
# # Streamlit plotting function
# st.pyplot(g)

fig = plt.figure(figsize=(2, 2))
sns.swarmplot(x="Color", y="Item Size", data=new_pumpkins)
st.pyplot(fig)

fig1 = sns.catplot(x="Color", y="Item Size",
            kind="violin", data=new_pumpkins)
st.pyplot(fig1)
from sklearn.model_selection import train_test_split
    
Selected_features = ['Origin','Item Size','Variety','City Name','Package']

X = new_pumpkins[Selected_features]
y = new_pumpkins['Color']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report 
from sklearn.linear_model import LogisticRegression

model = LogisticRegression()
model.fit(X_train, y_train)
predictions = model.predict(X_test)

st.write(classification_report(y_test, predictions))
st.write('Predicted labels: ', predictions)
st.write('Accuracy: ', accuracy_score(y_test, predictions))

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, predictions)
print(cm)

fig3 = plt.figure(figsize=(4,4))
sns.heatmap(cm, annot=True, fmt=".3f", linewidths=.5, square = True, cmap = 'Blues_r')
st.pyplot(fig3)


from sklearn.metrics import roc_curve, roc_auc_score

y_scores = model.predict_proba(X_test)
# calculate ROC curve
# fpr, tpr, thresholds = roc_curve(y_test, y_scores[:,1])
# fig5 = sns.lineplot([0, 1], [0, 1])
# sns.lineplot(fpr, tpr)

# st.pyplot(fig5)