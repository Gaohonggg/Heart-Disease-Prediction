import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
from xgboost import XGBClassifier
from sklearn.ensemble import StackingClassifier

df = pd.read_csv("cleveland.csv",header=None)
df.columns = ["age","sex","cp","trestbps","chol","fbs",
              "restecg","thalach","exang","oldpeak","slope",
              "ca","thal","target"]
df["target"] = df["target"].map({0:0,1:1,2:1,3:1,4:1})
df["thal"] = df["thal"].fillna( df["thal"].mean() )
df["ca"] = df["ca"].fillna( df["ca"].mean() )

# Vẽ quan hệ giữa age và target sử dụng seaborn
sns.set_context("paper",font_scale=1,rc={"font.size":3,"axes.titlesize":15,"axes.labelsize":10})
ax = sns.catplot(kind="count",data=df,x="age",hue="target",order=df["age"].sort_values().unique())
ax.ax.set_xticks( np.arange(0,80,5) )
plt.title("Variation of Age for each target class")
plt.show()

# Vẽ quan hệ giữa age và sex sử dụng seaborn
sns.catplot(kind="bar",data=df,y="age",x="sex",hue="target")
plt.title("Distribution of age vs sex with the target class")
plt.show()

# Chia tập dữ liệu
X = df.drop( columns=["target"] )
y = df["target"]
X_train, X_test, y_train, y_test = train_test_split(X,y,
                                                    test_size=0.2,
                                                    random_state=42)

# KNN
knn = KNeighborsClassifier(n_neighbors=5,
                           weights="uniform",
                           algorithm="auto",
                           leaf_size=30,
                           p=2,
                           metric="minkowski")
knn.fit(X_train,y_train)
knn_y_predict = knn.predict(X_test)

knn_ac_train = accuracy_score(knn.predict(X_train),y_train)
knn_ac_test = accuracy_score(knn_y_predict,y_test)
print("Accuracy for knn training: {}".format(round(knn_ac_train,2)))
print("Accuracy for knn test: {}".format(round(knn_ac_test,2)))
print("------------------------------------------------------")

# SVM
svm = SVC(kernel="rbf",
          random_state=42)
svm.fit(X_train,y_train)
svm_y_predict = svm.predict(X_test)

svm_ac_train = accuracy_score(svm.predict(X_train),y_train)
svm_ac_test = accuracy_score(svm_y_predict,y_test)
print("Accuracy for svm training: {}".format(round(svm_ac_train,2)))
print("Accuracy for svm test: {}".format(round(svm_ac_test,2)))
print("------------------------------------------------------")

# Naive Bayes
nb = GaussianNB()
nb.fit(X_train,y_train)
nb_y_predict = nb.predict(X_test)

nb_ac_train = accuracy_score(nb.predict(X_train),y_train)
nb_ac_test = accuracy_score(nb_y_predict,y_test)
print("Accuracy for nb training: {}".format(round(nb_ac_train,2)))
print("Accuracy for nb test: {}".format(round(nb_ac_test,2)))
print("------------------------------------------------------")

# Decision Tree
dt = DecisionTreeClassifier(criterion="gini",
                            max_depth=10,
                            min_samples_split=2)
dt.fit(X_train,y_train)
dt_y_predict = dt.predict(X_test)

dt_ac_train = accuracy_score(dt.predict(X_train),y_train)
dt_ac_test = accuracy_score(dt_y_predict,y_test)
print("Accuracy for dt training: {}".format(round(dt_ac_train,2)))
print("Accuracy for dt test: {}".format(round(dt_ac_test,2)))
print("------------------------------------------------------")

# Random Forest
rf = RandomForestClassifier(criterion="gini",
                            max_depth=10,
                            min_samples_split=2,
                            n_estimators=10,
                            random_state=42)
rf.fit(X_train,y_train)
rf_y_predict = rf.predict(X_test)

rf_ac_train = accuracy_score(rf.predict(X_train),y_train)
rf_ac_test = accuracy_score(rf_y_predict,y_test)
print("Accuracy for rf training: {}".format(round(rf_ac_train,2)))
print("Accuracy for rf test: {}".format(round(rf_ac_test,2)))
print("------------------------------------------------------")

# AdaBoost
ab = AdaBoostClassifier(n_estimators=50,
                        learning_rate=1.0)
ab.fit(X_train,y_train)
ab_y_predict = ab.predict(X_test)

ab_ac_train = accuracy_score(ab.predict(X_train),y_train)
ab_ac_test = accuracy_score(ab_y_predict,y_test)
print("Accuracy for ab training: {}".format(round(ab_ac_train,2)))
print("Accuracy for ab test: {}".format(round(ab_ac_test,2)))
print("------------------------------------------------------")

# GradientBoost
gb = GradientBoostingClassifier(learning_rate=0.1,
                                n_estimators=100,
                                subsample=1.0,
                                min_samples_split=2,
                                max_depth=3,
                                random_state=42)
gb.fit(X_train,y_train)
gb_y_predict = gb.predict(X_test)

gb_ac_train = accuracy_score(gb.predict(X_train),y_train)
gb_ac_test = accuracy_score(gb_y_predict,y_test)
print("Accuracy for gb training: {}".format(round(gb_ac_train,2)))
print("Accuracy for gb test: {}".format(round(gb_ac_test,2)))
print("------------------------------------------------------")

# XGBoost
xb = XGBClassifier(objective="binary:logistic",
                   random_state = 42,
                   n_estimators=100)
xb.fit(X_train,y_train)
xb_y_predict = xb.predict(X_test)

xb_ac_train = accuracy_score(xb.predict(X_train),y_train)
xb_ac_test = accuracy_score(xb_y_predict,y_test)
print("Accuracy for xb training: {}".format(round(xb_ac_train,2)))
print("Accuracy for xb test: {}".format(round(xb_ac_test,2)))
print("------------------------------------------------------")

# Stacking
dtc = DecisionTreeClassifier(random_state=42)
rfc = RandomForestClassifier(random_state=42)
knn = KNeighborsClassifier()
xg = XGBClassifier()
gc = GradientBoostingClassifier(random_state=42)
svc = SVC(kernel= "rbf",random_state=42)
ad = AdaBoostClassifier(random_state=42)

clf = [("dtc",dtc),("rfc",rfc),("knn",knn),
       ("gc",gc),("svc",svc),("ad",ad)]

st = StackingClassifier(estimators=clf,
                        final_estimator=xg)
st.fit(X_train,y_train)
st_y_predict = st.predict(X_test)

st_ac_train = accuracy_score(st.predict(X_train),y_train)
st_ac_test = accuracy_score(st_y_predict,y_test)
print("Accuracy for st training: {}".format(round(st_ac_train,2)))
print("Accuracy for st test: {}".format(round(st_ac_test,2)))
print("------------------------------------------------------")

