import pandas as pd
import matplotlib.pyplot as mpt
import numpy as np
from sklearn.tree import DecisionTreeClassifier

df = pd.read_csv('train.csv')
df_test = pd.read_csv('test.csv')

print(df.shape[0])
print(df.shape[1])
print(df_test.shape[0])
print(df_test.shape[1])

print(df.columns.difference(df_test.columns))

survived = 0
for a in df['Survived']:
    if a == 1:
        survived+=1

        

print(sum(pd.isna(df['Age'])))
print(sum(pd.isna(df['Embarked'])))
print(sum(pd.isna(df['Cabin'])))

print(survived)
print(len(df['Survived']) - survived)

print(df.groupby('Sex')['Survived'].mean())
print(df.groupby('Pclass')['Survived'].mean())

print(sum(pd.isna(df['Pclass'])))
print(sum(pd.isna(df['SibSp'])))
print(sum(pd.isna(df['Parch'])))

print(sum(pd.isna(df_test['Age'])))
print(sum(pd.isna(df_test['Embarked'])))
print(sum(pd.isna(df_test['Cabin'])))
print(sum(pd.isna(df_test['Pclass'])))
print(sum(pd.isna(df_test['SibSp'])))
print(sum(pd.isna(df_test['Parch'])))

print(sum(pd.isna(df['Cabin'])))

print(sum(pd.isna(df_test['Cabin'])))

med_age = df['Age'].median()

df['Age'] = df['Age'].fillna(med_age)
df_test['Age'] = df_test['Age'].fillna(med_age)


sex_map = {"female": 0, "male": 1}
df["Sex"] = df["Sex"].map(sex_map)
df_test["Sex"] = df_test["Sex"].map(sex_map)

features = ["Pclass", "Sex", "SibSp", "Parch", "Age"]

X = df[features]
y = df["Survived"]

X_test = df_test[features]

print(X)
print(y)
print(X_test)
model = DecisionTreeClassifier(max_depth = 5, random_state=42)
model.fit(X, y)

predictions = model.predict(X_test)
print(len(predictions))

submission = pd.DataFrame({
    "PassengerId": df_test["PassengerId"],
    "Survived": predictions
})

submission.to_csv("submission.csv", index = False)
