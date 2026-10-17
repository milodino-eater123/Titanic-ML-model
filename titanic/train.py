import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from helper import check_missing,clean_missing_results

train = pd.read_csv("titanic/train.csv")
train = clean_missing_results(train)

x = train.drop(["Survived","Name","Ticket", "PassengerId","Cabin"],axis=1)
y=train["Survived"]

x_train,x_test,y_train,y_test=train_test_split(
    x,y, test_size=0.2, random_state=13
)

numerical_columns = ["Age", "Fare","Parch","SibSp","Pclass"]
categorical_columns = ["Sex", "Embarked"]

num_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])
cat_pipeline = Pipeline([
    ("imputer", SimpleImputer(
        strategy="constant",
        fill_value="Unknown"
    )),
    ("encoder", OneHotEncoder(handle_unknown="ignore"))
])
preprocessor = ColumnTransformer([
    ("num", num_pipeline, numerical_columns),
    ("cat", cat_pipeline, categorical_columns)
])  

model = Pipeline([
    ("preprocessor", preprocessor),
    ("classifier", LogisticRegression(max_iter=1000))
])

model.fit(x_train,y_train)
print(model.score(x_test,y_test))



