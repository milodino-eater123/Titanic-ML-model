import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer

imputer = SimpleImputer(strategy="median")

train = pd.read_csv("titanic/train.csv")
x = train.drop("Survived",axis=1)
x = x.select_dtypes(include="number")
x = imputer.fit_transform(x)
y=train["Survived"]
x_train,x_test,y_train,y_test=train_test_split(
    x,y, test_size=0.2, random_state=13
)


model = LinearRegression()
model.fit(x_train,y_train)
predictions=model.predict(x_test)
print(predictions)
print(y_test)
