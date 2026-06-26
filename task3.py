import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error

# Dataset
data = {
    "Hours": [1,2,3,4,5,6,7,8,9,10],
    "Score": [15,25,40,50,60,65,75,82,88,95]
}

df = pd.DataFrame(data)

X = df[['Hours']]
y = df['Score']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = LinearRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

mae = mean_absolute_error(y_test, y_pred)

print("Mean Absolute Error:", round(mae,2))

results = pd.DataFrame({
    "Actual": y_test.values,
    "Predicted": y_pred
})

print(results)

new_hours = pd.DataFrame({"Hours": [4.5]})
predicted_score = model.predict(new_hours)

print("Predicted Score for 4.5 Hours:", round(predicted_score[0],2))
