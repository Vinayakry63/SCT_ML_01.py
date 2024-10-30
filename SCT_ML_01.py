import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
data=pd.read_csv("train.csv")
data.info()

input =['Id','YearBuilt','YrSold','GrLivArea', 'BedroomAbvGr', 'FullBath']
output = 'SalePrice'
x = data[input]
y = data[output]
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
model = LinearRegression()
model.fit(data[input],data[output])
y_val_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_val_pred)
rmse = mse ** 0.5
print(f'Validation RMSE: {rmse}')

test_data = pd.read_csv("test.csv")
test_data.info()

X_test = test_data[['Id','YearBuilt', 'YrSold', 'GrLivArea', 'BedroomAbvGr', 'FullBath']]
test_predictions = model.predict(X_test)
submissiondata = pd.DataFrame({
    'Id': test_data['Id'],
    'SalePrice': test_predictions
})
submissiondata.to_csv('sample_submission.csv', index=False)
print('Submission file created: sample_submission.csv')

plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_val_pred, color='blue', label='Predicted vs Actual')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2, label='Perfect Prediction (y=x)')
plt.xlabel('Actual Sale Price')
plt.ylabel('Predicted Sale Price')
plt.title('Actual vs Predicted Sale Price')
plt.legend()
plt.grid(True)
plt.show()