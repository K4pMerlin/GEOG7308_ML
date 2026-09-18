import numpy as np
import matplotlib.pylab as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

#load data
data_file=r'ice_cream_sales/ice.csv'
data=np.loadtxt(data_file, delimiter=',', skiprows=1)
temperature=data[:, 0:1]
sales=data[:, 1:2]

#display temperature and sales data in a scatter plot
plt.scatter(temperature, sales)         #scatter plot
plt.xlabel('Temperature')               #x-axis label
plt.ylabel('Sales')                     #y-axis label
plt.title('Ice Cream Sales vs Temperature') #title
plt.show()                           #display the plot

#define lienar regression model
scales_model=LinearRegression()

#train the model
scales_model.fit(temperature, sales)

#print the model parameters
a=scales_model.coef_[0][0]
b=scales_model.intercept_[0]
print('Coefficient: {0}, Intercept: {1}'.format(a, b))
#predict sales using the model
sales_pred=scales_model.predict(temperature)
#display the regression line
plt.plot(temperature, sales_pred, color='red') #regression line
plt.scatter(temperature, sales) #scatter plot
plt.xlabel('Temperature')    #x-axis label
plt.ylabel('Sales')         #y-axis label
plt.title('y={0}x + {1}'.format(a, b))    #title


#evaluate the model
mse=mean_squared_error(sales, sales_pred)
r2=r2_score(sales, sales_pred)
mae=mean_absolute_error(sales, sales_pred)
print('MSE: {0}, R2: {1}, MAE: {2}'.format(mse, r2, mae))

# add metrics on plot
plt.text(12, 600, 'MSE: {0:.2f}'.format(mse),fontsize=12)
plt.text(12, 570, 'MAE: {0:.2f}'.format(mae),fontsize=12)
plt.text(12, 540, '$R^2$: {0:.2f}'.format(r2),fontsize=12)


plt.show()
