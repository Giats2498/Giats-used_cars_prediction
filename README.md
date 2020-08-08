# Used Cars Price Prediction by 15 different models
[Car.gr](https://car.gr/) is the Greek largest collection of used vehicles for sale, yet it's very difficult to collect all of them in the same place. I built a scraper and expanded upon it later to create this dataset which includes every used vehicle entry on Car.gr.

## Project Overview 
* Created a tool that estimates used car price ```(R² ~ 0.73, MAE ~ 2988, RMSE ~ 6802)``` to help someone who wants to buy or to sell a car to best price.
* Scraped over **34K** used cars using python.
* Analysed and visualized all varriables using popular python's packages. 
* Built the **15** most popular models, the most complex models from them are tuned (optimized)
* Comparison of the optimal for each type models.
* Built a client facing API using flask 

## Code and Resources Used 
**Python Version:** 3.6  
**Packages:** pandas, numpy, prettytable, sklearn, matplotlib, seaborn, selenium, flask, json, pickle  
**Requirements:**  ```pip install -r requirements.txt```  
**Flask Productionization:** https://towardsdatascience.com/productionize-a-machine-learning-model-with-flask-and-heroku-8201260503d2

## Web Scraping
Used scraper to scrape over 34000 cars from Car.gr.

|Make_model                                                        |Classified_number|Price                |Category         |Registration|Mileage   |Fuel_type|Cubic_capacity|Power  |Transmission|Color           |Number_plate|Previous_owners|Drive_type|Airbags|Doors|Seats|Zip_code          |
|------------------------------------------------------------------|-----------------|---------------------|-----------------|------------|----------|---------|--------------|-------|------------|----------------|------------|---------------|----------|-------|-----|-----|------------------|
|Suzuki Grand Vitara '99                                           |28245137         |2.500 €              |4X4/Jeep/SUV     |Dec-99      |202,000 km|Gas/LPG  |2,000 cc      |120 bhp|Manual      |White           |-1          |-1             |-1        |-1     |5    |5    |ΣΙΝΔΟΣ 57400      |
|Chevrolet Spark ΓΝΗΣΙΑ ΧΛΜ-BOOK SERVICE '11                       |25809505         |4.950 €              |Compact/Hatchback|Aug-11      |91,221 km |Gasoline |1,200 cc      |82 bhp |Manual      |Other (Metallic)|even        |-1             |FWD       |10     |5    |5    |ΚΟΖΑΝΗ 50100      |
|Subaru Forester DIESEL-4X4-ΗΛΙΟΡΟΦΗ '09                           |25546984         |7.500 €              |4X4/Jeep/SUV     |Apr-09      |310,000 km|Petroleum|2,000 cc      |150 bhp|Manual      |Grey (Metallic) |odd         |-1             |4x4       |8      |5    |5    |ΚΟΖΑΝΗ 50100      |
|Renault Clio ΕΛΛΗΝΙΚΟ '19                                         |25916392         |10.000 €             |Compact/Hatchback|May-19      |11,300 km |Gasoline |900 cc        |76 bhp |Manual      |Grey (Metallic) |even        |1              |FWD       |4      |5    |5    |ΚΟΖΑΝΗ 50100      |
|Fiat Doblo '05                                                    |20318111         |3.500 € (Debatable)  |Crew Cab/Pickup  |Jan-05      |147,550 km|Petroleum|1,900 cc      |115 bhp|Manual      |White (Metallic)|-1          |1              |FWD       |4      |5    |2    |ΞΑΝΘΗ 67100       |

## EDA Highlights

+ This analysis gives the distribution of prices of vehicles based on vehicles types.
+ Output before the cleaning the data is shown below in order to highlight the importance of cleaning this dataset.
+ **Histogram** and **KDE** before performing data cleaning.
+ It is clearly visible that the dataset has **many outliers** and **inconsistent data** as year of registration **cannot be more than 2016 and less than 1970**.

![alt text](https://github.com/Giats2498/Giats-used_cars_prediction/blob/master/Images/vehicle-distribution.png "Logo Title Text 1")

> Boxplot of prices of vehicles based on the type of vehicles after cleaning the dataset. Based on the vehicle type how the prices vary is depictable from the boxplot. low, 25th, 50th(Median), 75th percentile, high can be estimated from this boxplot.

![alt text](https://github.com/Giats2498/Giats-used_cars_prediction/blob/master/Images/price-vehicleType-boxplot.png "Logo Title Text 1")

+ This analysis gives the number of cars which are available for sale in the entire dataset based on a particular brand. 

![alt text](https://github.com/Giats2498/Giats-used_cars_prediction/blob/master/Images/brand-vehicleCount.png "Logo Title Text 1")

> Barplot of average price of the vehicles for sale based of the type of the vehicle as well as based on the gearbox of the vehicle.

![alt text](https://github.com/Giats2498/Giats-used_cars_prediction/blob/master/Images/vehicletype-gearbox-price.png "Logo Title Text 1")

+ This analysis gives you the average price of the brand of vehicles and their types which are likely to be found in the dataset.

![alt text](https://github.com/Giats2498/Giats-used_cars_prediction/blob/master/Images/heatmap-price-brand-vehicleType.png "Logo Title Text 1")

+ This analysis gives you where are the most cars in Greek map.

![alt text](https://github.com/Giats2498/Giats-used_cars_prediction/blob/master/Images/map.PNG "Logo Title Text 1")

+ This analysis gives you the correlations of variables

![alt text](https://github.com/Giats2498/Giats-used_cars_prediction/blob/master/Images/correlations.PNG "Logo Title Text 1")

### A lot of more plots in [data_eda.ipynb](https://github.com/Giats2498/Giats-used_cars_prediction/blob/master/data_eda.ipynb)

## Model Building - Model performance

First, I cleaned the data and then I transformed the categorical variables into dummy variables. I also split the data into train and tests sets with a test size of 20%.   

I tried 15 different models and evaluated them using Mean Absolute Error, R² Squared and Root Mean Square Error. Finally I removed outliers of some variables.

+ R²
![alt text](https://github.com/Giats2498/Giats-used_cars_prediction/blob/master/Model_Images/R2.png "Logo Title Text 1")

+ MAE
![alt text](https://github.com/Giats2498/Giats-used_cars_prediction/blob/master/Model_Images/MAE.png "Logo Title Text 1")

+ RMSE
![alt text](https://github.com/Giats2498/Giats-used_cars_prediction/blob/master/Model_Images/RMSE.png "Logo Title Text 1")


|Model                     |r2_train               |r2_test                |mae_train            |mae_test              |rmse_train            |rmse_test             |
|--------------------------|-----------------------|-----------------------|---------------------|----------------------|----------------------|----------------------|
|Linear Regression         |0.7924784935542084     |0.6941084673222401     |2608.2296286624214   |2983.8250594808546    |5918.184623784651     |7180.436830730218     |
|Support Vector Machines   |-0.01708465371852852   |-0.023605843760369227  |5296.569412748024    |5527.900007065662     |13101.936925594746    |13135.104020216624    |
|Linear SVR                |0.19511306179515797    |0.20426129554374406    |6491.8418041145205   |6630.800795573904     |11655.325145727957    |11581.16847214583     |
|MLPRegressor              |0.6011686983410387     |0.6293190033678105     |3771.22028890373     |3924.57074119332      |8204.489651251866     |7904.370690605631     |
|Stochastic Gradient Decent|-4.5554461041683124e+29|-4.4592876992614056e+29|7.726503492707798e+18|7.668078140775438e+18 |8.768441523239984e+18 |8.669616450035291e+18 |
|Decision Tree Regressor   |0.9999965309649592     |0.5761727948705108     |0.6173247359762246   |2387.7148997134673    |24.196987321920865    |8452.040185700753     |
|Random Forest             |0.9771173493788213     |0.8733856477723306     |619.0730602532076    |1694.3923774236544    |1965.2152177275207    |4619.645039106734     |
|XGB                       |0.9412647378052772     |0.8546519304931895     |1601.7213469093685   |1942.9072586054713    |3148.5196601477933    |4949.619022112295     |
|LGBM                      |0.9643313847459783     |0.8756600671254887     |1169.611042757379    |1691.9634720655392    |2453.5791826371624    |4577.964835312427     |
|GradientBoostingRegressor |0.9202570083799428     |0.8452896145109368     |1802.2113089475442   |2056.839702305344     |3668.624109705781     |5106.541577015737     |
|RidgeRegressor            |0.7624913276813889     |0.72551927636735       |2743.9413310174095   |2987.6517517165125    |6331.355355865922     |6801.787554074726     |
|BaggingRegressor          |0.9687146795798722     |0.8585257786939564     |709.2667956270234    |1806.3653154515546    |2297.879534710752     |4883.2144179509605    |
|ExtraTreesRegressor       |0.99999652512859       |0.8701649214779824     |0.6438244971607489   |1632.9454520853235    |24.217333498867326    |4678.031701831685     |
|AdaBoostRegressor         |-4.208607947362048     |-4.204812194315979     |27970.004353470693   |27866.46911790678     |29649.52828252368     |29618.950766855796    |
|VotingRegressor           |-1.308303999632478e+28 |-1.2809494707899487e+28|1.311701561842855e+18|1.3019184249803901e+18|1.4859742650767357e+18|1.4693766610744218e+18|

### I choosed RidgeRegressor model for better perfomance

## Productionization 
In this step, I built a flask API endpoint that was hosted on a local webserver by following along with the TDS tutorial in the reference section above. The API endpoint takes in a request with a list of values from a car listing and returns an estimated price. 
