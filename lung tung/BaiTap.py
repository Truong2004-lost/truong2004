#country_of_interest = "United States"
country_of_interest = input("Country: ")

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from geopy.geocoders import Nominatim
geolocator = Nominatim(user_agent="MyApp")

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_absolute_error


fpath = open("Data.csv","r")
df = pd.read_csv(fpath)
months = ['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec']

df = df[df["Country"]==country_of_interest]
print(df)

def get_location(city, attr, debug=False):
    def log(data):
        if debug:
            print("{}: {}".format(city, data))
    try:
        if attr == "latitude":
            data = geolocator.geocode(city).latitude
            log(data)
            return data
        if attr == "longitude":
            data = geolocator.geocode(city).longitude
            log(data)
            return data
    except Exception as e:
        print(e)
        return float('NaN')
    
df["Latitude"] = df["City"].map(lambda x: get_location(x, "latitude"))
df["Longitude"] = df["City"].map(lambda x: get_location(x, "longitude"))
new_df_list = []
for index, row in df.iterrows():
    for mon in months:
        temp = {"Country":row["Country"], "City":row["City"], "Latitude":row["Latitude"], "Longitude":row["Longitude"], "Month":mon, "Hours":row[mon]}
        new_df_list.append(temp)
new_df = pd.DataFrame(new_df_list)
print(new_df)

df = new_df
df[df["Longitude"] > 125]

df = df[df["City"] != "Richmond (VA)"]

for month in df["Month"].unique():
    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(10,5))
    data = df[df["Month"]==month]
    sns.scatterplot(data, y="Latitude", x="Longitude", hue="Hours", ax=axes[0]).set(title="{}: Hours of Sunlight in {}".format(month, country_of_interest))
    sns.histplot(data, x="Hours", ax=axes[1]).set(title="{}: Hours of Sunlight Dist".format(month))
    axes[0].set_xlim()
    axes[1].set_xlim(min(df["Hours"]), max(df["Hours"]))
    
    plt.show()
    
df = df.dropna()
y = df["Hours"]
X = df

X = X[["Latitude","Longitude","Month"]]
def convert_month(month):
    month_map = {"Jan":1.0,"Feb":2.0, "Mar":3.0,
                 "Apr":4.0,"May":5.0,"Jun":6.0,
                 "Jul":7.0,"Aug":8.0,"Sep":9.0,
                 "Oct":10.0,"Nov":11.0,"Dec":12.0}
    return month_map[month]
X["Month"] = X["Month"].map(lambda x: convert_month(x))

X['month_sin'] = np.sin(2 * np.pi * X['Month']/12.0)
X['month_cos'] = np.cos(2 * np.pi * X['Month']/12.0)

X = X[["Latitude","Longitude","month_sin","month_cos"]]
print(X)

X.describe()

clf = RandomForestRegressor()
scores = cross_val_score(clf, X, y, cv=5)
print(scores)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=13)

clf.fit(X_train, y_train)
scores=clf.score(X_test, y_test)
scores=str(round(scores, 3)*100)

y_pred = clf.predict(X_test)
MAE=mean_absolute_error(y_test, y_pred)
MAE=str(round(MAE,3))

print(scores)
print(MAE)

f=open("X.txt", "w")
f.write("Country: ")
f.write(country_of_interest)
f.write("\n")
f.write("Scores: ")
f.write(scores)
f.write(" %")
f.write("\n")
f.write("MAE: ")
f.write(MAE)
f.write(" hours")
f.close()
