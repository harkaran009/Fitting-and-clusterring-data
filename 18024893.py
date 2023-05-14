#!/usr/bin/env python
# coding: utf-8

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import cluster_tools as ct
import sklearn.cluster as cluster
import sklearn.metrics as skmet
import errors as err
import scipy.optimize as opt
%matplotlib inline


df_co2 = pd.read_csv('co2.csv')
df_gdp = pd.read_csv('gdp.csv', skiprows=3)



print(df_co2.describe())
print(df_gdp.describe())



df_co2 = df_co2[df_co2["2019"].notna()]

print(df_co2.describe())

# alternative way of targetting one or more columns

df_gdp = df_gdp[df_gdp["2019"].notna()]

print(df_gdp.describe)

df_co2_2019 = df_co2[["Country Name", "Country Code", "2019"]].copy()
df_gdp_2019 = df_gdp[["Country Name", "Country Code", "2019"]].copy()



print(df_co2_2019.describe())
print(df_gdp_2019.describe())



df_2019 = pd.merge(df_co2_2019, df_gdp_2019, on="Country Name", how="outer")
print(df_2019.describe())
df_2019.to_excel("co2_gdp.xlsx")



print(df_2019.describe())
df_2019 = df_2019.dropna() # entries with one datum or less are useless.
print()
print(df_2019.describe())
# rename columns
df_2019 = df_2019.rename(columns={"2019_x":"CO2 emissions", "2019_y":"GDP"})



pd.plotting.scatter_matrix(df_2019, figsize=(12, 12), s=5, alpha=0.8)




print(df_2019.corr())



df_cluster = df_2019[["CO2 emissions", "GDP"]].copy()
# normalise
df_cluster, df_min, df_max = ct.scaler(df_cluster)



print("n score")

# loop over number of clusters

for ncluster in range(2, 10):
    
    # set up the clusterer with the number of expected clusters
    kmeans = cluster.KMeans(n_clusters=ncluster)

    # Fit the data, results are stored in the kmeans object
    kmeans.fit(df_cluster) # fit done on x,y pairs

    labels = kmeans.labels_

    # extract the estimated cluster centres
    cen = kmeans.cluster_centers_

    # calculate the silhoutte score
    print(ncluster, skmet.silhouette_score(df_cluster, labels))




ncluster = 5

# set up the clusterer with the number of expected clusters
kmeans = cluster.KMeans(n_clusters=ncluster)

# Fit the data, results are stored in the kmeans object
kmeans.fit(df_cluster) # fit done on x,y pairs
labels = kmeans.labels_

# extract the estimated cluster centres
cen = kmeans.cluster_centers_
xcen = cen[:, 0]
ycen = cen[:, 1]

# cluster by cluster
plt.figure(figsize=(8.0, 8.0))

cm = plt.cm.get_cmap('tab10')

plt.scatter(df_cluster["CO2 emissions"], df_cluster["GDP"], 10, labels,marker="o", cmap=cm)

plt.scatter(xcen, ycen, 45, "k", marker="d")

plt.xlabel("CO2 emissions")

plt.ylabel("GDP")

plt.show()



# move the cluster centres to the original scale
cen = ct.backscale(cen, df_min, df_max)
xcen = cen[:, 0]
ycen = cen[:, 1]
# cluster by cluster
plt.figure(figsize=(8.0, 8.0))
cm = plt.cm.get_cmap('tab10')
plt.scatter(df_2019["CO2 emissions"], df_2019["GDP"], 10, labels, marker="o",cmap=cm)
plt.scatter(xcen, ycen, 45, "k", marker="d")

plt.xlabel("CO2 emissions")
plt.ylabel("GDP")
plt.show()


# # world GDP historic DATA



df_world_gdp = pd.read_csv('world_gdp.csv')
df_world_gdp = df_world_gdp.dropna()

df_world_gdp['Year'] = df_world_gdp['Year'].astype(int)
df_world_gdp.dtypes






df_world_gdp.plot("Year", "World")
plt.show()





def exponential(t, n0, g):
    """Calculates exponential function with scale factor n0 and growth rate g."""
    t = t - 1990
    f = n0 * np.exp(g*t)
    return f




print(type(df_world_gdp["Year"].iloc[1]))

df_world_gdp["Year"] = pd.to_numeric(df_world_gdp["Year"])

print(type(df_world_gdp["Year"].iloc[1]))

param, covar = opt.curve_fit(exponential, df_world_gdp["Year"], df_world_gdp["World"], p0=(1.2e12, 0.03))

print("GDP 1960", param[0]/1e9)

print("growth rate", param[1])




plt.figure()
plt.plot(df_world_gdp["Year"], exponential(df_world_gdp["Year"], 1.2e12, 0.03), label = "trial fit")

plt.plot(df_world_gdp["Year"], df_world_gdp["World"])

plt.xlabel("Year")

plt.legend()

plt.show()





df_world_gdp["fit"] = exponential(df_world_gdp["Year"], *param)
df_world_gdp.plot("Year", ["World", "fit"])
plt.show()





def logistic(t, n0, g, t0):
    """Calculates the logistic function with scale factor n0 and growth rate g"""
    f = n0 / (1 + np.exp(-g*(t - t0)))
    
    return f






param, covar = opt.curve_fit(logistic, df_world_gdp["Year"], df_world_gdp["World"],p0=(1.2e12, 0.06, 1960.0))





sigma = np.sqrt(np.diag(covar))

df_world_gdp["fit"] = logistic(df_world_gdp["Year"], *param)

df_world_gdp.plot("Year", ["World", "fit"])

plt.show()

print("turning point", param[2], "+/-", sigma[2])

print("GDP at turning point", param[0]/1e9, "+/-", sigma[0]/1e9)

print("growth rate", param[1], "+/-", sigma[1])

df_world_gdp['fit']





df_world_gdp["trial"] = logistic(df_world_gdp["Year"], 3e12, 0.10, 1960)
df_world_gdp.plot("Year", ["World", "trial"])
plt.show()





year = np.arange(1960, 2031)
forecast = logistic(year, *param)



plt.figure()
plt.plot(df_world_gdp["Year"], df_world_gdp["World"], label="World GDP")
plt.plot(year, forecast, label="forecast")
plt.xlabel("year")
plt.ylabel("GDP")
plt.legend()
plt.show()



low, up = err.err_ranges(year, logistic, param, sigma)



plt.figure()
plt.plot(df_world_gdp["Year"], df_world_gdp["World"], label="World GDP")
plt.plot(year, forecast, label="forecast")
plt.fill_between(year, low, up, color="yellow", alpha=0.7)
plt.xlabel("year")
plt.ylabel("GDP")
plt.legend()
plt.show()



print(logistic(2030, *param)/1e9)

print(err.err_ranges(2030, logistic, param, sigma))

# assuming symmetrie estimate sigma
gdp2030 = logistic(2030, *param)/1e9

low, up = err.err_ranges(2030, logistic, param, sigma)

sig = np.abs(up-low)/(2.0 * 1e9)

print()

print("GDP 2030", gdp2030, "+/-", sig)



def poly(x, a, b, c, d, e):
    """ Calulates polynominal"""
    x = x - 1990
    f = a + b*x + c*x**2 + d*x**3 + e*x**4
    return f



param, covar = opt.curve_fit(poly, df_world_gdp["Year"], df_world_gdp["World"])
sigma = np.sqrt(np.diag(covar))
print(sigma)
year = np.arange(1960, 2031)

forecast = poly(year, *param)

low, up = err.err_ranges(year, poly, param, sigma)

df_world_gdp["fit"] = poly(df_world_gdp["Year"], *param)

plt.figure()
plt.plot(df_world_gdp["Year"], df_world_gdp["World"], label="GDP")

plt.plot(year, forecast, label="forecast")

plt.fill_between(year, low, up, color="yellow", alpha=0.7)
plt.xlabel("year")
plt.ylabel("GDP")
plt.legend()
plt.show()




