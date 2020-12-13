# %% [markdown]
# The earlier plot of the Air Passengers data set shows that our data contains both a seasonal and a trend component. The increase in amplitude of the seasonality indicates, I think, a multiplicative time series.

# %%
ap = pd.read_csv("../pycon2017/data/AirPassengers.csv")


# %%
ap.dropna(inplace=True)


# %%
# convert the ap index to a datetime
ap.set_index("Month", inplace=True)
ap.set_index(pd.to_datetime(ap.index), inplace=True)


# %%
ap.sample(10)


# %%
decomposition = seasonal_decompose(ap, model="multiplicative", freq=12)

# if you don't want to convert the index to a datetime for some reason, you can set the frequency explicitly
#decomposition = seasonal_decompose(ap, model="multiplicative", freq=12)


# %%
trend = decomposition.trend
seasonal = decomposition.seasonal
residual = decomposition.resid
decomposition.plot()

# %% [markdown]
# Using a log transformation to remove the trend (note to self, you need to learn how to test/validate that your removal of trend and seasonality actually works beyond a visual inspection).

# %%
removed_trend = ap["#Passengers"].apply(lambda x: np.log(x))
removed_trend.head()


# %%
# compare visually
ax1 = plt.subplot(121)
removed_trend.plot(title="Log-Transformed #Passengers", figsize=(10, 4), ax=ax1)
ax2 = plt.subplot(122)
ap.plot(title="Original #Passengers", ax=ax2)

# %% [markdown]
# We can see that the variance has been reduced. How do we know if this is good enough? Are there some test that we can apply?

# %%
decomposition_after_log_transform = seasonal_decompose(removed_trend)
decomposition_after_log_transform.plot()

# %% [markdown]
# Next, remove seasonality using differencing the log transformed values.

# %%
removed_season = removed_trend - removed_trend.shift()


# %%
ax1 = plt.subplot(121)
removed_season.plot(title="Log-Transformed And Differenced #Passengers", figsize=(10, 4), ax=ax1)
ax2 = plt.subplot(122)
ap.plot(title="Original #Passengers", ax=ax2)


# %%
removed_season.sample(10)


# %%
ap["residual"] = removed_season


# %%
ap.sample(10)


# %%
# Adding an entity id for tsfresh. We have just a single entity.
ap["id"] = "A"


# %%
# No NaN for tsfresh.
ap.dropna(inplace=True)


# %%
ap.reset_index(inplace=True)


# %%
ap.head()


# %%
ap["Month"].sample(5)


# %%
ap_sample = ap[["Month", "residual", "id"]].head(70)


# %%
ap_sample.sample(10)


# %%
ap_sample.dtypes


# %%

from tsfresh import extract_features
from tsfresh.utilities.dataframe_functions import impute
extracted_features = extract_features(ap_sample, column_id="id", column_sort="Month", column_value="residual")


# %%
extracted_features