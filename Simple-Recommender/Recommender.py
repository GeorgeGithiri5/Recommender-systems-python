from scipy import spatial

a = [1,2]
b = [2,4]
c = [2.5,4]
d = [4.5,5]

# euclidean distance
print(spatial.distance.euclidean(c,a))
print(spatial.distance.euclidean(c,b))
print(spatial.distance.euclidean(c,d))

# Cosine Distance
# Perfect for checking similarity since cosine ranges from -1 to 1. 
# cosine goes towards negative as the angle increase.

print(spatial.distance.cosine(c,a))
print(spatial.distance.cosine(c,b))
print(spatial.distance.cosine(c,d))
print(spatial.distance.cosine(a,b))

import pandas as pd
from surprise import Dataset
from surprise import Reader

rating_dict = {
    "item": [1,2,1,2,1,2,1,2,1],
    "user": ['A','A','B','B','C','C','D','D','E'],
    "rating": [1,2,2,4,2.5,4,4.5,5,3]
}

df = pd.DataFrame(rating_dict)
reader = Reader(rating_scale=(1,5))

# Load the dataframe
data = Dataset.load_from_df(df[["user","item","rating"]],reader)
# builtin Movielens-100k data
movielens = Dataset.load_builtin('ml-100k')

# Configure KnnwithMeans

from surprise import KNNWithMeans
sim_options = {
    "name":"cosine",
    "user_based": False
}

algo = KNNWithMeans(sim_options=sim_options)

# Predict the Rating of Movie E
trainSet = data.build_full_trainset()
algo.fit(trainSet)

prediction = algo.predict('E',2)
print(prediction.est)

# Checking Which similarity metric works best for our memory based approach
from surprise.model_selection import GridSearchCV

dataa = Dataset.load_builtin("ml-100k")
sim_options = {
    "name": ['msd','cosine'],
    "min_support": [3,4,5],
    "user_based": [False,True]
}

param_grid = {"sim_options": sim_options}
gs = GridSearchCV(KNNWithMeans,param_grid,measures=['rmse','mae'],cv=3)
gs.fit(dataa)

print(gs.best_score["rmse"])
print(gs.best_params["rmse"])