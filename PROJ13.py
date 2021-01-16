#%%
import pandas as pd
import numpy as np 
import re

class Movie():
    def __init__(self, movie_id, df, year, title):
        self.id = movie_id
        self.reviews = df
        self.year = year
        self.title = title
    
    def print_movie(self):
        print(f"movie id: {self.id}\nyear: {self.year}\ntitle: {self.title}")

    def print_reviews(self):
        display(self.reviews)

lines = ""
for i in range(1,5): #range(1,5) dla wszystkich plików
    with open(f"data/combined_data_{i}.txt", "r") as f:
        lines += f.read()

titles = pd.read_csv("data/movie_titles.csv", index_col="id")
movie_lines = re.split("\d+:\n", lines)

# %%
from io import StringIO

header = "user_id,rating,date\n"
movies = []
for i in range(len(movie_lines)-1):
    movies.append(Movie(i+1, pd.read_csv(StringIO(header + movie_lines[i+1])), titles.iloc[i]['year'], titles.iloc[i]['title']))
# %%
user_ids = []
for movie in movies:
    user_ids.extend(movie.reviews["user_id"])

user_ids = np.unique(user_ids)
#%%
from operator import itemgetter
movies_count = len(movies)
users_count = len(user_ids)

mapping = dict(zip(user_ids, list(range(users_count))))
table = np.zeros((users_count, movies_count), dtype=np.int64)

for movie in movies:
    reviews = movie.reviews["user_id"]
    table[list(itemgetter(*reviews)(mapping)), [movie.id-1]*len(reviews)] = np.array(movie.reviews["rating"])
    if movie.id % 100 == 0:
        print(movie.id)

# %%
df_record = pd.read_csv("moviesBands.csv", sep=';')
#%%
def custom_distance(v1, v2):
    seen_v1 = np.nonzero(v1)[0]
    seen_v2 = np.nonzero(v2)[0]
    common = seen_v1[np.isin(seen_v1, seen_v2)]
    return movies_count - len(common)/2 + np.sum(np.abs(v1[common] - v2[common]) - 2) #liczba_filmów - liczba_wspólnie_widzianych_filmów/2 + [(abs(różnica ratingów) - 2) dla każdego filmu] 

#query_id = 1488844
#record = table[mapping[query_id]] 
record = np.nan_to_num(np.array(df_record["Rating"]), 0) 
k = 5 # number of movie nearest soulmates <3

k_nearest = list(table[0:k])
k_nearest.sort(key=lambda x: custom_distance(x, record))

for i in range(k, len(table)):
    if custom_distance(table[i], record) < custom_distance(k_nearest[k-1], record):
        k_nearest[k-1] = table[i]
        k_nearest.sort(key=lambda x: custom_distance(x, record))
    if i % 10000 == 0:
        print(int(i*100/users_count), "%")

# %%
k_nearest = np.array(k_nearest)
mean_reviews = np.zeros((movies_count,2))

def calc_score(rev):
    nonzero = np.count_nonzero(rev)
    if nonzero == 0:
        return np.array([0, 0.5])
    positive = np.count_nonzero(rev == 5) + np.count_nonzero(rev == 4) + 1
    negative = np.count_nonzero(rev == 1) + np.count_nonzero(rev == 2) + 1
    return np.array([np.sum(rev)/nonzero, positive/(positive + negative)]) #(mean rating, laplace's rule of succession)

for i in range(movies_count):
    mean_reviews[i] = calc_score(k_nearest[:,i])

# %%
p = 50 # movies to recommend
p_best = list(zip(list(range(p)), np.zeros((p,2))))
seen_movies = np.nonzero(record)[0] 
for i in range(movies_count):
    if i in seen_movies:
        continue
    if mean_reviews[i][1] > p_best[p-1][1][1] or \
            (mean_reviews[i][1] == p_best[p-1][1][1] and mean_reviews[i][0] > p_best[p-1][1][0]):
        p_best[p-1] = (i,mean_reviews[i])
        p_best.sort(key=lambda x: (x[1][1],x[1][0]), reverse=True)

# %%
def print_query_guy(record):
    for movie_id in seen_movies:
        print(movies[movie_id].title, end="---|---")
    print()

def print_results(your=False, common=False):
    if your:
        print_query_guy(record)
    if common:
        for i in range(k):
            print(f"--------------------- neighbor {i} ---------------------")
            seen_movies2 = np.nonzero(k_nearest[i])[0]
            for movie_id in seen_movies2:
                for movie_id2 in seen_movies:
                    if movie_id == movie_id2:
                        print(f"{movies[movie_id].title} -----", "(him)", k_nearest[i][movie_id], ":", int(record[movie_id2]), "(you)")
    print("\n --------- RECOMMENDATIONS --------- \n")
    for best in p_best:
        if best[0] in seen_movies:
            continue
        print(movies[best[0]].title,"-----", "{:.2f}".format(best[1][0]), ":", f"{int(np.round(best[1][1]*100))}%")

print_results(True, True)
# %%
