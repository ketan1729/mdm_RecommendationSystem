import numpy as np
import pandas as pd
import math
from sklearn.model_selection import train_test_split
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error

K = 5


def get_train_and_test_data():
    df = pd.read_csv(r"ratings.csv")
    # df = pd.read_csv(r"ratings_2.csv")
    train, test = train_test_split(df, test_size=0.2, stratify=df['userId'])
    return train, test


def create_utility_mat():
    train_y = train_data['rating']
    train_x = pd.pivot_table(train_data, values='rating', index=['userId'], columns='movieId').reset_index()
    train_x['mean'] = train_x.iloc[:, 1:len(train_x.columns)].mean(axis=1)
    train_x = train_x.iloc[:, 1:len(train_x.columns)].apply(lambda row: row - row['mean'], axis=1)
    train_x = train_x.iloc[:, :-1].fillna(0)
    return train_x, train_y


def get_similarity_matrix():
    mat = cosine_similarity(x_train, x_train)
    for i in range(len(mat)):
        mat[i][i] = 0
    return mat


def predict_rating(user_id, movie_id):
    weighted_sum = 0
    sim_sum = 0
    count = 0

    target_users = train_data[train_data['movieId'] == movie_id]['userId'].unique()
    target_users = [x - 1 for x in target_users]
    target_users = sorted(target_users, key=lambda x: sim_mat[x][user_id])

    if len(target_users) == 0:
        # NA-TU     Only the test user has watched the movie - Need to fix
        return 2.75
    for user in target_users[::-1]:
        if count < K:
            sim_sum += sim_mat[user_id][user]
            weighted_sum += sim_mat[user_id][user] * train_data[train_data['movieId'] == movie_id][train_data['userId'] == user + 1][
                                'rating'].item()
            count += 1
    res = weighted_sum / sim_sum
    if math.isnan(res):
        # NAN    When the weighted sum is 0 and sim_sum is also 0 - Need to fix
        return 2.75
    return res


def get_ratings():
    pred = []
    c = 1
    for index, row in test_data.iterrows():
        print("For iter :", c)
        c += 1
        user_id = int(row['userId']) - 1
        movie_id = row['movieId']
        res = predict_rating(user_id, movie_id)
        pred.append(res)
    return pred


def compare():
    y_test = test_data['rating']
    for x, y in zip(y_test, y_pred):
        print("Pred : ", y, " Actual : ", x)


def calculate_rmse():
    y_test = test_data['rating']
    '''
    unwanted = []
    for i in range(len(y_pred)):
        if (not str(y_pred[i]).isdigit()) or math.isnan(y_pred[i]):
            unwanted.append(i)
    y_test.drop(y_test.index[unwanted])

    for ele in sorted(unwanted, reverse=True):
        del y_pred[ele]
    '''

    rms = mean_squared_error(y_test, y_pred, squared=False)
    print("RMSE: ", rms)

    mae = mean_absolute_error(y_test, y_pred)
    print("MAE: ", mae)


def get_predicted_ratings_mat():
    res_mat = []


if __name__ == '__main__':
    train_data, test_data = get_train_and_test_data()
    x_train, y_train = create_utility_mat()
    sim_mat = get_similarity_matrix()
    y_pred = get_ratings()
    compare()
    calculate_rmse()

    # print(get_top_k_similar_users_rated(train_data, sim_mat, 5, 1, 4))
