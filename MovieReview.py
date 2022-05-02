import pandas as pd
import math
import csv
from os import listdir
from os.path import isfile, join
from sklearn.model_selection import train_test_split
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error

K = 5


def get_train_and_test_data():
    df = pd.read_csv(r"ratings.csv")
    train, test = train_test_split(df, test_size=0.2, stratify=df['userId'])
    train.to_csv('trainData.csv', encoding='utf-8', index=False)
    test.to_csv('testData.csv', encoding='utf-8', index=False)
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
    target_users = sorted(target_users, key=lambda x: sim_mat[x][user_id - 1])

    if len(target_users) == 0:
        # NA-TU     Only the test user has watched the movie - Need to fix
        return 2.75
    for user in target_users[::-1]:
        if count < K:
            sim_sum += sim_mat[user_id - 1][user]
            weighted_sum += sim_mat[user_id - 1][user] * \
                            train_data[train_data['movieId'] == movie_id][train_data['userId'] == user + 1][
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
        user_id = int(row['userId'])
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


def generate_predicted_ratings_files(train_data, test_data):
    movies_train = train_data['movieId'].unique().tolist()
    movies_test = test_data['movieId'].unique().tolist()
    movies = []
    movies.extend(movies_train)
    movies.extend(movies_test)
    movies = set(movies)

    users = train_data['userId'].unique()
    user_bins = []
    bin_size = 10
    print("Number of bins = ", len(users) / bin_size)
    curr_bin = []
    for user in users:
        if len(curr_bin) == bin_size:
            user_bins.append(curr_bin)
            curr_bin = []
        curr_bin.append(user)

    lo = 0
    hi = len(user_bins)
    for user_bin_idx in range(hi):
        user_bin = user_bins[user_bin_idx]
        f_name = "PredictedRatings\\Bin" + str(user_bin_idx+1)+".csv"
        outcsv = open(f_name, 'w', newline='')
        writer = csv.writer(outcsv)
        writer.writerow(["user", "R1", "R2", "R3", "R4", "R5", "R6", "R7", "R8", "R9", "R10"])

        for user in user_bin:
            curr_rec = []
            res_row = [user]
            movies_to_predict = [x for x in movies if
                                 x not in train_data[train_data['userId'] == user]['movieId'].unique()]
            for movie in movies_to_predict:
                pred_mov_rating = predict_rating(user, movie)
                curr_rec.append((movie, pred_mov_rating))
            curr_rec_sorted_list = sorted(curr_rec, key=lambda x: x[1])
            res_row.extend([i[0] for i in curr_rec_sorted_list[:10]])
            writer.writerow(res_row)
        outcsv.close()


def get_recom_metrics():
    test_data_for_metrics = pd.read_csv("testData.csv")
    dir_path = "PredictedRatings"
    binfiles = [f for f in listdir(dir_path) if isfile(join(dir_path, f))]

    precision = 0.0
    recall = 0.0
    no_of_users = 0
    # df_pred_ratings = pd.DataFrame(columns=["user", "R1", "R2", "R3", "R4", "R5", "R6", "R7", "R8", "R9", "R10"])
    cols = ["R1", "R2", "R3", "R4", "R5", "R6", "R7", "R8", "R9", "R10"]
    for f in binfiles:
        curr_df = pd.read_csv("PredictedRatings//"+f)
        curr_df = curr_df.reset_index()
        for index, row in curr_df.iterrows():
            no_of_users += 1
            user_precision = 0.0
            user_recall = 0.0
            for c in cols:
                if row[c] in test_data_for_metrics[test_data_for_metrics['userId'] == row['user']]['movieId'].unique():
                    user_precision += 1
                    user_recall += 1
            precision += user_precision / 10
            recall += user_recall / len(test_data_for_metrics[test_data_for_metrics['userId'] == row['user']]['movieId'].unique())

    recall /= no_of_users
    precision /= no_of_users
    f_measure = 0.0
    if not (precision == 0.0 and recall == 0.0):
        f_measure = 2 * precision * recall / (precision + recall)
    return precision, recall, f_measure


if __name__ == '__main__':
    train_data, test_data = get_train_and_test_data()
    x_train, y_train = create_utility_mat()
    sim_mat = get_similarity_matrix()
    # y_pred = get_ratings()
    # compare()
    # calculate_rmse()
    #generate_predicted_ratings_files(train_data, test_data)
    o_precision, o_recall, o_f_measure = get_recom_metrics()
    print('Precision: ', o_precision)
    print('Recall: ', o_recall)
    print('F Measure: ', o_f_measure)

    # print(get_top_k_similar_users_rated(train_data, sim_mat, 5, 1, 4))
