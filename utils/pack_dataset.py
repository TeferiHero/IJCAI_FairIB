import pickle
import os

import numpy as np
import pandas as pd
import sklearn.model_selection as sk
# filename = "lastfm-360k"




if __name__ == '__main__':
    filename_mod = "-1to24"
    filename = "gender-bias"
    fileDirPath = f"../data/{filename}/process"
    os.makedirs(fileDirPath, mode=0o777, exist_ok=True)

    users = pd.read_csv(f'{fileDirPath}/Users.csv', sep=";", dtype={'Age': str})
    users_filtered = users[users['Age'].astype(str).str.isnumeric()].copy()
    users_filtered['Age'] = users_filtered['Age'].astype(int)

    # BARRIER = 40 # 2:1
    # BARRIER = 32 # 1:1
    # BARRIER = 25 # 1:2
    # BARRIER = 22 # 1:4
    # BARRIER = 20 # 1:6
    # BARRIER = 18 # 1:10
    BARRIER = 15 # 1:24


    # for i in range(10, 20):
    #     print(i)
    #     _, counts = np.unique((users_filtered['Age'] > i).astype(int), return_counts=True)
    #     print(counts[1]/counts[0])
    #     print(counts)
    #
    # # print(np.median(users_filtered['Age']))
    # quit()
    users_filtered['Age'] = (users_filtered['Age'] > BARRIER).astype(int)



    users_filtered['User-ID-converted'], _ = pd.factorize(users_filtered['User-ID'])

    df = pd.read_csv(f'{fileDirPath}/Ratings.csv', sep=";", dtype={'User-ID': int, 'ISBN': str})
    recommendations = pd.merge(users_filtered, df, on="User-ID", how="inner")

    recommendations['ISBN'], _ = pd.factorize(recommendations['ISBN'])
    recommendations['User-ID'] = recommendations['User-ID-converted']
    users_filtered['User-ID'] = users_filtered['User-ID-converted']

    users_filtered = users_filtered.drop(columns=['User-ID-converted'])
    recommendations = recommendations.drop(columns=['Age', 'User-ID-converted'], errors='ignore')


    isbn_counts = recommendations['ISBN'].value_counts()
    valid_isbn = isbn_counts[isbn_counts >= 6].index
    recommendations = recommendations[recommendations['ISBN'].isin(valid_isbn)]

    rec_counts = recommendations['User-ID'].value_counts()
    valid_users = rec_counts[rec_counts >= 4].index
    recommendations = recommendations[recommendations['User-ID'].isin(valid_users)]
    users_filtered = users_filtered[users_filtered['User-ID'].isin(valid_users)]

    recommendations['ISBN'], _ = pd.factorize(recommendations['ISBN'])
    users_filtered['User-ID-converted2'], _ = pd.factorize(users_filtered['User-ID'])

    recommendations['User-ID'] = recommendations['User-ID'].map(users_filtered.set_index('User-ID')['User-ID-converted2'])
    users_filtered['User-ID'] = users_filtered['User-ID-converted2']

    train, test = sk.train_test_split(recommendations, test_size=0.2, random_state=42)



    train_u2i = (
        train.groupby('User-ID')['ISBN']
        .apply(lambda x: sorted(x.tolist()))
        .to_dict()
    )
    test_u2i = (
        test.groupby('User-ID')['ISBN']
        .apply(lambda x: sorted(x.tolist()))
        .to_dict()
    )

    train_i2u = (
        train.groupby('ISBN')['User-ID']
        .apply(lambda x: sorted(x.tolist()))
        .to_dict()
    )
    test_i2u = (
        test.groupby('ISBN')['User-ID']
        .apply(lambda x: sorted(x.tolist()))
        .to_dict()
    )

    train_set = {
        'userid': np.array(train['User-ID']),
        'itemid': np.array(train['ISBN']),
        'rating': np.array(train['Rating'])
    }
    test_set = {
        'userid': np.array(test['User-ID']),
        'itemid': np.array(test['ISBN']),
        'rating': np.array(test['Rating'])
    }

    user_side_features = {
        'userid': np.array(users_filtered['User-ID']),
        'age': np.array(users_filtered['Age'])
    }

    # books = pd.read_csv(f'{fileDirPath}/Books.csv', sep=";", dtype={'ISBN': str})
    n_users = len(users_filtered)
    # n_items = len(books)
    n_items = max(recommendations['ISBN']) + 1
    print(n_users, n_items)

    for i in range(n_users):
        train_u2i.setdefault(i, [])
        test_u2i.setdefault(i, [])

    for i in range(n_items):
        train_i2u.setdefault(i, [])
        test_i2u.setdefault(i, [])

    with open(f'{fileDirPath}/process{filename_mod}.pkl', 'wb') as f:
        pickle.dump(train_u2i, f)
        pickle.dump(train_i2u, f)
        pickle.dump(test_u2i, f)
        pickle.dump(test_i2u, f)
        pickle.dump(train_set, f) # 'userid': [0,    0, ..., 6039, 6039], 'itemid': [1192,  744, ..., 3181,  299], 'rating':[5, 3, ..., 5, 2]}
        pickle.dump(test_set, f)
        pickle.dump(user_side_features, f) #dictionary with userid and lists for each feature (age, gender,occ,f)
        pickle.dump((n_users, n_items), f) # number of users and number of items

    def getvalues(data):
        count = 0
        maxim = 0
        suma = 0
        for i in data.values():
            # print(data)
            if type(i) is list:
                count+=len(i)
                maxim = max(maxim, max(i))
            else:
                count
                print(i)

        print(f"count = {count}")
        print(f"max = {maxim}")
    # head = [next(data) for _ in range(lines_number)]
    # print(data)
    # print(data2 == data)
    # print(user_side_features)

    def printFeatures(user_side_features):
        print("user side features")
        for i in user_side_features.keys():
            print(i)


    # printFeatures(user_side_features)