import pickle
import os
import pandas as pd
import sklearn.model_selection as sk
# filename = "lastfm-360k"



filename = "gender-bias"
fileDirPath = f"../data/{filename}/process"
os.makedirs(fileDirPath, mode=0o777, exist_ok=True)

users = pd.read_csv(f'{fileDirPath}/Users.csv', sep=";", dtype={'Age': str})
users_filtered = users[users['Age'].astype(str).str.isnumeric()].copy()

df = pd.read_csv(f'{fileDirPath}/Ratings.csv', sep=";", dtype={'User-ID': int, 'ISBN': str})
recommendations = pd.merge(users_filtered, df, on="User-ID", how="inner")
recommendations = recommendations.drop(columns=['Age'], errors='ignore')

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
    'userid': train['User-ID'].tolist(),
    'itemid': train['ISBN'].tolist(),
    'rating': train['Rating'].tolist()
}
test_set = {
    'userid': test['User-ID'].tolist(),
    'itemid': test['ISBN'].tolist(),
    'rating': test['Rating'].tolist()
}

user_side_features = {
    'userid': users_filtered['User-ID'].tolist(),
    'age': users_filtered['Age'].tolist()
}

books = pd.read_csv(f'{fileDirPath}/Books.csv', sep=";", dtype={'ISBN': str})
n_users = len(users_filtered)
n_items = len(books)

print(n_users, n_items)

# with open(f'{fileDirPath}/process.pkl', 'rb') as f:
    # pickle.dump(train_u2i, f)
    # pickle.dump(train_i2u, f)
    # pickle.dump(test_u2i, f)
    # pickle.dump(test_i2u, f)
    # pickle.dump(train_set, f) # 'userid': [0,    0, ..., 6039, 6039], 'itemid': [1192,  744, ..., 3181,  299], 'rating':[5, 3, ..., 5, 2]}
    # pickle.dump(test_set, f)
    # pickle.dump(user_side_features, f) #dictionary with userid and lists for each feature (age, gender,occ,f)
    # pickle.dump((n_users, n_items), f) # number of users and number of items

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
print(user_side_features)

def printFeatures(user_side_features):
    print("user side features")
    for i in user_side_features.keys():
        print(i)


printFeatures(user_side_features)