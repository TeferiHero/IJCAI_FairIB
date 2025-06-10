import pickle
import os
# filename = "lastfm-360k"



filename = "other"
fileDirPath = f"data/{filename}/process"
os.makedirs(fileDirPath, mode=0o777, exist_ok=True)

with open(f'{fileDirPath}/process.pkl', 'rb') as f:
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
print(user_side_features)

def printFeatures(user_side_features):
    print("user side features")
    for i in user_side_features.keys():
        print(i)


printFeatures(user_side_features)