import copy
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
from torch.utils.data import TensorDataset, DataLoader
from tqdm import tqdm


class DataSet:
    def __init__(self, fpath, split):
        self.fpath = fpath
        self.split = split

    def LSTM_load_data(self):
        train_data, test_data, uids, rids, t_name_dict = self.data_process()
        test_data = np.array(test_data).astype('float32')
        test_set = TensorDataset(
            torch.tensor(test_data)
        )
        # rating for train_data
        ratings = train_data.pop('rating')

        # train. val split
        k = int(self.split * len(train_data))  # self split is the percentage of the training data.
        train_ = np.array(train_data[0:k]).astype('float32')
        rating_ = np.array(ratings[0:k]).astype('float32')
        rating_ = rating_ - 1

        validation = np.array(train_data[k:]).astype('float32')
        validation_rating = np.array(ratings[k:]).astype('float32')
        validation_rating = validation_rating - 1

        

        train_set = TensorDataset(
            torch.tensor(train_),
            torch.tensor(rating_)
        )
        valid_set = TensorDataset(
            torch.tensor(validation),
            torch.tensor(validation_rating)
        )

        # test_set = TensorDataset(
        #     torch.tensor(test_data)
        # )
        return train_set, valid_set, test_set

    def data_aug(self, trainData, rating, x_times):
      x = trainData.loc[trainData['rating'] == rating]
      for i in range(x_times):
          trainData = pd.concat([trainData,x])
      return trainData

    def WideAndDeep_loader(self):
        train_data, test_data, uids, rids, t_name_dict = self.data_process()
        train_data = self.data_aug(train_data, 1, 10)
        train_data = self.data_aug(train_data, 2, 15)
        train_data = self.data_aug(train_data, 3, 8)
        train_data = self.data_aug(train_data, 4, 2)
        # we need to encoding the sparse_list
        sparse_list = ['user_id', 'recipe_id', 'name','contributor_id'] #, 'contributor_id'
        train_data = train_data.sample(frac = 1).reset_index(drop=True)
        category_feature_unique = [int(np.max(train_data[f]) + 1) for f in sparse_list]  # get the features
        category_feature_unique_test = [int(np.max(test_data[f]) + 1) for f in sparse_list]  # get the features
        dense_features = [f for f in train_data.columns.tolist() if f not in sparse_list and f != 'rating']
        category_feature_unique = list(map(max, category_feature_unique, category_feature_unique_test))


        ratings = train_data.pop('rating')  # pop rating from training data, now training data does not have rating column
        ratings = ratings - 1  # subtract 1 from the rating, make the ratnig in range [0,4]
        ratings_unique = set(ratings)
        # split dataset train and val
        k = int(self.split * len(train_data))  # self split is the percentage of the training data.
        train_part = train_data[0:k]  # traning set
        valid_part = train_data[k:]  # valid set

        ratings_train = np.array(ratings[0:k]).astype('float32')  # split the rating for tranining set
        # validation data TODO here can be replaced by whole dataset, no need for validation
        ratings_val = np.array(ratings[k:]).astype('float32')  # rating data for val

        # train sparse and dense
        train_sparse = np.array(pd.DataFrame([train_part.pop(x) for x in sparse_list]).T).astype('float32')
        train_dense = np.array(train_part).astype('float32')
        # val spars and dense
        val_sparse = np.array(pd.DataFrame([valid_part.pop(x) for x in sparse_list]).T).astype('float32')
        val_dense = np.array(valid_part).astype('float32')

        # test sparse and dense
        test_sparse = np.array(pd.DataFrame([test_data.pop(x) for x in sparse_list]).T).astype('float32')
        test_dense = np.array(test_data).astype('float32')

        train_set = TensorDataset(
            torch.tensor(train_sparse),
            torch.tensor(train_dense),
            torch.tensor(ratings_train)
        )
        valid_set = TensorDataset(
            torch.tensor(val_sparse),
            torch.tensor(val_dense),
            torch.tensor(ratings_val)
        )

        test_set = TensorDataset(
            torch.tensor(test_sparse),
            torch.tensor(test_dense),
        )
        return train_set, valid_set, test_set, category_feature_unique, dense_features, ratings_unique

    def data_process(self):
        """ get all datasets info """
        train = pd.read_csv(os.path.join(self.fpath, "train.csv"))
        test = pd.read_csv(os.path.join(self.fpath, "test.csv"))
        recipe_info = pd.read_csv(os.path.join(self.fpath, "recipes_info.csv"))

        # recipe_processing
        recipe_info = self.nutrition_expand(recipe_info)

        # recipe_numerical data minmax_scale
        recipe_info = self.data_clipping_and_scale(recipe_info, 'minutes', 0., 800.)  # minutes upper 800,  lower 0 for clipping
        recipe_info = self.data_clipping_and_scale(recipe_info, 'n_steps', 3., 22.)  # n_steps upper 22, lower 3
        recipe_info = self.data_clipping_and_scale(recipe_info, 'n_ingredients', 4., 22.)  # n_integradient upper 22, lower 4
        recipe_info = self.data_clipping_and_scale(recipe_info, 'calories', 0., 1500.)  # upper 1500, lower 0
        recipe_info = self.data_clipping_and_scale(recipe_info, 'totalFat', 0., 500.)  # upper 500, lower 0
        recipe_info = self.data_clipping_and_scale(recipe_info, 'sugar', 0., 400.)  # upper 400, lower 0
        recipe_info = self.data_clipping_and_scale(recipe_info, 'sodium', 0., 200.)  # upper 200, lower 0
        recipe_info = self.data_clipping_and_scale(recipe_info, 'protein', 0., 200.)  # upper 200, lower 0
        recipe_info = self.data_clipping_and_scale(recipe_info, 'saturatedFat', 0., 350.)  # upper 300, lower 0
        recipe_info = self.data_clipping_and_scale(recipe_info, 'carbohydrates', 0., 150.)  # upper 150, lower 0

        train_data = self.merge_data(train, recipe_info)  # merge tarin + recipe_info
        test_data = self.merge_data(test, recipe_info)  # merge test + recipe_info
        user_list = pd.concat([train[['user_id', 'recipe_id']], test[['user_id', 'recipe_id']]])
        # generate a temp dataframe
        recipe_concat = pd.DataFrame()
        recipe_concat[['user_id', 'recipe_id']] = recipe_info[['contributor_id', 'id']]
        # all uids and rids are saved into over_all_concat
        over_all_concat = pd.concat([user_list[['user_id', 'recipe_id']], recipe_concat[['user_id', 'recipe_id']]])
        uids, rids = self.mapping_ids(over_all_concat)  # uids is all users dictionary {user_id:index}   rids is all recipes dictionary {recipe_id:index}

        # Train: user_id, contributor_id, recipe_id replaced by uids and rids
        train_data['user_id'] = train_data.user_id.map(uids)
        train_data['contributor_id'] = train_data.contributor_id.map(uids)
        train_data['recipe_id'] = train_data.recipe_id.map(rids)

        # test:user_id, contributor_id, recipe_id replaced by uids and rids
        test_data['user_id'] = test_data.user_id.map(uids)
        test_data['contributor_id'] = test_data.contributor_id.map(uids)
        test_data['recipe_id'] = test_data.recipe_id.map(rids)

        # deal with the train Date and test Date
        train_data = self.date_processing(train_data, testing=False)
        test_data = self.date_processing(test_data, testing=True)

        # Name processing
        t_name_dict = dict()
        train_data, t_name_dict = self.name_replace(train_data, t_name_dict, testing=False)
        test_data, t_name_dict = self.name_replace(test_data, t_name_dict, testing=True)

        # drop unnessary columns
        train_data = train_data.drop(['date', 'submitted', 'tags', 'steps', 'description', 'ingredients', 'nutrition'], axis=1)
        test_data = test_data.drop(['date', 'submitted', 'tags', 'steps', 'description', 'ingredients', 'nutrition'], axis=1)

        return train_data, test_data, uids, rids, t_name_dict

        ## -------------------------------functions-----------------------------------------

    def merge_data(self, input_data, recipe_info):
        recipe_info = recipe_info.rename(columns={"id": "recipe_id"})
        result = pd.merge(input_data, recipe_info, how="left", on=["recipe_id"])
        result = result[result['user_id'].notna()]
        result['user_id'] = result['user_id'].astype('int64')
        return result

    def mapping_ids(self, concatenation):
        uids = dict()
        rids = dict()
        u_count, r_count = 0, 0
        for row in concatenation.itertuples():
            uid, rid = row[1], row[2]
            if uid not in uids.keys():
                uids[uid] = u_count
                u_count += 1
            if rid not in rids.keys():
                rids[rid] = r_count
                r_count += 1

        print('u_count, r_count ', u_count, r_count)
        return uids, rids

    def date_processing(self, input_data, testing=False):
        if testing:
            input_data['date'] = pd.to_datetime(input_data.date).astype('str')
            input_data[["year", "month", "day"]] = input_data["date"].str.split("-", expand=True).astype('int64')
            input_data["year"] = input_data["year"]
        else:
            input_data[["year", "month", "day"]] = input_data["date"].str.split("-", expand=True).astype('int64')
        input_data[["subyear", "submonth", "subday"]] = input_data["submitted"].str.split("-", expand=True).astype('int64')

        # normalize the date:
        # year [1999, 2018], month [1, 12], day[1, 31]
        input_data = self.min_max_normalize(input_data, 'year', 1999, 2018)
        input_data = self.min_max_normalize(input_data, 'subyear', 1999, 2018)
        input_data = self.min_max_normalize(input_data, 'month', 1, 12)
        input_data = self.min_max_normalize(input_data, 'submonth', 1, 12)
        input_data = self.min_max_normalize(input_data, 'day', 1, 31)
        input_data = self.min_max_normalize(input_data, 'subday', 1, 12)
        return input_data

    def min_max_normalize(self, result, col, lower_bound, upper_bound):
        result[col] = result[col].clip(lower_bound, upper_bound)
        min_ = lower_bound
        max_ = upper_bound
        result[col] = (result[col] - min_) / (max_ - min_)
        return result

    def data_clipping_and_scale(self, input_data, column, lower, upper):
        input_data[column] = input_data[column].clip(lower, upper)
        mean = input_data[column].mean()
        std = input_data[column].std()
        input_data[column] = (input_data[column] - mean) / (std + 1e-12)
        return input_data

    def nutrition_expand(self, recipe_info):
        """
        Expand the nutrition list to dataframe

        """
        recipe_info[['calories', 'totalFat', 'sugar', 'sodium',
                     'protein', 'saturatedFat', 'carbohydrates']] = recipe_info.nutrition.str.split(",", expand=True)
        recipe_info['calories'] = recipe_info['calories'].str.replace('[', '')
        recipe_info['carbohydrates'] = recipe_info['carbohydrates'].str.replace(']', '')
        recipe_info[['calories', 'totalFat', 'sugar', 'sodium',
                     'protein', 'saturatedFat', 'carbohydrates']] = recipe_info[['calories', 'totalFat', 'sugar', 'sodium',
                                                                                 'protein', 'saturatedFat', 'carbohydrates']].astype('float32')
        recipe_info['n_steps'] = recipe_info['n_steps'].astype('float32')
        recipe_info['n_ingredients'] = recipe_info['n_ingredients'].astype('float32')
        recipe_info['minutes'] = recipe_info['minutes'].astype('float32')

        return recipe_info

    # convert recipe name to index, the name has been saved in name_dict {name: index}
    def name_replace(self, input_data, name_dict, testing=False):
        if testing == False:
            names = input_data['name'].value_counts()  # find all names
            count_names = names.to_dict()  # save as a dictionary {name : freq}
            count_names = list(count_names.keys())  # get all names and save as a list
            name_dict = dict((v, i) for i, v in enumerate(count_names))  # save into the dictionary, and give a index for each name {name: index}
            input_data['name'] = input_data.name.map(name_dict)
            # index = input_data['name'].index[input_data['name'].apply(np.isnan)]
            # index = index.values
        else:
            input_data['name'] = input_data.name.map(name_dict)
            # index = input_data['name'].index[input_data['name'].apply(np.isnan)]
            # index = index.values
        # for i in tqdm(range(len(index))):
        input_data['name'] = input_data['name'].fillna(len(name_dict.values()))
        # input_data['name'] = input_data['name'].replace(np.nan, len(name_dict.values())) # if there has name is "NaN", replace it as last index
        input_data['name'] = input_data['name'].astype('int64')
        return input_data, name_dict


def wideAndDeep(path, split, batch, data_type='wideAndDeep'):
    data = DataSet(path, split)
    if data_type == 'wideAndDeep':
        print("Wide&Deep Data processing.............")
        df = data.WideAndDeep_loader()
    else:
        print("LSTM Data Processing...........")
        df = data.LSTM_load_data()
        training_data = DataLoader(df[0], shuffle=False, batch_size=batch, drop_last=True)
        validation_data = DataLoader(df[1], shuffle=False, batch_size=batch, drop_last=True)
        testing_data = DataLoader(df[2], shuffle=False, batch_size=batch, drop_last=False)
        return training_data, validation_data, testing_data

    # print(df[0].shape)
    training_data = DataLoader(df[0], shuffle=True, batch_size=batch, drop_last=True)
    validation_data = DataLoader(df[1], shuffle=False, batch_size=batch, drop_last=True)
    testing_data = DataLoader(df[2], shuffle=False, batch_size=batch, drop_last=False)
    print("------------------Done-------------------")
    return training_data, validation_data, testing_data, df[3], df[4], df[5]
