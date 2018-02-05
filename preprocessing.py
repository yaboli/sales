import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from keras.utils import np_utils
import collections
import time

file_name = 'data.csv'


def encode_one_hot(column, num):
    top_n_tuple = collections.Counter(column).most_common(num)
    top_n = []
    # add category to the list
    for tup in top_n_tuple:
        top_n.append(tup[0])
    # update column
    for i in range(0, len(column)):
        if column[i] not in top_n:
            column[i] = 'other'
    # encode column using one-hot encoding
    le = LabelEncoder()
    labels = le.fit_transform(column)
    encoded = np.zeros((len(column), num + 1), dtype=int)
    encoded[np.arange(len(column), dtype=int), labels] = 1
    return encoded


def tobits(s):
    result = []
    for c in s:
        bits = bin(ord(c))[2:]
        bits = '00000000'[len(bits):] + bits
        result.extend([int(b) for b in bits])
    return result


def encode_ascii(column):
    encoded = []
    for value in column:
        new_value = tobits(value)
        # deal with the case in sku_category when value has 4 chars
        if len(new_value) == 32:
            new_value = [0, 0, 0, 0, 0, 0, 0, 0] + new_value
        encoded.append(new_value)
    return encoded


def preprocess():

    start = time.time()

    # load data
    data = pd.read_csv(file_name)

    # drop rows with negative 'Order_Qty' value
    data = data[data.Order_Qty >= 0]

    # extract 'Order_Qty' column to be used as labels
    y = data['Order_Qty'].values
    # replace all non-zero values with one
    y[y > 0] = 1
    # drop 'Order_Qty' column
    data.drop('Order_Qty', axis=1, inplace=True)
    # total number of valid data sets
    data_size = len(y)
    dict_order_qty = dict(collections.Counter(y))
    keys_order_qty = list(dict_order_qty.keys())

    # extract 'Country' column
    countries = data['Country'].values
    # compute frequency of each country
    dict_country = dict(collections.Counter(countries))
    keys_country = list(dict_country.keys())
    le_country = LabelEncoder()
    labels_country = le_country.fit_transform(countries)
    country_enc = np.zeros((len(countries), len(keys_country)), dtype=int)
    country_enc[np.arange(len(countries), dtype=int), labels_country] = 1

    # extract 'Coverage' column
    coverage = data['Coverage'].values
    dict_coverage = dict(collections.Counter(coverage))
    keys_coverage = list(dict_coverage.keys())
    # one-hot encode 'Coverage'
    le_coverage = LabelEncoder()
    labels_coverage = le_coverage.fit_transform(coverage)
    coverage_enc = np.zeros((len(coverage), len(keys_coverage)), dtype=int)
    coverage_enc[np.arange(len(coverage), dtype=int), labels_coverage] = 1

    # extract 'SKU' column
    skus = [str(i) for i in data['SKU'].values]
    # compute frequency of each SKU
    dict_sku = dict(collections.Counter(skus))
    keys_sku = list(dict_sku.keys())
    # one-hot encode 'SKU'
    top_skus = 1000
    sku_enc = encode_one_hot(skus, top_skus)
    # le_sku = LabelEncoder()
    # labels_sku = le_sku.fit_transform(skus)
    # sku_enc = np.zeros((len(skus), len(keys_sku)), dtype=int)
    # sku_enc[np.arange(len(skus), dtype=int), labels_sku] = 1

    # extract 'SKU_Category' column
    sku_categories = [str(i) for i in data['SKU_Category'].values]
    # compute frequency of each SKU category
    dict_sku_category = dict(collections.Counter(sku_categories))
    keys_sku_category = list(dict_sku_category.keys())
    # one-hot encode 'SKU_Category'
    top_sku_categories = 300
    sku_category_enc = encode_one_hot(sku_categories, top_sku_categories)
    # le_sku_category = LabelEncoder()
    # labels_sku_category = le_sku_category.fit_transform(sku_categories)
    # sku_category_enc = np.zeros((len(sku_categories), len(keys_sku_category)), dtype=int)
    # sku_category_enc[np.arange(len(sku_categories), dtype=int), labels_sku_category] = 1

    # extract 'EB_Flag' column
    eb_flag = data['EB_Flag'].values
    # compute frequency of each EB_Flag
    dict_eb_flag = dict(collections.Counter(eb_flag))
    keys_eb_flag = list(dict_eb_flag.keys())
    # one-hot encode 'EB_Flag'
    le_eb_flag = LabelEncoder()
    labels_eb_flag = le_eb_flag.fit_transform(eb_flag)
    eb_flag_enc = np.zeros((len(eb_flag), 2), dtype=int)
    eb_flag_enc[np.arange(len(eb_flag), dtype=int), labels_eb_flag] = 1

    # extract 'RFQ_TYPE' column
    rfq_type = [str(i) for i in data['RFQ_TYPE'].values]
    # compute frequency of each RFQ_TYPE
    dict_rfq_type = dict(collections.Counter(rfq_type))
    keys_rfq_type = list(dict_rfq_type.keys())
    # One-hot encode RFQ_Type
    le_rfq_type = LabelEncoder()
    labels_rfq_type = le_rfq_type.fit_transform(rfq_type)
    rfq_type_enc = np.zeros((len(rfq_type), 9), dtype=int)
    rfq_type_enc[np.arange(len(rfq_type), dtype=int), labels_rfq_type] = 1

    # extract 'List_Price' column
    list_price = data['List_Price'].values
    # scale data to [0, 1]
    min_max_scaler = MinMaxScaler()
    list_price_norm = np.array(min_max_scaler.fit_transform(np.array(list_price).reshape(-1, 1)))

    # extract 'RFQ_Price' column
    rfq_price = data['RFQ_Price'].values
    # scale data to [0, 1]
    min_max_scaler = MinMaxScaler()
    rfq_price_norm = np.array(min_max_scaler.fit_transform(np.array(rfq_price).reshape(-1, 1)))

    country_enc = np.array(country_enc)
    coverage_enc = np.array(coverage_enc)
    sku_enc = np.array(sku_enc)
    sku_category_enc = np.array(sku_category_enc)
    eb_flag_enc = np.array(eb_flag_enc)
    rfq_type_enc = np.array(rfq_type_enc)

    # concatenate all encoded and normalized arrays
    X = np.concatenate((country_enc, coverage_enc), axis=1)
    X = np.concatenate((X, sku_enc), axis=1)
    X = np.concatenate((X, sku_category_enc), axis=1)
    X = np.concatenate((X, eb_flag_enc), axis=1)
    X = np.concatenate((X, rfq_type_enc), axis=1)
    X = np.concatenate((X, list_price_norm), axis=1)
    X = np.concatenate((X, rfq_price_norm), axis=1)
    print(X.shape)

    # split into train&cv and test sets
    test_size = 0.3
    X_train_and_cv, X_test, y_train_and_cv, y_test = train_test_split(X, y, test_size=test_size)

    # split into train and cv sets
    cv_size = 0.2
    X_train, X_cv, y_train, y_cv = train_test_split(X_train_and_cv,
                                                    y_train_and_cv,
                                                    test_size=cv_size)

    # # compute percentage of each key
    # for i in range(0, len(keys_order_qty)):
    #     key = keys_order_qty[i]
    #     dict_order_qty[key] = dict_order_qty.get(key) / data_size
    # for i in range(0, len(keys_country)):
    #     key = keys_country[i]
    #     dict_country[key] = dict_country.get(key) / data_size
    # for i in range(0, len(keys_sku)):
    #     key = keys_sku[i]
    #     dict_sku[key] = dict_sku.get(key) / data_size
    # for i in range(0, len(keys_sku_category)):
    #     key = keys_sku_category[i]
    #     dict_sku_category[key] = dict_sku_category.get(key) / data_size
    # for i in range(0, len(keys_eb_flag)):
    #     key = keys_eb_flag[i]
    #     dict_eb_flag[key] = dict_eb_flag.get(key) / data_size
    # for i in range(0, len(keys_rfq_type)):
    #     key = keys_rfq_type[i]
    #     dict_rfq_type[key] = dict_rfq_type.get(key) / data_size

    # # write result into a text file
    # file = open('result.txt', 'w')
    # file.write('Number of valid data sets: {}'.format(data_size))
    # file.write("\n\nCountry:")
    # file.write('\nKeys: {}'.format(keys_country))
    # file.write('\nNo. of keys: {}'.format(len(keys_country)))
    # file.write('\nFrequency of each category: {}'.format(dict_country))
    # file.write("\n\nCoverage:")
    # file.write('\nKeys: {}'.format(keys_coverage))
    # file.write('\nNo. of keys: {}'.format(len(keys_coverage)))
    # file.write('\nFrequency of each category: {}'.format(dict_coverage))
    # file.write("\n\nSKU:")
    # file.write('\nKeys: {}'.format(keys_sku))
    # file.write('\nNo. of keys: {}'.format(len(keys_sku)))
    # file.write('\nFrequency of each category: {}'.format(dict_sku))
    # file.write("\n\nSKU_Category:")
    # file.write('\nKeys: {}'.format(keys_sku_category))
    # file.write('\nNo. of keys: {}'.format(len(keys_sku_category)))
    # file.write('\nFrequency of each category: {}'.format(dict_sku_category))
    # file.write("\n\nEB_Flag:")
    # file.write('\nKeys: {}'.format(keys_eb_flag))
    # file.write('\nNo. of keys: {}'.format(len(keys_eb_flag)))
    # file.write('\nFrequency of each category: {}'.format(dict_eb_flag))
    # file.write("\n\nRFQ_Type:")
    # file.write('\nKeys: {}'.format(keys_rfq_type))
    # file.write('\nNo. of keys: {}'.format(len(keys_rfq_type)))
    # file.write('\nFrequency of each category: {}'.format(dict_rfq_type))
    # file.write("\n\nOrder_Qty:")
    # file.write('\nKeys: {}'.format(keys_order_qty))
    # file.write('\nNo. of keys: {}'.format(len(keys_order_qty)))
    # file.write('\nFrequency of each category: {}'.format(dict_order_qty))
    # file.close()

    end = time.time()

    print('\nPREPROCESSING COMPLETE\nTime elapsed: {:.2f} {}'.format((end - start), 'seconds'))

    return X_train, X_cv, X_test, y_train, y_cv, y_test


# preprocess()
