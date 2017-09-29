import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import collections


def preprocess():

    path = 'D:/'
    file_name = 'data.csv'

    # load data
    data = pd.read_csv(path + file_name)

    # extract 'Order_Qty' column to be used as labels
    y = data['Order_Qty'].values
    # replace all non-zero values with one
    y[y > 0] = 1
    # total number of data entries
    data_size = len(y)
    dict_order_qty = dict(collections.Counter(y))
    keys_order_qty = list(dict_order_qty.keys())
    for i in range(0, len(keys_order_qty)):
        key = keys_order_qty[i]
        dict_order_qty[key] = dict_order_qty.get(key) / data_size

    # extract 'Country' column
    countries = data['Country'].values
    # compute frequency of each country
    dict_country = dict(collections.Counter(countries))
    keys_country = list(dict_country.keys())
    for i in range(0, len(keys_country)):
        key = keys_country[i]
        dict_country[key] = dict_country.get(key) / data_size

    # extract 'SKU' column
    skus = data['SKU'].values
    # compute frequency of each SKU
    dict_sku = dict(collections.Counter(skus))
    keys_sku = list(dict_sku.keys())
    for i in range(0, len(keys_sku)):
        key = keys_sku[i]
        dict_sku[key] = dict_sku.get(key) / data_size

    # extract 'SKU_Category' column
    sku_categories = data['SKU_Category'].values
    # compute frequency of each SKU category
    dict_sku_category = dict(collections.Counter(sku_categories))
    keys_sku_category = list(dict_sku_category.keys())
    for i in range(0, len(keys_sku_category)):
        key = keys_sku_category[i]
        dict_sku_category[key] = dict_sku_category.get(key) / data_size

    # extract 'EB_Flag' column
    eb_flags = data['EB_Flag'].values
    # compute frequency of each EB_Flag
    dict_eb_flag = dict(collections.Counter(eb_flags))
    keys_eb_flag = list(dict_eb_flag.keys())
    for i in range(0, len(keys_eb_flag)):
        key = keys_eb_flag[i]
        dict_eb_flag[key] = dict_eb_flag.get(key) / data_size

    # extract 'RFQ_TYPE' column
    rfq_types = [str(i) for i in data['RFQ_TYPE'].values]
    # compute frequency of each RFQ_TYPE
    dict_rfq_type = dict(collections.Counter(rfq_types))
    keys_rfq_type = list(dict_rfq_type.keys())
    for i in range(0, len(keys_rfq_type)):
        key = keys_rfq_type[i]
        dict_rfq_type[key] = dict_rfq_type.get(key) / data_size

    # One-hot encoding of RFQ_Type
    le_rfq_types = LabelEncoder()
    labels_rfq_types = le_rfq_types.fit_transform(rfq_types)
    rfq_types_b = np.zeros((len(rfq_types), 9), dtype=int)
    rfq_types_b[np.arange(len(rfq_types), dtype=int), labels_rfq_types] = 1

    # print one-hot encoded samples
    print('\nOne-hot encoded samples (RFQ_Type):')
    print('3: {}'.format(rfq_types_b[10]))
    print('NaN: {}'.format(rfq_types_b[29839]))
    print('7: {}'.format(rfq_types_b[29997]))
    print('8: {}'.format(rfq_types_b[30005]))
    print('2: {}'.format(rfq_types_b[30112]))
    print('6: {}'.format(rfq_types_b[119100]))
    print('4: {}'.format(rfq_types_b[136174]))
    print('9: {}'.format(rfq_types_b[143901]))
    print('1: {}'.format(rfq_types_b[154238]))

    # write result into a text file
    file = open('result.txt', 'w')
    file.write("Country:")
    file.write('\nKeys: {}'.format(keys_country))
    file.write('\nNo. of keys: {}'.format(len(keys_country)))
    file.write('\nPercentage of each category: {}'.format(dict_country))
    file.write("\n\nSKU:")
    file.write('\nKeys: {}'.format(keys_sku))
    file.write('\nNo. of keys: {}'.format(len(keys_sku)))
    file.write('\nPercentage of each category: {}'.format(dict_sku))
    file.write("\n\nSKU_Category:")
    file.write('\nKeys: {}'.format(keys_sku_category))
    file.write('\nNo. of keys: {}'.format(len(keys_sku_category)))
    file.write('\nPercentage of each category: {}'.format(dict_sku_category))
    file.write("\n\nEB_Flag:")
    file.write('\nKeys: {}'.format(keys_eb_flag))
    file.write('\nNo. of keys: {}'.format(len(keys_eb_flag)))
    file.write('\nPercentage of each category: {}'.format(dict_eb_flag))
    file.write("\n\nRFQ_Type:")
    file.write('\nKeys: {}'.format(keys_rfq_type))
    file.write('\nNo. of keys: {}'.format(len(keys_rfq_type)))
    file.write('\nPercentage of each category: {}'.format(dict_rfq_type))
    file.write('\nOne-hot Encoded RFQ_Type:\n{}'.format(rfq_types_b))
    file.write("\n\nOrder_Qty:")
    file.write('\nKeys: {}'.format(keys_order_qty))
    file.write('\nNo. of keys: {}'.format(len(keys_order_qty)))
    file.write('\nPercentage of each category: {}'.format(dict_order_qty))
    file.close()

    return

preprocess()