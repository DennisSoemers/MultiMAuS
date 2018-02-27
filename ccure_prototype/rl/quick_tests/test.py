import pickle

data = pickle.load(open("D:\\Apps\\MultiMAuS-fork\\ccure_prototype\\data\\first_auth_layer_data.pickle", "rb"),
                   encoding='latin1')

for key in data:
    print("key = ", key)
    print("val = ", data[key])
    break
