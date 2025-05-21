
# open this dict in python

import json

BOOL_GLOBAL_OBJECT_NAMES_PATH = "/home/disi/Alessio/IPDN-clone/bool_global_object_names__20250507-101601_winsize20_minocc6.json"
with open(BOOL_GLOBAL_OBJECT_NAMES_PATH, "r") as f:
    bool_global_object_names = json.load(f)
# print the first 10 keys
print("type of bool_global_object_names:", type(bool_global_object_names))
print("len of bool_global_object_names:", len(bool_global_object_names))
print("first 10 keys:", list(bool_global_object_names.keys())[:10])
# print the first 10 values
print("first 10 values:", list(bool_global_object_names.values())[:10])



DICT_PATH = "/nfs/data_todi/jli/Alessio_works/RAM-clone/output_preprocess_object_ids/global_object_names__20250507-101601_winsize20_minocc6.json"

with open(DICT_PATH, "r") as f:
    global_object_names = json.load(f)

FOLDER_PROCESSING = "/home/disi/Alessio/IPDN-clone/preprocess-claude/"

# save only the keys in a list in a json file
keys = list(global_object_names.keys())
with open(f"{FOLDER_PROCESSING}global_object_names_keys.json", "w") as f:
    json.dump(keys, f)

# open the json file and read it
with open(f"{FOLDER_PROCESSING}global_object_names_keys.json", "r") as f:
    keys = json.load(f)

# save it on a .txt file each key in a new line
with open(f"{FOLDER_PROCESSING}global_object_names_keys.txt", "w") as f:
    for key in keys:
        f.write(key + "\n")

# now with this open list I will load a list of boolean values
# associated with the keys, creting a dict with the keys and the boolean values
# open the json file and read it
with open(f"{FOLDER_PROCESSING}global_object_names_keys.json", "r") as f:
    keys = json.load(f)
# create a dict with the keys and the boolean values
keys_dict = {key: False for key in keys}
# save the dict in a json file
        
# print the first 10 keys
print("type of keys:", type(keys))
print(keys[:10])


values_1_50= [False, True, True, True, True, True, True, True, False, True, True, True, True, True, True, True, True, True, True, False, True, True, True, True, True, False, True, True, True, True, True, False, True, True, False, True, True, True, True, True, False, True, True, False, True, True, True, True, False, True]
values_51_100 = [True, False, True, False, True, True, False, False, True, True, True, True, True, True, True, False, True, True, False, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, False, True, True, False, True, False, True, False, True, True, True, False, True]
values_101_150 = [True, True, False, True, False, True, True, False, True, True, True, False, True, False, True, False, False, False, True, False, True, True, False, True, True, True, True, True, True, True, True, True, True, True, True, False, True, True, False, True, True, True, True, True, True, True, True, True, False, True]
values_151_200 = [True, True, True, True, False, True, True, False, True, True, False, True, True, True, True, False, True, True, False, True, True, False, True, True, True, False, True, True, True, True, True, True, True, True, True, False, True, False, True, True, False, True, True, True, True, True, True, True, True, True]
values_201_280 = [True, True, False, True, True, True, True, True, True, True, True, True, True, False, True, True, True, False, True, True, False, True, True, True, False, True, True, True, True, True, True, True, True, True, True, False, False, True, False, False, False, True, True, True, False, False, True, True, True, True, True, True, True, True, True, True, True, False, False, True, True, False, False, True, True, True, True, True, True, True, True, True, True, True, False, False, True, True, True, True]

values_281_330 =  [True, True, True, True, True, True, True, True, True, True, True, True, True, False, True, False, True, True, False, True, False, True, True, True, True, True, False, False, True, True, False, True, True, False, True, True, True, True, True, True, True, True, True, False, False, True, True, True, False, True]
values_331_380 = [True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, False, False, False, True, False, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, False, True, True, True, True, False, True, True, True, True]
values_381_430 = [True, True, True, True, False, True, True, True, False, True, True, True, True, False, True, False, False, True, True, True, False, True, False, True, True, False, True, False, True, True, True, True, True, True, True, False, True, True, True, True, True, True, False, True, True, True, True, True, True, True]
values_431_500 =  [True, True, False, True, True, True, True, False, False, False, False, True, True, True, False, False, True, False, True, True, False, True, False, False, True, True, True, True, True, True, True, False, True, True, True, True, False, True, False, True, True, True, True, True, True, True, True, True, True, True, False, True, True, False, True, True, True, True, True, True, True, False, True, True, True, True, True, False, True, True]
values_501_550 =  [False, False, False, False, False, True, True, True, False, True, False, True, True, True, True, False, True, True, True, True, True, True, True, True, True, False, False, False, True, True, True, True, False, False, True, True, False, True, True, True, True, True, True, False, True, True, True, True, True, True]

values_551_600 = [False, False, False, False, False, True, True, True, False, True, False, True, True, True, True, False, True, True, True, True, True, True, True, True, True, False, False, False, True, True, True, True, False, False, True, True, False, True, True, True, True, True, True, False, True, True, True, True, True, True]
values_601_641 = [False, True, True, True, True, True, True, True, True, True, False, True, True, False, True, False, True, False, True, True, True, False, False, True, True, True, False, True, True, True, True, True, True, True, False, True, True, True, True, True, True]
# values_501_550 = []
# values_551_600 = []
# values_601_650 = []
# values_651_700 = []

print("len of values_1_50:", len(values_1_50))
print("len of values_51_100:", len(values_51_100))
print("len of values_101_150:", len(values_101_150))
print("len of values_151_200:", len(values_151_200))
print("len of values_201_280:", len(values_201_280))

print("len of values_281_330:", len(values_281_330))
print("len of values_331_380:", len(values_331_380))
print("len of values_381_430:", len(values_381_430))
print("len of values_431_500:", len(values_431_500))
print("len of values_501_550:", len(values_501_550))

print("len of values_551_600:", len(values_551_600))
print("len of values_601_641:", len(values_601_641))

all_values = values_1_50 + values_51_100 + values_101_150 + values_151_200 + values_201_280 + values_281_330 + values_331_380 + values_381_430 + values_431_500 + values_501_550 + values_551_600 + values_601_641
print("len of all values:", len(all_values))

print("length of keys:", len(keys))
# print("length of values:", len(values))

# now create a dict with the keys and the values
keys_values_dict = {key: value for key, value in zip(keys, all_values)}
# save the dict in a json file
with open(f"{FOLDER_PROCESSING}global_object_names_keys_values.json", "w") as f:
    json.dump(keys_values_dict, f)

# # save those values in a .txt file one row for each value
# with open("/nfs/data_todi/jli/Alessio_works/IPDN-clone/global_object_names_keys_values.txt", "w") as f:
#     for value in values:
#         f.write(str(value) + "\n")