import numpy as np
import pickle

favorite_color = {"lion": "yellow", "kitty": "red"}
favorite_number = {"a": 4, "b": 6}
mix_dict = {'fav_col': favorite_color,
            'fav_num': favorite_number}
pickle.dump(mix_dict, open("save.p", "wb"))

mix_dict = pickle.load(open("save.p", "rb"))

print(mix_dict)
a, b = mix_dict

print(type(a), b)
