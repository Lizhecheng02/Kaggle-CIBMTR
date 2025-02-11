import json


def get_top_100(json_dict):
    sorted_items = sorted(json_dict.items(), key=lambda item: item[1], reverse=True)
    top_100 = sorted_items[:100]
    return top_100


with open("category_comparison.json", "r") as f:
    json_dict = json.load(f)
top_100_strings = get_top_100(json_dict)

for string, value in top_100_strings:
    print(f"{string}: {value}")
