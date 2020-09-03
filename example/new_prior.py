import json

def new_prior(x):
    return (x[0] + x[1]) / 3

with open("rastrigin_archive.json") as fp:
    new_file = json.load(fp)

for _, v in new_file.items():
    print(v)
    v["performance"] = new_prior(v["centroid"])
    v["solution"] = [*v["centroid"], 0, 0, 0, 0]
    v["features"] = {"feature_a": v["centroid"][0], "feature_b": v["centroid"][1]}

with open("new_prior.json", "w") as fp:
    json.dump(new_file, fp)

