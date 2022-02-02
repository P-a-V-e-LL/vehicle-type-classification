import os

path = "" # root_dir/data folder to check
i = 0
l = []

for folder in os.listdir(path):
    n = os.path.join(path, folder)
    i += len(os.listdir(n))
    if len(os.listdir(n)) not in l:
        l.append(len(os.listdir(n)))

#print(sorted(l))
print(i)
#print(len(os.listdir(path)))
