import random

file = open("name_list.txt", "r")
contents = file.readlines()
N = len(contents)
choice = random.randint(0, N-1)
print("Randomly selected name is: ", contents[choice])
file.close()