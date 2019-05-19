#_autor: Thinkpad
#date: 2019/5/19
from collections import defaultdict

user_ratings = defaultdict(set)
u = 1
user_ratings[u].add(1)
user_ratings[u].add(2)

#键值对返回：键和多个值
for u ,i_items in user_ratings.items():
    print(u,i_items)