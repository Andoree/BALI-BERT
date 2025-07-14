

from collections import OrderedDict

lst = [1, 2, 3, 4, 5, 5, 4, 3, 2, 1]
s = list(OrderedDict.fromkeys(lst))
# s = set(lst)
print(lst)
print(s)

lst = [7, 8, 1, 2, 3, 4, 5, 8, 7, 5, 4, 3, 2, 1]
# s = set(lst)
s = list(OrderedDict.fromkeys(lst))
print(lst)
print(s)

lst = [79, 8, 116, 2, 37, 544, 50, 5, 2, 5, 8, 8, 2, 1, 6, 7]
# s = set(lst)
s = list(OrderedDict.fromkeys(lst))
print(lst)
print(s)
