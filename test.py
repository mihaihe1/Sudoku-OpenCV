from collections import deque

a = deque([(1,2)])
coord = a.popleft()
x = coord[0]
y = coord[1]
print(x, y)