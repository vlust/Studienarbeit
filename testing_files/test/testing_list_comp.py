from random import uniform

def add_random_cone(max_x, max_y, n):
        r_cones=[(uniform(0, max_x), uniform(0, max_y)) for _ in range(n)]
        return r_cones
print(add_random_cone(10, 10, 5))