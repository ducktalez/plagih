


if __name__ == "__main__":
    evo = Evolution()
    for _ in range(10):
        tree = evo.evolve_create_random(float, 4)
        print(tree)