


# if __name__ == "__main__":
#     df = pd.read_csv(Path(__file__).parent.parent.absolute() / f'benchmarks/mc/gp_files/samples200.csv').astype('float32')
#     evo = Evolution()
#     gp = ExplainableGP(evo, df)
#     pop = gp.gen_create_initial()
#     for _ in range(10):
#         gp.