

if __name__ == '__main__':
    for oo, ocls in op.items():
        print('operators:', oo, ocls.nlabel, ocls.xtype)

    for x in [FloatConstant(0.44), BoolConstant(True)]:
        print(f'x: {x.nlabel}, {x.xtype}')