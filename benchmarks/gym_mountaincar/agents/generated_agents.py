import math


class MTC_simple0:

    def decide(self, input):
        cartPos, cartVel = input
        action = 0 if (cartVel < 0) else 2
        return max(0, min(2, int(round(action))))


class MTC_simple7:

    def decide(self, input):
        cartPos, cartVel = input
        action = (((-3.425) * cartPos) + (lambda x, y: x / y if y != 0 else 0)(0.11, cartVel))
        return max(0, min(2, int(round(action))))


class MTC_simple10:

    def decide(self, input):
        cartPos, cartVel = input
        action = (min(max((lambda x, y: x / y if y != 0 else 0)(5.0, cartVel), (lambda x, y: x / y if y != 0 else 0)(4.0, cartPos)), 1.265) + 5.0)
        return max(0, min(2, int(round(action))))


class MTC_simple15:

    def decide(self, input):
        cartPos, cartVel = input
        action = ((((-cartPos) * abs(((-0.34) + (lambda x, y: x / y if y != 0 else 0)(0.235, cartVel)))) - 0.005) + (lambda x, y: x / y if y != 0 else 0)(0.235, cartVel))
        return max(0, min(2, int(round(action))))


class MTC_simple18:

    def decide(self, input):
        cartPos, cartVel = input
        action = ((((-cartPos) * max(cartPos, max(max(3.355, abs((lambda x, y: x / y if y != 0 else 0)(0.235, cartVel))), cartVel))) - 0.245) + (lambda x, y: x / y if y != 0 else 0)(0.235, cartVel))
        return max(0, min(2, int(round(action))))


class MTC_simple21:

    def decide(self, input):
        cartPos, cartVel = input
        action = ((cartPos * (cartVel - max(5.0, abs((cartPos * (cartPos + abs(((-0.1) + (lambda x, y: x / y if y != 0 else 0)(0.235, cartVel))))))))) + (lambda x, y: x / y if y != 0 else 0)(0.235,
                                                                                                                                                                                               cartVel))
        return max(0, min(2, int(round(action))))


class MTC_simple28:

    def decide(self, input):
        cartPos, cartVel = input
        action = ((((-cartPos) * max(max(3.0, abs((cartPos * (cartPos + max(max(cartVel, abs(((-0.34) + (lambda x, y: x / y if y != 0 else 0)(0.235, cartVel)))), 6.785))))), cartVel)) + cartVel) + (
            lambda x, y: x / y if y != 0 else 0)(0.235, cartVel))
        return max(0, min(2, int(round(action))))


class MTC_simple29:

    def decide(self, input):
        cartPos, cartVel = input
        action = ((((-cartPos) * max(cartPos,
                                     max(abs(((-cartPos) * (cartPos + max(max(cartVel, abs(((-0.34) + (lambda x, y: x / y if y != 0 else 0)(0.235, cartVel)))), 7.085)))), cartVel))) + cartVel) + (
                      lambda x, y: x / y if y != 0 else 0)(0.235, cartVel))
        return max(0, min(2, int(round(action))))


class MTC_simple31:

    def decide(self, input):
        cartPos, cartVel = input
        action = ((((-cartPos) * max(max(0.31, ((-cartPos) + 2.755)),
                                     abs((cartPos * ((cartPos - cartVel) + max(cartVel, abs(((-0.34) + (lambda x, y: x / y if y != 0 else 0)(0.235, cartVel))))))))) - cartVel) + (
                      lambda x, y: x / y if y != 0 else 0)(0.235, cartVel))
        return max(0, min(2, int(round(action))))


class MTC_simple35:

    def decide(self, input):
        cartPos, cartVel = input
        action = ((((-cartPos) * max(0.34, max(max(3.07, abs(((-cartPos) * (cartPos + max(max(cartVel, abs(((-0.34) + (lambda x, y: x / y if y != 0 else 0)(0.235, cartVel)))), cartVel))))),
                                               abs(abs(cartVel))))) + (12.57 * cartVel)) + (lambda x, y: x / y if y != 0 else 0)(0.235, cartVel))
        return max(0, min(2, int(round(action))))


class MTC_simple36:

    def decide(self, input):
        cartPos, cartVel = input
        action = ((((-cartPos) * max(cartPos, max(5.0, abs((cartPos * (cartPos + max(max(max(cartVel, abs(((-0.34) + (lambda x, y: x / y if y != 0 else 0)(0.235, cartVel)))), 0.235),
                                                                                     (lambda x, y: x / y if y != 0 else 0)((1.975 * cartPos), min(cartPos, cartVel))))))))) + cartVel) + (
                      lambda x, y: x / y if y != 0 else 0)(0.235, cartVel))
        return max(0, min(2, int(round(action))))


class MTC_simple37:

    def decide(self, input):
        cartPos, cartVel = input
        action = ((((-cartPos) * max(cartPos, max(max(5.0, abs((cartPos * (cartPos + max(max(max(cartVel, abs((cartVel + (lambda x, y: x / y if y != 0 else 0)(0.235, cartVel)))), abs(cartPos)),
                                                                                         (lambda x, y: x / y if y != 0 else 0)((1.825 * cartPos), min(cartPos, cartVel))))))), cartVel))) + cartVel) + (
                      lambda x, y: x / y if y != 0 else 0)(0.235, cartVel))
        return max(0, min(2, int(round(action))))


class MTC_simple38:

    def decide(self, input):
        cartPos, cartVel = input
        action = ((((-cartPos) * max(cartPos, max(max(5.0, abs((cartPos * (cartPos + max(max(max(cartVel, abs(((-0.115) + (lambda x, y: x / y if y != 0 else 0)(0.235, cartVel)))), abs(cartPos)),
                                                                                         (lambda x, y: x / y if y != 0 else 0)((1.825 * cartPos), min(cartPos, cartVel))))))), cartVel))) + cartVel) + (
                      lambda x, y: x / y if y != 0 else 0)(0.235, cartVel))
        return max(0, min(2, int(round(action))))


class MTC_simple39:

    def decide(self, input):
        cartPos, cartVel = input
        action = ((((-cartPos) * max(cartPos, max(abs(((-cartPos) * (cartPos + max(
            max((((12.57 * cartVel) + 2.235) + (lambda x, y: x / y if y != 0 else 0)(0.9, min(cartPos, cartVel))), abs(((-0.34) + (lambda x, y: x / y if y != 0 else 0)(0.235, cartVel)))), 7.0)))),
                                                  cartVel))) + cartVel) + (lambda x, y: x / y if y != 0 else 0)(0.235, cartVel))
        return max(0, min(2, int(round(action))))


class MTC_simple41:

    def decide(self, input):
        cartPos, cartVel = input
        action = ((((-cartPos) * max(cartPos, max(max((abs(3.355) - 0.33), abs((((cartVel - max(max(5.0, (cartVel + (lambda x, y: x / y if y != 0 else 0)(0.9, min(cartPos, cartVel)))),
                                                                                                abs((lambda x, y: x / y if y != 0 else 0)(0.235, cartVel)))) - min(cartPos, cartVel)) * abs(cartPos)))),
                                                  cartVel))) - 0.005) + (lambda x, y: x / y if y != 0 else 0)(0.235, cartVel))
        return max(0, min(2, int(round(action))))


class MTC_simple45:

    def decide(self, input):
        cartPos, cartVel = input
        action = ((((-cartPos) * max(max((abs(3.355) - 0.33), abs((cartPos * ((cartVel - max(
            max(max(cartVel, abs((lambda x, y: x / y if y != 0 else 0)(0.235, cartVel))), (((12.57 * cartVel) + 2.33) + (lambda x, y: x / y if y != 0 else 0)(0.9, min(cartPos, cartVel)))),
            abs(0.82))) - min(cartPos, cartVel))))), cartVel)) + 0.015) + (lambda x, y: x / y if y != 0 else 0)(0.235, cartVel))
        return max(0, min(2, int(round(action))))


class MTC_simple48:

    def decide(self, input):
        cartPos, cartVel = input
        action = ((((-cartPos) * max(max((cartVel - 0.33), abs((cartPos * ((cartVel - max(
            max(max(cartVel, abs((lambda x, y: x / y if y != 0 else 0)(0.235, cartVel))), (((12.57 * cartVel) + 2.33) + (lambda x, y: x / y if y != 0 else 0)(0.9, min(cartPos, cartVel)))),
            abs(((11.59 * cartPos) + 0.585)))) - min(cartPos, cartVel))))), cartVel)) + 0.015) + (lambda x, y: x / y if y != 0 else 0)(0.235, cartVel))
        return max(0, min(2, int(round(action))))


class MTC_simple49:

    def decide(self, input):
        cartPos, cartVel = input
        action = ((((-cartPos) * max(max((abs(3.355) - 0.33), abs((cartPos * ((cartVel - max(
            max(max(cartVel, abs((lambda x, y: x / y if y != 0 else 0)(0.235, cartVel))), (((12.57 * cartVel) + 2.33) + (lambda x, y: x / y if y != 0 else 0)(0.9, min(cartPos, cartVel)))),
            abs(((11.59 * cartPos) + 0.585)))) - min(cartPos, cartVel))))), cartVel)) + 0.015) + (lambda x, y: x / y if y != 0 else 0)(0.235, cartVel))
        return max(0, min(2, int(round(action))))


class MTC_simple56:

    def decide(self, input):
        cartPos, cartVel = input
        action = ((((-cartPos) * max(max(0.31, ((-cartPos) + 2.755)), max(abs(((((-cartPos) + cartVel) - max(
            max(max(5.0, abs((cartPos * ((cartPos - cartVel) + max(cartVel, abs(((-0.34) + (lambda x, y: x / y if y != 0 else 0)(0.235, cartVel)))))))),
                ((cartVel + 2.33) + (lambda x, y: x / y if y != 0 else 0)(0.895, min(cartPos, cartVel)))), cartPos)) * abs(cartPos))), cartVel))) - 0.005) + (lambda x, y: x / y if y != 0 else 0)(
            0.235, cartVel))
        return max(0, min(2, int(round(action))))


class MTC_simple60:

    def decide(self, input):
        cartPos, cartVel = input
        action = ((((-cartPos) * max(max((abs(3.355) - 0.33), abs((((cartVel - max(
            max(max(5.0, abs((cartPos * ((cartPos - cartVel) + max(cartVel, abs(((-0.34) + (lambda x, y: x / y if y != 0 else 0)(0.235, cartVel)))))))),
                (cartPos + (lambda x, y: x / y if y != 0 else 0)(0.9, min(cartPos, cartVel)))), abs(((cartPos * (cartVel + 11.0)) + 0.235)))) - min(cartPos, cartVel)) * abs(cartPos)))),
                                     cartPos)) + 0.015) + (lambda x, y: x / y if y != 0 else 0)(0.235, cartVel))
        return max(0, min(2, int(round(action))))


class MTC_simple61:

    def decide(self, input):
        cartPos, cartVel = input
        action = ((((-cartPos) * max(max((abs(3.355) - 0.33), abs((cartPos * ((cartVel - max(
            max(max(5.0, abs((cartPos * ((cartPos - cartVel) + max(0.34, abs(((-0.34) + (lambda x, y: x / y if y != 0 else 0)(0.235, cartVel)))))))),
                (((12.57 * cartVel) + 2.33) + (lambda x, y: x / y if y != 0 else 0)(0.9, min(cartPos, cartVel)))), abs(((11.59 * cartPos) + 0.585)))) - min(cartPos, cartVel))))),
                                     cartVel)) + 0.015) + (lambda x, y: x / y if y != 0 else 0)(0.235, cartVel))
        return max(0, min(2, int(round(action))))


class MTC_simple65:

    def decide(self, input):
        cartPos, cartVel = input
        action = ((((-cartPos) * max(max(0.31, ((-cartPos) + 2.755)), max(max(cartVel, abs((((cartVel - max(
            max(max(cartVel, abs((cartPos * (cartPos + max(0.34, abs(((-0.34) + (lambda x, y: x / y if y != 0 else 0)(0.235, cartVel)))))))),
                (2.565 + (lambda x, y: x / y if y != 0 else 0)(0.9, min(cartPos, cartVel)))), abs(((cartPos * ((cartPos + abs(cartVel)) + 11.0)) + cartPos)))) - min(cartPos, cartVel)) * abs(
            cartPos)))), cartVel))) - cartVel) + (lambda x, y: x / y if y != 0 else 0)(0.235, cartVel))
        return max(0, min(2, int(round(action))))


class MTC_simple68:

    def decide(self, input):
        cartPos, cartVel = input
        action = ((((-cartPos) * max(max(0.31, ((-cartPos) + 2.755)), max(max((-cartPos), abs((((cartVel - max(
            max(max(5.0, abs((cartPos * (cartPos + abs(((-0.34) + (lambda x, y: x / y if y != 0 else 0)(0.235, cartVel))))))),
                (((2.0 * cartVel) + 2.33) + (lambda x, y: x / y if y != 0 else 0)(0.9, min(cartPos, cartVel)))), abs(((cartPos * ((cartPos - abs(cartVel)) - 11.0)) + cartPos)))) - min(cartPos,
                                                                                                                                                                                        cartVel)) * abs(
            cartPos)))), cartVel))) - cartVel) + (lambda x, y: x / y if y != 0 else 0)(0.235, cartVel))
        return max(0, min(2, int(round(action))))


all_agents = [MTC_simple0, MTC_simple7, MTC_simple10, MTC_simple15, MTC_simple18, MTC_simple21, MTC_simple28, MTC_simple29, MTC_simple31, MTC_simple35, MTC_simple36, MTC_simple37, MTC_simple38,
              MTC_simple39, MTC_simple41, MTC_simple45, MTC_simple48, MTC_simple49, MTC_simple56, MTC_simple60, MTC_simple61, MTC_simple65, MTC_simple68]

agent_tuples = [('MTC_simple0', MTC_simple0()), ('MTC_simple7', MTC_simple7()), ('MTC_simple10', MTC_simple10()), ('MTC_simple15', MTC_simple15()), ('MTC_simple18', MTC_simple18()),
                ('MTC_simple21', MTC_simple21()), ('MTC_simple28', MTC_simple28()), ('MTC_simple29', MTC_simple29()), ('MTC_simple31', MTC_simple31()), ('MTC_simple35', MTC_simple35()),
                ('MTC_simple36', MTC_simple36()), ('MTC_simple37', MTC_simple37()), ('MTC_simple38', MTC_simple38()), ('MTC_simple39', MTC_simple39()), ('MTC_simple41', MTC_simple41()),
                ('MTC_simple45', MTC_simple45()), ('MTC_simple48', MTC_simple48()), ('MTC_simple49', MTC_simple49()), ('MTC_simple56', MTC_simple56()), ('MTC_simple60', MTC_simple60()),
                ('MTC_simple61', MTC_simple61()), ('MTC_simple65', MTC_simple65()), ('MTC_simple68', MTC_simple68())]
