import math


class MTC_simple0:

    def decide(self, input):
        cartPos, cartVel = input
        action = 0 if (cartVel < 0) else 2
        return max(0, min(2, int(round(action))))


class MTC_simple7:

    def decide(self, input):
        cartPos, cartVel = input
        action = (((-3.425) * cartPos) + (0.11 / cartVel))
        return max(0, min(2, int(round(action))))


class MTC_simple10:

    def decide(self, input):
        cartPos, cartVel = input
        action = (min(max((5.0 / cartVel), (4.0 / cartPos)), 1.265) + 5.0)
        return max(0, min(2, int(round(action))))


class MTC_simple15:

    def decide(self, input):
        cartPos, cartVel = input
        action = ((((-cartPos) * abs(((-0.34) + (0.235 / cartVel)))) - 0.005) + (0.235 / cartVel))
        return max(0, min(2, int(round(action))))


class MTC_simple18:

    def decide(self, input):
        cartPos, cartVel = input
        action = ((((-cartPos) * max(cartPos, max(max(3.355, abs((0.235 / cartVel))), cartVel))) - 0.245) + (0.235 / cartVel))
        return max(0, min(2, int(round(action))))


class MTC_simple21:

    def decide(self, input):
        cartPos, cartVel = input
        action = ((cartPos * (cartVel - max(5.0, abs((cartPos * (cartPos + abs(((-0.1) + (0.235 / cartVel))))))))) + (0.235 / cartVel))
        return max(0, min(2, int(round(action))))


class MTC_simple28:

    def decide(self, input):
        cartPos, cartVel = input
        action = ((((-cartPos) * max(max(3.0, abs((cartPos * (cartPos + max(max(cartVel, abs(((-0.34) + (0.235 / cartVel)))), 6.785))))), cartVel)) + cartVel) + (0.235 / cartVel))
        return max(0, min(2, int(round(action))))


class MTC_simple31:

    def decide(self, input):
        cartPos, cartVel = input
        action = ((((-cartPos) * max(max(0.31, ((-cartPos) + 2.755)), abs((cartPos * ((cartPos - cartVel) + max(cartVel, abs(((-0.34) + (0.235 / cartVel))))))))) - cartVel) + (0.235 / cartVel))
        return max(0, min(2, int(round(action))))


class MTC_simple35:

    def decide(self, input):
        cartPos, cartVel = input
        action = ((((-cartPos) * max(0.34, max(max(3.07, abs(((-cartPos) * (cartPos + max(max(cartVel, abs(((-0.34) + (0.235 / cartVel)))), cartVel))))), abs(abs(cartVel))))) + (12.57 * cartVel)) + (
                    0.235 / cartVel))
        return max(0, min(2, int(round(action))))


class MTC_simple36:

    def decide(self, input):
        cartPos, cartVel = input
        action = ((((-cartPos) * max(cartPos, max(5.0, abs(
            (cartPos * (cartPos + max(max(max(cartVel, abs(((-0.34) + (0.235 / cartVel)))), 0.235), ((1.975 * cartPos) / min(cartPos, cartVel))))))))) + cartVel) + (0.235 / cartVel))
        return max(0, min(2, int(round(action))))


class MTC_simple38:

    def decide(self, input):
        cartPos, cartVel = input
        action = ((((-cartPos) * max(cartPos,
                                     max(max(5.0, abs((cartPos * (cartPos + max(max(max(cartVel, abs(((-0.115) + (0.235 / cartVel)))), abs(cartPos)), ((1.825 * cartPos) / min(cartPos, cartVel))))))),
                                         cartVel))) + cartVel) + (0.235 / cartVel))
        return max(0, min(2, int(round(action))))


class MTC_simple46:

    def decide(self, input):
        cartPos, cartVel = input
        action = ((((-cartPos) * max(cartPos, max(max(5.0, abs(
            (cartPos * (cartPos + max(max(5.0, abs((cartPos * ((cartPos - cartVel) + max(cartVel, abs(((-0.34) + (0.235 / cartVel)))))))), ((1.825 * cartPos) / min(cartPos, cartVel))))))), 1.87))) + (
                               1.95 * cartVel)) + (0.235 / cartVel))
        return max(0, min(2, int(round(action))))


class MTC_simple48:

    def decide(self, input):
        cartPos, cartVel = input
        action = ((((-cartPos) * max(min(cartPos, (-2.415)), max(
            max((abs(3.355) - 0.33), abs((((cartVel - max(max(5.0, (((12.57 * cartVel) + 2.33) + (0.9 / min(cartPos, cartVel)))), abs((0.235 / cartVel)))) - min(cartPos, cartVel)) * abs(cartPos)))),
            cartVel))) - 0.005) + (0.235 / cartVel))
        return max(0, min(2, int(round(action))))


class MTC_simple53:

    def decide(self, input):
        cartPos, cartVel = input
        action = ((((-cartPos) * max(min(cartPos, (-2.415)), max(max((abs(3.355) - 0.33), abs(
            (((cartVel - max(max(max(5.0, abs((cartPos * cartVel))), (((12.57 * cartVel) + 2.28) + (0.9 / min(cartPos, cartVel)))), abs((0.235 / cartVel)))) - min(cartPos, cartVel)) * abs(cartPos)))),
                                                                 cartVel))) - 0.005) + (0.235 / cartVel))
        return max(0, min(2, int(round(action))))


class MTC_simple55:

    def decide(self, input):
        cartPos, cartVel = input
        action = ((((-cartPos) * max((cartPos * 2.0), max(max((abs(3.355) - 0.33), abs((((cartVel - max(
            max(max(5.0, abs((cartPos * ((cartPos - cartVel) + max(cartVel, abs(((-0.34) + (0.235 / cartVel)))))))), (2.135 + (0.9 / cartVel))), cartVel)) - min(cartPos, cartVel)) * abs(cartPos)))),
                                                          cartVel))) - 0.005) + (0.235 / cartVel))
        return max(0, min(2, int(round(action))))


class MTC_simple56:

    def decide(self, input):
        cartPos, cartVel = input
        action = ((((-cartPos) * max(max(0.31, ((-cartPos) + 2.755)), max(abs(((((-cartPos) + cartVel) - max(
            max(max(5.0, abs((cartPos * ((cartPos - cartVel) + max(cartVel, abs(((-0.34) + (0.235 / cartVel)))))))), ((cartVel + 2.33) + (0.895 / min(cartPos, cartVel)))), cartPos)) * abs(cartPos))),
                                                                          cartVel))) - 0.005) + (0.235 / cartVel))
        return max(0, min(2, int(round(action))))


class MTC_simple64:

    def decide(self, input):
        cartPos, cartVel = input
        action = ((((-cartPos) * max(max((abs(3.355) - 0.33), abs((((cartVel - max(
            max(max(5.0, abs((cartPos * ((cartPos - cartVel) + max(cartVel, abs(((-0.34) + (0.235 / cartVel)))))))), (((12.57 * cartVel) + 2.33) + (0.9 / min(cartPos, cartVel)))),
            abs(((cartPos * (cartVel + 11.0)) + 0.235)))) - min(cartPos, cartVel)) * abs(cartPos)))), cartVel)) + 0.015) + (0.235 / cartVel))
        return max(0, min(2, int(round(action))))


class MTC_simple66:

    def decide(self, input):
        cartPos, cartVel = input
        action = ((((-cartPos) * max(max((abs(3.355) - 0.33), abs((((cartVel - max(
            max(max(5.0, abs((cartPos * ((cartPos - cartVel) + max(cartVel, abs(((-0.34) + (0.235 / cartVel)))))))), (((12.57 * cartVel) + 2.33) + (0.9 / min(cartPos, cartVel)))),
            abs(((11.59 * cartPos) + 0.585)))) - min((0.9 / min(cartPos, cartVel)), cartVel)) * abs(cartPos)))), cartVel)) + 0.015) + (0.235 / cartVel))
        return max(0, min(2, int(round(action))))


class MTC_simple69:

    def decide(self, input):
        cartPos, cartVel = input
        action = ((((-cartPos) * max(max(0.31, ((-cartPos) + 2.755)), max(max((-cartPos), abs((((cartVel - max(
            max(max(5.0, abs((cartPos * ((cartPos - cartVel) + max(cartVel, abs(((-0.34) + (0.235 / cartVel)))))))), (abs(0.11) + (0.9 / min(cartPos, cartVel)))),
            abs(((cartPos * ((cartPos + abs(cartVel)) + 11.0)) + cartPos)))) - min(cartPos, cartVel)) * abs(cartPos)))), cartVel))) - 0.005) + (0.235 / cartVel))
        return max(0, min(2, int(round(action))))


all_agents = [MTC_simple0, MTC_simple7, MTC_simple10, MTC_simple15, MTC_simple18, MTC_simple21, MTC_simple28, MTC_simple31, MTC_simple35, MTC_simple36, MTC_simple38, MTC_simple46, MTC_simple48,
              MTC_simple53, MTC_simple55, MTC_simple56, MTC_simple64, MTC_simple66, MTC_simple69]

agent_tuples = [('MTC_simple0', MTC_simple0()), ('MTC_simple7', MTC_simple7()), ('MTC_simple10', MTC_simple10()), ('MTC_simple15', MTC_simple15()), ('MTC_simple18', MTC_simple18()),
                ('MTC_simple21', MTC_simple21()), ('MTC_simple28', MTC_simple28()), ('MTC_simple31', MTC_simple31()), ('MTC_simple35', MTC_simple35()), ('MTC_simple36', MTC_simple36()),
                ('MTC_simple38', MTC_simple38()), ('MTC_simple46', MTC_simple46()), ('MTC_simple48', MTC_simple48()), ('MTC_simple53', MTC_simple53()), ('MTC_simple55', MTC_simple55()),
                ('MTC_simple56', MTC_simple56()), ('MTC_simple64', MTC_simple64()), ('MTC_simple66', MTC_simple66()), ('MTC_simple69', MTC_simple69())]
