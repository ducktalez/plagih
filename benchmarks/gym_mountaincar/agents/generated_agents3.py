import math;


class dummyname_0:

    def decide(self, observation):
        cartPos, cartVel = observation[0], observation[1]
        return 0 if (cartVel < 0) else 2


class dummyname_10:

    def decide(self, observation):
        cartPos, cartVel = observation[0], observation[1]
        return (2.83 + ((math.sin((cartPos + 0.54)) + 0.495) / cartVel))


class dummyname_13:

    def decide(self, observation):
        cartPos, cartVel = observation[0], observation[1]
        return (2.0 + ((math.sin((min(cartPos, cartVel) + 0.54)) + max(cartPos, 0.495)) / cartVel))


class dummyname_15:

    def decide(self, observation):
        cartPos, cartVel = observation[0], observation[1]
        return (max(cartPos, (2.83 * (cartVel * cartPos))) + ((math.sin((cartPos + 0.54)) + 0.495) / cartVel))


class dummyname_18:

    def decide(self, observation):
        cartPos, cartVel = observation[0], observation[1]
        return (max(0.795, ((cartVel * (-1.0)) + 2.83)) + ((math.sin((cartPos + 0.54)) + max(cartPos, 0.495)) / cartVel))


class dummyname_19:

    def decide(self, observation):
        cartPos, cartVel = observation[0], observation[1]
        return (abs((max(0.795, (2.83 * (cartVel * (-1.0)))) + 0.965)) + ((math.sin((cartPos + 0.54)) + 0.495) / cartVel))


class dummyname_21:

    def decide(self, observation):
        cartPos, cartVel = observation[0], observation[1]
        return (abs((max(max(cartPos, 0.495), (2.83 * (cartVel * (-1.0)))) + 0.965)) + ((math.sin((cartPos + 0.54)) + 0.495) / cartVel))


class dummyname_23:

    def decide(self, observation):
        cartPos, cartVel = observation[0], observation[1]
        return (abs((cartVel + max(0.795, ((cartVel * (-1.0)) + 2.83)))) + ((math.sin((min(cartPos, 0.2327257667) + 0.53)) + max(cartPos, 0.495)) / cartVel))


class dummyname_26:

    def decide(self, observation):
        cartPos, cartVel = observation[0], observation[1]
        return (abs((max(0.795, ((cartVel * (-1.0)) + (0.72 / cartPos))) - 0.485)) + ((math.sin((min(cartPos, (-1.0)) + 0.53)) + max(cartPos, 0.495)) / cartVel))


class dummyname_33:

    def decide(self, observation):
        cartPos, cartVel = observation[0], observation[1]
        return (abs((0.455 + max(0.795, (2.83 * (max(0.54 if (((cartVel + math.sin(abs((-0.02)))) * (-1.0)) < cartPos) else cartVel, cartPos) * cartPos))))) + (
                    (math.sin((cartPos + 0.54)) + max(cartPos, 0.495)) / cartVel))


class dummyname_36:

    def decide(self, observation):
        cartPos, cartVel = observation[0], observation[1]
        return (abs(
            (0.275 + max(0.795, (2.83 * (max(0.54 if (((cartVel + math.sin(math.sin(abs((-0.02))))) * (-1.0)) < ((cartPos / cartVel) + 0.71)) else cartVel, cartPos) * cartPos))))) + (
                                                    (math.sin((cartPos + 0.54)) + 0.495) / cartVel))


class dummyname_38:

    def decide(self, observation):
        cartPos, cartVel = observation[0], observation[1]
        return (abs(
            (cartVel + max(0.795, (2.83 * (max(0.54 if (((cartVel + math.sin(abs(0.02))) * (-1.0)) < (((1.11 * cartPos) / cartVel) + 0.71)) else cartVel, cartPos) * cartPos))))) + (
                                                    (math.sin((cartPos + 0.54)) + max(cartPos, 0.495)) / cartVel))


class dummyname_48:

    def decide(self, observation):
        cartPos, cartVel = observation[0], observation[1]
        return (abs((0.54 + max(cartVel, (2.83 * (max(0.51 if ((max((-0.27), (cartVel + abs(abs(0.02)))) * (-1.0)) < (cartPos + 0.71)) else cartVel,
                                                                              ((math.sin((min(cartPos, ((49.715 * cartVel) + 0.185)) + 0.54)) + 0.49) / cartVel)) * cartPos))))) + (
                                                    (math.sin((cartPos + 0.54)) + 0.495) / cartVel))


class dummyname_55:

    def decide(self, observation):
        cartPos, cartVel = observation[0], observation[1]
        return (abs((0.54 + max(cartVel, (2.83 * (max(0.51 if ((max((-0.27), (cartVel + abs(abs(0.02)))) * (-1.0)) < (((cartPos * 3.0) / cartVel) + 0.71)) else cartVel,
                                                                              ((math.sin((min(cartPos, ((50.005 * cartVel) + 0.185)) + 0.54)) + 0.495) / cartVel)) * cartPos))))) + (
                                                    (math.sin((min(cartPos, abs(cartPos)) + 0.54)) + 0.495) / cartVel))


class dummyname_65:

    def decide(self, observation):
        cartPos, cartVel = observation[0], observation[1]
        return (abs((((-cartPos) + math.sin(((1.11 * cartPos) / cartVel))) + max(cartPos, (2.83 * (
                    max(0.51 if ((max(cartVel, (cartVel + abs(abs((-0.02))))) * (-1.0)) < (((1.11 * cartPos) / cartVel) + cartVel)) else cartVel,
                        ((math.sin((min(cartPos, ((cartVel / math.sin(abs((-0.02)))) + 0.185)) + 0.54)) + max(cartPos, 0.495)) / cartVel)) * cartPos))))) + (
                                                    (math.sin((cartPos + 0.54)) + 0.495) / cartVel))


all_agents = [
    ('dummyname_0', dummyname_0()),('dummyname_10', dummyname_10()),('dummyname_13', dummyname_13()),('dummyname_15', dummyname_15()),('dummyname_18', dummyname_18()),('dummyname_19', dummyname_19()),(
        'dummyname_21', dummyname_21()),('dummyname_23', dummyname_23()),('dummyname_26', dummyname_26()),('dummyname_33', dummyname_33()),('dummyname_36', dummyname_36()),('dummyname_38', dummyname_38()),(
        'dummyname_48', dummyname_48()),('dummyname_55', dummyname_55()),('dummyname_65', dummyname_65())]

tuples3 = [tuple(('b' + str(ii), agnt)) for ii, agnt in enumerate(all_agents)]
