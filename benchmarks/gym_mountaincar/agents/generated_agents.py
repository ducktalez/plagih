import math


class dummyname_0:

    def decide(self, observation):
        cartPos, cartVel = observation[0], observation[1]
        return 0 if (cartVel < 0) else 2


class dummyname_6:

    def decide(self, observation):
        cartPos, cartVel = observation[0], observation[1]
        return 0 if (cartVel < (cartPos * abs(cartVel))) else 2


class dummyname_12:

    def decide(self, observation):
        cartPos, cartVel = observation[0], observation[1]
        return 0 if (((-0.3) <= (cartPos + 0.725)) & ((cartPos + cartVel) < cartPos)) else 2


class dummyname_20:

    def decide(self, observation):
        cartPos, cartVel = observation[0], observation[1]
        return 0 if (((cartVel + 0.265) / cartVel) < (((((-0.95) * cartVel) - 0.25175) * max(cartPos, (0.1 / cartVel))) / cartVel)) else 2


class dummyname_22:

    def decide(self, observation):
        cartPos, cartVel = observation[0], observation[1]
        return 0 if (((cartVel + 0.265) / cartVel) < (((((-0.89775) * cartVel) - 0.25175) * max(cartPos, (cartPos + (0.1 / cartVel)))) / cartVel)) else 2


class dummyname_52:

    def decide(self, observation):
        cartPos, cartVel = observation[0], observation[1]
        return 0 if (((cartVel + 0.265) / cartVel) < (((((-0.8645) * cartVel) - 0.25175) * max(cartPos, max(
            (((((cartPos * cartVel) * ((-abs((cartVel * (cartPos + 0.395)))) - 1.615)) * abs(cartVel)) / (cartVel + math.sin((math.sin(((cartVel + 0.265) / cartVel)) / abs(0.91))))) + cartPos),
            (0.1 / cartVel)))) / cartVel)) else 2


all_agents = [dummyname_0, dummyname_6, dummyname_12, dummyname_20, dummyname_22, dummyname_52]
agnt_dict = {0: ('dummy1', dummyname_0()),
             6: ('dummy2', dummyname_6()),
             12: ('dummy3', dummyname_12()),
             20: ('dummy4', dummyname_20()),
             22: ('dummy5', dummyname_22()),
             52: ('dummy6', dummyname_52())
             }
