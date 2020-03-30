import math;

class dummyname_0:

    def decide(self, observation):
        cartPos, cartVel = observation[0], observation[1]
        return max(0, min(2, int(round(2 if ((cartVel<1) | ((cartPos<(-0.05)) & (cartVel<0.1))) else 0 if (((cartVel>(-0.45)) & (cartPos<0.02)) & (cartVel<(-0.05))) else 0 if (cartPos<0) else 2))))

class dummyname_25:

    def decide(self, observation):
        cartPos, cartVel = observation[0], observation[1]
        return max(0, min(2, int(round((((cartVel+0.81)/max(cartPos, cartVel))+(-0.57) if (((cartVel<=(-0.05)) & (cartVel>(-0.45))) & (cartPos<0.02)) else cartVel)))))

class dummyname_27:

    def decide(self, observation):
        cartPos, cartVel = observation[0], observation[1]
        return max(0, min(2, int(round((((3.335*cartPos)+max(cartVel, (cartPos/cartVel)))+((cartPos+(0.485*cartVel))/(cartPos*cartVel)))))))

class dummyname_28:

    def decide(self, observation):
        cartPos, cartVel = observation[0], observation[1]
        return max(0, min(2, int(round((((3.335*cartPos)+max(4.225, (cartPos/cartVel)))+((cartPos+(0.485*cartVel))/(cartPos*cartVel)))))))

class dummyname_29:

    def decide(self, observation):
        cartPos, cartVel = observation[0], observation[1]
        return max(0, min(2, int(round(((((2.335*cartPos)-cartVel)+max(4.225, (cartPos/cartVel)))+((cartPos+(0.485*cartVel))/(cartPos*cartVel)))))))

class dummyname_30:

    def decide(self, observation):
        cartPos, cartVel = observation[0], observation[1]
        return max(0, min(2, int(round((((((3.335*cartPos)-cartVel)+max(4.225, (cartPos/cartVel)))+0.485)+((cartPos+(0.485*cartVel))/(cartPos*cartVel)))))))

class dummyname_31:

    def decide(self, observation):
        cartPos, cartVel = observation[0], observation[1]
        return max(0, min(2, int(round(((((2.87*cartPos)-cartVel)+max(0.05, (cartPos/cartVel)))+(((cartVel*(-1.0))*(cartPos+(0.485*cartVel)))/cartPos))))))

class dummyname_35:

    def decide(self, observation):
        cartPos, cartVel = observation[0], observation[1]
        return max(0, min(2, int(round((((((3.26*cartPos)-cartVel)+(0.8*math.sin((0.235/cartVel))))+max(4.225, (cartPos/cartVel)))+((cartPos+(0.485*cartVel))/(cartPos*cartVel)))))))

class dummyname_37:

    def decide(self, observation):
        cartPos, cartVel = observation[0], observation[1]
        return max(0, min(2, int(round((max(0.93, cartPos)+(((cartPos+(0.485*min(0.28, cartVel)))*(max(max(cartVel, (((-cartPos)-(cartVel*2))-1.05)), cartPos)*(-1.0)))/cartPos))))))

class dummyname_39:

    def decide(self, observation):
        cartPos, cartVel = observation[0], observation[1]
        return max(0, min(2, int(round(((((2.87*cartPos)-cartVel)+max(0.05, (cartPos/cartVel)))+(((cartPos+(0.485*cartVel))*(max((cartVel+(0.93/(((32.335*cartPos)+cartVel)-0.555))), cartVel)*(-1.0)))/cartPos))))))

class dummyname_42:

    def decide(self, observation):
        cartPos, cartVel = observation[0], observation[1]
        return max(0, min(2, int(round(((cartPos*((-1.61)+(cartVel/cartPos)))+(((cartPos+(0.485*cartVel))*(max(((0.485*cartVel)+max(cartVel, (-0.105 if (cartVel<((cartPos+cartVel)+1.045)) else cartVel))), cartPos)*(-1.0)))/cartPos))))))

class dummyname_44:

    def decide(self, observation):
        cartPos, cartVel = observation[0], observation[1]
        return max(0, min(2, int(round(((cartPos*((-1.61)+(cartVel/cartPos)))+(((cartPos+(0.485*cartVel))*(max(((0.485*cartVel)+max(cartVel, (-0.105 if (cartVel<((cartPos+(0.555*cartVel))+1.045)) else cartVel))), cartPos)*(-1.0)))/cartPos))))))

class dummyname_45:

    def decide(self, observation):
        cartPos, cartVel = observation[0], observation[1]
        return max(0, min(2, int(round(((cartPos*((-1.865)+((0.435*cartVel)/cartPos)))+(((cartPos+(0.485*cartVel))*(max(((0.485*cartVel)+max(cartVel, (-0.215 if (cartVel<((cartPos+(0.555*cartVel))+1.045)) else cartVel))), cartPos)*(-1.0)))/cartPos))))))

class dummyname_51:

    def decide(self, observation):
        cartPos, cartVel = observation[0], observation[1]
        return max(0, min(2, int(round(((cartPos*(cartPos-1.865))+(((cartPos+(0.485*cartVel))*(max((cartVel+max(cartVel, (-(0.555-(0.43/(((32.335*cartPos)+cartVel)-0.675))) if (cartVel<=((cartPos+(0.555*cartVel))+1.045)) else cartVel))), cartPos)*(-1.0)))/cartPos))))))

class dummyname_53:

    def decide(self, observation):
        cartPos, cartVel = observation[0], observation[1]
        return max(0, min(2, int(round(((cartPos*((-1.865)+(cartVel/cartPos)))+(((cartPos+(0.485*cartVel))*(max(((0.655*cartVel)+max(cartVel, (-(0.78-(0.43/(((32.335*cartPos)+cartVel)-0.675))) if (cartVel<=((cartPos+(0.555*cartVel))+1.045)) else cartVel))), cartPos)*(-1.0)))/cartPos))))))

class dummyname_54:

    def decide(self, observation):
        cartPos, cartVel = observation[0], observation[1]
        return max(0, min(2, int(round((0.965+(((cartPos+(0.485*cartVel))*(max(((0.8*cartVel)+max(cartVel, (-(cartVel-(0.9*(((((0.9*cartPos)+cartVel)+max(cartVel, (0.465*max(0.555, cartVel))))-0.675)*(-cartPos)))) if (cartVel<((cartPos+cartVel)+1.05)) else cartVel))), cartPos)*(-1.0)))/cartPos))))))

class dummyname_55:

    def decide(self, observation):
        cartPos, cartVel = observation[0], observation[1]
        return max(0, min(2, int(round((0.965+(((cartPos+(0.485*cartVel))*(max(((0.8*cartVel)+max(cartVel, (-(cartVel-(0.9*(((((0.9*cartPos)+cartVel)+max(cartVel, (0.465*max(0.555, cartVel))))-0.675)*(-1.0)))) if (cartVel<((cartPos+cartVel)+1.05)) else cartVel))), cartPos)*(-1.0)))/cartPos))))))

class dummyname_62:

    def decide(self, observation):
        cartPos, cartVel = observation[0], observation[1]
        return max(0, min(2, int(round((max(0.93, cartPos)+(((cartPos+(0.485*min(0.28, cartVel)))*(max(max(cartVel, (-(0.1048071688-(0.43/((((0.9*cartPos)+cartVel)+max(((((-3.5087719298)*cartPos)/cartVel)-0.79560162), ((0.1066945606*cartPos)+0.1987447698)))-0.675))) if ((0.485*cartVel)<=((cartPos+(0.59*cartVel))+1.05)) else cartVel)), cartPos)*(-1.0)))/cartPos))))))



all_agents = [dummyname_0(), dummyname_25(), dummyname_27(), dummyname_28(), dummyname_29(), dummyname_30(), dummyname_31(), dummyname_35(), dummyname_37(), dummyname_39(), dummyname_42(), dummyname_44(), dummyname_45(), dummyname_51(), dummyname_53(), dummyname_54(), dummyname_55(), dummyname_62()]
tuples = [tuple(('a' + str(ii), agnt)) for ii, agnt in enumerate(all_agents)]