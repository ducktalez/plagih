from abc import ABC, abstractmethod


class NodeCreator(ABC):

    @abstractmethod
    def choose_operator(self, xt):
        pass

    @abstractmethod
    def choose_operator_match(self, xtype):
        pass

    @abstractmethod
    def choose_terminal(self, xt):
        pass

    @abstractmethod
    def choose_constant(self, xt):
        pass

    @abstractmethod
    def choose_symbol(self, xt):
        pass
