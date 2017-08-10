import abc
from core.util.interfaces import InterfaceMetaclass

class ScopeInterface(metaclass=InterfaceMetaclass):
    _modtype = 'ScopeInterface'
    _modclass = 'interface'

    @abc.abstractmethod
    def run_continuous(self):
        pass

    @abc.abstractmethod
    def stop_acquisition(self):
        pass