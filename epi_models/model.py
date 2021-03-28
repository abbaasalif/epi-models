from abc import ABC, abstractmethod
from enum import Enum

from .config.epidemic_params import covid_specific_parameters
from .params import CampParams


class Model(ABC):
    def __init__(self):
        pass

    @abstractmethod
    def id(self):
        """This serves to give the model a unique ID to distinguish it from other models"""

    def load_epidemic_parameters(self):
        # TODO: put the paraemters from compartmental models here can then later refactor out some when other models are brought in
        # This is read in via config/epidemic_parameters.py and modified there as research is updated
        for k, v in covid_specific_parameters.items():
            setattr(self, k, v)

    @abstractmethod
    def process_epidemic_parameters(self):
        """After reading in the epidemic parameters which are shared by all models some models require some further transformations before they can be plugged in to the model"""

    # def load_camp_parameters(self, camp_params: CampParams):
    #     # TODO: put the paraemters from compartmental models here can then later refactor out some when other models are brought in
    #     # here we need to have a few methods for reading the parameters in csv/json/cache/db this might need its own class - having a seperate class might be good so it only needs accessd once through DB etc
    #     # load all attributes from CampParams into the model
    #     for k,v in camp_params.__dict__.items():
    #         setattr(self, k, v)

    @abstractmethod
    def process_and_load_camp_parameters(self, camp_params: CampParams):
        """parse the camp parameters according to model needs and load them into the model object"""

    @abstractmethod
    def load_model_parameters(self):
        """Internal Parameters specific to the model"""

    @abstractmethod
    def run_single_simulation(self):
        pass

    @abstractmethod
    def run_multiple_simulations(self):
        pass


class ModelId(Enum):
    DeterministicCompartmentalModel = 0
    StochasticCompartmentalModel = 1
    AgentBasedModel = 2
    NetworkModel = 3


class ModelRunner(ABC):
    pass
