import numpy as np

class DeterministicCompartmentalModelScenario(object):
    def __init__(self, transmission_reduction_factor=1, remove_symptomatic_rate=0, remove_high_risk_rate=0, first_high_risk_category_n=2, icu_capacity=0):
        # baseline parameters that need to be run (Do nothing scenario)
        # self.transmission_reduction_factor = transmission_reduction_factor
        # self.remove_symptomatic_rate = remove_symptomatic_rate
        # self.first_high_risk_category_n = first_high_risk_category_n
        # self.remove_high_risk_rate = remove_high_risk_rate
        # self.ICU_capacity = icu_capacity
        param_dict = dict()
        param_dict['transmission_reduction_factor'] = transmission_reduction_factor
        param_dict['remove_symptomatic_rate'] = remove_symptomatic_rate
        param_dict['first_high_risk_category_n'] = first_high_risk_category_n
        param_dict['remove_high_risk_rate'] = remove_high_risk_rate
        param_dict['icu_capacity'] = icu_capacity
        self.baseline_param_dict = param_dict

    @staticmethod
    def _validate_input_params(transmission_reduction_factor, remove_symptomatic_rate, remove_high_risk_rate, first_high_risk_category_n, icu_capacity, infection_matrix):
        assert infection_matrix.shape == (8, 8)
        assert transmission_reduction_factor >= 0
        assert transmission_reduction_factor <= 1
        assert remove_symptomatic_rate >=0
        assert remove_high_risk_rate >= 0
        assert first_high_risk_category_n >= 0
        assert first_high_risk_category_n <= 8
        assert icu_capacity >= 0

    @staticmethod
    def apply_shielding(infection_matrix, shiled_increase=2, n_oldest_group=1):
        divider = -n_oldest_group  # determines which groups separated. -1 means only oldest group separated from the rest

        infection_matrix[:divider, :divider] = shiled_increase * infection_matrix[:divider, :divider]
        infection_matrix[:divider, divider:] = shiled_increase * infection_matrix[:divider, divider:]
        infection_matrix[divider:, :divider] = shiled_increase * infection_matrix[divider:, :divider]
        infection_matrix[divider:, divider] = shiled_increase * infection_matrix[divider:, divider:]

        return infection_matrix


class SingleInterventionScenario(DeterministicCompartmentalModelScenario):
    # TODO: give two intensity setting mode - one is constant where values stay the same throughout the duration or the other one is a linear decay or inverse exponential decay where the efficacy of the intervention decreases throughout the intervention cycles
    def __init__(self, start_times, end_times, infection_matrix, apply_shielding=False, transmission_reduction_factor_inter=1, remove_symptomatic_rate_inter = 0, remove_high_risk_rate_inter=0, first_high_risk_category_n_inter=2, icu_capacity_inter=0, inter_rate_change='Constant'):
        DeterministicCompartmentalModelScenario.__init__(self)
        self._validate_input_params(transmission_reduction_factor_inter, remove_symptomatic_rate_inter, remove_high_risk_rate_inter, first_high_risk_category_n_inter, icu_capacity_inter, infection_matrix)
        param_dict = dict()
        param_dict['transmission_reduction_factor'] = transmission_reduction_factor_inter
        param_dict['remove_symptomatic_rate'] = remove_symptomatic_rate_inter
        param_dict['first_high_risk_category_n'] = first_high_risk_category_n_inter
        param_dict['remove_high_risk_rate'] = remove_high_risk_rate_inter
        param_dict['icu_capacity'] = icu_capacity_inter
        self.intervention_param_dict = param_dict
        self._validate_input_time(start_times, end_times)
        self.start_times = start_times
        self.end_times = end_times
        self.baseline_param_dict['infection_matrix'] = infection_matrix
        if apply_shielding:
            infection_matrix =self.apply_shielding(infection_matrix)
        else:
            infection_matrix = infection_matrix
        self.intervention_param_dict['infection_matrix'] = infection_matrix
        self.inter_rate_change = inter_rate_change

    @staticmethod
    def _validate_input_time(start_times, end_times):
        assert len(start_times) >= 1
        assert len(start_times) == len(
            end_times), "intervention start times and duration given for each start time don't match"
        assert start_times[0] >= 0
        for start, end in zip(start_times, end_times):
            assert (end >= start)
        assert all(isinstance(t, int) for t in start_times)
        assert all(isinstance(t, int) for t in end_times)

    def intervention_params_at_time_t(self, t: int):
        for start, end in zip(self.start_times, self.end_times):
            if t in range(start,end):
                if self.inter_rate_change == 'Constant':
                    return self.intervention_param_dict
                elif self.inter_rate_change == 'Decay':
                    # first check which components are different to the baseline dict
                    intervention_dict = {k: self.intervention_param_dict[k] for k in self.baseline_param_dict if self.intervention_param_dict[k] != self.baseline_param_dict[k]}
                    baseline_dict_updated = {k: self.intervention_param_dict[k] for k in self.baseline_param_dict if self.intervention_param_dict[k] == self.baseline_param_dict[k]}
                    assert len(intervention_dict) == 1, "SingleInterventionScenario should have only one intervention only"
                    # use a linear decay formulae to update the rate
                    initial_rate = intervention_dict.values()[0]
                    current_rate = 0.3 * initial_rate / end * (start - t) + initial_rate
                    baseline_dict_updated[intervention_dict.keys()[0]] = current_rate
                    baseline_dict_updated["infection_matrix"] = self.intervention_param_dict['infection_matrix']
                    return baseline_dict_updated
            else:
                return self.baseline_param_dict


class MultipleInterventionScenario(DeterministicCompartmentalModelScenario):
    pass