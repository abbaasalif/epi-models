from model import Model,ModelId
import numpy as np
from config.compartmental_model import model_config_cm, CONTACT_MATRIX_DIR, compartment_index
import pandas as pd
from scipy.integrate import ode

class DeterministicCompartmentalModel(Model):

    def __init__(self, camp_params):
        super().__init__()
        # load parameters
        self.load_epidemic_parameters()
        self.load_model_parameters()
        self.load_camp_parameters(camp_params)
        # process parameters
        self.population_vector = self._compute_population_vector()
        self.process_epidemic_parameters()

    def id(self):
        return ModelId.DeterministicCompartmentalModel

    def process_epidemic_parameters(self):
        # simulate a range of R0 values in the compartmental model
        self.R_0_list = np.asarray([self.R0_low, self.R0_medium,
                                    self.R0_high])
        # rate of E -> I
        self.latent_rate = 1 / (np.float(self.Latent_period))
        # rate of I -> R
        self.removal_rate = 1 / (np.float(self.Infectious_period))
        # rate of I -> H
        self.hosp_rate = 1 / (np.float(self.Hosp_period))
        # rate of C -> D
        self.death_rate = 1 / (np.float(self.Death_period))
        # rate of C -> D with ICU
        self.death_rate_with_ICU = 1 / (np.float(self.Death_period_withICU))
        # probability of death with ICU
        self.death_prob_with_ICU = np.float(self.Death_prob_withICU)
        # infectiousness of people who are asymptomatic
        self.AsymptInfectiousFactor = np.float(self.Infectiousness_asymptomatic)
        # R_0 mu/N, N=1
        self.beta_list = [R_0 * self.removal_rate for R_0 in self.R_0_list]
        # infection matrix from contact matrix
        self.infection_matrix, self.im_beta_list, self.largest_eigenvalue = self._generate_infection_matrix()
        # transition probabilities between disease compartments are pre-loaded already

    def _generate_contact_matrix(self):
        """Squeeze 5-year gap, 16 age compartment POLYMOD contact matrix into 10-year gap, 8 age compartment used in this model"""
        # TODO Walk through this code and write some tests for it
        contact_matrix_path = CONTACT_MATRIX_DIR / f'{self.country}.csv'
        contact_matrix = pd.read_csv(contact_matrix_path).to_numpy()
        n_categories = len(self.age_limits) - 1
        ind_limits = np.array(self.age_limits / 5, dtype=int)
        p = np.zeros(16)
        for i in range(n_categories):
            p[ind_limits[i]: ind_limits[i + 1]] = self.population_vector[i] / (ind_limits[i + 1] - ind_limits[i])
        transformed_matrix = np.zeros((n_categories, n_categories))
        for i in range(n_categories):
            for j in range(n_categories):
                sump = sum(p[ind_limits[i]: ind_limits[i + 1]])
                b = contact_matrix[ind_limits[i]: ind_limits[i + 1], ind_limits[j]: ind_limits[j + 1]] * np.array(
                    p[ind_limits[i]: ind_limits[i + 1]]).transpose()
                v1 = b.sum() / sump
                transformed_matrix[i, j] = v1
        return transformed_matrix

    def _generate_infection_matrix(self):
        # TODO: write tests for it with known cases
        infection_matrix = self._generate_contact_matrix()
        assert infection_matrix.shape[0] == infection_matrix.shape[1], "Infection matrix is supposed to be a square matrix"

        next_generation_matrix = np.matmul(0.01 * np.diag(self.population_vector), infection_matrix)
        largest_eigenvalue = max(np.linalg.eig(next_generation_matrix)[0])  # max eigenvalue

        beta_list = np.linspace(self.beta_list[0], self.beta_list[2], 20)
        beta_list = np.real((1 / largest_eigenvalue) * beta_list)  # in case eigenvalue imaginary

        return infection_matrix, beta_list, largest_eigenvalue

    def _compute_population_vector(self):
        # generate population vector
        age0to5 = float(self.age_population_0_5)
        age6to9 = float(self.age_population_6_9)
        population_structure = np.asarray([age0to5 + age6to9, float(self.age_population_10_19),
                                           float(self.age_population_20_29),
                                           float(self.age_population_30_39),
                                           float(self.age_population_40_49),
                                           float(self.age_population_50_59),
                                           float(self.age_population_60_69),
                                           float(self.age_population_70+)])
        self.population_size = int(self.total_population)
        # load the population vector as a vector
        return population_structure / population_size * 100

    def load_model_parameters(self):
        # in toatl there are 11 disease compartments
        self.number_compartments = 11
        # 8 age compartments with 10 year gap in each
        self.ages = ['0-9', '10-19', '20-29', '30-39', '40-49', '50-59', '60-69', '70+']
        self.age_categories = len(self.ages)
        self.age_limits = np.array([0, 10, 20, 30, 40, 50, 60, 70, 80], dtype=int)
        # 11 disease state compartments
        self.calculated_categories = ['S', 'E', 'I', 'A', 'R', 'H', 'C', 'D', 'O', 'Q', 'U']
        self.change_in_categories = ['C' + category for category in self.calculated_categories]

        # model control parameters
        # These are unique model control params
        self.shield_decrease = np.float(model_config_cm["shiedling_reduction_between_groups"])
        self.shield_increase = np.float(model_config_cm["shielding_increase_within_group"])
        self.better_hygiene = np.float(model_config_cm["better_hygiene_infection_scale"])

        # we will get this from the UI default to 14 for now
        self.quarant_rate = 1 / (np.float(model_config_cm["default_quarantine_period"]))

    def generate_epidemic_parameter_ranges(self):
        """Generate ranges of parameters with input parameters as mean and some custom standard deviation around it"""

    def ode_equations(self, t, y, beta, latent_rate, removal_rate):
        """ t is the time step and y is the value of the eqaution at each time step and this equation is run through at each integration time step"""

        y2d = y.reshape(self.age_categories, self.number_compartments).T
        # the gradients of number of people with respect to time
        dydt2d = np.zeros(y2d.shape)

        # some calculations upfront to make the differential equations look clean later
        S_vec = y2d[compartment_index['S'], :]
        E_vec = y2d[compartment_index['E'], :]
        I_vec = y2d[compartment_index['I'], :]
        H_vec = y2d[compartment_index['H'], :]
        A_vec = y2d[compartment_index['A'], :]
        C_vec = y2d[compartment_index['C'], :]
        Q_vec = y2d[compartment_index['Q'], :]

        E_latent = latent_rate * y2d[compartment_index['E'], :]
        I_removed = removal_rate * I_vec
        Q_quarantined = self.quarant_rate * Q_vec

        total_I = sum(I_vec)
        total_H = sum(H_vec)

        first_high_risk_category_n = age_categories - remove_high_risk['n_categories_removed']
        S_removal = sum(y2d[compartment_index['S'], first_high_risk_category_n))

        # removing symptomatic individuals
        # these are put into Q ('quarantine');
        quarantine_sicks = (remove_symptomatic_rate / total_I) * I_vec  # no age bias in who is moved

        # Laying out differential equations:
        # S
        infection_I = np.dot(self.infection_matrix, I_vec)
        infection_A = np.dot(self.infection_matrix, A_vec)
        infection_total = (infection_I + self.AsymptInfectiousFactor * infection_A)
        offsite = high_risk_people_removal_rates / S_removal * S_vec
        dydt2d[compartment_index['S'], :] = (- transmission_reduction_factor * beta * S_vec * infection_total - offsite)

        # E
        dydt2d[compartment_index['E'], :] = (transmission_reduction_factor * beta * S_vec * infection_total - E_latent)

        # I
        dydt2d[compartment_index['I'], :] = (self.p_symptomatic * E_latent - I_removed - quarantine_sicks)

        # A
        A_removed = removal_rate * A_vec
        dydt2d[compartment_index['A'], :] = ((1 - self.p_symptomatic) * E_latent - A_removed)

        # H
        dydt2d[compartment_index['H'], :] = (self.p_hosp_given_symptomatic * I_removed - hosp_rate * H_vec
                              + death_rate_ICU * (1 - self.death_prob_with_ICU) *
                              np.minimum(C_vec, hospitalized_on_icu)  # recovered from ICU
                              + self.p_hosp_given_symptomatic * Q_quarantined
                              # proportion of removed people who were hospitalised once returned
                              )

        # Critical care (ICU)
        deaths_on_icu = death_rate_ICU * C_vec
        without_deaths_on_icu = C_vec - deaths_on_icu
        needing_care = hosp_rate * self.p_critical_given_hospitalised * H_vec  # number needing care

        # number who get icu care (these entered category C)
        icu_cared = np.minimum(needing_care, hospitalized_on_icu - without_deaths_on_icu)

        # amount entering is minimum of: amount of beds available**/number needing it
        # **including those that will be made available by new deaths
        # without ICU treatment
        dydt2d[compartment_index['C'], :] = (icu_cared - deaths_on_icu)

        # Uncared - no ICU
        deaths_without_icu = death_rate_NoICU * y2d[compartment_index['U'],:]  # died without ICU treatment (all cases that don't get treatment die)
        dydt2d[compartment_index['U'], :] = (needing_care - icu_cared - deaths_without_icu)  # without ICU treatment

        # R
        # proportion of removed people who recovered once returned
        dydt2d[compartment_index['R'], :] = (
                (1 - self.p_hosp_given_symptomatic) * I_removed + A_removed + hosp_rate * (1 - self.p_critical_given_hospitalised) * H_vec + (
                    1 - self.p_hosp_given_symptomatic) * Q_quarantined
        )

        # D
        dydt2d[compartment_index['D'], :] = (
                deaths_without_icu + self.death_prob_with_ICU * deaths_on_icu  # died despite attempted ICU treatment
        )

        # O
        dydt2d[compartment_index['O'], :] = offsite

        # Q
        dydt2d[compartment_index['Q'], :] = quarantine_sicks - Q_quarantined

        # here the ICU implementation involves as np.minimum TODO: simulate an experiment for the people needing care below the the actual ICU capacity and observe if there is any dubious behaviour

        return dydt2d.T.reshape(y.shape)

    def run_model(self, t_stop=200, initial_exposed=0, initial_symp=0, initial_asymp=0, intergrator_type='vode'):
        """"""
        # initialise the epidemic
        seir_matrix = np.zeros((self.number_compartments, 1))

        seir_matrix[compartment_index['E'], 0] = initial_exposed / self.population_size  # exposed
        seir_matrix[compartment_index['I'], 0] = initial_symp / self.population_size  # sympt
        seir_matrix[compartment_index['A'], 0] = initial_asymp / self.population_size  # asympt

        seir_matrix[compartment_index['S'], 0] = 1 - seir_matrix.sum()

        y_initial = np.dot(seir_matrix, self.population_vector.reshape(1, self.age_categories) / 100)

        # initial conditions
        y0 = y_initial.T.reshape(self.number_compartments * self.age_categories)

        sol = ode(self.ode_equations).set_f_params(beta, latent_rate, removal_rate, hosp_rate, death_rate_ICU, death_rate_no_ICU).set_integrator(intergrator_type, nsteps=2000)

        time_range = np.arange(t_stop+1)  # 1 time value per day

        sol.set_initial_value(y0, time_range[0])

        y_out = np.zeros((len(y0), len(time_range)))

        t_sim = 0
        y_out[:, 0] = sol.y
        for t in time_range[1:]:
            if sol.successful():
                sol.integrate(t)
                t_sim = t_sim + 1
                y_out[:, t_sim] = sol.y
            else:
                raise RuntimeError('ode solver unsuccessful')

        return y_out

    def run_single_simulation(self):
        pass

    def run_multiple_simulations(self):
        pass


class DeterministicCompartmentalModelScenarios(object):
    def __init__(self):
        # baseline parameters that need to be run
        pass

    def better_hygiene(self):
        pass

    def remove_highrisk_offsite(self):
        pass

    def isolate_symptomatic(self):
        pass

    def increase_icu_capacity(cls):
        pass

    def shielding(self):
        # this belongs to the shielding increase scenario where the infection matrix will be modified even more
        if self.control_dict['shielding']['used']:  # increase contact within group and decrease between groups
            divider = -1  # determines which groups separated. -1 means only oldest group separated from the rest

            infection_matrix[:divider, :divider] = self.shield_increase * infection_matrix[:divider, :divider]
            infection_matrix[:divider, divider:] = self.shield_decrease * infection_matrix[:divider, divider:]
            infection_matrix[divider:, :divider] = self.shield_decrease * infection_matrix[divider:, :divider]
            infection_matrix[divider:, divider] = self.shield_increase * infection_matrix[divider:, divider:]

        pass