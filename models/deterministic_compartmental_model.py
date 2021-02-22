from .model import Model,ModelId
import numpy as np
from .config.compartmental_model import Config
import pandas as pd
from scipy.integrate import ode
import dask
from dask.diagnostics import ProgressBar
import multiprocessing
from .params import CampParams


class DeterministicCompartmentalModel(Model):

    def __init__(self, camp_params: CampParams, num_iterations=1000):
        super().__init__()
        # load parameters
        self.load_epidemic_parameters()
        self.load_model_parameters()
        self.load_camp_parameters(camp_params)
        # process parameters
        self.process_epidemic_parameters()
        self.population_vector = self._compute_population_vector()
        self.infection_matrix, self.im_beta_list, self.largest_eigenvalue = self._generate_infection_matrix()

    def id(self):
        return ModelId.DeterministicCompartmentalModel

    def _generate_contact_matrix(self):
        """Squeeze 5-year gap, 16 age compartment POLYMOD contact matrix into 10-year gap, 8 age compartment used in this model"""
        # TODO Walk through this code and write some tests for it
        contact_matrix_path = Config.CONTACT_MATRIX_DIR / f'{self.country}.csv'
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
                                           float(self.age_population_70_and_above)])
        self.population_size = int(self.total_population)
        # load the population vector as a vector
        return population_structure / self.population_size * 100

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
        self.shield_decrease = np.float(Config.model_params["shiedling_reduction_between_groups"])
        self.shield_increase = np.float(Config.model_params["shielding_increase_within_group"])
        self.better_hygiene = np.float(Config.model_params["better_hygiene_infection_scale"])

        # we will get this from the UI default to 14 for now
        self.quarant_rate = 1 / (np.float(Config.model_params["default_quarantine_period"]))

    def process_epidemic_parameters(self):
        self.R_0_list = [self.R0_low, self.R0_medium, self.R0_high]
        removal_rate = 1 / (np.float(self.Infectious_period))
        self.beta_list = [R_0 * removal_rate for R_0 in self.R_0_list]

        self.p_symptomatic = np.array(self.p_symptomatic)
        self.p_hosp_given_symptomatic = np.array(self.p_hosp_given_symptomatic)
        self.p_critical_given_hospitalised = np.array(self.p_critical_given_hospitalised)

    def generate_epidemic_parameter_ranges(self, num_iterations, scale=1, lb=1, seed=42):
        """Generate ranges of parameters with input parameters as mean and some custom standard deviation around it default generate 1000 sets of parameters"""
        generated_params = {}
        np.random.seed(seed)
        generated_params['R0'] = np.random.normal(self.R_0_list[1], np.std(self.R_0_list), num_iterations)
        generated_params['LatentPeriod'] = np.random.normal(self.Latent_period, scale, num_iterations)
        generated_params['RemovalPeriod'] = np.random.normal(self.Infectious_period, scale, num_iterations)
        generated_params['HospPeriod'] = np.random.normal(self.Hosp_period, scale, num_iterations)
        generated_params['DeathICUPeriod'] = np.random.normal(self.Death_period_withICU, scale, num_iterations)
        generated_params['DeathNoICUPeriod'] = np.random.normal(self.Death_period, scale, num_iterations)
        generated_params_df = pd.DataFrame(generated_params)
        generated_params_df[generated_params_df <= 1] = lb
        generated_params_df['latentRate'] = 1 / generated_params_df['LatentPeriod']
        generated_params_df['removalRate'] = 1 / generated_params_df['RemovalPeriod']
        generated_params_df['hospRate'] = 1 / generated_params_df['HospPeriod']
        generated_params_df['deathRateICU'] = 1 / generated_params_df['DeathICUPeriod']
        generated_params_df['deathRateNoICU'] = 1 / generated_params_df['DeathNoICUPeriod']
        generated_params_df['beta'] = generated_params_df['removalRate'] * generated_params_df['R0'] / self.largest_eigenvalue
        return generated_params_df

    @staticmethod
    def _timing_function(t, time_vector):

        for t in range(ceil(len(time_vector) / 2)):
            if t >= time_vector[2 * ii] and t < time_vector[2 * ii + 1]:
                return True
        # if wasn't in any of these time interval
        return False

    def ode_equations(self, t, y, beta, latent_rate, removal_rate, hosp_rate, death_rate_ICU, death_rate_no_ICU, scenario):
        """ t is the time step and y is the value of the eqaution at each time step and this equation is run through at each integration time step"""
        # extract scenario dict for this time step:
        scenario_dict = scenario.intervention_params_at_time_t(t)
        y2d = y.reshape(self.age_categories, self.number_compartments).T
        # the gradients of number of people with respect to time
        dydt2d = np.zeros(y2d.shape)

        # some calculations upfront to make the differential equations look clean later
        S_vec = y2d[Config.compartment_index['S'], :]
        E_vec = y2d[Config.compartment_index['E'], :]
        I_vec = y2d[Config.compartment_index['I'], :]
        H_vec = y2d[Config.compartment_index['H'], :]
        A_vec = y2d[Config.compartment_index['A'], :]
        C_vec = y2d[Config.compartment_index['C'], :]
        Q_vec = y2d[Config.compartment_index['Q'], :]

        E_latent = latent_rate * y2d[Config.compartment_index['E'], :]
        I_removed = removal_rate * I_vec
        Q_quarantined = self.quarant_rate * Q_vec

        total_I = sum(I_vec)
        total_H = sum(H_vec)

        # Intervention: removing high risk population
        first_high_risk_category_n = self.age_categories - scenario_dict['first_high_risk_category_n']
        S_removal = sum(y2d[Config.compartment_index['S']], first_high_risk_category_n)

        # Intervention: removing symptomatic individuals
        # these are put into Q ('quarantine');
        quarantine_sicks = (scenario_dict['remove_symptomatic_rate'] / total_I) * I_vec  # no age bias in who is moved

        # ICU capacity
        if total_H > 0:  # can't divide by 0
            hospitalized_on_icu = (scenario_dict["icu_capacity"] / self.population_size) / total_H * H_vec
            # ICU beds allocated on a first come, first served basis based on the numbers in hospital
        else:
            hospitalized_on_icu = np.full(self.age_categories, (scenario_dict["icu_capacity"] / self.population_size))

        # Laying out differential equations:
        # S
        # Intervention: shielding
        infection_I = np.dot(self.infection_matrix, I_vec)
        infection_A = np.dot(self.infection_matrix, A_vec)
        infection_total = (infection_I + self.AsymptInfectiousFactor * infection_A)
        offsite = scenario_dict["remove_high_risk_rate"] / S_removal * S_vec
        # Intervention: transimission reduction via better hygiene
        dydt2d[Config.compartment_index['S'], :] = (- scenario_dict["transmission_reduction_factor"] * beta * S_vec * infection_total - offsite)

        # E
        dydt2d[Config.compartment_index['E'], :] = (scenario_dict["transmission_reduction_factor"] * beta * S_vec * infection_total - E_latent)

        # I
        dydt2d[Config.compartment_index['I'], :] = (self.p_symptomatic * E_latent - I_removed - quarantine_sicks)

        # A
        A_removed = removal_rate * A_vec
        dydt2d[Config.compartment_index['A'], :] = ((1 - self.p_symptomatic) * E_latent - A_removed)

        # H
        dydt2d[Config.compartment_index['H'], :] = (self.p_hosp_given_symptomatic * I_removed - hosp_rate * H_vec
                              + death_rate_ICU * (1 - self.death_prob_with_ICU) *
                              np.minimum(C_vec, hospitalized_on_icu)  # recovered from ICU
                              + self.p_hosp_given_symptomatic * Q_quarantined
                              # proportion of removed people who were hospitalised once returned
                              )

        # Intervention Critical care (ICU)
        deaths_on_icu = death_rate_ICU * C_vec
        without_deaths_on_icu = C_vec - deaths_on_icu
        needing_care = hosp_rate * self.p_critical_given_hospitalised * H_vec  # number needing care

        # number who get icu care (these entered category C)
        # print(f"at time {t} the hospitalized_on_icu value is {hospitalized_on_icu*20000}")
        # print(f"at time {t} the without_deaths_on_icu value is {without_deaths_on_icu*20000}")
        icu_cared = np.minimum(needing_care, hospitalized_on_icu - without_deaths_on_icu)

        # amount entering is minimum of: amount of beds available**/number needing it
        # **including those that will be made available by new deaths
        # without ICU treatment
        dydt2d[Config.compartment_index['C'], :] = (icu_cared - deaths_on_icu)

        # Uncared - no ICU
        deaths_without_icu = death_rate_no_ICU * y2d[Config.compartment_index['U'], :]  # died without ICU treatment (all cases that don't get treatment die)
        dydt2d[Config.compartment_index['U'], :] = (needing_care - icu_cared - deaths_without_icu)  # without ICU treatment

        # R
        # proportion of removed people who recovered once returned
        dydt2d[Config.compartment_index['R'], :] = (
                (1 - self.p_hosp_given_symptomatic) * I_removed + A_removed + hosp_rate * (1 - self.p_critical_given_hospitalised) * H_vec + (
                    1 - self.p_hosp_given_symptomatic) * Q_quarantined
        )

        # D
        dydt2d[Config.compartment_index['D'], :] = (
                deaths_without_icu + self.death_prob_with_ICU * deaths_on_icu  # died despite attempted ICU treatment
        )

        # O
        dydt2d[Config.compartment_index['O'], :] = offsite

        # Q
        dydt2d[Config.compartment_index['Q'], :] = quarantine_sicks - Q_quarantined

        # here the ICU implementation involves as np.minimum TODO: simulate an experiment for the people needing care below the the actual ICU capacity and observe if there is any dubious behaviour

        return dydt2d.T.reshape(y.shape)

    def run_model(self, scenario, t_stop=200, r0=None, beta=None, latent_rate=None, removal_rate=None, hosp_rate=None, death_rate_ICU=None, death_rate_no_ICU=None, initial_exposed=0, initial_symp=0, initial_asymp=0, intergrator_type='vode'):
        """high level function for running the model via differential equation solver from scipy"""
        # initialise the epidemic
        seir_matrix = np.zeros((self.number_compartments, 1))

        seir_matrix[Config.compartment_index['E'], 0] = initial_exposed / self.population_size  # exposed
        seir_matrix[Config.compartment_index['I'], 0] = initial_symp / self.population_size  # sympt
        seir_matrix[Config.compartment_index['A'], 0] = initial_asymp / self.population_size  # asympt

        seir_matrix[Config.compartment_index['S'], 0] = 1 - seir_matrix.sum()

        y_initial = np.dot(seir_matrix, self.population_vector.reshape(1, self.age_categories) / 100)

        # initial conditions
        y0 = y_initial.T.reshape(self.number_compartments * self.age_categories)

        sol = ode(self.ode_equations).set_f_params(beta, latent_rate, removal_rate, hosp_rate, death_rate_ICU, death_rate_no_ICU, scenario).set_integrator(intergrator_type, nsteps=5000)

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

        y_sum = np.zeros((self.number_compartments, t_sim + 1))
        for compartment in Config.longname.keys():
            for i in range(self.age_categories):  # age_categories
                y_sum[Config.compartment_index[compartment], :] += y_out[Config.compartment_index[compartment] + i * self.number_compartments,:]

        solution_frame = self.parse_model_output(y_out, y_sum, time_range, r0, latent_rate, removal_rate, hosp_rate, death_rate_ICU, death_rate_no_ICU)

        return solution_frame

    def parse_model_output(self, y_out, y_sum, time_range, r0, latent_rate, removal_rate, hosp_rate, death_rate_ICU, death_rate_no_ICU):
        """ to be run after a simulation to present simualtion results in a dataframe"""
        # setup column names
        AGE_SEP = ': '  # separate compartment and age in column name
        disease_compartment_col_names = [name for name in Config.longname.values()]
        disease_age_compartment_col_names = [name + AGE_SEP + age for age in self.ages for name in Config.longname.values()]
        disease_param_col_names = ['R0', 'latentRate', 'removalRate', 'hospRate', 'deathRateICU', 'deathRateNoIcu']
        time_col_name = ['Time']
        data_store = np.transpose(y_out)
        data_store_df = pd.DataFrame(data_store, columns = disease_age_compartment_col_names)
        data_store_df['Time'] = time_range
        data_store_df['R0'] = [r0] * len(time_range)
        data_store_df['latentRate'] = [latent_rate] * len(time_range)
        data_store_df['removalRate'] = [removal_rate] * len(time_range)
        data_store_df['hospRate'] = [hosp_rate] * len(time_range)
        data_store_df['deathRateICU'] = [death_rate_ICU] * len(time_range)
        data_store_df['deathRateNoIcu'] = [death_rate_no_ICU] * len(time_range)
        aggregated_compartment_output = np.transpose(y_sum)
        data_store_df = pd.concat([data_store_df, pd.DataFrame(aggregated_compartment_output, columns=disease_compartment_col_names)], axis=1)
        col_names = disease_compartment_col_names + disease_age_compartment_col_names + disease_param_col_names + time_col_name
        assert len(col_names) == len(data_store_df.columns)
        return data_store_df

    def run_single_simulation(self, scenario, num_iterations=1000, t_stop=200, initial_exposed=1, initial_symp=1, initial_asymp=1):
        # TODO: swap this for a dask distributed client so it is non-blocking
        # allow two implementation where one the initial seeds are fixed throughout
        # and the second one where initial exposed/symp/asymp are input as arrays
        generated_params_df = self.generate_epidemic_parameter_ranges(num_iterations)
        lazy_sols = []
        sols_raw = {} # raw output for solutions
        for index, row in generated_params_df.iterrows():
            lazy_result = dask.delayed(self.run_model)(scenario=scenario, t_stop=t_stop, r0=row["R0"], beta=row["beta"],
                                                       latent_rate=row['latentRate'],
                                                       removal_rate=row['removalRate'],
                                                       hosp_rate=row['hospRate'],
                                                       death_rate_ICU=row['deathRateICU'],
                                                       death_rate_no_ICU=row['deathRateNoICU'],
                                                       initial_symp = initial_symp,
                                                       initial_asymp = initial_asymp,
                                                       )
            lazy_sols.append(lazy_result)
        with dask.config.set(scheduler='processes', num_workers=multiprocessing.cpu_count()):
            with ProgressBar():
                sols = dask.compute(*lazy_sols)
        simulation_result_frame = pd.concat(sols, axis=0)
        return simulation_result_frame

    def run_multiple_simulations(self):
        pass


