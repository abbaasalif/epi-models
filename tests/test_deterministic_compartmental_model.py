import os
from pathlib import Path

import pytest
from numpy.testing import assert_array_less
from pandas.util.testing import assert_almost_equal

from epi_models import (
    CampParams,
    DeterministicCompartmentalModel,
    DeterministicCompartmentalModelRunner,
    SingleInterventionScenario,
)


@pytest.fixture
def instantiate_runner():
    result_set = {}
    base_dir = Path(os.path.dirname(__file__))
    camp_params = CampParams.load_from_json(
        base_dir + "//epi_models//config//sample_input.json"
    )
    num_iterations = 1
    runner = DeterministicCompartmentalModelRunner(
        camp_params, num_iterations=num_iterations
    )
    do_nothing_baseline, camp_baseline = runner.run_baselines()
    (
        better_hygiene_intervention_result,
        increase_icu_intervention_result,
        increase_remove_high_risk_result,
        better_isolation_intervention_result,
        shielding_intervention_result,
    ) = runner.run_different_scenarios()
    runner.run_better_hygiene_scenarios()
    do_nothing_baseline = do_nothing_baseline * camp_params.total_population
    camp_baseline = camp_baseline * camp_params.total_population
    result_set["do_nothing_baseline"] = do_nothing_baseline
    result_set["camp_baseline"] = camp_baseline
    result_set["camp_params"] = camp_params
    result_set["generated_params_df"] = runner.generated_params_df
    return result_set


def test_individual_age_compartments(instantiate_runner):
    result_set = instantiate_runner
    do_nothing_baseline = result_set["do_nothing_baseline"]
    for j in [
        "Susceptible",
        "Exposed",
        "Infected (symptomatic)",
        "Asymptomatically Infected",
        "Recovered",
        "Hospitalised",
        "Critical",
        "Deaths",
        "Offsite",
        "Quarantined",
        "No ICU Care",
    ]:
        for i in range(0, 201):
            assert_almost_equal(
                do_nothing_baseline[j][i],
                do_nothing_baseline[str(j) + ": 0-9"][i]
                + do_nothing_baseline[str(j) + ": 10-19"][i]
                + do_nothing_baseline[str(j) + ": 20-29"][i]
                + do_nothing_baseline[str(j) + ": 30-39"][i]
                + do_nothing_baseline[str(j) + ": 40-49"][i]
                + do_nothing_baseline[str(j) + ": 50-59"][i]
                + do_nothing_baseline[str(j) + ": 60-69"][i]
                + do_nothing_baseline[str(j) + ": 70+"][i],
            )


def test_individual_compartment(instantiate_runner):
    sum = 0
    result_set = instantiate_runner
    do_nothing_baseline = result_set["do_nothing_baseline"]
    camp_params = result_set["camp_params"]
    for j in [
        "Susceptible",
        "Exposed",
        "Infected (symptomatic)",
        "Asymptomatically Infected",
        "Recovered",
        "Hospitalised",
        "Critical",
        "Deaths",
        "Offsite",
        "Quarantined",
        "No ICU Care",
    ]:
        sum += do_nothing_baseline[j]
    for i in range(0, 201):
        assert_almost_equal(sum[i], camp_params.total_population)


def test_intervention_better_hygiene(instantiate_runner):
    result_set = instantiate_runner
    result_set["do_nothing_baseline"]
    camp_params = result_set["camp_params"]
    sim_groups = do_nothing_baseline.groupby("R0")
    model = DeterministicCompartmentalModel(camp_params, num_iterations=num_iterations)
    better_hygiene_6_month = SingleInterventionScenario(
        camp_params.total_population,
        [0],
        [180],
        model.infection_matrix,
        transmission_reduction_factor_inter=0.7,
    )
    better_hygiene_6_month_results = model.run_single_simulation(
        better_hygiene_6_month, generated_params_df=runner.generated_params_df
    )
    better_hygiene_6_month_results = (
        camp_params.total_population * better_hygiene_6_month_results
    )
    better_hygiene_6_month_groups = better_hygiene_6_month_results.groupby("R0")
    do_nothing_infected = 0
    for index, group in sim_groups:
        do_nothing_infected = group["Infected (symptomatic)"][1:31]
    do_nothing_hospitalised = 0
    for index, group in sim_groups:
        do_nothing_hospitalised = group["Hospitalised"][1:31]
    do_nothing_critical = 0
    for index, group in sim_groups:
        do_nothing_critical = group["Critical"][1:31]
    do_nothing_deaths = 0
    for index, group in sim_groups:
        do_nothing_deaths = group["Deaths"][1:31]
    better_hygiene_infected = 0
    for index, group in better_hygiene_6_month_groups:
        better_hygiene_infected = group["Infected (symptomatic)"][1:31]

    better_hygiene_hospitalised = 0
    for index, group in better_hygiene_6_month_groups:
        better_hygiene_hospitalised = group["Hospitalised"][1:31]

    better_hygiene_critical = 0
    for index, group in better_hygiene_6_month_groups:
        better_hygiene_critical = group["Critical"][1:31]

    better_hygiene_deaths = 0
    for index, group in better_hygiene_6_month_groups:
        better_hygiene_deaths = group["Deaths"][1:31]
    # testing infected is lesser in better hygiene scenarios than do nothing
    assert_array_less(better_hygiene_infected, do_nothing_infected)

    # testing hospitalised is lesser in better hygiene scenarios than do nothing
    assert_array_less(better_hygiene_hospitalised, do_nothing_hospitalised)

    # testing critcal is lesser in better hygiene scenarios than do nothing
    assert_array_less(better_hygiene_critical, do_nothing_critical)

    # testing deaths is lesser in better hygiene scenarios than do nothing
    assert_array_less(better_hygiene_deaths, do_nothing_deaths)


def test_intervention_isolation(instantiate_runner):
    result_set = instantiate_runner
    result_set["do_nothing_baseline"]
    camp_params = result_set["camp_params"]
    sim_groups = do_nothing_baseline.groupby("R0")
    do_nothing_baseline, camp_baseline = runner.run_baselines()
    (
        better_hygiene_intervention_result,
        increase_icu_intervention_result,
        increase_remove_high_risk_result,
        better_isolation_intervention_result,
        shielding_intervention_result,
    ) = runner.run_different_scenarios()
    runner.run_better_hygiene_scenarios()
    camp_params = camp_params
    do_nothing_baseline = do_nothing_baseline * camp_params.total_population
    camp_baseline = camp_baseline * camp_params.total_population
    sim_groups = do_nothing_baseline.groupby("R0")
    model = DeterministicCompartmentalModel(camp_params, num_iterations=num_iterations)
    iso_6_month = SingleInterventionScenario(
        camp_params.total_population,
        [0],
        [180],
        model.infection_matrix,
        isolation_capacity_inter=10000,
        remove_symptomatic_rate_inter=10,
    )
    iso_6_month_results = model.run_single_simulation(
        iso_6_month, generated_params_df=runner.generated_params_df
    )
    iso_6_month_results = camp_params.total_population * iso_6_month_results
    iso_6_month_results_groups = iso_6_month_results.groupby("R0")
    do_nothing_infected = 0
    for index, group in sim_groups:
        do_nothing_infected = group["Infected (symptomatic)"][1:31]
    do_nothing_hospitalised = 0
    for index, group in sim_groups:
        do_nothing_hospitalised = group["Hospitalised"][1:31]
    do_nothing_critical = 0
    for index, group in sim_groups:
        do_nothing_critical = group["Critical"][1:31]
    do_nothing_deaths = 0
    for index, group in sim_groups:
        do_nothing_deaths = group["Deaths"][1:31]
    isolation_infected = 0
    for index, group in iso_6_month_results_groups:
        isolation_infected = group["Infected (symptomatic)"][1:31]

    isolation_hospitalised = 0
    for index, group in iso_6_month_results_groups:
        isolation_hospitalised = group["Hospitalised"][1:31]

    isolation_critical = 0
    for index, group in iso_6_month_results_groups:
        isolation_critical = group["Critical"][1:31]

    isolation_deaths = 0
    for index, group in iso_6_month_results_groups:
        isolation_deaths = group["Deaths"][1:31]
    # testing infected is lesser in isolation scenarios than do nothing
    assert_array_less(isolation_infected, do_nothing_infected)

    # testing hospitalised is lesser in isolation scenarios than do nothing
    assert_array_less(isolation_hospitalised, do_nothing_hospitalised)

    # testing critical is lesser in isolation scenarios than do nothing
    assert_array_less(isolation_critical, do_nothing_critical)

    # testing deaths is lesser in isolation scenarios than do nothing
    assert_array_less(isolation_deaths, do_nothing_deaths)


def test_camp_baselines_with_do_nothing(instantiate_runner):
    result_set = instantiate_runner
    result_set["do_nothing_baseline"]
    camp_baseline = result_set["camp_baseline"]
    sim_groups = do_nothing_baseline.groupby("R0")
    sim_groups_camp = camp_baseline.groupby("R0")
    do_nothing_infected = 0
    for index, group in sim_groups:
        do_nothing_infected = group["Infected (symptomatic)"][1:31]
    do_nothing_hospitalised = 0
    for index, group in sim_groups:
        do_nothing_hospitalised = group["Hospitalised"][1:31]
    do_nothing_critical = 0
    for index, group in sim_groups:
        do_nothing_critical = group["Critical"][1:31]
    do_nothing_deaths = 0
    for index, group in sim_groups:
        do_nothing_deaths = group["Deaths"][1:31]
    camp_baseline_infected = 0
    for index, group in sim_groups_camp:
        camp_baseline_infected = group["Infected (symptomatic)"][1:31]
    camp_baseline_hospitalised = 0
    for index, group in sim_groups_camp:
        camp_baseline_hospitalised = group["Hospitalised"][1:31]
    camp_baseline_critical = 0
    for index, group in sim_groups_camp:
        camp_baseline_critical = group["Critical"][1:31]
    camp_baseline_deaths = 0
    for index, group in sim_groups_camp:
        camp_baseline_deaths = group["Deaths"][1:31]
    # testing infected is lesser in isolation scenarios than do nothing
    assert_array_less(camp_baseline_infected, do_nothing_infected)

    # testing hospitalised is lesser in isolation scenarios than do nothing
    assert_array_less(camp_baseline_hospitalised, do_nothing_hospitalised)

    # testing critical is lesser in isolation scenarios than do nothing
    assert_array_less(camp_baseline_critical, do_nothing_critical)

    # testing deaths is lesser in isolation scenarios than do nothing
    assert_array_less(camp_baseline_deaths, do_nothing_deaths)
