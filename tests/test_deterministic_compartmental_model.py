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
    base_dir = Path(os.path.dirname(__file__)).parents[0]
    camp_params = CampParams.load_from_json(
        base_dir / "epi_models" / "config" / "sample_input.json"
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
    result_set["do_nothing_baseline"] = do_nothing_baseline
    result_set["camp_baseline"] = camp_baseline
    result_set["camp_params"] = camp_params
    result_set["generated_params_df"] = runner.generated_params_df
    result_set["num_iterations"] = num_iterations
    result_set[
        "better_hygiene_intervention_result"
    ] = better_hygiene_intervention_result
    result_set["increase_icu_intervention_result"] = increase_icu_intervention_result
    result_set["increase_remove_high_risk_result"] = increase_remove_high_risk_result
    result_set[
        "better_isolation_intervention_result"
    ] = better_isolation_intervention_result
    result_set["shielding_intervention_result"] = shielding_intervention_result
    return result_set


def test_individual_age_compartments(instantiate_runner):
    result_set = instantiate_runner
    do_nothing_baseline = result_set["do_nothing_baseline"]
    for j in [
        "Susceptible",
        "Exposed",
        "Infected_symptomatic",
        "Infected_asymptomatic",
        "Recovered",
        "Hospitalised",
        "Critical",
        "Deaths",
        "Offsite",
        "Quarantined",
        "No_ICU_Care",
    ]:
        for i in range(0, 201):
            assert_almost_equal(
                do_nothing_baseline[j][i],
                do_nothing_baseline[str(j) + "_0_9"][i]
                + do_nothing_baseline[str(j) + "_10_19"][i]
                + do_nothing_baseline[str(j) + "_20_29"][i]
                + do_nothing_baseline[str(j) + "_30_39"][i]
                + do_nothing_baseline[str(j) + "_40_49"][i]
                + do_nothing_baseline[str(j) + "_50_59"][i]
                + do_nothing_baseline[str(j) + "_60_69"][i]
                + do_nothing_baseline[str(j) + "_70_above"][i],
            )


def test_individual_compartment(instantiate_runner):
    sum = 0
    result_set = instantiate_runner
    do_nothing_baseline = result_set["do_nothing_baseline"]
    camp_params = result_set["camp_params"]
    for j in [
        "Susceptible",
        "Exposed",
        "Infected_symptomatic",
        "Infected_asymptomatic",
        "Recovered",
        "Hospitalised",
        "Critical",
        "Deaths",
        "Offsite",
        "Quarantined",
        "No_ICU_Care",
    ]:
        sum += do_nothing_baseline[j]
    for i in range(0, 201):
        assert_almost_equal(sum[i], camp_params.total_population)


def test_intervention_better_hygiene(instantiate_runner):
    result_set = instantiate_runner
    do_nothing_baseline = result_set["do_nothing_baseline"]
    camp_params = result_set["camp_params"]
    num_iterations = result_set["num_iterations"]
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
        better_hygiene_6_month, generated_params_df=result_set["generated_params_df"]
    )
    better_hygiene_6_month_groups = better_hygiene_6_month_results.groupby("R0")
    do_nothing_infected = 0
    for index, group in sim_groups:
        do_nothing_infected = group["Infected_symptomatic"][1:31]
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
        better_hygiene_infected = group["Infected_symptomatic"][1:31]

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
    do_nothing_baseline = result_set["do_nothing_baseline"]
    camp_params = result_set["camp_params"]
    num_iterations = result_set["num_iterations"]
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
        iso_6_month, generated_params_df=result_set["generated_params_df"]
    )
    iso_6_month_results_groups = iso_6_month_results.groupby("R0")
    do_nothing_infected = 0
    for index, group in sim_groups:
        do_nothing_infected = group["Infected_symptomatic"][1:31]
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
        isolation_infected = group["Infected_symptomatic"][1:31]

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
    do_nothing_baseline = result_set["do_nothing_baseline"]
    camp_baseline = result_set["camp_baseline"]
    sim_groups = do_nothing_baseline.groupby("R0")
    sim_groups_camp = camp_baseline.groupby("R0")
    do_nothing_infected = 0
    for index, group in sim_groups:
        do_nothing_infected = group["Infected_symptomatic"][1:31]
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
        camp_baseline_infected = group["Infected_symptomatic"][1:31]
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


def test_run_different_scenarios(instantiate_runner):
    result_set = instantiate_runner
    do_nothing_baseline = result_set["do_nothing_baseline"]
    better_hygiene_intervention_result = result_set[
        "better_hygiene_intervention_result"
    ]
    increase_icu_intervention_result = result_set["increase_icu_intervention_result"]
    increase_remove_high_risk_result = result_set["increase_remove_high_risk_result"]
    better_isolation_intervention_result = result_set[
        "better_isolation_intervention_result"
    ]
    shielding_intervention_result = result_set["shielding_intervention_result"]
    sim_groups = do_nothing_baseline.groupby("R0")
    sim_group_better_hygiene = better_hygiene_intervention_result.groupby("R0")
    sim_group_increase_icu = increase_icu_intervention_result.groupby("R0")
    sim_group_increase_remove_high_risk = increase_remove_high_risk_result.groupby("R0")
    sim_group_better_isolation = better_hygiene_intervention_result.groupby("R0")
    sim_group_shielding_intervention = shielding_intervention_result.groupby("R0")
    do_nothing_infected = 0
    for index, group in sim_groups:
        do_nothing_infected = group["Infected_symptomatic"][1:31]
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
    for index, group in sim_group_better_hygiene:
        better_hygiene_infected = group["Infected_symptomatic"][1:31]
    better_hygiene_hospitalised = 0
    for index, group in sim_group_better_hygiene:
        better_hygiene_hospitalised = group["Hospitalised"][1:31]
    better_hygiene_critical = 0
    for index, group in sim_group_better_hygiene:
        better_hygiene_critical = group["Critical"][1:31]
    better_hygiene_deaths = 0
    for index, group in sim_group_better_hygiene:
        better_hygiene_deaths = group["Deaths"][1:31]
    increase_icu_infected = 0
    for index, group in sim_group_increase_icu:
        increase_icu_infected = group["Infected_symptomatic"][1:31]
    increase_icu_hospitalised = 0
    for index, group in sim_group_increase_icu:
        increase_icu_hospitalised = group["Hospitalised"][1:31]
    increase_icu_critical = 0
    for index, group in sim_group_increase_icu:
        increase_icu_critical = group["Critical"][1:31]
    increase_icu_deaths = 0
    for index, group in sim_group_increase_icu:
        increase_icu_deaths = group["Deaths"][1:31]
    increase_remove_high_risk_infected = 0
    for index, group in sim_group_increase_remove_high_risk:
        increase_remove_high_risk_infected = group["Infected_symptomatic"][1:31]
    increase_remove_high_risk_hospitalised = 0
    for index, group in sim_group_increase_remove_high_risk:
        increase_remove_high_risk_hospitalised = group["Hospitalised"][1:31]
    increase_remove_high_risk_critical = 0
    for index, group in sim_group_increase_remove_high_risk:
        increase_remove_high_risk_critical = group["Critical"][1:31]
    increase_remove_high_risk_deaths = 0
    for index, group in sim_group_increase_remove_high_risk:
        increase_remove_high_risk_deaths = group["Deaths"][1:31]
    better_isolation_infected = 0
    for index, group in sim_group_better_isolation:
        better_isolation_infected = group["Infected_symptomatic"][1:31]
    better_isolation_hospitalised = 0
    for index, group in sim_group_better_isolation:
        better_isolation_hospitalised = group["Hospitalised"][1:31]
    better_isolation_critical = 0
    for index, group in sim_group_better_isolation:
        better_isolation_critical = group["Critical"][1:31]
    better_isolation_deaths = 0
    for index, group in sim_group_better_isolation:
        better_isolation_deaths = group["Deaths"][1:31]
    shielding_intervention_infected = 0
    for index, group in sim_group_shielding_intervention:
        shielding_intervention_infected = group["Infected_symptomatic"][1:31]
    shielding_intervention_hospitalised = 0
    for index, group in sim_group_shielding_intervention:
        shielding_intervention_hospitalised = group["Hospitalised"][1:31]
    shielding_intervention_critical = 0
    for index, group in sim_group_shielding_intervention:
        shielding_intervention_critical = group["Critical"][1:31]
    shielding_intervention_deaths = 0
    for index, group in sim_group_shielding_intervention:
        shielding_intervention_deaths = group["Deaths"][1:31]

    # testing infected is lesser in isolation scenarios than do nothing
    assert_array_less(better_hygiene_infected, do_nothing_infected)

    # testing hospitalised is lesser in isolation scenarios than do nothing
    assert_array_less(better_hygiene_hospitalised, do_nothing_hospitalised)

    # testing critical is lesser in isolation scenarios than do nothing
    assert_array_less(better_hygiene_critical, do_nothing_critical)

    # testing deaths is lesser in isolation scenarios than do nothing
    assert_array_less(better_hygiene_deaths, do_nothing_deaths)

    # # testing infected is lesser in isolation scenarios than do nothing
    # assert_array_less(increase_icu_infected, do_nothing_infected)

    # # testing hospitalised is lesser in isolation scenarios than do nothing
    # assert_array_less(increase_icu_hospitalised, do_nothing_hospitalised)

    # # testing critical is lesser in isolation scenarios than do nothing
    # assert_array_less(increase_icu_critical, do_nothing_critical)

    # # testing deaths is lesser in isolation scenarios than do nothing
    # assert_array_less(increase_icu_deaths, do_nothing_deaths)

    # testing infected is lesser in isolation scenarios than do nothing
    assert_array_less(increase_remove_high_risk_infected, do_nothing_infected)

    # testing hospitalised is lesser in isolation scenarios than do nothing
    assert_array_less(increase_remove_high_risk_hospitalised, do_nothing_hospitalised)

    # testing critical is lesser in isolation scenarios than do nothing
    assert_array_less(increase_remove_high_risk_critical, do_nothing_critical)

    # testing deaths is lesser in isolation scenarios than do nothing
    assert_array_less(increase_remove_high_risk_deaths, do_nothing_deaths)

    # testing infected is lesser in isolation scenarios than do nothing
    assert_array_less(better_isolation_infected, do_nothing_infected)

    # testing hospitalised is lesser in isolation scenarios than do nothing
    assert_array_less(better_isolation_hospitalised, do_nothing_hospitalised)

    # testing critical is lesser in isolation scenarios than do nothing
    assert_array_less(better_isolation_critical, do_nothing_critical)

    # testing deaths is lesser in isolation scenarios than do nothing
    assert_array_less(better_isolation_deaths, do_nothing_deaths)

    # # testing infected is lesser in isolation scenarios than do nothing
    # assert_array_less(shielding_intervention_infected, do_nothing_infected)

    # # testing hospitalised is lesser in isolation scenarios than do nothing
    # assert_array_less(shielding_intervention_hospitalised, do_nothing_hospitalised)

    # # testing critical is lesser in isolation scenarios than do nothing
    # assert_array_less(shielding_intervention_critical, do_nothing_critical)

    # # testing deaths is lesser in isolation scenarios than do nothing
    # assert_array_less(shielding_intervention_deaths, do_nothing_deaths)
