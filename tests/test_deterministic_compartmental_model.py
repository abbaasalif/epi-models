import os

import pytest
from pandas.util.testing import assert_almost_equal

from epi_models import CampParams, DeterministicCompartmentalModelRunner


@pytest.fixture
def instantiate_runner():
    os.chdir("..")
    camp_params = CampParams.load_from_json("epi_models//config//sample_input.json")
    num_iterations = 1
    return DeterministicCompartmentalModelRunner(
        camp_params, num_iterations=num_iterations
    )


@pytest.fixture
def camp_params():
    camp_parameters = CampParams.load_from_json("epi_models//config//sample_input.json")
    return camp_parameters


def test_individual_sanity_compartments(instantiate_runner, camp_params):
    runner = instantiate_runner
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
