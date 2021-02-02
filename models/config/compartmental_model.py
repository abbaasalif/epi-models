from pathlib import Path

model_config_cm = {
    "shiedling_reduction_between_groups": 0.25,
    "shielding_increase_within_group": 2,
    "default_quarantine_period": 5,
    "better_hygiene_infection_scale": 0.7,
}

CONTACT_MATRIX_DIR = Path(__file__) / "contact_matrices"

index = {'S': 0,
         'E': 1,
         'I': 2,
         'A': 3,
         'R': 4,
         'H': 5,
         'C': 6,
         'D': 7,
         'O': 8,
         'Q': 9,
         'U': 10,
         'C S': 11,
         'CE': 12,
         'CI': 13,
         'CA': 14,
         'CR': 15,
         'CH': 16,
         'CC': 17,
         'CD': 18,
         'CO': 19,
         'CQ': 20,
         'CU': 21,
         'Ninf': 22,
         }
