from typing import Sequence, Set

INTERACTION_LIST: Sequence[str] = (
    'Hydrophobic',
    'PiStacking_P',
    'PiStacking_T',
    'PiCation_lring',
    'PiCation_pring',
    'HBond_ldon',
    'HBond_pdon',
    'SaltBridge_lneg',
    'SaltBridge_pneg',
    'XBond'
)

NUM_INTERACTION_TYPES: int = 10

HYDROPHOBIC = 0
PISTACKING_P = 1
PISTACKING_T = 2
PICATION_LRING = 3
PICATION_PRING = 4
HBOND_LDON = 5
HBOND_PDON = 6
SALTBRIDGE_LNEG = 7
SALTBRIDGE_PNEG = 8
XBOND = 9

# PLIP Distance + 0.5 A
INTERACTION_DIST = {
    HYDROPHOBIC: 4.5,       # 4.0 + 0.5
    PISTACKING_P: 6.0,      # 5.5 + 0.5
    PISTACKING_T: 6.0,      # 5.5 + 0.5
    PICATION_LRING: 6.5,    # 6.0 + 0.5
    PICATION_PRING: 6.5,    # 6.0 + 0.5
    HBOND_LDON: 4.5,        # 4.1 + 0.5 - 0.1 (to be devided to 0.5)
    HBOND_PDON: 4.5,        # 4.1 + 0.5 - 0.1
    SALTBRIDGE_LNEG: 6.0,   # 5.5 + 0.5
    SALTBRIDGE_PNEG: 6.0,   # 5.5 + 0.5
    XBOND: 4.5,             # 4.0 + 0.5
}

LONG_INTERACTION: Set[int] = {
    PISTACKING_P,
    PISTACKING_T,
    PICATION_PRING,
    PICATION_LRING,
    SALTBRIDGE_LNEG,
    SALTBRIDGE_PNEG
}

SHORT_INTERACTION: Set[int] = {
    HYDROPHOBIC,
    HBOND_LDON,
    HBOND_PDON,
    XBOND,
}

