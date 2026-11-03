FAM = {
    "F1": "Material substitution/modification",
    "F2": "Product modification",
    "F3": "Process/equipment modification",
    "F4": "Inventory, material management, spill/leak prevention",
    "F5": "Operating practices and training",
}
W = {
    "F5": ["W13", "W14", "W15", "W19"],
    "F4": ["W21", "W22", "W23", "W24", "W25", "W29", "W31", "W32", "W33", "W36", "W39", "W54", "W55"],
    "F1": ["W41", "W42", "W43", "W49", "W53", "W56", "W61", "W73"],
    "F3": [
        "W35",
        "W50",
        "W51",
        "W52",
        "W57",
        "W58",
        "W59",
        "W60",
        "W63",
        "W64",
        "W65",
        "W66",
        "W67",
        "W68",
        "W71",
        "W72",
        "W74",
        "W75",
        "W78",
    ],
    "F2": ["W81", "W82", "W83", "W84", "W89"],
}
S = {
    "F1": ["S01", "S02", "S03", "S04", "S05", "S06"],
    "F2": ["S11", "S12", "S13", "S14"],
    "F3": ["S21", "S22", "S23", "S24", "S25"],
    "F4": ["S31", "S32", "S33", "S34", "S35"],
    "F5": ["S41", "S42", "S43", "S44"],
}
C2F = {c: f for d in (W, S) for f, cs in d.items() for c in cs}
