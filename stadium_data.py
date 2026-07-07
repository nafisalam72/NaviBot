"""
stadium_data.py
---------------
Provides mock static stadium data for FIFA World Cup 2026 venues.

This module simulates the data that would ordinarily come from a real-time
IoT/GPS system. It stores section-level details (restrooms, exits, food
stalls, wheelchair accessibility, and live crowd levels) for each of the
primary FIFA 2026 host stadiums.

Assumptions
-----------
- Data is mock/static because real-time GPS feeds are unavailable in this demo.
- Crowd levels (1–10) are simulated; a production system would ingest live
  sensor data from stadium management platforms.
- Sections A–E are representative; real stadiums have many more zones.
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# Type alias for section data
# ---------------------------------------------------------------------------
SectionData = dict[str, object]
StadiumSections = dict[str, SectionData]
StadiumRegistry = dict[str, StadiumSections]

# ---------------------------------------------------------------------------
# FIFA 2026 Host Stadiums – Mock Data
# ---------------------------------------------------------------------------

STADIUMS: StadiumRegistry = {
    "MetLife Stadium": {
        "A": {
            "nearest_restroom": "Restroom Block A1 – Gate 101 (30 m north)",
            "nearest_exit": "Exit Gate 101 – Field Level, North Tunnel",
            "food_stalls": [
                "Hot Dogs & Burgers – Stall A-01",
                "Tacos & Nachos – Stall A-02",
                "Beverages & Snacks – Stall A-03",
            ],
            "wheelchair_accessible": True,
            "crowd_level": 3,
            "alternative_route": (
                "Take Concourse B elevator to Level 2, then proceed south to Gate 201."
            ),
        },
        "B": {
            "nearest_restroom": "Restroom Block B2 – Near Gate 201 (20 m east)",
            "nearest_exit": "Exit Gate 201 – Upper Concourse, East Ramp",
            "food_stalls": [
                "Pizza Corner – Stall B-01",
                "Vegan Delights – Stall B-02",
            ],
            "wheelchair_accessible": True,
            "crowd_level": 7,
            "alternative_route": (
                "Use the West Concourse elevator to Level 1 and exit via Gate 102."
            ),
        },
        "C": {
            "nearest_restroom": "Restroom Block C3 – Gate 301 (15 m west)",
            "nearest_exit": "Exit Gate 301 – South Tunnel, Ground Level",
            "food_stalls": [
                "Sushi Express – Stall C-01",
                "BBQ Grill – Stall C-02",
                "Ice Cream & Desserts – Stall C-03",
            ],
            "wheelchair_accessible": False,
            "crowd_level": 5,
            "alternative_route": (
                "Route through Concourse A is wheelchair-inaccessible here; "
                "use the North accessible ramp via Gate 101."
            ),
        },
        "D": {
            "nearest_restroom": "Restroom Block D4 – Gate 401 (40 m south)",
            "nearest_exit": "Exit Gate 401 – West Concourse, Level 3",
            "food_stalls": [
                "Loaded Fries – Stall D-01",
                "World Cuisine Hub – Stall D-02",
            ],
            "wheelchair_accessible": True,
            "crowd_level": 9,
            "alternative_route": (
                "Section D is critically crowded. Use Section E via internal corridor "
                "and exit through Gate 501 (estimated 5 min detour)."
            ),
        },
        "E": {
            "nearest_restroom": "Restroom Block E5 – Gate 501 (25 m north)",
            "nearest_exit": "Exit Gate 501 – Premium Level, North-West",
            "food_stalls": [
                "Gourmet Sandwiches – Stall E-01",
                "Fresh Juice Bar – Stall E-02",
                "Family Meal Combo – Stall E-03",
            ],
            "wheelchair_accessible": True,
            "crowd_level": 2,
            "alternative_route": (
                "Section E has low crowd density – this is already the recommended route."
            ),
        },
    },
    "AT&T Stadium": {
        "A": {
            "nearest_restroom": "Restroom Block A – Near Gate 1 (25 m)",
            "nearest_exit": "Gate 1 – Main Entrance Level",
            "food_stalls": ["Texas BBQ – Stall A-01", "Burger Ranch – Stall A-02"],
            "wheelchair_accessible": True,
            "crowd_level": 4,
            "alternative_route": "Low crowd – direct route via main concourse.",
        },
        "B": {
            "nearest_restroom": "Restroom Block B – Near Gate 2 (35 m)",
            "nearest_exit": "Gate 2 – East Side",
            "food_stalls": ["Mexican Street Food – Stall B-01", "Hot Dogs – Stall B-02"],
            "wheelchair_accessible": True,
            "crowd_level": 6,
            "alternative_route": "Use Gate 3 via internal west passage to avoid congestion.",
        },
        "C": {
            "nearest_restroom": "Restroom Block C – Near Gate 3 (20 m)",
            "nearest_exit": "Gate 3 – West Side",
            "food_stalls": ["Salad & Wraps – Stall C-01", "Smoothie Stand – Stall C-02"],
            "wheelchair_accessible": True,
            "crowd_level": 8,
            "alternative_route": "Gate 3 is congested. Proceed to Gate 4 via South Concourse.",
        },
        "D": {
            "nearest_restroom": "Restroom Block D – Near Gate 4 (30 m)",
            "nearest_exit": "Gate 4 – South Side",
            "food_stalls": ["Classic Nachos – Stall D-01"],
            "wheelchair_accessible": False,
            "crowd_level": 3,
            "alternative_route": (
                "Section D is not wheelchair accessible. "
                "Use Section A accessible entrance via Gate 1."
            ),
        },
        "E": {
            "nearest_restroom": "Restroom Block E – Near Gate 5 (15 m)",
            "nearest_exit": "Gate 5 – VIP North Level",
            "food_stalls": [
                "Gourmet Bites – Stall E-01",
                "Espresso Bar – Stall E-02",
                "Frozen Desserts – Stall E-03",
            ],
            "wheelchair_accessible": True,
            "crowd_level": 1,
            "alternative_route": "Section E is virtually empty – ideal for low-crowd experience.",
        },
    },
    "SoFi Stadium": {
        "A": {
            "nearest_restroom": "Restroom Hub A – Tunnel 10 (20 m)",
            "nearest_exit": "Exit 10 – North Plaza",
            "food_stalls": ["California Roll Bar – Stall A-01", "Vegan Bites – Stall A-02"],
            "wheelchair_accessible": True,
            "crowd_level": 5,
            "alternative_route": "Moderate crowd – proceed via main north concourse.",
        },
        "B": {
            "nearest_restroom": "Restroom Hub B – Tunnel 20 (30 m)",
            "nearest_exit": "Exit 20 – East Plaza",
            "food_stalls": ["Grill House – Stall B-01", "Asian Fusion – Stall B-02"],
            "wheelchair_accessible": True,
            "crowd_level": 7,
            "alternative_route": "Use Tunnel 10 to Exit 10 to bypass East Plaza congestion.",
        },
        "C": {
            "nearest_restroom": "Restroom Hub C – Tunnel 30 (25 m)",
            "nearest_exit": "Exit 30 – South Plaza",
            "food_stalls": ["Churros Stand – Stall C-01", "Fresh Lemonade – Stall C-02"],
            "wheelchair_accessible": True,
            "crowd_level": 4,
            "alternative_route": "Low-moderate crowd – south plaza route is clear.",
        },
        "D": {
            "nearest_restroom": "Restroom Hub D – Tunnel 40 (40 m)",
            "nearest_exit": "Exit 40 – West Plaza",
            "food_stalls": ["Seafood Corner – Stall D-01"],
            "wheelchair_accessible": False,
            "crowd_level": 9,
            "alternative_route": (
                "CRITICAL: Section D is very crowded and not wheelchair accessible. "
                "Use Section E via internal corridor for both concerns."
            ),
        },
        "E": {
            "nearest_restroom": "Restroom Hub E – Tunnel 50 (10 m)",
            "nearest_exit": "Exit 50 – Premium West",
            "food_stalls": [
                "Artisan Sandwiches – Stall E-01",
                "Craft Beer Bar – Stall E-02",
            ],
            "wheelchair_accessible": True,
            "crowd_level": 2,
            "alternative_route": "Section E is lightly populated – ideal diversion route.",
        },
    },
}

# ---------------------------------------------------------------------------
# Crowd-level thresholds
# ---------------------------------------------------------------------------

CROWD_THRESHOLDS: dict[str, tuple[int, int]] = {
    "Low": (1, 3),
    "Medium": (4, 6),
    "High": (7, 10),
}

# Default stadium used when none is specified
DEFAULT_STADIUM: str = "MetLife Stadium"


def get_stadium_names() -> list[str]:
    """Return a list of all available stadium names.

    Returns
    -------
    list[str]
        Sorted list of stadium identifiers present in the STADIUMS registry.
    """
    return sorted(STADIUMS.keys())


def get_section_data(stadium: str, section: str) -> SectionData | None:
    """Retrieve data for a specific section within a stadium.

    Parameters
    ----------
    stadium : str
        The name of the stadium (case-sensitive, must match STADIUMS keys).
    section : str
        The section identifier (e.g. ``"A"``, ``"B"``).

    Returns
    -------
    SectionData or None
        A dictionary of section attributes, or ``None`` if the stadium or
        section does not exist in the registry.
    """
    stadium_sections: StadiumSections | None = STADIUMS.get(stadium)
    if stadium_sections is None:
        return None
    return stadium_sections.get(section.upper())


def categorise_crowd(crowd_level: int) -> str:
    """Convert a numeric crowd level (1–10) to a human-readable category.

    Parameters
    ----------
    crowd_level : int
        A numeric value between 1 (empty) and 10 (completely packed).

    Returns
    -------
    str
        One of ``"Low"``, ``"Medium"``, or ``"High"``.

    Raises
    ------
    ValueError
        If ``crowd_level`` is outside the 1–10 range.
    """
    if not 1 <= crowd_level <= 10:
        raise ValueError(f"crowd_level must be between 1 and 10, got {crowd_level}.")
    for label, (low, high) in CROWD_THRESHOLDS.items():
        if low <= crowd_level <= high:
            return label
    return "Unknown"  # Unreachable, but satisfies type checkers
