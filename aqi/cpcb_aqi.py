"""
CPCB-compatible AQI calculation for Mumbai AQI V3.

Input:
    Raw pollutant concentrations.

Expected units:
    PM2.5 -> µg/m³
    PM10  -> µg/m³
    NO2   -> µg/m³
    SO2   -> µg/m³
    O3    -> µg/m³

Output:
    Individual pollutant sub-indices
    Overall AQI

IMPORTANT:
    The pollutant input values must be concentrations,
    NOT AQI sub-indices.
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple


# ============================================================
# CPCB BREAKPOINT TABLE
# ============================================================
#
# Each tuple:
#
# (
#     concentration_low,
#     concentration_high,
#     aqi_low,
#     aqi_high
# )
#
# Formula:
#
# Ip = ((I_high - I_low) / (B_high - B_low))
#      * (Cp - B_low) + I_low
#
# ============================================================

CPCB_BREAKPOINTS = {

    "PM2.5": [
        (0.0, 30.0, 0, 50),
        (31.0, 60.0, 51, 100),
        (61.0, 90.0, 101, 200),
        (91.0, 120.0, 201, 300),
        (121.0, 250.0, 301, 400),
        (250.0, float("inf"), 401, 500),
    ],

    "PM10": [
        (0.0, 50.0, 0, 50),
        (51.0, 100.0, 51, 100),
        (101.0, 250.0, 101, 200),
        (251.0, 350.0, 201, 300),
        (351.0, 430.0, 301, 400),
        (430.0, float("inf"), 401, 500),
    ],

    "NO2": [
        (0.0, 40.0, 0, 50),
        (41.0, 80.0, 51, 100),
        (81.0, 180.0, 101, 200),
        (181.0, 280.0, 201, 300),
        (281.0, 400.0, 301, 400),
        (400.0, float("inf"), 401, 500),
    ],

    "SO2": [
        (0.0, 40.0, 0, 50),
        (41.0, 80.0, 51, 100),
        (81.0, 380.0, 101, 200),
        (381.0, 800.0, 201, 300),
        (801.0, 1600.0, 301, 400),
        (1600.0, float("inf"), 401, 500),
    ],

    "O3": [
        (0.0, 50.0, 0, 50),
        (51.0, 100.0, 51, 100),
        (101.0, 168.0, 101, 200),
        (169.0, 208.0, 201, 300),
        (209.0, 748.0, 301, 400),
        (748.0, float("inf"), 401, 500),
    ],
}


# ============================================================
# REQUIRED POLLUTANT COUNT
# ============================================================

MIN_REQUIRED_POLLUTANTS = 3


# ============================================================
# UTILITY
# ============================================================

def _to_float(value: Any) -> Optional[float]:
    """
    Safely convert input to float.
    """

    if value is None:
        return None

    try:
        value = float(value)
    except (TypeError, ValueError):
        return None

    if value < 0:
        return None

    return value


# ============================================================
# SINGLE POLLUTANT SUB-INDEX
# ============================================================

def calculate_sub_index(
    pollutant: str,
    concentration: Any,
) -> Optional[float]:
    """
    Calculate CPCB AQI sub-index for one pollutant.

    Parameters
    ----------
    pollutant:
        PM2.5, PM10, NO2, SO2, or O3

    concentration:
        Raw pollutant concentration in µg/m³.

    Returns
    -------
    float or None
        AQI sub-index.
    """

    if pollutant not in CPCB_BREAKPOINTS:
        raise ValueError(
            f"Unsupported pollutant: {pollutant}"
        )

    concentration = _to_float(concentration)

    if concentration is None:
        return None

    breakpoints = CPCB_BREAKPOINTS[pollutant]

    for (
        b_low,
        b_high,
        i_low,
        i_high,
    ) in breakpoints:

        # Last interval extends to infinity.
        if concentration >= b_low and concentration <= b_high:

            # Avoid division by zero.
            if b_high == b_low:
                return float(i_low)

            # For infinite upper interval,
            # cap the concentration at the upper
            # AQI boundary for our 0-500 model.
            if b_high == float("inf"):
                return 500.0

            sub_index = (
                (
                    (i_high - i_low)
                    / (b_high - b_low)
                )
                * (concentration - b_low)
                + i_low
            )

            return round(sub_index, 2)

    return None


# ============================================================
# CALCULATE ALL SUB-INDICES
# ============================================================

def calculate_sub_indices(
    pollutants: Dict[str, Any]
) -> Dict[str, Optional[float]]:
    """
    Calculate sub-index for every supported pollutant.
    """

    return {
        pollutant: calculate_sub_index(
            pollutant,
            pollutants.get(pollutant),
        )
        for pollutant in CPCB_BREAKPOINTS
    }


# ============================================================
# CALCULATE OVERALL AQI
# ============================================================

def calculate_cpcb_aqi(
    pollutants: Dict[str, Any]
) -> Tuple[Optional[float], Dict[str, Optional[float]]]:
    """
    Calculate overall CPCB AQI.

    Rules used here:
    - At least 3 valid pollutants are required.
    - PM2.5 or PM10 must be present.
    - Overall AQI = maximum valid pollutant sub-index.

    Returns:
        (
            overall_aqi,
            sub_indices
        )
    """

    sub_indices = calculate_sub_indices(pollutants)

    valid_sub_indices = {
        pollutant: value
        for pollutant, value in sub_indices.items()
        if value is not None
    }

    # Need at least 3 pollutants.
    if len(valid_sub_indices) < MIN_REQUIRED_POLLUTANTS:
        return None, sub_indices

    # CPCB AQI should include particulate matter.
    particulate_available = (
        sub_indices.get("PM2.5") is not None
        or sub_indices.get("PM10") is not None
    )

    if not particulate_available:
        return None, sub_indices

    overall_aqi = max(valid_sub_indices.values())

    return round(overall_aqi, 2), sub_indices


# ============================================================
# CONVENIENCE FUNCTION
# ============================================================

def calculate_aqi_details(
    pollutants: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Return a complete AQI result suitable for
    the rest of the pipeline.
    """

    aqi, sub_indices = calculate_cpcb_aqi(
        pollutants
    )

    if aqi is None:
        category = None
    elif aqi <= 50:
        category = "Good"
    elif aqi <= 100:
        category = "Satisfactory"
    elif aqi <= 200:
        category = "Moderate"
    elif aqi <= 300:
        category = "Poor"
    elif aqi <= 400:
        category = "Very Poor"
    else:
        category = "Severe"

    return {
        "AQI": aqi,
        "category": category,
        "sub_indices": sub_indices,
    }


# ============================================================
# TEST FUNCTION
# ============================================================

if __name__ == "__main__":

    sample = {
        "PM2.5": 27.8,
        "PM10": 12.1,
        "NO2": 0.06,
        "SO2": 0.26,
        "O3": 36.43,
    }

    result = calculate_aqi_details(sample)

    print("\n====================================")
    print("CPCB AQI TEST")
    print("====================================")

    print("\nRaw pollutants:")

    for key, value in sample.items():
        print(f"  {key}: {value}")

    print("\nSub-indices:")

    for key, value in result["sub_indices"].items():
        print(f"  {key}: {value}")

    print("\nOverall AQI:")
    print(f"  {result['AQI']}")

    print("\nCategory:")
    print(f"  {result['category']}")

    print("====================================")