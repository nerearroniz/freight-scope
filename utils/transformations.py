
import numpy as np
import pandas as pd

MUNICIPALITY_COL = 'Municipality'
YEAR_COL = 'Year'
VARIABLE_COL = 'Variable'
VALUE_COL = 'Value'

def transform_relative_first(
    group,
    baseline_year,
    year_col=YEAR_COL,
    value_col=VALUE_COL
):
    """
    Calculate proportional change relative to a fixed baseline year.

    Formula:
        X_t / X_baseline - 1

    The baseline is the same calendar year for all municipalities
    within each variable.
    """

    group = group.sort_values(year_col).copy()

    baseline = group.loc[
        group[year_col] == baseline_year,
        value_col
    ]

    # No unique baseline available
    if len(baseline) != 1:
        group['Value_transformed'] = np.nan
        return group

    baseline = baseline.iloc[0]

    if pd.isna(baseline) or baseline == 0:
        group['Value_transformed'] = np.nan
        return group

    group['Value_transformed'] = (
        group[value_col] / baseline
    ) - 1

    return group


def transform_log(
    group,
    value_col=VALUE_COL
):
    """
    Apply log transformation to each observation.

    Formula:
        log(X)

    Only positive values are transformed.
    """

    group = group.copy()

    group['Value_transformed'] = np.nan

    mask = group[value_col] > 0

    group.loc[mask, 'Value_transformed'] = np.log(
        group.loc[mask, value_col]
    )

    return group


def transform_log_change(
    group,
    baseline_year,
    year_col=YEAR_COL,
    value_col=VALUE_COL
):
    """
    Calculate logarithmic change relative to a fixed baseline year.

    Formula:
        log(X_t / X_baseline)

    Only positive values can be transformed.
    """

    group = group.sort_values(year_col).copy()

    baseline = group.loc[
        group[year_col] == baseline_year,
        value_col
    ]

    # No unique baseline available
    if len(baseline) != 1:
        group['Value_transformed'] = np.nan
        return group

    baseline = baseline.iloc[0]

    if pd.isna(baseline) or baseline <= 0:
        group['Value_transformed'] = np.nan
        return group

    group['Value_transformed'] = np.nan

    mask = group[value_col] > 0

    group.loc[mask, 'Value_transformed'] = np.log(
        group.loc[mask, value_col] / baseline
    )

    return group



TRANSFORMATIONS = {
    'relative_first': transform_relative_first,
    'log': transform_log,
    'log_change': transform_log_change,
}


def transform_data(
    df,
    transformation='relative_first',
    baseline_year=None,
    municipality_col=MUNICIPALITY_COL,
    year_col=YEAR_COL,
    variable_col=VARIABLE_COL,
    value_col=VALUE_COL
):
    """
    Transform each municipality-variable time series.

    Parameters
    ----------
    transformation : str
        'relative_first', 'log', or 'log_change'

    baseline_year : int or None
        Required for 'relative_first' and 'log_change'.

    Returns
    -------
    pd.DataFrame
        Original long-format dataframe with an additional
        'Value_transformed' column.
    """

    valid_transformations = {
        'relative_first',
        'log',
        'log_change'
    }

    if transformation not in valid_transformations:
        raise ValueError(
            f"Unknown transformation '{transformation}'. "
            f"Choose from: {valid_transformations}"
        )

    if (
        transformation in {'relative_first', 'log_change'}
        and baseline_year is None
    ):
        raise ValueError(
            f"'{transformation}' requires a baseline_year."
        )

    df = df.copy()

    # Make sure years are ordered before transformation
    df = df.sort_values(
        [municipality_col, variable_col, year_col]
    )

    if transformation == 'relative_first':

        df = (
            df
            .groupby(
                [municipality_col, variable_col],
                group_keys=False
            )
            .apply(
                transform_relative_first,
                baseline_year=baseline_year,
                year_col=year_col,
                value_col=value_col
            )
        )

    elif transformation == 'log':

        df = (
            df
            .groupby(
                [municipality_col, variable_col],
                group_keys=False
            )
            .apply(
                transform_log,
                value_col=value_col
            )
        )

    elif transformation == 'log_change':

        df = (
            df
            .groupby(
                [municipality_col, variable_col],
                group_keys=False
            )
            .apply(
                transform_log_change,
                baseline_year=baseline_year,
                year_col=year_col,
                value_col=value_col
            )
        )

    return df.reset_index(drop=True)

