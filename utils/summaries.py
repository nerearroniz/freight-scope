import pandas as pd

# Trend analysis function

def analyse_trends(
    df,
    category,
    variables,
    year_col="Year",
    sort_by=None,
    sort_year=None,
    ascending=False,
    decimals=1,
    inter_perc=True,
    show_growth=True,
):
    """
    Print a visual text-based trend summary.

    For each category and year:

        Count row:
            Number of observations in the category/year.
            If inter_perc=True, also shows the percentage
            of total observations for that year.

        Growth row:
            Growth of Count compared with the previous year.
            Controlled by show_growth.

        Variable row:
            SUM of the variable.
            If inter_perc=True, also shows the percentage
            of the total sum for that variable/year.

        Mean ± SD row:
            Mean ± SD for the variable.

            Only shown when there is more than one
            observation for the category/year.

        Q5–Q95 row:
            5th–95th percentile range.

            Only shown when there is more than one
            observation for the category/year.

        Variable Growth row:
            Growth of the variable SUM compared with
            the previous year.
            Controlled by show_growth.

    Parameters
    ----------
    df : pandas.DataFrame
        Input dataframe.

    category : str
        Column defining the categories/groups.

    variables : list
        Variables to analyse.

    year_col : str, default="Year"
        Column containing the year.

    sort_by : str or None, default=None
        Variable used to sort categories based on SUM.

    sort_year : int or None, default=None
        Year used for sorting. If None, the mean SUM
        across years is used.

    ascending : bool, default=False
        Sorting direction.

    decimals : int, default=1
        Number of decimals for numerical values.

    inter_perc : bool, default=True
        If True, show percentages for both Count and
        variable SUM.

    show_growth : bool, default=True
        If True, show Growth for Count and all variables.
    """

    data = df.copy()

    # =========================================================
    # CHECK COLUMNS
    # =========================================================

    required = [category, year_col] + variables

    missing_columns = [
        col
        for col in required
        if col not in data.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Columns not found in dataframe: {missing_columns}"
        )

    # =========================================================
    # CHECK NaNs
    # =========================================================

    missing = data[required].isna().sum()
    missing = missing[missing > 0]

    if len(missing) > 0:

        print("\n⚠️ Missing values detected:")

        for col, n in missing.items():
            print(f"   {col}: {n:,}")

        raise ValueError(
            "\nNaN values detected. Please check your data."
        )

    # =========================================================
    # CONVERT VARIABLES TO NUMERIC
    # =========================================================

    for var in variables:

        data[var] = pd.to_numeric(
            data[var],
            errors="raise"
        )

    # =========================================================
    # YEARS
    # =========================================================

    years = sorted(
        data[year_col].unique()
    )

    # =========================================================
    # COUNT
    # =========================================================

    # Count is common to all variables, so calculate it once.

    counts = (
        data
        .groupby([category, year_col])
        .size()
        .unstack(fill_value=0)
    )

    # Percentage of total observations for each year.

    total_counts = counts.sum(axis=0)

    count_percentages = (
        counts
        .div(total_counts, axis=1)
        * 100
    )

    # =========================================================
    # VARIABLE SUMMARIES
    # =========================================================

    summaries = {}

    for var in variables:

        grouped = data.groupby(
            [category, year_col]
        )[var]

        # -----------------------------------------------------
        # SUM
        # -----------------------------------------------------

        sums = (
            grouped
            .sum()
            .unstack(fill_value=0)
        )

        # -----------------------------------------------------
        # PERCENTAGE OF TOTAL SUM
        # -----------------------------------------------------

        total_sums = sums.sum(axis=0)

        percentages = (
            sums
            .div(total_sums, axis=1)
            * 100
        )

        # -----------------------------------------------------
        # MEAN
        # -----------------------------------------------------

        mean = (
            grouped
            .mean()
            .unstack()
        )

        # -----------------------------------------------------
        # STANDARD DEVIATION
        # -----------------------------------------------------

        sd = (
            grouped
            .std()
            .unstack()
        )

        # -----------------------------------------------------
        # 5TH PERCENTILE
        # -----------------------------------------------------

        q5 = (
            grouped
            .quantile(0.05)
            .unstack()
        )

        # -----------------------------------------------------
        # 95TH PERCENTILE
        # -----------------------------------------------------

        q95 = (
            grouped
            .quantile(0.95)
            .unstack()
        )

        summaries[var] = {
            "sum": sums,
            "percentage": percentages,
            "mean": mean,
            "sd": sd,
            "q5": q5,
            "q95": q95,
        }

    # =========================================================
    # SORTING
    # =========================================================

    if sort_by is None:

        sorted_categories = sorted(
            data[category].unique()
        )

    elif sort_by in variables:

        # Sort according to SUM of selected variable.

        sort_values = summaries[sort_by]["sum"]

        if sort_year is None:

            # Mean SUM across years.
            sort_values = sort_values.mean(axis=1)

        else:

            if sort_year not in sort_values.columns:
                raise ValueError(
                    f"sort_year {sort_year} not found in years: "
                    f"{list(sort_values.columns)}"
                )

            sort_values = sort_values[sort_year]

        sorted_categories = (
            sort_values
            .sort_values(
                ascending=ascending
            )
            .index
        )

    else:

        raise ValueError(
            f"sort_by must be None or one of {variables}"
        )

    # =========================================================
    # FORMATTING
    # =========================================================

    width = max(
        70,
        25 + 30 * len(years)
    )

    def format_growth(value):

        if pd.isna(value):
            return "—"

        if value > 0:
            return f"🟢↑ {value:.1f}%"

        elif value < 0:
            return f"🔴↓ {abs(value):.1f}%"

        else:
            return "⚪→ 0.0%"

    # =========================================================
    # PRINT CATEGORIES
    # =========================================================

    for idx in sorted_categories:

        print("\n" + "═" * width)
        print(f"  {str(idx).upper()}")
        print("═" * width)

        # -----------------------------------------------------
        # HEADER
        # -----------------------------------------------------

        print(
            f"{'':25}",
            end=""
        )

        for year in years:

            print(
                f"{year:>30}",
                end=""
            )

        print()

        print("─" * width)

        # =====================================================
        # COUNT
        # =====================================================

        category_counts = counts.loc[idx]

        category_count_percentages = (
            count_percentages.loc[idx]
        )

        print(
            f"{'Count':25}",
            end=""
        )

        for year in years:

            count_value = category_counts.loc[year]

            if inter_perc:

                percentage_value = (
                    category_count_percentages.loc[year]
                )

                text = (
                    f"{int(count_value):,} "
                    f"({percentage_value:.1f}%)"
                )

            else:

                text = f"{int(count_value):,}"

            print(
                f"{text:>30}",
                end=""
            )

        print()

        # =====================================================
        # COUNT GROWTH
        # =====================================================

        if show_growth:

            count_growth = (
                category_counts
                .pct_change()
                * 100
            )

            print(
                f"{'  Growth':25}",
                end=""
            )

            for year in years:

                text = format_growth(
                    count_growth.loc[year]
                )

                print(
                    f"{text:>30}",
                    end=""
                )

            print()

        # -----------------------------------------------------
        # SEPARATOR
        # -----------------------------------------------------

        print("─" * width)

        # =====================================================
        # VARIABLES
        # =====================================================

        for var in variables:

            sums = summaries[var]["sum"].loc[idx]

            percentages = (
                summaries[var]["percentage"].loc[idx]
            )

            # -------------------------------------------------
            # SUM + PERCENTAGE
            # -------------------------------------------------

            print(
                f"{var:25}",
                end=""
            )

            for year in years:

                sum_value = sums.loc[year]

                percentage_value = (
                    percentages.loc[year]
                )

                if inter_perc:

                    text = (
                        f"{sum_value:.{decimals}f} "
                        f"({percentage_value:.1f}%)"
                    )

                else:

                    text = (
                        f"{sum_value:.{decimals}f}"
                    )

                print(
                    f"{text:>30}",
                    end=""
                )

            print()

            # -------------------------------------------------
            # STATISTICS
            # -------------------------------------------------

            mean = summaries[var]["mean"].loc[idx]
            sd = summaries[var]["sd"].loc[idx]
            q5 = summaries[var]["q5"].loc[idx]
            q95 = summaries[var]["q95"].loc[idx]

            # Only show statistics if there is at least
            # one year with more than one observation.

            has_multiple = (
                category_counts > 1
            ).any()

            if has_multiple:

                # =============================================
                # MEAN ± SD
                # =============================================

                print(
                    f"{'  Mean ± SD':25}",
                    end=""
                )

                for year in years:

                    n = category_counts.loc[year]

                    if n > 1:

                        mean_value = mean.loc[year]
                        sd_value = sd.loc[year]

                        text = (
                            f"{mean_value:.{decimals}f} ± "
                            f"{sd_value:.{decimals}f}"
                        )

                    else:

                        text = "—"

                    print(
                        f"{text:>30}",
                        end=""
                    )

                print()

                # =============================================
                # Q5–Q95
                # =============================================

                print(
                    f"{'  Q5–Q95':25}",
                    end=""
                )

                for year in years:

                    n = category_counts.loc[year]

                    if n > 1:

                        q5_value = q5.loc[year]
                        q95_value = q95.loc[year]

                        text = (
                            f"{q5_value:.{decimals}f}–"
                            f"{q95_value:.{decimals}f}"
                        )

                    else:

                        text = "—"

                    print(
                        f"{text:>30}",
                        end=""
                    )

                print()

            # -------------------------------------------------
            # VARIABLE GROWTH
            # -------------------------------------------------

            if show_growth:

                growth = (
                    sums
                    .pct_change()
                    * 100
                )

                print(
                    f"{'  Growth':25}",
                    end=""
                )

                for year in years:

                    text = format_growth(
                        growth.loc[year]
                    )

                    print(
                        f"{text:>30}",
                        end=""
                    )

                print()

        # =====================================================
        # END CATEGORY
        # =====================================================

        print("═" * width)




# Summary function

def grouped_summary(
    df,
    group_col,
    variables,
    count_name="Count",
    decimals=0,
    dropna=False,
    sort_by="Count",
    ascending=False
):
    """
    Generic grouped summary.

    Parameters
    ----------
    df : DataFrame
        Input data, one row per establishment.

    group_col : str
        Variable used to define groups, e.g. 'Sector', 'CNAE',
        or 'MUNICIPIO'.

    variables : list of str
        Numeric variables to summarize.

    count_name : str
        Name of the establishment count column.

    decimals : int
        Number of decimals for statistics.

    dropna : bool
        If False, NaNs raise an error.
        If True, NaNs are reported and dropped.

    sort_by : str
        Variable used to sort the output.
        Use "Count" or any variable in `variables`.

    ascending : bool
        Sort ascending if True, descending if False.
    """

    data = df.copy()

    # ---------------------------------------------------------
    # Check missing values
    # ---------------------------------------------------------

    columns_to_check = [group_col] + variables

    missing = data[columns_to_check].isna().sum()
    missing = missing[missing > 0]

    if len(missing) > 0:

        print("\n⚠️ Missing values detected:")

        for column, n in missing.items():
            print(f"   {column}: {n:,} missing values")

        if not dropna:
            raise ValueError(
                "\nNaN values detected. "
                "Set dropna=True if you want to remove them."
            )

        print("\n⚠️ Dropping rows containing NaN values...")

        before = len(data)
        data = data.dropna(subset=columns_to_check)
        after = len(data)

        print(f"   Rows before: {before:,}")
        print(f"   Rows after:  {after:,}")
        print(f"   Rows dropped: {before - after:,}")

    # ---------------------------------------------------------
    # Count
    # ---------------------------------------------------------

    counts = data.groupby(group_col).size()

    result = pd.DataFrame(index=counts.index)

    count_pct = counts / counts.sum() * 100

    result[f"{count_name} (%)"] = [
        f"{int(n):,} ({p:.1f}%)"
        for n, p in zip(counts, count_pct)
    ]

    # ---------------------------------------------------------
    # Variables
    # ---------------------------------------------------------

    for var in variables:

        data[var] = pd.to_numeric(data[var], errors="coerce")

        grouped = data.groupby(group_col)[var]

        total = grouped.sum()
        mean = grouped.mean()
        sd = grouped.std()
        q05 = grouped.quantile(0.05)
        q95 = grouped.quantile(0.95)

        total_pct = total / total.sum() * 100

        result[f"{var} (%)"] = [
            f"{value:,.{decimals}f} ({pct:.1f}%)"
            for value, pct in zip(total, total_pct)
        ]

        result[
            f"Mean {var} per establishment ± SD (q5–q95)"
        ] = [
            f"{m:.{decimals}f} ± {s:.{decimals}f} "
            f"({q1:.{decimals}f}–{q9:.{decimals}f})"
            for m, s, q1, q9 in zip(
                mean, sd, q05, q95
            )
        ]

    # ---------------------------------------------------------
    # Sort
    # ---------------------------------------------------------

    if sort_by == "Count":
        sort_values = counts

    elif sort_by in variables:
        sort_values = data.groupby(group_col)[sort_by].sum()

    else:
        raise ValueError(
            f"sort_by must be 'Count' or one of {variables}"
        )

    result = result.loc[
        sort_values.sort_values(
            ascending=ascending
        ).index
    ]

    result.index.name = group_col

    return result
