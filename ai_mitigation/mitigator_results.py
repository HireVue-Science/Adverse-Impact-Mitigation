import pandas as pd

from .demo_utils import convert_df_to_demo_dicts
from .utils import calc_cohens_d_pairwise, calc_model_score, get_model_metric


class MitigatorResults:
    """Base class that has methods to calculate results. This is intended to be inherited from
    the predictor removal mitigation class and the multi penalty optimization class.
    """

    def calc_results(self, pairs):
        """calculate bias/validity tradeoff"""
        df_results = pd.DataFrame()
        metric = get_model_metric(self.base_model)

        demo_dicts = convert_df_to_demo_dicts(self.demo)

        for col in self.df_ypred.columns:
            y_pred = self.df_ypred[col].values
            score, _ci = calc_model_score(self.y, y_pred, metric)
            df_results.loc[col, metric] = score

            for demo_dict in demo_dicts.values():
                d_pairwise = calc_cohens_d_pairwise(y_pred, demo_dict).d
                for pair in pairs:
                    if pair not in d_pairwise:
                        continue
                    pair_str = "-".join(pair)
                    df_results.loc[col, pair_str] = d_pairwise[pair]
        pair_strs = ["-".join(pair) for pair in pairs]
        df_results = df_results[[metric] + pair_strs]
        return df_results

    def calc_results_with_ci(self, pairs):
        """calculate bias/validity tradeoff"""
        metric = get_model_metric(self.base_model)
        metric_ci = metric + " (CI)"
        pair_strs = ["-".join(pair) for pair in pairs]
        tuples = [("", metric), ("", metric_ci)]
        for pair_str in pair_strs:
            tuples += [(pair_str, "d"), (pair_str, "d (CI)")]
        columns = pd.MultiIndex.from_tuples(tuples)

        df_results = pd.DataFrame(columns=columns, index=self.df_ypred.columns, dtype="object")

        demo_dicts = convert_df_to_demo_dicts(self.demo)

        for col in self.df_ypred.columns:
            y_pred = self.df_ypred[col].values
            score, ci = calc_model_score(self.y, y_pred, metric=metric)
            df_results.loc[col, ("", metric)] = round(score, 3)
            df_results.at[col, ("", metric_ci)] = [round(ci[0], 3), round(ci[1], 3)]

            for demo_dict in demo_dicts.values():
                d_info = calc_cohens_d_pairwise(y_pred, demo_dict)
                for pair in pairs:
                    if pair not in d_info.d:
                        continue
                    pair_str = "-".join(pair)
                    df_results.loc[col, (pair_str, "d")] = round(d_info.d[pair], 3)
                    ci = d_info.d_ci[pair]
                    ci = [round(ci[0], 3), round(ci[1], 3)]
                    df_results.at[col, (pair_str, "d (CI)")] = ci

        # the metric and 'd' columns should be numeric
        for col in df_results.columns[::2]:
            df_results[col] = pd.to_numeric(df_results[col])

        return df_results
