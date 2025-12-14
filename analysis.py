import numpy as np
from scipy.stats import (
    ks_2samp,
    cramervonmises_2samp,
    anderson_ksamp,
    mannwhitneyu,
    ttest_ind,
    kruskal,
    fisher_exact,
    chi2_contingency,
    levene,
)

def _is_constant(x):
    x = np.asarray(x, dtype=float)
    return np.allclose(x, x[0])

def safe_welch_ttest(x, y):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if _is_constant(x) and _is_constant(y) and np.isclose(x[0], y[0]):
        return ("Welch t-test", np.nan, 1.0, "degenerate(all equal)")
    stat, p = ttest_ind(x, y, equal_var=False, nan_policy="omit")
    if np.isnan(p):
        return ("Welch t-test", float(stat), 1.0, "degenerate(var=0)")
    return ("Welch t-test", float(stat), float(p), "ok")

def safe_mannwhitney(x, y, alternative):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if _is_constant(x) and _is_constant(y) and np.isclose(x[0], y[0]):
        return ("Mann–Whitney", np.nan, 1.0, "degenerate(all equal)")
    stat, p = mannwhitneyu(x, y, alternative=alternative)
    return ("Mann–Whitney", float(stat), float(p), "ok")

def safe_kruskal(*groups):
    groups = [np.asarray(g, dtype=float) for g in groups]
    flat = np.concatenate(groups)
    if _is_constant(flat):
        return ("Kruskal", np.nan, 1.0, "degenerate(all equal)")
    stat, p = kruskal(*groups, nan_policy="omit")
    return ("Kruskal", float(stat), float(p), "ok")

def safe_chi2_or_fisher(a_binary, b_binary):
    a = np.asarray(a_binary, dtype=int)
    b = np.asarray(b_binary, dtype=int)
    A1, A0 = int((a == 1).sum()), int((a == 0).sum())
    B1, B0 = int((b == 1).sum()), int((b == 0).sum())
    table = np.array([[A1, A0], [B1, B0]], dtype=int)

    if np.any(table.sum(axis=0) == 0) or np.any(table.sum(axis=1) == 0):
        return ("Binary test", np.nan, 1.0, "degenerate")

    if np.any(table == 0):
        odds, p = fisher_exact(table, alternative="two-sided")
        return ("Fisher exact", float(odds), float(p), "ok")

    chi2, p, _, exp = chi2_contingency(table)
    if np.any(exp == 0):
        odds, p = fisher_exact(table, alternative="two-sided")
        return ("Fisher exact", float(odds), float(p), "fallback(exp=0)")
    return ("Chi-square", float(chi2), float(p), "ok")

def run_hypothesis_tests(results_df, return_paths):
    # H1
    short_all = np.concatenate(return_paths["short_gamma"])
    long_all = np.concatenate(return_paths["long_gamma"])

    out = {}
    out["H1_KS"] = ks_2samp(short_all, long_all)
    out["H1_CvM"] = cramervonmises_2samp(short_all, long_all)
    out["H1_AD"] = anderson_ksamp([short_all, long_all])

    # H2 crashes_z
    sg = results_df.loc[results_df.scenario == "short_gamma", "crashes_z"].values
    lg = results_df.loc[results_df.scenario == "long_gamma", "crashes_z"].values
    nh = results_df.loc[results_df.scenario == "no_hedge", "crashes_z"].values

    out["H2_means"] = (float(sg.mean()), float(lg.mean()), float(nh.mean()))
    out["H2_MW_short_gt_long"] = safe_mannwhitney(sg, lg, alternative="greater")
    out["H2_Welch"] = safe_welch_ttest(sg, lg)
    out["H2_Kruskal_3"] = safe_kruskal(sg, lg, nh)

    sg_any = (sg > 0).astype(int)
    lg_any = (lg > 0).astype(int)
    out["H2_any_crash_binary"] = safe_chi2_or_fisher(sg_any, lg_any)

    # H3 drawdown
    sg_mdd = results_df.loc[results_df.scenario == "short_gamma", "max_drawdown"].values
    lg_mdd = results_df.loc[results_df.scenario == "long_gamma", "max_drawdown"].values
    nh_mdd = results_df.loc[results_df.scenario == "no_hedge", "max_drawdown"].values

    out["H3_means"] = (float(sg_mdd.mean()), float(lg_mdd.mean()), float(nh_mdd.mean()))
    out["H3_MW_short_lt_long"] = safe_mannwhitney(sg_mdd, lg_mdd, alternative="less")
    out["H3_Welch"] = safe_welch_ttest(sg_mdd, lg_mdd)
    out["H3_KS"] = ks_2samp(sg_mdd, lg_mdd)
    out["H3_Kruskal_3"] = safe_kruskal(sg_mdd, lg_mdd, nh_mdd)
    out["H3_Levene_var"] = levene(sg_mdd, lg_mdd, center="median")

    return out
