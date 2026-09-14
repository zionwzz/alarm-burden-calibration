#!/usr/bin/env python3
"""Generate the simulation tables and findings paragraphs from the saved simulation outputs.

Reads `simulation/results/` (written by `run_submission_simulations.py`) and writes three LaTeX
fragments to the generated-tables directory (`alarmreplay/paths.py`, ALARM_TABLES): the main-text
table of principal contrasts, the supplement's data-generating laws, complete comparisons,
diagnostics and patient-count tables with their prose, and the main-text findings paragraphs. The
simulation keys its scenarios A, B, C, D, E, F, G, I, J; everything printed here carries the
manuscript's letters A to I through `alarmreplay/scenarios.py`. The manuscript's own copies of
these fragments were edited for the final text; the numbers are the ones generated here.
"""
from __future__ import annotations
import argparse
import json
import os
import sys
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from alarmreplay.paths import TABLES  # noqa: E402
from alarmreplay.scenarios import PAPER_LETTER as L, PAPER_ORDER, SHORT_NAME as SHORT  # noqa: E402

RES=ROOT/'simulation/results'
OUT=Path(TABLES)
METHODS={'replay':'Replay','lrc_d':'Duration-only','lrc_dz_pre':'Feature-augmented',
         'lrc_dz':'Feature map selected in sample','baseline_oracle':'Baseline-reward oracle',
         'offset':'Fixed-offset comparator'}
ORDER=['A','B','J','E','F']          # simulation keys of the main-text table, in its row order
ALL=PAPER_ORDER                      # simulation keys in the manuscript's order

def f(x,nd=2):return '---' if x is None else f'{x:.{nd}f}'

def lets(keys):
    """Manuscript letters of simulation keys, sorted, as 'A, B, C'."""
    return ', '.join(sorted(L[k] for k in keys))


def comparisons_prose(data):
    holds=[sc for sc in ALL if sc not in ('E','F') and not (sc in ('B','C','J'))]   # both calibrated estimators unbiased
    spread=max(abs(data[sc][k]['esd']-data[sc]['baseline_oracle']['esd']) for sc in holds for k in ('lrc_d','lrc_dz_pre'))
    inmap=max(abs(data[sc]['lrc_dz']['bias']-data[sc]['lrc_dz_pre']['bias']) for sc in ALL)
    ef=[(data[sc]['lrc_dz_pre']['bias'],data[sc]['lrc_dz_pre']['esd']) for sc in ('E','F')]
    return (f"Table~\\ref{{stab:simfull}} gives bias, empirical standard deviation, root mean squared error, coverage and the Monte Carlo standard error of the bias for all six estimators in every scenario; Figure~\\ref{{sfig:simreps}} shows the replicate distributions behind it. Three features are visible there that the summary statistics compress. Wherever the calibrated estimators are unbiased ({lets(holds)}), their empirical standard deviation is within {spread:.2f} points of the baseline-reward oracle's, so nothing is lost by estimating the reward instead of knowing it. The map selected in sample reproduces the independently selected map almost replicate by replicate: the two biases differ by at most {inmap:.2f} points in any scenario. In {L['E']} and {L['F']} the whole distribution of every estimator moves away from zero by several times its width: the feature-augmented bias is {ef[0][0]:.2f} points in {L['E']} and {ef[1][0]:.2f} in {L['F']} against empirical standard deviations of {ef[0][1]:.2f} and {ef[1][1]:.2f}, which is why the intervals in those scenarios are narrow and wrong at once.")

def diagnostics_prose(data):
    tm=[data[sc]['_truth_mcse'] for sc in ALL]
    z_clock=[100*data[sc]['_zero_inflation']['zero_share'] for sc in ALL if sc not in ('I','J')]; z_stoch=[100*data[sc]['_zero_inflation']['zero_share'] for sc in ('I','J')]
    uns={sc:100*data[sc]['_diagnostics']['unsupported_share'] for sc in ALL}; nonzero=[sc for sc in ALL if uns[sc]>=0.05]
    coar={sc:100*data[sc]['_diagnostics']['fallback_share_prespecified'] for sc in ALL}; rest=[coar[sc] for sc in ALL if sc!='C']
    return (f"Table~\\ref{{stab:simdiag}} records the precision of the population truths and three sample diagnostics. The truth Monte Carlo standard error is between {min(tm):.4f} and {max(tm):.4f} points, small against every bias the tables report. Observed zeros make up {min(z_clock):.1f}--{max(z_clock):.1f}\\% of excursions where they arise from the persistence clock alone and {min(z_stoch):.1f}--{max(z_stoch):.1f}\\% under stochastic non-annunciation ({lets(['I','J'])}). The share of candidate replayed time in duration bins with no baseline excursion is zero in every scenario except "+', '.join(f"{L[sc]} ({uns[sc]:.1f}\\%)" for sc in nonzero)+f", where it is the identification failure by construction. Between {min(rest):.1f}\\% and {max(rest):.1f}\\% of the cells of the independently selected map are coarsened to their parent duration rate in the 300-patient scenarios, against {coar['C']:.1f}\\% in {L['C']}, where 60 patients leave many cells below the five-excursion minimum.")

def sweep_prose(data,sweep):
    rows=sweep['sizes']; first,last=rows[0]['lrc_dz_pre'],rows[-1]['lrc_dz_pre']
    n0,n1=rows[0]['n_patients'],rows[-1]['n_patients']
    expected=first['esd']*(n0/n1)**0.5
    ratios=[r['lrc_dz_pre']['se_over_sd'] for r in rows]
    covs=[r['lrc_dz_pre']['coverage'] for r in rows]
    zero={}
    for n in (600,1200):
        p=RES/f'ZERO_CHECK_{n}.json'
        if p.exists(): zero[n]=json.loads(p.read_text())['lrc_dz_pre']
    jtxt=''
    if zero:
        jtxt=(f" In {L['J']}, with feature-dependent non-annunciation, coverage is "+' and '.join(f"{v['coverage']:.3f}" for v in zero.values())+" at "+' and '.join(f"{n:,}".replace(',','\\,') for n in zero)+" patients, with mean bootstrap standard error to empirical standard deviation ratios of "+' and '.join(f"{v['se_over_sd']:.2f}" for v in zero.values())+".")
    return (f"Table~\\ref{{stab:simsweep}} and panels C and D of the simulation figure in the main text follow the feature-augmented estimator as the patient count grows. In {L['A']} the empirical standard deviation falls from {first['esd']:.3f} points at {n0} patients to {last['esd']:.3f} at {n1:,} ({expected:.3f} expected at the $n^{{-1/2}}$ rate), coverage moves from {covs[0]:.3f} to {covs[-1]:.3f}, and the ratio of the mean bootstrap standard error to the empirical standard deviation stays between {min(ratios):.2f} and {max(ratios):.2f}.{jtxt} The bootstrap therefore tracks the sampling variability at every count examined, and the coverage shortfall at 300 patients is not accompanied by a bias that grows with the sample.").replace('2,400','2\\,400').replace('1,200','1\\,200')

def build(data,sweep):
    lines=[r'\begin{table}[htbp]\centering\small',
    r'\caption{Simulation performance for the principal estimator and comparators in selected scenarios from Table~\ref{tab:simdesign}.}',
    r'\label{tab:sim}',r'\begin{tabular}{@{}llrrr@{}}\toprule',
    r'Scenario & Method & Bias & RMSE & Coverage\\\midrule']
    for sc in ORDER:
        for j,key in enumerate(['replay','lrc_d','lrc_dz_pre']):
            v=data[sc][key]
            name=f'{L[sc]}: {SHORT[sc]}' if j==0 else ''
            lines.append(f'{name} & {METHODS[key]} & {f(v["bias"])} & {f(v["rmse"])} & {f(v.get("coverage"),3)}'+r'\\')
        lines.append(r'\addlinespace')
    lines.extend([r'\bottomrule\end{tabular}',
    r'\tightnote{Bias and root mean squared error (RMSE) are percentage points for the true linked-burden reduction. Each scenario uses 300 patients and 500 replicates; coverage uses the first 400 replicates and 200 patient bootstrap resamples. The feature map is selected on an independent sample. Nominal coverage is 0.95. Replay has no interval. Incomplete support and transport failure deliberately violate identification assumptions; the intervals are not expected to cover the scientific target in these settings. Complete Monte Carlo diagnostics and additional scenarios are in the supplement.}',r'\end{table}'])
    sup=[r'\subsection{Data-generating laws}',
    f'''Figure~\\ref{{sfig:simlaws}} draws the laws that follow: the duration distributions at the two limits, the reward laws with and without overshoot dependence, and the non-annunciation probabilities of scenarios {L['I']} and {L['J']}.''',
    f'''Patient effects are $F_i=G_i/k$, where $G_i\\sim\\mathrm{{Gamma}}(k,1)$ independently, with $k=4$ except in scenario {L['G']}, where $k=1.2$. Each patient has $N_i=\\max\\{{1,\\mathrm{{Poisson}}(25F_i)\\}}$ excursions. The same $N_i$ and $F_i$ are used at the baseline and candidate rules. Conditional on them, duration draws at the two rules are independent, with Gamma shape 1.6 and scale $s_uF_i^{{0.3}}$. The baseline scale is 14; the candidate scale is 11, except in {L['E']} where it is 20. Durations are rounded upward to two-second values and capped at 400 seconds. In {L['E']} only, baseline durations are capped at 100 seconds, producing a point mass there and leaving some candidate cells unobserved at baseline. These experiments simulate excursion measures, not a nested pair of sample paths.''',
    f'''Overshoot classes $z=0,1,2,3$ have baseline probabilities $(0.25,0.30,0.25,0.20)$. Candidate probabilities are unchanged in {_and(['A','D','E','G'])} and equal $(0.45,0.28,0.17,0.10)$ otherwise. Define $b(d)=(d-8)_+$ and a mean positive reward $m(d,z)=b(d)g(d,z)$. The multiplier is 0.85 in {_and(['A','E','F','G','I','J'])}; it is $(0.45,0.70,0.85,0.92)_z$ in {_and(['B','C'])} and $\\min\\{{0.95,0.55+0.004d\\}}$ in {L['D']}. In {L['F']}, the candidate reward is additionally multiplied by 0.8. Baseline positive rewards are Gamma with scale 3 and shape $m/3$, up to a numerical shape guard of order $10^{{-9}}$; rewards with $m=0$ are set to zero.''',
    f'''Scenarios {L['I']} and {L['J']} add stochastic non-annunciation: the positive reward is replaced by zero with probability $p(d,z)$. In {L['I']}, $p=0.40\\{{0.20+0.80\\exp[-(d-2)/30]\\}}$; in {L['J']}, 0.40 is replaced by $(0.65,0.45,0.25,0.12)_z$. Thus the conditional mean is $a_u(d,z)=\\{{1-p(d,z)\\}}m_u(d,z)$. The same cells can contain both observed zeros and positive rewards. No positivity assumption on $a_0$ is imposed.''',
    f'''The simulation map has 30 nominal two-second bins below 60 seconds and pooled bins $[60,120)$ and $[120,\\infty)$. It uses the same count/seconds carrier functions as the clinical analysis but a smaller partition. Feature-map selection uses a minimum of five excursions. Scenario {L['C']} uses 60 patients to examine sparse cells; all other main scenarios use 300. Independent map selection uses an additional sample of the same size. A secondary analysis selects the map within the estimation sample. The baseline-reward oracle integrates the known $a_0(d,z)$ excursion by excursion, including where baseline observations are absent. In {L['F']} this oracle still uses the baseline reward, so it does not know the deliberately changed candidate reward. The fixed-offset comparator matches observed baseline linked seconds with a persistence model, using mean duration within each pooled cell.''',
    r'\subsection{Complete estimator comparisons}',
    comparisons_prose(data),
    r'\begin{longtable}{@{}llrrrrr@{}}',
    r'\caption{Simulation accuracy and Monte Carlo uncertainty.}\label{stab:simfull}\\',
    r'\toprule Method & & Bias & Empirical SD & RMSE & Coverage & MCSE(bias)\\\midrule\endfirsthead',
    r'\toprule Method & & Bias & Empirical SD & RMSE & Coverage & MCSE(bias)\\\midrule\endhead',
    r'\bottomrule\endfoot']
    for sc in ALL:
        d=data[sc]
        sup.append(r'\multicolumn{7}{@{}l}{\textbf{'+f'{L[sc]}: {SHORT[sc]}'+'}'+f'; truth {d["_truth"]:.3f}'+r'\%, $n='+str(d['_diagnostics']['n_patients'])+r'$}\\')
        for key in METHODS:
            v=d[key]
            sup.append(METHODS[key]+' & & '+ ' & '.join([f(v['bias']),f(v['esd']),f(v['rmse']),f(v.get('coverage'),3),f(v['mcse_bias'],3)])+r'\\')
        sup.append(r'\addlinespace')
    sup += [r'\end{longtable}',
    r'''Bias, empirical standard deviation and root mean squared error are percentage points. Bias MCSE combines the independent replicate mean and population-truth Monte Carlo errors. Coverage has binomial MCSE $\{\widehat c(1-\widehat c)/400\}^{1/2}$ conditional on the estimated truth. The common truth error is assessed separately by rescoring the saved intervals one truth MCSE above and below its estimate; that sensitivity range is not an additional formally estimated coverage standard error.''',
    r'\subsection{Truth precision, overlap and observed zeros}',
    diagnostics_prose(data),
    r'\begin{table}[htbp]\centering\small',r'\caption{Population-truth precision and sample diagnostics.}\label{stab:simdiag}',
    r'\begin{tabular}{@{}lrrrr@{}}\toprule',
    r'Scenario & Truth MCSE & Zero rewards (\%) & Unobserved duration share (\%) & Coarsened cells (\%)\\\midrule']
    for sc in ALL:
        d=data[sc];diag=d['_diagnostics']
        sup.append(L[sc]+' & '+' & '.join([f(d['_truth_mcse'],4),f(100*d['_zero_inflation']['zero_share'],1),f(100*diag['unsupported_share'],1),f(100*diag['fallback_share_prespecified'],1)])+r'\\')
    sup +=[r'\bottomrule\end{tabular}',r'\tightnote{Truth MCSE is in percentage points, based on four million independent patients for each scenario. Unobserved duration share is target replayed time in bins without baseline observations. Coarsened cells is a proportion of cells in the independently selected feature map, not a proportion of burden. Zero-reward shares are averages over the simulated estimation samples.}',r'\end{table}',
    r'\subsection{Patient-count sensitivity}',
    sweep_prose(data,sweep),
    r'\begin{table}[htbp]\centering\small',r'\caption{Independent-map feature calibration as patient count increases.}\label{stab:simsweep}',
    r'\begin{tabular}{@{}llrrrrr@{}}\toprule',
    r'Scenario & Patients & Bias & Empirical SD & Coverage & MCSE(coverage) & Mean SE/SD\\\midrule']
    for row in sweep['sizes']:
        v=row['lrc_dz_pre'];sup.append(L['A']+' & '+str(row['n_patients'])+' & '+' & '.join([f(v['bias'],3),f(v['esd'],3),f(v['coverage'],3),f(v['mcse_coverage'],3),f(v['se_over_sd'],3)])+r'\\')
    for n in [600,1200]:
        p=RES/f'ZERO_CHECK_{n}.json'
        if p.exists():
            v=json.loads(p.read_text())['lrc_dz_pre'];sup.append(L['J']+' & '+str(n)+' & '+' & '.join([f(v['bias'],3),f(v['esd'],3),f(v['coverage'],3),f(v['mcse_coverage'],3),f(v['se_over_sd'],3)])+r'\\')
    sup +=[r'\bottomrule\end{tabular}',r'\tightnote{Each row uses 400 replicates, with 200 patient bootstrap resamples per replicate. Mean SE/SD compares the mean bootstrap standard error with empirical variability. '+L['A']+' uses a common duration-only reward; '+L['J']+' includes feature-dependent stochastic non-annunciation. Rows use the same scenario-specific high-precision population truth.}',r'\end{table}']
    a,b,c,d,e,ff,g,i,j=(data[x] for x in ['A','B','C','D','E','F','G','I','J'])
    def cov(sc,key):return data[sc][key]['coverage']
    holds=['A','G','D']
    hold_bias=max(abs(data[sc][key]['bias']) for sc in holds for key in ['lrc_d','lrc_dz_pre'])
    hold_cov=[cov(sc,key) for sc in holds for key in ['lrc_d','lrc_dz_pre']]
    mcse_cov=(0.95*0.05/400)**0.5
    j_short=(0.95-cov('J','lrc_dz_pre'))/mcse_cov
    def zero_json(n):
        p=RES/f'ZERO_CHECK_{n}.json'
        return json.loads(p.read_text())['lrc_dz_pre'] if p.exists() else None
    j600,j1200=zero_json(600),zero_json(1200)
    sweep_rows=sweep['sizes']
    ratio_all=[r['lrc_dz_pre']['se_over_sd'] for r in sweep_rows if r['n_patients']<2400]+[v['se_over_sd'] for v in (j600,j1200) if v]
    ratio_dev=max(abs(x-1) for x in ratio_all)
    r2400=[r['lrc_dz_pre'] for r in sweep_rows if r['n_patients']==2400][0]
    bias_sd=max([abs(r['lrc_dz_pre']['bias_over_sd']) for r in sweep_rows]+[abs(v['bias_over_sd']) for v in (j600,j1200) if v])
    words={1:'one',2:'two',3:'three',4:'four',5:'five'}
    j_words=words.get(int(round(j_short)),f'{j_short:.0f}')
    map_diff=max(abs(data[sc]['lrc_dz']['bias']-data[sc]['lrc_dz_pre']['bias']) for sc in ALL)
    off=[abs(data[sc]['offset']['bias']) for sc in ALL]
    truth_mcse=max(data[sc]['_truth_mcse'] for sc in ALL)
    b_cov_phrase='none of the' if cov('B','lrc_d')==0 else f"{cov('B','lrc_d'):.3f} of"
    text=(f"Figure~\\ref{{fig:simulation}} summarises all nine scenarios and Table~\\ref{{tab:sim}} gives the principal contrasts. Where the transport condition holds ({', '.join(L[k] for k in holds)}), calibration removes the replay error: replay understated the reduction by {abs(a['replay']['bias']):.2f}, {abs(g['replay']['bias']):.2f} and {abs(d['replay']['bias']):.2f} percentage points, both calibrated estimators were within {hold_bias:.2f} points of the truth, and their intervals covered it in {min(hold_cov):.3f}--{max(hold_cov):.3f} of replicates against a nominal 0.95, whose binomial Monte Carlo standard error is {mcse_cov:.3f}. The population truths themselves carry a Monte Carlo error of at most {truth_mcse:.3f} points.\n\n"
    f"When the reward depends on overshoot and its composition shifts ({L['B']}), duration-only calibration inherits a bias of {b['lrc_d']['bias']:.2f} points and its intervals cover the truth in {b_cov_phrase} replicates, whereas feature-augmented calibration has bias {b['lrc_dz_pre']['bias']:.2f} points and coverage {cov('B','lrc_dz_pre'):.3f}. With 60 patients ({L['C']}) the feature-augmented bias is {c['lrc_dz_pre']['bias']:.2f} points, its empirical standard deviation widens to {c['lrc_dz_pre']['esd']:.2f}, and coverage is {cov('C','lrc_dz_pre'):.3f}.\n\n"
    f"Under stochastic non-annunciation the zeros enter as observations. When the miss probability depends on duration only ({L['I']}), both calibrated estimators are unbiased ({i['lrc_d']['bias']:.2f} and {i['lrc_dz_pre']['bias']:.2f} points); when it also depends on overshoot ({L['J']}), duration-only calibration is biased by {j['lrc_d']['bias']:.2f} points and feature-augmented calibration by {j['lrc_dz_pre']['bias']:.2f}. Feature-augmented coverage in {L['J']} was {cov('J','lrc_dz_pre'):.3f} at 300 patients, about {j_words} Monte Carlo standard errors below nominal, and {j600['coverage']:.3f} and {j1200['coverage']:.3f} at 600 and 1\\,200 patients; in {L['A']} it was "+', '.join(f"{r['lrc_dz_pre']['coverage']:.3f}" for r in sweep_rows[:-1])+f" and {sweep_rows[-1]['lrc_dz_pre']['coverage']:.3f} at 300, 600, 1\\,200 and 2\\,400 patients. Across these runs the empirical standard deviation fell as $n^{{-1/2}}$, the mean bootstrap standard error was within {100*ratio_dev:.0f}\\% of it below 2\\,400 patients and {100*(r2400['se_over_sd']-1):.0f}\\% above it at 2\\,400, where coverage was {r2400['coverage']:.3f}, and the bias never exceeded {bias_sd:.2f} of a standard deviation, so the shortfall reflects interval width in small samples rather than a persistent centring error.\n\n"
    f"When an identification condition fails, the intervals do not protect against it. In {L['E']} the candidate rule places {100*e['_diagnostics']['unsupported_share']:.1f}\\% of its replayed time in duration cells with no baseline excursion; replay and both calibrated estimators are biased by {min(e['lrc_d']['bias'],e['lrc_dz_pre']['bias']):.2f} points or more, while the baseline-reward oracle, which integrates the known reward over the unobserved cells, is unbiased ({e['baseline_oracle']['bias']:.2f}). In {L['F']} the reward itself moves with the limit and even the oracle is biased by {ff['baseline_oracle']['bias']:.2f} points. The narrow spread of both rows in Figure~\\ref{{fig:simulation}}A is the signature of these failures: the error is systematic, and no sampling interval can reveal it. Among the secondary comparators, selecting the map in sample instead of independently changed no bias by more than {map_diff:.2f} points, and the fixed-offset comparator was biased by between {min(off):.1f} and {max(off):.1f} points in absolute value across the scenarios.\n")
    return '\n'.join(lines)+'\n','\n'.join(sup)+'\n',text

def _and(keys):
    """Manuscript letters of simulation keys, sorted, as 'A, B and C'."""
    ls=sorted(L[k] for k in keys)
    return ls[0] if len(ls)==1 else ', '.join(ls[:-1])+' and '+ls[-1]

def main():
 p=argparse.ArgumentParser(description=__doc__.splitlines()[0]);p.add_argument('--src',type=Path,default=RES/'SIMULATION_LINKED_REWARD.json');p.add_argument('--sweep',type=Path,default=RES/'SIMULATION_SAMPLE_SIZE.json');p.add_argument('--dest',type=Path,default=OUT/'TAB_SIMULATION_MAIN.tex');p.add_argument('--dest-supp',type=Path,default=OUT/'TAB_SIMULATION_SUPP.tex');p.add_argument('--findings',type=Path,default=OUT/'simulation_findings.tex');a=p.parse_args()
 for src in (a.src,a.sweep):
  if not src.exists(): raise SystemExit(f"{src} is missing: run simulation/run_submission_simulations.py first (the saved results are not distributed)")
 main,sup,findings=build(json.loads(a.src.read_text()),json.loads(a.sweep.read_text()))
 for dest,text in [(a.dest,main),(a.dest_supp,sup),(a.findings,findings)]:dest.parent.mkdir(parents=True,exist_ok=True);dest.write_text(text)
 print(f'Generated simulation tables and findings from the saved results into {os.path.relpath(a.dest.parent, ROOT)}/.')
if __name__=='__main__':main()
