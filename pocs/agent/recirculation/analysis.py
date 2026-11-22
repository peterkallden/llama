"""Locked low-dose comparison and paired exploratory statistics (standard library only)."""
import argparse
import itertools
import json
import math
import random
import statistics
from pathlib import Path


def pair(row):
    return row['source_layer_input'], row['destination_layer_input']


def configuration(row):
    return {key: row[key] for key in ('source_layer_input', 'destination_layer_input', 'alpha', 'beta')}


def ranks(values):
    return [1 + sum(v > x for v in values) + (values.count(x) - 1) / 2 for x in values]


def correlation(a, b):
    a, b = ranks(a), ranks(b)
    ma, mb = statistics.mean(a), statistics.mean(b)
    den = math.sqrt(sum((x-ma)**2 for x in a) * sum((x-mb)**2 for x in b))
    return sum((x-ma)*(y-mb) for x, y in zip(a, b)) / den if den else None


def lock(selection):
    candidates = selection['candidates']
    pairs = list(dict.fromkeys(pair(c) for c in candidates))
    if len(pairs) != 3:
        raise ValueError('expected the same three candidate pairs')
    low = [next(c for c in candidates if pair(c) == p and math.isclose(c['alpha'], .01, abs_tol=1e-8)) for p in pairs]
    low_winner = max(low, key=lambda c: c['mean_margin_change'])
    chosen_pairs = {'cosine': pair(selection['geometry_ranked_pair']),
                    'margin': pair(selection['selected']), 'low_dose_response': pair(low_winner)}
    comparisons = []
    for method, p in chosen_pairs.items():
        rows = [c for c in candidates if pair(c) == p]
        best = max(sorted(rows, key=lambda c: c['alpha']), key=lambda c: c['mean_margin_change'])
        small = next(c for c in low if pair(c) == p)
        for name, row in [('common_alpha', small), ('selected_dose', best)]:
            comparisons.append({'method': method, 'comparison': name, 'configuration': configuration(row)})
    selection['locked_comparisons'] = comparisons
    selection['low_dose_response'] = {'definition': 'Observed mean margin gain at alpha=0.01; not a general derivative',
        'ranking': [dict(configuration(c), mean_margin_change=c['mean_margin_change'], baseline_anchored_sensitivity=c['mean_margin_change']/c['alpha']) for c in sorted(low, key=lambda c: -c['mean_margin_change'])]}
    prediction = []
    for dose in (.03, .1, .2):
        rows = [next(c for c in candidates if pair(c) == p and math.isclose(c['alpha'], dose, abs_tol=1e-7)) for p in pairs]
        best = max(c['mean_margin_change'] for c in rows)
        prediction.append({'alpha': dose, 'spearman': correlation([c['mean_margin_change'] for c in low], [c['mean_margin_change'] for c in rows]),
                           'low_dose_winner_is_best': any(pair(c) == pair(low_winner) and c['mean_margin_change'] == best for c in rows)})
    selection['low_dose_prediction'] = prediction
    return selection


def describe(values):
    if not values or not all(math.isfinite(x) for x in values):
        raise ValueError('empty or nonfinite paired differences')
    rng = random.Random(1729)
    n = len(values)
    samples = sorted(sum(values[rng.randrange(n)] for _ in range(n))/n for _ in range(10000))
    return {'mean': statistics.mean(values), 'median': statistics.median(values),
            'paired_bootstrap_95_percentile': [samples[249], samples[9749]],
            'improved': sum(x > 0 for x in values), 'regressed': sum(x < 0 for x in values), 'unchanged': sum(x == 0 for x in values)}


def lag_curve(case, branch):
    arms = case['arms']
    key = branch + '_path_future_logprobs'
    base, target, control = [arms[a][key] for a in ('baseline', 'targeted_recirculation', 'targeted_norm_matched_control')]
    if not len(base) == len(target) == len(control):
        raise ValueError('unaligned continuation paths')
    rows, cb, cc, continuation_b, continuation_c = [], 0., 0., 0., 0.
    first = None
    for i, (b, t, c) in enumerate(zip(base, target, control)):
        tolerance = 1e-4 + 1e-4 * max(abs(b), abs(t), abs(c))
        if i == 0 and (abs(t-b) > tolerance or abs(c-b) > tolerance):
            raise ValueError('targeted lag 1 changed with identical prefix')
        gain, donor_gain = t-b, t-c
        cb += gain
        cc += donor_gain
        position = case['update_token_position'] + i + 1
        continuation = position >= case['prompt_token_count']
        if continuation:
            continuation_b += gain
            continuation_c += donor_gain
        if first is None and abs(gain) > tolerance:
            first = i+1
        rows.append({'lag': i+1, 'token_position': position, 'is_continuation': continuation,
                     'logprob_gain_vs_baseline': gain, 'logprob_gain_vs_control': donor_gain,
                     'cumulative_logprob_gain_vs_baseline': cb, 'cumulative_logprob_gain_vs_control': cc,
                     'continuation_cumulative_gain_vs_baseline': continuation_b,
                     'continuation_cumulative_gain_vs_control': continuation_c})
    return {'first_effect_lag': first, 'effects': rows}


def summarize(events):
    reports = [e for e in events if e.get('event') == 'locked_comparison']
    if len(reports) != 6:
        raise ValueError('expected six locked method/comparison reports')
    output = {'bootstrap': {'seed': 1729, 'resamples': 10000, 'exploratory': True, 'multiple_comparison_correction': False}, 'comparisons': [], 'between_methods': [], 'low_dose_retention': []}
    for event in reports:
        report = event['report']
        cases = report['cases']
        entry = {'method': event['method'], 'comparison': event['comparison'], 'configuration': report['configuration'], 'summary': report['summary'], 'against': {}, 'by_task_type': {}, 'cases': cases}
        for arm in ('baseline', 'targeted_norm_matched_control'):
            entry['against'][arm] = describe([c['arms']['targeted_recirculation']['answer_margin']-c['arms'][arm]['answer_margin'] for c in cases])
        for task in sorted({c['task_type'] for c in cases}):
            values = [c['targeted_margin_change_vs_baseline'] for c in cases if c['task_type'] == task]
            entry['by_task_type'][task] = dict(describe(values), regression_fraction=sum(x < 0 for x in values)/len(values))
        for case in cases:
            case['expected_branch_lag'] = lag_curve(case, 'expected')
            case['incorrect_branch_lag'] = lag_curve(case, 'incorrect')
        output['comparisons'].append(entry)
    for kind in ('common_alpha', 'selected_dose'):
        group = [e for e in reports if e['comparison'] == kind]
        for a, b in itertools.combinations(group, 2):
            left, right = a['report']['cases'], b['report']['cases']
            if [c['case_id'] for c in left] != [c['case_id'] for c in right]:
                raise ValueError('unpaired cases')
            output['between_methods'].append({'comparison': kind, 'left': a['method'], 'right': b['method'],
                'identical_configuration': a['report']['configuration'] == b['report']['configuration'],
                'difference': describe([x['arms']['targeted_recirculation']['answer_margin']-y['arms']['targeted_recirculation']['answer_margin'] for x, y in zip(left, right)])})
    for method in ('cosine', 'margin', 'low_dose_response'):
        small = next(e for e in reports if e['method'] == method and e['comparison'] == 'common_alpha')
        large = next(e for e in reports if e['method'] == method and e['comparison'] == 'selected_dose')
        low_values = [c['targeted_margin_change_vs_baseline'] for c in small['report']['cases']]
        high_values = [c['targeted_margin_change_vs_baseline'] for c in large['report']['cases']]
        lo, hi = statistics.mean(low_values), statistics.mean(high_values)
        output['low_dose_retention'].append({'method': method, 'low_gain': lo, 'selected_gain': hi,
            'ratio': lo/hi if hi > 0 else None, 'undefined_reason': None if hi > 0 else 'selected_gain_not_positive',
            'cases': [{'low_gain': l, 'selected_gain': h, 'ratio': l/h if h > 0 else None, 'undefined_reason': None if h > 0 else 'selected_gain_not_positive'} for l, h in zip(low_values, high_values)]})
    return output


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('mode', choices=['lock', 'report'])
    parser.add_argument('input', type=Path)
    parser.add_argument('output', type=Path)
    args = parser.parse_args()
    result = lock(json.loads(args.input.read_text())) if args.mode == 'lock' else summarize([json.loads(line) for line in args.input.open()])
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')


if __name__ == '__main__':
    main()
