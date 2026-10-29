#!/usr/bin/env python3
"""Summarize the production concept-graft smoke without manufacturing evidence."""
import argparse
import json
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('log', help='Structured concept-synthesis-model stdout')
    parser.add_argument('--runtime-log', help='Separate server trace with actual model arms')
    args = parser.parse_args()
    searches = []
    malformed_records = 0
    with open(args.log, encoding='utf-8') as source:
        for line in source:
            prefix = 'flydelta_concept_synthesis_search='
            if line.startswith(prefix):
                try:
                    searches.append(json.loads(line[len(prefix):]))
                except json.JSONDecodeError:
                    malformed_records += 1
    runtime_arms = []
    if args.runtime_log:
        with open(args.runtime_log, encoding='utf-8') as source:
            for line in source:
                prefix = 'flydelta_daemon_arm '
                if line.startswith(prefix):
                    try:
                        runtime_arms.append(json.loads(line[len(prefix):]))
                    except json.JSONDecodeError:
                        malformed_records += 1
    initial = next((x for x in searches if x.get('phase') == 'initial'), None)
    grafts = [x for x in searches if str(x.get('phase', '')).startswith('concept-graft-')]
    result = {'schema': 'flydelta_graft_verification_v1', 'initial_present': initial is not None,
              'graft_count': len(grafts), 'malformed_records': malformed_records, 'grafts': []}
    for graft in grafts:
        arms = graft.get('arms_detail', [])
        rounds = [r for w in graft.get('whirlpool', []) for r in w.get('rounds', [])]
        evidence = [a.get('evidence_ref', '') for a in arms if a.get('evidence_ref')]
        matched = [a for a in runtime_arms if a.get('arm_id') and any(
            a['arm_id'] in ref for ref in evidence)]
        result['grafts'].append({
            'phase': graft.get('phase'), 'direction_ref': graft.get('graft_direction_ref'),
            'differs_from_initial': bool(initial and graft.get('graft_direction_ref') and
                                         graft.get('graft_direction_ref') != initial.get('graft_direction_ref')),
            'where_rounds': len(rounds), 'probed_layers': [r.get('probed_layers', []) for r in rounds],
            'dose_evaluated_arms': sum(bool(a.get('dose_evaluated')) for a in arms),
            'full_generation_arms': sum(a.get('execution_class') == 'full' for a in arms),
            'host_evaluated_arms': sum(bool(a.get('host_evaluated')) for a in arms),
            'verifier_known_arms': sum(bool(a.get('verifier_known')) for a in arms),
            'raw_responses_available': any(a.get('generation_available') and a.get('content') for a in matched),
            'matched_runtime_arms': [{k: a.get(k) for k in ('arm_id', 'wave_id', 'phase', 'layers',
                'executed_alpha', 'overlay_applied', 'host_evaluated', 'verifier_known',
                'verifier_passed', 'content')} for a in matched],
            'trace_fields_complete': all('dose_evaluated' in a and 'execution_class' in a for a in arms),
            'search_status': graft.get('search_status'),
        })
    result['status'] = 'observed' if initial and grafts and not malformed_records and all(
        x['differs_from_initial'] and x['where_rounds'] and x['trace_fields_complete']
        for x in result['grafts']) else 'incomplete'
    result['interpretation'] = ('WHERE observations only; dose, host outcomes and actual model outputs '
                                'must be assessed separately. Missing fields are not successful checks.')
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result['status'] == 'observed' else 2


if __name__ == '__main__':
    sys.exit(main())
