#!/usr/bin/env python3
"""Tests for model smoke diagnostic classification."""
import importlib.util
import pathlib
import tempfile
import unittest


spec = importlib.util.spec_from_file_location(
    "model_profile", pathlib.Path(__file__).with_name("agent-model-test-profile.py"))
profile = importlib.util.module_from_spec(spec)
spec.loader.exec_module(profile)


class SmokeDiagnosticsTest(unittest.TestCase):
    def test_precondition_event_and_observations_are_preserved(self):
        with tempfile.TemporaryDirectory() as directory:
            log = pathlib.Path(directory) / "test.log"
            log.write_text('model_smoke_event {"version":1,"kind":"precondition_not_observed",'
                           '"missing_stage":"baseline","observed_candidates":[{"tool":"dataset.schema"}]}\n')
            events = profile.smoke_events(log)
        self.assertEqual(profile.skipped_status(events), "inconclusive_precondition")
        self.assertEqual(events[0]["observed_candidates"][0]["tool"], "dataset.schema")

    def test_configuration_and_unclassified_skips(self):
        self.assertEqual(profile.skipped_status([{"kind": "configuration_error"}]), "configuration_error")
        self.assertEqual(profile.skipped_status([]), "skipped_unclassified")

    def test_malformed_and_unknown_events_do_not_classify(self):
        with tempfile.TemporaryDirectory() as directory:
            log = pathlib.Path(directory) / "test.log"
            log.write_text('model_smoke_event broken\nmodel_smoke_event {"version":2,"kind":"configuration_error"}\n')
            self.assertEqual(profile.smoke_events(log), [])


if __name__ == "__main__":
    unittest.main()
