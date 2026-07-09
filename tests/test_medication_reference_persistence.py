import json
import os
import shutil
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import ai_helpers
import patient_care_storage as storage


class MedicationReferencePersistenceTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self._orig_root = storage._DATA_ROOT
        storage._DATA_ROOT = type(self._orig_root)(self.temp_dir)

    def tearDown(self):
        storage._DATA_ROOT = self._orig_root
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    @patch("ai_helpers.supabase")
    def test_warfarin_reference_persists_locally_when_supabase_lacks_patient_id(self, mock_supabase):
        mock_supabase.table.return_value.insert.return_value.execute.side_effect = RuntimeError(
            "patient_id column missing"
        )
        mock_supabase.table.return_value.select.return_value.eq.return_value.order.return_value.execute.side_effect = RuntimeError(
            "patient_id column missing"
        )

        saved = ai_helpers.upsert_medication_reference(
            medication_name="Warfarin",
            image_b64="ZmFrZV9pbWFnZQ==",
            pill_strength=3.0,
            strength_unit="mg",
            brand="Coumadin",
            patient_id="42",
            back_image_b64="YmFja19pbWFnZQ==",
        )
        self.assertTrue(saved)

        refs = ai_helpers.get_medication_references("42")
        self.assertEqual(len(refs), 1)
        self.assertEqual(refs[0]["medication_name"], "Warfarin")
        self.assertEqual(refs[0]["image_b64"], "ZmFrZV9pbWFnZQ==")

        meta = json.loads(refs[0]["description"])
        self.assertEqual(meta["pill_strength"], 3.0)
        self.assertEqual(meta["back_image_b64"], "YmFja19pbWFnZQ==")

    @patch("ai_helpers.supabase")
    def test_upsert_replaces_existing_warfarin_row_for_same_patient(self, mock_supabase):
        mock_supabase.table.return_value.insert.return_value.execute.side_effect = RuntimeError("fail")

        self.assertTrue(
            ai_helpers.upsert_medication_reference(
                medication_name="Warfarin",
                image_b64="first",
                patient_id="7",
                back_image_b64="back1",
            )
        )
        self.assertTrue(
            ai_helpers.upsert_medication_reference(
                medication_name="Warfarin",
                image_b64="second",
                patient_id="7",
                back_image_b64="back2",
            )
        )

        refs = ai_helpers.get_medication_references("7")
        self.assertEqual(len(refs), 1)
        self.assertEqual(refs[0]["image_b64"], "second")


if __name__ == "__main__":
    unittest.main(verbosity=2)
