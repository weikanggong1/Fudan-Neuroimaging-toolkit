import copy,json,unittest
from pathlib import Path
from verify_con11_origin import select_con11_case

class ActualManifestSchemaTests(unittest.TestCase):
    def setUp(self):self.fixture=json.loads(Path(__file__).with_name('actual_manifest_CON11_fixture.json').read_text())
    def test_actual_case_id_and_unprefixed_subject(self):
        case=select_con11_case(self.fixture)
        self.assertEqual(case['case_id'],'sub-CON11');self.assertEqual(case['subject'],'CON11')
        self.assertEqual(self.fixture['provenance']['source_manifest_sha256'],'d707f7a990372e50fb27de0023b2cddfc4eb74c7e5cdd9d909e41b90c2fa1b88')
    def test_duplicate_actual_case_is_rejected(self):
        value=copy.deepcopy(self.fixture);value['cases'].append(copy.deepcopy(value['cases'][0]))
        with self.assertRaisesRegex(ValueError,'unique canonical case_id'):select_con11_case(value)
    def test_inconsistent_subject_is_rejected(self):
        value=copy.deepcopy(self.fixture);value['cases'][0]['subject']='CON10'
        with self.assertRaisesRegex(ValueError,'subject disagrees'):select_con11_case(value)
    def test_missing_case_is_rejected(self):
        value=copy.deepcopy(self.fixture);value['cases']=[]
        with self.assertRaisesRegex(ValueError,'unique canonical case_id'):select_con11_case(value)
if __name__=='__main__':unittest.main()
