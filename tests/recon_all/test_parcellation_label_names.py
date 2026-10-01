"""用标准库核查比较器标签名称，不加载影像或替代真实 benchmark。"""

import ast
from pathlib import Path
import unittest


class ParcellationLabelNamesTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        root = Path(__file__).resolve().parents[2]
        source = root / "validation/recon_all/python_gpu_port/compare_parcellation_dice.py"
        tree = ast.parse(source.read_text(), filename=str(source))
        nodes = [node for node in tree.body
                 if (isinstance(node, ast.Assign)
                     and any(isinstance(target, ast.Name)
                             and target.id == "FILLED_LABEL_NAMES" for target in node.targets))
                 or (isinstance(node, ast.FunctionDef) and node.name == "_label_name")]
        namespace = {}
        exec(compile(ast.Module(body=nodes, type_ignores=[]), str(source), "exec"), namespace)
        cls.lookup = staticmethod(namespace["_label_name"])

    def test_filled_uses_hemisphere_codes(self):
        self.assertEqual(self.lookup(volume_name="filled.mgz", label=255,
                                     label_names={255: "CC_Anterior"}), "Left-Hemisphere-Fill")
        self.assertEqual(self.lookup(volume_name="filled.mgz", label=127,
                                     label_names={127: "LUT-other-region"}), "Right-Hemisphere-Fill")

    def test_other_segmentations_keep_the_global_lut(self):
        self.assertEqual(self.lookup(volume_name="aseg.mgz", label=255,
                                     label_names={255: "CC_Anterior"}), "CC_Anterior")
        self.assertEqual(self.lookup(volume_name="aparc+aseg.mgz", label=1001,
                                     label_names={1001: "ctx-lh-bankssts"}), "ctx-lh-bankssts")

    def test_unknown_fill_code_is_not_reinterpreted_as_a_region(self):
        self.assertIsNone(self.lookup(volume_name="filled.mgz", label=252,
                                     label_names={252: "CC_Mid_Anterior"}))
        self.assertIsNone(self.lookup(volume_name="aseg.mgz", label=999999,
                                     label_names={}))


if __name__ == "__main__":
    unittest.main()
