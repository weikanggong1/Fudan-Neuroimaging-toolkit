"""A real T1 SynthMorph affine must keep its voxel-LTA bytes after removing Surfa."""

from pathlib import Path

from fnit.recon_all.input_talairach_chain import write_voxel_lta_from_ras


def test_real_t1_ras_to_voxel_lta_matches_prior_output(tmp_path: Path) -> None:
    fixture_dir = Path(__file__).parent / "data"
    output = tmp_path / "talairach.xfm.lta"
    write_voxel_lta_from_ras(
        source_lta=fixture_dir / "talairach_ras_real.lta",
        output_lta=output,
    )
    assert output.read_bytes() == (fixture_dir / "talairach_voxel_oracle_real.lta").read_bytes()
