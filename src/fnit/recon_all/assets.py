"""Download verified FreeSurfer 8.2 data assets for the fixed single-T1 profile."""

import argparse
import hashlib
from importlib.resources import files
import json
import os
import re
import shutil
import tarfile
from pathlib import Path
from urllib.request import Request, urlopen

from fnit.weights import cache_dir, verify_file


# Relative path: (size, SHA-256, key extension for direct annex-backed files).
# Network assets passed full GET, byte-count, and SHA-256 checks against
# FreeSurfer 8.2 files; the small BrainstemSS LUT is bundled and hash-checked.
ASSET_FILES = {
    "ASegStatsLUT.txt": (
        3240, "36eb822e91174a7a4f99f22f17f19e6692225476b815aa0fdaf0622fd32e2511", ".txt"),
    "FreeSurferColorLUT.txt": (
        116487, "da55c6a47d316ee16d2609afd5c93cb8078e08e7e3afec5bb6283027d7c33e7b", ".txt"),
    "SubCorticalMassLUT.txt": (
        281, "65957c41244153d6614aeaf08c16ee956c09e2c8b5e198aa64c81946f75e93b6", ".txt"),
    "WMParcStatsLUT.txt": (
        25476, "1eb362b9d929797179106ba37eecaa8eff2830f1509be8e7b56238e6ac7c6dbf", ".txt"),
    "average/RB_all_2020-01-02.gca": (
        71651552, "2fcd276a39800f01f93a4c8828ae6d0a8cea3d8b8b9fe1599d4ee54e806be93e", ".gca"),
    "average/RB_all_withskull_2020_01_02.gca": (
        76631396, "0cc9b5a76f80555507ff86bfccce31bb7a308b8541dcad8c2a465fb915534870", ".gca"),
    "average/BrainstemSS/atlas/AtlasMesh.gz": (
        1726705, "90b0c6a6ade8aa388ef7c682b652ffc6bbd602271fd6b6068f47197514b2df3f", ".gz"),
    "average/BrainstemSS/atlas/AtlasDump.mgz": (
        48827, "14523eaf5e7f596be68d251a50f98ef277f5b7ef8904cca786a3efef8e9e3e7f", ".mgz"),
    "average/BrainstemSS/atlas/compressionLookupTable.txt": (
        1291, "8c343757d9ee13ed2d02daeb5f5f5fc764a6adb850b19b9d1b0ad352c96ca15b", ".txt"),
    "average/colortable_BA.txt": (
        810, "83aaf7a79ce4fdb98c9f6175b318dfaceed276ebf7422cdfaa195f1f71a2512f", ".txt"),
    "average/colortable_BA_thresh.txt": (
        915, "025b467483825cba420d67c475a7f0af8f7eb092ae53010e7adddfe4a853443b", ".txt"),
    "average/colortable_vpnl.txt": (
        793, "bf73534d4fe5b46e120611b8886061cf546d871f3620c225604649f6fc0f195d", ".txt"),
    "average/lh.CDaparc.atlas.acfb40.noaparc.i12.2016-08-02.gcs": (
        21724500, "f726bca3747676e465aaad4572e525dc5370d0a073c435fd41c1beaff2ad6add", ".gcs"),
    "average/lh.DKTaparc.atlas.acfb40.noaparc.i12.2016-08-02.gcs": (
        22003357, "6db52238d4c6c5867edd645daf23a7ce1bf753caffa792c9972db40af8229214", ".gcs"),
    "average/lh.DKaparc.atlas.acfb40.noaparc.i12.2016-08-02.gcs": (
        18842086, "721d6f53e8e3edae2c3798d52c5878026f1cdbe6cb286659ae48b920651a4cb6", ".gcs"),
    "average/lh.folding.atlas.acfb40.noaparc.i12.2016-08-02.tif": (
        2857642, "92f1dc820d778a67c143d4ea82597fd71a89ddc738dfa555b7ca32ed21e95c97", ".tif"),
    "average/mca-dura.prior.warp.mni152.1.0mm.lh.nii.gz": (
        41792, "0ea9f9ffdcbc38b139e5ca374dc322cb60773c355ffbccf21efd812d8c0ee458", ".nii.gz"),
    "average/mca-dura.prior.warp.mni152.1.0mm.rh.nii.gz": (
        41927, "5772a838e2fdd2ecf0c83de27bc572532cd8b9b8a1b00d919a385aab34e5710e", ".nii.gz"),
    "average/mni305.cor.mgz": (
        3285041, "5358b6adbf18d5c42b272b19bdcdda1a8f28b1d5ce4b719d539e86f35d1737e9", ".mgz"),
    "average/mni305.cor.stripped.mgz": (
        1342430, "fff93f13255a8d393c0e787fbfcfaf5eb379e88e955bda7e04e31552568878a4", ".mgz"),
    "average/mni_icbm152_nlin_asym_09c/reg-targets/mni152.1.0mm.cropped.nii.gz": (
        7745204, "ef89b7aa615dcb4a68a1cad725a9b99314227f9777ed86f42aae61e4e89da19a", ""),
    "average/mni_icbm152_nlin_asym_09c/reg-targets/mni152.1.0mm.nii.gz": (
        17977963, "e4e1a25b66fef6b2cde8e4916d4a03d4369de6f24b0d3d4736baffc3d4a8c575", ""),
    "average/mni_icbm152_nlin_asym_09c/reg-targets/reg.1.0mm.cropped.to.1.0mm.lta": (
        1620, "2a80b62a9baceaa8c4312bf46d5468353b10ed2990e1704028907700cb13d790", ""),
    "average/mni_icbm152_nlin_asym_09c/reg-targets/reg.1.0mm.to.1.0mm.cropped.lta": (
        1620, "2715856af855b0308350bbe32f8f932e3432d4c96acbbe107eee9dbb1c2b2e07", ""),
    "average/rh.CDaparc.atlas.acfb40.noaparc.i12.2016-08-02.gcs": (
        21898679, "878f4caae1f7d439ebeae5499537b9b5c1146bf7c369f5fb4cfc1df67e6d0b29", ".gcs"),
    "average/rh.DKTaparc.atlas.acfb40.noaparc.i12.2016-08-02.gcs": (
        21987412, "3f6e2ec1f24081f5a3ad2cafcdc1aaa156a2a635b78c30808dbeb8aaf9421498", ".gcs"),
    "average/rh.DKaparc.atlas.acfb40.noaparc.i12.2016-08-02.gcs": (
        18610793, "64e98c907af4242cb79b645a384c97ed37854b7b127a18ad406084b29dc486cf", ".gcs"),
    "average/rh.folding.atlas.acfb40.noaparc.i12.2016-08-02.tif": (
        2858470, "1064e4c0ae0440f5071b4369f29a09860f1abf6eddc90a271d10cd0ff7e60043", ".tif"),
    "average/vsinus.no-sp.prior.mni152.1.0mm.mgz": (
        269881, "b46661dda5cdde2a9c43cd2bad7ca096bd96bf2ba6c60f784d54cd8a75485126", ".0mm.mgz"),
    "lib/bem/ic4.tri": (
        207432, "95cc8e50f48df9b2a7a8558c9837becfa486f7562f2d9ae6fee4cc7aad341d13", ".tri"),
    "lib/bem/ic7.tri": (
        13368214, "cdd2761f0921a05d4959eb5b200fc3c65c0344a575809b0b09458299d90a1e6d", ".tri"),
    "subjects/fsaverage/surf/lh.sphere.reg": (
        5898546, "fddcf0eeecc6e0f62142164f7c0ac24fda1f0be8cb653d3f037970184c5ba98f", ".reg"),
    "subjects/fsaverage/surf/rh.sphere.reg": (
        5898546, "fddcf0eeecc6e0f62142164f7c0ac24fda1f0be8cb653d3f037970184c5ba98f", ".reg"),
    "subjects/fsaverage/label/lh.BA1_exvivo.label": (166374, "8884afe5b712013b7faf517d9c30950fb0d934737611cb39ee66ed4168935863", ""),
    "subjects/fsaverage/label/lh.BA1_exvivo.thresh.label": (46008, "a52bfb2ca169f96b2e611f5b6d0ffe6e822178c774b93c62daee700aa553f02f", ""),
    "subjects/fsaverage/label/lh.BA2_exvivo.label": (320897, "b4fc82af5491263068c0c12d04d5298c6fdc1f996cc832154905c1d4354145fd", ""),
    "subjects/fsaverage/label/lh.BA2_exvivo.thresh.label": (94863, "1fb1811a8a29ebcb5060e23f988aad62b8a764b085f3e183075d2358ce92276d", ""),
    "subjects/fsaverage/label/lh.BA3a_exvivo.label": (163588, "addeb991e3c083d5e27c8d14956654456fe3949f60122ce2b1ca635e75c3d52f", ""),
    "subjects/fsaverage/label/lh.BA3a_exvivo.thresh.label": (68084, "963d4eb18f9f3362966d998cd7b440372889481ba440b2e2894f266a2e2b6889", ""),
    "subjects/fsaverage/label/lh.BA3b_exvivo.label": (239495, "476ad4b6a72b4979b8a53545b1c4b1c6b82b1732dacfd51df3bcab05efe5410e", ""),
    "subjects/fsaverage/label/lh.BA3b_exvivo.thresh.label": (90376, "86aac6cb4b65109f71afd11bcac8b1c2369c4e5d57f3bdd74184d78fc553dc61", ""),
    "subjects/fsaverage/label/lh.BA44_exvivo.label": (167585, "62631737d99bf4c20c6586a9c13b17e43d4b53ea6d43b6f194c8697e0211d9f2", ""),
    "subjects/fsaverage/label/lh.BA44_exvivo.thresh.label": (83185, "8d7a087c409a08869aacd06a74be36b7d95bce3ef9ceedab6aaa6625f31323b0", ""),
    "subjects/fsaverage/label/lh.BA45_exvivo.label": (137141, "7c7a92b25ba2b7d1e21318d7b315039d9fa10d20527ad3740f83557745a2bfaa", ""),
    "subjects/fsaverage/label/lh.BA45_exvivo.thresh.label": (50422, "d68196b0aaaf83c964c8333e2c4198a3e490379a3002bc9d913dc08a89122020", ""),
    "subjects/fsaverage/label/lh.BA4a_exvivo.label": (231744, "59fa939431954f5fb85f9b9344cfece29f6ab60293f79b6331e7abb242370317", ""),
    "subjects/fsaverage/label/lh.BA4a_exvivo.thresh.label": (103857, "0d36c09329bd36395643c66d07f0bbf34e30e699ee6567c2770d00df3058e7f2", ""),
    "subjects/fsaverage/label/lh.BA4p_exvivo.label": (163313, "9dcf4ce34422bf6024cd70e31a8494fac428a1a14855c67c5812a0b1aa843632", ""),
    "subjects/fsaverage/label/lh.BA4p_exvivo.thresh.label": (69838, "1b1451f5e52847d3285783a68f3088ece65dee0bfd88695a7d35c6e2f5e31e94", ""),
    "subjects/fsaverage/label/lh.BA6_exvivo.label": (538881, "3432dfa143bfa910a251585c2359603e1972fa3e6acd7d7f8dd79c60b525d759", ""),
    "subjects/fsaverage/label/lh.BA6_exvivo.thresh.label": (311318, "673c58990444ab7924b9c6ea8a6e26bac548079c96740732df066d44e6b220ed", ""),
    "subjects/fsaverage/label/lh.MT_exvivo.label": (85306, "9149feef1c571b16bc0ac14795f3a4d3ca3367a4f2cd371706d831696e49a1b1", ""),
    "subjects/fsaverage/label/lh.MT_exvivo.thresh.label": (23084, "79699fad47bcbb6b8935a567bb8b1928ffa19b295c8b4330681e0839e6df2e2c", ""),
    "subjects/fsaverage/label/lh.V1_exvivo.label": (192278, "cd138deccaa711411a89f183d7d18b0f38c612c3ff454044dc26b2f21194c325", ""),
    "subjects/fsaverage/label/lh.V1_exvivo.thresh.label": (151679, "cfd0cb7f4f8a1b55681609b64f414d1030ba2f30dd499228e8d91eae96f9ba9c", ""),
    "subjects/fsaverage/label/lh.V2_exvivo.label": (337758, "123cf0910e014b7f830626e1475725c87a2315308f00c856420c4b6de8ed2d93", ""),
    "subjects/fsaverage/label/lh.V2_exvivo.thresh.label": (149833, "ade1be8c79444d764e804d13bdb8d4be0479b966c4226a9a785409c55069caf3", ""),
    "subjects/fsaverage/label/lh.entorhinal_exvivo.label": (57500, "4f34b9e31377d15ef78f786bfc0a9d553cb7ec4702450139a437acf266f4091b", ""),
    "subjects/fsaverage/label/lh.entorhinal_exvivo.thresh.label": (21693, "a0c864c87a9a6636c799c28b0df0a3166624632702bec8c34a9c6661cd820749", ""),
    "subjects/fsaverage/label/lh.perirhinal_exvivo.label": (55014, "b242367b90204b8d441264c0a07abd2e21f4884d9f214a3d936c37190111317a", ""),
    "subjects/fsaverage/label/lh.perirhinal_exvivo.thresh.label": (20742, "6ddc7237cfe35854eb30704ed5deb8d22e194ddc058b5386753617322027d1e0", ""),
    "subjects/fsaverage/label/rh.BA1_exvivo.label": (155760, "e13ca0b5263354a70fab89a90f9d662366ac718c27604f99a31a3ed487c1ff4d", ""),
    "subjects/fsaverage/label/rh.BA1_exvivo.thresh.label": (38863, "a8588c380b9981e6ff51f789d679a5297b94c091ef9cac35187ea838f0e58699", ""),
    "subjects/fsaverage/label/rh.BA2_exvivo.label": (265610, "5aec9b5dbcb35cdb44b384ac9e8ee194ca0a27cb93aa16a5d1a480ea6bf9c9a9", ""),
    "subjects/fsaverage/label/rh.BA2_exvivo.thresh.label": (119206, "c33ead1dcebd8c91aed9c9ebb03e5ef485edc8d69c5efececee6e29e3e60abda", ""),
    "subjects/fsaverage/label/rh.BA3a_exvivo.label": (156488, "472a099485fe350b6c84c6a5c63d391c28e12b2596ce1e9d467a5150619292d3", ""),
    "subjects/fsaverage/label/rh.BA3a_exvivo.thresh.label": (74978, "cd96191d63460a2e11ee6a0006389eadbb8a9f4f5a55fbe97b97ff675ea4880f", ""),
    "subjects/fsaverage/label/rh.BA3b_exvivo.label": (177307, "63c174c0f7d69b1a5e42f0db8b2232accaef4580ea1a2e1a9d1c235db8d654af", ""),
    "subjects/fsaverage/label/rh.BA3b_exvivo.thresh.label": (96608, "7d0c61837aa898154351842ecd032f8b9c63bbf5311461b7cfe06abbec7cd99c", ""),
    "subjects/fsaverage/label/rh.BA44_exvivo.label": (269815, "f13f41bc0b42fec614552701d4a728dd4358003a29fe7f29da1ff4d35c0113d6", ""),
    "subjects/fsaverage/label/rh.BA44_exvivo.thresh.label": (43100, "0e59b5513e7e388e53cbe79a7ff947b8b41e116d2a324d01850f08d17d8eaa4c", ""),
    "subjects/fsaverage/label/rh.BA45_exvivo.label": (211071, "788be813f4a949a15ba988698811dc861980b0a11386ef2c0c74f5398eadc819", ""),
    "subjects/fsaverage/label/rh.BA45_exvivo.thresh.label": (50322, "1cc883c2c40c6c492935e1b6bcd5f65375c2555e7bae1ce2cf74b36fae130806", ""),
    "subjects/fsaverage/label/rh.BA4a_exvivo.label": (224244, "bf88639551db4ea0455b487bbf7a427b2f1a0d1ba96fb41108079dd0728e39c0", ""),
    "subjects/fsaverage/label/rh.BA4a_exvivo.thresh.label": (60471, "4789c770f40af484816afbfb78a851a194f32c8d9be52b113064b13f5ba84d28", ""),
    "subjects/fsaverage/label/rh.BA4p_exvivo.label": (175244, "ca5e5d64c62dafda926ef1b631c99abf784dd0e725702b85b12d705e460a899d", ""),
    "subjects/fsaverage/label/rh.BA4p_exvivo.thresh.label": (65740, "69f1ad431d2ea5f0ab2d247203b3825b29fc3c810821089bcec39ce3bf5d27f9", ""),
    "subjects/fsaverage/label/rh.BA6_exvivo.label": (475963, "9730d02af8fa2de766d7445d154cbbea10db515cd9f2bc0a0c24faf3c8b38af6", ""),
    "subjects/fsaverage/label/rh.BA6_exvivo.thresh.label": (300267, "e439228197eab77e9d5cba83dda7864768d8c80b94d33438fb41bcc7c91e11d5", ""),
    "subjects/fsaverage/label/rh.MT_exvivo.label": (78819, "09ee210fff0ed9d913a28b3984a4d1a7682416fc44c111f02b50064bb894671a", ""),
    "subjects/fsaverage/label/rh.MT_exvivo.thresh.label": (11674, "18d2a133d66c65ecfcc9e96de6e1a3324e9c0845a37d63805f1d0dfaa2a680d3", ""),
    "subjects/fsaverage/label/rh.V1_exvivo.label": (189237, "0906451073e85f131ad809a0c68477546e7040a3882a4789bfeb2d7be71e72c2", ""),
    "subjects/fsaverage/label/rh.V1_exvivo.thresh.label": (140751, "d0be64272e50bc6014daa61dba149eddf97d606443da85dbd85a96ee50efd188", ""),
    "subjects/fsaverage/label/rh.V2_exvivo.label": (324193, "c5a77d07674240814bcab88d08a531e3a10ea26089afcad85f712e40219b2bbe", ""),
    "subjects/fsaverage/label/rh.V2_exvivo.thresh.label": (151056, "007df4237059f4bf3a6053535a6dd20fdb2bb1989185bd758e9620474c8fbb46", ""),
    "subjects/fsaverage/label/rh.entorhinal_exvivo.label": (46475, "2d4e7bb6ffed893edccc2c8ed37bc7a65703e59eec639b20ebb1e20ec02370e2", ""),
    "subjects/fsaverage/label/rh.entorhinal_exvivo.thresh.label": (31152, "870334737b498638501c1e9ec3c20e1ca2c73c03f1de17a7d26244d7b96459ee", ""),
    "subjects/fsaverage/label/lh.FG1.mpm.vpnl.label": (19982, "3e807da35c8272b44b52ff4732c4ecda49b712a95f611ffb1ba99337ec6b7d5f", ""),
    "subjects/fsaverage/label/lh.FG2.mpm.vpnl.label": (34086, "e66ae1795cd0b59f4268180c0d9990a898a4b5808f5f249b56ce09150bf729a9", ""),
    "subjects/fsaverage/label/lh.FG3.mpm.vpnl.label": (89996, "001d7eb1733fbd9d44df48ead11f32b5751365fe8a1d6d21c84d199a0f769c4c", ""),
    "subjects/fsaverage/label/lh.FG4.mpm.vpnl.label": (101583, "7593ff98336e328dc574c73e2c77962a51d73a568c79fad12f9bce98cfc127d9", ""),
    "subjects/fsaverage/label/lh.hOc1.mpm.vpnl.label": (180234, "ac2b3b33512cb9aab20aa5cf4dad70de2c8dd94000bc4d6bcc4160e1ee2f9183", ""),
    "subjects/fsaverage/label/lh.hOc2.mpm.vpnl.label": (137176, "b94415047e6ce9756ef852d0acbd8f6068c617ce185d81ab55dae7df0d59d038", ""),
    "subjects/fsaverage/label/lh.hOc3v.mpm.vpnl.label": (61482, "501ed11cb9764799df355049593932fad8becd4d130c239e8fa8381c4acacaeb", ""),
    "subjects/fsaverage/label/lh.hOc4v.mpm.vpnl.label": (48322, "69e8d4b6a1a6db6d81f282ca7ec8b2de3f6d295f6177f32dfc3f8412e23addfb", ""),
    "subjects/fsaverage/label/rh.FG1.mpm.vpnl.label": (25517, "284ad692e80ee1b9a88a938c37198b06e983b554846f148cd2dfeed935ae63b4", ""),
    "subjects/fsaverage/label/rh.FG2.mpm.vpnl.label": (34157, "e6fcb5563d7ecc9efc976174375e802b44cdb05271f5cbefa97cf48bc608b65b", ""),
    "subjects/fsaverage/label/rh.FG3.mpm.vpnl.label": (71590, "9b0b284d53c5ab827713ebe34ba9162539f6661ffced5e959c8f6721633dc0d1", ""),
    "subjects/fsaverage/label/rh.FG4.mpm.vpnl.label": (75176, "0bb4c93b1ba774d83d0018669fce3299901f2d64723a14392d83e0037623dbb5", ""),
    "subjects/fsaverage/label/rh.hOc1.mpm.vpnl.label": (166863, "077dd20fcb605fd9f68fa54ed196c2d51698f9d437293f7174de2c231fd03810", ""),
    "subjects/fsaverage/label/rh.hOc2.mpm.vpnl.label": (124947, "c54b111f4b59ba6ed3b11edf104a35d8da93e6e52185b82144f176d118518523", ""),
    "subjects/fsaverage/label/rh.hOc3v.mpm.vpnl.label": (57189, "2a1996ea90ada1ceb43d1f58d1136d35b544abe91989ac5dad75bcdf47ef05c4", ""),
    "subjects/fsaverage/label/rh.hOc4v.mpm.vpnl.label": (48095, "f698568b8d1264ae17b7228bd685153bf6fc6d2fa67fd79414cbe568b23066a8", ""),
    "subjects/fsaverage/label/rh.perirhinal_exvivo.label": (33698, "5f7debb15d42d60ebc0fc4e4a02d383704a6f20995f58a5d59bb8220620fc9da", ""),
    "subjects/fsaverage/label/rh.perirhinal_exvivo.thresh.label": (13128, "e8eaa170f7d1aa232a4d4879e00354171a901c6c67a5b159b4724df12e694034", ""),
}
ANNEX_BASE = "https://surfer.nmr.mgh.harvard.edu/pub/dist/freesurfer/repo/annex.git/annex/objects"
BUNDLED_BRAINSTEM_LUT = "average/BrainstemSS/atlas/compressionLookupTable.txt"
SOURCE_BASE = ("https://raw.githubusercontent.com/freesurfer/freesurfer/"
               "d932c45b7941662ea380a05efef580568b98d41a/distribution")
FSAVERAGE_BASE = ("https://www.freesurfer.net/pub/dist/freesurfer/"
                  "tutorial_versions_centos6/freesurfer")
SOURCE_FILES = frozenset({
    "ASegStatsLUT.txt",
    "FreeSurferColorLUT.txt",
    "SubCorticalMassLUT.txt",
    "WMParcStatsLUT.txt",
    "average/colortable_BA.txt",
    "average/colortable_BA_thresh.txt",
    "average/colortable_vpnl.txt",
})
FSAVERAGE_ARCHIVE_MEMBERS = frozenset(
    name for name in ASSET_FILES
    if name.startswith("subjects/fsaverage/label/")
    and (".mpm.vpnl." in name or "rh.perirhinal_exvivo" in name))
FSAVERAGE_FILES = frozenset(
    name for name in ASSET_FILES
    if name.startswith("subjects/fsaverage/label/")
    and name not in FSAVERAGE_ARCHIVE_MEMBERS)
ARCHIVE_MEMBERS = frozenset(name for name in ASSET_FILES
                            if name.startswith("average/mni_icbm152_nlin_asym_09c/reg-targets/"))
ARCHIVE_SIZE = 514649342
ARCHIVE_SHA256 = "29f8b3dec88feaa133c65ee9342fd9d875cac4e9c08e43a7537cd5d614b227d8"
FSAVERAGE_ARCHIVE_SIZE = 320193429
FSAVERAGE_ARCHIVE_SHA256 = "586cbe3513db2872ad885486a042ebbde1cb5ca66dd3255994e8736901ce147f"


# The fixed recon-all profile and optional BrainstemSS data have verified sources.
PENDING_FILES = {}


def _annex_url(size, sha256, extension):
    key = f"SHA256E-s{size}--{sha256}{extension}"
    digest = hashlib.md5(key.encode("ascii")).hexdigest()
    return f"{ANNEX_BASE}/{digest[:3]}/{digest[3:6]}/{key}/{key}"


def asset_url(name):
    if name == BUNDLED_BRAINSTEM_LUT:
        raise ValueError(f"{name} is included in the FNIT package")
    if name in SOURCE_FILES:
        return f"{SOURCE_BASE}/{name}"
    if name in FSAVERAGE_FILES:
        return f"{FSAVERAGE_BASE}/{name}"
    if name in ARCHIVE_MEMBERS:
        return _annex_url(ARCHIVE_SIZE, ARCHIVE_SHA256, ".tar.gz")
    if name in FSAVERAGE_ARCHIVE_MEMBERS:
        return _annex_url(FSAVERAGE_ARCHIVE_SIZE, FSAVERAGE_ARCHIVE_SHA256, ".tar.gz")
    size, sha256, extension = ASSET_FILES[name]
    return _annex_url(size, sha256, extension)


def configured_dir():
    config = cache_dir() / "recon_all_assets.json"
    if not config.is_file():
        return None
    value = json.loads(config.read_text(encoding="utf-8"))["directory"]
    if not isinstance(value, str) or not Path(value).is_absolute():
        raise ValueError(f"Invalid asset directory in {config}")
    return Path(value)


def save_config(directory):
    config = cache_dir() / "recon_all_assets.json"
    config.parent.mkdir(parents=True, exist_ok=True)
    part = config.with_name(config.name + ".tmp")
    part.write_text(json.dumps({"directory": str(directory)}, indent=2) + "\n", encoding="utf-8")
    part.replace(config)


def download_asset(name, directory, verify_only=False):
    size, sha256, _ = ASSET_FILES[name]
    target = Path(directory) / name
    if verify_file(target, size, sha256):
        return target
    if verify_only:
        raise ValueError(f"Missing or invalid reconstruction asset: {target}")
    if name == BUNDLED_BRAINSTEM_LUT:
        data = files("fnit").joinpath("gems/data/brainstem_compressionLookupTable.txt").read_bytes()
        if len(data) != size or hashlib.sha256(data).hexdigest() != sha256:
            raise ValueError("Bundled brainstem lookup table failed SHA-256 verification")
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(data)
        return target
    if name in ARCHIVE_MEMBERS:
        archive = Path(directory) / ".downloads" / "mni_icbm152_nlin_asym_09c.tar.gz"
        archive_size, archive_sha = ARCHIVE_SIZE, ARCHIVE_SHA256
        members, prefix = ARCHIVE_MEMBERS, "average/"
    elif name in FSAVERAGE_ARCHIVE_MEMBERS:
        archive = Path(directory) / ".downloads" / "fsaverage.tar.gz"
        archive_size, archive_sha = FSAVERAGE_ARCHIVE_SIZE, FSAVERAGE_ARCHIVE_SHA256
        members, prefix = FSAVERAGE_ARCHIVE_MEMBERS, "subjects/"
    else:
        return _download_verified(asset_url(name), target, size, sha256)
    _download_verified(asset_url(name), archive, archive_size, archive_sha)
    _extract_archive_members(archive, Path(directory), members, prefix)
    archive.unlink()
    if not verify_file(target, size, sha256):
        raise ValueError(f"Archive member failed size or SHA-256 verification: {name}")
    return target


def _download_verified(url, target, size, sha256):
    if verify_file(target, size, sha256):
        return target
    target.parent.mkdir(parents=True, exist_ok=True)
    part = target.with_name(target.name + ".part")
    if verify_file(part, size, sha256):
        part.replace(target)
        return target
    if part.exists() and part.stat().st_size > size:
        part.unlink()
    if part.exists() and part.stat().st_size == size:
        part.unlink()
    offset = part.stat().st_size if part.exists() else 0
    headers = {"User-Agent": "Mozilla/5.0"}
    if offset:
        headers["Range"] = f"bytes={offset}-"
    with urlopen(Request(url, headers=headers), timeout=60) as response:
        if offset and response.status == 206:
            match = re.fullmatch(r"bytes (\d+)-(\d+)/(\d+)",
                                 response.headers.get("Content-Range", ""))
            if not match or int(match[1]) != offset or int(match[3]) != size:
                raise ValueError(f"Unexpected HTTP Content-Range for {target}")
            mode = "ab"
        elif response.status == 200:
            mode = "wb"
        else:
            raise ValueError(f"Unexpected HTTP status {response.status} for {target}")
        with part.open(mode) as stream:
            for block in iter(lambda: response.read(8 * 1024 * 1024), b""):
                stream.write(block)
    if not verify_file(part, size, sha256):
        part.unlink(missing_ok=True)
        raise ValueError(f"Downloaded asset failed size or SHA-256 verification: {target}")
    part.replace(target)
    return target


def _extract_archive_members(archive, directory, members, prefix):
    seen = set()
    with tarfile.open(archive, "r|gz") as stream:
        for member in stream:
            name = prefix + member.name.removeprefix("./")
            if name not in members:
                continue
            if not member.isfile() or name in seen:
                raise ValueError(f"Invalid atlas archive member: {name}")
            seen.add(name)
            size, sha256, _ = ASSET_FILES[name]
            target = directory / name
            if verify_file(target, size, sha256):
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            part = target.with_name(target.name + ".part")
            with stream.extractfile(member) as source, part.open("wb") as output:
                shutil.copyfileobj(source, output)
            if not verify_file(part, size, sha256):
                part.unlink(missing_ok=True)
                raise ValueError(f"Archive member failed size or SHA-256 verification: {name}")
            part.replace(target)
    if seen != members:
        raise ValueError(f"Atlas archive is missing expected members: {sorted(members - seen)}")


# Data for native_free.run_recon_all_python and the validated MCA/dura,
# venous-sinus segmentation stages. Other verified assets remain available
# through --all for isolated stage validation.
CORE_ASSETS = (
    "ASegStatsLUT.txt", "FreeSurferColorLUT.txt",
    "SubCorticalMassLUT.txt", "WMParcStatsLUT.txt",
    "average/RB_all_2020-01-02.gca", "average/mni305.cor.stripped.mgz",
    *(f"average/{hemi}.{atlas}.atlas.acfb40.noaparc.i12.2016-08-02.gcs"
      for hemi in ("lh", "rh") for atlas in ("DKaparc", "CDaparc", "DKTaparc")),
    *(f"average/{hemi}.folding.atlas.acfb40.noaparc.i12.2016-08-02.tif"
      for hemi in ("lh", "rh")),
    "lib/bem/ic4.tri", "lib/bem/ic7.tri",
    "average/mca-dura.prior.warp.mni152.1.0mm.lh.nii.gz",
    "average/mca-dura.prior.warp.mni152.1.0mm.rh.nii.gz",
    "average/vsinus.no-sp.prior.mni152.1.0mm.mgz",
)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--asset", action="append", choices=ASSET_FILES,
                        help="Asset to install; repeat to select multiple (default: Python recon-all assets)")
    parser.add_argument("--all", action="store_true", help="Install all verified stage-validation assets")
    parser.add_argument("--dest", type=Path, help="Asset directory; saved for later runs")
    parser.add_argument("--verify-only", action="store_true", help="Check local files without downloading")
    args = parser.parse_args(argv)
    if args.asset and args.all:
        parser.error("--asset and --all are mutually exclusive")
    directory = Path(args.dest or os.environ.get("FNIT_ASSETS")
                     or configured_dir() or cache_dir() / "recon_all_assets").expanduser().resolve()
    for name in (args.asset or (ASSET_FILES if args.all else CORE_ASSETS)):
        print(f"Verified {download_asset(name, directory, verify_only=args.verify_only)}")
    if not args.verify_only:
        save_config(directory)
        print(f"Configured reconstruction asset directory: {directory}")


if __name__ == "__main__":
    main()
