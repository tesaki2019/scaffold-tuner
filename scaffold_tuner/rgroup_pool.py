# -*- coding: utf-8 -*-
"""
Unified R-group pool for Scaffold-tuner v2.

All 25 R-groups are treated as ONE common candidate pool.
Historical descriptor-specific categories are not used for candidate selection.

Recommended workflow:
    parent R-group
      -> evaluate replacements against the unified pool
      -> select candidates consistent with requested descriptor delta
      -> build whole molecule
      -> validate exact descriptor delta with RDKit

Evidence metadata is documentary only and MUST NOT affect candidate ranking
or selection.

Important wording for manuscripts:
    "literature-informed and manually curated unified R-group pool"
Do NOT claim that all 25 entries are individually validated replacement rules.
"""

REFERENCES = {'takeuchi2021': {'citation': 'Takeuchi K, Kunimoto R, Bajorath J. Systematic mapping of R-group space enables the generation of an R-group replacement system for medicinal chemistry. Eur J Med Chem. 2021;225:113771.', 'doi': '10.1016/j.ejmech.2021.113771'}, 'kenny2022': {'citation': 'Kenny PW. Hydrogen-Bond Donors in Drug Design. J Med Chem. 2022;65(21):14261-14275.', 'doi': '10.1021/acs.jmedchem.2c01147'}, 'ward2014': {'citation': 'Ward SE, Beswick P. What does the aromatic ring number mean for drug design? Expert Opin Drug Discov. 2014;9(9):995-1003.', 'doi': '10.1517/17460441.2014.932346'}, 'pollock2015': {'citation': 'Pollock J, et al. Rational Design of Orthogonal Multipolar Interactions with Fluorine in Protein-Ligand Complexes. J Med Chem. 2015;58(18):7465-7474.', 'doi': '10.1021/acs.jmedchem.5b00975'}, 'lovering2009': {'citation': 'Lovering F, Bikker J, Humblet C. Escape from Flatland: Increasing Saturation as an Approach to Improving Clinical Success. J Med Chem. 2009;52(21):6752-6756.', 'doi': '10.1021/jm901241e'}, 'veber2002': {'citation': 'Veber DF, et al. Molecular Properties That Influence the Oral Bioavailability of Drug Candidates. J Med Chem. 2002;45(12):2615-2623.', 'doi': '10.1021/jm020017n'}}

R_GROUP_POOL = [{'id': 1, 'name': 'amino', 'smiles': '[*:1]N', 'evidence_level': 'explicit_rgroup_or_replacement', 'references': ['takeuchi2021', 'kenny2022'], 'evidence_note': "Amino is named in Takeuchi's preferred replacement discussion; NH2/NH donor motifs are explicitly discussed by Kenny."}, {'id': 2, 'name': 'methylamino', 'smiles': '[*:1]NC', 'evidence_level': 'literature_rationale', 'references': ['kenny2022'], 'evidence_note': 'N-H donor chemistry is supported, but this exact R-group replacement was not verified in the supplied papers.'}, {'id': 3, 'name': 'hydroxy', 'smiles': '[*:1]O', 'evidence_level': 'explicit_rgroup_or_replacement', 'references': ['takeuchi2021', 'kenny2022'], 'evidence_note': "Hydroxyl is named in Takeuchi's preferred replacements and OH donor motifs are explicitly discussed by Kenny."}, {'id': 4, 'name': '3-pyrrolyl', 'smiles': '[*:1]c1ccc[nH]1', 'evidence_level': 'explicit_motif', 'references': ['kenny2022'], 'evidence_note': 'Pyrrole is explicitly listed as an H-bond donor motif; not verified as a preferred R-group replacement.'}, {'id': 5, 'name': 'indolyl', 'smiles': '[*:1]c1ccc2[nH]ccc2c1', 'evidence_level': 'explicit_motif', 'references': ['kenny2022'], 'evidence_note': 'Indole is explicitly listed as an H-bond donor motif; not verified as a preferred R-group replacement.'}, {'id': 6, 'name': 'methyl', 'smiles': '[*:1]C', 'evidence_level': 'explicit_rgroup_or_replacement', 'references': ['takeuchi2021'], 'evidence_note': 'Methyl is explicitly reported among the most frequent R-groups.'}, {'id': 7, 'name': 'ethyl', 'smiles': '[*:1]CC', 'evidence_level': 'literature_rationale', 'references': ['veber2002'], 'evidence_note': 'Simple alkyl R-group retained from the prior pool; exact replacement not verified.'}, {'id': 8, 'name': 'fluoro', 'smiles': '[*:1]F', 'evidence_level': 'explicit_rgroup_or_replacement', 'references': ['takeuchi2021', 'pollock2015'], 'evidence_note': "Fluorine is named in Takeuchi's replacement hierarchy and directly studied as a substitution by Pollock."}, {'id': 9, 'name': 'chloro', 'smiles': '[*:1]Cl', 'evidence_level': 'explicit_rgroup_or_replacement', 'references': ['takeuchi2021'], 'evidence_note': 'Chlorine is explicitly named as a preferred replacement for methoxy.'}, {'id': 10, 'name': 'isopropyl', 'smiles': '[*:1]C(C)C', 'evidence_level': 'literature_rationale', 'references': ['lovering2009'], 'evidence_note': 'Saturated alkyl design rationale is supported; exact replacement not verified.'}, {'id': 11, 'name': 'methoxy', 'smiles': '[*:1]OC', 'evidence_level': 'explicit_rgroup_or_replacement', 'references': ['takeuchi2021'], 'evidence_note': 'Methoxy is explicitly used as a central R-group in the replacement hierarchy.'}, {'id': 12, 'name': 'cyano', 'smiles': '[*:1]C#N', 'evidence_level': 'literature_rationale', 'references': [], 'evidence_note': 'Retained from the prior curated pool; exact literature support not verified in the supplied papers.'}, {'id': 13, 'name': 'acetyl', 'smiles': '[*:1]C(=O)C', 'evidence_level': 'literature_rationale', 'references': [], 'evidence_note': 'Retained from the prior curated pool; exact literature support not verified in the supplied papers.'}, {'id': 14, 'name': 'dimethylamino', 'smiles': '[*:1]N(C)C', 'evidence_level': 'literature_rationale', 'references': ['kenny2022'], 'evidence_note': 'Amine/H-bond chemistry is supported; exact R-group replacement not verified.'}, {'id': 15, 'name': 'methylsulfonyl', 'smiles': '[*:1]S(=O)(=O)C', 'evidence_level': 'literature_rationale', 'references': ['kenny2022'], 'evidence_note': 'Sulfonyl/sulfonamide H-bond chemistry is discussed; exact methylsulfonyl replacement not verified.'}, {'id': 16, 'name': 'phenyl', 'smiles': '[*:1]c1ccccc1', 'evidence_level': 'explicit_rgroup_or_replacement', 'references': ['takeuchi2021', 'ward2014'], 'evidence_note': 'Phenyl is explicitly among the most frequent R-groups; aromatic-ring design is reviewed by Ward & Beswick.'}, {'id': 17, 'name': 'pyridyl', 'smiles': '[*:1]c1ccncc1', 'evidence_level': 'literature_rationale', 'references': ['ward2014'], 'evidence_note': 'Heteroaromatic design is supported; exact pyridyl replacement not verified in the supplied papers.'}, {'id': 18, 'name': '2-furyl', 'smiles': '[*:1]c1ccco1', 'evidence_level': 'literature_rationale', 'references': ['ward2014'], 'evidence_note': 'Heteroaromatic design is supported; exact furyl replacement not verified.'}, {'id': 19, 'name': '2-thienyl', 'smiles': '[*:1]c1cccs1', 'evidence_level': 'literature_rationale', 'references': ['ward2014'], 'evidence_note': 'Heteroaromatic design is supported; exact thienyl replacement not verified.'}, {'id': 20, 'name': 'n-propyl', 'smiles': '[*:1]CCC', 'evidence_level': 'literature_rationale', 'references': ['veber2002'], 'evidence_note': 'Flexible alkyl R-group retained from the prior pool; exact replacement not verified.'}, {'id': 21, 'name': 'n-butyl', 'smiles': '[*:1]CCCC', 'evidence_level': 'literature_rationale', 'references': ['veber2002'], 'evidence_note': 'Flexible alkyl-chain rationale is consistent with rotatable-bond considerations.'}, {'id': 22, 'name': 'methoxymethyl', 'smiles': '[*:1]COC', 'evidence_level': 'literature_rationale', 'references': ['veber2002'], 'evidence_note': 'Flexible heteroatom-containing side chain retained from the prior pool; exact replacement not verified.'}, {'id': 23, 'name': 'acetonyl', 'smiles': '[*:1]CC(=O)C', 'evidence_level': 'literature_rationale', 'references': ['veber2002'], 'evidence_note': 'Flexible carbonyl-containing side chain retained from the prior pool; exact replacement not verified.'}, {'id': 24, 'name': 'cyclopropyl', 'smiles': '[*:1]C1CC1', 'evidence_level': 'literature_rationale', 'references': ['lovering2009'], 'evidence_note': 'Rigid saturated ring is consistent with saturation/3D design rationale; exact replacement not verified.'}, {'id': 25, 'name': 'cyclohexyl', 'smiles': '[*:1]C1CCCCC1', 'evidence_level': 'literature_rationale', 'references': ['lovering2009'], 'evidence_note': 'Saturated cycloalkyl design rationale is supported; exact replacement not verified.'}]

R_GROUP_SMILES = tuple(item["smiles"] for item in R_GROUP_POOL)
R_GROUP_BY_NAME = {item["name"]: item for item in R_GROUP_POOL}

def get_rgroup_pool():
    return list(R_GROUP_POOL)

def get_rgroup_smiles():
    return list(R_GROUP_SMILES)

def validate_pool():
    required = {"id", "name", "smiles", "evidence_level", "references", "evidence_note"}
    names = set()
    smiles = set()

    for item in R_GROUP_POOL:
        missing = required - set(item)
        if missing:
            raise ValueError(f"Missing fields for R-group {item!r}: {sorted(missing)}")
        if item["name"] in names:
            raise ValueError(f"Duplicate R-group name: {item['name']}")
        if item["smiles"] in smiles:
            raise ValueError(f"Duplicate R-group SMILES: {item['smiles']}")
        names.add(item["name"])
        smiles.add(item["smiles"])

    if len(R_GROUP_POOL) != 25:
        raise ValueError(f"Expected 25 R-groups, found {len(R_GROUP_POOL)}")

    return True
