import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem

from .utils.decorator import timeit

# Modified from code under the following license:
__author__ = "Steven Kearnes"
__copyright__ = "Copyright 2014, Stanford University"
__license__ = "3-clause BSD"


class RDKitConformerGenerator:
    def __init__(
        self,
        max_low_energy_conformers: int = 1000,
        num_conformers_to_embed: int = 5000,
        rmsd_threshold: float = 0.5,
        skip_cistrans_opt: bool = True,
        num_threads: int = 0,  # 0 uses all available cores
    ) -> None:
        self.max_low_energy_conformers = max_low_energy_conformers
        self.num_conformers_to_embed = num_conformers_to_embed
        self.rmsd_threshold = rmsd_threshold
        self.num_threads = num_threads

        # Use ETKDG method for macrocycles
        self.embedding_params = AllChem.ETKDGv3()
        self.embedding_params.maxIterations = 10 * self.num_conformers_to_embed
        self.embedding_params.pruneRmsThresh = 0.01
        self.embedding_params.useRandomCoords = True
        self.embedding_params.useMacrocycleTorsions = True

        self.skip_cistrans = skip_cistrans_opt
        self.embedding_params.numThreads = self.num_threads

    def __call__(self, mol: Chem.Mol) -> Chem.Mol:
        return self.generate_conformers(mol)

    def generate_conformers(
        self,
        mol: Chem.Mol,
    ) -> Chem.Mol:
        mol = self.embed_molecule(mol)
        if not mol.GetNumConformers():
            raise RuntimeError("No conformers generated")
        print(f"Generated {mol.GetNumConformers()} conformers")

        mol = self.optimize_conformers(mol, skip_cistrans=self.skip_cistrans)
        mol = self.filter_conformers(mol)
        print(f"Filtered to {mol.GetNumConformers()} conformers")

        return mol

    @timeit(unit="min")
    def embed_molecule(self, mol: Chem.Mol) -> Chem.Mol:
        AllChem.EmbedMultipleConfs(
            mol,
            numConfs=self.num_conformers_to_embed,
            params=self.embedding_params,
        )
        return mol

    @timeit(unit="min")
    def optimize_conformers(
        self, mol: Chem.Mol, skip_cistrans: bool = True
    ) -> Chem.Mol:
        # Don't optimize if cis/trans stereo
        if skip_cistrans and not all(
            bond.GetStereo() is Chem.BondStereo.STEREONONE for bond in mol.GetBonds()
        ):
            return mol
        AllChem.MMFFOptimizeMoleculeConfs(mol, numThreads=self.num_threads)
        return mol

    def get_conformer_energies(self, mol: Chem.Mol) -> np.ndarray:
        AllChem.MMFFSanitizeMolecule(mol)
        mmff_props = AllChem.MMFFGetMoleculeProperties(mol)

        energies = np.empty((mol.GetNumConformers(),), dtype=float)
        for i, conf in enumerate(mol.GetConformers()):
            ff = AllChem.MMFFGetMoleculeForceField(mol, mmff_props, confId=conf.GetId())
            energy = ff.CalcEnergy()
            energies[i] = energy

        return energies

    @timeit(unit="min")
    def filter_conformers(self, mol: Chem.Mol) -> Chem.Mol:
        confs = np.array(mol.GetConformers())
        energies = self.get_conformer_energies(mol)
        sort = np.argsort(energies)

        mol_no_h = Chem.RemoveHs(mol)  # Remove hydrogens to speed up substruct match
        keep = [sort[0]]  # Always keep the lowest-energy conformer
        discard = []

        # Precompute atom map for GetBestRMS
        atom_idxs = [a.GetIdx() for a in mol_no_h.GetAtoms()]
        matches = mol_no_h.GetSubstructMatches(mol_no_h, uniquify=False)
        atom_map = [list(zip(match, atom_idxs)) for match in matches]

        for i in sort[1:]:
            if len(keep) >= self.max_low_energy_conformers:
                discard.append(i)
                continue

            # Want GetBestRMS in case there's symmetry
            rmsds = [
                AllChem.GetBestRMS(
                    mol_no_h,
                    mol_no_h,
                    confs[j].GetId(),
                    confs[i].GetId(),
                    map=atom_map,
                )
                for j in keep
            ]
            if np.all(np.array(rmsds) >= self.rmsd_threshold):
                keep.append(i)
            else:
                discard.append(i)

        new_mol = Chem.Mol(mol)
        new_mol.RemoveAllConformers()
        conf_ids = [conf.GetId() for conf in confs]
        for i in keep:
            conf = mol.GetConformer(conf_ids[i])
            new_mol.AddConformer(conf, assignId=True)

        return new_mol


def generate_rdkit_conformers(mol: Chem.Mol, **kwargs) -> Chem.Mol:
    conformer_generator = RDKitConformerGenerator(**kwargs)
    return conformer_generator(mol)
