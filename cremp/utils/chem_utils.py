import pickle
from pathlib import Path

from rdkit import Chem


def canonicalize_smiles(smi: str) -> str:
    return Chem.MolToSmiles(Chem.MolFromSmiles(smi))


def save_mol(mol: Chem.Mol, path: str | Path) -> None:
    with open(path, "wb") as f:
        pickle.dump(mol, f)


def load_mol(path: str | Path) -> Chem.Mol:
    with open(path, "rb") as f:
        return pickle.load(f)


def conf_to_xyz(conformer: Chem.Conformer, comment: str = "") -> str:
    if not conformer.HasOwningMol():
        raise ValueError("Conformer must belong to a molecule")
    mol = conformer.GetOwningMol()

    xyz = [f"{conformer.GetNumAtoms()}\n{comment}\n"]
    xyz.extend(
        f"{atom.GetSymbol()} {x: .8f} {y: .8f} {z: .8f}\n"
        for atom, (x, y, z) in zip(mol.GetAtoms(), conformer.GetPositions())
    )
    xyz = "".join(xyz)

    return xyz


def mol_to_xyz(mol: Chem.Mol) -> str:
    return "".join(map(conf_to_xyz, mol.GetConformers()))


def write_coordinates(
    conformer: Chem.Conformer, path: str | Path, comment: str = ""
) -> None:
    xyz = conf_to_xyz(conformer, comment=comment)

    with open(path, "w") as f:
        f.write(xyz)


def mol_to_sdf(path: Path, mol: Chem.Mol) -> None:
    writer = Chem.SDWriter(str(path))
    for cid in range(mol.GetNumConformers()):
        writer.write(mol, confId=cid)
