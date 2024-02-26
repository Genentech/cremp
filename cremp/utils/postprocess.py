import math
import pickle
import re
import tarfile
from pathlib import Path
from typing import Any, IO

from rdkit import Chem

from .chem_utils import load_mol

NUMBER_PATTERN = re.compile(r"\d+")


def parse_energies(file: IO[bytes], encoding: str = "utf-8") -> list[float]:
    energies = []

    for line in file:
        line = line.decode(encoding)

        try:
            energy = float(line)  # Works for number of atoms and energy
        except ValueError:
            continue

        energies.append(energy)

    # Keep every 2nd item to filter out number of atoms
    energies = [energies[i] for i in range(1, len(energies), 2)]
    return energies


def parse_ensemble_data(
    file: IO[bytes], encoding: str = "utf-8", include_time: bool = False
) -> dict[str, Any] | None:
    read = False

    ensemble_data: dict[str, Any] = {}
    conf_data: list[dict[str, Any]] = []

    ensemble_patterns_and_types = {
        "ensembleenergy": ("ensemble average energy", float),
        "ensembleentropy": ("ensemble entropy", float),
        "ensemblefreeenergy": ("ensemble free energy", float),
        "lowestenergy": ("E lowest", float),
        "poplowestpct": ("population of lowest in %", float),
        "temperature": ("T /K", float),
        "uniqueconfs": ("number of unique conformers", int),
        "totalconfs": ("total number unique points", int),
    }
    conf_idxs_and_types = {
        "set": (5, int),
        "degeneracy": (6, int),
        "totalenergy": (2, float),
        "relativeenergy": (1, float),
        "boltzmannweight": (4, float),
        "conformerweights": (3, lambda x: [float(x)]),
    }

    for line in file:
        line = line.decode(encoding).strip()

        if read:
            if line.startswith("Erel/kcal"):
                line = next(file).decode(encoding).strip()
                conf_info = line.split()
                single_conf_data: dict[str, Any] = {}

                while len(conf_info) > 4:
                    if len(conf_info) == 8:
                        # Append data for a single conformer upon encountering a new one
                        if single_conf_data:
                            if (
                                len(single_conf_data["conformerweights"])
                                != single_conf_data["degeneracy"]
                            ):
                                raise ValueError(f"Fewer rotamers than expected")
                            conf_data.append(single_conf_data)

                        single_conf_data = {
                            key: type_(conf_info[idx])
                            for key, (idx, type_) in conf_idxs_and_types.items()
                        }
                    else:
                        single_conf_data["conformerweights"].append(float(conf_info[3]))

                    line = next(file).decode(encoding).strip()
                    conf_info = line.split()

                # Append final conformer after exiting loop
                if single_conf_data:
                    if (
                        len(single_conf_data["conformerweights"])
                        != single_conf_data["degeneracy"]
                    ):
                        raise ValueError(f"Fewer rotamers than expected")
                    conf_data.append(single_conf_data)

            for key, (pattern, type_) in ensemble_patterns_and_types.items():
                if line.startswith(pattern):
                    ensemble_data[key] = type_(line.split()[-1])

            if include_time:
                if line.startswith("Overall wall time"):
                    h, m, s = map(int, NUMBER_PATTERN.findall(line))
                    wall_time = h + m / 60 + s / 3600
                    ensemble_data["wall_time"] = wall_time

        if "Final Geometry Optimization" in line:
            read = True

    if not conf_data or not ensemble_data:
        return None
    ensemble_data["conformers"] = conf_data

    if (
        sum(len(d["conformerweights"]) for d in conf_data)
        != ensemble_data["totalconfs"]
    ):
        raise ValueError("Fewer rotamers than expected")

    return ensemble_data


def parse_dir(
    path: str | Path,
    mol_file_name: str = "mol_crest.pickle",
    conformers_file_name: str = "crest/crest_conformers.xyz",
    crest_log_file_name: str = "crest/crest.out",
    include_time: bool = False,
) -> dict[str, Any] | None:
    dir_path = Path(path)

    mol_path = dir_path / mol_file_name
    if mol_path.exists():
        mol = load_mol(mol_path)
    else:
        return None

    crest_log_path = dir_path / crest_log_file_name
    if crest_log_path.exists():
        with open(crest_log_path, "rb") as f:
            ensemble_data = parse_ensemble_data(f, include_time=include_time)
    else:
        return None

    if ensemble_data is None:
        return None

    if not (
        ensemble_data["uniqueconfs"]
        == len(ensemble_data["conformers"])
        == mol.GetNumConformers()
    ):
        raise ValueError(f"Mismatch in number of conformers in '{path}'")

    ensemble_data["smiles"] = Chem.MolToSmiles(Chem.RemoveHs(mol))
    ensemble_data["rd_mol"] = mol
    ensemble_data["charge"] = sum(atom.GetFormalCharge() for atom in mol.GetAtoms())

    # Attempt to use more sig figs for energies
    crest_conformers_path = dir_path / conformers_file_name
    if crest_conformers_path.exists():
        with open(crest_conformers_path, "rb") as f:
            energies = parse_energies(f)

        if energies:
            for single_conf_data, energy in zip(ensemble_data["conformers"], energies):
                if math.isclose(single_conf_data["totalenergy"], energy, abs_tol=1e-5):
                    single_conf_data["totalenergy"] = energy

    return ensemble_data


def parse_archive(
    path: str | Path,
    mol_file_name: str = "mol_crest.pickle",
    conformers_file_name: str = "crest/crest_conformers.xyz",
    crest_log_file_name: str = "crest/crest.out",
    include_time: bool = False,
) -> dict[str, Any] | None:
    with tarfile.open(path, "r:gz") as tar:
        try:
            mol_member = tar.getmember(mol_file_name)
        except KeyError:
            return None

        with tar.extractfile(mol_member) as f:
            mol = pickle.load(f)

        try:
            crest_log_member = tar.getmember(crest_log_file_name)
        except KeyError:
            return None

        with tar.extractfile(crest_log_member) as f:
            ensemble_data = parse_ensemble_data(f, include_time=include_time)

        if ensemble_data is None:
            return None

        if not (
            ensemble_data["uniqueconfs"]
            == len(ensemble_data["conformers"])
            == mol.GetNumConformers()
        ):
            raise ValueError(f"Mismatch in number of conformers in '{path}'")

        ensemble_data["smiles"] = Chem.MolToSmiles(Chem.RemoveHs(mol))
        ensemble_data["rd_mol"] = mol
        ensemble_data["charge"] = sum(atom.GetFormalCharge() for atom in mol.GetAtoms())

        # Attempt to use more sig figs for energies
        try:
            crest_conformers_member = tar.getmember(conformers_file_name)
        except KeyError:
            pass
        else:
            with tar.extractfile(crest_conformers_member) as f:
                energies = parse_energies(f)

            if energies:
                for single_conf_data, energy in zip(
                    ensemble_data["conformers"], energies
                ):
                    if math.isclose(
                        single_conf_data["totalenergy"], energy, abs_tol=1e-5
                    ):
                        single_conf_data["totalenergy"] = energy

        return ensemble_data
