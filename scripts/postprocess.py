#!/usr/bin/env python
import json
import multiprocessing
import pickle
from functools import partial
from pathlib import Path
from typing import Any

import pandas as pd
import typer
from openeye import oechem
from rdkit import Chem
from rdkit.Chem import rdDetermineBonds

from cremp.utils.chem_utils import canonicalize_smiles, mol_to_xyz, mol_to_sdf
from cremp.utils.postprocess import parse_archive, parse_dir

IMS = oechem.oemolistream()
IMS.SetFormat(oechem.OEFormat_XYZ)


def uniqueconfs(path: str | Path) -> int:
    with open(path, "rb") as f:
        d = pickle.load(f)
    return d["uniqueconfs"]


def oemol_to_rdsmiles(oemol: oechem.OEMol) -> str:
    return canonicalize_smiles(oechem.OEMolToSmiles(oemol))


def get_wrong_smiles(data: dict[str, Any]) -> dict[int, str]:
    wrong_smiles = {}  # Maps conformer ID to re-identified SMILES
    xyz_allconfs = mol_to_xyz(data["rd_mol"])
    IMS.openstring(xyz_allconfs)

    oemols = []
    oemol = oechem.OEGraphMol()
    for oemol in IMS.GetOEGraphMols():
        oemols.append(oechem.OEGraphMol(oemol))

    for i, oemol in enumerate(oemols):
        reidentified_smiles = oemol_to_rdsmiles(oemol)
        if reidentified_smiles != data["smiles"]:
            wrong_smiles[i] = reidentified_smiles

    return wrong_smiles


def get_wrong_smiles_helper(
    index_and_row: tuple[int, pd.Series], processed_dir: Path
) -> dict[int, str]:
    _, row = index_and_row
    with open(processed_dir / row["pickle_path"], "rb") as f:
        ensemble_data = pickle.load(f)

    conf_id_to_reidentified_smiles = get_wrong_smiles(ensemble_data)
    return conf_id_to_reidentified_smiles


def redetermine_bonds(mol: Chem.Mol, charge: int = 0) -> Chem.Mol:
    rwmol = Chem.RWMol(mol)
    for bond in mol.GetBonds():
        rwmol.RemoveBond(bond.GetBeginAtomIdx(), bond.GetEndAtomIdx())
    mol = rwmol.GetMol()
    rdDetermineBonds.DetermineBonds(mol, charge=charge)
    return mol


def process_crest_output(
    index_and_row: tuple[int, pd.Series],
    processed_dir: Path,
    output_is_zipped: bool = False,
    include_time: bool = True,
) -> Path | None:
    i, row = index_and_row
    processed_file = processed_dir / f"{i}.pickle"

    if output_is_zipped:
        ensemble_data = parse_archive(row["archive"], include_time=include_time)
    else:
        ensemble_data = parse_dir(row["archive"], include_time=include_time)

    if ensemble_data is not None:
        assert ensemble_data["smiles"] == row["smiles"]
        with open(processed_file, "wb") as f:
            pickle.dump(ensemble_data, f)
        return processed_file
    else:
        return None


def get_metadata(
    index_and_row: tuple[int, pd.Series],
    processed_dir: Path,
    ignore: set[str] = {"smiles", "rd_mol", "conformers"},
) -> pd.Series:
    i, row = index_and_row

    with open(processed_dir / row["pickle_path"], "rb") as f:
        ensemble_data = pickle.load(f)

    return pd.Series(
        {k: v for k, v in ensemble_data.items() if k not in ignore}, dtype=object, name=i
    )


def write_sdf_and_json(
    index_and_row: tuple[int, pd.Series], pickle_dir: Path, sdf_and_json_dir: Path
) -> None:
    _, row = index_and_row

    pickle_path = pickle_dir / row["pickle_path"]
    with open(pickle_path, "rb") as f:
        ensemble_data = pickle.load(f)
    mol = ensemble_data.pop("rd_mol")

    sdf_path = sdf_and_json_dir / f"{pickle_path.stem}.sdf"
    json_path = sdf_and_json_dir / f"{pickle_path.stem}.json"

    mol_to_sdf(sdf_path, mol)
    with open(json_path, "w") as f:
        json.dump(ensemble_data, f, indent=4)


def main(
    processed_dir: str,  # Location to write processed pickle files to
    summary_path: str,  # Location to write summary CSV file to
    crest_input_dir: str,  # Location of CREST input CSV files
    crest_output_dir: str,  # Location of CREST output folders/archives
    output_is_zipped: bool = False,  # Should be True if CREST output folders are zipped as .tar.gz
    save_walltime: bool = True,  # Extract and save wall time
    include_sdf: bool = True,
    ncpu: int = multiprocessing.cpu_count(),
) -> None:
    """
    `crest_input_dir` should contain CSV files with a 'smiles' column.
    `crest_output_dir` should contain folders or archives with the output of CREST runs,
    where each folder/archive has the name of the corresponding input CSV file (minus
    the extension). Within each folder/archive, there should be numbered folders, each
    containing the output files for a row in the input CSV file.
    """
    # Join the data from the input CSV files with the paths to the corresponding output archives
    crest_input_dir = Path(crest_input_dir)
    crest_output_dir = Path(crest_output_dir)

    print(f"Combining input files in '{crest_input_dir}'")
    dfs = []
    for csv_path in crest_input_dir.glob("*.csv"):  # Modify this pattern if needed
        df = pd.read_csv(csv_path)

        output_dir = crest_output_dir / csv_path.stem
        if not output_dir.exists():
            continue

        if output_is_zipped:
            output_files = {
                int(p.name.split(".")[0]): p for p in output_dir.glob("*.tar.gz")
            }
        else:
            output_files = {int(p.name): p for p in output_dir.glob("*") if p.is_dir()}
        output_files = {i: output_files[i] for i in sorted(output_files)}

        if 0 not in output_files.keys():  # Assume folders are 1-indexed
            output_files = {i - 1: p for i, p in output_files.items()}

        # Perhaps some rows were removed from the end of the input CSV
        if len(output_files) > len(df):
            output_file_idxs = [i for i in output_files.keys() if i < len(df)]
            output_paths = [output_files[i] for i in output_file_idxs]
        else:
            output_file_idxs = list(output_files.keys())
            output_paths = list(output_files.values())

        df = df.iloc[output_file_idxs].copy()
        df["archive"] = output_paths
        dfs.append(df)

    df = pd.concat(dfs, ignore_index=True)
    df["smiles"] = df["smiles"].map(canonicalize_smiles)

    # Extract mol and ensemble metadata from each output and save to numbered pickle files
    processed_par_dir = Path(processed_dir)
    processed_dir = processed_par_dir / "pickle"
    processed_dir.mkdir(parents=True, exist_ok=True)

    print(f"Parsing output and saving to '{processed_dir}'")
    pfunc = partial(
        process_crest_output,
        processed_dir=processed_dir,
        output_is_zipped=output_is_zipped,
        include_time=save_walltime,
    )
    with multiprocessing.Pool(ncpu) as pool:
        processed_files = pool.map(pfunc, df.iterrows())
    df["processed_file"] = processed_files

    # Delete duplicates keeping the molecule with more unique conformers
    print("Deleting duplicates")
    removed_idxs = []
    df = df[~df["processed_file"].isna()]
    for smi, row in (
        df[df.duplicated(subset="smiles", keep=False)]
        .reset_index()
        .groupby(by="smiles")
        .agg(list)
        .iterrows()
    ):
        *remove, keep = sorted(row["processed_file"], key=uniqueconfs)
        for path in remove:
            print(f"Removing '{path}'")
            path.unlink()
            removed_idxs.append(int(path.stem))

    df = df.drop(index=removed_idxs)
    assert df[df.duplicated(subset="smiles", keep=False)].empty

    # Write joined input CSVs to a summary CSV and rename pickle files based on AA sequence
    summary_path = Path(summary_path)
    print(f"Writing summary path to '{summary_path}'")
    df["processed_file_new"] = df["smiles"].map(
        lambda s: processed_dir / f"{s.replace('/', '_')}.pickle"
    )
    assert df[df.duplicated(subset="processed_file_new")].empty
    df["pickle_path"] = df["processed_file_new"].map(lambda p: p.name)

    for old_path, new_path in zip(df["processed_file"], df["processed_file_new"]):
        old_path.rename(new_path)

    df[
        [
            c
            for c in df.columns
            if c not in {"archive", "processed_file", "processed_file_new"}
        ]
    ].to_csv(summary_path, index=False)

    # Extract metadata from pickle files and add it to the summary CSV
    print("Extracting metadata")
    pfunc = partial(get_metadata, processed_dir=processed_dir)
    with multiprocessing.Pool(ncpu) as pool:
        metadata = pool.map(pfunc, df.iterrows())
    meta_df = pd.DataFrame(data=metadata)
    df = pd.concat([df, meta_df], axis=1)

    if oechem.OEChemIsLicensed():
        # Identify CREST runs where reactions occurred (i.e., SMILES changed)
        # Usually a proton transfer if there are charged AAs
        print("Identifying chemical reactions")
        pfunc = partial(get_wrong_smiles_helper, processed_dir=processed_dir)
        with multiprocessing.Pool(ncpu) as pool:
            reidentified_smiles = pool.map(pfunc, df.iterrows())
        new_smiles_dict = {i: d for i, d in zip(df.index, reidentified_smiles) if d}

        # Keep ensembles that underwent reaction if all conformers have the same new SMILES
        print("Processing chemical reactions")
        df_copy = df.copy()
        removed_idxs = []
        for i, smi_dict in new_smiles_dict.items():
            row = df_copy.loc[i]
            path = processed_dir / row["pickle_path"]
            smi0 = smi_dict[next(iter(smi_dict))]
            if len(smi_dict) == row["uniqueconfs"] and all(
                smi == smi0 for smi in smi_dict.values()
            ):
                # Modify SMILES and mol
                print(f"Modifying '{path}'")
                with open(path, "rb") as f:
                    ensemble_data = pickle.load(f)
                new_mol = redetermine_bonds(
                    ensemble_data["rd_mol"], charge=ensemble_data["charge"]
                )
                new_smiles = Chem.MolToSmiles(Chem.RemoveHs(new_mol))
                df_copy.loc[i, "smiles"] = new_smiles
                ensemble_data["rd_mol"] = new_mol
                ensemble_data["smiles"] = new_smiles
                with open(path, "wb") as f:
                    pickle.dump(ensemble_data, f)
            else:
                print(f"Removing '{path}'")
                path.unlink()
                removed_idxs.append(i)

        df = df_copy.drop(index=removed_idxs)
    else:
        print("OpenEye license not found, skipping chemical reaction detection")

    # TODO: Check amino acid stereo - D/L could have flipped during CREST run

    df[
        [
            c
            for c in df.columns
            if c not in {"archive", "processed_file", "processed_file_new"}
        ]
    ].to_csv(summary_path, index=False)

    # Make SDF and JSON files
    if include_sdf:
        processed_sdf_json_dir = processed_par_dir / "sdf_and_json"
        processed_sdf_json_dir.mkdir(exist_ok=True)

        print(f"Writing SDF and JSON files to '{processed_sdf_json_dir}'")
        pfunc = partial(
            write_sdf_and_json,
            pickle_dir=processed_dir,
            sdf_and_json_dir=processed_sdf_json_dir,
        )
        with multiprocessing.Pool(ncpu) as pool:
            pool.map(pfunc, df.iterrows())

    print("Done")


if __name__ == "__main__":
    typer.run(main)
