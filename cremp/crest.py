import shutil
import subprocess
from pathlib import Path

from rdkit import Chem

from .utils.chem_utils import write_coordinates
from .utils.decorator import timeit


class CRESTSampler:
    def __init__(
        self,
        gfn: int = 2,
        solvent: str | None = None,
        skip_cross: bool = False,
        num_threads: int = 1,
        **kwargs,  # Other CREST keyword args
    ) -> None:
        self.gfn = gfn
        self.solvent = solvent
        self.skip_cross = skip_cross
        self.num_threads = num_threads
        self.kwargs = kwargs

    def __call__(
        self,
        mol: Chem.Mol,
        work_dir: str | Path = ".",
        scratch_dir: Path | None = None,
        keepdir: bool = False,
    ) -> tuple[Chem.Mol, dict[int, float]] | None:
        return self.sample_conformers(
            mol, work_dir=work_dir, scratch_dir=scratch_dir, keepdir=keepdir
        )

    def sample_conformers(
        self,
        mol: Chem.Mol,
        work_dir: str | Path = ".",
        scratch_dir: Path | None = None,
        keepdir: bool = False,
    ) -> tuple[Chem.Mol, dict[int, float]] | None:
        # Generally assumes there's only one conformer in mol
        conformer = mol.GetConformer()

        work_dir = Path(work_dir)
        work_dir.mkdir()
        name = work_dir.name

        xyz_path = work_dir / f"{name}.xyz"
        smi = Chem.MolToSmiles(Chem.RemoveHs(mol))
        write_coordinates(conformer, xyz_path, comment=smi)

        charge = sum(atom.GetFormalCharge() for atom in mol.GetAtoms())
        out_path = work_dir / f"{name}.out"
        err_path = work_dir / f"{name}.err"
        try:
            self.run_crest(
                xyz_path,
                out_path=out_path,
                charge=charge,
                err_path=err_path,
                scratch_dir=scratch_dir,
                keepdir=keepdir,
            )
        except subprocess.CalledProcessError:
            print(f"WARNING: CREST sampling in '{work_dir}' failed for '{smi}'")
            return None

        output_conf_path = work_dir / "crest_conformers.xyz"
        if not output_conf_path.exists():
            print(f"WARNING: Conformer path could not be found at '{output_conf_path}'")
            return None

        parsed_atoms, conformers, energies = self.parse_results(output_conf_path)
        # According to GEOM paper, CREST sometimes changes atom order,
        # so don't check atoms and just return new mol?
        self._check_atoms(mol, parsed_atoms)
        # Also check for bond changes or leave that for post-processing?

        mol = self.copy_mol_with_new_conformers(mol, conformers)
        energies = dict(enumerate(energies))

        return mol, energies

    @timeit(unit="h", with_args=True, skip_first_arg=True)
    def run_crest(
        self,
        xyz_path: Path,
        charge: int = 0,
        out_path: Path | None = None,
        err_path: Path | None = None,
        scratch_dir: Path | None = None,
        keepdir: bool = False,
    ) -> None:
        xyz_path = xyz_path.resolve()
        run_dir = xyz_path.parent
        xyz_file = xyz_path.name

        crest_executable = shutil.which("crest")
        command = [
            crest_executable,
            xyz_file,
            "-T",
            str(self.num_threads),
            f"--gfn{self.gfn}",
            "--chrg",
            str(charge),
        ]
        if self.solvent is not None:
            solvent_model = "gbsa" if self.solvent.lower() == "methanol" else "alpb"
            command.extend([f"--{solvent_model}", self.solvent])
        if self.skip_cross:
            command.append("--nocross")
        if scratch_dir is not None:
            command.extend(["--scratch", str(scratch_dir)])
        if keepdir:
            command.append("--keepdir")
        for arg, val in self.kwargs.items():
            command.extend([f"--{arg}", str(val)])
        subprocess_kwargs = dict(cwd=run_dir, check=True)

        if out_path is not None and err_path is not None:
            with open(out_path, "w") as fout, open(err_path, "w") as ferr:
                subprocess.run(command, stdout=fout, stderr=ferr, **subprocess_kwargs)
        elif out_path is not None:
            with open(out_path, "w") as fout:
                subprocess.run(command, stdout=fout, **subprocess_kwargs)
        elif err_path is not None:
            with open(err_path, "w") as ferr:
                subprocess.run(command, stderr=ferr, **subprocess_kwargs)
        else:
            subprocess.run(command, **subprocess_kwargs)

    @staticmethod
    def parse_results(
        path: Path,
    ) -> tuple[list[str], list[Chem.Conformer], list[float]]:
        atoms = []
        conformers = []
        energies = []

        with open(path) as f:
            while True:
                try:
                    num_atoms = int(next(f))
                except StopIteration:
                    break

                atoms = []
                conformer = Chem.Conformer(num_atoms)
                energy = float(next(f))

                for i in range(num_atoms):
                    atom, *position = next(f).split()
                    atoms.append(atom)
                    conformer.SetAtomPosition(i, [float(p) for p in position])

                conformers.append(conformer)
                energies.append(energy)

        return atoms, conformers, energies

    @staticmethod
    def _check_atoms(mol: Chem.Mol, atom_symbols: list[str]) -> None:
        for i, (atom_symbol, true_atom) in enumerate(zip(atom_symbols, mol.GetAtoms())):
            true_atom_symbol = true_atom.GetSymbol()
            if atom_symbol != true_atom_symbol:
                raise ValueError(
                    f"Atom '{atom_symbol}' at index {i} does not match expected atom '{true_atom_symbol}'"
                )

    @staticmethod
    def copy_mol_with_new_conformers(
        mol: Chem.Mol, conformers: list[Chem.Conformer]
    ) -> Chem.Mol:
        mol = Chem.Mol(mol, quickCopy=True)
        for conformer in conformers:
            mol.AddConformer(conformer, assignId=True)
        return mol


def sample_conformers_crest(
    mol,
    gfn: int = 2,
    solvent: str | None = None,
    skip_cross: bool = False,
    work_dir: str | Path = ".",
    scratch_dir: Path | None = None,
    keepdir: bool = False,
    num_threads: int = 1,
    **kwargs,
) -> tuple[Chem.Mol, dict[int, float]] | None:
    sampler = CRESTSampler(
        gfn=gfn,
        solvent=solvent,
        skip_cross=skip_cross,
        num_threads=num_threads,
        **kwargs,
    )
    return sampler(mol, work_dir=work_dir, scratch_dir=scratch_dir, keepdir=keepdir)
