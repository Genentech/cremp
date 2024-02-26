import multiprocessing
import shutil
import subprocess
from pathlib import Path

from rdkit import Chem

from .utils.chem_utils import write_coordinates
from .utils.decorator import timeit


class XTBConformerOptimizer:
    def __init__(
        self,
        gfn: int = 2,
        solvent: str | None = None,
        num_threads: int | None = None,  # None uses all available cores
    ) -> None:
        self.gfn = gfn
        self.solvent = solvent
        self.num_threads = num_threads

    def __call__(
        self, conformer: Chem.Conformer, opt_dir: str | Path = "."
    ) -> tuple[Chem.Conformer, float] | None:
        return self.optimize_conformer(conformer, opt_dir=opt_dir)

    def optimize_conformer(
        self, conformer: Chem.Conformer, opt_dir: str | Path = "."
    ) -> tuple[Chem.Conformer, float] | None:
        if not conformer.HasOwningMol():
            raise ValueError("Conformer must belong to a molecule")

        opt_dir = Path(opt_dir)
        opt_dir.mkdir()
        name = opt_dir.name
        mol = conformer.GetOwningMol()

        xyz_path = opt_dir / f"{name}.xyz"
        smi = Chem.MolToSmiles(Chem.RemoveHs(mol))
        write_coordinates(conformer, xyz_path, comment=smi)

        charge = sum(atom.GetFormalCharge() for atom in mol.GetAtoms())
        out_path = opt_dir / f"{name}.out"
        err_path = opt_dir / f"{name}.err"
        try:
            self.run_xtb(xyz_path, charge=charge, out_path=out_path, err_path=err_path)
        except subprocess.CalledProcessError:
            print(
                f"WARNING: xTB optimization in '{opt_dir}' of conformer {conformer.GetId()} of '{smi}' failed"
            )
            return None

        output_coord_path = opt_dir / "xtbopt.xyz"
        if not output_coord_path.exists():
            print(
                f"WARNING: Optimized coordinate file could not be found at '{output_coord_path}'"
            )
            return None

        parsed_atoms, new_conformer, energy = self.parse_results(output_coord_path)
        self._check_atoms(conformer, parsed_atoms)

        return new_conformer, energy

    @timeit(unit="min", with_args=True, skip_first_arg=True)
    def run_xtb(
        self,
        xyz_path: Path,
        charge: int = 0,
        out_path: Path | None = None,
        err_path: Path | None = None,
    ) -> None:
        xyz_path = xyz_path.resolve()
        run_dir = xyz_path.parent
        xyz_file = xyz_path.name

        xtb_executable = shutil.which("xtb")
        command = [
            xtb_executable,
            xyz_file,
            "--opt",
            "--gfn",
            str(self.gfn),
            "--chrg",
            str(charge),
        ]
        if self.solvent is not None:
            solvent_model = "gbsa" if self.solvent.lower() == "methanol" else "alpb"
            command.extend([f"--{solvent_model}", self.solvent])

        subprocess_kwargs = dict(
            cwd=run_dir,
            check=True,
            env={"OMP_NUM_THREADS": f"{self.num_threads},1"}
            if self.num_threads is not None
            else None,
        )

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
    def parse_results(path: Path) -> tuple[list[str], Chem.Conformer, float]:
        with open(path) as f:
            atoms = []
            num_atoms = int(next(f))
            conformer = Chem.Conformer(num_atoms)

            # Get energy from comment line
            energy = float(next(f).split()[1])

            for i, line in enumerate(f):
                atom, *position = line.split()
                atoms.append(atom)
                conformer.SetAtomPosition(i, [float(p) for p in position])

        return atoms, conformer, energy

    @staticmethod
    def _check_atoms(conformer: Chem.Conformer, atom_symbols: list[str]) -> None:
        mol = conformer.GetOwningMol()
        for i, (atom_symbol, true_atom) in enumerate(zip(atom_symbols, mol.GetAtoms())):
            true_atom_symbol = true_atom.GetSymbol()
            if atom_symbol != true_atom_symbol:
                raise ValueError(
                    f"Atom '{atom_symbol}' at index {i} does not match expected atom '{true_atom_symbol}'"
                )


# Need to define at top level for multiprocessing
def _optimize_conformer_helper(params) -> tuple[Chem.Conformer, float] | None:
    conf_id, mol, opt_dir, name, optimizer = params
    conformer = mol.GetConformer(conf_id)
    conf_dir = opt_dir / f"{name}_{conf_id}"

    opt_res = optimizer(conformer, opt_dir=conf_dir)
    if opt_res is None:
        print(
            f"WARNING: Conformer {conf_id} of '{Chem.MolToSmiles(Chem.RemoveHs(mol))}' was not updated"
        )
        return opt_res
    else:
        # Can't pickle Chem.Conformer, so make mol instead
        new_conformer, energy = opt_res
        mol_copy = Chem.Mol(mol, quickCopy=True)
        mol_copy.AddConformer(new_conformer)
        return mol_copy, energy


class XTBMolOptimization:
    def __init__(self, mol: Chem.Mol) -> None:
        self.mol = Chem.Mol(mol)  # Make copy
        self.energies: dict[int, float] = {}

    def __call__(
        self,
        optimizer: XTBConformerOptimizer,
        opt_dir: str | Path = ".",
        num_proc: int = 1,
    ) -> tuple[Chem.Mol, dict[int, float]]:
        return self.optimize_mol(optimizer, opt_dir=opt_dir, num_proc=num_proc)

    @timeit(unit="min")
    def optimize_mol(
        self,
        optimizer: XTBConformerOptimizer,
        opt_dir: str | Path = ".",
        num_proc: int = 1,
    ) -> tuple[Chem.Mol, dict[int, float]]:
        # Repeated calls to optimize will continue to operate on the same mol
        opt_dir = Path(opt_dir)
        opt_dir.mkdir(parents=True)
        name = opt_dir.name

        # We can't loop over the conformer sequence b/c we update it while iterating,
        # so we get the initial list of IDs instead
        conf_ids = [conformer.GetId() for conformer in self.mol.GetConformers()]
        params_list = [
            (conf_id, self.mol, opt_dir, name, optimizer) for conf_id in conf_ids
        ]

        with multiprocessing.Pool(num_proc) as pool:
            opt_results = pool.map(_optimize_conformer_helper, params_list)

        for conf_id, opt_result in zip(conf_ids, opt_results):
            if opt_result is not None:
                new_conformer_mol, energy = opt_result
                new_conformer = new_conformer_mol.GetConformer()
                self.update_conformer(conf_id, new_conformer)
                self.energies[conf_id] = energy

        return self.mol, self.energies

    def get_mol_with_lowest_energy_conf(self) -> Chem.Mol:
        min_energy_conf_id = min(self.energies, key=self.energies.get)
        min_energy_conformer = self.mol.GetConformer(min_energy_conf_id)

        new_mol = Chem.Mol(self.mol, quickCopy=True)  # Copy without conformers
        new_mol.AddConformer(min_energy_conformer, assignId=True)

        return new_mol

    def update_conformer(
        self, conf_id: int, new_conformer: Chem.Conformer
    ) -> Chem.Conformer:
        new_conformer.SetId(conf_id)
        self.mol.RemoveConformer(conf_id)
        self.mol.AddConformer(new_conformer, assignId=False)

        conformer = self.mol.GetConformer(conf_id)
        return conformer


def optimize_conformers_xtb(
    mol: Chem.Mol,
    gfn: int = 2,
    solvent: str | None = None,
    opt_dir: str | Path = ".",
    num_proc: int = 1,  # Parallelize across conformer opts
    num_threads: int = 1,  # Parallelize within each opt
) -> tuple[Chem.Mol, Chem.Mol, dict[int, float]]:
    optimizer = XTBConformerOptimizer(gfn=gfn, solvent=solvent, num_threads=num_threads)
    mol_optimization = XTBMolOptimization(mol)
    mol, energies = mol_optimization(optimizer, opt_dir=opt_dir, num_proc=num_proc)
    mol_min = mol_optimization.get_mol_with_lowest_energy_conf()
    return mol, mol_min, energies
