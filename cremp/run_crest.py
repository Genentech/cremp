#!/usr/bin/env python
import shutil
from pathlib import Path
from typing import Optional

import typer
from rdkit import Chem

from cremp.crest import sample_conformers_crest
from cremp.rdconf import generate_rdkit_conformers
from cremp.utils.chem_utils import load_mol, save_mol
from cremp.xtb import optimize_conformers_xtb

app = typer.Typer(add_completion=False)


def _load_helper(
    smi: str | None = None,
    molpath: str | None = None,
    add_hs: bool = False,  # always adds hs when loading from smiles
) -> Chem.Mol:
    if smi is not None and molpath is not None:
        raise ValueError("Specify 'smi' OR 'molpath', not both")
    elif smi is not None:
        mol = Chem.MolFromSmiles(smi)
        mol = Chem.AddHs(mol)
    elif molpath is not None:
        mol = load_mol(molpath)
        if add_hs:
            mol = Chem.AddHs(mol, addCoords=True)
    else:
        raise ValueError("Must specify 'smi' or 'molpath'")

    return mol


@app.command()
def rdconf(
    smi: Optional[str] = None,
    molpath: Optional[str] = None,
    savepath: str = typer.Option(...),
    nembed: int = 5000,
    nmax: int = 50,
    thresh: float = 0.5,
    skip_cistrans: bool = True,
    nthread: int = 1,
):
    mol = _load_helper(smi=smi, molpath=molpath)
    mol = generate_rdkit_conformers(
        mol,
        max_low_energy_conformers=nmax,
        num_conformers_to_embed=nembed,
        rmsd_threshold=thresh,
        skip_cistrans_opt=skip_cistrans,
        num_threads=nthread,
    )
    save_mol(mol, savepath)


@app.command()
def xtbopt(
    molpath: str = typer.Option(...),
    savepath: str = typer.Option(...),
    minsavepath: Optional[str] = None,
    solvent: Optional[str] = None,
    add_hs: bool = False,
    workdir: str = ".",
    del_scratch: bool = False,
    nthread: int = 1,
    useprocs: bool = True,
):
    mol = _load_helper(molpath=molpath, add_hs=add_hs)

    # Parallelize across conformers opts by default instead of within each opt
    if useprocs:
        nproc = nthread
        nthread = 1
    else:
        nproc = 1

    mol, mol_min, _ = optimize_conformers_xtb(
        mol, solvent=solvent, opt_dir=workdir, num_proc=nproc, num_threads=nthread
    )
    save_mol(mol, savepath)

    if minsavepath is not None:
        save_mol(mol_min, minsavepath)

    if del_scratch:
        shutil.rmtree(workdir)


@app.command()
def crest(
    molpath: str = typer.Option(...),
    savepath: str = typer.Option(...),
    solvent: Optional[str] = None,
    skip_cross: bool = False,
    ewin: Optional[float] = None,  # Energy window
    rthr: Optional[float] = None,  # RMSD threshold
    ethr: Optional[float] = None,  # Energy threshold
    workdir: str = ".",
    scratch: Optional[str] = None,
    keepdir: bool = False,
    nthread: int = 1,
):
    mol = _load_helper(molpath=molpath)

    kwargs = {}
    if ewin is not None:
        kwargs['ewin'] = ewin
    if rthr is not None:
        kwargs['rthr'] = rthr
    if ethr is not None:
        kwargs['ethr'] = ethr

    result = sample_conformers_crest(
        mol,
        solvent=solvent,
        skip_cross=skip_cross,
        work_dir=workdir,
        scratch_dir=scratch,
        keepdir=keepdir,
        num_threads=nthread,
        **kwargs,
    )

    if result is not None:
        mol, _ = result
        save_mol(mol, savepath)


@app.callback(
    invoke_without_command=True,
    context_settings={"allow_extra_args": True, "ignore_unknown_options": True}
)
def main(
    ctx: typer.Context,
    smi: Optional[str] = None,
    molpath: Optional[str] = None,
    save_prefix: str = "mol",
    nconf_rdkit: int = 5000,
    nconf_xtb: int = 50,
    solvent: Optional[str] = None,
    skip_cross: bool = False,
    ewin: Optional[float] = None,  # Energy window
    rthr: Optional[float] = None,  # RMSD threshold
    ethr: Optional[float] = None,  # Energy threshold
    workdir: str = ".",
    scratch: Optional[str] = None,
    del_xtb_scratch: bool = False,
    keepdir: bool = False,
    nthread: int = 1,
) -> None:
    if ctx.invoked_subcommand is None:
        work_dir = Path(workdir)
        work_dir.mkdir(parents=True)

        rdconf_mol_path = work_dir / f"{save_prefix}_rdconf.pickle"
        rdconf(
            smi=smi,
            molpath=molpath,
            savepath=rdconf_mol_path,
            nembed=nconf_rdkit,
            nmax=nconf_xtb,
            nthread=nthread,
        )

        if scratch is not None:
            scratch = Path(scratch)
            opt_dir = scratch / "xtb_opt"
            crest_scratch = scratch / "crest"
        else:
            opt_dir = work_dir / "xtb_opt"
            crest_scratch = scratch

        xtbopt_mol_path = work_dir / f"{save_prefix}_xtbopt.pickle"
        xtbopt_min_mol_path = work_dir / f"{save_prefix}_xtbopt_min.pickle"
        xtbopt(
            molpath=rdconf_mol_path,
            savepath=xtbopt_mol_path,
            minsavepath=xtbopt_min_mol_path,
            solvent=solvent,
            workdir=opt_dir,
            nthread=nthread,
        )

        if scratch is not None:
            if del_xtb_scratch:
                shutil.rmtree(opt_dir)

        crest_dir = work_dir / "crest"
        crest_mol_path = work_dir / f"{save_prefix}_crest.pickle"
        crest(
            molpath=xtbopt_min_mol_path,
            savepath=crest_mol_path,
            solvent=solvent,
            skip_cross=skip_cross,
            ewin=ewin,
            rthr=rthr,
            ethr=ethr,
            workdir=crest_dir,
            scratch=crest_scratch,
            keepdir=keepdir,
            nthread=nthread,
        )


if __name__ == "__main__":
    app()
