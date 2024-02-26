# CREMP

Generate conformer ensembles using CREST. Used to generate the [CREMP database](https://zenodo.org/doi/10.5281/zenodo.7931444) described in [CREMP: Conformer-Rotamer Ensembles of Macrocyclic Peptides for Machine Learning](https://arxiv.org/abs/2305.08057).

![cover](assets/cremp.png)

## Installation

```bash
conda env create -f env.yml
conda activate cremp
pip install -e .
```

To enable post-processing utilities, you need to install OpenEye and obtain a license:
```bash
conda install -c openeye openeye-toolkits
```

## Usage

Run the whole pipeline with `run_crest.py`. This embeds conformers with RDKit ETKDGv3, optimizes the low-energy ones with xTB, and subsequently runs CREST. Run `run_crest.py --help` for an overview of the options.

Alternatively, three subcommands are possible to run individual steps in the pipeline. These are

- `run_crest.py rdconf`
- `run_crest.py xtbopt`
- `run_crest.py crest`

To post-process the CREST output, use `scripts/postprocess.py`. See the instructions in the docstring of the `main` function for details about the expected file formats and directory structure.

## Downloading and using CREMP

The published CREMP dataset is available on [Zenodo](https://zenodo.org/doi/10.5281/zenodo.7931444). After downloading and extracting `pickle.tar.gz`, molecules can be loaded as follows:

```python
from cremp.utils.chem_utils import load_mol

mol_path = 'pickle/A.c.Men.S.pickle'  # Replace with the path of your choice

mol_dict = load_mol(mol_path)
mol = mol_dict['rd_mol']  # Get RDKit molecule containing conformers

print(mol.GetNumConformers())
```

## License

Licensed under the MIT License. See [LICENSE](LICENSE) for additional details.

## Citation

If using the CREMP dataset, please cite

```
@misc{grambow2023cremp,
    title={{CREMP}: Conformer-Rotamer Ensembles of Macrocyclic Peptides for Machine Learning}, 
    author={Colin A. Grambow and Hayley Weir and Christian N. Cunningham and Tommaso Biancalani and Kangway V. Chuang},
    year={2023},
    eprint={2305.08057},
    archivePrefix={arXiv},
    primaryClass={q-bio.BM}
}
```
