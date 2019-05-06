# Active regulatory regions prediction dataset renderer
Simple python tool to render dataset that can be used for training models for active regulatory regions prediction.

## How to get it?
Just clone the repo.

## Which genome does it use by default?
By default it uses [hg19](https://www.ncbi.nlm.nih.gov/assembly/GCF_000001405.13/), as it is the genome used in the labeled data currently available from the Wasserman team.

## Dependencies
This package will use the package [bedtools](https://bedtools.readthedocs.io/en/latest/) to elaborate the bed files. A setup for the package is available [here](https://github.com/LucaCappelletti94/wasserman/blob/master/info/bedtools.md).

## Rendering the dataset
Just run the following:
```bash
python run.py
```
