# RNA Modification Detection using Nanopore Direct RNA Sequencing via improved Transformer
![Image text](https://github.com/faded53222/NSWord/blob/main/figures/whole_structure.png)

## Abstract
RNA modifications are a common occurrence in transcriptome and play a crucial role in various biological processes. Nanopore direct RNA sequencing (DRS) provides raw current signal readings, which carry information of modifications. Supervised machine learning methods using DRS are advantageous for RNA modification detection. However, existing methods for RNA modification detection do not adequately capture sequential signal features within and between reads. Here, we represent NSWord, an improved transformer model with novel self-attention blocks that integrates the transcript sequence and its signal reads to produce a comprehensive site-level prediction. NSWord outperforms existing deep learning methods, particularly in its ability to utilize a greater number of reads to produce more accurate predictions. Additionally, we conducted a series of studies using the SHAP method, investigating the factors influencing the RNA modifications from a perspective of interpretability.

## Installation
1. **Clone the repository**:

    ```bash
    git clone https://github.com/faded53222/NSWord.git
    cd NSWord
    ```

2. **Create a virtual environment** (optional but recommended):

    ```bash
    python -m venv virtual
    source virtual/bin/activate  # On Windows use `virtual\Scripts\activate`
    ```

3. **Install dependencies**:

    ```bash
    pip install -r requirements.txt
    ```

## Dataprep
NSWord dataprep requires eventalign.txt from ``nanopolish eventalign``:
```
    nanopolish eventalign --reads reads.fastq --bam reads.sorted.bam --genome transcript.fa --scale-events --signal-index --summary /path/to/summary.txt  --threads 50 > /path/to/eventalign.txt
```
This function segments raw fast5 signals to each position within the transcriptome, allowing for predictions of modifications based on the segmented signals. In order to run eventalign, users will need:
* ``reads.fastq``: fastq file generated from basecalling the raw .fast5 files
* ``reads.sorted.bam``: sorted bam file obtained from aligning reads.fastq to the reference transcriptome file
* ``transcript.fa``: reference transcriptome file

See [Nanopolish](https://github.com/jts/nanopolish) for more information.

After getting nanopolish eventalign results, we need to preprocess the segmented raw signal file using ``make_index.py``, ``process.py`` and ``process_neg.py``.

``make_index.py`` builds index for faster running. ``process.py`` gets positive samples for the dataset. And ``process_neg.py`` gets negative samples with the same 5-mer motifs as positive ones.

Main functions are within those files, take care with '####' annotations before running them.

RNA modification sites such as those in ``m6Asites.txt`` need to be converted to ENST coordinates with ``ENSG_to_ENST.ipynb`` to be used for searching positive samples in ``process.py``.

## Usage

We offer two ways to run the project:
1. **Jupyter Notebook Version**: This version allows users to run and interact with code blocks step by step.
2. **Command-line Python Version**: A step-by-step Python script that can be executed directly from the command line.

1. Using the Jupyter Notebook Version


Run ``NSWord.ipynb`` block by block to train and test some conventional NSWord models. 

Run specific blocks in ``Draw_Graphs.ipynb`` to get the gragh you need.

``NSWord_extra.ipynb`` is for investigating "the impact of limiting the length or number of signal reads" and exploring "the role of transcript sequence in modification prediction".

You can also get the result of [m6Anet](https://github.com/GoekeLab/m6anet/tree/master) model for predicting the same dataset by running ``m6Anet.ipynb``

# Citing
If you use NSWord in your research, please cite ####
