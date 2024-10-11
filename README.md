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
```bash
    nanopolish eventalign --reads reads.fastq --bam reads.sorted.bam --genome transcript.fa --scale-events --signal-index --summary /path/to/summary.txt  --threads 50 > /path/to/eventalign.txt
```
This function segments raw fast5 signals to each position within the transcriptome, allowing for predictions of modifications based on the segmented signals. In order to run eventalign, users will need:
* ``reads.fastq``: fastq file generated from basecalling the raw .fast5 files
* ``reads.sorted.bam``: sorted bam file obtained from aligning reads.fastq to the reference transcriptome file
* ``transcript.fa``: reference transcriptome file

See [Nanopolish](https://github.com/jts/nanopolish) for more information.

Example:
```bash
cd ecode/data_process
wget http://sg-nex-data.s3.amazonaws.com/data/sequencing_data_ont/fast5/SGNex_Hct116_directRNA_replicate3_run4/SGNex_Hct116_directRNA_replicate3_run4.tar.gz
mkdir SGNex_Hct116_directRNA_replicate3_run4_fast5
tar -zxvf SGNex_Hct116_directRNA_replicate3_run4.tar.gz -C SGNex_Hct116_directRNA_replicate3_run4_fast5
wget http://sg-nex-data.s3.amazonaws.com/data/sequencing_data_ont/fastq/SGNex_Hct116_directRNA_replicate3_run4/SGNex_Hct116_directRNA_replicate3_run4.fastq.gz
nanopolish index -d /SGNex_Hct116_directRNA_replicate3_run4_fast5 SGNex_Hct116_directRNA_replicate3_run4.fastq.gz

wget http://sg-nex-data.s3.amazonaws.com/data/annotations/transcriptome_fasta/Homo_sapiens.GRCh38.cdna.ncrna.fa
wget http://sg-nex-data.s3.amazonaws.com/data/annotations/transcriptome_fasta/Homo_sapiens.GRCh38.cdna.ncrna.fa.fai
minimap2 -ax map-ont -t 8 Homo_sapiens.GRCh38.cdna.ncrna.fa SGNex_Hct116_directRNA_replicate3_run4.fastq.gz | samtools sort -o SGNex_Hct116_directRNA_replicate3_run4.sorted.bam -T SGNex_Hct116_directRNA_replicate3_run4.tmp
samtools index SGNex_Hct116_directRNA_replicate3_run4.sorted.bam

nanopolish eventalign \
    --threads=10 \
    --signal-index \
    --min-mapping-quality=20 \
    --reads SGNex_Hct116_directRNA_replicate3_run4.fastq.gz \
    --bam SGNex_Hct116_directRNA_replicate3_run4.sorted.bam \
    --genome Homo_sapiens.GRCh38.cdna.ncrna.fa \
    --scale-events > SGNex_Hct116_directRNA_replicate3_run4.eventalign.txt
```

After getting nanopolish eventalign results, we need to preprocess the segmented raw signal file using ``make_index.py``, ``process.py``, ``process_neg_approach1.py`` and ``process_neg_approach2.py``.

The restriction files, which contain the sites to be extracted from the events results, are obtained in ``process_sites.ipynb``.

``make_index.py`` builds index for faster running. ``process.py`` gets positive samples for the dataset. ``process_neg_approach1.py`` gets half of the negative samples with the same 5-mer motifs as positive ones. And ``process_neg_approach2.py`` gets the other half of the negative samples by selecting sites that are m6A modifiable in other cell-lines but not in Hct116.

Example:
```bash
python make_index.py --input SGNex_Hct116_directRNA_replicate3_run4.eventalign
python process.py -i SGNex_Hct116_directRNA_replicate3_run4.eventalign --restrict_file Hct116_ENST
python process_neg_appoarch1.py -i SGNex_Hct116_directRNA_replicate3_run4.eventalign -r Hct116_ENST
python process_neg_appoarch2.py -i SGNex_Hct116_directRNA_replicate3_run4.eventalign -r others_reduced_by_Hct116_ENST
```

The preprocessing results include an ``.index`` index file and a ``.json`` data file, both of which should be placed in the ``edata/Dataset/m6A`` folder for subsequent use.

## Usage

The majority of the project's work is presented in Jupyter Notebook format.

``NSWord.ipynb`` encompass the primary processes of the project, including dataset creation, model structure, training, testing and SHAP interpretability.

``Draw_Graphs.ipynb`` is responsible for generating the various graphs used for analysis.

``m6Anet.ipynb`` contains an implementation of [m6Anet](https://github.com/GoekeLab/m6anet/tree/master), with identical training data and tasks as in ``NSWord.ipynb``.

In addition, we also provide a basic Python and command-line version for model training and testing, located in the ``python_ver`` folder.

Example:
```bash
cd NSWord/python_ver
python create_dataset.py --path ../edata/DataSet/m6A --use_file_name use_files --save_name m6A_NSWord
python train.py --load_dataset_name m6A_NSWord --epochs 150 --learning_rate 0.001 --seq_reduce 16 -- read_reduce 0
python test.py --load_dataset_name m6A_NSWord --load_model_name NSWord_222000_50_50reads_9sites --seq_reduce 16 -- read_reduce 0
```
