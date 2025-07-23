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
wget http://sg-nex-data.s3.amazonaws.com/data/annotations/transcriptome_fasta/Homo_sapiens.GRCh38.cdna.ncrna.fa
wget http://sg-nex-data.s3.amazonaws.com/data/annotations/transcriptome_fasta/Homo_sapiens.GRCh38.cdna.ncrna.fa.fai

wget http://sg-nex-data.s3.amazonaws.com/data/sequencing_data_ont/fast5/SGNex_Hct116_directRNA_replicate3_run4/SGNex_Hct116_directRNA_replicate3_run4.tar.gz
mkdir SGNex_Hct116_directRNA_replicate3_run4_fast5
tar -zxvf SGNex_Hct116_directRNA_replicate3_run4.tar.gz -C SGNex_Hct116_directRNA_replicate3_run4_fast5
wget http://sg-nex-data.s3.amazonaws.com/data/sequencing_data_ont/fastq/SGNex_Hct116_directRNA_replicate3_run4/SGNex_Hct116_directRNA_replicate3_run4.fastq.gz
nanopolish index -d /SGNex_Hct116_directRNA_replicate3_run4_fast5 SGNex_Hct116_directRNA_replicate3_run4.fastq.gz

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
All data processing steps and python files from this point onward are also contained in the ``ecode/data_process`` directory.

---

The restriction files, which contain the sites to be extracted from the events results, are already included in this repository and can be obtained in ``process_sites.ipynb``.

Example of getting the m6A sites restriction file of the Hct116 cell line:
```bash
wget http://sg-nex-data.s3.amazonaws.com/data/annotations/m6ACE_seq_reference_table/Hct116_m6ACEsites.txt -O Hct116.txt
pyensembl install --release 110 --species homo_sapiens
python ENSG_to_ENST.py -i Hct116
```

---

After getting nanopolish eventalign results, we need to preprocess the segmented raw signal file using ``make_index.py``, ``process.py``, ``process_neg_approach1.py`` and ``process_neg_approach2.py``.
``make_index.py`` builds index for faster running. ``process.py`` gets positive samples for the dataset. ``process_neg_approach1.py`` gets half of the negative samples with the same 5-mer motifs as positive ones. And ``process_neg_approach2.py`` gets the other half of the negative samples by selecting sites that are m6A modifiable in other cell-lines but not in Hct116. Run the commands with '-help' for parameter details.

Example:
```bash
python make_index.py --input SGNex_Hct116_directRNA_replicate3_run4.eventalign
python process.py -i SGNex_Hct116_directRNA_replicate3_run4.eventalign --restrict_file Hct116_ENST
python process_neg_approach1.py -i SGNex_Hct116_directRNA_replicate3_run4.eventalign -r Hct116_ENST
python process_neg_approach2.py -i SGNex_Hct116_directRNA_replicate3_run4.eventalign -r others_reduced_by_Hct116_ENST
```
The processing results include an ``.index`` index file and a ``.json`` data file.

Files ``edata/Dataset/m6A/final_data_example.index`` and ``edata/Dataset/m6A/final_data_example.json`` demonstrate the expected post-processing results.

---

- Execute the commands provided in the example code blocks above would generate the final processed files for ``SGNex_Hct116_directRNA_replicate3_run4``.

Next: 

- Re-run the same code blocks, replacing ``SGNex_Hct116_directRNA_replicate3_run4`` with ``SGNex_Hct116_directRNA_replicate3_run1``, ``SGNex_Hct116_directRNA_replicate4_run3`` (``SGNex_HepG2_directRNA_replicate1_run3`` for cross cell line predictive performance validation).
    
- Store all outputs in the ``edata/Dataset/m6A`` directory.

These files constitute the dataset used by the models in demonstration codes.

The exact processed data used for model evaluation in the notebooks can be downloaded [here](https://drive.google.com/drive/folders/19L5-yIUrHiIotUJoltECkWRWmc21THFm?usp=sharing)

---

As shown above, the Nanopore sequencing data and m6A modification sites analyzed in the Hct116 cell line were sourced from [SG-NEx](https://github.com/GoekeLab/sg-nex-data).

## Usage

The majority of the project's work is presented in Jupyter Notebook format.

``NSWord.ipynb`` encompasses the primary processes of the project, including dataset creation, model structure, training, testing and SHAP interpretability.

``Draw_Graphs.ipynb`` is responsible for generating the various graphs used for analysis.

``m6Anet.ipynb`` contains an implementation of [m6Anet](https://github.com/GoekeLab/m6anet/tree/master), with identical training data and tasks as in ``NSWord.ipynb``.

In addition, we provide a basic Python and command-line version for model training and testing, located in the ``python_ver`` folder. Run the commands with '-help' for parameter details.

Example:
```bash
cd NSWord/python_ver
python create_dataset.py --path m6A --use_file_name use_files --save_name m6A_NSWord
python train.py --load_dataset_name m6A_NSWord --epochs 150 --learning_rate 0.001 --seq_reduce 16 -- read_reduce 0
python test.py --load_dataset_name m6A_NSWord --load_model_name NSWord_222000_50_50reads_9sites --seq_reduce 16 -- read_reduce 0
```

## Test or Sanity Check with RNA004 data

This section shows the evaluation of the model using RNA004 data. Given the small size of the [test data](https://epi2me.nanoporetech.com/rna-mod-validation-data/), it can also serve as a sanity check.

The structure of the model need not to be modified for using RNA004 data, because models like NSWord are fundamentally event-based rather than certain-version-signal-data-dependent: as long as future data are still suitable be converted into eventaligned events, the model's performance would remain largely consistent.

Learn more about [pod5 data format](https://github.com/nanoporetech/pod5-file-format) and [f5c](https://github.com/hasindu2008/f5c/releases/tag/v1.3) used to convert raw signals to events.

Detailed data fetching:
```bash
cd NSWord/RNA004

wget https://42basepairs.com/download/s3/ont-open-data/rna-modbase-validation_2025.03/references/sampled_context_strands.fa
wget https://42basepairs.com/download/s3/ont-open-data/rna-modbase-validation_2025.03/basecalls/m6A_rep1.bam
wget https://42basepairs.com/download/s3/ont-open-data/rna-modbase-validation_2025.03/basecalls/control_rep1.bam
wget https://42basepairs.com/download/s3/ont-open-data/rna-modbase-validation_2025.03/subset/m6A_rep1.pod5
wget https://42basepairs.com/download/s3/ont-open-data/rna-modbase-validation_2025.03/subset/control_rep1.pod5

pod5 convert to_fast5 control_rep1.pod5 --output control_rep1_fast5/
pod5 convert to_fast5 m6A_rep1.pod5 --output m6A_rep1_fast5/
samtools fastq -0 control_rep1.fastq control_rep1.bam
samtools fastq -0 m6A_rep1.fastq m6A_rep1.bam

minimap2 -ax map-ont -t 8 sampled_context_strands.fa m6A_rep1.fastq | samtools sort -o m6A_rep1-ref.sorted.bam -T m6A_rep1.tmp
samtools index m6A_rep1-ref.sorted.bam
minimap2 -ax map-ont -t 8 sampled_context_strands.fa control_rep1.fastq | samtools sort -o control_rep1-ref.sorted.bam -T control_rep1.tmp
samtools index control_rep1-ref.sorted.bam

VERSION=v1.5
wget "https://github.com/hasindu2008/f5c/releases/download/$VERSION/f5c-$VERSION-binaries.tar.gz" && tar xvf f5c-$VERSION-binaries.tar.gz && cd f5c-$VERSION/
cd f5c-v1.5/scripts
export HDF5_PLUGIN_PATH=$HOME/.local/hdf5/lib/plugin

wget https://raw.githubusercontent.com/hasindu2008/f5c/v1.3/test/rna004-models/rna004.nucleotide.5mer.model

/f5c-v1.5/f5c_x86_64_linux index -d m6A_rep1_fast5/ m6A_rep1.fastq
/f5c-v1.5/f5c_x86_64_linux eventalign --rna -b m6A_rep1-ref.sorted.bam -r m6A_rep1.fastq -g sampled_context_strands.fa -o m6A_rep1.eventalign.txt --kmer-model rna004.nucleotide.5mer.model 
/f5c-v1.5/f5c_x86_64_linux index -d control_rep1_fast5/ control_rep1.fastq
/f5c-v1.5/f5c_x86_64_linux eventalign --rna -b control_rep1-ref.sorted.bam -r control_rep1.fastq -g sampled_context_strands.fa -o control_rep1.eventalign.txt --kmer-model rna004.nucleotide.5mer.model 

```
