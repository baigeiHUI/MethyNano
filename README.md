# MethyNano

## MethyNano: Supervised Contrastive Pretraining Enables Robust and Generalizable Methylation Detection from Nanopore Sequencing

## Project Overview

5-Methylcytosine (5mC) plays an important role in gene regulation and development. Although nanopore sequencing enables direct detection of 5mC, existing methods still face limitations, including weak generalization across species and sequence contexts (CpG/CHG/CHH), as well as suboptimal integration of sequence and current signals.

MethyNano is a deep learning framework that uses supervised contrastive pretraining for 5mC detection from nanopore reads. By learning more discriminative and stable representations, the contrastive objective improves sensitivity in rare sequence contexts and reduces prediction uncertainty in challenging regions. Across datasets from A. thaliana, O. sativa, and H. sapiens, MethyNano achieves strong performance compared with existing methods. Cross-species and cross-motif experiments further demonstrate its generalization ability, and ablation studies show that the model architecture effectively integrates critical sequence and signal features.

## Installation

### 0. Create a New Environment

```
conda create -n MethyNano python=3.10
conda activate MethyNano
```

### 1. Clone the Project

```
git clone https://github.com/baigeiHUI/MethyNano.git
cd MethyNano
```

### 2. Install Requirements

```
pip install -r requirements.txt
```

MethyNano was tested with PyTorch `2.6.0+cu118` (CUDA `11.8`). Please check your local CUDA version and install a compatible PyTorch build from the official PyTorch website: https://pytorch.org

For Linux GPU environments, install the CUDA-enabled PyTorch wheel before or after installing `requirements.txt`, for example:

```
pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/cu118
```

## Data Preprocessing

### Process Bisulfite Sequencing Results

Install the required command-line tools with Anaconda:

```
conda install -y -c bioconda bismark bowtie2 samtools cutadapt
```

Modify `FASTQ`, `REF_FA`, and `WORK` in `scripts/pipeline_bismark.sh` to specify the bisulfite sequencing file, reference genome file, and output directory. Set `SAMPLE` to your sample prefix. Then run the script to generate a BED file containing site-level methylation labels:

```
./scripts/pipeline_bismark.sh
```

### Basecalling

We use Dorado (`v0.7.2`) with the `dna_r10.4.1_e8.2_400bps_hac@v4.2.0` model for basecalling. The `--emit-moves` option is required because it stores the move table in each BAM record.

```
dorado basecaller dna_r10.4.1_e8.2_400bps_hac@v4.2.0 <pod5_files> \
--device <device> \
--reference <reference.fasta> \
--emit-moves > calls.bam
```

Sort the BAM file by genomic coordinates and create a `.bai` index:

```
samtools sort -o calls.sorted.bam calls.bam
samtools index calls.sorted.bam
```

### Extract Features

Extract read-level signals from POD5 files and generate 13-mer features:

```
python scripts/extract_pod5_signal.py \
--pod5 <pod5_directory> \
--bam calls.sorted.bam \
-r reference.fasta \
-o output_signal.tsv \
-p 16

python scripts/get_13mer_features.py \
--signal_file output_signal.tsv \
--output 13merBasicFeature.tsv \
--clip 6 \
--motif NNNNNNCNNNNNN
```

Split the BED file into fully methylated and fully unmethylated subsets. This step outputs `methylation_1.bed` and `methylation_0.bed`.

```
python "scripts/split_BED_ into_methylated_and_unmethylated .py" \
--bed <bed_file> \
--output_prefix methylation \
--chunksize 200000
```

Align BED methylation labels with 13-mer features by chromosome and genomic position:

```
python scripts/alignment.py \
--bed <bed_file> \
--tsv path/to/13merBasicFeatures \
--output_file path/to/output.csv \
--chunksize 200000
```

### Build the Dataset

Construct train, validation, and test datasets from the aligned CSV files:

```
python scripts/csv2dataset.py
```

## Train Your Own Model

MethyNano uses a two-stage training strategy:

* Contrastive pretraining: A dual-branch architecture with shared weights learns feature representations from input samples. The contrastive objective encourages discriminative methylation signal embeddings.
* Classification fine-tuning: After pretraining, the contrastive projection head is discarded and replaced with a classification head. The model is then fine-tuned to predict methylation status.

### Contrastive Pretraining

```
python train_contrastive.py \
--train_csv <train.csv> \
--val_csv <val.csv> \
--batch_size 512 \
--epochs 50 \
--lr 1e-3 \
--ckpt_dir <save_path>
```

### Classification Fine-Tuning

```
python finetune_cls.py \
--train_csv <train.csv> \
--val_csv <val.csv> \
--batch_size 512 \
--epochs 30 \
--lr 8e-4 \
--ckpt_dir <save_path> \
--logdir <log_save_path> \
--resume <pretrained_ckpt>
```

The `--resume` argument loads weights from contrastive pretraining. If `--resume` is omitted, the classification model is trained from scratch.

### Evaluation

Open `testDemo.ipynb` in Jupyter, set `CKPT_PATH` and `TEST_CSV` to your checkpoint and test CSV paths, and then run the notebook.

## Methylation Calling with a Trained Model

Use `predict.py` to call methylation directly from nanopore POD5 reads, a coordinate-sorted BAM file, a reference FASTA file, and a trained MethyNano checkpoint.

Before calling, basecall with Dorado using `--emit-moves`, then sort and index the BAM file as described above. After that, run methylation calling:

```
python predict.py \
--pod5 <pod5_file_or_directory> \
--bam calls.sorted.bam \
--reference reference.fasta \
--ckpt <ckpt_path> \
--output /path/to/methynano_calls.csv \
--recursive \
--workers 4 \
--feature-batch-size 512 \
--gpu-batch-size 2048 \
--threshold 0.5 \
--device auto \
--fp16
```

The output CSV contains one row per called candidate site, including `read_id`, genomic `start_pos` and `end_pos`, the 13-mer sequence, the positive methylation probability `prob_pos`, and the binary prediction `label_pred`.

Common options:

* `--pod5` accepts one or more POD5 files or directories.
* `--recursive` searches for POD5 files recursively under input directories.
* `--motif` defaults to `......C......`, which calls cytosines in a 13-mer window.
* `--require-query-ref-match` keeps only sites where the query 13-mer exactly matches the reference 13-mer.
* `--fp16` enables half-precision inference on CUDA GPUs.
