# MethyNano

---
## MethyNano: supervised contrastive pretraining enables robust and generalizable methylation detection from nanopore sequencing

---
### Project Overview
5-Methylcytosine (5mC) plays an important role in gene regulation and development. 
Although nanopore sequencing has enabled direct detection of 5mC, existing methods 
still face several limitations, including poor generalization across species and sequence 
contexts (CpG/CHG/CHH), as well as suboptimal integration of sequence and current 
signals. Here, we present MethyNano, a deep learning framework incorporating a 
contrastive learning strategy to detect 5mC from nanopore reads. By encouraging more 
discriminative and stable representations, the contrastive objective improves the 
model’s sensitivity to rare sequence contexts and reduces its prediction uncertainty in 
challenging regions. Across datasets from A. thaliana, O. sativa, and H. sapiens, 
MethyNano achieves superior performance on key metrics compared with other 
existing methods. Extensive cross-species and cross-motif experiments demonstrate the 
robust generalization performance of MethyNano, while dimensionality-reduction 
visualizations of learned features provide an intuitive view of the model’s efficient 
representation capability. Moreover, our ablation studies show that MethyNano’s 
architecture enables more effective integration of critical features, leading to higher 
predictive accuracy.

---
### Installation

#### 0.Create new environment (e.g. Conda)
```
conda create -n MethyNano python=3.10
```

#### 1.Cloning the Project
First, you need to clone the project repository from GitHub to your local machine. You can do this by running the following command in your terminal:
   
```
git clone https://github.com/baigeiHUI/MethyNano.git
cd MethyNano
```
#### 2.Activate Environment
```
conda activate MethyNano
```
#### 3.Installing Requirements
To install the required packages, run the following command in your terminal:
```
pip install -r requirements.txt
```
Note: We use PyTorch version `2.6.0+cu118` (compiled with CUDA `11.8`).
Please check your local CUDA version and install a compatible PyTorch version from the official PyTorch website: https://pytorch.org

---
### Data preprocessing
#### Basecalling
We use Dorado (`v0.7.2`)  with the `dna_r10.4.1_e8.2_400bps_hac@v4.2.0` model for basecalling.
 ```
 dorado basecaller dna_r10.4.1_e8.2_400bps_hac@v4.2.0 <pod5_files> \
 --device <device> \
 --reference <reference> \
 --emit-moves > /path/to/output/calls.bam
 ```
Note: `--emit-moves` will output move table field for each entry in bam file.
#### Extract features
* Use samtools to sort alignments by genomic coordinates and create a` .bai` index file.
```
samtools sort -o calls.sorted.bam calls.bam
samtools index calls.sorted.bam
```
+ Extract signal files and features using the following python scripts.
```
python scripts/extract_pod5_signal.py --pod5 <pod5_files> --bam calls.sorted.bam -r reference.fasta -o output_signal.tsv -p 16
python scripts/get_13mer_features.py --signal_file output_signal.tsv --output 13merBasicFeature.tsv --clip 4 --motif NNNNNNCNNNNNN
```
+ Split the` BED` file into` fully positive (methylated)` and `fully negative (unmethylated)` subsets,
which output two files: `methylation_1.bed` and `methylation_0.bed`.
```
python scripts/split_BED_ into_methylated_and_unmethylated.py \
--bed <bed_file> \
--output_prefix prefix \
--chunksize 200000
```

* Align the  BED file you required with 13-mer basic features based on chromosome and genomic position information
```
python scripts/alignment.py --bed <bed_file> \
--tsv  path/to/13merBasicFeatures \
--output_file  path/to/output.csv \
--chunksize 200000
```
#### Build  dataset
Construct dataset from the aligned output CSV files.
```
python scripts/csv2dataset.py 
```
---
###  Train your own model
Our model employs a two-stage training strategy:
* Contrastive Pretraining: A dual-branch architecture with shared weights is used to learn feature representations from input samples, optimized via contrastive loss to capture discriminative methylation signal embeddings.
+ Classification Fine-tuning: After pretraining, the contrastive projection head is discarded and replaced with a classification head, which is then fine-tuned on the downstream task to produce final methylation status predictions.

#### Contrastive Pretraining
```
python train_contrastive.py --train_csv <train.csv> \
--val_csv <val.csv> \
--batch_size 512 \
--epochs 100 \
--lr 1e-3 \
--ckpt_dir  <save_path>
```
#### Classification Fine-tuning
```
python finetune_cls.py --train_csv <train.csv> \
--val_csv <val.csv> \ 
--batch_size 512 \
--epochs 30 \
--lr 8e-4 \
--ckpt_dir <save_path> \
--logdir <log_save_path> \
--resume <pretrained_ckpt>
```
Note: 
The `--resume` argument indicates that the model will be fine-tuned using weights from contrastive pretraining. 
If this argument is not provided, the model will be trained from scratch solely for classification.

#### Evaluation
Open `testDemo.ipynb` in Jupyter, set the `CKPT_PATH` and `TEST_CSV` variables to your checkpoint and test CSV paths, 
then execute the notebook.