#!/usr/bin/env bash
set -euo pipefail

FASTQ="/mnt/disk3/baigeihui_Data/Athaliana_BsSeq/CRR1048701_f1.fq"
REF_FA="/mnt/disk3/baigeihui_Data/Athaliana_Ref/ncbi_dataset/data/GCF_000001735.4/GCF_000001735.4_TAIR10.1_genomic.fa"
WORK="/mnt/disk3/baigeihui_Data/Athaliana_BsSeq/bismark_run_CRR1048701_f1"
SAMPLE="CRR1048701_f1"
THREADS=16

mkdir -p "$WORK/genome" "$WORK/tmp"

cp "$REF_FA" "$WORK/genome/"
REF_COPY="$WORK/genome/$(basename "$REF_FA")"

bismark_genome_preparation --bowtie2 "$WORK/genome"

cutadapt \
  -a AGATCGGAAGAGCACACGTCTGAACTCCAGTCAC \
  --max-n 0.1 \
  -m 60 \
  --trim-n \
  -q 15,15 \
  -j "$THREADS" \
  -o "$WORK/${SAMPLE}.cut.fastq.gz" \
  "$FASTQ" \
  > "$WORK/${SAMPLE}.cutadapt.log" 2>&1

bismark \
  --parallel "$THREADS" \
  -o "$WORK" \
  --temp_dir "$WORK/tmp" \
  --bowtie2 \
  -N 1 \
  -L 30 \
  --genome "$WORK/genome" \
  "$WORK/${SAMPLE}.cut.fastq.gz" \
  > "$WORK/${SAMPLE}.bismark.log" 2>&1

samtools sort -@ "$THREADS" \
  -o "$WORK/${SAMPLE}.cut_bismark_bt2.sorted.bam" \
  "$WORK/${SAMPLE}.cut_bismark_bt2.bam"

samtools sort -n -@ "$THREADS" \
  -o "$WORK/${SAMPLE}.cut_bismark_bt2.name.bam" \
  "$WORK/${SAMPLE}.cut_bismark_bt2.sorted.bam"

cd "$WORK"
bismark_methylation_extractor \
  -s \
  --no_overlap \
  --ignore 4 \
  --comprehensive \
  --parallel "$THREADS" \
  --bedGraph \
  --cytosine_report \
  --zero_based \
  --CX \
  --genome_folder "$WORK/genome" \
  "$WORK/${SAMPLE}.cut_bismark_bt2.name.bam" \
  > "$WORK/${SAMPLE}.extract.log" 2>&1

COV=$(find "$WORK" -maxdepth 1 -type f \( -name "*.cov" -o -name "*zero.cov" \) | head -n 1)

if [ -z "${COV:-}" ]; then
  echo "ERROR: no .cov file found. Please check the output under $WORK"
  exit 1
fi

python "bedcov2bedmethyl.py" \
  --cov "$COV" \
  --genome "$REF_COPY" \
  --motif C \
  --mloc_in_motif 0 \
  --out "$WORK/${SAMPLE}.allC.bed"

echo "DONE"
echo "BED file: $WORK/${SAMPLE}.allC.bed"
