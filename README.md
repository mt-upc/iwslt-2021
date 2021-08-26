
# UPC's Speech Translation System for IWSLT 2021

System submitted to the [IWSLT 2021](https://iwslt.org/2021/) [offline speech translation task](https://iwslt.org/2021/offline) by the [UPC Machine Translation group](https://mt.cs.upc.edu).

The pre-print of the submssion is available on [arxiv](https://arxiv.org/abs/2105.04512).

<em>This paper describes the submission to the IWSLT 2021 offline speech translation task by the UPC Machine Translation group. The task consists of building a system capable of translating English audio recordings extracted from TED talks into German text. Submitted systems can be either cascade or end-to-end and use a custom or given segmentation. Our submission is an end-to-end speech translation system, which combines pre-trained models (Wav2Vec 2.0 and mBART) with coupling modules between the encoder and decoder, and uses an efficient fine-tuning technique, which trains only 20% of its total parameters. We show that adding an Adapter to the system and pre-training it, can increase the convergence speed and the final result, with which we achieve a BLEU score of 27.3 on the MuST-C test set. Our final model is an ensemble that obtains 28.22 BLEU score on the same set. Our submission also uses a custom segmentation algorithm that employs pre-trained Wav2Vec 2.0 for identifying periods of untranscribable text and can bring improvements of 2.5 to 3 BLEU score on the IWSLT 2019 test set, as compared to the result with the given segmentation.</em>

Results             |  Learning Curves
:-------------------------:|:-------------------------:
![](/readme_figures/tbl.PNG) | ![](/readme_figures/fig.PNG)

Cite this work as:
```
@inproceedings{gallego2021iwslt,
    title = "End-to-End Speech Translation with Pre-trained Models and Adapters: UPC at IWSLT 2021",
    author = "G{\'a}llego, Gerard I. and Tsiamas, Ioannis and Escolano, Carlos and Fonollosa, Jos{\'e} A. R.  and Costa-juss{\`a}, Marta R.",
    booktitle = "Proceedings of the 18th International Conference on Spoken Language Translation (IWSLT 2021)",
    month = aug,
    year = "2021",
    address = "Bangkok, Thailand (online)",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2021.iwslt-1.11",
    doi = "10.18653/v1/2021.iwslt-1.11",
    pages = "110--119",
}
```

## Setting up the environment

Set the environment variables:

```bash
export IWSLT_ROOT=...        # where you'll clone this repository
export FAIRSEQ_ROOT=...      # where you'll clone our Fairseq fork
```

Clone this repository to `$IWSLT_ROOT`:

```bash
git clone https://github.com/mt-upc/iwslt-2021.git ${IWSLT_ROOT} 
```
Create a conda environment using the `environment.yml` file and activate it:

```bash
conda env create -f ${IWSLT_ROOT}/environment.yml && \
conda activate iwslt21
```
We have been working on some PR of Fairseq, but they have not been merged yet. Clone our Fairseq fork and install it.

```bash
git clone -b iwslt-2021 https://github.com/gegallego/fairseq.git ${FAIRSEQ_ROOT} && \
pip install --editable ${FAIRSEQ_ROOT}
```

Install NVIDIA's [apex](https://github.com/NVIDIA/apex) library for faster training.

```bash
git clone https://github.com/NVIDIA/apex && cd apex
pip install -v --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" \
  --global-option="--deprecated_fused_adam" --global-option="--xentropy" \
  --global-option="--fast_multihead_attn" ./
cd .. && rm -rf apex
```

## Pre-trained models
In this project we use a pre-trained Wav2Vec 2.0 encoder and a pre-trained mBART decoder.

Set the environment variables:

```bash
export WAV2VEC_ROOT=...      # where you'll download the Wav2Vec checkpoint
export MBART_ROOT=...        # where you'll download the mBART checkpoint
```

Download the "Wav2Vec 2.0 Large (LV-60) + Self Training" version:

```bash
wget https://dl.fbaipublicfiles.com/fairseq/wav2vec/wav2vec_vox_960h_pl.pt -P ${WAV2VEC_ROOT}
```

It is necessary to make a few preparations to load this checkpoint into our model:

```bash
python ./scripts/prepare_wav2vec.py --checkpoint $WAV2VEC_ROOT/wav2vec_vox_960h_pl.pt
```

Download the "mBART 50 finetuned one-to-many" version:

```bash
mkdir -p ${MBART_ROOT} && \
wget https://dl.fbaipublicfiles.com/fairseq/models/mbart50/mbart50.ft.1n.tar.gz -O - | \
  tar -xz --strip-components 1 -C ${MBART_ROOT}
```


## Datasets
Set the environment variables:

```bash
export MUSTC_ROOT=...        # where you'll download MuST-C data
export COVOST_ROOT=...       # where you'll download CoVoST data
export EUROPARLST_ROOT=...   # where you'll download Europarl-ST data
export IWSLT_TEST_ROOT=...   # where you'll download IWSLT test sets
export DATA_ROOT=...         # where you'll combine the datasets
```

### Download

#### MuST-C
Download MuST-C v2 to `$MUSTC_ROOT`:

```bash
mkdir -p ${MUSTC_ROOT} && \
wget http://ftp.scc.kit.edu/pub/mirror/IAR/MUSTC_v2.0_en-de.tar.gz -O - | \
  tar -xz -C ${MUSTC_ROOT}
```

#### CoVoST
[Download](https://commonvoice.mozilla.org/en/datasets) the English split from Common Voice v4. Unpack it to `${COVOST_ROOT}/en/`.

This dataset contains MP3 files at 48 kHz. We need to convert them to WAV to load them with the fairseq speech_to_text modules. Moreover, we'll need resample them to 16 kHz, which is the sample rate needed by Wav2Vec.

```bash
for f in ${COVOST_ROOT}/*/clips/*.mp3; do
  ffmpeg -i $f -ar 16000 -hide_banner -loglevel error "${f%.mp3}.wav" && rm $f
done
sed 's/\.mp3\t/\.wav\t/g' ${COVOST_ROOT}/**/*.tsv
```

#### Europarl-ST
Download Europarl-ST v1.1 to `$EUROPARLST_ROOT`:

```bash
mkdir -p ${EUROPARLST_ROOT} && \
wget https://www.mllp.upv.es/europarl-st/v1.1.tar.gz -O - | \
  tar -xz --strip-components 1 -C ${EUROPARLST_ROOT}
```

This dataset contains M4A files at 44.1 kHz. As we did with CoVoST, we need to convert them to WAV and resample them to 16 kHz.

```bash
for f in ${EUROPARLST_ROOT}/*/audios/*.m4a; do
  ffmpeg -i $f -ar 16000 -hide_banner -loglevel error "${f%.m4a}.wav" && rm $f
done
```

#### IWSLT test sets
Download IWSLT test sets to `$IWSLT_TEST_ROOT`:

```bash
mkdir -p ${IWSLT_TEST_ROOT}/tst2020/ && \
wget https://surfdrive.surf.nl/files/index.php/s/KVYz2OgcVZJGPYK/download -O - | \
  tar -xz --strip-components 1 -C ${IWSLT_TEST_ROOT}/tst2020/
```
```bash
mkdir -p ${IWSLT_TEST_ROOT}/tst2021/ && \
wget https://surfdrive.surf.nl/files/index.php/s/U3blQuKJ2M0L14K/download -O - | \
  tar -xz --strip-components 1 -C ${IWSLT_TEST_ROOT}/tst2021/
```

### Preparation
Add some extra tokens to the mBART dictionary:

```bash
cat $MBART_ROOT/ML50_langs.txt | cut -d'_' -f1 | sed 's/^/<lang:/g' | \
  sed 's/$/> 1/g' >> $MBART_ROOT/dict.en_XX.txt && \
echo "<mask> 1" >> $MBART_ROOT/dict.en_XX.txt

cat $MBART_ROOT/ML50_langs.txt | cut -d'_' -f1 | sed 's/^/<lang:/g' | \
  sed 's/$/> 1/g' >> $MBART_ROOT/dict.de_DE.txt && \
echo "<mask> 1" >> $MBART_ROOT/dict.de_DE.txt
```

Copy the mBART vocabulary to the datasets directories.

```bash
# MuST-C
cp $MBART_ROOT/dict.en_XX.txt $MUSTC_ROOT/en-de/spm_bpe250000_asr.txt
cp $MBART_ROOT/dict.de_DE.txt $MUSTC_ROOT/en-de/spm_bpe250000_st.txt
cp $MBART_ROOT/sentence.bpe.model $MUSTC_ROOT/en-de/spm_bpe250000_asr.model
cp $MBART_ROOT/sentence.bpe.model $MUSTC_ROOT/en-de/spm_bpe250000_st.model

# CoVoST
cp $MBART_ROOT/dict.en_XX.txt $COVOST_ROOT/en/spm_bpe250000_asr_en.txt
cp $MBART_ROOT/dict.de_DE.txt $COVOST_ROOT/en/spm_bpe250000_st_en_de.txt
cp $MBART_ROOT/sentence.bpe.model $COVOST_ROOT/en/spm_bpe250000_asr_en.model
cp $MBART_ROOT/sentence.bpe.model $COVOST_ROOT/en/spm_bpe250000_st_en_de.model

# Europarl-ST
cp $MBART_ROOT/dict.en_XX.txt $EUROPARLST_ROOT/en/spm_bpe250000_en-de_asr.txt
cp $MBART_ROOT/dict.de_DE.txt $EUROPARLST_ROOT/en/spm_bpe250000_en-de_st.txt
cp $MBART_ROOT/sentence.bpe.model $EUROPARLST_ROOT/en/spm_bpe250000_en-de_asr.model
cp $MBART_ROOT/sentence.bpe.model $EUROPARLST_ROOT/en/spm_bpe250000_en-de_st.model
```

Run the scripts that will generate the TSV files.

```bash
# MuST-C
python $FAIRSEQ_ROOT/examples/speech_to_text/prep_mustc_data.py \
  --data-root $MUSTC_ROOT --task asr --vocab-type bpe \
  --vocab-size 250000 --use-audio-input --prepend-tgt-lang-tag

python $FAIRSEQ_ROOT/examples/speech_to_text/prep_mustc_data.py \
  --data-root $MUSTC_ROOT --task st --vocab-type bpe \
  --vocab-size 250000 --use-audio-input --prepend-tgt-lang-tag

# CoVoST
python $FAIRSEQ_ROOT/examples/speech_to_text/prep_covost_data.py \
  --data-root $COVOST_ROOT --vocab-type bpe --vocab-size 250000 \
  --src-lang en --use-audio-input --prepend-tgt-lang-tag

python $FAIRSEQ_ROOT/examples/speech_to_text/prep_covost_data.py \
  --data-root $COVOST_ROOT --vocab-type bpe --vocab-size 250000 \
  --src-lang en --tgt-lang de --use-audio-input --prepend-tgt-lang-tag

# Europarl-ST
python $FAIRSEQ_ROOT/examples/speech_to_text/prep_europarlst_data.py \
  --data-root $EUROPARLST_ROOT --lang-pair en-de --task asr \
  --vocab-type bpe --vocab-size 250000 \
  --use-audio-input --prepend-tgt-lang-tag

python $FAIRSEQ_ROOT/examples/speech_to_text/prep_europarlst_data.py \
  --data-root $EUROPARLST_ROOT --lang-pair en-de --task st \
  --vocab-type bpe --vocab-size 250000 \
  --use-audio-input --prepend-tgt-lang-tag

# IWSLT test sets
python $IWSLT_ROOT/scripts/prepare_iwslt_tst.py --data-root $IWSLT_TEST_ROOT
```

### Filtering
Run the filtering script to produce a clean version of a dataset. The script does:
- Text filtering by formatting the target text of an example and removal of empty examples after text filtering.
- Noise filtering by removing completelly an example based on predictions from an ASR system.

```bash
# MuST-C
python $IWSLT_ROOT/scripts/filtering/filter_tsv.py \
  --dataset_name MUSTC --tsv_root $MUSTC_ROOT/en-de \
  --asr_batch_size 24 --asr_wer_threshold 0.5

# CoVoST
python $IWSLT_ROOT/scripts/filtering/filter_tsv.py \
  --dataset_name COVOST --tsv_root $COVOST_ROOT/en \
  --asr_batch_size 24 --asr_wer_threshold 0.5

# Europarl-ST
python $IWSLT_ROOT/scripts/filtering/filter_tsv.py \
  --dataset_name EUROPARLST --tsv_root $EUROPARLST_ROOT/en \
  --asr_batch_size 24 --asr_wer_threshold 0.5
```

### Merge
Generate symlinks pointing to dataset files in `$DATA_ROOT`:

```bash
ln -s $MUSTC_ROOT/en-de/config_st.yaml $DATA_ROOT/config.yaml
ln -s $MUSTC_ROOT/en-de/spm_bpe250000_st.{txt,model} $DATA_ROOT/

ln -s $MUSTC_ROOT/en-de/train_st_filtered.tsv $DATA_ROOT/train_mustc.tsv
ln -s $MUSTC_ROOT/en-de/dev_st.tsv $DATA_ROOT/dev_mustc.tsv
ln -s $MUSTC_ROOT/en-de/tst-COMMON_st.tsv $DATA_ROOT/tst-COMMON_mustc.tsv
ln -s $MUSTC_ROOT/en-de/tst-HE_st.tsv $DATA_ROOT/tst-HE_mustc.tsv

ln -s $COVOST_ROOT/en/train_st_en_de_filtered.tsv $DATA_ROOT/train_covost.tsv
ln -s $COVOST_ROOT/en/dev_st_en_de_filtered.tsv $DATA_ROOT/train_dev_covost.tsv
ln -s $COVOST_ROOT/en/test_st_en_de.tsv $DATA_ROOT/test_covost.tsv

ln -s $EUROPARLST_ROOT/en/train_en-de_st_filtered.tsv $DATA_ROOT/train_europarlst.tsv
ln -s $EUROPARLST_ROOT/en/dev_en-de_st_filtered.tsv $DATA_ROOT/train_dev_europarlst.tsv
ln -s $EUROPARLST_ROOT/en/test_en-de_st.tsv $DATA_ROOT/test_europarlst.tsv

ln -s $IWSLT_TEST_ROOT/tst2020.tsv $DATA_ROOT/tst2020_iwslt.tsv
ln -s $IWSLT_TEST_ROOT/tst2021.tsv $DATA_ROOT/tst2021_iwslt.tsv
```

## Experiments
Set the environment variables:

```bash
export SAVE_DIR=...          # where the checkpoints, hydra logs and tensorboard logs will be saved
```

NOTE: When launching the trainings, take into account:
- If you use a different number of GPUs, remember to set the appropriate value for `+optimization.update_freq='[x]'` (where `x = old_update_freq * old_n_gpus / new_n_gpus`).

- We recommend to set `+dataset.num_workers=y` for a better performance (where `y = n_cpu // 2`).

### 1) LNA-ED
Run the following command to train the model:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 \
fairseq-hydra-train \
  --config-dir ${IWSLT_ROOT}/config/ \
  --config-name lna_ed.yaml
```

Generate predictions of the MuST-C dev and test sets:

```bash
for subset in {dev_mustc,tst-COMMON_mustc}; do
  fairseq-generate ${DATA_ROOT} \
    --path ${SAVE_DIR}/lna_ed/ckpts/checkpoint_best.pt \
    --results-path ${SAVE_DIR}/lna_ed/results/ \
    --user-dir ${IWSLT_ROOT}/fairseq_modules \
    --task speech_to_text_iwslt21 --gen-subset $subset \
    --max-source-positions 960000 --max-tokens 960000   \
    --skip-invalid-size-inputs-valid-test --prefix-size 1 \
    --beam 5 --scoring sacrebleu
done
```

### 2) LNA-ED + Adapter
Run the following command to train the model:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 \
fairseq-hydra-train \
  --config-dir ${IWSLT_ROOT}/config/ \
  --config-name lna_ed_adapt.yaml
```

Run the following command to fine-tune it with just with MuST-C:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 \
fairseq-hydra-train \
  --config-dir ${IWSLT_ROOT}/config/ \
  --config-name lna_ed_adapt_ft.yaml
```

Generate predictions of the MuST-C dev and test sets:

```bash
for experiment in {lna_ed_adapt, lna_ed_adapt_ft}; do
  for subset in {dev_mustc,tst-COMMON_mustc}; do
    fairseq-generate ${DATA_ROOT} \
      --path ${SAVE_DIR}/${experiment}/ckpts/checkpoint_best.pt \
      --results-path ${SAVE_DIR}/${experiment}/results/ \
      --user-dir ${IWSLT_ROOT}/fairseq_modules \
      --task speech_to_text_iwslt21 --gen-subset $subset \
      --max-source-positions 960000 --max-tokens 960000   \
      --skip-invalid-size-inputs-valid-test --prefix-size 1 \
      --beam 5 --scoring sacrebleu
  done
done
```

### 3) LNA-ED + Adapter (2 Steps)
Run the following command to train the first step of the experiment:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 \
fairseq-hydra-train \
  --config-dir ${IWSLT_ROOT}/config/ \
  --config-name lna_ed_adapt_2step_1.yaml
```

Run this command to train the second step:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 \
fairseq-hydra-train \
  --config-dir ${IWSLT_ROOT}/config/ \
  --config-name lna_ed_adapt_2step_2.yaml
```

Generate predictions of the MuST-C dev and test sets:

```bash
for subset in {dev_mustc,tst-COMMON_mustc}; do
  fairseq-generate ${DATA_ROOT} \
    --path ${SAVE_DIR}/lna_ed_adapt_2step_2/ckpts/checkpoint_best.pt \
    --results-path ${SAVE_DIR}/lna_ed_adapt_2step_2/results/ \
    --user-dir ${IWSLT_ROOT}/fairseq_modules \
    --task speech_to_text_iwslt21 --gen-subset $subset \
    --max-source-positions 960000 --max-tokens 960000   \
    --skip-invalid-size-inputs-valid-test --prefix-size 1 \
    --beam 5 --scoring sacrebleu
done
```

## Prepare Test Sets with provided and custom Segmentation

To get token predictions from Wav2Vec 2.0

```bash
for subset in {tst2020,tst2021}; do
  python ${IWSLT_ROOT}/scripts/segmentation/get_predictions.py --wav_dir ${IWSLT_TEST_ROOT}/${subset}/wavs
done
```

Use the predictions to segment the audio files with the maximum segment length of 22, which was found to maximize BLEU score on the tst2019.

```bash
for subset in {tst2020,tst2021}; do
  python ${IWSLT_ROOT}scripts/segmentation/segment_audio.py --dataset_root ${IWSLT_TEST_ROOT}/${subset} --max_segm_len 22
done
```

Prepare test sets for generation

```bash
for subset in {tst2020,tst2021}; do
  python ${IWSLT_ROOT}/scripts/prepare_iwslt_tst.py -d ${IWSLT_TEST_ROOT}/${subset}
  python ${IWSLT_ROOT}/scripts/prepare_iwslt_tst.py -d ${IWSLT_TEST_ROOT}/${subset} -c
done
```

Generate predictions for the two test sets for given and own segmentation

```bash
experiment_name=...
for subset in {tst2020,tst2020_own_22,tst2021,tst2021_own_22}; do
  fairseq-generate ${DATA_ROOT} \
    --path ${SAVE_DIR}/${experiment_name}/ckpts/checkpoint_best.pt \
    --results-path ${SAVE_DIR}/${experiment_name}/results/ \
    --user-dir ${IWSLT_ROOT}/fairseq_modules \
    --task speech_to_text_iwslt21 --gen-subset $subset \
    --max-source-positions 960000 --max-tokens 960000   \
    --prefix-size 1 \
    --beam 5 --scoring sacrebleu
done
```
