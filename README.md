# iwslt-2021
Systems submitted to IWSLT 2021 by the MT-UPC group.

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
git clone -b tmp https://github.com/gegallego/fairseq.git ${FAIRSEQ_ROOT} && \
pip install --editable ${FAIRSEQ_ROOT}
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

Download the "mBART 50 finetuned one-to-many" version:
```bash
mkdir -p ${MBART_ROOT} && wget https://dl.fbaipublicfiles.com/fairseq/models/mbart50/mbart50.ft.1n.tar.gz -O - | tar -xz --strip-components 1 -C ${MBART_ROOT}
```


## Data Preparation

Set the environment variables:
```bash
export MUSTC_ROOT=...        # where you'll download MuST-C data
export COVOST_ROOT=...       # where you'll download CoVoST data
export EUROPARLST_ROOT=...   # where you'll download Europarl-ST data
export DATA_ROOT=...         # where you'll combine the datasets
```
### MuST-C
Download MuST-C v2 to `$MUSTC_ROOT`:
```bash
mkdir -p ${MUSTC_ROOT} && wget http://ftp.scc.kit.edu/pub/mirror/IAR/MUSTC_v2.0_en-de.tar.gz -O - | tar -xz -C ${MUSTC_ROOT}
```

Copy the mBART vocabulary to `$MUSTC_ROOT/en-de/`.
```bash
cp $MBART_ROOT/dict.en_XX.txt $MUSTC_ROOT/en-de/spm_bpe250000_asr.txt
cp $MBART_ROOT/dict.de_DE.txt $MUSTC_ROOT/en-de/spm_bpe250000_st.txt
cp $MBART_ROOT/sentence.bpe.model $MUSTC_ROOT/en-de/spm_bpe250000_asr.model
cp $MBART_ROOT/sentence.bpe.model $MUSTC_ROOT/en-de/spm_bpe250000_st.model
```

You also need to add some extra tokens to the dictionary:
```bash
cat $MBART_ROOT/ML50_langs.txt | cut -d'_' -f1 | sed 's/^/<lang:/g' | sed 's/$/> 1/g' >> $MUSTC_ROOT/en-de/spm_bpe250000_asr.txt && \
echo "<mask> 1" >> $MUSTC_ROOT/en-de/spm_bpe250000_asr.txt

cat $MBART_ROOT/ML50_langs.txt | cut -d'_' -f1 | sed 's/^/<lang:/g' | sed 's/$/> 1/g' >> $MUSTC_ROOT/en-de/spm_bpe250000_st.txt && \
echo "<mask> 1" >> $MUSTC_ROOT/en-de/spm_bpe250000_st.txt
```

Run the scripts that will generate the TSV files.
```bash
python $FAIRSEQ_ROOT/examples/speech_to_text/prep_mustc_data.py \
  --data-root $MUSTC_ROOT --task asr --vocab-type bpe \
  --vocab-size 250000 --use-audio-input --prepend-tgt-lang-tag

python $FAIRSEQ_ROOT/examples/speech_to_text/prep_mustc_data.py \
  --data-root $MUSTC_ROOT --task st --vocab-type bpe \
  --vocab-size 250000 --use-audio-input --prepend-tgt-lang-tag
```

### CoVoST
[Download](https://commonvoice.mozilla.org/en/datasets) the English split from Common Voice v4. Unpack it to `${COVOST_ROOT}/en/`.

This dataset contains MP3 files at 48 kHz. We need to convert them to WAV to load them with the fairseq speech_to_text modules. Moreover, we'll need resample them to 16 kHz, which is the sample rate needed by Wav2Vec.

```bash
for f in ${COVOST_ROOT}/*/clips/*.mp3; do
    ffmpeg -i $f -ar 16000 -hide_banner -loglevel error "${f%.mp3}.wav" && rm $f
done
sed 's/\.mp3\t/\.wav\t/g' ${COVOST_ROOT}/**/*.tsv
```

Copy the mBART vocabulary to `$COVOST_ROOT/en/`.
```bash
cp $MBART_ROOT/dict.en_XX.txt $COVOST_ROOT/en/spm_bpe250000_asr_en.txt
cp $MBART_ROOT/dict.de_DE.txt $COVOST_ROOT/en/spm_bpe250000_st_en_de.txt
cp $MBART_ROOT/sentence.bpe.model $COVOST_ROOT/en/spm_bpe250000_asr_en.model
cp $MBART_ROOT/sentence.bpe.model $COVOST_ROOT/en/spm_bpe250000_st_en_de.model
```

You also need to add some extra tokens to the dictionary:
```bash
cat $MBART_ROOT/ML50_langs.txt | cut -d'_' -f1 | sed 's/^/<lang:/g' | sed 's/$/> 1/g' >> $COVOST_ROOT/en/spm_bpe250000_asr_en.txt && \
echo "<mask> 1" >> $COVOST_ROOT/en/spm_bpe250000_asr_en.txt

cat $MBART_ROOT/ML50_langs.txt | cut -d'_' -f1 | sed 's/^/<lang:/g' | sed 's/$/> 1/g' >> $COVOST_ROOT/en/spm_bpe250000_st_en_de.txt && \
echo "<mask> 1" >> $COVOST_ROOT/en/spm_bpe250000_st_en_de.txt
```

Run the scripts that will generate the TSV files.
```bash
python $FAIRSEQ_ROOT/examples/speech_to_text/prep_covost_data.py \
  --data-root $COVOST_ROOT --vocab-type bpe --vocab-size 250000 \
  --src-lang en --use-audio-input --prepend-tgt-lang-tag

python $FAIRSEQ_ROOT/examples/speech_to_text/prep_covost_data.py \
  --data-root $COVOST_ROOT --vocab-type bpe --vocab-size 250000 \
  --src-lang en --tgt-lang de --use-audio-input --prepend-tgt-lang-tag
```


### Europarl-ST
Download Europarl-ST v1.1 to `$EUROPARLST_ROOT`:
```bash
mkdir -p ${EUROPARLST_ROOT} && wget https://www.mllp.upv.es/europarl-st/v1.1.tar.gz -O - | tar -xz --strip-components 1 -C ${EUROPARLST_ROOT}
```

This dataset contains M4A files at 44.1 kHz. As we did with CoVoST, we need to convert them to WAV and resample them to 16 kHz.

```bash
for f in ${EUROPARLST_ROOT}/*/audios/*.m4a; do
    ffmpeg -i $f -ar 16000 -hide_banner -loglevel error "${f%.m4a}.wav" && rm $f
done
```

Copy the mBART vocabulary to `$EUROPARLST_ROOT/en/`.
```bash
cp $MBART_ROOT/dict.en_XX.txt $EUROPARLST_ROOT/en/spm_bpe250000_en-de_asr.txt
cp $MBART_ROOT/dict.de_DE.txt $EUROPARLST_ROOT/en/spm_bpe250000_en-de_st.txt
cp $MBART_ROOT/sentence.bpe.model $EUROPARLST_ROOT/en/spm_bpe250000_en-de_asr.model
cp $MBART_ROOT/sentence.bpe.model $EUROPARLST_ROOT/en/spm_bpe250000_en-de_st.model
```

You also need to add some extra tokens to the dictionary:
```bash
cat $MBART_ROOT/ML50_langs.txt | cut -d'_' -f1 | sed 's/^/<lang:/g' | sed 's/$/> 1/g' >> $EUROPARLST_ROOT/en/spm_bpe250000_en-de_asr.txt && \
echo "<mask> 1" >> $EUROPARLST_ROOT/en/spm_bpe250000_en-de_asr.txt

cat $MBART_ROOT/ML50_langs.txt | cut -d'_' -f1 | sed 's/^/<lang:/g' | sed 's/$/> 1/g' >> $EUROPARLST_ROOT/en/spm_bpe250000_en-de_st.txt && \
echo "<mask> 1" >> $EUROPARLST_ROOT/en/spm_bpe250000_en-de_st.txt
```

Run the scripts that will generate the TSV files.
```bash
python $FAIRSEQ_ROOT/examples/speech_to_text/prep_europarlst_data.py \
  --data-root $EUROPARLST_ROOT --lang-pair en-de --task asr \
  --vocab-type bpe --vocab-size 250000 \
  --use-audio-input --prepend-tgt-lang-tag

python $FAIRSEQ_ROOT/examples/speech_to_text/prep_europarlst_data.py \
  --data-root $EUROPARLST_ROOT --lang-pair en-de --task st \
  --vocab-type bpe --vocab-size 250000 \
  --use-audio-input --prepend-tgt-lang-tag
```

## Data Filtering

Run the filtering script to produce a clean version of a dataset.
The script takes as input the path of the dataset, reads the (train split) TSV files
for the two tasks (ASR and ST) and produces two new TSV files. The cleaner can optionally
run an ASR system (asr_inference.py) to remove noisy and potential inconsitent examples.
PS: Tested it for MuST-C for now. Will run it for the other two when we have the data,
but since their TSV files are in the same format, should be fine.

```bash
python scripts/filtering/filter_tsv.py \
  --tsv_root $MUSTC_ROOT/en-de --duration_sec_threshold 30 \
  --apply_asr_filtering --asr_batch_size 24 --asr_wer_threshold 0.5
```

## Combine datasets
Generate symlinks pointing to dataset files in `$DATA_ROOT`:
```bash
ln -s $MUSTC_ROOT/en-de/config_st.yaml $DATA_ROOT/config.yaml
ln -s $MUSTC_ROOT/en-de/spm_bpe250000_st.{txt,model} $DATA_ROOT/

ln -s $MUSTC_ROOT/en-de/train_st_filtered.tsv $DATA_ROOT/train_mustc.tsv
ln -s $MUSTC_ROOT/en-de/dev_st.tsv $DATA_ROOT/dev_mustc.tsv
ln -s $MUSTC_ROOT/en-de/tst-COMMON_st.tsv $DATA_ROOT/tst-COMMON_mustc.tsv
ln -s $MUSTC_ROOT/en-de/tst-HE_st.tsv $DATA_ROOT/tst-HE_mustc.tsv

ln -s $COVOST_ROOT/en/train_st_en_de_filtered.tsv $DATA_ROOT/train_covost.tsv
ln -s $COVOST_ROOT/en/dev_st_en_de.tsv $DATA_ROOT/dev_covost.tsv
ln -s $COVOST_ROOT/en/test_st_en_de.tsv $DATA_ROOT/test_covost.tsv

ln -s $EUROPARLST_ROOT/en/train_en-de_st_filtered.tsv $DATA_ROOT/train_europarlst.tsv
ln -s $EUROPARLST_ROOT/en/dev_en-de_st.tsv $DATA_ROOT/dev_europarlst.tsv
ln -s $EUROPARLST_ROOT/en/test_en-de_st.tsv $DATA_ROOT/test_europarlst.tsv
```
