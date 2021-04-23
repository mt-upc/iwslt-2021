export IWSLT_TST_ROOT=/home/usuaris/scratch/ioannis.tsiamas/datasets/speech_translation/IWSLT/
export RANGE=5,25
export RESULTS_ROOT=/home/usuaris/scratch/ioannis.tsiamas/iwslt-2021/save_dir/lna_ed_adapt/results/
export MWER_ROOT=/home/usuaris/veu/ioannis.tsiamas/mwerSegmenter/

1. get token predictions from wav2vec for tst2019

	python ${IWSLT_ROOT}scripts/segmentation/get_predictions.py --wav_dir ${IWSLT_TST_ROOT}IWSLT.tst2019/wavs

	saves the predictions of wav2vec for each wav file at ${IWSLT_TST_ROOT}/IWSLT.tst2019/token_predictions.json

2. produce segmentation for N values of max_segm_len for tst2019

	python ${IWSLT_ROOT}scripts/segmentation/segment_audio.py --dataset_root ${IWSLT_TST_ROOT}IWSLT.tst2019 --max_segm_len $RANGE

	saves the yaml segmentation for each $max_segm_len in $RANGE at ${IWSLT_TST_ROOT}IWSLT.tst2019/own_segmentation/IWSLT.TED.tst2019.en-de_own_${max_segm_len}.yaml

3. produce tsv files for these N yaml segmentations

	python ${IWSLT_ROOT}scripts/prepare_iwslt_tst.py -d ${IWSLT_TST_ROOT}IWSLT.tst2019 -c

	saves the tsv file for each $max_segm_len at ${IWSLT_TST_ROOT}IWSLT.tst2019_own_${max_segm_len}.tsv

4. run generation script for these N segmentations

	sbatch ${IWSLT_ROOT}jobs/gen4custom.sh

	saves generation output files for each $max_segm_len at ${RESULTS_ROOT}/generate-IWSLT.tst2019_own_${max_segm_len}-checkpoint.best_loss_169.50.pt.txt

	rename them to generate-IWSLT.tst2019_own_${max_segm_len}-checkpoint_best.pt.txt

	move them to folder ${RESULTS_ROOT}IWSLT.tst2019/

5. process the output of the generation into the required format

	python ${IWSLT_ROOT}scripts/format_generation_output.py -p ${RESULTS_ROOT}IWSLT.tst2019/

	saves the txt files at ${RESULTS_ROOT}IWSLT.tst2019/generate-IWSLT.tst2019_own_${max_segm_len}-checkpoint_best.pt_formated.txt

	copy them to ${MWER_ROOT}files/

- python ${IWSLT_ROOT}scripts/correct_order.py

6. run mwerSegmenter script to score each generation

	bash ${MWER_ROOT}align_segmentations_.sh

	saves the aligned generations at ${MWER_ROOT}files/own_segm_${max_segm_len}.xml

7. get the value of max_segm_len that corresponds to highest BLEU

	python ${IWSLT_ROOT}scripts/segmentation/score_segmentations.py -s ${MWER_ROOT}files/ -r ${MWER_ROOT}files/IWSLT.TED.tst2019.en-de.de.xml


for 2020 & 2021 and the best value $max_segm_len=22

python ${IWSLT_ROOT}scripts/segmentation/get_predictions.py --wav_dir ${IWSLT_TST_ROOT}IWSLT.tst2020/wavs
python ${IWSLT_ROOT}scripts/segmentation/get_predictions.py --wav_dir ${IWSLT_TST_ROOT}IWSLT.tst2021/wavs

mkdir ${IWSLT_TST_ROOT}IWSLT.tst2020/own_segmentation
mkdir ${IWSLT_TST_ROOT}IWSLT.tst2021/own_segmentation

python ${IWSLT_ROOT}scripts/segmentation/segment_audio.py --dataset_root ${IWSLT_TST_ROOT}IWSLT.tst2020 --max_segm_len 22
python ${IWSLT_ROOT}scripts/segmentation/segment_audio.py --dataset_root ${IWSLT_TST_ROOT}IWSLT.tst2021 --max_segm_len 22

python ${IWSLT_ROOT}scripts/prepare_iwslt_tst.py -d ${IWSLT_TST_ROOT}IWSLT.tst2020 -c
python ${IWSLT_ROOT}scripts/prepare_iwslt_tst.py -d ${IWSLT_TST_ROOT}IWSLT.tst2021 -c
python ${IWSLT_ROOT}scripts/prepare_iwslt_tst.py -d ${IWSLT_TST_ROOT}IWSLT.tst2020
python ${IWSLT_ROOT}scripts/prepare_iwslt_tst.py -d ${IWSLT_TST_ROOT}IWSLT.tst2021

/home/usuaris/scratch/ioannis.tsiamas/datasets/speech_translation/IWSLT/IWSLT.tst2021.tsv
/home/usuaris/scratch/ioannis.tsiamas/datasets/speech_translation/IWSLT/IWSLT.tst2020.tsv
/home/usuaris/scratch/ioannis.tsiamas/datasets/speech_translation/IWSLT/IWSLT.tst2021_own_22.tsv
/home/usuaris/scratch/ioannis.tsiamas/datasets/speech_translation/IWSLT/IWSLT.tst2020_own_22.tsv