# iwslt-2021
Systems submitted to IWSLT 2021 by the MT-UPC group.

## Setting up the environment
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

## Data Preparation

### MuST-C
Download MuST-C v2 to `$MUSTC_ROOT`:
```bash
mkdir -p ${MUSTC_ROOT} && wget http://ftp.scc.kit.edu/pub/mirror/IAR/MUSTC_v2.0_en-de.tar.gz -O - | tar -xz -C ${MUSTC_ROOT}
```

### CoVoST
[Download](https://commonvoice.mozilla.org/en/datasets) the English split from Common Voice v4. Unpack it to `${COVOST_ROOT}/en/`.

### Europarl-ST
Download Europarl-ST v1.1 to `$EUROPARLST_ROOT`:
```bash
mkdir -p ${EUROPARLST_ROOT} && wget https://www.mllp.upv.es/europarl-st/v1.1.tar.gz -O - | tar -xz --strip-components 1 -C ${EUROPARLST_ROOT}
```



