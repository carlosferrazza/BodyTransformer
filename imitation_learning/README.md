# Imitation Learning Instructions

## Installation
To install the requirements in a fresh conda env:
```
conda create --name body_transformer python=3.10.9
conda activate body_transformer
pip install -r requirements.txt
```

## Train the policy

To train the policy on Adroit Hand tasks, run the following command:
```
python run_adroit.py --env door-expert-v2
```

To train the policy on MoCapAct, first download and process the MoCapAct dataset (see section below), then run the following command:
```
python run_mocapact.py --data_dir $MOCAPACT_DIR
```

## Prepare the MoCapAct data

Download the MoCapAct dataset, following <b>either</b> of the following two options:
1) Follow the instructions on the [official website](https://github.com/microsoft/MoCapAct#dataset). We recommend following the AzCopy instructions therein.
2) If you are on x86 Linux, run the following commands:
```
export MOCAPACT_DIR=SET_YOUR_TARGET_DIR_HERE
cd azcopy
./download.sh $MOCAPACT_DIR
cd $MOCAPACT_DIR
tar xzf small.tar.gz --transform='s/.*\///'
```
As a result of this step, your $MOCAPACT_DIR should be populated with many .hdf5 files.

Then, run the following command to process the data (this may take a while):
```
cd BodyTransformer
PYTHONPATH=. python utils/mocapact_utils.py $MOCAPACT_DIR
```