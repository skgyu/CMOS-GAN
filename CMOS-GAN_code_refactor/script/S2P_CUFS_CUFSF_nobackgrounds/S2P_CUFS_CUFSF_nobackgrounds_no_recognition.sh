python train/train_crossmodality.py       --config script/S2P_CUFS_CUFSF_nobackgrounds/CrossModal_S2P_CUFS_CUFSF_nobackgrounds_step1_1.yaml --no_recognition

python train/train_crossmodality.py       --config script/S2P_CUFS_CUFSF_nobackgrounds/CrossModal_S2P_CUFS_CUFSF_nobackgrounds_step1_1.yaml\
,script/S2P_CUFS_CUFSF_nobackgrounds/CrossModal_S2P_CUFS_CUFSF_nobackgrounds_step1_2.yaml  --no_recognition

python train/train_crossmodality.py       --config script/S2P_CUFS_CUFSF_nobackgrounds/CrossModal_S2P_CUFS_CUFSF_nobackgrounds_step2_1.yaml  --no_recognition

python train/train_crossmodality.py       --config script/S2P_CUFS_CUFSF_nobackgrounds/CrossModal_S2P_CUFS_CUFSF_nobackgrounds_step2_1.yaml\
,script/S2P_CUFS_CUFSF_nobackgrounds/CrossModal_S2P_CUFS_CUFSF_nobackgrounds_step2_2.yaml  --no_recognition

#bash script/S2P_CUFS_CUFSF_nobackgrounds/S2P_CUFS_CUFSF_nobackgrounds_no_recognition.sh