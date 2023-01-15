python train/train_crossmodality.py       --config script/S2P_CUFS_CUFSF/CrossModal_S2P_CUFS_CUFSF_step1_1.yaml  --no_recognition

python train/train_crossmodality.py       --config script/S2P_CUFS_CUFSF/CrossModal_S2P_CUFS_CUFSF_step1_1.yaml\
,script/S2P_CUFS_CUFSF/CrossModal_S2P_CUFS_CUFSF_step1_2.yaml  --no_recognition

python train/train_crossmodality.py       --config script/S2P_CUFS_CUFSF/CrossModal_S2P_CUFS_CUFSF_step2_1.yaml --no_recognition

python train/train_crossmodality.py       --config script/S2P_CUFS_CUFSF/CrossModal_S2P_CUFS_CUFSF_step2_1.yaml\
,script/S2P_CUFS_CUFSF/CrossModal_S2P_CUFS_CUFSF_step2_2.yaml  --no_recognition

#bash script/S2P_CUFS_CUFSF/S2P_CUFS_CUFSF_no_recognition.sh