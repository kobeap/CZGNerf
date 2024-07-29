#!/bin/bash

python main.py /mnt/data1/zpf/ernerf/girl3/girl3_lan_3ddfa --workspace /mnt/data1/zpf/ernerf/girl3/girl3_lan_3ddfa/add_loss_nolip_optmwdn_trial_girl3 -O --iters 100000 --asr_model hubert 
python main.py /mnt/data1/zpf/ernerf/girl3/girl3_lan_3ddfa --workspace /mnt/data1/zpf/ernerf/girl3/girl3_lan_3ddfa/add_loss_nolip_optmwdn_trial_girl3 -O --iters 125000 --finetune_lips --patch_size 32 --asr_model hubert
python main.py /mnt/data1/zpf/ernerf/girl3/girl3_lan_3ddfa --workspace /mnt/data1/zpf/ernerf/girl3/girl3_lan_3ddfa/add_loss_nolip_optmwdn_trial_girl3_torso -O --torso --iters 125000 --head_ckpt /mnt/data1/zpf/ernerf/girl3/girl3_lan_3ddfa/add_loss_nolip_optmwdn_trial_girl3/checkpoints/ngp_ep0062.pth --iters 200000 --asr_model hubert
python main.py /mnt/data1/zpf/ernerf/girl3/girl3_lan_3ddfa --workspace /mnt/data1/zpf/ernerf/girl3/girl3_lan_3ddfa/add_loss_nolip_optmwdn_trial_girl3_torso -O --torso --test  --asr_model hubert --smooth_path
