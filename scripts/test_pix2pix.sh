set -ex
CUDA_CACHE_PATH='/home/user/cudacache' python test.py --dataroot ./datasets/facades --name facades_pix2pix --model pix2pix --which_model_netG unet_256 --input_nc 4 --output_nc 5 --which_direction BtoA --dataset_mode aligned --norm batch
