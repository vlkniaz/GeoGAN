set -ex
CUDA_CACHE_PATH='/home/user/cudacache' python test.py --dataroot ./datasets/ISPRS --name isprs_geo2label --model geo2label --which_model_netG unet_256 --loadSize 256 --fineSize 256 --input_nc 4 --output_nc 5 --which_direction AtoB --dataset_mode geo --norm batch
