from boxx import *

data_root= "/data/ws/blender_syn"
dataset_names = []
for dataset_name in listdir(data_root):
    dir = (pathjoin(data_root, dataset_name, 'dataset_ln'))
    if isdir(dir):
        dataset_names.append(dataset_name)

def f(dataset_name):
    cmd = f"""rlaunch --cpu 24 --gpu 8 --memory 90000 --  python tools/train_net.py  --num-gpus 8 \
	--config-file configs/COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_1x.yaml \
    DATASETS.TRAIN '("{dataset_name}_train",)' DATASETS.TEST '("{dataset_name}_val",)' \
    MODEL.ROI_HEADS.NUM_CLASSES 1 OUTPUT_DIR ./output/{dataset_name}_v2"""
    os.system(cmd)


# dataset_names = dataset_names[:2]
mapmp(f, dataset_names, pool=len(dataset_names))

