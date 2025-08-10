## Transformer-based Detection of Abnormal Rice Growth using Drone-based Multispectral Imaging

## Preparation (Pre-requisites)


### Code - git clone
```bash
git clone https://github.com/ironluffy/RiceSeg.git
```

### Install MMSegmentation
```bash
cd mmsegmentation
pip install -v -e .
```

### Project directory structure
```bash
.
├── RiceSeg
│   ├── README.md
│   ├── src
│   ├── mmsegmentation
│   ├── sample_dataset
│   ├── pretrained_ckpt
│   └── .gitignore
└── index.html
```


## INFERENCE
```bash
python3 ./mmsegmentation/tools/test.py .{config file path} {checkpoint_path} --eval mIoU --show-dir {output path}
```


###### Sample command using the provided best checkpoint
```bash
python3 mmsegmentation/tools/test.py mmsegmentation/configs/rice/segformer_mit-b4_lovasz_gne_chw.py ./pretrained_ckpt/best.pth --eval mIoU 
```

