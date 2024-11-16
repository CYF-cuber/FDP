# FDP
Micro-Expression Recognition via Fine-Grained Dynamic Perception

Our model is implemented with PyTorch 1.13.0 and Python 3.9. 

## Preparation
1.Install requried packages:
```
$ pip install -r requirements.txt
```

3.Make training and testing datasets:
The datasets like [CASME II](http://casme.psych.ac.cn/casme/c2) and [SAMM](https://helward.mmu.ac.uk/STAFF/M.Yap/dataset.php), and should follow such folder structure.
```
├──CASME2_data_5/
│  ├── disgust
│  │   ├── 01_EP19_05f
│  │   │   ├── img1.jpg
│  │   │   ├── img2.jpg
│  │   │   ├── ......
│  ├── surprise
│  │   ├── ......

├──SAMM_data_5/
│  ├── anger
│  │   ├── 006_1_2
│  │   │   ├── 006_05562.jpg
│  │   │   ├── 006_05563.jpg
│  │   │   ├── ......
│  ├── contempt
│  │   ├── ......

```
Running data.py makes datasets for each subject:
```
$ python data.py --dataset SAMM
```
## Train & Evaluate 
If you have already made ME datasets, you can simply train FDP like this:
```
$ python train.py --dataset SAMM
```

