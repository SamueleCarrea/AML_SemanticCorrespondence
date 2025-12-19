# SPair-71k setup

Place the SPair-71k dataset under this `data/` directory with the official structure:

```
data/
└── SPair-71k/
    ├── JPEGImages/
    ├── PairAnnotation/
    │   └── <category>/*.json
    ├── ImageSets/
    │   └── main/{train,val,test}.txt
    ├── symmetry.txt
    └── keypoints/
        └── <category>.json
```

Notes:
- `symmetry.txt` is optional but recommended if you leverage symmetric correspondences.
- The `keypoints/` directory should contain the category-wise JSON files from the SPair-71k release (one file per class with keypoint definitions).
- Split lists (`train.txt`, `val.txt`, `test.txt`) should match the filenames of the pair annotation JSONs (without extensions).

You can then initialize the dataset with:

```python
from dataset import SPairDataset

dataset = SPairDataset(root="data/SPair-71k", split="train", long_side=480)
```
