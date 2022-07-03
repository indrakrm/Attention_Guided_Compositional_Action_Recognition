

The file containing annotations can be downloaded from:

https://drive.google.com/open?id=1XqZC2jIHqrLPugPOVJxCH_YWa275PBrZ in four parts,
it containes a dictionary mapping each video id, the name of the video file to the list of per-frame annotations. The annotations assume that the frame rate of the videos is 12.
An example of per-frame annotation is shown below, the names and number of "something's" in the frame correspond to the fields
'gt_placeholders' and 'nr_instances', the frame path is given in the field 'name', 'labels' is a list of object's and hand's bounding boxes and names.

```
   [
    {'gt_placeholders': ['pillow'],
     'labels': [{'box2d': {'x1': 97.64950730138266,
                          'x2': 427,
                          'y1': 11.889318166856967,
                          'y2': 239.92858832368972},
                          'category': 'pillow',
                          'standard_category': '0000'}},
                {'box2d': {'x1': 210.1160330781122,
                          'x2': 345.4329005999551,
                          'y1': 78.65516045335991,
                          'y2': 209.68758889799403},
                          'category': 'hand',
                          'standard_category': 'hand'}}],
     'name': '2/0001.jpg',
     'nr_instances': 2}, 
     {...},
     ...
     {...},
     ]
```


# Training
To train the model
```python train.py --model coord_latent_nl --num_frames 16 --logname experiment_name --batch_size 12 
                   --coord_feature_dim 256 --root_frames /path/to/frames 
                   --json_data_train dataset_splits/compositional/train.json 
                   --json_data_val dataset_splits/compositional/validation.json 
                   --json_file_labels dataset_splits/compositional/labels.json
                   --tracked_boxes /path/to/bounding_box_annotations.json
```

Place the data in the folder /path/to/frames each video bursted into frames in a separate folder. The ground-truth box annotations 
can be found in the google drive in parts and have to be concatenated in a single json file.

The models that are using appearance features are initialized with I3D network pre-trained on Kinetics.

