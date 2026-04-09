# pix2pix-filtered-30k

This folder contains the filtered 30k dataset processing pipeline and supporting metadata for the text-guided-image-to-3d project.  
Use the documentation below to find the dataset recovery instructions and the filtering/introduction notes.

## What to read

- To understand how to get or recover the dataset, read:  
  [`dataset_creation/README_extracting_the_images_using_json.md`](dataset_creation/README_extracting_the_images_using_json.md)

- To understand the filtering pipeline and general introduction, read:  
  [`dataset_creation/README_pix2pix_filtering_Introduction.md`](dataset_creation/README_pix2pix_filtering_Introduction.md)

## Repository tree

```text
pix2pix-filtered-30k/
├── .gitattributes
├── README.md
└── dataset_creation_files/
    ├── README_extracting_the_images_using_json.md
    ├── README_pix2pix_filtering_Introduction.md
    ├── recover_dataset.py
    ├── stage1_filter.py
    ├── stage2_3_runpod_v2.py
    ├── stage3_fast.py
    ├── final_indices.json
    ├── metadata.jsonl
    └── stage1_output/
