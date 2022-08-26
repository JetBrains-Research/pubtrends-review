* Download [PubMedCentral Author Manuscript Collection](https://ftp.ncbi.nlm.nih.gov/pub/pmc/manuscript/xml/) 
into `~/pmc_dataset` folder.\
  Expected folder content:
  ```
  author_manuscript_xml.PMC001xxxxxx.baseline.2022-06-16.filelist.csv
  author_manuscript_xml.PMC001xxxxxx.baseline.2022-06-16.filelist.txt
  author_manuscript_xml.PMC001xxxxxx.baseline.2022-06-16.tar.gz
  ...
  ```
* Extract all downloaded `tar.gz` files.\
  Expected extracted folders:
  ```
  PMC001xxxxxx
  PMC002xxxxxx
  PMC003xxxxxx
  PMC004xxxxxx
  PMC005xxxxxx
  PMC006xxxxxx
  PMC007xxxxxx
  PMC008xxxxxx
  PMC009xxxxxx
  ```
* Launch `dataset/analyze_archive.ipynb` to analyze downloaded archive.
* Launch `dataset/prepare_dataset.ipynb` to create tables required for model training.
* Refer to `train/README.md` for instruction on model training.