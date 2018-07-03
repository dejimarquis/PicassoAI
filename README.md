#PicassoAI
PicassoAI is a machine learning model built to generate 
Picasso like paintings of the Cubism era. This project 
was done as a proof a concept and it's inspired by my newly found interest
in art and Picasso

### Notes
 - An ensemble of gradient boosted trees was used to train the model
 - Data from Google images
 - `cd to Project directory` then `mkdir img && mkdir generated_imgs`
 - `python scrapeImages.py -s "picasso cubism paintings" -n 100 -d "Projectdirectory/img"`
 - `python picassoai.py`
 
 
 ###Some Interesting Results
 ![Alt text](./generated_imgs/picassoai_58.jpg?raw=true "PicassoAI_58")
 
 ![Alt text](./generated_imgs/picassoai_20.jpg?raw=true "PicassoAI_20")
 
 ![Alt text](./generated_imgs/picassoai_33.jpg?raw=true "PicassoAI_33")