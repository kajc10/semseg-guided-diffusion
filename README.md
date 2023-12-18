# semseg-guided-diffusion

Train and test semantic segmentation guided diffusion models.

You will need to install:  **pytorch-lightning, torchvision, hydra, wandb, omegaconf** (and **skimage** for BDD preprocessing)\
Use: `pip install -r requirements.txt`

The data needs to be in a specific format. Our experiments utilized the [Berkeley Deep Drive Dataset](https://bdd-data.berkeley.edu/). To see detailed steps for that specific dataset, see `data/BDD_instructions.md`.

For custom datasets, you should have a fromat presented in `data/sample` . Two separate folders: one for the images, one for the masks and a `colormap.json` file, that contains mappings for the given classes.
You need to edit `config/base.yaml` with the correct paths. Make sure to set correct class number and image size as well.


# Train
Make sure you have configured everything in  `config/base.yaml` !
Then use `python train.py`. 


# Test
Make sure you have configured everything in  `config/base.yaml` !
Then use `python test.py`. 

:warning: More detailed instructions coming soon... :warning: