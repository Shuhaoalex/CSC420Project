# CSC420Project

To train the model, simply run the following code in the root directory of this project, `python src/train.py path_to_job_folder`.
For example, to train the edge generating model, run `python src/train.py job_edge/`. To make changes to the configurations of the model, change properties in the `configurations.json` file located in each job folder. Examples configuration files can be found in side the existing job folders. Weights for the models will be saved to the `weights` folder inside job folders.

To test the model's performance on different images, run `python src/test.py job_test/`. We use a `flist` file as our dataset descriptor, detailed instruction for generating such descriptor file can be found in `datasets/instruction.txt`. We have our trained weights in the `job_test` folder.
