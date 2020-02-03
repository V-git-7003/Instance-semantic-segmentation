# Classifier for Rabies infected Neurons
The model classifies three different possible classes-Neuron, background and partial Neuron-for a larger agenda of detecting/annotating rabies infected neurons from brain scans of mice. The brain scans are of the size 10000 times 10000 (pixel unit). General cell detection algorithms (Ex. Unet-instance semantic segmentation) do not perform well on this case due to low signal to noise ratio. The rabies neurons on average have a diameter of 25-30 pixel size which compared to even a moderate cropped image size of 512 times 512 yields a very low signal to noise ratio. 

Hence, we use a classification algorithm that classifies cropped images of size 50 times 50 into three different possible classes-a single neuron, just background tissue and many neurons or partial neurons. Using this classifier we should achieve annotation (coordinates of rabies-neurons in the bigger image) of rabies-neurons in the bigger image. The above model can classify with almost 99% validation and test accuracy for our data. The 'Model_that_worked.h5' is a pretrained model that works with 99% accuracy on the test data. 

To train, the images of respective classes are put in the folder Data/Train/Images, Data/Train/Background, and Data/Train/Partial, and the test images from the three classes are put together in the folder Data/Test. The classifier can be run by the executing the file 'Binary_Classification.py', which calls 'Data_Preparation.py' for preparing images for training. In addition, 'Data_Preparation.py' file reads images of all three classes and appends the images with class index for training.
The other notebook files are just to check the functionality of the code.


