# facebook_ml
Machine Learning project aimed at creating a machine learning that recommends products akin to facebook marketplace

1. Exploring & cleaning the datasets:

Before starting any form of EDA it is important to examine the datasets & clean them ready for usage in any analysis/ models. For this project there are two datasets: a raw text 'products' csv file, this contains all information about the product i.e. price, location, description etc and an images folder containing the raw images associated to the 'products' dataset.

Advertisement Dataset:

- Data is loaded in from the S3 bucket into a jupyter notebook (this allows for rapid troubleshooting, once all cleaning has been completed this can be transferred into a python method), pandas was used to turn the dataset into a dataframe.

- Before any cleaning it is a good idea to have a look at the raw file to see any obvious improvements. a combination of this & the use of the pandas df.dtype and df.head gives us a good indication of the next steps to be taken to clean this file.

Cleaning operations:

1. by looking at the datatypes of our rows most are object, in order to save hastle we convert these to str type, this will later allow us to manipulate them later on

2. from looking at the csv file some non-ascii characters have been used in two of the columns (product description & product name), for this we use a simple encode/decode function to rid the df of these

3. our df has a inbuilt index column & a meaningless additional index column, this is dropped from the table, whilst dropping this column i also chose to drop columns with nan data inputs, this meaning our new df will have inputs for each value

4. the price function contains a '£' symbol in each price, to make these values into floats we simply use a str replace & convert the results into float64 datatype

5. product_name includes the location of the item in the header, this information is already known in the location column so we use a str split & index the zeroth element so that product name is only the product

6. category & location both contain two pieces of information, here i have split the category into an additional subcaregory column & with location it is split into county & the original location


Images Dataset:

raw images come in a variety of sizes & aspect ratios, in order to train the machine learning model firstly we will need to clean these images beforehand.Images here are modified through the PIL import & glob is used to iterate through the files within our image folder.

Cleaning Operations:

1. Images are formed in a for loop with an enumerate function(this makes for easy naming of the photos).

2. Firstly our loop opens the image & created a black background at the specified limit of 512 x 512 pixels to be overlayed later.

2. the maximum dimension of each image is found & compared to our maximum acceptable size, this is computed into a ratio factor which will transform the image to the correct size.

3. Background image is overlayed with the product image, image is centred on this background & saved with the enumerate function from before

2. Creating Simple Machine Learning Models:

Product Details Regression:

- A simple model is utiliser here, using one hot encoding products are assigned into their categories paired with their price

- So we are going to be testing the linear dependance category of the price, so y is the price of the item & X is the category

- As this is a very basic test as expected the model doesnt perform well at all, with a MSE of its definetly not ideal

Image Multi Class Classification:

- For the Images again we will be one hot encoding to catogrise products, however we will need to have. numerical value for each category for our images model

- Images are opened alike before using PIL, in this current form our model has no way to analyse the image, for this we will need to transform the image into a readable format

- to do this we can transform the PIL image into a pytorch Tensor, in this form we will have a three channel tensor for each image, this however is still not readable by our model.

- The tensor must be flattened & turned into a flat numpy array for usage

- In this form while trainable is not ideal as we lose massive amounts of information on the images that could be used in a model i.e. do pixels that are neighbors have any effect? This will be fixed at a later point using neural networks

- This flattened numpy array is joined to the class number in a tuple for training in sklearn's logistic regression model with X=(no. of labels, no of features) and y = (no of labels)

- This model only produces an accuracy of 15% this will be drastically increased with the usage of a CNN

- While making this image dataset i had tried for ages to correctly split the categories & index them into the correct tuples, this can be seen in the multiclas_logreg file, for anyone looking to mimic this, for the love of all things good just use the inbuilt sklearn LabelEncoder.

- My other main pause with this step happened when my categories were completeley unbalanced, as it turns out in my data cleaning when removing duplicate rows i had accidently removed rows which contained different pictures of the same product, always good to double check this next time :)
