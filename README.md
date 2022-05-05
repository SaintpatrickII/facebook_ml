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
