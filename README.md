# TGS-Salt-Identification-Challenge
## Kaggle TGS Challenge

The enviroment you need:
* python3
* tensorflow
* Keras
* numpy
* pandas
* PIL

Image Augemention
>> Read the file "ImageGenator"
>> python images && masks .py and rename the path
>> Visualization your data

Model Network
>> python src/Bn.py

Test and Submission
>> python src/test.py



### submission1:
try original unet -- batch_size = 8 -- 0.599

### submission2:
> Remove black image  -- batch_size = 8 -- 0.607

### submission3:
> Image Auge -- color*3 && flip*3 && rotate*3, batch_size = 8, loss = 0.22 -- 0.613

### submission4:
> Image Auge -- bn layers,loss = 0.20675 -- 0.661

### submission4:
> Put the dropout layer in before output layer and put regularzition on some layers.(Ing...)
