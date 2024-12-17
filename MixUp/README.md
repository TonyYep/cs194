synthesis.py is the main execution program, while pipline.py contains the required runtime functions, including the autogen framework and the image fusion Mixup function.

	melt_image5() -- The function implementing Mixup takes the coordinates of patches from two images and replaces the patch in one image with the corresponding patch from the other image.
	encode_image() -- By encoding the image in base64 format, it can be recognized by GPT.
	get_important_patch() -- Returns the coordinates of the most important and least important patches in the image to prepare for image fusion.
	get_best_fuse() -- Returns the best fused image.
	adjust_coors() -- Prevent the coordinates provided by GPT from exceeding the image dimensions.
	autogen_framework() -- Function to implement the autogen multi-agent collaboration framework.
	main() -- A function to perform image fusion, integrating all the above components together.

	processing() -- Using this single function, the entire process can be completed, including image fusion, label fusion, image-label correspondence, generating one-hot labels, and retaining the results of random image sampling.


To run the program, place the ImageNet_subset folder in the same directory and put label.txt inside the ImageNet_subset folder.
- synthesis.py
- pipeline.py
- ImageNet_subset
	- train
		- label1
			- .jpg
			- .jpg
			- ...
		- ...
	- val
	- label.txt