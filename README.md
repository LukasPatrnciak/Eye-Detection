# Eye-Detection
The aim of this assignment is to implement a program in the Python programming language that will be able to find the iris of the eye in an image. As part of this assignment, we have a database of photographs (with a .csv file with Ground Triths localization circles - iris_annotation.csv). 

The task is to apply geometric object localization methods to these photographs and evaluate their success compared to the 'true' positions. First, it is necessary to load and display any image from the provided dataset and then it will be necessary to find circles bounding the pupil, iris, upper and lower eyelids using the Hough transform.

Subsequently, an evaluation of the success of this detection will be added to the program. The next step concerns searching the entire database of images, while we will search for optimal parameters using a grid search. The last step will concern iris segmentation according to circles.
