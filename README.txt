INSTRUCTIONS TO RUN THE CODE
############################
use template.py
1) I have written all my methods in the template.py itself, inorder to avoid the having many different files (at the cost of heavy template.py)

2) Kindly keep the directory structure entact, do not copy paste files from one folder to other, the paths given in the code are relative and may throw error if files are moved here and there.

3) When you will run the script It will automatically load the data and start running the regression with the values I have tuned with, if one just want to load the saved pickel file to test, see in code for instructions (in comments)

BEHAVIOUR OF GRADIENT DESCENT
#############################
I first did some feature engineering on the data set which was loaded. Some important points are:

1) Even though the dataset had timestamp (which made me think at once) that it's a time series data, after some preliminary observation, It wasn't really the time series data (atleast not in the native form), So i decided to ditch the timestamp column as well as the id column

2) As for the first attempt I only tried the regression with first degree feature vectors, the behaviour which cropped up was, even though the gradient descent seemed to work fine for the unregularised part, as soon as the regularisation term was involved, gredient descent seemed to be overshooting the optimal point much more often than in the normal case, also the initial alpha (step size) which was to be used for descent requrired to be much more smaller in case of p=[1,2] than for lambda = 0 case.

3) Final feature engineering which was used was to consider feature vectors which have all the first and second order term(including the cross second order term), I even tried to include the cubic terms, the result was not so very awesome, hence I stuck two 2 degree. Once the 2 degree PHI matrix was created, I normalised my dataset using the ZSCORE method, where I subtracted the features with their respective means and divided it bby it's standard deviation. I attempt to remove the outliers from the dataset, it only made the result worst, so I reverted the changes.

4) The ZSCORE transformation, needs to be applied to the test data as well for (using the meanVector and stdVector of the testing data), before doing the prediction.
