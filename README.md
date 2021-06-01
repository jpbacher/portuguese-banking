# portuguese-banking
Determining which potential customers' characteristics are best at predicting a purchase of a bank's product


## Table of Contents
* [Abstract](#Abstract)
* [Project Structure](#ProjectStructure)
* [Limitations](#Limitations)
* [Considerations/Decisions](#Considerations/Decisions)
* [Plots](#Plots)
* [Conclusion](#Conclusion)
* [Contact](#Contact)

## Abstract
In this assessment, our objective is to help exexecutives at a large bank understand which characteristics of potential customers have the greatest influence in purchasing a bank product. The bank is interested in using a predictive model to score each customer's propensity to purchase a product. The data used to build our model can be found: [Bank Marketing Data Set](https://archive.ics.uci.edu/ml/datasets/Bank+Marketing).

## ProjectStructure
The *notebooks* directory contains Jupyter notebooks for the discovery process:
    
  *1.0-jpb-eda.ipynb*: EDA of the data
    
  *2.0-jpb-preprocessing.ipynb*: preprocessing the data for a given model
    
  *3.0-jpb-model-selection-assessment.ipynb*: tuning various models and selectinng the best model

The *packages* directory would have contained an actual package, with setup information, logging, testing, a  prediction script, to name a few, if more time was available; however, there still are various scripts for training the model we deemed suitable from discovery:
    
*config* directory:
        
  *config.py*: contains global variables for various scripts
    
*processing* directory:
        
  *data_managemennt.py*: helper functions to load the data, clean the data, engineer features, and save our final pipeline.
    
  *pipeline.py*: pipeline from scikit-learn's Pipeline class, where we standardize our numerical features, dummy encode the categorical features, and apply our XGBoost Classifier to the processed data.
    
 *train_pipeline.py*: script that uses functions from *data_management.py* and our pipeline from *pipeline.py* to train our model on all of the data points given.
    
## Limitations
Given the short time constraint, there were numerous areas we were unable to pursue further. 

First and most importantly, a deeper understanding of the predictors to our response variable would provide valuable information - this comprehension would allow us to do meaningful feature engineering, which could provide better signal for a given model. 

Second, we could have attempted various preprocessing techniques for our tree-based models, possibly a simple imputation of -1's to missing values, as opposed to creating a new category given. Also, we could have attempted target encoding on some of our categorical features, as opposed to just dummy encoding. By doing the former, it would give our numerical features a 'better chance' of having greater importance to the model. 

Third, spending more time tuning our hyperparmaeters, especially for more complex models like XGBoost. 

Last, as stated above, we could have created a package of our model that could be used in different applications.

## Considerations/Decisions
In this evaluation, we assumed there was an equal cost for a False Negative to a False Positive; so, we chose the F1 score to optimize the models, and kept the threshold at 0.5. If there were different costs, we would create a new scorer based on the costs, and adjusted our probability thresholds to determine the optimal (minimal) cost.

In our training data, the response variable, whether a customer subscribed to particular bank product, was somewhat balanced (88% did not subscribe, 12% did subscribe). Therefore, we never chose to use the SMOTE algorithm, or undersample the majority class.

We chose a regularized Logistic Regression model as our baseline, and found after tuning both RandomForest and XGBoost models, these models produced better tradeoffs between the recall and precision scores. Though the Logistic model had less False Negatives (0.63 recall) than the other two models (0.56 recall) in the validation set, XGBoost model had much less False Positives (0.45 precision) than Logisitic (0.33 precision). Hence, we chose the XGBoost as our final model.

## Plots
We quickly inspected the feature importances of our leading model, and discovered that communication through cellular (*contact* variable) was the most important predictor. Also, another important predictor in determining a customer purchase is whether the bank has made contact in the past (*prev_contact_no*). And for some particular reason, customers are more willing to make a puchase when the last contact month of year is an end to a fiscal quarter (March, September, December). ![Importances](https://github.com/jpbacher/portuguese-banking/tree/master/notebooks/visuals/feat_importance.png)

## Conclusion
Our discovery shows an auspicious venue in customer purchase prediction with the given data. Further examination may use different encoding techniques to the categorical features, as well as acquiring more potential customer-specific traits, such as location of the individual, income-levels or credit-scores (if possible).

## Contact
Created by [@jpbacher](https://www.linkedin.com/in/joshbacher) - thank you for your time, please feel free to connect with me.
