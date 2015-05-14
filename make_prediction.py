
__author__ = "Sander Martijn Kerkdijk"
__copyright__ = "2015,DataMining"
__credits__ = ["Felix Wanders", "Dimi Gerakas"]
__license__ = "GPL"
__version__ = "1.0.4"
__maintainer__ = "Sander Martijn Kerkdijk"
__email__ = "skk420@few.vu.nl"
__status__ = "Beta"

import data_io
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.datasets import make_gaussian_quantiles
from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import GradientBoostingClassifier



def main():
    class bcolors:
        HEADER = '\033[95m'
        OKBLUE = '\033[94m'
        OKGREEN = '\033[92m'
        WARNING = '\033[93m'
        FAIL = '\033[91m'
        ENDC = '\033[0m'
        BOLD = '\033[1m'
        UNDERLINE = '\033[4m'

    print bcolors.HEADER + "Start Training" + bcolors.HEADER
    print bcolors.OKBLUE + "Reading and making Trainingset" + bcolors.OKBLUE

    train = data_io.read_train()
    train.fillna(0, inplace=True)

    train_sample = train[:1250000].fillna(value=0)       # change the samplesize over here

    # list of features that can be removed if you want
    feature_names = list(train_sample.columns)
    feature_names.remove("click_bool")
    feature_names.remove("booking_bool")
    feature_names.remove("gross_bookings_usd")
    feature_names.remove("date_time")
    feature_names.remove("position")

    features = train_sample[feature_names].values
    target = train_sample["booking_bool"].values

    print bcolors.OKGREEN + "Training Dataset" + bcolors.OKGREEN

    # check over here , you can find the algorithms at http://scikit-learn.org/stable/modules/ensemble.html

    # random forest
    classifier = RandomForestClassifier(n_estimators=3200,  verbose=2,n_jobs=-1,min_samples_split=10,random_state=1)

    # extra Trees (better then random forest) (best till now!)
    #classifier = ExtraTreesClassifier(n_estimators=300,  verbose=2, n_jobs=-1, min_samples_split=10,random_state=1)

    # Adaboost
    #classifier = AdaBoostClassifier(DecisionTreeClassifier(max_depth=2),n_estimators=600,learning_rate=1)



    # Knearest neighbour with bagging
    #classifier = BaggingClassifier(KNeighborsClassifier(), max_samples=0.5, max_features=0.5)


    # Gradient Boosting  BEST SOLUTION (i suppose,will try tomorrow)
    # classifier = GradientBoostingClassifier(loss='deviance', learning_rate=0.1,  n_estimators=100,  subsample=1.0, min_samples_split=2, min_samples_leaf=1, max_depth=3, init=None, random_state=None, max_features=None, verbose=0)


    classifier.fit(features, target)

    print bcolors.OKBLUE + "Saving Classifier" + bcolors.OKBLUE
    data_io.save_model(classifier)


    print bcolors.OKGREEN + "Start Making Predictions On Testset" + bcolors.OKGREEN

    print bcolors.OKBLUE + "Reading Testset" + bcolors.OKBLUE

    test = data_io.read_test()
    test.fillna(0, inplace=True)

    feature_names = list(test.columns)
    feature_names.remove("date_time")

    features = test[feature_names].values

    classifier = data_io.load_model()

    print bcolors.OKGREEN + "Make Predictions" + bcolors.OKGREEN
    predictions = classifier.predict_proba(features)[:,1]

    print bcolors.OKBLUE + "Calculate NDcg" + bcolors.OKBLUE
    predictions = list(-1.0*predictions)

    print bcolors.OKBLUE +  "Sort Predictions" + bcolors.OKBLUE
    recommendations = zip(test["srch_id"], test["prop_id"], predictions)

    print bcolors.OKGREEN + "Writing Predictions To Outputfile" + bcolors.OKGREEN


    data_io.write_submission(recommendations)

    print ""
    print bcolors.ENDC + "Thats all folks,goodbye!" + bcolors.ENDC

if __name__=="__main__":
    main()

