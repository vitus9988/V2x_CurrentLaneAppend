from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn.pipeline import make_pipeline
import joblib
from sklearn.model_selection import GridSearchCV



def origin_model(leanrDataPath):
    data = pd.read_csv(leanrDataPath)

    X = data.drop(columns=['CURRENT_LANE','SPEED'], axis=1)
    y = data["CURRENT_LANE"]
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=121, shuffle=True)

    pipe = make_pipeline(
        RandomForestClassifier(n_estimators = 400, max_depth = None,
                            min_samples_split = 2, min_samples_leaf =1,
                            max_features='auto',
                            class_weight ='balanced' ,n_jobs=-1, 
                            random_state=2, oob_score=True)
    )


    pipe.fit(X_train, y_train)
    print("학습 정확도", pipe.score(X_train, y_train))
    print("테스트 정확도", pipe.score(X_test, y_test))

    joblib.dump(pipe, 'rad_fore.pkl')

    
    
def gridtest(leanrDataPath):
    data = pd.read_csv(leanrDataPath)

    X = data.drop(columns=['CURRENT_LANE'], axis=1)
    y = data["CURRENT_LANE"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=121, shuffle=True)

    param_grid = {
    'n_estimators': [10, 50, 100, 200],
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'max_features': ['auto', 'sqrt', 'log2']
    }
    
    rf = RandomForestClassifier(random_state=42)
    
    grid_search = GridSearchCV(
    estimator=rf,
    param_grid=param_grid,
    scoring='accuracy',
    cv=5,
    verbose=0,
    n_jobs=-1
    )
    
    grid_search.fit(X_train, y_train)
    best_params = grid_search.best_params_
    print("Best parameters found:", best_params)



if __name__ == "__main__":
    origin_model()