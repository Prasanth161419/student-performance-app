import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from sklearn.model_selection import train_test_split,GridSearchCV
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.tree import DecisionTreeRegressor
import joblib

# Cell 1: Load and display head
df=pd.read_csv("student_habits_performance.csv")
print("--- df.head() ---")
print(df.head())
print("\n")

# Cell 2: Display info
print("--- df.info() ---")
df.info()
print("\n")

# Cell 3: Display shape
print("--- df.shape ---")
print(df.shape)
print("\n")

# Cell 4: Display head(2)
print("--- df.head(2) ---")
print(df.head(2))
print("\n")

# Cell 5: Imports (already imported above, but included for completeness)
# import numpy as np
# import matplotlib.pyplot as plt
# import seaborn as sns

# Cell 6: Set seaborn style
sns.set(style="whitegrid")

# Cell 7: Check total NaN
print("--- df.isna().sum().sum() ---")
print(df.isna().sum().sum())
print("\n")

# Cell 8: Display full DataFrame (will be truncated by default)
print("--- df ---")
print(df)
print("\n")

# Cell 9: Check duplicated
print("--- df.duplicated().sum() ---")
print(df.duplicated().sum())
print("\n")

# Cell 10: Filter warnings
warnings.filterwarnings("ignore")

# Cell 11: Describe numerical
print("--- df.describe() ---")
print(df.describe())
print("\n")

# Cell 12: Describe object
print("--- df.describe(include=\"object\") ---")
print(df.describe(include="object"))
print("\n")

# Cell 13: Check NaN by column
print("--- df.isna().sum() ---")
print(df.isna().sum())
print("\n")

# Cell 14: Define categorical columns
categorical_cols=["gender","part_time_job","diet_quality","parental_education_level","internet_quality","extracurricular_participation"]

# Cell 15: Print value counts for categorical
print("--- Value Counts for Categorical Columns ---")
for col in categorical_cols:
    print(f"value counts for {col}:\n{df[col].value_counts()}")
print("\n")

# Cell 16: Plot histograms
print("--- Plotting Histograms ---")
df.hist(bins=20,edgecolor="black")
plt.tight_layout()
plt.show()
print("\n")

# Cell 17: Plot count plots
print("--- Plotting Count Plots ---")
for col in categorical_cols:
    sns.countplot(data=df,x=col)
    plt.title(f"distribution pf {col}")
    plt.xticks(rotation=45)
    plt.show()
print("\n")
    
# Cell 18: Plot correlation heatmap
print("--- Plotting Correlation Matrix ---")
sns.heatmap(df.corr(numeric_only=True),annot=True,cmap="coolwarm",fmt=".2f")
plt.title("Correlation Matrix")
plt.show()
print("\n")

# Cell 19: Get numerical column names
print("--- df.describe().columns ---")
print(df.describe().columns)
print("\n")

# Cell 20: Define numerical features
num_features=['age', 'study_hours_per_day', 'social_media_hours', 'netflix_hours',
       'attendance_percentage', 'sleep_hours', 'exercise_frequency',
       'mental_health_rating']

# Cell 21: Plot scatter plots
print("--- Plotting Scatter Plots (Features vs Exam Score) ---")
for feature in num_features:
    sns.scatterplot(data=df,x=feature,y="exam_score")
    plt.title(f"{feature} vs Exam Score")
    plt.show()
print("\n")
    
# Cell 22: Imports (already imported above)
# from sklearn.model_selection import train_test_split,GridSearchCV
# from sklearn.preprocessing import LabelEncoder
# from sklearn.linear_model import LinearRegression, Ridge, Lasso
# from sklearn.ensemble import RandomForestRegressor
# from sklearn.metrics import mean_squared_error, r2_score
# from sklearn.tree import DecisionTreeRegressor

# Cell 23: Display columns
print("--- df.columns ---")
print(df.columns)
print("\n")

# Cell 24: Display head(2) again
print("--- df.head(2) ---")
print(df.head(2))
print("\n")

# Cell 25: Define model features
features=["study_hours_per_day","attendance_percentage","mental_health_rating",'sleep_hours','part_time_job']

# Cell 26: Define target
target="exam_score"

# Cell 27: Create model DataFrame
df_model=df[features+[target]].copy()

# Cell 28: Display model DataFrame
print("--- df_model ---")
print(df_model)
print("\n")

# Cell 29: Initialize LabelEncoder
le=LabelEncoder()

# Cell 30: Apply LabelEncoder
df_model["part_time_job"]=le.fit_transform(df_model["part_time_job"])

# Cell 31: Display encoded model DataFrame
print("--- Encoded df_model ---")
print(df_model)
print("\n")

# Cell 32: Define X
X=df_model[features]

# Cell 33: Define y
y=df_model[target]

# Cell 34: Split data
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2)

# Cell 35: Check y_test length
print("--- len(y_test) ---")
print(len(y_test))
print("\n")

# Cell 36: Check y_train length
print("--- len(y_train) ---")
print(len(y_train))
print("\n")

# Cell 37: Define models dictionary
models={
    "linearregression":{
        "model":LinearRegression(),
        "params":{}
    },
    "DecisionTree":{
        "model":DecisionTreeRegressor(),
        "params":{"max_depth":[3,5,10],"min_samples_split": [2,5]}
        },
        "RandomForest":{
            "model":RandomForestRegressor(),
            "params":{"n_estimators":[50,100],"max_depth":[5,10]}
        }
    }

# Cell 38: Initialize results list
best_models=[]

# Cell 39: Run GridSearchCV
print("--- Running GridSearchCV ---")
for name, config in models.items():
    print(f"Training {name}")
    grid = GridSearchCV(
        config["model"],
        config["params"],
        cv=5,
        scoring="neg_mean_squared_error"
    )
    grid.fit(X_train, y_train)

    y_pred = grid.predict(X_test)

    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    
    r2 = r2_score(y_test, y_pred)

    best_models.append({
        "model": name,
        "best_params": grid.best_params_,
        "rmse": rmse,
        "r2": r2
    })
print("\n")

# Cell 40: Create results DataFrame
results_df=pd.DataFrame(best_models)

# Cell 41: Display sorted results
print("--- Model Results (sorted by RMSE) ---")
print(results_df.sort_values(by="rmse"))
print("\n")

# Cell 42: Get best model row
best_row=results_df.sort_values(by="rmse").iloc[0]

# Cell 43: Display best row
print("--- Best Model Row ---")
print(best_row)
print("\n")

# Cell 44: Get best model name
best_model_name=best_row["model"]

# Cell 45: Display best model name
print("--- Best Model Name ---")
print(best_model_name)
print("\n")

# Cell 46: Get best model config
best_model_config=models[best_model_name]

# Cell 47: Display best model config
print("--- Best Model Config ---")
print(best_model_config)
print("\n")

# Cell 48: Define final model
final_model=best_model_config["model"]

# Cell 49: Fit final model on all data
print("--- Fitting Final Model on All Data ---")
final_model.fit(X,y)
print("\n")

# Cell 50: Predict on test set with final model
print("--- Final Model Predictions on X_test ---")
print(final_model.predict(X_test))
print("\n")

# Cell 51: Save model with joblib
print("--- Saving Model ---")
joblib.dump(final_model,"best_model.pkl")
print("\n")

# Cell 52: Load and predict with saved model
print("--- Loading and Predicting with Saved Model ---")
print(joblib.load("best_model.pkl").predict(X_test))
print("\n")