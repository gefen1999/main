import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
import re

def load_data(path):
    df = pd.read_csv(path)
    # extract numeric problem number
    df['problem_num'] = df['problem_num'].str.extract(r'(\d+)').astype(int)
    return df

def build_model():
    numeric_features = ['task', 'problem_num', 'a1', 'pa1', 'a2',
                        'b1', 'pb1', 'b2', 'corrAB', 'N']
    text_features = ['A_Lab', 'B_Lab']

    preprocessor = ColumnTransformer([
        ('num', 'passthrough', numeric_features),
        ('txt', TfidfVectorizer(), 'A_Lab'),
        ('txt2', TfidfVectorizer(), 'B_Lab')
    ])
    model = RandomForestRegressor(n_estimators=200, random_state=42)
    return Pipeline([
        ('preprocessor', preprocessor),
        ('model', model)
    ])

def train(path):
    df = load_data(path)
    X = df.drop('A_rates', axis=1)
    y = df['A_rates']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    pipeline = build_model()
    pipeline.fit(X_train, y_train)
    preds = pipeline.predict(X_test)
    mae = mean_absolute_error(y_test, preds)
    print(f"MAE: {mae:.4f}")
    return pipeline

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Train model on CPC dataset')
    parser.add_argument('--csv', default='gpt_cpc_1000 oneshot DM (1).csv', help='Path to dataset CSV')
    args = parser.parse_args()
    train(args.csv)
