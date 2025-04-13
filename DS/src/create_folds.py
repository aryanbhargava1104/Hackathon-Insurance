import pandas as pd
from sklearn import model_selection

if __name__ == "__main__":
    df = pd.read_csv("./input/train.csv")
    
    y_label = df["Fraud_Label"]  

    df["kfold"] = -1

    # Shuffle the DataFrame
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)

    # Initialize StratifiedKFold
    kf = model_selection.StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    # Assign folds
    for fold, (train_idx, val_idx) in enumerate(kf.split(X=df, y=y_label)):
        print(f"Fold {fold}: Train = {len(train_idx)}, Valid = {len(val_idx)}")
        df.loc[val_idx, "kfold"] = fold

    # Save the new DataFrame with folds
    df.to_csv("./input/train_folds.csv", index=False)