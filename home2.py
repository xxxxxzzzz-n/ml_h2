import marimo

__generated_with = "0.18.1"
app = marimo.App()


@app.cell
def _():
    import marimo as mo
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    import seaborn as sns

    from sklearn.model_selection import train_test_split
    from sklearn.compose import ColumnTransformer
    from sklearn.preprocessing import StandardScaler, OneHotEncoder
    from sklearn.pipeline import Pipeline
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.metrics import mean_squared_error, mean_absolute_error
    return (
        ColumnTransformer,
        OneHotEncoder,
        Pipeline,
        RandomForestRegressor,
        StandardScaler,
        mean_absolute_error,
        mean_squared_error,
        mo,
        np,
        pd,
        plt,
        sns,
        train_test_split,
    )


@app.cell
def _(pd):
    items = pd.read_csv("blinkit_order_items.csv")
    orders = pd.read_csv("blinkit_orders.csv")
    products = pd.read_csv("blinkit_products.csv")

    df = (
        items.merge(orders, on="order_id", how="left")
             .merge(products, on="product_id", how="left")
    )

    df.head()
    return (df,)


@app.cell
def _(df, np, pd):
    df_clean = df.copy()


    if "order_date" in df_clean.columns:
        df_clean["order_date"] = pd.to_datetime(df_clean["order_date"], errors="coerce")
        df_clean["month"] = df_clean["order_date"].dt.month
        df_clean["weekday"] = df_clean["order_date"].dt.weekday
        df_clean["hour"] = df_clean["order_date"].dt.hour


    num_cols = df_clean.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = df_clean.select_dtypes(include=["object"]).columns.tolist()


    id_cols = ["order_id", "product_id", "customer_id", "delivery_partner_id"]
    for col in id_cols:
        if col in num_cols:
            num_cols.remove(col)
        if col in cat_cols:
            cat_cols.remove(col)


    if "quantity" in num_cols:
        num_cols.remove("quantity")


    df_clean[num_cols] = df_clean[num_cols].fillna(df_clean[num_cols].median())
    df_clean[cat_cols] = df_clean[cat_cols].fillna("Unknown")


    y = df_clean["quantity"]
    feature_cols = num_cols + cat_cols
    X = df_clean[feature_cols]
    return X, y


@app.cell
def _(mo):
    n_estimators_slider = mo.ui.slider(50, 500, 50, 200, label='Number of Trees (n_estimators)')
    max_depth_slider = mo.ui.slider(3, 30, 1, 10, label='Maximum Tree Depth (max_depth)')
    test_size_slider = mo.ui.slider(10, 40, 5, 20, label='Test size (%)')
    sample_frac_slider = mo.ui.slider(0.1, 1.0, 0.1, 0.5, label='Sample Fraction (to speed up training)')
    mo.vstack([n_estimators_slider, max_depth_slider, test_size_slider, sample_frac_slider])
    return (
        max_depth_slider,
        n_estimators_slider,
        sample_frac_slider,
        test_size_slider,
    )


@app.cell
def _(
    ColumnTransformer,
    OneHotEncoder,
    Pipeline,
    RandomForestRegressor,
    StandardScaler,
    X,
    max_depth_slider,
    mean_absolute_error,
    mean_squared_error,
    n_estimators_slider,
    np,
    sample_frac_slider,
    test_size_slider,
    train_test_split,
    y,
):
    def run_model_block():
        n_estimators = n_estimators_slider.value
        max_depth = max_depth_slider.value
        test_size = test_size_slider.value / 100
        sample_frac = sample_frac_slider.value


        X_sample, _, y_sample, _ = train_test_split(
            X, y, train_size=sample_frac, random_state=42
        )

        X_train, X_test, y_train, y_test = train_test_split(
            X_sample, y_sample, test_size=test_size, random_state=42
        )

        numeric_features = X_train.select_dtypes(include=[np.number]).columns.tolist()
        categorical_features = [c for c in X_train.columns if c not in numeric_features]

        preprocess = ColumnTransformer(
            transformers=[
                ("num", StandardScaler(), numeric_features),
                ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features),
            ]
        )

        model = RandomForestRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            random_state=42,
            n_jobs=-1
        )

        pipe = Pipeline([
            ("prep", preprocess),
            ("model", model),
        ])

        pipe.fit(X_train, y_train)
        y_pred = pipe.predict(X_test)

        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        mae = mean_absolute_error(y_test, y_pred)

        return (
        X_test,
        X_train,
        max_depth,
        n_estimators,
        numeric_features,
        pipe,
        rmse,
        mae,         
        sample_frac,
        test_size,
        y_pred,
        y_test,
    )
    return (run_model_block,)


@app.cell
def _(mo, run_model_block):
    (
        X_test2,
        X_train2,
        max_depth2,
        n_estimators2,
        numeric_features2,
        pipe2,
        rmse2,
        mae2,
        sample_frac2,
        test_size2,
        y_pred2,
        y_test2
    ) = run_model_block()

    mo.hstack([
        mo.md(f"""
        <div style="
            background:#e7f0fd; padding:12px; border-radius:10px; width:180px;
            text-align:center; box-shadow:0 2px 6px rgba(0,0,0,0.1);
        ">
            <h3 style="margin:0; color:#1a73e8;">RMSE</h3>
            <p style="font-size:26px; margin:4px 0;"><b>{rmse2:.4f}</b></p>
        </div>
        """),

        mo.md(f"""
        <div style="
            background:#e8f7e6; padding:12px; border-radius:10px; width:180px;
            text-align:center; box-shadow:0 2px 6px rgba(0,0,0,0.1);
        ">
            <h3 style="margin:0; color:#2e7d32;">MAE</h3>
            <p style="font-size:26px; margin:4px 0;"><b>{mae2:.4f}</b></p>
        </div>
        """)
    ])
    return


@app.cell
def _(pd, plt, run_model_block, sns):
    (
        X_test,
        X_train,
        max_depth,
        n_estimators,
        numeric_features,
        pipe,
        rmse,
        mae,
        sample_frac,
        test_size,
        y_pred,
        y_test
    ) = run_model_block()
    importances = pipe.named_steps["model"].feature_importances_

    # Feature names from numeric + encoded categorical
    cat_encoded = pipe.named_steps["prep"].transformers_[1][1].get_feature_names_out().tolist()
    feat_names = numeric_features + cat_encoded

    imp_df = pd.DataFrame({
        "feature": feat_names,
        "importance": importances
    }).sort_values("importance", ascending=False).head(15)

    plt.figure(figsize=(8,5))
    sns.barplot(x="importance", y="feature", data=imp_df)
    plt.title("Top Feature Importances")
    plt.show()
    return (
        X_test,
        X_train,
        max_depth,
        n_estimators,
        rmse,
        sample_frac,
        test_size,
        y_pred,
        y_test,
    )


@app.cell
def _(plt, sns, y_pred, y_test):
    plt.figure(figsize=(6,6))
    sns.scatterplot(x=y_test, y=y_pred, alpha=0.3)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], "r--")
    plt.xlabel("Real")
    plt.ylabel("Predicted")
    plt.title("Real vs Predicted")
    plt.show()
    return


@app.cell
def _(plt, sns, y_pred, y_test):
    err = y_test - y_pred
    plt.figure(figsize=(7,4))
    sns.histplot(err, bins=20, kde=True)
    plt.title("Prediction Error Distribution")
    plt.show()
    return


@app.cell
def _(X_test, X_train, plt):
    sizes = [len(X_train), len(X_test)]
    plt.figure(figsize=(5,4))
    plt.bar(["Train", "Test"], sizes, color=["#4a90e2", "#7ed321"])
    plt.title("Train vs Test Sizes")
    plt.show()
    return


@app.cell
def _(max_depth, mo, n_estimators, sample_frac, test_size):
    mo.md(f"""
    ### ⚙️ Model Parameters

    - **n_estimators:** {n_estimators}  
    - **max_depth:** {max_depth}  
    - **test size:** {test_size}  
    - **sample fraction:** {sample_frac}
    """)
    return


@app.cell
def _(max_depth, mo, rmse, sample_frac):
    insights = []

    if rmse > 2:
        insights.append("⚠️ The model’s error is a bit high. Try increasing the training sample size or reducing the tree depth")

    if max_depth > 20:
        insights.append("🌲 The model may be overfitting — the trees are too deep. Consider lowering max_depth")

    if sample_frac < 0.3:
        insights.append("📉 The model is training on too little data. Increasing the sample fraction should improve performance")

    if len(insights) == 0:
        insights.append("✅ The model looks stable and well-balanced")

    mo.md("### 🔍 Insights:\n" + "\n".join([f"- {i}" for i in insights]))
    return


@app.cell
def _(plt, y_pred, y_test):
    plt.figure(figsize=(6, 5))
    plt.scatter(y_test, y_pred, alpha=0.3)
    plt.xlabel("Real quantity")
    plt.ylabel("Predicted quantity")
    plt.title("Real vs Predicted")
    plt.show()


    error = y_test - y_pred
    plt.figure(figsize=(6, 4))
    plt.hist(error, bins=20)
    plt.title("Error distribution")
    plt.show()
    return


if __name__ == "__main__":
    app.run()
