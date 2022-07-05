import plotly.express as px


def plot_n_top_models(df, dataset_name, n, fs_type=None):

    df = df[df['dataset'] == dataset_name].sort_values(
        by='best_score', ascending=False)
    if fs_type is not None:
        df = df[df['algorithm'].str.contains(fs_type)]

    df = df.iloc[0:n, :]
    df['best_score'] = df['best_score'].round(decimals=4)
    n = df.shape[0]

    fig = px.bar(df, x='best_score', y='algorithm', text='best_score', orientation='h',
                 labels={'best_score': 'score'},
                 color='best_score', color_continuous_scale=px.colors.sequential.Oryel)

    fig.update_traces(hovertemplate="<br>".join([
        f"Score:"+": %{x:.4f}",
        "Model: %{y}",
        f"Dataset: {dataset_name}"
    ]))

    fig.update_layout(
        yaxis=dict(autorange="reversed")
    )

    t = f'{n} top scores for dataset: {dataset_name}'

    if fs_type is not None:
        t += f' (feature selection method: {fs_type})'

    fig.update_layout(title=t)
    fig.show()


def plot_n_most_selected_features(counter, models_count, dataset_name, n):
    
    counter = counter.iloc[0:n, :]
    fig = px.bar(counter, x=counter.index.astype(str), y='count', text='count',
                    color='count', color_continuous_scale=px.colors.sequential.Oryel)

    fig.update_layout(title=f'{n} most frequently selected features among {models_count} models. Dataset: {dataset_name}')
    fig.update_xaxes(title="Feature original index")
    fig.show()