import pandas as pd
import random

def filter_top_features(data, top_n_indices, threshold, m):
    features = data.x.detach().cpu().numpy()
    df_features = pd.DataFrame(features)

    categorical_features = []
    numerical_features = []
    unique_values_dict = {}

    for column in top_n_indices:
        unique_values = df_features[column].nunique()
        unique_values_dict[column] = unique_values

        # threshold to determine categorical features
        if unique_values < threshold:
            categorical_features.append(column)
        else:
            numerical_features.append(column)

    # sorted_columns = sorted(unique_values_dict, key=unique_values_dict.get)
    # # top m categorical features
    # # m = int(n/2)  # Set the number of top categorical features to pick
    # # top_m_categorical_features = categorical_features[:m]  # Get the first `m` categorical features
    # # top_m_categorical_features = sorted_columns[:m]
    # top_m_categorical_features = random.sample(sorted_columns, min(m, len(sorted_columns)))

    # if len(categorical_features) < m:
    #     print(f"Warning: Only {len(categorical_features)} valid features found; returning all.")
    #     top_m_categorical_features = categorical_features
    # else:
    #     top_m_categorical_features = random.sample(categorical_features, m)


    # randomly sample additional features if the satisfied number < m
    if len(categorical_features) < m:
        remaining = m - len(categorical_features)
        print(f"Warning: Only {len(categorical_features)} valid features found; random choose additional {remaining} features.")

        # exclude already chosen features
        remaining_features = list(set(top_n_indices) - set(categorical_features))
        
        additional_features = random.sample(remaining_features, min(remaining, len(remaining_features)))
        categorical_features.extend(additional_features)


    # top_m_categorical_features = categorical_features[:m]
    if len(categorical_features) >= m:
        top_m_categorical_features = random.sample(categorical_features, m)
    else:
        top_m_categorical_features = categorical_features

    # print(f"Top {m} categorical features: {top_m_categorical_features}")
    # print(f"Random choose {m} categorical features.")
    return df_features, top_m_categorical_features